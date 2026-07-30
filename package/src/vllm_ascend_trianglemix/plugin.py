"""Non-invasive vLLM-Ascend integration for TriangleMix.

The plugin deliberately patches only four narrow seams:

* the 0.23 model runner, to attach a scheduler-owned final prompt length and
  fresh step token to supported B=1 eager metadata;
* the metadata builder, to build one immutable routing plan per scheduler
  generation;
* ``forward``, to remember the vLLM layer name on the backend instance; and
* ``forward_fused_infer_attention``, to dispatch eligible prefill calls to the
  single-launch paged-KV operator.

Everything else, including decode and every fallback, remains on the
vLLM-Ascend implementation installed by the user.  In particular, no source
file under ``site-packages`` is copied or replaced.
"""

from __future__ import annotations

import functools
import inspect
import itertools
import json
import logging
import os
import sys
import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from .config import PluginConfig, resolve_plugin_config
from .kernel import (
    SUPPORTED_CACHE_BLOCK_SIZE,
    SUPPORTED_HEAD_SIZE,
    SUPPORTED_KV_HEADS,
    SUPPORTED_QUERY_HEADS,
    TriangleMixConfig,
    extract_layer_index,
    triangle_direct_paged_attention,
)
from .planning import FallbackReason, TriangleBatchPlan, build_batch_plan
from .stats import runtime_stats


# Use vLLM's configured logger hierarchy so requested INFO-level runtime
# counter snapshots are observable in worker logs.  A package-local logger
# inherits the process root's WARNING level in the supported vLLM stack and
# silently drops hit/fallback/performance evidence.
logger = logging.getLogger("vllm.trianglemix.plugin")

_PATCH_LOCK = threading.Lock()
_RUNNER_PATCH_LOCK = threading.Lock()
_REGISTERED = False
_PATCH_MARKER = "_vllm_ascend_trianglemix_plugin_patched"
_RUNNER_PATCH_MARKER = "_vllm_ascend_trianglemix_runner_patched"
_CONFIG_ATTR = "_vllm_ascend_trianglemix_config"
_NATIVE_READY_ATTR = "_vllm_ascend_trianglemix_native_ready"
_LAYER_NAME_ATTR = "_vllm_ascend_trianglemix_layer_name"
_LAYER_SELECTED_ATTR = "_vllm_ascend_trianglemix_layer_selected"
_KV_CAPABILITY_ATTR = "_vllm_ascend_trianglemix_kv_capability"
_PLAN_ATTR = "_vllm_ascend_trianglemix_plan"
_PLAN_FINGERPRINT_ATTR = (
    "_vllm_ascend_trianglemix_plan_fingerprint"
)
_PROMPT_LENS_ATTR = "_vllm_ascend_trianglemix_prompt_lens"
_PROMPT_SOURCE_ATTR = "_vllm_ascend_trianglemix_prompt_source"
_GENERATION_ATTR = "_vllm_ascend_trianglemix_builder_generation"
_RUNNER_PROMPT_ATTR = "_vllm_ascend_trianglemix_runner_prompt_lens"
_RUNNER_STEP_ATTR = "_vllm_ascend_trianglemix_runner_step"
_GRAPH_ATTR = "_vllm_ascend_trianglemix_graph_capture"
_BUILDER_GENERATIONS = itertools.count(1)
_RUNNER_STEPS = itertools.count(1)
_RUNNER_REJECTIONS: set[tuple[int, str]] = set()
_SUPPORTED_VLLM_VERSION = "0.23.0"
_SUPPORTED_ASCEND_VERSION = "0.23.0rc1"
_RUNNER_MODULE = "vllm_ascend.worker.model_runner_v1"
_ATTRIBUTE_MISSING = object()


class _UnsupportedCompatibilityError(RuntimeError):
    pass


def _installed_versions() -> tuple[str, str]:
    try:
        from importlib.metadata import version

        return version("vllm"), version("vllm-ascend")
    except Exception as exc:
        raise _UnsupportedCompatibilityError(
            "could not resolve installed vllm/vllm-ascend versions"
        ) from exc


def _without_local_suffix(value: str) -> str:
    return value.split("+", 1)[0]


def _validate_versions(versions: tuple[str, str]) -> None:
    vllm_version, ascend_version = versions
    if (
        _without_local_suffix(vllm_version)
        != _SUPPORTED_VLLM_VERSION
        or _without_local_suffix(ascend_version)
        != _SUPPORTED_ASCEND_VERSION
    ):
        raise _UnsupportedCompatibilityError(
            "unsupported_version: expected "
            f"vllm={_SUPPORTED_VLLM_VERSION}[+local] and "
            f"vllm-ascend={_SUPPORTED_ASCEND_VERSION}[+local], got "
            f"vllm={vllm_version!r}, "
            f"vllm-ascend={ascend_version!r}"
        )


def _default_key(value: object) -> str:
    if value is inspect.Parameter.empty:
        return "required"
    if value is None:
        return "none"
    if value is False:
        return "false"
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return f"enum:{name}"
    return repr(value)


def _signature_key(function: object) -> tuple[
    tuple[str, str, str], ...
]:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError) as exc:
        raise _UnsupportedCompatibilityError(
            f"cannot inspect signature for {function!r}"
        ) from exc
    return tuple(
        (
            parameter.name,
            parameter.kind.name,
            _default_key(parameter.default),
        )
        for parameter in signature.parameters.values()
    )


def _required(name: str) -> tuple[str, str, str]:
    return name, "POSITIONAL_OR_KEYWORD", "required"


def _optional(
    name: str,
    default: str = "none",
) -> tuple[str, str, str]:
    return name, "POSITIONAL_OR_KEYWORD", default


_ATTENTION_SIGNATURES: dict[str, tuple[tuple[str, str, str], ...]] = {
    "builder.__init__": tuple(
        _required(name)
        for name in (
            "self",
            "kv_cache_spec",
            "layer_names",
            "vllm_config",
            "device",
        )
    ),
    "builder.build": (
        _required("self"),
        _required("common_prefix_len"),
        _required("common_attn_metadata"),
        _optional("fast_build", "false"),
    ),
    "builder.build_for_graph_capture": (
        _required("self"),
        _required("common_attn_metadata"),
        _optional("attn_state", "enum:DecodeOnly"),
    ),
    "backend.__init__": (
        *tuple(
            _required(name)
            for name in (
                "self",
                "num_heads",
                "head_size",
                "scale",
                "num_kv_heads",
                "alibi_slopes",
                "sliding_window",
                "kv_cache_dtype",
                "logits_soft_cap",
                "attn_type",
                "kv_sharing_target_layer_name",
            )
        ),
        _optional("sinks"),
        ("kwargs", "VAR_KEYWORD", "required"),
    ),
    "backend.forward": (
        *tuple(
            _required(name)
            for name in (
                "self",
                "layer",
                "query",
                "key",
                "value",
                "kv_cache",
                "attn_metadata",
            )
        ),
        _optional("output"),
        _optional("output_scale"),
        _optional("output_block_scale"),
    ),
    "backend.forward_fused_infer_attention": (
        *tuple(
            _required(name)
            for name in (
                "self",
                "query",
                "key",
                "value",
                "attn_metadata",
                "output",
            )
        ),
        _optional("kv_cache"),
    ),
}

_RUNNER_SIGNATURE = (
    *tuple(
        _required(name)
        for name in (
            "self",
            "num_tokens",
            "num_reqs",
            "max_query_len",
        )
    ),
    _optional("num_tokens_padded"),
    _optional("num_reqs_padded"),
    _optional("ubatch_slices"),
    _optional("logits_indices"),
    _optional("use_spec_decode", "false"),
    _optional("for_cudagraph_capture", "false"),
    _optional("num_scheduled_tokens"),
    _optional("num_scheduled_tokens_np"),
    _optional("cascade_attn_prefix_lens"),
)


def _disabled_config() -> PluginConfig:
    return PluginConfig(
        kernel=TriangleMixConfig(
            enabled=False,
            layer_indices=frozenset(),
        )
    )


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _strict_requested(vllm_config: object | None) -> bool:
    additional = getattr(vllm_config, "additional_config", None)
    if isinstance(additional, dict):
        section = additional.get("trianglemix", additional)
        if isinstance(section, dict) and "strict" in section:
            return _bool_value(section["strict"])
    return _bool_value(
        os.getenv("VLLM_ASCEND_TRIANGLE_MIX_STRICT", "false")
    )


def _resolve_config(owner: object) -> PluginConfig:
    cached = getattr(owner, _CONFIG_ATTR, None)
    if isinstance(cached, PluginConfig):
        return cached
    vllm_config = getattr(owner, "vllm_config", None)
    try:
        config = resolve_plugin_config(vllm_config)
    except Exception:
        runtime_stats().record_runtime_error(stage="config")
        if _strict_requested(vllm_config):
            raise
        logger.exception(
            "TriangleMix configuration is invalid; using official FIA"
        )
        config = _disabled_config()
    setattr(owner, _CONFIG_ATTR, config)
    runtime_stats().configure(
        config.stats_recent_capacity,
        config.stats_log_interval,
    )
    return config


def _ensure_native_ready(config: PluginConfig) -> bool:
    """Load the bundled native runtime only for an enabled configuration."""
    if not config.enabled:
        return False
    try:
        # Importing native is intentionally delayed.  Feature-off workers must
        # not inspect package resources, alter OPP paths, or dlopen libraries.
        from .native import ensure_native_loaded

        status = ensure_native_loaded(True, strict=config.strict)
        return bool(getattr(status, "loaded", status))
    except Exception:
        runtime_stats().record_runtime_error(stage="native_bootstrap")
        if config.strict:
            raise
        logger.exception(
            "TriangleMix native runtime is unavailable; using official FIA"
        )
        return False


def _as_int_list(value: object | None) -> list[int] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        return [int(item) for item in value]  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return None


def _state_name(value: object) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value)


@dataclass(frozen=True)
class _MetadataFingerprint:
    generation: int | None
    prompt_step: int | None
    state_name: str
    query_ends: tuple[int, ...]
    seq_lens: tuple[int, ...]
    prompt_lens: tuple[int, ...]
    num_decodes: int
    num_prefills: int


@dataclass(frozen=True)
class _PlanCacheEntry:
    plan: TriangleBatchPlan
    fingerprint: _MetadataFingerprint
    prompt_lens: tuple[int, ...]
    prompt_source: str
    generation: int | None
    prompt_step: int | None


@dataclass(frozen=True)
class _PromptResolution:
    values: tuple[int, ...]
    source: str
    step_token: int | None = None

    @property
    def available(self) -> bool:
        return bool(self.values) and self.source != "missing"


class _MetadataPlanStore:
    """Fallback cache for metadata classes that disallow new attributes."""

    def __init__(self, capacity: int = 2048) -> None:
        self._capacity = capacity
        self._lock = threading.Lock()
        self._items: OrderedDict[
            int,
            tuple[
                weakref.ReferenceType[Any] | None,
                object,
                _PlanCacheEntry,
            ],
        ] = OrderedDict()
        self._graph_items: OrderedDict[
            int, tuple[weakref.ReferenceType[Any] | None, object]
        ] = OrderedDict()

    def get(self, metadata: object) -> _PlanCacheEntry | None:
        plan = getattr(metadata, _PLAN_ATTR, None)
        fingerprint = getattr(
            metadata,
            _PLAN_FINGERPRINT_ATTR,
            None,
        )
        prompt_lens = getattr(metadata, _PROMPT_LENS_ATTR, None)
        prompt_source = getattr(
            metadata,
            _PROMPT_SOURCE_ATTR,
            None,
        )
        if (
            isinstance(plan, TriangleBatchPlan)
            and isinstance(fingerprint, _MetadataFingerprint)
            and isinstance(prompt_lens, tuple)
            and isinstance(prompt_source, str)
        ):
            return _PlanCacheEntry(
                plan=plan,
                fingerprint=fingerprint,
                prompt_lens=prompt_lens,
                prompt_source=prompt_source,
                generation=fingerprint.generation,
                prompt_step=fingerprint.prompt_step,
            )
        identity = id(metadata)
        with self._lock:
            entry = self._items.get(identity)
            if entry is None:
                return None
            reference, strong_reference, cache_entry = entry
            cached_object = (
                reference() if reference is not None else strong_reference
            )
            if cached_object is not metadata:
                self._items.pop(identity, None)
                return None
            self._items.move_to_end(identity)
            return cache_entry

    def put(
        self,
        metadata: object,
        entry: _PlanCacheEntry,
    ) -> None:
        try:
            setattr(metadata, _PLAN_ATTR, entry.plan)
            setattr(
                metadata,
                _PLAN_FINGERPRINT_ATTR,
                entry.fingerprint,
            )
            setattr(
                metadata,
                _PROMPT_LENS_ATTR,
                entry.prompt_lens,
            )
            setattr(
                metadata,
                _PROMPT_SOURCE_ATTR,
                entry.prompt_source,
            )
            return
        except (AttributeError, TypeError):
            pass
        try:
            reference: weakref.ReferenceType[Any] | None = weakref.ref(
                metadata
            )
            strong_reference: object = None
        except TypeError:
            reference = None
            strong_reference = metadata
        with self._lock:
            self._items[id(metadata)] = (
                reference,
                strong_reference,
                entry,
            )
            self._items.move_to_end(id(metadata))
            while len(self._items) > self._capacity:
                self._items.popitem(last=False)

    def mark_graph(self, metadata: object) -> None:
        try:
            reference: weakref.ReferenceType[Any] | None = weakref.ref(
                metadata
            )
            strong_reference: object = None
        except TypeError:
            reference = None
            strong_reference = metadata
        with self._lock:
            self._graph_items[id(metadata)] = (
                reference,
                strong_reference,
            )
            self._graph_items.move_to_end(id(metadata))
            while len(self._graph_items) > self._capacity:
                self._graph_items.popitem(last=False)

    def is_graph(self, metadata: object) -> bool:
        identity = id(metadata)
        with self._lock:
            entry = self._graph_items.get(identity)
            if entry is None:
                return False
            reference, strong_reference = entry
            cached_object = (
                reference() if reference is not None else strong_reference
            )
            if cached_object is not metadata:
                self._graph_items.pop(identity, None)
                return False
            self._graph_items.move_to_end(identity)
            return True


_METADATA_PLANS = _MetadataPlanStore()


def _prompt_lens(
    *,
    metadata: object,
    common_metadata: object | None,
) -> _PromptResolution:
    # Prefer an official field already materialised on AscendMetadata.  This
    # avoids depending on the builder's positional calling convention.
    for name in (
        "prompt_lens_list",
        "prompt_lens_cpu",
        "num_prompt_tokens_cpu",
    ):
        value = _as_int_list(getattr(metadata, name, None))
        if value:
            return _PromptResolution(
                values=tuple(value),
                source=f"metadata:{name}",
            )
    runner_prompt = _as_int_list(
        getattr(metadata, _RUNNER_PROMPT_ATTR, None)
    )
    runner_step = getattr(metadata, _RUNNER_STEP_ATTR, None)
    if (
        runner_prompt
        and isinstance(runner_step, int)
        and not isinstance(runner_step, bool)
        and runner_step > 0
    ):
        return _PromptResolution(
            values=tuple(runner_prompt),
            source="runner_private",
            step_token=runner_step,
        )
    if common_metadata is not None:
        for name in (
            "num_prompt_tokens_cpu",
            "prompt_lens_cpu",
            "prompt_lens",
        ):
            value = _as_int_list(
                getattr(common_metadata, name, None)
            )
            if value:
                return _PromptResolution(
                    values=tuple(value),
                    source=f"common:{name}",
                )
    return _PromptResolution(values=(), source="missing")


def _builder_generation(metadata: object) -> int | None:
    value = getattr(metadata, _GENERATION_ATTR, None)
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    ):
        return value
    return None


def _start_builder_generation(metadata: object) -> int | None:
    generation = next(_BUILDER_GENERATIONS)
    try:
        setattr(metadata, _GENERATION_ATTR, generation)
    except (AttributeError, TypeError):
        # Common-only prompt provenance cannot be made safe on a metadata
        # class that rejects the scheduler-step token.  Explicit official
        # prompt fields remain independently revalidatable.
        return None
    return generation


def _metadata_fingerprint(
    metadata: object,
    *,
    prompt_lens: list[int] | tuple[int, ...] | None,
    generation: int | None,
    prompt_step: int | None,
) -> _MetadataFingerprint:
    query_ends = _as_int_list(
        getattr(metadata, "actual_seq_lengths_q", None)
    )
    seq_lens = _as_int_list(getattr(metadata, "seq_lens_list", None))
    if seq_lens is None:
        seq_lens = _as_int_list(getattr(metadata, "seq_lens", None))
    return _MetadataFingerprint(
        generation=generation,
        prompt_step=prompt_step,
        state_name=_state_name(
            getattr(metadata, "attn_state", "")
        ),
        query_ends=tuple(query_ends or ()),
        seq_lens=tuple(seq_lens or ()),
        prompt_lens=tuple(prompt_lens or ()),
        num_decodes=int(getattr(metadata, "num_decodes", 0)),
        num_prefills=int(getattr(metadata, "num_prefills", 0)),
    )


def _build_plan_entry(
    *,
    metadata: object,
    common_metadata: object | None,
    config: PluginConfig,
    generation: int | None,
) -> _PlanCacheEntry:
    query_ends = _as_int_list(
        getattr(metadata, "actual_seq_lengths_q", None)
    )
    seq_lens = _as_int_list(getattr(metadata, "seq_lens_list", None))
    if seq_lens is None:
        seq_lens = _as_int_list(getattr(metadata, "seq_lens", None))

    # The final prompt length is scheduler-owned state.  Never infer it from
    # the current sequence length: doing so makes intermediate chunks appear
    # to contain the final dense tail.
    prompt = _prompt_lens(
        metadata=metadata,
        common_metadata=common_metadata,
    )
    if prompt.source.startswith("common:") and generation is None:
        prompt = _PromptResolution(values=(), source="missing")
    plan = build_batch_plan(
        state_name=_state_name(getattr(metadata, "attn_state", "")),
        cumulative_query_ends=query_ends,
        seq_lens=seq_lens,
        prompt_lens=(
            prompt.values if prompt.available else None
        ),
        num_decodes=int(getattr(metadata, "num_decodes", 0)),
        num_prefills=int(getattr(metadata, "num_prefills", 0)),
        config=config.kernel,
    )
    return _PlanCacheEntry(
        plan=plan,
        fingerprint=_metadata_fingerprint(
            metadata,
            prompt_lens=prompt.values,
            generation=generation,
            prompt_step=prompt.step_token,
        ),
        prompt_lens=prompt.values,
        prompt_source=prompt.source,
        generation=generation,
        prompt_step=prompt.step_token,
    )


def _missing_plan_entry(
    metadata: object,
    config: PluginConfig,
    generation: int | None,
) -> _PlanCacheEntry:
    plan = build_batch_plan(
        state_name=_state_name(getattr(metadata, "attn_state", "")),
        cumulative_query_ends=None,
        seq_lens=None,
        prompt_lens=None,
        num_decodes=int(getattr(metadata, "num_decodes", 0)),
        num_prefills=int(getattr(metadata, "num_prefills", 0)),
        config=config.kernel,
    )
    return _PlanCacheEntry(
        plan=plan,
        fingerprint=_metadata_fingerprint(
            metadata,
            prompt_lens=None,
            generation=generation,
            prompt_step=None,
        ),
        prompt_lens=(),
        prompt_source="missing",
        generation=generation,
        prompt_step=None,
    )


def _get_or_build_plan(
    metadata: object,
    config: PluginConfig,
) -> TriangleBatchPlan:
    cached = _METADATA_PLANS.get(metadata)
    current_generation = _builder_generation(metadata)
    if cached is not None:
        if _METADATA_PLANS.is_graph(metadata):
            return cached.plan
        same_core = (
            _metadata_fingerprint(
                metadata,
                prompt_lens=cached.prompt_lens,
                generation=current_generation,
                prompt_step=cached.prompt_step,
            )
            == cached.fingerprint
        )
        current_prompt = _prompt_lens(
            metadata=metadata,
            common_metadata=None,
        )
        if current_prompt.available:
            current_fingerprint = _metadata_fingerprint(
                metadata,
                prompt_lens=current_prompt.values,
                generation=current_generation,
                prompt_step=current_prompt.step_token,
            )
            if current_fingerprint == cached.fingerprint:
                return cached.plan
        elif (
            same_core
            and (
                (
                    cached.prompt_source.startswith("common:")
                    and current_generation is not None
                    and current_generation == cached.generation
                )
                or cached.prompt_source == "missing"
            )
        ):
            # A prompt captured from CommonAttentionMetadata is attached to
            # this exact immutable scheduler fingerprint for reuse across
            # layers.  It is never carried into a changed scheduler step.
            return cached.plan
    try:
        entry = _build_plan_entry(
            metadata=metadata,
            common_metadata=None,
            config=config,
            generation=current_generation,
        )
    except Exception:
        runtime_stats().record_runtime_error(stage="metadata_plan")
        if config.strict:
            raise
        entry = _missing_plan_entry(
            metadata,
            config,
            current_generation,
        )
    _METADATA_PLANS.put(metadata, entry)
    return entry.plan


def _mark_graph_capture(metadata: object) -> None:
    _METADATA_PLANS.mark_graph(metadata)
    try:
        setattr(metadata, _GRAPH_ATTR, True)
    except (AttributeError, TypeError):
        # AscendMetadata is a regular dataclass in supported releases.  If a
        # future slotted type appears, runtime capture detection remains the
        # authoritative fallback gate.
        pass


def _metadata_builder_patch(builder_cls: type[Any]) -> None:
    if getattr(builder_cls, _PATCH_MARKER, False):
        return
    original_build = builder_cls.build

    @functools.wraps(original_build)
    def build(self: object, *args: Any, **kwargs: Any) -> object:
        metadata = original_build(self, *args, **kwargs)
        config = _resolve_config(self)
        if not config.enabled:
            return metadata
        generation = _start_builder_generation(metadata)
        common_metadata = kwargs.get("common_attn_metadata")
        if common_metadata is None and len(args) >= 2:
            common_metadata = args[1]
        try:
            entry = _build_plan_entry(
                metadata=metadata,
                common_metadata=common_metadata,
                config=config,
                generation=generation,
            )
        except Exception:
            runtime_stats().record_runtime_error(stage="metadata_build")
            if config.strict:
                raise
            logger.exception(
                "TriangleMix metadata planning failed; using official FIA"
            )
            entry = _missing_plan_entry(
                metadata,
                config,
                generation,
            )
        _METADATA_PLANS.put(metadata, entry)
        return metadata

    builder_cls.build = build

    original_graph_build = getattr(
        builder_cls, "build_for_graph_capture", None
    )
    if callable(original_graph_build):

        @functools.wraps(original_graph_build)
        def build_for_graph_capture(
            self: object, *args: Any, **kwargs: Any
        ) -> object:
            metadata = original_graph_build(self, *args, **kwargs)
            _mark_graph_capture(metadata)
            return metadata

        builder_cls.build_for_graph_capture = build_for_graph_capture
    setattr(builder_cls, _PATCH_MARKER, True)


def _runner_prompt_candidate(
    runner: object,
    arguments: dict[str, object],
) -> tuple[int, ...] | None:
    try:
        num_tokens = int(arguments["num_tokens"])
        num_reqs = int(arguments["num_reqs"])
    except (KeyError, TypeError, ValueError):
        return None
    if num_reqs != 1:
        return None
    num_reqs_padded = arguments.get("num_reqs_padded")
    if num_reqs_padded is not None:
        try:
            if int(num_reqs_padded) != 1:
                return None
        except (TypeError, ValueError):
            return None
    num_tokens_padded = arguments.get("num_tokens_padded")
    if num_tokens_padded is not None:
        try:
            if int(num_tokens_padded) != num_tokens:
                return None
        except (TypeError, ValueError):
            return None
    if (
        arguments.get("ubatch_slices") is not None
        or bool(arguments.get("use_spec_decode", False))
        or bool(arguments.get("for_cudagraph_capture", False))
        or getattr(runner, "speculative_config", None) is not None
        or bool(getattr(runner, "use_async_spec_decode", False))
    ):
        return None
    input_batch = getattr(runner, "input_batch", None)
    values = getattr(
        input_batch,
        "num_prompt_tokens_cpu_tensor",
        None,
    )
    if values is None:
        return None
    device = getattr(values, "device", None)
    if device is not None and str(device) != "cpu":
        return None
    try:
        prompt_lens = _as_int_list(values[:1])
    except (IndexError, TypeError):
        return None
    if (
        prompt_lens is None
        or len(prompt_lens) != 1
        or prompt_lens[0] <= 0
    ):
        return None
    return tuple(prompt_lens)


def _attention_metadata_leaves(value: object) -> tuple[object, ...]:
    leaves: list[object] = []
    seen: set[int] = set()

    def visit(item: object) -> None:
        identity = id(item)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
            return
        if item is not None and (
            hasattr(item, "attn_state")
            or hasattr(item, "actual_seq_lengths_q")
            or hasattr(item, "seq_lens_list")
        ):
            leaves.append(item)

    visit(value)
    return tuple(leaves)


def _clear_runner_prompt(metadata: object) -> None:
    for name in (_RUNNER_PROMPT_ATTR, _RUNNER_STEP_ATTR):
        try:
            delattr(metadata, name)
        except AttributeError:
            try:
                setattr(metadata, name, None)
            except (AttributeError, TypeError):
                pass


def _attach_runner_prompt(
    metadata: object,
    *,
    prompt_lens: tuple[int, ...],
    step_token: int,
) -> bool:
    _clear_runner_prompt(metadata)
    try:
        setattr(metadata, _RUNNER_PROMPT_ATTR, prompt_lens)
        setattr(metadata, _RUNNER_STEP_ATTR, step_token)
    except (AttributeError, TypeError):
        _clear_runner_prompt(metadata)
        return False
    return True


def _patch_model_runner_module(
    runner_module: object,
    *,
    versions: tuple[str, str] | None = None,
) -> bool:
    _validate_versions(versions or _installed_versions())
    runner_cls = getattr(runner_module, "NPUModelRunner", None)
    if not isinstance(runner_cls, type):
        raise _UnsupportedCompatibilityError(
            "unsupported_version: NPUModelRunner is missing"
        )
    if bool(getattr(runner_cls, _RUNNER_PATCH_MARKER, False)):
        return True
    original = getattr(
        runner_cls,
        "_build_attention_metadata",
        None,
    )
    if _signature_key(original) != _RUNNER_SIGNATURE:
        raise _UnsupportedCompatibilityError(
            "unsupported_version: NPUModelRunner."
            "_build_attention_metadata signature is not allowlisted"
        )
    signature = inspect.signature(original)

    @functools.wraps(original)
    def build_attention_metadata(
        self: object,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        prompt_lens = _runner_prompt_candidate(
            self,
            dict(bound.arguments),
        )
        result = original(self, *args, **kwargs)
        metadata_root = (
            result[0]
            if isinstance(result, tuple) and result
            else result
        )
        step_token = next(_RUNNER_STEPS)
        for metadata in _attention_metadata_leaves(metadata_root):
            _clear_runner_prompt(metadata)
            if prompt_lens is not None:
                _attach_runner_prompt(
                    metadata,
                    prompt_lens=prompt_lens,
                    step_token=step_token,
                )
        return result

    captured = [
        (
            runner_cls,
            "_build_attention_metadata",
            vars(runner_cls).get(
                "_build_attention_metadata",
                _ATTRIBUTE_MISSING,
            ),
        ),
        (
            runner_cls,
            _RUNNER_PATCH_MARKER,
            vars(runner_cls).get(
                _RUNNER_PATCH_MARKER,
                _ATTRIBUTE_MISSING,
            ),
        ),
    ]
    try:
        runner_cls._build_attention_metadata = build_attention_metadata
        setattr(runner_cls, _RUNNER_PATCH_MARKER, True)
    except Exception:
        _restore_attributes(captured)
        raise
    return True


def _ensure_model_runner_patch(*, strict: bool) -> bool:
    runner_module = sys.modules.get(_RUNNER_MODULE)
    if runner_module is None:
        # The supported 0.23 runner imports attention_v1 before constructing
        # an attention backend.  Absence here is not enough evidence to import
        # an unknown runner module from a plugin hook.
        return False
    with _RUNNER_PATCH_LOCK:
        try:
            return _patch_model_runner_module(runner_module)
        except _UnsupportedCompatibilityError as exc:
            if strict:
                raise
            rejection = (id(runner_module), str(exc))
            if rejection not in _RUNNER_REJECTIONS:
                _RUNNER_REJECTIONS.add(rejection)
                runtime_stats().record_runtime_error(
                    stage="unsupported_version"
                )
                logger.warning(
                    "TriangleMix runner seam rejected "
                    "(reason=unsupported_version); "
                    "official FIA remains active: %s",
                    exc,
                )
            return False


def _dtype_name(dtype: object) -> str:
    return str(dtype).removeprefix("torch.")


def _tensor_stride(tensor: object) -> tuple[int, ...] | None:
    stride = getattr(tensor, "stride", None)
    if not callable(stride):
        return None
    try:
        return tuple(int(item) for item in stride())
    except (TypeError, ValueError):
        return None


def _tensor_signature(tensor: object) -> tuple[object, ...]:
    shape = tuple(int(item) for item in getattr(tensor, "shape", ()))
    return (
        id(tensor),
        shape,
        _dtype_name(getattr(tensor, "dtype", None)),
        str(getattr(tensor, "device", None)),
        _tensor_stride(tensor),
    )


@dataclass
class _KvCapabilityCache:
    signature: tuple[object, ...] | None = None
    reason: FallbackReason = FallbackReason.KV_CACHE_UNSUPPORTED

    def check(
        self,
        key_cache: object,
        value_cache: object,
    ) -> FallbackReason:
        signature = (
            *_tensor_signature(key_cache),
            *_tensor_signature(value_cache),
        )
        if signature == self.signature:
            return self.reason
        self.signature = signature
        self.reason = _validate_static_kv(key_cache, value_cache)
        return self.reason


def _is_contiguous(tensor: object) -> bool:
    predicate = getattr(tensor, "is_contiguous", None)
    return bool(callable(predicate) and predicate())


def _validate_static_kv(
    key_cache: object,
    value_cache: object,
) -> FallbackReason:
    key_shape = tuple(int(item) for item in getattr(key_cache, "shape", ()))
    value_shape = tuple(
        int(item) for item in getattr(value_cache, "shape", ())
    )
    if (
        len(key_shape) != 4
        or key_shape[1:]
        != (
            SUPPORTED_CACHE_BLOCK_SIZE,
            SUPPORTED_KV_HEADS,
            SUPPORTED_HEAD_SIZE,
        )
        or value_shape != key_shape
        or _dtype_name(getattr(key_cache, "dtype", None)) != "bfloat16"
        or _dtype_name(getattr(value_cache, "dtype", None)) != "bfloat16"
        or getattr(key_cache, "device", None)
        != getattr(value_cache, "device", None)
        or not _is_contiguous(key_cache)
        or not _is_contiguous(value_cache)
    ):
        return FallbackReason.KV_CACHE_UNSUPPORTED
    return FallbackReason.NONE


def _validate_query(
    backend: object,
    query: object,
    output: object,
    plan: TriangleBatchPlan,
) -> FallbackReason:
    shape = tuple(int(item) for item in getattr(query, "shape", ()))
    if (
        len(shape) != 3
        or shape[1:] != (SUPPORTED_QUERY_HEADS, SUPPORTED_HEAD_SIZE)
        or shape[0] != plan.requests[0].query_len
        or _dtype_name(getattr(query, "dtype", None)) != "bfloat16"
        or not _is_contiguous(query)
        or int(getattr(backend, "num_heads", -1))
        != SUPPORTED_QUERY_HEADS
        or int(getattr(backend, "num_kv_heads", -1))
        != SUPPORTED_KV_HEADS
        or int(getattr(backend, "head_size", -1))
        != SUPPORTED_HEAD_SIZE
    ):
        return FallbackReason.QUERY_UNSUPPORTED
    output_shape = tuple(
        int(item) for item in getattr(output, "shape", ())
    )
    if (
        not output_shape
        or output_shape[0] < shape[0]
        or _dtype_name(getattr(output, "dtype", None)) != "bfloat16"
        or getattr(output, "device", None)
        != getattr(query, "device", None)
        or not _is_contiguous(output)
    ):
        return FallbackReason.QUERY_UNSUPPORTED
    return FallbackReason.NONE


def _validate_block_table(
    block_table: object,
    *,
    query: object,
    seq_len: int,
) -> FallbackReason:
    shape = tuple(
        int(item) for item in getattr(block_table, "shape", ())
    )
    required_pages = (
        seq_len + SUPPORTED_CACHE_BLOCK_SIZE - 1
    ) // SUPPORTED_CACHE_BLOCK_SIZE
    if (
        len(shape) != 2
        or shape[0] != 1
        or shape[1] < required_pages
        or _dtype_name(getattr(block_table, "dtype", None)) != "int32"
        or getattr(block_table, "device", None)
        != getattr(query, "device", None)
        or not _is_contiguous(block_table)
    ):
        return FallbackReason.BLOCK_TABLE_UNSUPPORTED
    return FallbackReason.NONE


def _tensor_parallel_world_size(attention_module: object) -> int | None:
    getter = getattr(
        attention_module,
        "get_tensor_model_parallel_world_size",
        None,
    )
    if callable(getter):
        try:
            return int(getter())
        except Exception:
            return None
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_world_size,
        )

        return int(get_tensor_model_parallel_world_size())
    except Exception:
        return None


def _context_parallel_enabled(attention_module: object) -> bool | None:
    getter = getattr(attention_module, "enable_cp", None)
    if not callable(getter):
        return None
    try:
        return bool(getter())
    except Exception:
        return None


def _capturing(attention_module: object, metadata: object) -> bool:
    if bool(getattr(metadata, _GRAPH_ATTR, False)) or (
        _METADATA_PLANS.is_graph(metadata)
    ):
        return True
    context = getattr(attention_module, "_EXTRA_CTX", None)
    # Supported attention_v1 modules always expose _EXTRA_CTX.  Treat a
    # missing/invalid context as unsupported instead of risking a custom
    # launch during graph capture.
    if context is None:
        return True
    try:
        return bool(getattr(context, "capturing"))
    except Exception:
        return True


def _model_reason(
    backend: object,
    metadata: object,
    attention_module: object,
    config: PluginConfig,
) -> FallbackReason:
    if _capturing(attention_module, metadata):
        return FallbackReason.GRAPH_CAPTURE
    tp_world_size = _tensor_parallel_world_size(attention_module)
    if tp_world_size != 1:
        return FallbackReason.TENSOR_PARALLEL
    context_parallel = _context_parallel_enabled(attention_module)
    if context_parallel is not False:
        return FallbackReason.CONTEXT_PARALLEL
    if not bool(getattr(metadata, "causal", False)):
        return FallbackReason.NON_CAUSAL
    if not config.kernel.has_supported_geometry:
        return FallbackReason.GEOMETRY_UNSUPPORTED
    if (
        getattr(backend, "sliding_window", None) is not None
        or getattr(backend, "sinks", None) is not None
        or getattr(backend, "alibi_slopes", None) is not None
        or bool(getattr(backend, "enable_hamming_sparse", False))
        or bool(getattr(backend, "enable_c8_quant", False))
    ):
        return FallbackReason.MODEL_UNSUPPORTED
    attention_type = str(getattr(backend, "attn_type", "")).upper()
    if "ENCODER" in attention_type:
        return FallbackReason.MODEL_UNSUPPORTED
    return FallbackReason.NONE


def _dispatch_reason(
    backend: object,
    query: object,
    metadata: object,
    output: object,
    attention_module: object,
    config: PluginConfig,
    plan: TriangleBatchPlan,
) -> FallbackReason:
    if _capturing(attention_module, metadata):
        return FallbackReason.GRAPH_CAPTURE
    if not plan.direct:
        return plan.primary_reason
    if not bool(getattr(backend, _NATIVE_READY_ATTR, False)):
        return FallbackReason.ADAPTER_UNAVAILABLE
    reason = _model_reason(
        backend,
        metadata,
        attention_module,
        config,
    )
    if reason is not FallbackReason.NONE:
        return reason
    reason = _validate_query(backend, query, output, plan)
    if reason is not FallbackReason.NONE:
        return reason
    key_cache = getattr(backend, "key_cache", None)
    value_cache = getattr(backend, "value_cache", None)
    if key_cache is None or value_cache is None:
        return FallbackReason.KV_CACHE_UNSUPPORTED
    cache = getattr(backend, _KV_CAPABILITY_ATTR, None)
    if not isinstance(cache, _KvCapabilityCache):
        cache = _KvCapabilityCache()
        setattr(backend, _KV_CAPABILITY_ATTR, cache)
    reason = cache.check(key_cache, value_cache)
    if reason is not FallbackReason.NONE:
        return reason
    if (
        getattr(key_cache, "device", None)
        != getattr(query, "device", None)
    ):
        return FallbackReason.KV_CACHE_UNSUPPORTED
    block_table = getattr(metadata, "block_tables", None)
    if block_table is None:
        return FallbackReason.BLOCK_TABLE_UNSUPPORTED
    return _validate_block_table(
        block_table,
        query=query,
        seq_len=plan.requests[0].seq_len,
    )


def _output_view(
    output: Any,
    *,
    num_tokens: int,
    num_heads: int,
    head_size: int,
) -> Any:
    view = output[:num_tokens]
    if tuple(int(item) for item in view.shape) == (
        num_tokens,
        num_heads,
        head_size,
    ):
        return view
    return view.view(num_tokens, num_heads, head_size)


def _log_stats_if_due(stats: object) -> None:
    payload = stats.structured_log_if_due()
    if payload is not None:
        logger.info(
            "%s",
            json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )


def _backend_patch(
    backend_cls: type[Any],
    attention_module: object,
) -> None:
    if getattr(backend_cls, _PATCH_MARKER, False):
        return
    original_init = backend_cls.__init__
    original_forward = backend_cls.forward
    original_fia = backend_cls.forward_fused_infer_attention

    @functools.wraps(original_init)
    def init(self: object, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if hasattr(self, "_triangle_mix_layer_enabled"):
            # Development snapshots may initialise their old hook here.  Its
            # predicate is disabled below; pin the per-layer flag as well so
            # the old forward wrapper neither reselects nor logs a layer.
            setattr(self, "_triangle_mix_layer_enabled", False)
        config = _resolve_config(self)
        setattr(self, _LAYER_NAME_ATTR, None)
        setattr(self, _LAYER_SELECTED_ATTR, False)
        setattr(self, _KV_CAPABILITY_ATTR, _KvCapabilityCache())
        if config.enabled:
            _ensure_model_runner_patch(strict=config.strict)
        setattr(
            self,
            _NATIVE_READY_ATTR,
            _ensure_native_ready(config) if config.enabled else False,
        )

    @functools.wraps(original_forward)
    def forward(
        self: object,
        layer: object,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        config = _resolve_config(self)
        layer_name = str(getattr(layer, "layer_name", ""))
        setattr(self, _LAYER_NAME_ATTR, layer_name)
        setattr(
            self,
            _LAYER_SELECTED_ATTR,
            config.enabled and config.kernel.uses_layer(layer_name),
        )
        return original_forward(self, layer, *args, **kwargs)

    @functools.wraps(original_fia)
    def forward_fused_infer_attention(
        self: object,
        query: Any,
        key: Any,
        value: Any,
        attn_metadata: object,
        output: Any,
        kv_cache: Any = None,
    ) -> Any:
        config = _resolve_config(self)
        if (
            not config.enabled
            or not bool(getattr(self, _LAYER_SELECTED_ATTR, False))
            or attn_metadata is None
        ):
            return original_fia(
                self,
                query,
                key,
                value,
                attn_metadata,
                output,
                kv_cache,
            )

        plan = _get_or_build_plan(attn_metadata, config)
        reason = _dispatch_reason(
            self,
            query,
            attn_metadata,
            output,
            attention_module,
            config,
            plan,
        )
        layer_name = getattr(self, _LAYER_NAME_ATTR, None)
        layer_index = (
            extract_layer_index(layer_name)
            if isinstance(layer_name, str)
            else None
        )
        stats = runtime_stats()
        if reason is not FallbackReason.NONE:
            # Graph-capture metadata is a reusable dummy, not a scheduled user
            # request.  Keep its per-layer FIA counter without contaminating
            # request planner eligibility.
            if reason is not FallbackReason.GRAPH_CAPTURE:
                stats.record_planner_decision(plan)
            stats.record_layer_fallback(
                plan_id=plan.plan_id,
                layer_index=layer_index,
                reason=reason,
                request_indices=tuple(
                    request.request_index
                    for request in plan.requests[:plan.batch_size]
                ),
            )
            _log_stats_if_due(stats)
            return original_fia(
                self,
                query,
                key,
                value,
                attn_metadata,
                output,
                kv_cache,
            )

        request = plan.requests[0]
        num_tokens = request.query_len
        output_view = _output_view(
            output,
            num_tokens=num_tokens,
            num_heads=int(getattr(self, "num_heads")),
            head_size=int(getattr(self, "head_size")),
        )
        started = time.perf_counter_ns()
        try:
            triangle_direct_paged_attention(
                query=query[:num_tokens],
                key_cache=getattr(self, "key_cache"),
                value_cache=getattr(self, "value_cache"),
                block_table=getattr(attn_metadata, "block_tables"),
                seq_len=request.seq_len,
                prompt_len=request.prompt_len,
                output=output_view,
                softmax_scale=float(getattr(self, "scale")),
                config=config.kernel,
            )
        except Exception:
            stats.record_runtime_error(stage="direct_launch")
            if config.strict:
                raise
            logger.exception(
                "TriangleMix direct launch failed at layer %s; "
                "using official FIA",
                layer_name,
            )
            reason = FallbackReason.DIRECT_LAUNCH_ERROR
            stats.record_planner_decision(plan)
            stats.record_layer_fallback(
                plan_id=plan.plan_id,
                layer_index=layer_index,
                reason=reason,
                request_indices=tuple(
                    request.request_index
                    for request in plan.requests[:plan.batch_size]
                ),
            )
            _log_stats_if_due(stats)
            return original_fia(
                self,
                query,
                key,
                value,
                attn_metadata,
                output,
                kv_cache,
            )
        host_enqueue_ns = time.perf_counter_ns() - started
        stats.record_planner_decision(plan)
        stats.record_layer_dispatch(
            plan_id=plan.plan_id,
            layer_index=layer_index,
            saved_qk=request.saved_qk,
            host_enqueue_ns=host_enqueue_ns,
            request_index=request.request_index,
        )
        _log_stats_if_due(stats)
        return output

    backend_cls.__init__ = init
    backend_cls.forward = forward
    backend_cls.forward_fused_infer_attention = (
        forward_fused_infer_attention
    )

    # Some 0.23.0rc1 development snapshots contain an older in-tree
    # TriangleMix hook.  Disable only its predicate so all plugin fallbacks
    # reach the snapshot's unchanged official FIA implementation.
    legacy_predicate = getattr(
        backend_cls, "_can_use_triangle_mix", None
    )
    if callable(legacy_predicate):
        setattr(
            backend_cls,
            "_vllm_ascend_trianglemix_legacy_can_use",
            legacy_predicate,
        )

        def legacy_disabled(
            self: object, *args: Any, **kwargs: Any
        ) -> bool:
            del self, args, kwargs
            return False

        backend_cls._can_use_triangle_mix = legacy_disabled
        legacy_loader = getattr(
            attention_module,
            "load_triangle_mix_adapter_if_enabled",
            None,
        )
        if callable(legacy_loader):
            setattr(
                attention_module,
                "_vllm_ascend_trianglemix_legacy_loader",
                legacy_loader,
            )

            def legacy_loader_disabled(
                enabled: bool, path: object
            ) -> bool:
                del enabled, path
                return False

            # The wheel owns native bootstrap.  Prevent development snapshots
            # from pre-registering an external adapter before our bundled OPP
            # and adapter are loaded.
            attention_module.load_triangle_mix_adapter_if_enabled = (
                legacy_loader_disabled
            )
    setattr(backend_cls, _PATCH_MARKER, True)


def _preflight_attention_module(
    attention_module: object,
    *,
    versions: tuple[str, str],
) -> tuple[type[Any], type[Any]]:
    _validate_versions(versions)
    builder_cls = getattr(
        attention_module,
        "AscendAttentionMetadataBuilder",
        None,
    )
    backend_cls = getattr(
        attention_module,
        "AscendAttentionBackendImpl",
        None,
    )
    if not isinstance(builder_cls, type) or not isinstance(
        backend_cls, type
    ):
        raise _UnsupportedCompatibilityError(
            "unsupported_version: vLLM-Ascend attention_v1 "
            "builder or backend class is missing"
        )
    methods = {
        "builder.__init__": builder_cls.__init__,
        "builder.build": getattr(builder_cls, "build", None),
        "builder.build_for_graph_capture": getattr(
            builder_cls,
            "build_for_graph_capture",
            None,
        ),
        "backend.__init__": backend_cls.__init__,
        "backend.forward": getattr(backend_cls, "forward", None),
        "backend.forward_fused_infer_attention": getattr(
            backend_cls,
            "forward_fused_infer_attention",
            None,
        ),
    }
    mismatches = [
        name
        for name, expected in _ATTENTION_SIGNATURES.items()
        if _signature_key(methods[name]) != expected
    ]
    if mismatches:
        raise _UnsupportedCompatibilityError(
            "unsupported_version: attention_v1 signature allowlist "
            f"mismatch for {', '.join(mismatches)}"
        )
    return builder_cls, backend_cls


def _restore_attributes(
    captured: list[tuple[object, str, object]],
) -> None:
    for owner, name, value in reversed(captured):
        if value is _ATTRIBUTE_MISSING:
            try:
                delattr(owner, name)
            except AttributeError:
                pass
        else:
            setattr(owner, name, value)


def _patch_attention_module(
    attention_module: object,
    *,
    versions: tuple[str, str] | None = None,
) -> None:
    builder_cls, backend_cls = _preflight_attention_module(
        attention_module,
        versions=versions or _installed_versions(),
    )
    targets = (
        (
            builder_cls,
            (
                "build",
                "build_for_graph_capture",
                _PATCH_MARKER,
            ),
        ),
        (
            backend_cls,
            (
                "__init__",
                "forward",
                "forward_fused_infer_attention",
                "_can_use_triangle_mix",
                "_vllm_ascend_trianglemix_legacy_can_use",
                _PATCH_MARKER,
            ),
        ),
        (
            attention_module,
            (
                "load_triangle_mix_adapter_if_enabled",
                "_vllm_ascend_trianglemix_legacy_loader",
            ),
        ),
    )
    captured = [
        (owner, name, vars(owner).get(name, _ATTRIBUTE_MISSING))
        for owner, names in targets
        for name in names
    ]
    try:
        _metadata_builder_patch(builder_cls)
        _backend_patch(backend_cls, attention_module)
    except Exception:
        _restore_attributes(captured)
        raise


def register() -> None:
    """vLLM ``general_plugins`` entry point."""
    global _REGISTERED
    with _PATCH_LOCK:
        if _REGISTERED:
            return
        try:
            # vLLM-Ascend 0.23 initializes DeviceOperator through the
            # top-level ops package.  Importing attention_v1 first re-enters
            # device_op while it is only partially initialized and raises a
            # circular-import error during vLLM general-plugin discovery.
            import vllm_ascend.ops  # noqa: F401
            from vllm_ascend.attention import attention_v1

            _patch_attention_module(attention_v1)
        except _UnsupportedCompatibilityError:
            runtime_stats().record_runtime_error(
                stage="unsupported_version"
            )
            if _strict_requested(None):
                raise
            logger.warning(
                "TriangleMix plugin rejected the installed framework "
                "(reason=unsupported_version); official FIA remains active",
                exc_info=True,
            )
            return
        except Exception:
            runtime_stats().record_runtime_error(stage="plugin_register")
            if _strict_requested(None):
                raise
            logger.exception(
                "TriangleMix plugin could not patch vLLM-Ascend; "
                "the official backend remains active"
            )
            return
        _REGISTERED = True


__all__ = ["register"]
