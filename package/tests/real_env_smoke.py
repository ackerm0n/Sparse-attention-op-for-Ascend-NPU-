#!/usr/bin/env python3
"""Read-only smoke test against an installed vLLM-Ascend environment.

This script intentionally does not construct a real attention backend, load a
model, allocate tensors, or touch an NPU.  It follows the official Ascend
runtime's import order (``vllm_ascend.ops`` before ``attention_v1``), applies
the wheel's general-plugin entry point, and uses cloned wrappers with sentinel
originals to exercise feature-off and DecodeOnly control flow.  Every
class/module attribute changed by registration is restored before the process
exits.

The only output is one JSON object.  Exit status is zero exactly when every
check passes.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import sys
import traceback
import types
from importlib import metadata
from pathlib import Path
from typing import Any, Callable


# Importing an installed package must not create or refresh site-packages
# bytecode as a side effect of this read-only smoke.
sys.dont_write_bytecode = True

ENTRY_POINT_GROUP = "vllm.general_plugins"
ENTRY_POINT_NAME = "trianglemix"
EXPECTED_ENTRY_POINT = "vllm_ascend_trianglemix.plugin:register"
ASCEND_OPS_MODULE = "vllm_ascend.ops"
ATTENTION_MODULE = "vllm_ascend.attention.attention_v1"
NATIVE_MODULE = "vllm_ascend_trianglemix.native"
ENVIRONMENT_KEYS = (
    "ASCEND_CUSTOM_OPP_PATH",
    "LD_LIBRARY_PATH",
)
FEATURE_ENVIRONMENT = {
    # Keep automatic plugin discovery from pre-patching the classes before
    # this script captures their true installed identities.  The TriangleMix
    # entry point is loaded and invoked explicitly below.
    "VLLM_PLUGINS": "ascend",
    "VLLM_ASCEND_ENABLE_TRIANGLE_MIX": "0",
    "VLLM_ASCEND_TRIANGLE_MIX_LAYERS": "",
}
_MISSING = object()


def _entry_points() -> list[metadata.EntryPoint]:
    discovered = metadata.entry_points()
    if hasattr(discovered, "select"):
        return list(
            discovered.select(
                group=ENTRY_POINT_GROUP,
                name=ENTRY_POINT_NAME,
            )
        )
    return [
        item
        for item in discovered.get(ENTRY_POINT_GROUP, ())
        if item.name == ENTRY_POINT_NAME
    ]


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signature(value: object) -> str | None:
    if not callable(value):
        return None
    return str(inspect.signature(value))


def _make_cell(value: object) -> object:
    def capture() -> object:
        return value

    assert capture.__closure__ is not None
    return capture.__closure__[0]


def _clone_with_freevars(
    function: Callable[..., Any],
    replacements: dict[str, object],
) -> Callable[..., Any]:
    """Clone a wrapper while replacing selected closed-over originals."""
    closure = function.__closure__
    if closure is None:
        raise RuntimeError(
            f"{function.__qualname__} has no closure to probe"
        )
    cells = list(closure)
    freevars = function.__code__.co_freevars
    for name, value in replacements.items():
        try:
            index = freevars.index(name)
        except ValueError as exc:
            raise RuntimeError(
                f"{function.__qualname__} has no {name!r} free variable; "
                f"found {freevars!r}"
            ) from exc
        cells[index] = _make_cell(value)
    clone = types.FunctionType(
        function.__code__,
        function.__globals__,
        name=f"{function.__name__}_smoke_probe",
        argdefs=function.__defaults__,
        closure=tuple(cells),
    )
    clone.__kwdefaults__ = function.__kwdefaults__
    return clone


class _Restorer:
    def __init__(self) -> None:
        self._items: list[tuple[object, str, object]] = []

    def save(self, owner: object, name: str) -> None:
        namespace = vars(owner)
        self._items.append(
            (owner, name, namespace.get(name, _MISSING))
        )

    def restore(self) -> None:
        for owner, name, value in reversed(self._items):
            if value is _MISSING:
                try:
                    delattr(owner, name)
                except AttributeError:
                    pass
            else:
                setattr(owner, name, value)


def _environment_snapshot() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in ENVIRONMENT_KEYS}


def _restore_environment(snapshot: dict[str, str | None]) -> None:
    for name, value in snapshot.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _distribution_name(entry_point: metadata.EntryPoint) -> str | None:
    distribution = getattr(entry_point, "dist", None)
    if distribution is None:
        return None
    try:
        return str(distribution.metadata["Name"])
    except Exception:
        return str(getattr(distribution, "name", None))


def _record_check(
    report: dict[str, Any],
    name: str,
    passed: bool,
    evidence: object | None = None,
) -> None:
    item: dict[str, object] = {"passed": bool(passed)}
    if evidence is not None:
        item["evidence"] = evidence
    report["checks"][name] = item


def _method_state(
    builder_cls: type[Any],
    backend_cls: type[Any],
) -> dict[str, dict[str, object]]:
    owners = {
        "builder": (
            builder_cls,
            ("build", "build_for_graph_capture"),
        ),
        "backend": (
            backend_cls,
            (
                "__init__",
                "forward",
                "forward_fused_infer_attention",
                "_can_use_triangle_mix",
            ),
        ),
    }
    result: dict[str, dict[str, object]] = {}
    for prefix, (owner, names) in owners.items():
        for name in names:
            value = getattr(owner, name, None)
            result[f"{prefix}.{name}"] = {
                "present": callable(value),
                "identity": id(value) if callable(value) else None,
                "signature": _signature(value),
            }
    return result


def _feature_off_init_probe(
    plugin_module: types.ModuleType,
    wrapped_init: Callable[..., Any],
) -> dict[str, object]:
    original_called = 0

    def original_init_sentinel(self: object) -> None:
        nonlocal original_called
        original_called += 1
        self.vllm_config = types.SimpleNamespace(
            additional_config={
                "trianglemix": {
                    "enabled": False,
                    "layers": "",
                }
            }
        )

    probe = _clone_with_freevars(
        wrapped_init,
        {"original_init": original_init_sentinel},
    )
    backend = types.SimpleNamespace()
    native_before = NATIVE_MODULE in sys.modules
    environment_before = _environment_snapshot()
    probe(backend)
    environment_after = _environment_snapshot()
    native_after = NATIVE_MODULE in sys.modules
    ready = getattr(
        backend,
        plugin_module._NATIVE_READY_ATTR,
        None,
    )
    config = getattr(backend, plugin_module._CONFIG_ATTR, None)
    return {
        "passed": (
            original_called == 1
            and ready is False
            and config is not None
            and not config.enabled
            and native_before == native_after
            and environment_before == environment_after
        ),
        "original_init_calls": original_called,
        "native_imported_before": native_before,
        "native_imported_after": native_after,
        "native_ready": ready,
        "config_enabled": (
            None if config is None else bool(config.enabled)
        ),
        "environment_unchanged": (
            environment_before == environment_after
        ),
    }


def _decode_fallback_probe(
    plugin_module: types.ModuleType,
    attention_module: types.ModuleType,
    wrapped_fia: Callable[..., Any],
) -> dict[str, object]:
    original_calls: list[dict[str, object]] = []
    sentinel_result = object()

    def original_fia_sentinel(
        self: object,
        query: object,
        key: object,
        value: object,
        attn_metadata: object,
        output: object,
        kv_cache: object = None,
    ) -> object:
        del self, query, key, value, output
        original_calls.append(
            {
                "state": getattr(
                    getattr(attn_metadata, "attn_state", None),
                    "name",
                    None,
                ),
                "kv_cache_forwarded": kv_cache == "kv-cache-sentinel",
            }
        )
        return sentinel_result

    probe = _clone_with_freevars(
        wrapped_fia,
        {"original_fia": original_fia_sentinel},
    )
    kernel_config = plugin_module.TriangleMixConfig(
        enabled=True,
        layer_indices=frozenset({0}),
    )
    config = plugin_module.PluginConfig(kernel=kernel_config)
    backend = types.SimpleNamespace(
        num_heads=32,
        num_kv_heads=8,
        head_size=128,
        scale=0.088,
    )
    setattr(backend, plugin_module._CONFIG_ATTR, config)
    setattr(backend, plugin_module._LAYER_SELECTED_ATTR, True)
    setattr(
        backend,
        plugin_module._LAYER_NAME_ATTR,
        "model.layers.0.self_attn",
    )
    setattr(backend, plugin_module._NATIVE_READY_ATTR, False)

    metadata_object = types.SimpleNamespace(
        attn_state=attention_module.AscendAttentionState.DecodeOnly,
        actual_seq_lengths_q=[1],
        seq_lens_list=[4096],
        prompt_lens_list=[4096],
        num_decodes=1,
        num_prefills=0,
        causal=True,
        block_tables=None,
    )
    plugin_module.runtime_stats().snapshot(reset=True)
    result = probe(
        backend,
        object(),
        object(),
        object(),
        metadata_object,
        object(),
        "kv-cache-sentinel",
    )
    plan = getattr(
        metadata_object,
        plugin_module._PLAN_ATTR,
        None,
    )
    snapshot = plugin_module.runtime_stats().snapshot(reset=True)
    reason = (
        None if plan is None else plan.primary_reason.value
    )
    return {
        "passed": (
            result is sentinel_result
            and len(original_calls) == 1
            and original_calls[0]["state"] == "DecodeOnly"
            and original_calls[0]["kv_cache_forwarded"] is True
            and reason == "state_unsupported"
            and snapshot["counters"].get("single_launch", 0) == 0
        ),
        "original_fia_calls": len(original_calls),
        "original_call": (
            original_calls[0] if original_calls else None
        ),
        "plan_reason": reason,
        "single_launch": snapshot["counters"].get(
            "single_launch", 0
        ),
    }


def main() -> int:
    report: dict[str, Any] = {
        "schema": 1,
        "checks": {},
        "evidence": {},
        "errors": [],
    }
    original_environment = {
        name: os.environ.get(name)
        for name in (*ENVIRONMENT_KEYS, *FEATURE_ENVIRONMENT)
    }
    for name, value in FEATURE_ENVIRONMENT.items():
        os.environ[name] = value

    restorer = _Restorer()
    plugin_module: types.ModuleType | None = None
    attention_module: types.ModuleType | None = None
    builder_cls: type[Any] | None = None
    backend_cls: type[Any] | None = None
    before_methods: dict[str, dict[str, object]] | None = None
    source_path: str | None = None
    source_hash_before: str | None = None

    try:
        entry_points = _entry_points()
        entry_evidence = [
            {
                "name": item.name,
                "value": item.value,
                "distribution": _distribution_name(item),
            }
            for item in entry_points
        ]
        _record_check(
            report,
            "wheel_entry_point_unique",
            len(entry_points) == 1,
            entry_evidence,
        )
        if len(entry_points) != 1:
            raise RuntimeError(
                f"expected one {ENTRY_POINT_NAME!r} entry point, "
                f"found {len(entry_points)}"
            )
        entry_point = entry_points[0]
        _record_check(
            report,
            "wheel_entry_point_target",
            entry_point.value == EXPECTED_ENTRY_POINT,
            entry_point.value,
        )

        # vLLM-Ascend initializes ops before attention in normal engine
        # startup. Importing attention first creates an upstream circular
        # device_op/experts_selector dependency on this supported release.
        ascend_ops_module = importlib.import_module(ASCEND_OPS_MODULE)
        _record_check(
            report,
            "official_ascend_ops_bootstrap",
            ascend_ops_module.__name__ == ASCEND_OPS_MODULE,
            str(Path(ascend_ops_module.__file__).resolve()),
        )

        # Import the real installed attention module before loading
        # TriangleMix so the identities below are true pre-patch evidence.
        attention_module = importlib.import_module(ATTENTION_MODULE)
        builder_cls = attention_module.AscendAttentionMetadataBuilder
        backend_cls = attention_module.AscendAttentionBackendImpl
        source_path = str(Path(attention_module.__file__).resolve())
        source_hash_before = _sha256(source_path)
        before_methods = _method_state(builder_cls, backend_cls)
        marker = "_vllm_ascend_trianglemix_plugin_patched"
        _record_check(
            report,
            "classes_unpatched_before_registration",
            not bool(getattr(builder_cls, marker, False))
            and not bool(getattr(backend_cls, marker, False)),
            {
                "builder_marker": bool(
                    getattr(builder_cls, marker, False)
                ),
                "backend_marker": bool(
                    getattr(backend_cls, marker, False)
                ),
            },
        )

        restorer.save(builder_cls, "build")
        restorer.save(builder_cls, "build_for_graph_capture")
        restorer.save(builder_cls, marker)
        restorer.save(backend_cls, "__init__")
        restorer.save(backend_cls, "forward")
        restorer.save(
            backend_cls,
            "forward_fused_infer_attention",
        )
        restorer.save(backend_cls, "_can_use_triangle_mix")
        restorer.save(
            backend_cls,
            "_vllm_ascend_trianglemix_legacy_can_use",
        )
        restorer.save(
            attention_module,
            "load_triangle_mix_adapter_if_enabled",
        )
        restorer.save(
            attention_module,
            "_vllm_ascend_trianglemix_legacy_loader",
        )

        environment_before_register = _environment_snapshot()
        native_before_register = NATIVE_MODULE in sys.modules
        register = entry_point.load()
        plugin_module = importlib.import_module(register.__module__)
        restorer.save(plugin_module, "_REGISTERED")
        register()
        environment_after_register = _environment_snapshot()
        native_after_register = NATIVE_MODULE in sys.modules

        after_methods = _method_state(builder_cls, backend_cls)
        expected_patched = (
            "builder.build",
            "builder.build_for_graph_capture",
            "backend.__init__",
            "backend.forward",
            "backend.forward_fused_infer_attention",
        )
        identities_changed = {
            name: (
                before_methods[name]["identity"]
                != after_methods[name]["identity"]
            )
            for name in expected_patched
        }
        signatures_preserved = {
            name: (
                before_methods[name]["signature"]
                == after_methods[name]["signature"]
            )
            for name in expected_patched
        }
        _record_check(
            report,
            "actual_class_methods_patched",
            all(identities_changed.values()),
            identities_changed,
        )
        _record_check(
            report,
            "actual_method_signatures_preserved",
            all(signatures_preserved.values()),
            signatures_preserved,
        )
        _record_check(
            report,
            "feature_off_registration_has_no_native_or_env_side_effect",
            (
                native_before_register == native_after_register
                and not native_after_register
                and environment_before_register
                == environment_after_register
            ),
            {
                "native_imported_before": native_before_register,
                "native_imported_after": native_after_register,
                "environment_unchanged": (
                    environment_before_register
                    == environment_after_register
                ),
            },
        )

        feature_off_probe = _feature_off_init_probe(
            plugin_module,
            backend_cls.__init__,
        )
        _record_check(
            report,
            "feature_off_backend_init_probe",
            bool(feature_off_probe.pop("passed")),
            feature_off_probe,
        )

        decode_probe = _decode_fallback_probe(
            plugin_module,
            attention_module,
            backend_cls.forward_fused_infer_attention,
        )
        _record_check(
            report,
            "decode_only_delegates_to_original_fia",
            bool(decode_probe.pop("passed")),
            decode_probe,
        )

        legacy_before = before_methods[
            "backend._can_use_triangle_mix"
        ]["present"]
        legacy_after = getattr(
            backend_cls,
            "_can_use_triangle_mix",
            None,
        )
        if legacy_before:
            legacy_disabled = (
                callable(legacy_after)
                and legacy_after(types.SimpleNamespace()) is False
            )
            legacy_evidence: object = {
                "present_before": True,
                "disabled_result": (
                    legacy_after(types.SimpleNamespace())
                    if callable(legacy_after)
                    else None
                ),
            }
        else:
            legacy_disabled = not callable(legacy_after)
            legacy_evidence = {"present_before": False}
        _record_check(
            report,
            "legacy_predicate_disabled_or_absent",
            legacy_disabled,
            legacy_evidence,
        )

        legacy_loader_original = getattr(
            attention_module,
            "_vllm_ascend_trianglemix_legacy_loader",
            None,
        )
        loader_after = getattr(
            attention_module,
            "load_triangle_mix_adapter_if_enabled",
            None,
        )
        if callable(legacy_loader_original):
            loader_disabled = (
                callable(loader_after)
                and loader_after(True, "/must/not/load.so") is False
                and NATIVE_MODULE not in sys.modules
            )
        else:
            loader_disabled = True
        _record_check(
            report,
            "legacy_external_adapter_loader_disabled_or_absent",
            loader_disabled,
            {
                "legacy_loader_present": callable(
                    legacy_loader_original
                ),
                "native_imported": NATIVE_MODULE in sys.modules,
            },
        )

        kernel_module = importlib.import_module(
            "vllm_ascend_trianglemix.kernel"
        )
        _record_check(
            report,
            "no_triangle_adapter_loaded",
            getattr(
                kernel_module,
                "_LOADED_ADAPTER_PATH",
                None,
            )
            is None,
            {
                "adapter_path": getattr(
                    kernel_module,
                    "_LOADED_ADAPTER_PATH",
                    None,
                )
            },
        )
        _record_check(
            report,
            "attention_source_unchanged_while_patched",
            _sha256(source_path) == source_hash_before,
            {
                "path": source_path,
                "sha256_before": source_hash_before,
                "sha256_after": _sha256(source_path),
            },
        )
        report["evidence"]["plugin_module"] = str(
            Path(plugin_module.__file__).resolve()
        )
        report["evidence"]["attention_module"] = source_path
        report["evidence"]["methods_before"] = before_methods
        report["evidence"]["methods_after_patch"] = after_methods
    except Exception as exc:
        report["errors"].append(
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        try:
            restorer.restore()
            if (
                builder_cls is not None
                and backend_cls is not None
                and before_methods is not None
            ):
                restored_methods = _method_state(
                    builder_cls,
                    backend_cls,
                )
                restored = all(
                    restored_methods[name]["identity"]
                    == before_methods[name]["identity"]
                    and restored_methods[name]["signature"]
                    == before_methods[name]["signature"]
                    for name in before_methods
                )
                _record_check(
                    report,
                    "actual_class_methods_restored",
                    restored,
                    restored_methods,
                )
                report["evidence"][
                    "methods_after_restore"
                ] = restored_methods
            if source_path is not None and source_hash_before is not None:
                _record_check(
                    report,
                    "attention_source_unchanged_after_restore",
                    _sha256(source_path) == source_hash_before,
                )
        except Exception as exc:
            report["errors"].append(
                {
                    "type": type(exc).__name__,
                    "message": f"restore failed: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            _restore_environment(original_environment)

    report["ok"] = (
        not report["errors"]
        and bool(report["checks"])
        and all(
            item["passed"]
            for item in report["checks"].values()
        )
    )
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
