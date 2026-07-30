"""Typed TriangleMix configuration with vLLM additional-config support."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from .kernel import TriangleMixConfig, parse_layer_indices


_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def _coerce_bool(value: object, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in _TRUE_VALUES:
            return True
        if normalised in _FALSE_VALUES:
            return False
    raise ValueError(
        f"TriangleMix {name} must be a boolean, got {value!r}"
    )


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    return default if value is None else _coerce_bool(value, name=name)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _value(
    section: Mapping[str, Any],
    name: str,
    env_name: str,
    default: Any,
) -> Any:
    if name in section:
        return section[name]
    env_value = os.getenv(env_name)
    if env_value is None:
        return default
    if isinstance(default, bool):
        return _env_bool(env_name, default)
    if isinstance(default, int):
        return int(env_value)
    return env_value


@dataclass(frozen=True)
class PluginConfig:
    """Process configuration shared by the planner and patched backend."""

    kernel: TriangleMixConfig
    strict: bool = False
    stats_recent_capacity: int = 256
    stats_log_interval: int = 0

    @property
    def enabled(self) -> bool:
        return self.kernel.enabled and bool(self.kernel.layer_indices)


def resolve_plugin_config(vllm_config: object | None) -> PluginConfig:
    """Resolve typed config, preferring ``additional_config.trianglemix``."""
    additional = _mapping(
        getattr(vllm_config, "additional_config", None)
    )
    section = _mapping(additional.get("trianglemix", additional))
    layers_value = _value(
        section,
        "layers",
        "VLLM_ASCEND_TRIANGLE_MIX_LAYERS",
        "",
    )
    if isinstance(layers_value, (list, tuple, set)):
        layers_text = ",".join(str(item) for item in layers_value)
    else:
        layers_text = str(layers_value)

    kernel = TriangleMixConfig(
        enabled=_coerce_bool(
            _value(
                section,
                "enabled",
                "VLLM_ASCEND_ENABLE_TRIANGLE_MIX",
                False,
            ),
            name="enabled",
        ),
        layer_indices=parse_layer_indices(layers_text),
        sink_tokens=int(
            _value(
                section,
                "sink_tokens",
                "VLLM_ASCEND_TRIANGLE_MIX_SINK_TOKENS",
                8,
            )
        ),
        local_window=int(
            _value(
                section,
                "local_window",
                "VLLM_ASCEND_TRIANGLE_MIX_LOCAL_WINDOW",
                512,
            )
        ),
        last_rows=int(
            _value(
                section,
                "last_rows",
                "VLLM_ASCEND_TRIANGLE_MIX_LAST_ROWS",
                128,
            )
        ),
        direct_min_seq_len=int(
            _value(
                section,
                "min_seq_len",
                "VLLM_ASCEND_TRIANGLE_MIX_DIRECT_MIN_SEQ_LEN",
                0,
            )
        ),
        direct_min_sparse_rows=int(
            _value(
                section,
                "min_sparse_rows",
                "VLLM_ASCEND_TRIANGLE_MIX_DIRECT_MIN_SPARSE_ROWS",
                128,
            )
        ),
        direct_min_saved_qk=int(
            _value(
                section,
                "min_saved_qk",
                "VLLM_ASCEND_TRIANGLE_MIX_DIRECT_MIN_SAVED_QK",
                913_152,
            )
        ),
        direct_split_min_sparse_rows=int(
            _value(
                section,
                "split_min_sparse_rows",
                "VLLM_ASCEND_TRIANGLE_MIX_DIRECT_SPLIT_MIN_SPARSE_ROWS",
                192,
            )
        ),
        direct_split_min_saved_qk=int(
            _value(
                section,
                "split_min_saved_qk",
                "VLLM_ASCEND_TRIANGLE_MIX_DIRECT_SPLIT_MIN_SAVED_QK",
                1_299_264,
            )
        ),
    )
    recent_capacity = int(
        _value(
            section,
            "stats_recent_capacity",
            "VLLM_ASCEND_TRIANGLE_MIX_STATS_RECENT_CAPACITY",
            256,
        )
    )
    log_interval = int(
        _value(
            section,
            "stats_log_interval",
            "VLLM_ASCEND_TRIANGLE_MIX_STATS_LOG_INTERVAL",
            0,
        )
    )
    if recent_capacity < 0 or log_interval < 0:
        raise ValueError("TriangleMix stats capacities must be non-negative")
    return PluginConfig(
        kernel=kernel,
        strict=_coerce_bool(
            _value(
                section,
                "strict",
                "VLLM_ASCEND_TRIANGLE_MIX_STRICT",
                False,
            ),
            name="strict",
        ),
        stats_recent_capacity=recent_capacity,
        stats_log_interval=log_interval,
    )
