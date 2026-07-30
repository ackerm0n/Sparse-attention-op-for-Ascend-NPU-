#
# Copyright 2026 TriangleMix contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Single-launch paged-KV TriangleMix prefill attention for Ascend.

The custom MIX kernel consumes vLLM's paged KV cache and the complete query
chunk directly.  Inside that one launch it splits query tiles by absolute
position, runs dense causal attention for prefix/tail rows, runs sink plus
local-window attention for sparse rows, and merges every KV interval with one
online-softmax state.

The caller-owned output tensor is written in place.  There is no KV gather,
token pruning, query padding, intermediate attention output, or auxiliary FIA
launch around the sparse region.
"""

from __future__ import annotations

import math
import os
import re
import threading
from dataclasses import dataclass

import torch

SUPPORTED_HEAD_SIZE = 128
SUPPORTED_CACHE_BLOCK_SIZE = 128
SUPPORTED_QUERY_HEADS = 32
SUPPORTED_KV_HEADS = 8
SUPPORTED_SINK_TOKENS = 8
SUPPORTED_LOCAL_WINDOW = 512
SUPPORTED_LAST_ROWS = 128
DEFAULT_BLOCK_M = 32
DEFAULT_BLOCK_N = 512

_ADAPTER_LOAD_LOCK = threading.Lock()
_LOADED_ADAPTER_PATH: str | None = None


@dataclass(frozen=True)
class TriangleMixConfig:
    """Runtime configuration for direct paged TriangleMix attention."""

    enabled: bool
    layer_indices: frozenset[int]
    sink_tokens: int = SUPPORTED_SINK_TOKENS
    local_window: int = SUPPORTED_LOCAL_WINDOW
    last_rows: int = SUPPORTED_LAST_ROWS
    direct_min_seq_len: int = 0
    direct_min_sparse_rows: int = 128
    direct_min_saved_qk: int = 913_152
    direct_split_min_sparse_rows: int = 192
    direct_split_min_saved_qk: int = 1_299_264

    def __post_init__(self) -> None:
        if self.sink_tokens <= 0:
            raise ValueError("TriangleMix sink_tokens must be positive")
        if self.local_window < 0:
            raise ValueError(
                "TriangleMix local_window must be non-negative"
            )
        if self.last_rows <= 0:
            raise ValueError("TriangleMix last_rows must be positive")
        for field_name in (
            "direct_min_seq_len",
            "direct_min_sparse_rows",
            "direct_min_saved_qk",
            "direct_split_min_sparse_rows",
            "direct_split_min_saved_qk",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(
                    f"TriangleMix {field_name} must be non-negative"
                )

    @property
    def has_supported_geometry(self) -> bool:
        """Whether the fixed production operator ABI matches this config."""
        return (
            self.sink_tokens == SUPPORTED_SINK_TOKENS
            and self.local_window == SUPPORTED_LOCAL_WINDOW
            and self.last_rows == SUPPORTED_LAST_ROWS
        )

    def uses_layer(self, layer_name: str) -> bool:
        """Return whether ``layer_name`` is configured as a Triangle layer."""
        layer_index = extract_layer_index(layer_name)
        return (
            layer_index is not None
            and layer_index in self.layer_indices
        )


@dataclass(frozen=True)
class TriangleSparseSpan:
    """Absolute-coordinate plan for one scheduler prefill call."""

    q0: int
    q1: int
    s0: int
    s1: int

    @property
    def query_rows(self) -> int:
        return self.q1 - self.q0

    @property
    def sparse_rows(self) -> int:
        return max(0, self.s1 - self.s0)

    @property
    def has_sparse_middle(self) -> bool:
        return self.s0 < self.s1

    @property
    def has_dense_prefix(self) -> bool:
        return self.has_sparse_middle and self.q0 < self.s0

    @property
    def has_dense_tail(self) -> bool:
        return self.has_sparse_middle and self.s1 < self.q1

    @property
    def is_split(self) -> bool:
        return self.has_dense_prefix or self.has_dense_tail

    @property
    def saved_qk(self) -> int:
        return triangle_saved_qk(self.s0, self.s1)


def extract_layer_index(layer_name: str) -> int | None:
    """Extract a transformer layer index from a vLLM layer name."""
    match = re.search(
        r"(?:^|\.)layers\.(\d+)(?:\.|$)",
        layer_name,
    )
    return int(match.group(1)) if match is not None else None


def parse_layer_indices(value: str) -> frozenset[int]:
    """Parse comma-separated indices and inclusive ranges."""
    indices: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start < 0 or end < start:
                raise ValueError(
                    f"Invalid TriangleMix layer range: {item!r}"
                )
            indices.update(range(start, end + 1))
        else:
            index = int(item)
            if index < 0:
                raise ValueError(
                    f"Invalid TriangleMix layer index: {item!r}"
                )
            indices.add(index)
    return frozenset(indices)


def triangle_streaming_start(config: TriangleMixConfig) -> int:
    """First absolute query row with any QK work that can be skipped."""
    return config.sink_tokens + config.local_window + 1


def triangle_sparse_span(
    *,
    query_len: int,
    seq_len: int,
    prompt_len: int,
    config: TriangleMixConfig,
) -> TriangleSparseSpan:
    """Return the exact absolute-coordinate sparse interval ``[s0, s1)``."""
    if query_len < 0:
        raise ValueError("TriangleMix query_len must be non-negative")
    if seq_len < query_len:
        raise ValueError(
            "TriangleMix seq_len must be at least query_len"
        )
    if prompt_len < seq_len:
        raise ValueError(
            "TriangleMix prompt_len must be at least seq_len"
        )

    q0 = seq_len - query_len
    q1 = seq_len
    s0 = max(q0, triangle_streaming_start(config))
    s1 = min(q1, max(0, prompt_len - config.last_rows))
    return TriangleSparseSpan(q0=q0, q1=q1, s0=s0, s1=s1)


def triangle_sparse_query_rows(
    *,
    query_len: int,
    seq_len: int,
    prompt_len: int,
    config: TriangleMixConfig,
) -> int:
    """Return the number of rows dispatched to the direct sparse operator."""
    return triangle_sparse_span(
        query_len=query_len,
        seq_len=seq_len,
        prompt_len=prompt_len,
        config=config,
    ).sparse_rows


def triangle_saved_qk(sparse_start: int, sparse_end: int) -> int:
    """Return the exact number of causal QK positions skipped by the ABI.

    For the fixed sink-8/local-512 geometry, sparse row ``q`` skips
    ``q - 520`` positions.  Summing that arithmetic sequence over
    ``[sparse_start, sparse_end)`` gives the closed form below.
    """
    if sparse_end <= sparse_start:
        return 0
    rows = sparse_end - sparse_start
    return rows * (sparse_start + sparse_end - 1041) // 2


def triangle_direct_eligible(
    *,
    query_len: int,
    seq_len: int,
    prompt_len: int,
    config: TriangleMixConfig,
) -> bool:
    """Apply only performance/geometry gates for a structurally valid call."""
    if not config.has_supported_geometry:
        return False
    span = triangle_sparse_span(
        query_len=query_len,
        seq_len=seq_len,
        prompt_len=prompt_len,
        config=config,
    )
    if not span.has_sparse_middle:
        return False
    if seq_len < config.direct_min_seq_len:
        return False
    if span.sparse_rows < config.direct_min_sparse_rows:
        return False
    if span.saved_qk < config.direct_min_saved_qk:
        return False
    if span.is_split and (
        span.sparse_rows < config.direct_split_min_sparse_rows
        or span.saved_qk < config.direct_split_min_saved_qk
    ):
        return False
    return True


def _normalise_adapter_path(path: str | os.PathLike[str] | None) -> str:
    if path is None or not os.fspath(path).strip():
        raise ValueError(
            "VLLM_ASCEND_TRIANGLE_MIX_ADAPTER_PATH is empty"
        )
    resolved = os.path.realpath(
        os.path.abspath(os.path.expanduser(os.fspath(path)))
    )
    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            f"TriangleMix Torch adapter does not exist: {resolved}"
        )
    return resolved


def _triangle_sparse_out_is_registered() -> bool:
    try:
        packet = torch.ops.trianglemix.triangle_paged_sparse_attention
        packet.out
    except AttributeError:
        return False
    return True


def load_triangle_mix_adapter(
    path: str | os.PathLike[str] | None,
) -> str:
    """Load the private Torch adapter once per process.

    Reusing the same absolute path is idempotent.  Loading a second adapter
    from a different path is rejected because Torch operator registrations
    cannot be safely replaced in a live worker process.
    """
    resolved = _normalise_adapter_path(path)
    global _LOADED_ADAPTER_PATH
    with _ADAPTER_LOAD_LOCK:
        if _LOADED_ADAPTER_PATH is not None:
            if _LOADED_ADAPTER_PATH != resolved:
                raise RuntimeError(
                    "TriangleMix adapter is already loaded from "
                    f"{_LOADED_ADAPTER_PATH}; refusing {resolved}"
                )
            return resolved
        torch.ops.load_library(resolved)
        if not _triangle_sparse_out_is_registered():
            raise RuntimeError(
                "TriangleMix adapter did not register "
                "trianglemix::triangle_paged_sparse_attention.out"
            )
        _LOADED_ADAPTER_PATH = resolved
        return resolved


def load_triangle_mix_adapter_if_enabled(
    enabled: bool,
    path: str | os.PathLike[str] | None,
) -> bool:
    """Load the adapter only when the feature is explicitly enabled."""
    if not enabled:
        return False
    load_triangle_mix_adapter(path)
    return True


def triangle_mix_adapter_loaded() -> bool:
    """Return whether this process has successfully loaded the adapter."""
    with _ADAPTER_LOAD_LOCK:
        return _LOADED_ADAPTER_PATH is not None


def _validate_direct_tensors(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    output: torch.Tensor,
    seq_len: int,
) -> None:
    if (
        query.ndim != 3
        or tuple(query.shape[1:])
        != (SUPPORTED_QUERY_HEADS, SUPPORTED_HEAD_SIZE)
    ):
        raise ValueError(
            "TriangleMix query must be [Tq, 32, 128]"
        )
    if query.dtype != torch.bfloat16:
        raise ValueError("TriangleMix query must be BF16")
    if output.shape != query.shape or output.dtype != query.dtype:
        raise ValueError(
            "TriangleMix output must match query shape and dtype"
        )
    expected_cache_tail = (
        SUPPORTED_CACHE_BLOCK_SIZE,
        SUPPORTED_KV_HEADS,
        SUPPORTED_HEAD_SIZE,
    )
    if (
        key_cache.ndim != 4
        or tuple(key_cache.shape[1:]) != expected_cache_tail
        or value_cache.shape != key_cache.shape
    ):
        raise ValueError(
            "TriangleMix KV caches must be [pages, 128, 8, 128]"
        )
    if (
        key_cache.dtype != query.dtype
        or value_cache.dtype != query.dtype
    ):
        raise ValueError("TriangleMix query and KV cache dtypes must match")
    if (
        block_table.ndim != 2
        or block_table.shape[0] != 1
        or block_table.dtype != torch.int32
    ):
        raise ValueError(
            "TriangleMix block table must be INT32 [1, max_pages]"
        )
    if any(
        not tensor.is_contiguous()
        for tensor in (
            query,
            key_cache,
            value_cache,
            block_table,
            output,
        )
    ):
        raise ValueError("TriangleMix tensors must be contiguous")
    if any(
        tensor.device != query.device
        for tensor in (
            key_cache,
            value_cache,
            block_table,
            output,
        )
    ):
        raise ValueError("TriangleMix tensors must share one device")
    required_pages = math.ceil(seq_len / SUPPORTED_CACHE_BLOCK_SIZE)
    if block_table.shape[1] < required_pages:
        raise ValueError(
            "TriangleMix block table is too short for seq_len"
        )


def _run_triangle_single_launch_out(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start: int,
    seq_len: int,
    prompt_len: int,
    softmax_scale: float,
    output: torch.Tensor,
) -> None:
    """Run the complete query chunk through one direct-paged MIX kernel."""
    torch.ops.trianglemix.triangle_paged_sparse_attention.out(
        query,
        key_cache,
        value_cache,
        block_table,
        query_start,
        seq_len,
        prompt_len,
        softmax_scale,
        out=output,
    )


def triangle_direct_paged_attention(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    prompt_len: int,
    output: torch.Tensor,
    softmax_scale: float,
    config: TriangleMixConfig,
) -> torch.Tensor:
    """Execute dense prefix, sparse middle, and dense tail in one launch."""
    span = triangle_sparse_span(
        query_len=query.shape[0],
        seq_len=seq_len,
        prompt_len=prompt_len,
        config=config,
    )
    if not config.has_supported_geometry:
        raise ValueError("TriangleMix operator geometry does not match config")
    if not span.has_sparse_middle:
        raise ValueError(
            "TriangleMix direct path requires a non-empty sparse middle"
        )
    _validate_direct_tensors(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        output=output,
        seq_len=seq_len,
    )
    _run_triangle_single_launch_out(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        query_start=span.q0,
        seq_len=span.q1,
        prompt_len=prompt_len,
        softmax_scale=softmax_scale,
        output=output,
    )
    return output


def estimate_tile_counts(
    seq_len: int,
    config: TriangleMixConfig,
    block_m: int = DEFAULT_BLOCK_M,
    block_n: int = DEFAULT_BLOCK_N,
) -> tuple[int, int]:
    """Return theoretical ``(triangle, dense)`` full-prompt QK tile counts."""
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    if block_m <= 0 or block_n <= 0:
        raise ValueError("block sizes must be positive")

    triangle_tiles = 0
    dense_tiles = 0
    for query_start in range(0, seq_len, block_m):
        query_end = min(seq_len, query_start + block_m)
        dense_tiles += math.ceil(query_end / block_n)
        if query_end <= triangle_streaming_start(config):
            triangle_tiles += math.ceil(query_end / block_n)
            continue
        sink_tiles = math.ceil(config.sink_tokens / block_n)
        local_begin = max(
            config.sink_tokens,
            query_start - config.local_window,
        )
        local_tiles = (
            math.ceil(query_end / block_n)
            - local_begin // block_n
        )
        triangle_tiles += sink_tiles + max(0, local_tiles)
    return triangle_tiles, dense_tiles
