#!/usr/bin/env python3
"""Independent TrianglePagedSparseAttention v2 CPU reference.

This module deliberately has no dependency on the custom operator.  The
pure-Python path keeps the scheduling and online-softmax contract testable on
machines without PyTorch.  ``torch_reference_attention`` is the higher
throughput oracle used by the NPU validation script on the Ascend server.

The fixed production semantics are:

* Q is ``[Tq, 32, 128]`` and paged K/V are
  ``[num_pages, 128, 8, 128]``;
* query head ``h`` uses KV head ``h // 4``;
* sparse rows attend to ``[0, 8) U [q - 512, q + 1)``;
* overlapping or adjacent sink/local intervals are merged;
* rows before ``sparse_begin = 521`` and the final 128 prompt rows are dense;
* sink and local tokens participate in one joint softmax.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


Vector = Sequence[float]
Intervals = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class TriangleGeometry:
    """Shape and schedule constants.

    Tests may use smaller tensor dimensions while retaining the same
    scheduling rules.  ``PRODUCTION_GEOMETRY`` is the only geometry intended
    for the NPU operator.
    """

    query_heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    page_size: int = 128
    q_tile: int = 32
    sink_tokens: int = 8
    local_window: int = 512
    dense_tail: int = 128

    def __post_init__(self) -> None:
        positive = (
            self.query_heads,
            self.kv_heads,
            self.head_dim,
            self.page_size,
            self.q_tile,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("head counts, dimensions, page size, and tile must be positive")
        if self.query_heads % self.kv_heads:
            raise ValueError("query_heads must be divisible by kv_heads")
        if min(self.sink_tokens, self.local_window, self.dense_tail) < 0:
            raise ValueError("schedule sizes must be non-negative")

    @property
    def group_size(self) -> int:
        return self.query_heads // self.kv_heads

    @property
    def sparse_begin(self) -> int:
        # At q == sink + window, local begins exactly at sink_end and the
        # union remains dense.  The first row with a real gap is one later.
        return self.sink_tokens + self.local_window + 1


PRODUCTION_GEOMETRY = TriangleGeometry()


@dataclass(frozen=True)
class QuerySpan:
    """One C++-compatible absolute dense or sparse query span."""

    begin: int
    end: int
    sparse: bool

    def __post_init__(self) -> None:
        if self.begin < 0 or self.end <= self.begin:
            raise ValueError("query span must be a non-empty half-open range")

    @property
    def row_count(self) -> int:
        return self.end - self.begin


def split_query_spans(
    query_start: int,
    query_tokens: int,
    sparse_begin: int,
    sparse_end_position: int,
    geometry: TriangleGeometry = PRODUCTION_GEOMETRY,
) -> tuple[QuerySpan, ...]:
    """Split every Q tile at exact dense/sparse row boundaries.

    Row offsets are relative to the possibly unaligned ``query_start``.  A
    span never crosses a q-tile boundary, sparseBegin, or sparseEnd.
    """

    if query_start < 0:
        raise ValueError("query_start must be non-negative")
    if query_tokens <= 0:
        raise ValueError("query_tokens must be positive")
    if sparse_begin < 0 or sparse_end_position < 0:
        raise ValueError("sparse boundaries must be non-negative")

    spans: list[QuerySpan] = []
    for tile_row_begin in range(0, query_tokens, geometry.q_tile):
        tile_row_end = min(query_tokens, tile_row_begin + geometry.q_tile)
        tile_begin = query_start + tile_row_begin
        tile_end = query_start + tile_row_end
        boundaries = {tile_begin, tile_end}
        for absolute_boundary in (sparse_begin, sparse_end_position):
            if tile_begin < absolute_boundary < tile_end:
                boundaries.add(absolute_boundary)
        ordered = sorted(boundaries)
        tile_spans: list[QuerySpan] = []
        for begin, end in zip(ordered, ordered[1:]):
            is_sparse = sparse_begin <= begin < sparse_end_position
            candidate = QuerySpan(
                begin=begin,
                end=end,
                sparse=is_sparse,
            )
            # A degenerate boundary ordering (for example sparseEnd before
            # sparseBegin in a short prompt) can create adjacent dense pieces.
            # Merge those pieces, but never merge across a q-tile boundary.
            if (
                tile_spans
                and tile_spans[-1].sparse == candidate.sparse
                and tile_spans[-1].end == candidate.begin
            ):
                previous = tile_spans.pop()
                candidate = QuerySpan(
                    begin=previous.begin,
                    end=candidate.end,
                    sparse=candidate.sparse,
                )
            tile_spans.append(candidate)
        spans.extend(tile_spans)
    return tuple(spans)


def submitted_span_intervals(
    span: QuerySpan,
    seq_len: int,
    geometry: TriangleGeometry = PRODUCTION_GEOMETRY,
) -> Intervals:
    """Return the KV intervals a fast-path query span may submit.

    A sparse span submits only sink and the union local band. Per-row causal
    and window boundary masks further narrow the local interval inside the
    kernel; the skipped middle is never submitted to QK/PV.
    """

    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    q_begin = span.begin
    q_end = min(span.end, seq_len)
    if q_end <= 0:
        return ()
    if not span.sparse:
        return ((0, q_end),)

    sink_end = min(geometry.sink_tokens, seq_len)
    local_begin = min(max(0, q_begin - geometry.local_window), q_end)
    if local_begin <= sink_end:
        return ((0, q_end),)
    intervals: list[tuple[int, int]] = []
    if sink_end:
        intervals.append((0, sink_end))
    if local_begin < q_end:
        intervals.append((local_begin, q_end))
    return tuple(intervals)


def sparse_end(prompt_len: int, geometry: TriangleGeometry = PRODUCTION_GEOMETRY) -> int:
    """Return the first query position in the final dense tail."""

    if prompt_len < 0:
        raise ValueError("prompt_len must be non-negative")
    return max(0, prompt_len - geometry.dense_tail)


def selected_intervals(
    query_position: int,
    seq_len: int,
    prompt_len: int,
    geometry: TriangleGeometry = PRODUCTION_GEOMETRY,
) -> Intervals:
    """Return exact half-open logical-KV intervals for one query row."""

    if query_position < 0:
        raise ValueError("query_position must be non-negative")
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    if prompt_len < 0:
        raise ValueError("prompt_len must be non-negative")

    causal_end = min(query_position + 1, seq_len)
    if causal_end <= 0:
        return ()

    row_is_sparse = (
        geometry.sparse_begin <= query_position
        and query_position < sparse_end(prompt_len, geometry)
    )
    if not row_is_sparse:
        return ((0, causal_end),)

    sink_end = min(geometry.sink_tokens, causal_end)
    local_begin = min(max(0, query_position - geometry.local_window), causal_end)

    # Adjacent intervals are one continuous dense prefix.  This is important
    # at q == sparse_begin - 1.
    if local_begin <= sink_end:
        return ((0, causal_end),)

    intervals: list[tuple[int, int]] = []
    if sink_end:
        intervals.append((0, sink_end))
    if local_begin < causal_end:
        intervals.append((local_begin, causal_end))
    return tuple(intervals)


def selected_token_indices(
    query_position: int,
    seq_len: int,
    prompt_len: int,
    geometry: TriangleGeometry = PRODUCTION_GEOMETRY,
) -> tuple[int, ...]:
    """Expand ``selected_intervals`` into logical token indices."""

    return tuple(
        token
        for begin, end in selected_intervals(
            query_position,
            seq_len,
            prompt_len,
            geometry,
        )
        for token in range(begin, end)
    )


def physical_location(
    logical_token: int,
    block_table: Sequence[int],
    geometry: TriangleGeometry = PRODUCTION_GEOMETRY,
) -> tuple[int, int]:
    """Map one logical token to ``(physical_page, offset_in_page)``."""

    if logical_token < 0:
        raise ValueError("logical_token must be non-negative")
    logical_page, page_offset = divmod(logical_token, geometry.page_size)
    if logical_page >= len(block_table):
        raise ValueError("block_table does not cover logical_token")
    physical_page = int(block_table[logical_page])
    if physical_page < 0:
        raise ValueError("block_table contains a negative physical page")
    return physical_page, page_offset


def paged_vector(
    cache: Sequence[Sequence[Sequence[Vector]]],
    block_table: Sequence[int],
    logical_token: int,
    kv_head: int,
    geometry: TriangleGeometry = PRODUCTION_GEOMETRY,
) -> Vector:
    """Read one logical ``[D]`` K/V vector from a BSND paged cache."""

    if not 0 <= kv_head < geometry.kv_heads:
        raise ValueError("kv_head is out of range")
    physical_page, page_offset = physical_location(
        logical_token,
        block_table,
        geometry,
    )
    try:
        vector = cache[physical_page][page_offset][kv_head]
    except IndexError as error:
        raise ValueError("cache shape is inconsistent with geometry/table") from error
    if len(vector) != geometry.head_dim:
        raise ValueError("cache vector has the wrong head dimension")
    return vector


def _dot(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise ValueError("dot-product vectors have different dimensions")
    return math.fsum(float(x) * float(y) for x, y in zip(left, right))


def joint_softmax_weighted_sum(
    scores: Sequence[float],
    values: Sequence[Vector],
) -> list[float]:
    """Stable softmax over all scores followed by one weighted sum."""

    if not scores or len(scores) != len(values):
        raise ValueError("scores and values must have the same non-zero length")
    dimension = len(values[0])
    if dimension == 0 or any(len(vector) != dimension for vector in values):
        raise ValueError("values must have one consistent non-zero dimension")
    maximum = max(float(score) for score in scores)
    weights = [math.exp(float(score) - maximum) for score in scores]
    denominator = math.fsum(weights)
    return [
        math.fsum(weight * float(vector[index]) for weight, vector in zip(weights, values))
        / denominator
        for index in range(dimension)
    ]


def online_softmax_weighted_sum(
    blocks: Iterable[tuple[Sequence[float], Sequence[Vector]]],
) -> list[float]:
    """Online softmax carrying one state across every supplied KV block.

    Supplying sink and local as two blocks must produce the same output as
    concatenating them before a single softmax.  Resetting ``row_max``,
    ``row_sum``, or the accumulator between blocks is therefore incorrect.
    """

    row_max = -math.inf
    row_sum = 0.0
    accumulator: list[float] | None = None
    observed = 0

    for scores, values in blocks:
        if len(scores) != len(values):
            raise ValueError("each score block must match its value block")
        if not scores:
            continue
        dimension = len(values[0])
        if dimension == 0 or any(len(vector) != dimension for vector in values):
            raise ValueError("value blocks must have a consistent dimension")
        if accumulator is None:
            accumulator = [0.0] * dimension
        elif len(accumulator) != dimension:
            raise ValueError("all value blocks must have the same dimension")

        block_max = max(float(score) for score in scores)
        new_max = max(row_max, block_max)
        old_correction = 0.0 if row_max == -math.inf else math.exp(row_max - new_max)
        weights = [math.exp(float(score) - new_max) for score in scores]
        accumulator = [
            accumulator[index] * old_correction
            + math.fsum(
                weight * float(vector[index])
                for weight, vector in zip(weights, values)
            )
            for index in range(dimension)
        ]
        row_sum = row_sum * old_correction + math.fsum(weights)
        row_max = new_max
        observed += len(scores)

    if accumulator is None or observed == 0 or row_sum == 0.0:
        raise ValueError("online softmax requires at least one finite score")
    return [value / row_sum for value in accumulator]


def reference_attention_python(
    query: Sequence[Sequence[Vector]],
    key_cache: Sequence[Sequence[Sequence[Vector]]],
    value_cache: Sequence[Sequence[Sequence[Vector]]],
    block_table: Sequence[int],
    query_start: int,
    seq_len: int,
    prompt_len: int,
    scale: float,
    geometry: TriangleGeometry = PRODUCTION_GEOMETRY,
) -> list[list[list[float]]]:
    """Pure-Python paged GQA reference with a joint sink/local softmax."""

    if query_start < 0:
        raise ValueError("query_start must be non-negative")
    if query_start + len(query) > seq_len:
        raise ValueError("query rows extend beyond seq_len")
    required_pages = (seq_len + geometry.page_size - 1) // geometry.page_size
    if len(block_table) < required_pages:
        raise ValueError("block_table does not cover seq_len")

    result: list[list[list[float]]] = []
    for row_index, query_row in enumerate(query):
        if len(query_row) != geometry.query_heads:
            raise ValueError("query row has the wrong number of heads")
        query_position = query_start + row_index
        tokens = selected_token_indices(
            query_position,
            seq_len,
            prompt_len,
            geometry,
        )
        output_row: list[list[float]] = []
        for query_head, query_vector in enumerate(query_row):
            if len(query_vector) != geometry.head_dim:
                raise ValueError("query vector has the wrong head dimension")
            kv_head = query_head // geometry.group_size
            keys = [
                paged_vector(
                    key_cache,
                    block_table,
                    token,
                    kv_head,
                    geometry,
                )
                for token in tokens
            ]
            values = [
                paged_vector(
                    value_cache,
                    block_table,
                    token,
                    kv_head,
                    geometry,
                )
                for token in tokens
            ]
            scores = [_dot(query_vector, key) * float(scale) for key in keys]
            output_row.append(joint_softmax_weighted_sum(scores, values))
        result.append(output_row)
    return result


def reference_attention_ragged_python(
    packed_query: Sequence[Sequence[Vector]],
    key_cache: Sequence[Sequence[Sequence[Vector]]],
    value_cache: Sequence[Sequence[Sequence[Vector]]],
    block_tables: Sequence[Sequence[int]],
    query_starts: Sequence[int],
    seq_lens: Sequence[int],
    prompt_lens: Sequence[int],
    query_lengths: Sequence[int],
    scale: float,
    geometry: TriangleGeometry = PRODUCTION_GEOMETRY,
) -> list[list[list[float]]]:
    """Reference a packed ragged batch using the frozen batch-one contract.

    The production op is invoked once per sequence.  This helper validates
    the vLLM-style packed slicing and ragged final query tiles independently.
    """

    batch_size = len(query_lengths)
    metadata = (
        block_tables,
        query_starts,
        seq_lens,
        prompt_lens,
    )
    if any(len(items) != batch_size for items in metadata):
        raise ValueError("ragged metadata lengths disagree")
    if sum(query_lengths) != len(packed_query):
        raise ValueError("query_lengths do not cover packed_query")

    output: list[list[list[float]]] = []
    offset = 0
    for index, query_length in enumerate(query_lengths):
        if query_length <= 0:
            raise ValueError("each ragged query length must be positive")
        sequence_query = packed_query[offset : offset + query_length]
        output.extend(
            reference_attention_python(
                sequence_query,
                key_cache,
                value_cache,
                block_tables[index],
                query_starts[index],
                seq_lens[index],
                prompt_lens[index],
                scale,
                geometry,
            )
        )
        offset += query_length
    return output


def torch_reference_attention(
    query_bf16: Any,
    key_cache_bf16: Any,
    value_cache_bf16: Any,
    block_table: Any,
    query_start: int,
    seq_len: int,
    prompt_len: int,
    scale: float,
    geometry: TriangleGeometry = PRODUCTION_GEOMETRY,
) -> Any:
    """Vectorized PyTorch CPU oracle for full Qwen3 dimensions.

    PyTorch is imported lazily so the schedule and pure-Python reference stay
    runnable in a lightweight local environment.
    """

    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("torch_reference_attention requires PyTorch") from error

    expected_query_shape = (
        query_bf16.shape[0],
        geometry.query_heads,
        geometry.head_dim,
    )
    if tuple(query_bf16.shape) != expected_query_shape:
        raise ValueError("query shape does not match geometry")
    expected_cache_tail = (
        geometry.page_size,
        geometry.kv_heads,
        geometry.head_dim,
    )
    if tuple(key_cache_bf16.shape[1:]) != expected_cache_tail:
        raise ValueError("key cache shape does not match geometry")
    if tuple(value_cache_bf16.shape) != tuple(key_cache_bf16.shape):
        raise ValueError("value cache shape must equal key cache shape")
    if query_start + query_bf16.shape[0] > seq_len:
        raise ValueError("query rows extend beyond seq_len")

    table = block_table.reshape(-1).to(device="cpu", dtype=torch.int64)
    required_pages = (seq_len + geometry.page_size - 1) // geometry.page_size
    if table.numel() < required_pages:
        raise ValueError("block_table does not cover seq_len")
    physical_pages = table[:required_pages]
    if bool((physical_pages < 0).any()):
        raise ValueError("block_table contains a negative physical page")

    logical_key = (
        key_cache_bf16
        .index_select(0, physical_pages)
        .float()
        .reshape(-1, geometry.kv_heads, geometry.head_dim)[:seq_len]
    )
    logical_value = (
        value_cache_bf16
        .index_select(0, physical_pages)
        .float()
        .reshape(-1, geometry.kv_heads, geometry.head_dim)[:seq_len]
    )
    query = query_bf16.float()
    result = torch.empty_like(query)

    for row in range(query.shape[0]):
        query_position = query_start + row
        intervals = selected_intervals(
            query_position,
            seq_len,
            prompt_len,
            geometry,
        )
        token_parts = [
            torch.arange(begin, end, dtype=torch.int64)
            for begin, end in intervals
        ]
        token_indices = (
            token_parts[0]
            if len(token_parts) == 1
            else torch.cat(token_parts)
        )
        keys = logical_key.index_select(0, token_indices)
        values = logical_value.index_select(0, token_indices)
        grouped_query = query[row].reshape(
            geometry.kv_heads,
            geometry.group_size,
            geometry.head_dim,
        )
        # [Hkv, G, selected_tokens] followed by one softmax over the full
        # concatenated sink+local token dimension.
        scores = torch.einsum("hgd,thd->hgt", grouped_query, keys)
        probabilities = torch.softmax(scores * float(scale), dim=-1)
        grouped_output = torch.einsum("hgt,thd->hgd", probabilities, values)
        result[row] = grouped_output.reshape(
            geometry.query_heads,
            geometry.head_dim,
        )
    return result
