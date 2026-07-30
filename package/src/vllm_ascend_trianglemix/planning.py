"""Per-scheduler-step TriangleMix routing plans."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .kernel import TriangleMixConfig, TriangleSparseSpan, triangle_sparse_span


class FallbackReason(str, Enum):
    NONE = "none"
    STATE_UNSUPPORTED = "state_unsupported"
    MIXED_DECODE = "mixed_decode"
    BATCH_UNSUPPORTED = "batch_unsupported"
    MISSING_METADATA = "missing_metadata"
    INVALID_LENGTHS = "invalid_lengths"
    NO_SPARSE_MIDDLE = "no_sparse_middle"
    BELOW_MIN_SEQ_LEN = "below_min_seq_len"
    BELOW_MIN_SPARSE_ROWS = "below_min_sparse_rows"
    BELOW_MIN_SAVED_QK = "below_min_saved_qk"
    BELOW_SPLIT_MIN_SPARSE_ROWS = "below_split_min_sparse_rows"
    BELOW_SPLIT_MIN_SAVED_QK = "below_split_min_saved_qk"
    GEOMETRY_UNSUPPORTED = "geometry_unsupported"
    ADAPTER_UNAVAILABLE = "adapter_unavailable"
    DIRECT_LAUNCH_ERROR = "direct_launch_error"
    STATIC_CAPABILITY = "static_capability"
    GRAPH_CAPTURE = "graph_capture"
    CONTEXT_PARALLEL = "context_parallel"
    TENSOR_PARALLEL = "tensor_parallel"
    NON_CAUSAL = "non_causal"
    MODEL_UNSUPPORTED = "model_unsupported"
    QUERY_UNSUPPORTED = "query_unsupported"
    KV_CACHE_UNSUPPORTED = "kv_cache_unsupported"
    BLOCK_TABLE_UNSUPPORTED = "block_table_unsupported"


@dataclass(frozen=True)
class TriangleRequestPlan:
    request_index: int
    q_begin: int
    q_end: int
    seq_len: int
    prompt_len: int
    span: TriangleSparseSpan | None
    eligible: bool
    reason: FallbackReason

    @property
    def query_len(self) -> int:
        return self.q_end - self.q_begin

    @property
    def saved_qk(self) -> int:
        return 0 if self.span is None else self.span.saved_qk


@dataclass(frozen=True)
class TriangleBatchPlan:
    plan_id: int
    state_name: str
    batch_size: int
    requests: tuple[TriangleRequestPlan, ...]
    direct: bool
    routing_ns: int

    @property
    def primary_reason(self) -> FallbackReason:
        for request in self.requests:
            if request.reason is not FallbackReason.NONE:
                return request.reason
        return FallbackReason.NONE


_PLAN_IDS = itertools.count(1)
_PREFILL_STATES = {
    "PrefillNoCache",
    "PrefillCacheHit",
    "ChunkedPrefill",
}


def _request_reason(
    *,
    query_len: int,
    seq_len: int,
    prompt_len: int,
    config: TriangleMixConfig,
) -> tuple[TriangleSparseSpan | None, FallbackReason]:
    if query_len <= 0 or seq_len < query_len or prompt_len < seq_len:
        return None, FallbackReason.INVALID_LENGTHS
    span = triangle_sparse_span(
        query_len=query_len,
        seq_len=seq_len,
        prompt_len=prompt_len,
        config=config,
    )
    if not config.has_supported_geometry:
        return span, FallbackReason.GEOMETRY_UNSUPPORTED
    if not span.has_sparse_middle:
        return span, FallbackReason.NO_SPARSE_MIDDLE
    if seq_len < config.direct_min_seq_len:
        return span, FallbackReason.BELOW_MIN_SEQ_LEN
    if span.sparse_rows < config.direct_min_sparse_rows:
        return span, FallbackReason.BELOW_MIN_SPARSE_ROWS
    if span.saved_qk < config.direct_min_saved_qk:
        return span, FallbackReason.BELOW_MIN_SAVED_QK
    if span.is_split:
        if span.sparse_rows < config.direct_split_min_sparse_rows:
            return span, FallbackReason.BELOW_SPLIT_MIN_SPARSE_ROWS
        if span.saved_qk < config.direct_split_min_saved_qk:
            return span, FallbackReason.BELOW_SPLIT_MIN_SAVED_QK
    return span, FallbackReason.NONE


def build_batch_plan(
    *,
    state_name: str,
    cumulative_query_ends: Sequence[int] | None,
    seq_lens: Sequence[int] | None,
    prompt_lens: Sequence[int] | None,
    num_decodes: int,
    num_prefills: int,
    config: TriangleMixConfig,
) -> TriangleBatchPlan:
    """Build one immutable plan reused by every selected attention layer."""
    started = time.perf_counter_ns()
    query_ends = tuple(int(value) for value in (cumulative_query_ends or ()))
    sequences = tuple(int(value) for value in (seq_lens or ()))
    prompts = tuple(int(value) for value in (prompt_lens or ()))
    batch_size = len(sequences)
    global_reason = FallbackReason.NONE
    if state_name not in _PREFILL_STATES:
        global_reason = FallbackReason.STATE_UNSUPPORTED
    elif num_decodes:
        global_reason = FallbackReason.MIXED_DECODE
    elif (
        not query_ends
        or not sequences
        or not prompts
        or len(query_ends) != batch_size
        or len(prompts) != batch_size
        or num_prefills != batch_size
    ):
        global_reason = FallbackReason.MISSING_METADATA
    elif batch_size != 1:
        global_reason = FallbackReason.BATCH_UNSUPPORTED

    requests: list[TriangleRequestPlan] = []
    previous_end = 0
    for index in range(batch_size):
        cumulative_end = (
            query_ends[index] if index < len(query_ends) else previous_end
        )
        query_len = cumulative_end - previous_end
        seq_len = sequences[index]
        prompt_len = prompts[index] if index < len(prompts) else seq_len
        span, local_reason = _request_reason(
            query_len=query_len,
            seq_len=seq_len,
            prompt_len=prompt_len,
            config=config,
        )
        reason = global_reason if global_reason is not FallbackReason.NONE else local_reason
        requests.append(
            TriangleRequestPlan(
                request_index=index,
                q_begin=seq_len - query_len,
                q_end=seq_len,
                seq_len=seq_len,
                prompt_len=prompt_len,
                span=span,
                eligible=reason is FallbackReason.NONE,
                reason=reason,
            )
        )
        previous_end = cumulative_end

    if not requests:
        requests.append(
            TriangleRequestPlan(
                request_index=0,
                q_begin=0,
                q_end=0,
                seq_len=0,
                prompt_len=0,
                span=None,
                eligible=False,
                reason=(
                    global_reason
                    if global_reason is not FallbackReason.NONE
                    else FallbackReason.MISSING_METADATA
                ),
            )
        )
    direct = len(requests) == 1 and requests[0].eligible
    return TriangleBatchPlan(
        plan_id=next(_PLAN_IDS),
        state_name=state_name,
        batch_size=batch_size,
        requests=tuple(requests),
        direct=direct,
        routing_ns=time.perf_counter_ns() - started,
    )
