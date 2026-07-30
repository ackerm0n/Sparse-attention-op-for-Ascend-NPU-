#!/usr/bin/env python3
"""Pure-Python case and span definitions for the NPU wrapper harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

QUERY_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
PAGE_SIZE = 128
SINK_TOKENS = 8
LOCAL_WINDOW = 512
DENSE_TAIL = 128
STREAMING_START = SINK_TOKENS + LOCAL_WINDOW + 1


@dataclass(frozen=True)
class WrapperCase:
    """One batch-one chunked-prefill geometry."""

    name: str
    query_start: int
    query_tokens: int
    prompt_len: int
    seed: int
    coverage: str

    def __post_init__(self) -> None:
        if self.query_start < 0:
            raise ValueError(f"{self.name}: query_start must be non-negative")
        if self.query_tokens <= 0:
            raise ValueError(f"{self.name}: query_tokens must be positive")
        if self.prompt_len < self.seq_len:
            raise ValueError(
                f"{self.name}: prompt_len must be at least seq_len"
            )

    @property
    def seq_len(self) -> int:
        return self.query_start + self.query_tokens

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "seq_len": self.seq_len}


@dataclass(frozen=True)
class SpanPlan:
    """Exact outer-wrapper absolute-coordinate dispatch plan."""

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
    def prefix_rows(self) -> int:
        if not self.has_sparse_middle:
            return self.query_rows
        return self.s0 - self.q0

    @property
    def tail_rows(self) -> int:
        if not self.has_sparse_middle:
            return 0
        return self.q1 - self.s1

    @property
    def saved_qk(self) -> int:
        if not self.has_sparse_middle:
            return 0
        rows = self.s1 - self.s0
        return rows * (self.s0 + self.s1 - 1041) // 2

    @property
    def dense_qk(self) -> int:
        rows = self.query_rows
        return rows * (self.q0 + self.q1 + 1) // 2

    @property
    def candidate_qk(self) -> int:
        return self.dense_qk - self.saved_qk

    @property
    def split_kind(self) -> str:
        if not self.has_sparse_middle:
            return "dense_fallback"
        has_prefix = self.q0 < self.s0
        has_tail = self.s1 < self.q1
        if has_prefix and has_tail:
            return "prefix_middle_tail"
        if has_prefix:
            return "prefix_sparse"
        if has_tail:
            return "sparse_tail"
        return "full_sparse"

    @property
    def stage_count(self) -> int:
        if not self.has_sparse_middle:
            return 1
        return (
            1
            + int(self.q0 < self.s0)
            + int(self.s1 < self.q1)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "query_rows": self.query_rows,
            "sparse_rows": self.sparse_rows,
            "prefix_rows": self.prefix_rows,
            "tail_rows": self.tail_rows,
            "saved_qk": self.saved_qk,
            "dense_qk": self.dense_qk,
            "candidate_qk": self.candidate_qk,
            "qk_fraction": self.candidate_qk / self.dense_qk,
            "split_kind": self.split_kind,
            "stage_count": self.stage_count,
            "has_sparse_middle": self.has_sparse_middle,
        }


def plan_for_case(case: WrapperCase) -> SpanPlan:
    q0 = case.query_start
    q1 = case.seq_len
    s0 = max(q0, STREAMING_START)
    s1 = min(q1, max(0, case.prompt_len - DENSE_TAIL))
    return SpanPlan(q0=q0, q1=q1, s0=s0, s1=s1)


CORRECTNESS_CASES = (
    WrapperCase(
        name="prefix_middle_tail",
        query_start=500,
        query_tokens=100,
        prompt_len=650,
        seed=4101,
        coverage=(
            "dense prefix [500,521), sparse [521,522), "
            "dense tail [522,600)"
        ),
    ),
    WrapperCase(
        name="full_sparse",
        query_start=521,
        query_tokens=65,
        prompt_len=1024,
        seed=4102,
        coverage="one custom stage covering [521,586)",
    ),
    WrapperCase(
        name="prefix_only_split",
        query_start=500,
        query_tokens=65,
        prompt_len=1024,
        seed=4103,
        coverage="dense prefix followed by sparse middle, no dense tail",
    ),
    WrapperCase(
        name="tail_split",
        query_start=864,
        query_tokens=64,
        prompt_len=1024,
        seed=4104,
        coverage="32 sparse rows followed by 32 final dense rows",
    ),
    WrapperCase(
        name="chunk_query_start_gt_zero",
        query_start=2048,
        query_tokens=65,
        prompt_len=4096,
        seed=4105,
        coverage="long chunk with q0>0 and randomized paged KV mapping",
    ),
    WrapperCase(
        name="prompt_649_no_middle",
        query_start=0,
        query_tokens=649,
        prompt_len=649,
        seed=4106,
        coverage="s0 == s1 == 521; wrapper must use one dense fallback",
    ),
    WrapperCase(
        name="prompt_650_first_middle",
        query_start=0,
        query_tokens=650,
        prompt_len=650,
        seed=4107,
        coverage="first one-row sparse middle at absolute row 521",
    ),
    WrapperCase(
        name="streaming_521_no_middle",
        query_start=0,
        query_tokens=521,
        prompt_len=1024,
        seed=4108,
        coverage="q1 == streaming start; no custom launch",
    ),
)


PROMPT_8320_CHUNK2048_ROUTE = (
    WrapperCase(
        "route_p8320_chunk0_q0_0_t2048",
        0,
        2048,
        8320,
        5118,
        (
            "complete 8320-token/chunk-2048 route: first chunk; "
            "measure whether split-launch overhead requires dense fallback"
        ),
    ),
    WrapperCase(
        "route_p8320_chunk1_q0_2048_t2048",
        2048,
        2048,
        8320,
        5119,
        "complete 8320-token/chunk-2048 route: second chunk",
    ),
    WrapperCase(
        "route_p8320_chunk2_q0_4096_t2048",
        4096,
        2048,
        8320,
        5120,
        "complete 8320-token/chunk-2048 route: third chunk",
    ),
    WrapperCase(
        "route_p8320_chunk3_q0_6144_t2048",
        6144,
        2048,
        8320,
        5121,
        "complete 8320-token/chunk-2048 route: fourth chunk",
    ),
    WrapperCase(
        "route_p8320_chunk4_q0_8192_t128",
        8192,
        128,
        8320,
        5122,
        (
            "complete 8320-token/chunk-2048 route: final 128 rows; "
            "structural dense fallback because all rows are in the dense tail"
        ),
    ),
)


def _make_full_sparse_freeze_cases() -> tuple[WrapperCase, ...]:
    """Fine saved-QK scans at realizable 8320-route history positions."""
    points = (
        (2048, 448),
        (2048, 480),
        (2048, 496),
        (2048, 504),
        (2048, 508),
        (2048, 512),
        (2048, 516),
        (2048, 520),
        (2048, 528),
        (2048, 544),
        (2048, 576),
        (4096, 192),
        (4096, 208),
        (4096, 224),
        (4096, 232),
        (4096, 240),
        (4096, 244),
        (4096, 248),
        (4096, 256),
        (4096, 272),
        (4096, 288),
        (6144, 128),
        (6144, 136),
        (6144, 144),
        (6144, 152),
        (6144, 156),
        (6144, 160),
        (6144, 164),
        (6144, 168),
        (6144, 176),
        (6144, 192),
    )
    return tuple(
        WrapperCase(
            name=f"freeze_full_q0_{query_start}_tq_{query_tokens}",
            query_start=query_start,
            query_tokens=query_tokens,
            prompt_len=8320,
            seed=6100 + index,
            coverage=(
                "full-sparse fallback freeze near saved-QK crossover at an "
                "8320-route history position"
            ),
        )
        for index, (query_start, query_tokens) in enumerate(points)
    )


def _make_split_freeze_cases() -> tuple[WrapperCase, ...]:
    """Realizable split scans for the production 2048-token chunk budget."""
    prefix_query_lengths = (
        1792,
        1920,
        1984,
        2016,
    )
    cases = [
        WrapperCase(
            name=f"freeze_prefix_q0_0_tq_{query_tokens}",
            query_start=0,
            query_tokens=query_tokens,
            prompt_len=8320,
            seed=6200 + index,
            coverage=(
                "prefix-split fallback freeze around the realizable first "
                "8320-route chunk under a 2048-token scheduler budget"
            ),
        )
        for index, query_tokens in enumerate(prefix_query_lengths)
    ]

    tail_history_positions = (
        521,
        1024,
        1536,
        2048,
        2560,
        3072,
        3584,
        3712,
        3776,
        3840,
        3904,
        3968,
        4032,
        4096,
        4608,
        5120,
        5632,
        5888,
        6144,
        6400,
        6656,
        6912,
        7040,
        7168,
        7424,
        7552,
        7616,
        7648,
        7680,
        7712,
        7744,
        7808,
        7936,
    )
    cases.extend(
        WrapperCase(
            name=f"freeze_tail_q0_{query_start}_tq_512",
            query_start=query_start,
            query_tokens=512,
            prompt_len=query_start + 512,
            seed=6300 + index,
            coverage=(
                "128-row tail split with saved-QK varied around the split "
                "crossover"
            ),
        )
        for index, query_start in enumerate(tail_history_positions)
    )

    tail_query_lengths = (385, 400, 416, 448, 480, 496)
    cases.extend(
        WrapperCase(
            name=f"freeze_tail_q0_7680_tq_{query_tokens}",
            query_start=7680,
            query_tokens=query_tokens,
            prompt_len=8192,
            seed=6400 + index,
            coverage=(
                "fixed 384-row sparse tail split with tail launch varied "
                "from one to 128 rows"
            ),
        )
        for index, query_tokens in enumerate(tail_query_lengths)
    )

    # Joint saved-QK/sparse-row boundary: every call ends at its final
    # prompt position and therefore contains exactly a 128-row dense tail.
    # The pairs bracket the ~1.30M saved-QK crossover with progressively
    # shorter sparse launches as history grows.
    joint_boundary = (
        (4096, 320),
        (4096, 336),
        (4096, 352),
        (4096, 368),
        (4096, 384),
        (5120, 240),
        (5120, 256),
        (5120, 272),
        (5120, 288),
        (5120, 304),
        (6144, 192),
        (6144, 208),
        (6144, 224),
        (6144, 240),
        (6144, 256),
        (7680, 144),
        (7680, 160),
        (7680, 176),
        (7680, 192),
        (7680, 208),
    )
    cases.extend(
        WrapperCase(
            name=(
                f"freeze_tail_joint_q0_{query_start}_"
                f"rows_{sparse_rows}"
            ),
            query_start=query_start,
            query_tokens=sparse_rows + 128,
            prompt_len=query_start + sparse_rows + 128,
            seed=6500 + index,
            coverage=(
                "joint split fallback freeze near 1.30M saved QK with "
                "a 128-row final dense tail"
            ),
        )
        for index, (query_start, sparse_rows) in enumerate(joint_boundary)
    )
    return tuple(cases)


FALLBACK_FULL_SPARSE_FREEZE_CASES = _make_full_sparse_freeze_cases()
FALLBACK_SPLIT_FREEZE_CASES = _make_split_freeze_cases()


DEFAULT_SWEEP_CASES = (
    WrapperCase(
        "sweep_prefix_middle_tail",
        500,
        400,
        1024,
        5101,
        "three-stage short split",
    ),
    WrapperCase(
        "sweep_prefix_sparse",
        500,
        300,
        1024,
        5102,
        "two-stage dense-prefix split",
    ),
    WrapperCase(
        "sweep_full_sparse_boundary",
        521,
        375,
        1024,
        5103,
        "minimum full-sparse interval ending at dense-tail boundary",
    ),
    WrapperCase(
        "sweep_sparse_tail",
        800,
        224,
        1024,
        5104,
        "two-stage dense-tail split",
    ),
    WrapperCase(
        "sweep_q0_521_t32",
        521,
        32,
        8192,
        5105,
        "small saved-QK full-sparse call",
    ),
    WrapperCase(
        "sweep_q0_1024_t32",
        1024,
        32,
        8192,
        5106,
        "history sweep, Tq=32",
    ),
    WrapperCase(
        "sweep_q0_1024_t128",
        1024,
        128,
        8192,
        5107,
        "query-size sweep, Tq=128",
    ),
    WrapperCase(
        "sweep_q0_1024_t512",
        1024,
        512,
        8192,
        5108,
        "query-size sweep, Tq=512",
    ),
    WrapperCase(
        "sweep_q0_2048_t32",
        2048,
        32,
        8192,
        5109,
        "history sweep, q0=2048 and Tq=32",
    ),
    WrapperCase(
        "sweep_q0_2048_t128",
        2048,
        128,
        8192,
        5110,
        "history/query sweep, q0=2048 and Tq=128",
    ),
    WrapperCase(
        "sweep_q0_2048_t512",
        2048,
        512,
        8192,
        5111,
        "history/query sweep, q0=2048 and Tq=512",
    ),
    WrapperCase(
        "sweep_prompt4096_q0_2048_t512",
        2048,
        512,
        4096,
        5117,
        "same q0/Tq at an intermediate final prompt length",
    ),
    WrapperCase(
        "sweep_q0_2048_t1024",
        2048,
        1024,
        8192,
        5112,
        "history/query sweep, q0=2048 and Tq=1024",
    ),
    WrapperCase(
        "sweep_q0_4096_t128",
        4096,
        128,
        8192,
        5113,
        "long-history Tq=128",
    ),
    WrapperCase(
        "sweep_q0_4096_t512",
        4096,
        512,
        8192,
        5114,
        "long-history Tq=512",
    ),
    WrapperCase(
        "sweep_q0_4096_t2048",
        4096,
        2048,
        8192,
        5115,
        "long-history Tq=2048",
    ),
    WrapperCase(
        "sweep_prompt8192_tail",
        7680,
        512,
        8192,
        5116,
        "same long prompt with explicit final dense-tail split",
    ),
) + PROMPT_8320_CHUNK2048_ROUTE


ALL_NAMED_SWEEP_CASES = (
    DEFAULT_SWEEP_CASES
    + FALLBACK_FULL_SPARSE_FREEZE_CASES
    + FALLBACK_SPLIT_FREEZE_CASES
)


def cross_product_cases(
    query_starts: Iterable[int],
    query_lengths: Iterable[int],
    prompt_lengths: Iterable[int],
    *,
    seed: int,
) -> tuple[WrapperCase, ...]:
    """Build a deterministic valid cross-product latency matrix."""
    cases: list[WrapperCase] = []
    index = 0
    for prompt_len in sorted(set(prompt_lengths)):
        for query_start in sorted(set(query_starts)):
            for query_tokens in sorted(set(query_lengths)):
                seq_len = query_start + query_tokens
                if (
                    query_start < 0
                    or query_tokens <= 0
                    or prompt_len < seq_len
                ):
                    continue
                cases.append(
                    WrapperCase(
                        name=(
                            f"axis_q0_{query_start}_tq_{query_tokens}"
                            f"_p_{prompt_len}"
                        ),
                        query_start=query_start,
                        query_tokens=query_tokens,
                        prompt_len=prompt_len,
                        seed=seed + index,
                        coverage="user-requested query_start/Tq/prompt cross-product",
                    )
                )
                index += 1
    if not cases:
        raise ValueError("axis cross-product produced no valid cases")
    return tuple(cases)
