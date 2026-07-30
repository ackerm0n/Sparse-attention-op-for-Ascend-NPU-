#!/usr/bin/env python3
"""Contract tests for dense/sparse query-span splitting in the fast path."""

from __future__ import annotations

import unittest
from pathlib import Path

from triangle_v2_reference import (
    PRODUCTION_GEOMETRY,
    QuerySpan,
    selected_token_indices,
    split_query_spans,
    submitted_span_intervals,
)


GEOMETRY = PRODUCTION_GEOMETRY
ROOT = Path(__file__).resolve().parents[1]


class QuerySpanContractTest(unittest.TestCase):
    def assert_exact_row_coverage(
        self,
        spans: tuple[QuerySpan, ...],
        *,
        query_start: int,
        query_tokens: int,
        sparse_begin: int,
        sparse_end: int,
    ) -> None:
        covered_rows: list[int] = []
        for span in spans:
            self.assertLessEqual(span.row_count, GEOMETRY.q_tile)
            row_begin = span.begin - query_start
            row_end = span.end - query_start
            self.assertGreaterEqual(row_begin, 0)
            self.assertLessEqual(row_end, query_tokens)
            self.assertEqual(
                row_begin // GEOMETRY.q_tile,
                (row_end - 1) // GEOMETRY.q_tile,
            )
            covered_rows.extend(range(row_begin, row_end))
            for query_position in range(span.begin, span.end):
                self.assertEqual(
                    span.sparse,
                    sparse_begin <= query_position < sparse_end,
                )
        self.assertEqual(covered_rows, list(range(query_tokens)))
        self.assertEqual(len(covered_rows), len(set(covered_rows)))

    def assert_sparse_spans_skip_middle(
        self,
        spans: tuple[QuerySpan, ...],
        *,
        query_start: int,
        seq_len: int,
        prompt_len: int,
    ) -> None:
        for span in spans:
            if not span.sparse:
                continue
            intervals = submitted_span_intervals(
                span,
                seq_len,
                GEOMETRY,
            )
            local_begin = max(0, span.begin - GEOMETRY.local_window)
            sink_end = min(GEOMETRY.sink_tokens, seq_len)
            self.assertGreater(local_begin, sink_end)
            self.assertEqual(
                intervals,
                (
                    (0, sink_end),
                    (local_begin, span.end),
                ),
            )

            # No submitted interval may intersect [sink_end, local_begin).
            for interval_begin, interval_end in intervals:
                overlap = max(
                    0,
                    min(interval_end, local_begin)
                    - max(interval_begin, sink_end),
                )
                self.assertEqual(overlap, 0)

            # Applying the per-row causal/window boundary to the submitted
            # union must reconstruct the exact row oracle, without middle.
            submitted = {
                token
                for interval_begin, interval_end in intervals
                for token in range(interval_begin, interval_end)
            }
            for query_position in range(span.begin, span.end):
                after_boundary_mask = {
                    token
                    for token in submitted
                    if (
                        token <= query_position
                        and (
                            token < GEOMETRY.sink_tokens
                            or query_position - token
                            <= GEOMETRY.local_window
                        )
                    )
                }
                expected = set(
                    selected_token_indices(
                        query_position,
                        seq_len,
                        prompt_len,
                        GEOMETRY,
                    )
                )
                self.assertEqual(after_boundary_mask, expected)

    def test_q_tile_crosses_sparse_begin_521_exactly(self) -> None:
        query_start = 512
        query_tokens = 32
        sparse_begin = 521
        sparse_end = 8192
        spans = split_query_spans(
            query_start,
            query_tokens,
            sparse_begin,
            sparse_end,
            GEOMETRY,
        )
        self.assertEqual(
            spans,
            (
                QuerySpan(512, 521, False),
                QuerySpan(521, 544, True),
            ),
        )
        self.assert_exact_row_coverage(
            spans,
            query_start=query_start,
            query_tokens=query_tokens,
            sparse_begin=sparse_begin,
            sparse_end=sparse_end,
        )
        self.assert_sparse_spans_skip_middle(
            spans,
            query_start=query_start,
            seq_len=query_start + query_tokens,
            prompt_len=8320,
        )

    def test_q_tile_crosses_sparse_end_prompt_minus_128_exactly(self) -> None:
        prompt_len = 1024
        sparse_begin = 521
        sparse_end = prompt_len - 128
        query_start = 880
        query_tokens = 32
        spans = split_query_spans(
            query_start,
            query_tokens,
            sparse_begin,
            sparse_end,
            GEOMETRY,
        )
        self.assertEqual(
            spans,
            (
                QuerySpan(880, 896, True),
                QuerySpan(896, 912, False),
            ),
        )
        self.assert_exact_row_coverage(
            spans,
            query_start=query_start,
            query_tokens=query_tokens,
            sparse_begin=sparse_begin,
            sparse_end=sparse_end,
        )
        self.assert_sparse_spans_skip_middle(
            spans,
            query_start=query_start,
            seq_len=query_start + query_tokens,
            prompt_len=prompt_len,
        )

    def test_unaligned_query_start_and_ragged_last_tile(self) -> None:
        query_start = 503
        query_tokens = 67
        sparse_begin = 521
        sparse_end = 8192
        spans = split_query_spans(
            query_start,
            query_tokens,
            sparse_begin,
            sparse_end,
            GEOMETRY,
        )
        self.assertEqual(
            spans,
            (
                QuerySpan(503, 521, False),
                QuerySpan(521, 535, True),
                QuerySpan(535, 567, True),
                QuerySpan(567, 570, True),
            ),
        )
        self.assertEqual(spans[-1].row_count, 3)
        self.assert_exact_row_coverage(
            spans,
            query_start=query_start,
            query_tokens=query_tokens,
            sparse_begin=sparse_begin,
            sparse_end=sparse_end,
        )
        self.assert_sparse_spans_skip_middle(
            spans,
            query_start=query_start,
            seq_len=query_start + query_tokens,
            prompt_len=8320,
        )

    def test_ragged_tile_crosses_dense_tail_boundary(self) -> None:
        prompt_len = 1024
        sparse_begin = 521
        sparse_end = prompt_len - 128
        query_start = 883
        query_tokens = 30
        spans = split_query_spans(
            query_start,
            query_tokens,
            sparse_begin,
            sparse_end,
            GEOMETRY,
        )
        self.assertEqual(
            spans,
            (
                QuerySpan(883, 896, True),
                QuerySpan(896, 913, False),
            ),
        )
        self.assert_exact_row_coverage(
            spans,
            query_start=query_start,
            query_tokens=query_tokens,
            sparse_begin=sparse_begin,
            sparse_end=sparse_end,
        )
        self.assert_sparse_spans_skip_middle(
            spans,
            query_start=query_start,
            seq_len=query_start + query_tokens,
            prompt_len=prompt_len,
        )

    def test_cpp_fast_path_binds_each_query_span_independently(self) -> None:
        schedule_source = (
            ROOT / "op_kernel/triangle_schedule.h"
        ).read_text()
        fast_path_source = (
            ROOT / "op_kernel/triangle_paged_fia_fast_path.h"
        ).read_text()

        self.assertIn("struct QuerySpan {", schedule_source)
        self.assertIn("uint32_t begin;", schedule_source)
        self.assertIn("uint32_t end;", schedule_source)
        self.assertIn("uint32_t sparse;", schedule_source)
        self.assertIn("struct QuerySpanSchedule {", schedule_source)
        self.assertIn("QuerySpan span[3];", schedule_source)
        self.assertIn("SplitQueryTile(", schedule_source)

        split_call = fast_path_source.index(
            "const QuerySpanSchedule querySpans = SplitQueryTile("
        )
        span_loop = fast_path_source.index(
            "spanIndex < querySpans.count",
            split_call,
        )
        span_query_row = fast_path_source.index(
            "const uint32_t spanQueryRow",
            span_loop,
        )
        build_schedule = fast_path_source.index(
            "BuildTileSchedule(",
            span_query_row,
        )
        tile_ordinal = fast_path_source.index(
            "uint32_t tileOrdinal = 0;",
            build_schedule,
        )
        first_flag = fast_path_source.index(
            "const bool first = tileOrdinal == 0U;",
            tile_ordinal,
        )
        last_flag = fast_path_source.index(
            "tileOrdinal + 1U == totalTiles;",
            first_flag,
        )
        self.assertLess(
            split_call,
            span_loop,
        )
        self.assertLess(span_loop, span_query_row)
        self.assertLess(span_query_row, build_schedule)
        self.assertLess(build_schedule, tile_ordinal)
        self.assertLess(tile_ordinal, first_flag)
        self.assertLess(first_flag, last_flag)

        self.assertIn(
            "querySpan.end - querySpan.begin",
            fast_path_source,
        )
        self.assertIn(
            "const uint32_t qBegin = querySpan.begin;",
            fast_path_source,
        )
        self.assertIn(
            "const uint32_t qEnd = querySpan.end;",
            fast_path_source,
        )
        self.assertIn(
            "static_cast<uint64_t>(spanQueryRow)",
            fast_path_source,
        )
        self.assertNotIn(
            "BuildTileSchedule(\n                queryTileBegin",
            fast_path_source,
        )

    def test_fast_path_drains_final_output_events_before_exit(self) -> None:
        fast_path_source = (
            ROOT / "op_kernel/triangle_paged_fia_fast_path.h"
        ).read_text()
        finalize_call = fast_path_source.index("FinalizeEvents();")
        init_definition = fast_path_source.index(
            "__aicore__ inline void InitVectorEvents()"
        )
        finalize_definition = fast_path_source.index(
            "__aicore__ inline void FinalizeEvents()"
        )
        private_section = fast_path_source.index(
            "private:",
            finalize_call,
        )
        self.assertLess(finalize_call, private_section)
        self.assertLess(private_section, finalize_definition)

        init_body = fast_path_source[
            init_definition:finalize_definition
        ]
        finalize_body = fast_path_source[finalize_definition:]
        final_output_wait = (
            "WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID6);"
        )
        final_vector_set = (
            "SetFlag<HardEvent::MTE3_V>(EVENT_ID2);"
        )
        final_vector_wait = (
            "WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);"
        )
        full_pipe_barrier = "PipeBarrier<PIPE_ALL>();"
        self.assertEqual(init_body.count(final_vector_set), 1)
        self.assertEqual(finalize_body.count(final_vector_wait), 1)
        self.assertIn(final_output_wait, finalize_body)
        self.assertIn(full_pipe_barrier, finalize_body)
        self.assertLess(
            finalize_body.index(final_output_wait),
            finalize_body.index(full_pipe_barrier),
        )
        self.assertLess(
            finalize_body.index(final_vector_wait),
            finalize_body.index(full_pipe_barrier),
        )
        self.assertIn(
            "SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID6);",
            fast_path_source,
        )


if __name__ == "__main__":
    unittest.main()
