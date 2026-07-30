#!/usr/bin/env python3
"""CPU/static contracts; these tests never import torch or touch an NPU."""

from __future__ import annotations

import ast
import pathlib
import unittest

from harness_spec import (
    ALL_NAMED_SWEEP_CASES,
    CORRECTNESS_CASES,
    DEFAULT_SWEEP_CASES,
    FALLBACK_FULL_SPARSE_FREEZE_CASES,
    FALLBACK_SPLIT_FREEZE_CASES,
    PROMPT_8320_CHUNK2048_ROUTE,
    WrapperCase,
    cross_product_cases,
    plan_for_case,
)

ROOT = pathlib.Path(__file__).resolve().parent


class SpanAndMatrixContractTest(unittest.TestCase):
    def test_exact_prefix_middle_tail_coordinates(self) -> None:
        case = next(
            case
            for case in CORRECTNESS_CASES
            if case.name == "prefix_middle_tail"
        )
        plan = plan_for_case(case)
        self.assertEqual(
            (plan.q0, plan.q1, plan.s0, plan.s1),
            (500, 600, 521, 522),
        )
        self.assertEqual(plan.split_kind, "prefix_middle_tail")
        self.assertEqual(
            (plan.prefix_rows, plan.sparse_rows, plan.tail_rows),
            (21, 1, 78),
        )

    def test_every_required_dispatch_class_is_present(self) -> None:
        plans = {
            case.name: plan_for_case(case)
            for case in CORRECTNESS_CASES
        }
        kinds = {plan.split_kind for plan in plans.values()}
        self.assertTrue(
            {
                "prefix_middle_tail",
                "prefix_sparse",
                "sparse_tail",
                "full_sparse",
                "dense_fallback",
            }.issubset(kinds)
        )
        self.assertGreater(
            next(
                case.query_start
                for case in CORRECTNESS_CASES
                if case.name == "chunk_query_start_gt_zero"
            ),
            0,
        )

    def test_649_650_and_521_fallback_boundaries(self) -> None:
        by_name = {case.name: case for case in CORRECTNESS_CASES}
        before = plan_for_case(by_name["prompt_649_no_middle"])
        first = plan_for_case(by_name["prompt_650_first_middle"])
        streaming = plan_for_case(
            by_name["streaming_521_no_middle"]
        )
        self.assertFalse(before.has_sparse_middle)
        self.assertEqual((before.s0, before.s1), (521, 521))
        self.assertTrue(first.has_sparse_middle)
        self.assertEqual((first.s0, first.s1), (521, 522))
        self.assertFalse(streaming.has_sparse_middle)

    def test_saved_qk_closed_form_and_work_accounting(self) -> None:
        case = WrapperCase(
            "known",
            query_start=500,
            query_tokens=400,
            prompt_len=1024,
            seed=1,
            coverage="known arithmetic example",
        )
        plan = plan_for_case(case)
        self.assertEqual((plan.s0, plan.s1), (521, 896))
        self.assertEqual(plan.saved_qk, 70500)
        self.assertEqual(
            plan.candidate_qk,
            plan.dense_qk - plan.saved_qk,
        )

    def test_default_sweep_is_genuinely_multidimensional(self) -> None:
        plans = [plan_for_case(case) for case in DEFAULT_SWEEP_CASES]
        self.assertGreaterEqual(
            len({case.query_start for case in DEFAULT_SWEEP_CASES}),
            5,
        )
        self.assertGreaterEqual(
            len({case.query_tokens for case in DEFAULT_SWEEP_CASES}),
            5,
        )
        self.assertGreaterEqual(
            len({case.prompt_len for case in DEFAULT_SWEEP_CASES}),
            3,
        )
        self.assertGreaterEqual(len({plan.split_kind for plan in plans}), 4)
        self.assertGreaterEqual(len({plan.saved_qk for plan in plans}), 10)

    def test_axis_cross_product_filters_invalid_cells(self) -> None:
        cases = cross_product_cases(
            query_starts=[0, 500],
            query_lengths=[32, 256],
            prompt_lengths=[128, 1024],
            seed=7000,
        )
        self.assertTrue(cases)
        self.assertTrue(all(case.seq_len <= case.prompt_len for case in cases))
        self.assertEqual(len({case.name for case in cases}), len(cases))

    def test_complete_8320_chunk2048_route_is_in_default_sweep(self) -> None:
        route_geometry = [
            (
                case.query_start,
                case.query_tokens,
                case.prompt_len,
            )
            for case in PROMPT_8320_CHUNK2048_ROUTE
        ]
        self.assertEqual(
            route_geometry,
            [
                (0, 2048, 8320),
                (2048, 2048, 8320),
                (4096, 2048, 8320),
                (6144, 2048, 8320),
                (8192, 128, 8320),
            ],
        )
        default_names = {case.name for case in DEFAULT_SWEEP_CASES}
        self.assertTrue(
            all(
                case.name in default_names
                for case in PROMPT_8320_CHUNK2048_ROUTE
            )
        )
        route_plans = [
            plan_for_case(case)
            for case in PROMPT_8320_CHUNK2048_ROUTE
        ]
        self.assertEqual(route_plans[0].split_kind, "prefix_sparse")
        self.assertTrue(
            all(
                plan.split_kind == "full_sparse"
                for plan in route_plans[1:4]
            )
        )
        self.assertEqual(route_plans[4].split_kind, "dense_fallback")

    def test_fallback_freeze_matrix_brackets_both_coarse_boundaries(
        self,
    ) -> None:
        names = [case.name for case in ALL_NAMED_SWEEP_CASES]
        self.assertEqual(len(names), len(set(names)))
        full_plans = [
            plan_for_case(case)
            for case in FALLBACK_FULL_SPARSE_FREEZE_CASES
        ]
        self.assertTrue(
            all(plan.split_kind == "full_sparse" for plan in full_plans)
        )
        self.assertLess(min(plan.saved_qk for plan in full_plans), 913152)
        self.assertGreater(max(plan.saved_qk for plan in full_plans), 913152)
        self.assertEqual(
            {
                case.query_start
                for case in FALLBACK_FULL_SPARSE_FREEZE_CASES
            },
            {2048, 4096, 6144},
        )

        split_plans = [
            plan_for_case(case)
            for case in FALLBACK_SPLIT_FREEZE_CASES
        ]
        self.assertLessEqual(
            max(case.query_tokens for case in ALL_NAMED_SWEEP_CASES),
            2048,
        )
        self.assertTrue(
            {"prefix_sparse", "sparse_tail"}.issubset(
                {plan.split_kind for plan in split_plans}
            )
        )
        tail_saved = [
            plan.saved_qk
            for plan in split_plans
            if plan.split_kind == "sparse_tail"
        ]
        tail_rows = [
            plan.sparse_rows
            for plan in split_plans
            if plan.split_kind == "sparse_tail"
        ]
        self.assertLess(min(tail_saved), 1_000_000)
        self.assertLess(min(tail_saved), 2822976)
        self.assertGreater(max(tail_saved), 2822976)
        self.assertIn(2822976, tail_saved)
        self.assertLess(min(tail_rows), 192)
        self.assertGreaterEqual(max(tail_rows), 384)


class RuntimeSourceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.common = (ROOT / "npu_harness_common.py").read_text()
        cls.correctness = (
            ROOT / "npu_wrapper_correctness.py"
        ).read_text()
        cls.crossover = (
            ROOT / "npu_wrapper_crossover.py"
        ).read_text()

    def test_runtime_sources_parse(self) -> None:
        ast.parse(self.common)
        ast.parse(self.correctness)
        ast.parse(self.crossover)

    def test_exact_candidate_and_adapter_are_loaded_by_path(self) -> None:
        self.assertIn("spec_from_file_location(", self.common)
        self.assertIn(
            "module.load_triangle_mix_adapter(str(adapter_path))",
            self.common,
        )
        self.assertNotIn("from vllm", self.common)
        self.assertNotIn("from vllm", self.correctness)

    def test_dense_and_direct_out_paths_are_real(self) -> None:
        self.assertIn(
            "torch.ops.npu.npu_fused_infer_attention_score.out(",
            self.common,
        )
        self.assertIn(
            "wrapper.triangle_direct_paged_attention(",
            self.common,
        )
        self.assertIn(
            "torch.ops.trianglemix."
            "triangle_paged_sparse_attention.out(",
            self.common,
        )

    def test_independent_fp32_triangle_reference_is_a_hard_gate(self) -> None:
        self.assertIn(
            "def fp32_triangle_middle_reference(",
            self.common,
        )
        self.assertIn("torch.softmax(scores, dim=-1)", self.common)
        self.assertIn("query_position - 512", self.common)
        self.assertIn("sink_indices = torch.arange(8", self.common)
        self.assertIn(
            '"middle_vs_cpu_fp32_triangle"',
            self.common,
        )
        self.assertIn(
            "triangle_reference_max_abs",
            self.common,
        )

    def test_random_pages_lock_sync_and_json_are_mandatory(self) -> None:
        self.assertIn("torch.randperm(", self.common)
        self.assertIn("fcntl.LOCK_EX | fcntl.LOCK_NB", self.common)
        self.assertIn("torch.npu.synchronize()", self.common)
        self.assertIn("exclusive_npu_lock(", self.correctness)
        self.assertIn("json.dumps(report", self.correctness)

    def test_latency_uses_events_exact_abba_and_reports_dimensions(self) -> None:
        self.assertIn("event_sample_ms(", self.crossover)
        self.assertIn("repeat_index % 2", self.crossover)
        self.assertIn("exact ABBA", self.crossover)
        self.assertIn("torch.npu.synchronize()", self.crossover)
        self.assertIn("exclusive_npu_lock(", self.crossover)
        self.assertIn('"saved_qk"', self.crossover)
        self.assertIn('"split_kind"', self.crossover)
        self.assertIn(
            "query_start: int",
            (ROOT / "harness_spec.py").read_text(),
        )
        self.assertIn("--query-starts", self.crossover)
        self.assertIn("--query-lengths", self.crossover)
        self.assertIn("--prompt-lengths", self.crossover)
        self.assertIn("--case-group", self.crossover)
        self.assertIn('"fallback-freeze"', self.crossover)
        self.assertIn(
            '"conservative_dense_minus_direct_ms"',
            self.crossover,
        )
        self.assertIn("tukey_outlier_count", self.common)
        self.assertIn("median_absolute_deviation_ms", self.common)
        self.assertIn(
            "summarize_prompt_8320_route(",
            self.crossover,
        )
        self.assertIn(
            '"single_layer_robust_policy_sum_ms"',
            self.crossover,
        )
        self.assertIn("json.dumps(report", self.crossover)


if __name__ == "__main__":
    unittest.main(verbosity=2)
