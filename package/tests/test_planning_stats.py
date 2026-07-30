from __future__ import annotations

import os
import sys
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")

from vllm_ascend_trianglemix.config import resolve_plugin_config
from vllm_ascend_trianglemix.kernel import TriangleMixConfig
from vllm_ascend_trianglemix.planning import (
    FallbackReason,
    build_batch_plan,
)
from vllm_ascend_trianglemix.stats import TriangleMixRuntimeStats


def config(**overrides: object) -> TriangleMixConfig:
    values: dict[str, object] = {
        "enabled": True,
        "layer_indices": frozenset({1, 3, 5}),
        "sink_tokens": 8,
        "local_window": 512,
        "last_rows": 128,
        "direct_min_seq_len": 0,
        "direct_min_sparse_rows": 128,
        "direct_min_saved_qk": 913_152,
        "direct_split_min_sparse_rows": 192,
        "direct_split_min_saved_qk": 1_299_264,
    }
    values.update(overrides)
    return TriangleMixConfig(**values)


class ConfigTests(unittest.TestCase):
    def test_additional_config_string_false_is_false(self) -> None:
        holder = types.SimpleNamespace(
            additional_config={
                "trianglemix": {
                    "enabled": "false",
                    "strict": "off",
                    "layers": "1,3-4",
                }
            }
        )
        resolved = resolve_plugin_config(holder)
        self.assertFalse(resolved.kernel.enabled)
        self.assertFalse(resolved.strict)
        self.assertEqual(resolved.kernel.layer_indices, frozenset({1, 3, 4}))

    def test_invalid_boolean_is_rejected(self) -> None:
        holder = types.SimpleNamespace(
            additional_config={
                "trianglemix": {
                    "enabled": "sometimes",
                    "layers": "1",
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            resolve_plugin_config(holder)

    def test_environment_is_compatibility_fallback(self) -> None:
        env = {
            "VLLM_ASCEND_ENABLE_TRIANGLE_MIX": "yes",
            "VLLM_ASCEND_TRIANGLE_MIX_LAYERS": "2-3",
        }
        with patch.dict(os.environ, env, clear=True):
            resolved = resolve_plugin_config(None)
        self.assertTrue(resolved.enabled)
        self.assertEqual(resolved.kernel.layer_indices, frozenset({2, 3}))


class PlanningTests(unittest.TestCase):
    def plan(
        self,
        *,
        state: str = "PrefillNoCache",
        query_ends: list[int] | None = None,
        seq_lens: list[int] | None = None,
        prompt_lens: list[int] | None = None,
        num_decodes: int = 0,
        num_prefills: int = 1,
    ):
        return build_batch_plan(
            state_name=state,
            cumulative_query_ends=query_ends,
            seq_lens=seq_lens,
            prompt_lens=prompt_lens,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            config=config(),
        )

    def test_full_prompt_split_crossover_boundary(self) -> None:
        before = self.plan(
            query_ends=[2260],
            seq_lens=[2260],
            prompt_lens=[2260],
        )
        at = self.plan(
            query_ends=[2261],
            seq_lens=[2261],
            prompt_lens=[2261],
        )
        self.assertFalse(before.direct)
        self.assertEqual(
            before.primary_reason,
            FallbackReason.BELOW_SPLIT_MIN_SAVED_QK,
        )
        self.assertTrue(at.direct)
        self.assertEqual(at.requests[0].saved_qk, 1_300_078)

    def test_chunked_prefill_uses_final_prompt_length(self) -> None:
        first = self.plan(
            state="ChunkedPrefill",
            query_ends=[2048],
            seq_lens=[2048],
            prompt_lens=[8320],
        )
        middle = self.plan(
            state="ChunkedPrefill",
            query_ends=[2048],
            seq_lens=[4096],
            prompt_lens=[8320],
        )
        final_tail = self.plan(
            state="ChunkedPrefill",
            query_ends=[128],
            seq_lens=[8320],
            prompt_lens=[8320],
        )
        self.assertFalse(first.direct)
        self.assertEqual(
            first.primary_reason,
            FallbackReason.BELOW_SPLIT_MIN_SAVED_QK,
        )
        self.assertTrue(middle.direct)
        self.assertFalse(final_tail.direct)
        self.assertEqual(
            final_tail.primary_reason,
            FallbackReason.NO_SPARSE_MIDDLE,
        )

    def test_prefix_cache_state_is_plannable(self) -> None:
        plan = self.plan(
            state="PrefillCacheHit",
            query_ends=[2048],
            seq_lens=[8320],
            prompt_lens=[8320],
        )
        self.assertTrue(plan.direct)
        self.assertEqual(plan.requests[0].q_begin, 6272)
        self.assertEqual(plan.requests[0].q_end, 8320)

    def test_decode_and_mixed_batches_always_fallback(self) -> None:
        decode = self.plan(
            state="DecodeOnly",
            query_ends=[1],
            seq_lens=[4096],
            prompt_lens=[4096],
            num_decodes=1,
            num_prefills=0,
        )
        mixed = self.plan(
            state="ChunkedPrefill",
            query_ends=[1, 2049],
            seq_lens=[4096, 4096],
            prompt_lens=[4096, 8320],
            num_decodes=1,
            num_prefills=1,
        )
        self.assertEqual(
            decode.primary_reason,
            FallbackReason.STATE_UNSUPPORTED,
        )
        self.assertEqual(
            mixed.primary_reason,
            FallbackReason.MIXED_DECODE,
        )

    def test_batch_query_lengths_are_cumulative_differences(self) -> None:
        plan = self.plan(
            query_ends=[2048, 3072],
            seq_lens=[4096, 7168],
            prompt_lens=[8320, 8192],
            num_prefills=2,
        )
        self.assertEqual(
            [request.query_len for request in plan.requests],
            [2048, 1024],
        )
        self.assertTrue(
            all(
                request.reason is FallbackReason.BATCH_UNSUPPORTED
                for request in plan.requests
            )
        )

    def test_missing_prompt_lengths_fail_closed_to_fia(self) -> None:
        plan = self.plan(
            query_ends=[4096],
            seq_lens=[4096],
            prompt_lens=None,
        )
        self.assertFalse(plan.direct)
        self.assertEqual(
            plan.primary_reason,
            FallbackReason.MISSING_METADATA,
        )

    def test_non_abi_geometry_has_specific_fallback(self) -> None:
        plan = build_batch_plan(
            state_name="PrefillNoCache",
            cumulative_query_ends=[4096],
            seq_lens=[4096],
            prompt_lens=[4096],
            num_decodes=0,
            num_prefills=1,
            config=config(local_window=1024),
        )
        self.assertFalse(plan.direct)
        self.assertEqual(
            plan.primary_reason,
            FallbackReason.GEOMETRY_UNSUPPORTED,
        )


class StatsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stats = TriangleMixRuntimeStats()
        self.stats.configure(4)

    def test_plan_is_counted_once_across_layers(self) -> None:
        plan = build_batch_plan(
            state_name="PrefillNoCache",
            cumulative_query_ends=[4096],
            seq_lens=[4096],
            prompt_lens=[4096],
            num_decodes=0,
            num_prefills=1,
            config=config(),
        )
        self.stats.record_planner_decision(plan)
        self.stats.record_planner_decision(plan)
        snapshot = self.stats.snapshot()
        self.assertEqual(snapshot["counters"]["request_total"], 1)
        self.assertEqual(
            snapshot["counters"]["request_planner_eligible"],
            1,
        )
        self.assertEqual(
            snapshot["performance"]["planner_eligibility_rate"],
            1.0,
        )

    def test_runtime_outcome_cannot_override_planner_result(self) -> None:
        plan = build_batch_plan(
            state_name="PrefillNoCache",
            cumulative_query_ends=[4096],
            seq_lens=[4096],
            prompt_lens=[4096],
            num_decodes=0,
            num_prefills=1,
            config=config(),
        )
        self.stats.record_request_decision(
            plan,
            direct=False,
            reason=FallbackReason.STATIC_CAPABILITY,
        )
        snapshot = self.stats.snapshot()
        self.assertEqual(
            snapshot["counters"]["request_planner_eligible"],
            1,
        )
        self.assertEqual(snapshot["fallback_reasons"], {})
        self.assertEqual(
            snapshot["recent"][0]["requests"][0]["reason"],
            "none",
        )

    def test_layer_and_host_enqueue_performance_counters(self) -> None:
        self.stats.record_layer_dispatch(
            plan_id=1,
            layer_index=7,
            saved_qk=2_000_000,
            host_enqueue_ns=31_000,
        )
        self.stats.record_layer_dispatch(
            plan_id=1,
            layer_index=7,
            saved_qk=2_000_000,
            host_enqueue_ns=41_000,
        )
        self.stats.record_layer_fallback(
            layer_index=9,
            reason=FallbackReason.GRAPH_CAPTURE,
        )
        snapshot = self.stats.snapshot()
        self.assertEqual(snapshot["counters"]["single_launch"], 2)
        self.assertEqual(snapshot["layers"]["direct"]["7"], 2)
        self.assertEqual(snapshot["layers"]["fia"]["9"], 1)
        self.assertEqual(
            snapshot["performance"]["host_enqueue_us_per_launch"],
            36.0,
        )
        self.assertEqual(
            snapshot["performance"]["host_enqueue_us_max"],
            41.0,
        )
        self.assertEqual(
            snapshot["performance"]["host_enqueue_timing_scope"],
            "host_wall_time_not_device_latency",
        )

    def test_recent_links_actual_layers_to_plan_and_request(self) -> None:
        plan = build_batch_plan(
            state_name="PrefillNoCache",
            cumulative_query_ends=[4096],
            seq_lens=[4096],
            prompt_lens=[4096],
            num_decodes=0,
            num_prefills=1,
            config=config(),
        )
        self.stats.record_planner_decision(plan)
        self.stats.record_layer_dispatch(
            plan_id=plan.plan_id,
            layer_index=3,
            request_index=0,
            saved_qk=2_000_000,
            host_enqueue_ns=31_000,
        )
        self.stats.record_layer_fallback(
            plan_id=plan.plan_id,
            layer_index=5,
            reason=FallbackReason.ADAPTER_UNAVAILABLE,
            request_indices=(0,),
        )

        recent = self.stats.snapshot()["recent"][0]
        execution = recent["execution"]
        self.assertEqual(execution["observed_route"], "mixed")
        self.assertEqual(execution["layer_direct"], 1)
        self.assertEqual(execution["layer_fia"], 1)
        self.assertEqual(execution["layer_hit_rate"], 0.5)
        self.assertEqual(
            execution["fallback_reasons"],
            {"adapter_unavailable": 1},
        )
        self.assertEqual(execution["single_launch"], 1)
        self.assertEqual(
            execution["estimated_saved_qk"],
            2_000_000,
        )
        self.assertEqual(
            execution["host_enqueue_ns_total"],
            31_000,
        )
        self.assertEqual(
            execution["timing_scope"],
            "host_enqueue_not_device_latency",
        )
        self.assertEqual(
            recent["requests"][0]["execution"],
            execution,
        )
        self.assertEqual(
            [
                (
                    event["layer_index"],
                    event["route"],
                    event["reason"],
                    event["request_slots"],
                )
                for event in recent["layer_events"]
            ],
            [
                (3, "direct", "none", [0]),
                (5, "fia", "adapter_unavailable", [0]),
            ],
        )

    def test_recent_updates_are_thread_safe(self) -> None:
        plan = build_batch_plan(
            state_name="PrefillNoCache",
            cumulative_query_ends=[4096],
            seq_lens=[4096],
            prompt_lens=[4096],
            num_decodes=0,
            num_prefills=1,
            config=config(),
        )
        self.stats.record_planner_decision(plan)

        def record(index: int) -> None:
            if index % 2 == 0:
                self.stats.record_layer_dispatch(
                    plan_id=plan.plan_id,
                    layer_index=index % 4,
                    request_index=0,
                    saved_qk=100,
                    host_enqueue_ns=index + 1,
                )
            else:
                self.stats.record_layer_fallback(
                    plan_id=plan.plan_id,
                    layer_index=index % 4,
                    reason=FallbackReason.STATIC_CAPABILITY,
                    request_indices=(0,),
                )

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(record, range(64)))

        snapshot = self.stats.snapshot()
        execution = snapshot["recent"][0]["execution"]
        self.assertEqual(snapshot["counters"]["layer_direct"], 32)
        self.assertEqual(snapshot["counters"]["layer_fia"], 32)
        self.assertEqual(execution["layer_direct"], 32)
        self.assertEqual(execution["layer_fia"], 32)
        self.assertEqual(execution["single_launch"], 32)
        self.assertEqual(execution["estimated_saved_qk"], 3_200)
        self.assertEqual(execution["host_enqueue_ns_total"], 1_024)
        self.assertEqual(
            execution["fallback_reasons"],
            {"static_capability": 32},
        )
        self.assertEqual(
            len(snapshot["recent"][0]["layer_events"]),
            64,
        )

    def test_recent_capacity_eviction_and_snapshot_isolation(
        self,
    ) -> None:
        self.stats.configure(2)
        plans = []
        for _ in range(3):
            plan = build_batch_plan(
                state_name="PrefillNoCache",
                cumulative_query_ends=[4096],
                seq_lens=[4096],
                prompt_lens=[4096],
                num_decodes=0,
                num_prefills=1,
                config=config(),
            )
            plans.append(plan)
            self.stats.record_planner_decision(plan)
            self.stats.record_layer_dispatch(
                plan_id=plan.plan_id,
                layer_index=3,
                request_index=0,
                saved_qk=100,
                host_enqueue_ns=10,
            )

        snapshot = self.stats.snapshot()
        self.assertEqual(
            [entry["plan_id"] for entry in snapshot["recent"]],
            [plans[1].plan_id, plans[2].plan_id],
        )
        snapshot["recent"][0]["execution"]["layer_direct"] = 999
        self.assertEqual(
            self.stats.snapshot()["recent"][0]["execution"][
                "layer_direct"
            ],
            1,
        )

        self.stats.record_layer_fallback(
            plan_id=plans[0].plan_id,
            layer_index=5,
            reason=FallbackReason.STATIC_CAPABILITY,
            request_indices=(0,),
        )
        self.assertEqual(
            [entry["plan_id"] for entry in self.stats.snapshot()["recent"]],
            [plans[1].plan_id, plans[2].plan_id],
        )
        self.stats.configure(1)
        self.assertEqual(
            [entry["plan_id"] for entry in self.stats.snapshot()["recent"]],
            [plans[2].plan_id],
        )
        self.stats.configure(0)
        self.assertEqual(self.stats.snapshot()["recent"], [])

    def test_per_plan_layer_event_sample_is_bounded(self) -> None:
        plan = build_batch_plan(
            state_name="PrefillNoCache",
            cumulative_query_ends=[4096],
            seq_lens=[4096],
            prompt_lens=[4096],
            num_decodes=0,
            num_prefills=1,
            config=config(),
        )
        self.stats.record_planner_decision(plan)
        for layer_index in range(300):
            self.stats.record_layer_dispatch(
                plan_id=plan.plan_id,
                layer_index=layer_index,
                request_index=0,
                saved_qk=100,
                host_enqueue_ns=10,
            )

        snapshot = self.stats.snapshot()
        recent = snapshot["recent"][0]
        self.assertEqual(
            len(recent["layer_events"]),
            self.stats._LAYER_EVENT_CAPACITY,
        )
        self.assertEqual(recent["execution"]["layer_direct"], 300)
        self.assertEqual(
            recent["execution"]["layer_events_dropped"],
            44,
        )
        self.assertEqual(
            recent["requests"][0]["execution"][
                "layer_events_dropped"
            ],
            44,
        )
        self.assertEqual(
            snapshot["counters"]["recent_layer_events_dropped"],
            44,
        )

    def test_reset_clears_every_counter_family(self) -> None:
        plan = build_batch_plan(
            state_name="PrefillNoCache",
            cumulative_query_ends=[4096],
            seq_lens=[4096],
            prompt_lens=[4096],
            num_decodes=0,
            num_prefills=1,
            config=config(),
        )
        self.stats.record_planner_decision(plan)
        self.stats.record_layer_dispatch(
            plan_id=plan.plan_id,
            layer_index=3,
            saved_qk=100,
            host_enqueue_ns=10,
        )
        self.stats.record_runtime_error(stage="dispatch")
        reset_snapshot = self.stats.snapshot(reset=True)
        self.assertEqual(reset_snapshot["counters"]["runtime_error"], 1)
        self.assertEqual(len(reset_snapshot["recent"]), 1)
        after = self.stats.snapshot()
        self.assertEqual(after["counters"], {})
        self.assertEqual(after["fallback_reasons"], {})
        self.assertEqual(after["layers"], {"direct": {}, "fia": {}})
        self.assertEqual(after["recent"], [])

    def test_missing_batch_metadata_is_not_a_phantom_request(self) -> None:
        plan = build_batch_plan(
            state_name="ChunkedPrefill",
            cumulative_query_ends=None,
            seq_lens=None,
            prompt_lens=None,
            num_decodes=0,
            num_prefills=0,
            config=config(),
        )
        self.stats.record_planner_decision(plan)
        snapshot = self.stats.snapshot()
        self.assertEqual(snapshot["counters"]["request_total"], 0)
        self.assertEqual(
            snapshot["counters"].get(
                "request_planner_ineligible",
                0,
            ),
            0,
        )
        self.assertEqual(snapshot["fallback_reasons"], {})
        self.assertEqual(snapshot["recent"][0]["requests"], [])

    def test_structured_log_interval_is_compact_and_worker_local(self) -> None:
        self.stats.configure(4, log_interval=2)
        payload = None
        for _ in range(2):
            plan = build_batch_plan(
                state_name="PrefillNoCache",
                cumulative_query_ends=[4096],
                seq_lens=[4096],
                prompt_lens=[4096],
                num_decodes=0,
                num_prefills=1,
                config=config(),
            )
            self.stats.record_planner_decision(plan)
            candidate = self.stats.structured_log_if_due()
            if candidate is not None:
                payload = candidate
        self.assertIsNotNone(payload)
        self.assertEqual(payload["event"], "trianglemix_runtime_stats")
        self.assertEqual(payload["scope"], "worker_local")
        self.assertEqual(payload["request_boundary"], 2)
        self.assertEqual(
            payload["stats"]["counters"]["request_total"],
            2,
        )
        self.assertNotIn("recent", payload["stats"])
        self.assertIsNone(self.stats.structured_log_if_due())

    def test_structured_log_is_independent_of_recent_retention(self) -> None:
        self.stats.configure(0, log_interval=1)
        plan = build_batch_plan(
            state_name="PrefillNoCache",
            cumulative_query_ends=[4096],
            seq_lens=[4096],
            prompt_lens=[4096],
            num_decodes=0,
            num_prefills=1,
            config=config(),
        )
        self.stats.record_planner_decision(plan)
        self.stats.record_layer_dispatch(
            plan_id=plan.plan_id,
            layer_index=3,
            request_index=0,
            saved_qk=2_000_000,
            host_enqueue_ns=31_000,
        )

        payload = self.stats.structured_log_if_due()
        self.assertIsNotNone(payload)
        self.assertEqual(
            [delta["kind"] for delta in payload["recent_delta"]],
            ["planner", "layer_execution"],
        )
        self.assertEqual(payload["recent_delta"][1]["request_slots"], [0])
        self.assertEqual(self.stats.snapshot()["recent"], [])

        serialized = repr(payload["recent_delta"])
        for forbidden in (
            "prompt_len",
            "seq_len",
            "q_begin",
            "q_end",
            "token_id",
            "token_ids",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_boundary_plan_layers_emit_supplemental_deltas(self) -> None:
        self.stats.configure(4, log_interval=1)
        plan = build_batch_plan(
            state_name="PrefillNoCache",
            cumulative_query_ends=[4096],
            seq_lens=[4096],
            prompt_lens=[4096],
            num_decodes=0,
            num_prefills=1,
            config=config(),
        )
        self.stats.record_planner_decision(plan)

        boundary = self.stats.structured_log_if_due()
        self.assertIsNotNone(boundary)
        self.assertEqual(boundary["request_boundary"], 1)
        self.assertEqual(
            [delta["kind"] for delta in boundary["recent_delta"]],
            ["planner"],
        )

        self.stats.record_layer_dispatch(
            plan_id=plan.plan_id,
            layer_index=3,
            request_index=0,
            saved_qk=2_000_000,
            host_enqueue_ns=31_000,
        )
        direct = self.stats.structured_log_if_due()
        self.assertIsNotNone(direct)
        self.assertEqual(direct["request_boundary"], 1)
        self.assertEqual(
            [delta["kind"] for delta in direct["recent_delta"]],
            ["layer_execution"],
        )
        self.assertEqual(direct["recent_delta"][0]["route"], "direct")

        self.stats.record_layer_fallback(
            plan_id=plan.plan_id,
            layer_index=5,
            reason=FallbackReason.ADAPTER_UNAVAILABLE,
            request_indices=(0,),
        )
        fallback = self.stats.structured_log_if_due()
        self.assertIsNotNone(fallback)
        self.assertEqual(fallback["request_boundary"], 1)
        self.assertEqual(
            fallback["recent_delta"][0]["reason"],
            "adapter_unavailable",
        )
        self.assertEqual(
            fallback["delta_mode"],
            "incremental_not_final_plan_snapshot",
        )

    def test_structured_delta_capacity_and_reset(self) -> None:
        self.stats.configure(0, log_interval=1)
        plan = build_batch_plan(
            state_name="PrefillNoCache",
            cumulative_query_ends=[4096],
            seq_lens=[4096],
            prompt_lens=[4096],
            num_decodes=0,
            num_prefills=1,
            config=config(),
        )
        with patch.object(
            TriangleMixRuntimeStats,
            "_LOG_DELTA_CAPACITY",
            2,
        ):
            self.stats.record_planner_decision(plan)
            self.stats.record_layer_dispatch(
                plan_id=plan.plan_id,
                layer_index=3,
                request_index=0,
                saved_qk=2_000_000,
                host_enqueue_ns=31_000,
            )
            self.stats.record_layer_fallback(
                plan_id=plan.plan_id,
                layer_index=5,
                reason=FallbackReason.ADAPTER_UNAVAILABLE,
                request_indices=(0,),
            )
            payload = self.stats.structured_log_if_due()

        self.assertIsNotNone(payload)
        self.assertEqual(len(payload["recent_delta"]), 2)
        self.assertTrue(
            all(
                delta["kind"] == "layer_execution"
                for delta in payload["recent_delta"]
            )
        )
        self.assertEqual(
            payload["stats"]["counters"][
                "structured_log_deltas_dropped"
            ],
            1,
        )

        second = build_batch_plan(
            state_name="PrefillNoCache",
            cumulative_query_ends=[4096],
            seq_lens=[4096],
            prompt_lens=[4096],
            num_decodes=0,
            num_prefills=1,
            config=config(),
        )
        self.stats.record_planner_decision(second)
        reset_snapshot = self.stats.snapshot(reset=True)
        self.assertEqual(
            reset_snapshot["counters"]["request_total"],
            2,
        )
        self.assertIsNone(self.stats.structured_log_if_due())
        self.assertEqual(self.stats.snapshot()["counters"], {})


if __name__ == "__main__":
    unittest.main()
