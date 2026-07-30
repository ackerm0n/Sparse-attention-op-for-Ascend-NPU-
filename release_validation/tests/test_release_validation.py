from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from release_validation.counters import (
    aggregate_worker_events,
    extract_stats_events,
)
from release_validation.model_smoke import evaluate_mode
from release_validation.model_smoke_worker import END_MARKER, START_MARKER
from release_validation.run import main as unified_main
from release_validation.ttft_abba import ORDER, summarize_abba


ROOT = Path(__file__).resolve().parents[2]


def stats_line(
    *,
    request_boundary: int,
    counters: dict[str, int],
    fallback_reasons: dict[str, int] | None = None,
    pid: int = 41,
) -> str:
    reasons = dict(fallback_reasons or {})
    normalized_counters = dict(counters)
    request_total = normalized_counters.get("request_total", 0)
    request_planner_ineligible = sum(reasons.values())
    normalized_counters.setdefault(
        "request_planner_ineligible",
        request_planner_ineligible,
    )
    normalized_counters.setdefault(
        "request_planner_eligible",
        request_total - request_planner_ineligible,
    )
    normalized_counters.setdefault(
        "layer_direct",
        normalized_counters.get("single_launch", 0),
    )
    normalized_counters.setdefault(
        "layer_fia",
        sum(
            value
            for name, value in normalized_counters.items()
            if name.startswith("layer_fia_reason:")
        ),
    )
    direct_layers = (
        {"selected": normalized_counters["layer_direct"]}
        if normalized_counters["layer_direct"]
        else {}
    )
    fia_layers = (
        {"official": normalized_counters["layer_fia"]}
        if normalized_counters["layer_fia"]
        else {}
    )
    value = {
        "event": "trianglemix_runtime_stats",
        "scope": "worker_local",
        "worker": {
            "pid": pid,
            "rank": "0",
            "local_rank": "0",
        },
        "request_boundary": request_boundary,
        "stats": {
            "counters": normalized_counters,
            "fallback_reasons": reasons,
            "layers": {"direct": direct_layers, "fia": fia_layers},
            "performance": {},
        },
    }
    return "INFO worker payload " + json.dumps(value, sort_keys=True)


def marker(prefix: str, name: str) -> str:
    return prefix + json.dumps({"name": name}, sort_keys=True)


class CounterParserTests(unittest.TestCase):
    def test_mixed_log_parser_and_latest_snapshot_aggregation(self) -> None:
        text = "\n".join(
            (
                "unrelated log",
                stats_line(
                    request_boundary=1,
                    counters={"request_total": 1, "single_launch": 2},
                ),
                stats_line(
                    request_boundary=2,
                    counters={"request_total": 2, "single_launch": 5},
                ),
            )
        )
        events = extract_stats_events(text, source="worker.log")
        self.assertEqual(len(events), 2)
        report = aggregate_worker_events(events)
        self.assertEqual(report["worker_count"], 1)
        self.assertEqual(
            report["aggregate"]["counters"]["request_total"],
            2,
        )
        self.assertEqual(
            report["aggregate"]["counters"]["single_launch"],
            5,
        )
        self.assertEqual(report["counter_regressions"], [])

    def test_counter_regression_is_reported(self) -> None:
        text = "\n".join(
            (
                stats_line(
                    request_boundary=2,
                    counters={"request_total": 2},
                ),
                stats_line(
                    request_boundary=1,
                    counters={"request_total": 1},
                ),
            )
        )
        report = aggregate_worker_events(
            extract_stats_events(text, source="worker.log")
        )
        self.assertIn(
            "request_total",
            {
                item["counter"]
                for item in report["counter_regressions"]
            },
        )


class ModelSmokeContractTests(unittest.TestCase):
    @staticmethod
    def _case(
        name: str,
        *,
        prompts: list[int],
        max_tokens: int = 1,
        cached: int | None = None,
        request_count: int = 1,
        prompt_hash: str | None = None,
    ) -> dict[str, object]:
        return {
            "name": name,
            "status": "PASS",
            "prompt_lengths": prompts,
            "request_count": request_count,
            "max_tokens": max_tokens,
            "prompt_token_ids_sha256": [
                prompt_hash or f"prompt-{index}"
                for index in range(request_count)
            ],
            "completions": [
                {
                    "output_tokens": max_tokens,
                    "cached_tokens": cached,
                }
                for _ in range(request_count)
            ],
        }

    def test_eager_contract_uses_markers_and_decode_counter_delta(self) -> None:
        log = "\n".join(
            (
                stats_line(
                    request_boundary=1,
                    counters={"request_total": 1, "single_launch": 0},
                ),
                marker(START_MARKER, "chunked_prefill"),
                stats_line(
                    request_boundary=2,
                    counters={"request_total": 2, "single_launch": 24},
                ),
                marker(END_MARKER, "chunked_prefill"),
                marker(START_MARKER, "batch2_fallback"),
                stats_line(
                    request_boundary=3,
                    counters={
                        "request_total": 3,
                        "single_launch": 24,
                        "layer_fia_reason:batch_unsupported": 1,
                    },
                    fallback_reasons={"batch_unsupported": 1},
                ),
                stats_line(
                    request_boundary=4,
                    counters={
                        "request_total": 4,
                        "single_launch": 24,
                        "layer_fia_reason:batch_unsupported": 1,
                    },
                    fallback_reasons={"batch_unsupported": 2},
                ),
                marker(END_MARKER, "batch2_fallback"),
                marker(START_MARKER, "sustained_decode"),
                stats_line(
                    request_boundary=5,
                    counters={
                        "request_total": 5,
                        "single_launch": 48,
                        "layer_fia_reason:batch_unsupported": 1,
                    },
                    fallback_reasons={"batch_unsupported": 2},
                ),
                stats_line(
                    request_boundary=6,
                    counters={
                        "request_total": 6,
                        "single_launch": 48,
                        "layer_fia_reason:batch_unsupported": 1,
                        "layer_fia_reason:state_unsupported": 1,
                    },
                    fallback_reasons={
                        "batch_unsupported": 2,
                        "state_unsupported": 1,
                    },
                ),
                stats_line(
                    request_boundary=7,
                    counters={
                        "request_total": 7,
                        "single_launch": 48,
                        "layer_fia_reason:batch_unsupported": 1,
                        "layer_fia_reason:state_unsupported": 2,
                    },
                    fallback_reasons={
                        "batch_unsupported": 2,
                        "state_unsupported": 2,
                    },
                ),
                stats_line(
                    request_boundary=8,
                    counters={
                        "request_total": 8,
                        "single_launch": 48,
                        "layer_fia_reason:batch_unsupported": 1,
                        "layer_fia_reason:state_unsupported": 3,
                    },
                    fallback_reasons={
                        "batch_unsupported": 2,
                        "state_unsupported": 3,
                    },
                ),
                marker(END_MARKER, "sustained_decode"),
            )
        )
        report = {
            "status": "PASS",
            "cases": [
                self._case("chunked_prefill", prompts=[8320]),
                self._case(
                    "prefix_cache_seed",
                    prompts=[4352],
                    cached=0,
                    prompt_hash="same-full-prompt",
                ),
                self._case(
                    "prefix_cache_repeat",
                    prompts=[4352],
                    cached=4224,
                    prompt_hash="same-full-prompt",
                ),
                self._case(
                    "prefix_cache_shared",
                    prompts=[4352],
                    cached=4096,
                    prompt_hash="shared-prefix-other-suffix",
                ),
                self._case(
                    "batch2_fallback",
                    prompts=[768, 768],
                    request_count=2,
                ),
                self._case(
                    "sustained_decode",
                    prompts=[8193],
                    max_tokens=3,
                ),
            ],
            "prefix_cache_evidence": {
                "block_size": 128,
                "prefix_cache_reset_succeeded": True,
                "shared_prefix_tokens": 4096,
                "expected_repeat_cached_tokens": 4224,
                "expected_shared_cached_tokens": 4096,
                "seed_prefix_block_hashes": ["same-prefix-block"] * 32,
                "shared_request_prefix_block_hashes": (
                    ["same-prefix-block"] * 32
                ),
                "seed_suffix_block_hashes": ["seed-a", "seed-b"],
                "shared_request_suffix_block_hashes": [
                    "shared-a",
                    "shared-b",
                ],
                "suffixes_are_distinct": True,
                "shared_hit_is_partial": True,
                "shared_prefix_blocks": 32,
                "evidence_scope": "synthetic scheduler contract",
            },
        }
        evaluation = evaluate_mode(
            "eager",
            report,
            log,
            chunk_budget=2048,
        )
        self.assertEqual(evaluation["status"], "PASS")
        self.assertTrue(
            evaluation["checks"]["decode_remained_official"]["passed"]
        )

    def test_graph_contract_requires_explicit_graph_reason(self) -> None:
        log = "\n".join(
            (
                marker(START_MARKER, "graph_capture_replay"),
                stats_line(
                    request_boundary=1,
                    counters={
                        "request_total": 1,
                        "layer_fia_reason:graph_capture": 24,
                    },
                    fallback_reasons={"graph_capture": 1},
                ),
                marker(END_MARKER, "graph_capture_replay"),
            )
        )
        report = {
            "status": "PASS",
            "cases": [
                self._case("graph_capture_replay", prompts=[2048], max_tokens=16)
            ],
        }
        evaluation = evaluate_mode(
            "graph",
            report,
            log,
            chunk_budget=2048,
        )
        self.assertEqual(evaluation["status"], "PASS")

    def test_graph_contract_rejects_any_custom_launch(self) -> None:
        log = "\n".join(
            (
                marker(START_MARKER, "graph_capture_replay"),
                stats_line(
                    request_boundary=1,
                    counters={
                        "request_total": 1,
                        "single_launch": 1,
                        "layer_fia_reason:graph_capture": 24,
                    },
                    fallback_reasons={"graph_capture": 1},
                ),
                marker(END_MARKER, "graph_capture_replay"),
            )
        )
        report = {
            "status": "PASS",
            "cases": [
                self._case(
                    "graph_capture_replay",
                    prompts=[2048],
                    max_tokens=16,
                )
            ],
        }
        evaluation = evaluate_mode(
            "graph",
            report,
            log,
            chunk_budget=2048,
        )
        self.assertEqual(evaluation["status"], "FAIL")
        self.assertFalse(
            evaluation["checks"][
                "graph_capture_bypassed_custom_kernel"
            ]["passed"]
        )


class AbbaContractTests(unittest.TestCase):
    def test_position_matched_end_to_end_summary(self) -> None:
        runs: list[dict[str, object]] = []
        for cycle in range(4):
            for position, variant in enumerate(ORDER):
                latency = 1.0 if variant == "dense" else 0.8
                runs.append(
                    {
                        "variant": variant,
                        "result": {
                            "runs": [
                                {
                                    "run": index + 1,
                                    "prompt": f"p{index}",
                                    "prompt_tokens": 8192 + index,
                                    "latency_s": latency,
                                }
                                for index in range(2)
                            ]
                        },
                    }
                )
        summary = summarize_abba(
            runs,
            cycles=4,
            bootstrap_samples=100,
            seed=7,
        )
        self.assertEqual(summary["measurement"], "end_to_end_ttft")
        self.assertFalse(summary["attention_microbenchmark_used"])
        self.assertAlmostEqual(summary["overall_gain_percent"], 20.0)
        self.assertGreater(
            summary["bootstrap"]["gain_fraction_p2_5"],
            0,
        )


class UnifiedEntryTests(unittest.TestCase):
    def test_parse_counters_command_is_locally_executable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "worker.log"
            output = root / "counters.json"
            log.write_text(
                stats_line(
                    request_boundary=1,
                    counters={"request_total": 1},
                )
                + "\n",
                encoding="utf-8",
            )
            code = unified_main(
                [
                    "parse-counters",
                    str(log),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(code, 0)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))["status"],
                "PASS",
            )

    def test_validation_sources_have_no_private_path_defaults(self) -> None:
        forbidden = ("/mnt/", "/Users/", "siyuan.tong")
        for path in (ROOT / "release_validation").glob("*.py"):
            source = path.read_text(encoding="utf-8")
            for marker_value in forbidden:
                self.assertNotIn(marker_value, source, str(path))

    def test_installed_correctness_has_no_wrapper_or_adapter_cli(self) -> None:
        source = (
            ROOT
            / "release_validation"
            / "installed_wheel_correctness.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('parser.add_argument("--wrapper"', source)
        self.assertNotIn('parser.add_argument("--adapter"', source)
        self.assertIn("ensure_native_loaded", source)
        self.assertIn("triangle_direct_paged_attention", source)


if __name__ == "__main__":
    unittest.main()
