from __future__ import annotations

import asyncio
import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path, PurePosixPath

from release_validation.counters import (
    aggregate_worker_events,
    extract_stats_events,
)
from release_validation.installed_artifact import audit_installed_wheel
from release_validation.installed_wheel_crossover import (
    _is_exact_abba,
    _load_correctness_prerequisite,
    _normalize_attention_record,
    build_route_cases,
)
from release_validation.model_smoke import (
    _decode_official_gate,
    _marker_analysis,
    evaluate_mode,
)
from release_validation.model_smoke_worker import (
    END_MARKER,
    START_MARKER,
    _expected_repeat_cached_tokens,
    _run_paused_request_batch,
)
from release_validation.run import main as unified_main


def _marker(prefix: str, name: str) -> str:
    return prefix + json.dumps({"name": name}, sort_keys=True)


def _concurrency_stats_line(
    *,
    request_total: int,
    batch_fallback_total: int,
    single_launch_total: int = 0,
    pid: int = 73,
) -> str:
    counters = {
        "request_total": request_total,
        "request_planner_eligible": (
            request_total - batch_fallback_total
        ),
        "request_planner_ineligible": batch_fallback_total,
        "layer_direct": single_launch_total,
        "single_launch": single_launch_total,
        "layer_fia": batch_fallback_total,
    }
    fallback_reasons: dict[str, int] = {}
    fia_layers: dict[str, int] = {}
    if batch_fallback_total:
        counters["layer_fia_reason:batch_unsupported"] = (
            batch_fallback_total
        )
        fallback_reasons["batch_unsupported"] = batch_fallback_total
        fia_layers["selected"] = batch_fallback_total
    value = {
        "event": "trianglemix_runtime_stats",
        "scope": "worker_local",
        "worker": {"pid": pid, "rank": "0", "local_rank": "0"},
        "request_boundary": request_total,
        "stats": {
            "counters": counters,
            "fallback_reasons": fallback_reasons,
            "layers": {
                "direct": (
                    {"selected": single_launch_total}
                    if single_launch_total
                    else {}
                ),
                "fia": fia_layers,
            },
            "performance": {},
        },
    }
    return "INFO " + json.dumps(value, sort_keys=True)


def _decode_stats_line(
    *,
    request_total: int,
    state_fallback_total: int,
    layer_fia_total: int,
    pid: int = 91,
) -> str:
    value = {
        "event": "trianglemix_runtime_stats",
        "scope": "worker_local",
        "worker": {"pid": pid, "rank": "0", "local_rank": "0"},
        "request_boundary": request_total,
        "stats": {
            "counters": {
                "request_total": request_total,
                "request_planner_eligible": (
                    request_total - state_fallback_total
                ),
                "request_planner_ineligible": state_fallback_total,
                "layer_direct": 0,
                "single_launch": 0,
                "layer_fia": layer_fia_total,
                "layer_fia_reason:state_unsupported": layer_fia_total,
            },
            "fallback_reasons": (
                {"state_unsupported": state_fallback_total}
                if state_fallback_total
                else {}
            ),
            "layers": {
                "direct": {},
                "fia": (
                    {"selected": layer_fia_total}
                    if layer_fia_total
                    else {}
                ),
            },
            "performance": {},
        },
    }
    return "INFO " + json.dumps(value, sort_keys=True)


def _concurrency_fixture(
    *,
    split_level: int | None = None,
    late_launch_level: int | None = None,
) -> tuple[dict[str, object], str]:
    levels = [1, 2, 4, 8, 16]
    lines: list[str] = []
    request_total = 0
    fallback_total = 0
    launch_total = 0
    cases: list[dict[str, object]] = []
    for level in levels:
        name = f"concurrency_{level}"
        lines.append(_marker(START_MARKER, name))
        if split_level == level:
            first = max(1, level // 2)
            lines.append(
                _concurrency_stats_line(
                    request_total=request_total + first,
                    batch_fallback_total=(
                        fallback_total + first if level > 1 else 0
                    ),
                    single_launch_total=launch_total,
                )
            )
        request_total += level
        if level > 1:
            fallback_total += level
        lines.append(
            _concurrency_stats_line(
                request_total=request_total,
                batch_fallback_total=fallback_total,
                single_launch_total=launch_total,
            )
        )
        if late_launch_level == level:
            launch_total += 1
            lines.append(
                _concurrency_stats_line(
                    request_total=request_total,
                    batch_fallback_total=fallback_total,
                    single_launch_total=launch_total,
                )
            )
        lines.append(_marker(END_MARKER, name))
        cases.append(
            {
                "name": name,
                "status": "PASS",
                "true_async_tasks": level,
                "async_collector_tasks": level,
                "request_count": level,
                "admission": {
                    "mode": "keep",
                    "cache_cleared": False,
                    "paused_verified_after_all_adds": True,
                    "submitted_before_resume": level,
                    "async_collector_tasks": level,
                    "all_requests_submitted_before_resume": True,
                },
            }
        )
    return (
        {
            "status": "PASS",
            "concurrency_levels": levels,
            "cases": cases,
        },
        "\n".join(lines),
    )


class _FakeDistribution:
    def __init__(self, root: Path, files: list[str]) -> None:
        self._root = root
        self.files = [PurePosixPath(name) for name in files]
        self.metadata = {"Name": "vllm-ascend-trianglemix"}
        self.version = "1.2.3"

    def locate_file(self, name: object) -> Path:
        return self._root / Path(str(name))


def _write_test_wheel(root: Path, payload: bytes) -> Path:
    wheel = root / "candidate-1.2.3-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "vllm_ascend_trianglemix/__init__.py",
            payload,
        )
        archive.writestr(
            "candidate-1.2.3.dist-info/METADATA",
            (
                "Metadata-Version: 2.1\n"
                "Name: vllm-ascend-trianglemix\n"
                "Version: 1.2.3\n\n"
            ),
        )
    return wheel


class InstalledArtifactTests(unittest.TestCase):
    def test_exact_manifest_match_and_content_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "vllm_ascend_trianglemix"
            package.mkdir()
            installed = package / "__init__.py"
            installed.write_bytes(b"exact-payload\n")
            wheel = _write_test_wheel(root, b"exact-payload\n")
            distribution = _FakeDistribution(
                root,
                ["vllm_ascend_trianglemix/__init__.py"],
            )

            matching = audit_installed_wheel(
                wheel,
                distribution=distribution,
            )
            self.assertTrue(matching["passed"], matching["errors"])

            installed.write_bytes(b"tampered\n")
            mismatching = audit_installed_wheel(
                wheel,
                distribution=distribution,
            )
            self.assertFalse(mismatching["passed"])
            self.assertIn(
                "installed_payload_mismatch",
                {item["kind"] for item in mismatching["errors"]},
            )

    def test_unrecorded_installed_payload_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "vllm_ascend_trianglemix"
            package.mkdir()
            (package / "__init__.py").write_bytes(b"exact-payload\n")
            (package / "rogue.py").write_text("unexpected = True\n")
            wheel = _write_test_wheel(root, b"exact-payload\n")
            report = audit_installed_wheel(
                wheel,
                distribution=_FakeDistribution(
                    root,
                    ["vllm_ascend_trianglemix/__init__.py"],
                ),
            )
            self.assertFalse(report["passed"])
            self.assertIn(
                "installed_unrecorded_payload",
                {item["kind"] for item in report["errors"]},
            )


class CounterFailClosedTests(unittest.TestCase):
    def test_malformed_worker_snapshot_is_rejected(self) -> None:
        malformed = {
            "event": "trianglemix_runtime_stats",
            "scope": "global",
            "worker": {"pid": "not-an-integer"},
            "request_boundary": -1,
            "stats": {
                "counters": {"request_total": "one"},
                "fallback_reasons": {},
                "layers": {"direct": [], "fia": {}},
                "performance": {},
            },
        }
        report = aggregate_worker_events(
            extract_stats_events(json.dumps(malformed), source="bad.log")
        )
        kinds = {item["kind"] for item in report["schema_errors"]}
        self.assertIn("invalid_scope", kinds)
        self.assertIn("missing_worker_pid", kinds)
        self.assertIn("invalid_request_boundary", kinds)
        self.assertIn("invalid_counter_value", kinds)
        self.assertIn("invalid_layer_mapping", kinds)


class ConcurrencyGateTests(unittest.TestCase):
    class _Output:
        finished = True

    class _Collector:
        def __init__(self, events: list[str], request_id: str) -> None:
            self.events = events
            self.request_id = request_id

        def get_nowait(self) -> None:
            return None

        async def get(self) -> object:
            self.events.append(f"drain:{self.request_id}")
            await asyncio.sleep(0)
            return ConcurrencyGateTests._Output()

    class _Engine:
        def __init__(self) -> None:
            self.events: list[str] = []
            self.paused = False

        async def pause_generation(self, **_: object) -> None:
            self.events.append("pause")
            self.paused = True

        async def add_request(self, **values: object) -> object:
            request_id = str(values["request_id"])
            if not self.paused:
                raise AssertionError("request admitted while running")
            self.events.append(f"add:{request_id}")
            return ConcurrencyGateTests._Collector(
                self.events,
                request_id,
            )

        async def is_paused(self) -> bool:
            self.events.append("is_paused")
            return self.paused

        async def resume_generation(self) -> None:
            self.events.append("resume")
            self.paused = False

        async def abort(self, request_id: str) -> None:
            self.events.append(f"abort:{request_id}")

    def test_paused_admission_orders_all_adds_before_resume(self) -> None:
        engine = self._Engine()
        outputs, evidence = asyncio.run(
            _run_paused_request_batch(
                engine,
                [
                    {
                        "request_id": f"r{index}",
                        "prompt": {"prompt_token_ids": [index]},
                        "params": object(),
                    }
                    for index in range(4)
                ],
            )
        )
        self.assertEqual(len(outputs), 4)
        self.assertLess(
            max(
                engine.events.index(f"add:r{index}")
                for index in range(4)
            ),
            engine.events.index("resume"),
        )
        self.assertTrue(evidence["all_requests_submitted_before_resume"])

    def test_levels_1_2_4_8_16_are_one_scheduler_step(self) -> None:
        worker_report, log = _concurrency_fixture()
        evaluation = evaluate_mode(
            "concurrency",
            worker_report,
            log,
            chunk_budget=2048,
        )
        self.assertEqual(evaluation["status"], "PASS")
        self.assertEqual(
            [item["concurrency"] for item in evaluation["concurrency_matrix"]],
            [1, 2, 4, 8, 16],
        )
        self.assertTrue(
            all(
                item["status"] == "PASS"
                for item in evaluation["concurrency_matrix"]
            )
        )
        self.assertEqual(
            {
                item["crossover_status"]
                for item in evaluation["concurrency_matrix"]
                if item["concurrency"] > 1
            },
            {"not_applicable_abi_batch1"},
        )

    def test_split_scheduler_batch_fails_closed(self) -> None:
        worker_report, log = _concurrency_fixture(split_level=4)
        evaluation = evaluate_mode(
            "concurrency",
            worker_report,
            log,
            chunk_budget=2048,
        )
        self.assertEqual(evaluation["status"], "FAIL")
        level_four = next(
            item
            for item in evaluation["concurrency_matrix"]
            if item["concurrency"] == 4
        )
        self.assertEqual(level_four["status"], "FAIL")
        self.assertEqual(
            level_four["scheduler_gate"]["evidence"][0][
                "request_total_delta"
            ],
            2,
        )

    def test_late_layer_custom_launch_fails_closed(self) -> None:
        worker_report, log = _concurrency_fixture(late_launch_level=8)
        evaluation = evaluate_mode(
            "concurrency",
            worker_report,
            log,
            chunk_budget=2048,
        )
        level_eight = next(
            item
            for item in evaluation["concurrency_matrix"]
            if item["concurrency"] == 8
        )
        self.assertEqual(evaluation["status"], "FAIL")
        self.assertEqual(level_eight["status"], "FAIL")
        self.assertEqual(
            level_eight["scheduler_gate"]["evidence"][0][
                "segment_single_launch_delta"
            ],
            1,
        )

    def test_extra_worker_fails_tp1_contract(self) -> None:
        worker_report, log = _concurrency_fixture()
        lines = log.splitlines()
        first_end = next(
            index
            for index, line in enumerate(lines)
            if END_MARKER in line
        )
        lines.insert(
            first_end,
            _concurrency_stats_line(
                request_total=1,
                batch_fallback_total=0,
                pid=74,
            ),
        )
        evaluation = evaluate_mode(
            "concurrency",
            worker_report,
            "\n".join(lines),
            chunk_budget=2048,
        )
        self.assertEqual(evaluation["status"], "FAIL")
        self.assertFalse(
            evaluation["checks"]["exactly_one_worker_observed"]["passed"]
        )


class ModelSmokeFailClosedTests(unittest.TestCase):
    def test_malformed_pass_report_returns_structured_fail(self) -> None:
        evaluation = evaluate_mode(
            "eager",
            {
                "status": "PASS",
                "cases": [
                    {
                        "name": "chunked_prefill",
                        "status": "PASS",
                        "prompt_lengths": [],
                        "completions": [],
                    }
                ],
            },
            "",
            chunk_budget=2048,
        )
        self.assertEqual(evaluation["status"], "FAIL")
        self.assertIsInstance(evaluation["checks"], dict)

    def test_duplicate_and_unmatched_markers_fail(self) -> None:
        log = "\n".join(
            (
                _marker(START_MARKER, "graph_capture_replay"),
                _marker(START_MARKER, "graph_capture_replay"),
                _concurrency_stats_line(
                    request_total=1,
                    batch_fallback_total=0,
                ),
                _marker(END_MARKER, "graph_capture_replay"),
                _marker(END_MARKER, "graph_capture_replay"),
            )
        )
        evaluation = evaluate_mode(
            "graph",
            {
                "status": "PASS",
                "cases": [
                    {
                        "name": "graph_capture_replay",
                        "status": "PASS",
                    }
                ],
            },
            log,
            chunk_budget=2048,
        )
        self.assertEqual(evaluation["status"], "FAIL")
        self.assertFalse(
            evaluation["checks"]["scenario_marker_contract"]["passed"]
        )
        self.assertTrue(
            evaluation["checks"]["scenario_marker_contract"]["evidence"][
                "parser_errors"
            ]
        )

    def test_many_layer_events_do_not_count_as_decode_steps(self) -> None:
        lines = [
            _decode_stats_line(
                request_total=1,
                state_fallback_total=0,
                layer_fia_total=0,
            ),
            _marker(START_MARKER, "sustained_decode"),
        ]
        lines.extend(
            _decode_stats_line(
                request_total=2,
                state_fallback_total=1,
                layer_fia_total=24,
            )
            for _ in range(24)
        )
        lines.append(_marker(END_MARKER, "sustained_decode"))
        log = "\n".join(lines)
        ranges, errors, _ = _marker_analysis(log)
        self.assertEqual(errors, [])
        gate = _decode_official_gate(
            extract_stats_events(log, source="decode.log"),
            ranges,
            expected_decode_steps=2,
        )
        self.assertFalse(gate["passed"])
        self.assertEqual(
            gate["evidence"][0]["observed_decode_steps"],
            1,
        )
        self.assertEqual(
            gate["evidence"][0]["layer_log_event_count"],
            24,
        )


class PrefixCacheContractTests(unittest.TestCase):
    def test_aligned_repeat_recomputes_the_final_cache_block(self) -> None:
        self.assertEqual(_expected_repeat_cached_tokens(4352, 128), 4224)
        self.assertEqual(_expected_repeat_cached_tokens(4353, 128), 4352)

    def test_zero_self_reported_hits_cannot_pass(self) -> None:
        def case(
            name: str,
            *,
            cached: int,
            prompt_hash: str,
        ) -> dict[str, object]:
            return {
                "name": name,
                "status": "PASS",
                "prompt_lengths": [4352],
                "prompt_token_ids_sha256": [prompt_hash],
                "completions": [{"cached_tokens": cached}],
            }

        evaluation = evaluate_mode(
            "eager",
            {
                "status": "PASS",
                "cases": [
                    case(
                        "prefix_cache_seed",
                        cached=0,
                        prompt_hash="same-full-prompt",
                    ),
                    case(
                        "prefix_cache_repeat",
                        cached=0,
                        prompt_hash="same-full-prompt",
                    ),
                    case(
                        "prefix_cache_shared",
                        cached=0,
                        prompt_hash="other-full-prompt",
                    ),
                ],
                "prefix_cache_evidence": {
                    "block_size": 128,
                    "prefix_cache_reset_succeeded": True,
                    "shared_prefix_tokens": 4096,
                    "shared_prefix_blocks": 32,
                    "expected_repeat_cached_tokens": 0,
                    "expected_shared_cached_tokens": 0,
                    "seed_prefix_block_hashes": ["same"] * 32,
                    "shared_request_prefix_block_hashes": ["same"] * 32,
                    "seed_suffix_block_hashes": ["suffix-a"],
                    "shared_request_suffix_block_hashes": ["suffix-b"],
                    "suffixes_are_distinct": True,
                    "shared_hit_is_partial": True,
                },
            },
            "",
            chunk_budget=2048,
        )
        repeat = evaluation["checks"]["prefix_cache_repeat_hit"]
        shared = evaluation["checks"]["prefix_cache_shared_partial_hit"]
        self.assertFalse(repeat["passed"])
        self.assertFalse(shared["passed"])
        self.assertEqual(repeat["evidence"]["expected_cached_tokens"], 4224)
        self.assertEqual(
            shared["evidence"]["expected_shared_cached_tokens"],
            4096,
        )


class CrossoverContractTests(unittest.TestCase):
    class _Case:
        def __init__(self, **values: object) -> None:
            self.__dict__.update(values)

    def test_route_cases_cover_every_prompt_token_once(self) -> None:
        lengths = [5, 9]
        chunks = [4, 6]
        cases = build_route_cases(
            lengths=lengths,
            chunk_sizes=chunks,
            seed=11,
            wrapper_case_type=self._Case,
        )
        for prompt_len in lengths:
            for chunk in chunks:
                selected = [
                    case
                    for case in cases
                    if case.prompt_len == prompt_len
                    and f"_c{chunk}_" in case.name
                ]
                self.assertEqual(
                    [case.query_start for case in selected],
                    list(range(0, prompt_len, chunk)),
                )
                self.assertEqual(
                    sum(case.query_tokens for case in selected),
                    prompt_len,
                )
                self.assertEqual(
                    selected[-1].query_start + selected[-1].query_tokens,
                    prompt_len,
                )

    def test_exact_abba_order_contract(self) -> None:
        self.assertTrue(
            _is_exact_abba(
                [
                    "dense->direct",
                    "direct->dense",
                    "direct->dense",
                    "dense->direct",
                ]
            )
        )
        self.assertFalse(
            _is_exact_abba(["dense->direct", "dense->direct"])
        )
        self.assertFalse(_is_exact_abba(["dense->direct"]))

    def test_legacy_gain_field_is_removed_from_record(self) -> None:
        record = _normalize_attention_record(
            {"end_to_end_gain_percent": 7.5}
        )
        self.assertNotIn("end_to_end_gain_percent", record)
        self.assertEqual(record["attention_gain_percent"], 7.5)

    def test_correctness_prerequisite_rejects_other_wheel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "correctness.json"
            path.write_text(
                json.dumps(
                    {
                        "suite": (
                            "trianglemix_installed_wheel_npu_correctness"
                        ),
                        "status": "PASS",
                        "wheel": {"sha256": "wheel-a"},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exact --wheel"):
                _load_correctness_prerequisite(path, "wheel-b")

    def test_unified_installed_crossover_help(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                unified_main(["installed-crossover", "--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("--correctness-report", output.getvalue())
        self.assertIn("--chunk-sizes", output.getvalue())


if __name__ == "__main__":
    unittest.main()
