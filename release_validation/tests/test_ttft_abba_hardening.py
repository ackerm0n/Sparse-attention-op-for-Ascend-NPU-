from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from release_validation.common import sha256_file, sha256_json
from release_validation.counters import (
    aggregate_worker_events,
    extract_stats_events,
)
from release_validation.installed_wheel_ttft_runner import (
    _bundled_artifact_manifest,
    _triangle_environment,
)
from release_validation.ttft_abba import (
    _counter_contract,
    _run_once,
    _runner_result_contract,
    _runner_result_path,
    main,
)


def _valid_sparse_counter_report() -> dict[str, object]:
    event = {
        "event": "trianglemix_runtime_stats",
        "scope": "worker_local",
        "worker": {"pid": 41, "rank": "0", "local_rank": "0"},
        "request_boundary": 1,
        "stats": {
            "counters": {
                "request_total": 1,
                "request_planner_eligible": 1,
                "request_planner_ineligible": 0,
                "layer_direct": 1,
                "single_launch": 1,
                "layer_fia": 0,
            },
            "fallback_reasons": {},
            "layers": {"direct": {"5": 1}, "fia": {}},
            "performance": {},
        },
    }
    return aggregate_worker_events(
        extract_stats_events(json.dumps(event), source="runner.log")
    )


class RunnerResultPathTests(unittest.TestCase):
    def test_result_must_be_one_file_under_dedicated_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_dir = root / "dedicated"
            result_dir.mkdir()
            result = result_dir / "result.json"
            result.write_text("{}\n", encoding="utf-8")
            self.assertEqual(
                _runner_result_path(
                    f"result_json={result}\n",
                    root,
                    result_dir=result_dir,
                ),
                result.resolve(),
            )

            outside = root / "outside.json"
            outside.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside"):
                _runner_result_path(
                    f"result_json={outside}\n",
                    root,
                    result_dir=result_dir,
                )
            with self.assertRaisesRegex(ValueError, "multiple"):
                _runner_result_path(
                    f"result_json={result}\nresult_json={result}\n",
                    root,
                    result_dir=result_dir,
                )


class RunnerIdentityTests(unittest.TestCase):
    def test_identity_and_reported_environment_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            legacy = root / "legacy.py"
            legacy.write_text("# fixture\n", encoding="utf-8")
            wheel = root / "candidate.whl"
            wheel.write_bytes(b"fixture-wheel")
            wheel_hash = sha256_file(wheel)
            environment = {
                "PYTHONDONTWRITEBYTECODE": "1",
                "VLLM_ASCEND_ENABLE_TRIANGLE_MIX": "1",
                "VLLM_ASCEND_TRIANGLE_MIX_LAYERS": "5,7",
                "VLLM_ASCEND_TRIANGLE_MIX_STRICT": "1",
                "VLLM_ASCEND_TRIANGLE_MIX_STATS_LOG_INTERVAL": "1",
            }
            comparison_config = {
                "wheel_sha256": wheel_hash,
                "installed_payload_manifest_sha256": "manifest-fixture",
                "native_artifact_bundle_sha256": "bundle-fixture",
                "triangle_environment_except_enable": {
                    name: value
                    for name, value in environment.items()
                    if name != "VLLM_ASCEND_ENABLE_TRIANGLE_MIX"
                }
            }
            result = {
                "experiment_id": "exp-sparse",
                "model": str(model),
                "legacy_script": str(legacy),
                "variant": "sparse",
                "python_dont_write_bytecode": "1",
                "wheel": {
                    "path": str(wheel),
                    "sha256": wheel_hash,
                },
                "installed_wheel_audit": {
                    "passed": True,
                    "wheel_sha256": wheel_hash,
                    "installed_payload_manifest_sha256": "manifest-fixture",
                },
                "native_provenance": {
                    "bundled_only": True,
                    "inspection": {"compatible": True},
                    "load_status": {"loaded": True, "state": "ready"},
                    "parent_bootstrap_loaded": True,
                    "bundled_artifacts": {
                        "all_files_hashed": True,
                        "bundle_sha256": "bundle-fixture",
                    },
                },
                "comparison_config": comparison_config,
                "comparison_fingerprint_sha256": sha256_json(
                    comparison_config
                ),
                "triangle_mix_environment": dict(environment),
            }
            exact = _runner_result_contract(
                result,
                experiment_id="exp-sparse",
                model=model,
                legacy_script=legacy,
                wheel=wheel,
                variant="sparse",
                environment=environment,
            )
            self.assertTrue(exact["passed"], exact["errors"])

            falsified = dict(result)
            falsified["experiment_id"] = "other-experiment"
            falsified["triangle_mix_environment"] = {
                **environment,
                "VLLM_ASCEND_ENABLE_TRIANGLE_MIX": "0",
            }
            rejected = _runner_result_contract(
                falsified,
                experiment_id="exp-sparse",
                model=model,
                legacy_script=legacy,
                wheel=wheel,
                variant="sparse",
                environment=environment,
            )
            self.assertFalse(rejected["passed"])
            self.assertFalse(
                rejected["checks"]["experiment_id_matches"]
            )
            self.assertFalse(rejected["checks"]["environment_matches"])


class CounterContractTests(unittest.TestCase):
    def test_dense_zero_and_sparse_positive_launch_contracts(self) -> None:
        empty = aggregate_worker_events([])
        dense = _counter_contract(empty, variant="dense")
        self.assertTrue(dense["passed"], dense)
        self.assertEqual(dense["single_launch"], 0)

        sparse_report = _valid_sparse_counter_report()
        sparse = _counter_contract(sparse_report, variant="sparse")
        self.assertTrue(sparse["passed"], sparse)
        self.assertEqual(sparse["single_launch"], 1)

        self.assertFalse(
            _counter_contract(empty, variant="sparse")["passed"]
        )
        self.assertFalse(
            _counter_contract(sparse_report, variant="dense")["passed"]
        )

    def test_any_counter_schema_or_invariant_error_fails(self) -> None:
        report = _valid_sparse_counter_report()
        report["schema_errors"] = [{"kind": "fixture_error"}]
        contract = _counter_contract(report, variant="sparse")
        self.assertFalse(contract["passed"])
        self.assertFalse(
            contract["checks"]["counter_schema_and_invariants"]
        )


class InstalledRunnerContractTests(unittest.TestCase):
    def test_release_environment_rejects_external_native_overrides(self) -> None:
        clean = {
            "PYTHONDONTWRITEBYTECODE": "1",
            "VLLM_PLUGINS": "ascend,trianglemix",
            "VLLM_ASCEND_ENABLE_TRIANGLE_MIX": "1",
            "VLLM_ASCEND_TRIANGLE_MIX_LAYERS": "5,7",
            "VLLM_ASCEND_TRIANGLE_MIX_STRICT": "1",
            "VLLM_ASCEND_TRIANGLE_MIX_STATS_LOG_INTERVAL": "1",
        }
        with mock.patch.dict(os.environ, clean, clear=True):
            self.assertEqual(
                _triangle_environment("sparse"),
                {
                    name: value
                    for name, value in clean.items()
                    if name != "PYTHONDONTWRITEBYTECODE"
                },
            )
        with mock.patch.dict(
            os.environ,
            {
                **clean,
                "VLLM_ASCEND_TRIANGLE_MIX_ADAPTER_PATH": "/old/adapter.so",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "external adapter"):
                _triangle_environment("sparse")
        with mock.patch.dict(
            os.environ,
            {**clean, "ASCEND_CUSTOM_OPP_PATH": "/old/opp"},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "external"):
                _triangle_environment("sparse")

    def test_comparison_environment_differs_only_by_removed_enable(self) -> None:
        base = {
            "PYTHONDONTWRITEBYTECODE": "1",
            "VLLM_PLUGINS": "ascend,trianglemix",
            "VLLM_ASCEND_TRIANGLE_MIX_LAYERS": "5,7",
            "VLLM_ASCEND_TRIANGLE_MIX_STRICT": "1",
            "VLLM_ASCEND_TRIANGLE_MIX_STATS_LOG_INTERVAL": "1",
        }
        observed: list[dict[str, str]] = []
        for variant, enabled in (("dense", "0"), ("sparse", "1")):
            with mock.patch.dict(
                os.environ,
                {
                    **base,
                    "VLLM_ASCEND_ENABLE_TRIANGLE_MIX": enabled,
                },
                clear=True,
            ):
                environment = _triangle_environment(variant)
            observed.append(
                {
                    name: value
                    for name, value in environment.items()
                    if name != "VLLM_ASCEND_ENABLE_TRIANGLE_MIX"
                }
            )
        self.assertEqual(observed[0], observed[1])

    def test_bundled_manifest_hashes_adapter_and_complete_opp_tree(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = root / "vllm_ascend_trianglemix" / "_native"
            vendor = native / "opp" / "vendors" / "trianglemix"
            library = vendor / "op_api" / "lib" / "libcust_opapi.so"
            library.parent.mkdir(parents=True)
            library.write_bytes(b"library")
            (vendor / "version.info").write_text(
                "custom_opp_compiler_version=9.0.1\n",
                encoding="utf-8",
            )
            adapter = native / "adapter.so"
            adapter.write_bytes(b"adapter")
            inspection = types.SimpleNamespace(
                adapter_path=str(adapter),
                cust_opapi_path=str(library),
                opp_vendor_path=str(vendor),
            )
            manifest = _bundled_artifact_manifest(
                inspection,
                distribution_root=root,
            )
            self.assertTrue(manifest["all_files_hashed"])
            self.assertEqual(manifest["opp_file_count"], 2)
            self.assertEqual(
                manifest["adapter"]["sha256"],
                sha256_file(adapter),
            )
            self.assertRegex(manifest["bundle_sha256"], r"^[0-9a-f]{64}$")


class PopenRunTests(unittest.TestCase):
    def test_independent_runs_record_distinct_pids(self) -> None:
        runner_source = r'''
import argparse
import hashlib
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--wheel")
parser.add_argument("--model")
parser.add_argument("--legacy-script")
parser.add_argument("--results-dir", type=Path)
parser.add_argument("--variant")
parser.add_argument("--experiment-id")
args = parser.parse_args()
args.results_dir.mkdir(parents=True, exist_ok=True)
wheel_sha256 = hashlib.sha256(Path(args.wheel).read_bytes()).hexdigest()
result = {
    "experiment_id": args.experiment_id,
    "model": args.model,
    "legacy_script": args.legacy_script,
    "variant": args.variant,
    "python_dont_write_bytecode": os.getenv("PYTHONDONTWRITEBYTECODE"),
    "wheel": {"path": args.wheel, "sha256": wheel_sha256},
    "installed_wheel_audit": {
        "passed": True,
        "wheel_sha256": wheel_sha256,
        "installed_payload_manifest_sha256": "fixture-manifest",
    },
    "native_provenance": {
        "bundled_only": True,
        "inspection": {"compatible": True},
        "load_status": (
            {"loaded": True, "state": "ready"}
            if args.variant == "sparse"
            else None
        ),
        "parent_bootstrap_loaded": args.variant == "sparse",
        "bundled_artifacts": {
            "all_files_hashed": True,
            "bundle_sha256": "fixture-native-bundle",
        },
    },
    "triangle_mix_environment": {
        name: os.getenv(name)
        for name in (
            "VLLM_ASCEND_ENABLE_TRIANGLE_MIX",
            "VLLM_ASCEND_TRIANGLE_MIX_LAYERS",
            "VLLM_ASCEND_TRIANGLE_MIX_STRICT",
            "VLLM_ASCEND_TRIANGLE_MIX_STATS_LOG_INTERVAL",
        )
    },
    "runs": [
        {
            "run": 1,
            "prompt": "fixture",
            "prompt_tokens": 16,
            "latency_s": 1.0,
        }
    ],
}
result["comparison_config"] = {
    "wheel_sha256": wheel_sha256,
    "installed_payload_manifest_sha256": "fixture-manifest",
    "native_artifact_bundle_sha256": "fixture-native-bundle",
    "triangle_environment_except_enable": {
        name: value
        for name, value in result["triangle_mix_environment"].items()
        if name != "VLLM_ASCEND_ENABLE_TRIANGLE_MIX"
    }
}
payload = json.dumps(
    result["comparison_config"],
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
result["comparison_fingerprint_sha256"] = hashlib.sha256(payload).hexdigest()
path = args.results_dir / "result.json"
path.write_text(json.dumps(result), encoding="utf-8")
if args.variant == "sparse":
    event = {
        "event": "trianglemix_runtime_stats",
        "scope": "worker_local",
        "worker": {"pid": os.getpid(), "rank": "0", "local_rank": "0"},
        "request_boundary": 1,
        "stats": {
            "counters": {
                "request_total": 1,
                "request_planner_eligible": 1,
                "request_planner_ineligible": 0,
                "layer_direct": 1,
                "single_launch": 1,
                "layer_fia": 0,
            },
            "fallback_reasons": {},
            "layers": {"direct": {"5": 1}, "fia": {}},
            "performance": {},
        },
    }
    print(json.dumps(event), flush=True)
print(f"result_json={path.resolve()}", flush=True)
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = root / "runner.py"
            runner.write_text(runner_source, encoding="utf-8")
            model = root / "model"
            model.mkdir()
            legacy = root / "legacy.py"
            legacy.write_text("# fixture\n", encoding="utf-8")
            wheel = root / "candidate.whl"
            wheel.write_bytes(b"fixture-wheel")
            args = argparse.Namespace(
                python=Path(sys.executable),
                runner=runner,
                model=model,
                legacy_script=legacy,
                wheel=wheel,
                experiment_id="fixture",
                layers="5",
                process_timeout=10.0,
            )
            artifacts = root / "artifacts"
            dense = _run_once(
                args=args,
                cycle=1,
                position=1,
                variant="dense",
                artifact_dir=artifacts,
                runner_args=[],
            )
            sparse = _run_once(
                args=args,
                cycle=1,
                position=2,
                variant="sparse",
                artifact_dir=artifacts,
                runner_args=[],
            )
            self.assertEqual(dense["status"], "PASS", dense)
            self.assertEqual(sparse["status"], "PASS", sparse)
            self.assertIsInstance(dense["pid"], int)
            self.assertIsInstance(sparse["pid"], int)
            self.assertNotEqual(dense["pid"], sparse["pid"])
            self.assertEqual(
                dense["result"]["python_dont_write_bytecode"],
                "1",
            )
            self.assertEqual(
                sparse["result"]["python_dont_write_bytecode"],
                "1",
            )


class WheelPreflightTests(unittest.TestCase):
    def test_failed_exact_wheel_audit_starts_no_runner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / "candidate.whl"
            wheel.write_bytes(b"not-used-by-mocked-audit")
            runner = root / "runner.py"
            runner.write_text("# not executed\n", encoding="utf-8")
            model = root / "model"
            model.mkdir()
            legacy = root / "legacy.py"
            legacy.write_text("# fixture\n", encoding="utf-8")
            output = root / "report.json"
            with (
                mock.patch(
                    "release_validation.ttft_abba.audit_installed_wheel",
                    return_value={
                        "passed": False,
                        "errors": [{"kind": "payload_mismatch"}],
                    },
                ),
                mock.patch(
                    "release_validation.ttft_abba._run_once"
                ) as run_once,
            ):
                code = main(
                    [
                        "--wheel",
                        str(wheel),
                        "--runner",
                        str(runner),
                        "--model",
                        str(model),
                        "--legacy-script",
                        str(legacy),
                        "--layers",
                        "5",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(code, 1)
            run_once.assert_not_called()
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "FAIL")
            self.assertFalse(
                report["checks"]["exact_installed_wheel_payload"]
            )
            self.assertEqual(report["runs"], [])


if __name__ == "__main__":
    unittest.main()
