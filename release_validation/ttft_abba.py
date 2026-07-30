#!/usr/bin/env python3
"""Orchestrate independent-process dense/sparse/sparse/dense TTFT cycles."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .common import (
    environment_fingerprint,
    is_relative_to,
    parse_json_string_list,
    percentile,
    sample_summary,
    serialise_error,
    sha256_file,
    sha256_json,
    write_json,
)
from .counters import aggregate_worker_events, extract_stats_events
from .installed_artifact import audit_installed_wheel


RESULT_PATTERN = re.compile(r"^result_json=(.+)$", re.MULTILINE)
ORDER = ("dense", "sparse", "sparse", "dense")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_RUNNER = Path(__file__).with_name(
    "installed_wheel_ttft_runner.py"
).resolve()
REQUIRED_RESULT_ENVIRONMENT = (
    "VLLM_ASCEND_ENABLE_TRIANGLE_MIX",
    "VLLM_ASCEND_TRIANGLE_MIX_LAYERS",
    "VLLM_ASCEND_TRIANGLE_MIX_STRICT",
)
COUNTER_ERROR_FIELDS = (
    "schema_errors",
    "counter_regressions",
    "snapshot_invariant_errors",
    "aggregate_invariant_errors",
)


def _latency_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    value = result.get("runs")
    if not isinstance(value, list) or not value:
        raise ValueError("TTFT runner result has no non-empty runs list")
    records: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or "latency_s" not in item:
            raise ValueError(f"invalid TTFT record at index {index}")
        records.append(item)
    return records


def _record_key(item: dict[str, Any], index: int) -> tuple[object, ...]:
    return (
        index,
        item.get("run"),
        item.get("prompt"),
        item.get("prompt_tokens"),
    )


def summarize_abba(
    runs: list[dict[str, Any]],
    *,
    cycles: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if len(runs) != cycles * len(ORDER):
        raise ValueError("ABBA run count does not match cycle count")
    dense_samples: list[float] = []
    sparse_samples: list[float] = []
    paired: list[dict[str, Any]] = []
    cycle_summaries: list[dict[str, Any]] = []

    for cycle in range(cycles):
        group = runs[cycle * 4 : cycle * 4 + 4]
        if tuple(item["variant"] for item in group) != ORDER:
            raise ValueError(f"cycle {cycle + 1} is not D-S-S-D")
        result_records = [
            _latency_records(item["result"]) for item in group
        ]
        signatures = [
            [
                _record_key(record, index)
                for index, record in enumerate(records)
            ]
            for records in result_records
        ]
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise ValueError(
                f"cycle {cycle + 1} runner schedules are not position matched"
            )
        dense_cycle = [
            float(record["latency_s"])
            for records in (result_records[0], result_records[3])
            for record in records
        ]
        sparse_cycle = [
            float(record["latency_s"])
            for records in (result_records[1], result_records[2])
            for record in records
        ]
        dense_samples.extend(dense_cycle)
        sparse_samples.extend(sparse_cycle)
        dense_mean = statistics.fmean(dense_cycle)
        sparse_mean = statistics.fmean(sparse_cycle)
        cycle_summaries.append(
            {
                "cycle": cycle + 1,
                "dense_mean_seconds": dense_mean,
                "sparse_mean_seconds": sparse_mean,
                "gain_seconds": dense_mean - sparse_mean,
                "gain_fraction": (dense_mean - sparse_mean) / dense_mean,
            }
        )
        for dense_index, sparse_index in ((0, 1), (3, 2)):
            for index, (dense, sparse) in enumerate(
                zip(
                    result_records[dense_index],
                    result_records[sparse_index],
                )
            ):
                dense_latency = float(dense["latency_s"])
                sparse_latency = float(sparse["latency_s"])
                paired.append(
                    {
                        "cycle": cycle + 1,
                        "dense_position": dense_index + 1,
                        "sparse_position": sparse_index + 1,
                        "record_key": list(_record_key(dense, index)),
                        "dense_seconds": dense_latency,
                        "sparse_seconds": sparse_latency,
                        "delta_seconds": dense_latency - sparse_latency,
                        "gain_fraction": (
                            dense_latency - sparse_latency
                        )
                        / dense_latency,
                    }
                )

    dense_mean = statistics.fmean(dense_samples)
    sparse_mean = statistics.fmean(sparse_samples)
    overall_gain = (dense_mean - sparse_mean) / dense_mean
    generator = random.Random(seed)
    bootstrap: list[float] = []
    if bootstrap_samples:
        for _ in range(bootstrap_samples):
            sampled = [
                cycle_summaries[generator.randrange(cycles)]
                for _ in range(cycles)
            ]
            sampled_dense = statistics.fmean(
                item["dense_mean_seconds"] for item in sampled
            )
            sampled_sparse = statistics.fmean(
                item["sparse_mean_seconds"] for item in sampled
            )
            bootstrap.append(
                (sampled_dense - sampled_sparse) / sampled_dense
            )
    return {
        "measurement": "end_to_end_ttft",
        "attention_microbenchmark_used": False,
        "order_per_cycle": list(ORDER),
        "cycles": cycle_summaries,
        "dense_seconds": sample_summary(dense_samples),
        "sparse_seconds": sample_summary(sparse_samples),
        "overall_gain_fraction": overall_gain,
        "overall_gain_percent": overall_gain * 100.0,
        "paired": paired,
        "paired_delta_seconds": sample_summary(
            item["delta_seconds"] for item in paired
        ),
        "bootstrap": {
            "unit": "D-S-S-D cycle",
            "samples": bootstrap_samples,
            "seed": seed,
            "gain_fraction_p2_5": (
                percentile(bootstrap, 0.025) if bootstrap else overall_gain
            ),
            "gain_fraction_p97_5": (
                percentile(bootstrap, 0.975) if bootstrap else overall_gain
            ),
        },
    }


def _runner_result_path(
    log_text: str,
    cwd: Path,
    *,
    result_dir: Path | None = None,
) -> Path:
    matches = RESULT_PATTERN.findall(log_text)
    if not matches:
        raise ValueError("TTFT runner did not print result_json=<path>")
    if len(matches) != 1:
        raise ValueError("TTFT runner printed multiple result_json paths")
    path = Path(matches[0].strip())
    resolved = (path if path.is_absolute() else cwd / path).resolve()
    if result_dir is not None and not is_relative_to(
        resolved,
        result_dir.resolve(),
    ):
        raise ValueError(
            "TTFT runner result is outside its dedicated results directory"
        )
    if not resolved.is_file():
        raise ValueError("TTFT runner result is not a regular file")
    return resolved


def _path_value_matches(value: object, expected: Path) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        return Path(value).resolve() == expected.resolve()
    except (OSError, RuntimeError, ValueError):
        return False


def _runner_result_contract(
    result: object,
    *,
    experiment_id: str,
    model: Path,
    legacy_script: Path,
    wheel: Path,
    variant: str,
    environment: dict[str, str],
) -> dict[str, Any]:
    """Validate the identity and environment asserted by one runner JSON."""

    if not isinstance(result, dict):
        return {
            "passed": False,
            "checks": {
                "result_is_object": False,
                "experiment_id_matches": False,
                "model_matches": False,
                "legacy_script_matches": False,
                "wheel_matches": False,
                "installed_wheel_audit_passed": False,
                "bundled_native_provenance": False,
                "bytecode_writes_disabled": False,
                "comparison_fingerprint_valid": False,
                "comparison_provenance_matches": False,
                "comparison_environment_excludes_only_enable": False,
                "variant_matches": False,
                "environment_matches": False,
            },
            "environment_field": None,
            "errors": [{"kind": "result_root_not_object"}],
        }

    errors: list[dict[str, object]] = []
    checks = {
        "result_is_object": True,
        "experiment_id_matches": result.get("experiment_id")
        == experiment_id,
        "model_matches": _path_value_matches(result.get("model"), model),
        "legacy_script_matches": _path_value_matches(
            result.get("legacy_script"),
            legacy_script,
        ),
        "wheel_matches": False,
        "installed_wheel_audit_passed": False,
        "bundled_native_provenance": False,
        "bytecode_writes_disabled": (
            environment.get("PYTHONDONTWRITEBYTECODE") == "1"
            and result.get("python_dont_write_bytecode") == "1"
        ),
        "comparison_fingerprint_valid": False,
        "comparison_provenance_matches": False,
        "comparison_environment_excludes_only_enable": False,
        "variant_matches": result.get("variant") == variant,
        "environment_matches": True,
    }
    for name in (
        "experiment_id_matches",
        "model_matches",
        "legacy_script_matches",
        "variant_matches",
        "bytecode_writes_disabled",
    ):
        if not checks[name]:
            errors.append({"kind": name})

    wheel_record = result.get("wheel")
    if isinstance(wheel_record, dict):
        checks["wheel_matches"] = (
            _path_value_matches(wheel_record.get("path"), wheel)
            and wheel_record.get("sha256") == sha256_file(wheel)
        )
    if not checks["wheel_matches"]:
        errors.append({"kind": "wheel_matches"})

    child_audit = result.get("installed_wheel_audit")
    checks["installed_wheel_audit_passed"] = (
        isinstance(child_audit, dict)
        and child_audit.get("passed") is True
        and child_audit.get("wheel_sha256") == sha256_file(wheel)
    )
    if not checks["installed_wheel_audit_passed"]:
        errors.append({"kind": "installed_wheel_audit_passed"})

    native_provenance = result.get("native_provenance")
    artifact_manifest: object = None
    if isinstance(native_provenance, dict):
        inspection = native_provenance.get("inspection")
        load_status = native_provenance.get("load_status")
        artifact_manifest = native_provenance.get("bundled_artifacts")
        checks["bundled_native_provenance"] = (
            native_provenance.get("bundled_only") is True
            and isinstance(inspection, dict)
            and inspection.get("compatible") is True
            and isinstance(artifact_manifest, dict)
            and artifact_manifest.get("all_files_hashed") is True
            and (
                (
                    variant == "sparse"
                    and isinstance(load_status, dict)
                    and load_status.get("loaded") is True
                    and load_status.get("state") == "ready"
                )
                or (
                    variant == "dense"
                    and load_status is None
                    and native_provenance.get("parent_bootstrap_loaded")
                    is False
                )
            )
        )
    if not checks["bundled_native_provenance"]:
        errors.append({"kind": "bundled_native_provenance"})

    comparison_config = result.get("comparison_config")
    fingerprint = result.get("comparison_fingerprint_sha256")
    checks["comparison_fingerprint_valid"] = (
        isinstance(comparison_config, dict)
        and isinstance(fingerprint, str)
        and re.fullmatch(r"[0-9a-f]{64}", fingerprint) is not None
        and sha256_json(comparison_config) == fingerprint
    )
    if not checks["comparison_fingerprint_valid"]:
        errors.append({"kind": "comparison_fingerprint_valid"})
    if isinstance(comparison_config, dict) and isinstance(
        artifact_manifest,
        dict,
    ):
        checks["comparison_provenance_matches"] = (
            comparison_config.get("wheel_sha256") == sha256_file(wheel)
            and comparison_config.get(
                "installed_payload_manifest_sha256"
            )
            == (
                child_audit.get("installed_payload_manifest_sha256")
                if isinstance(child_audit, dict)
                else None
            )
            and comparison_config.get(
                "native_artifact_bundle_sha256"
            )
            == artifact_manifest.get("bundle_sha256")
        )
    if not checks["comparison_provenance_matches"]:
        errors.append({"kind": "comparison_provenance_matches"})

    environment_candidates = [
        name
        for name in ("triangle_mix_environment", "triangle_environment")
        if name in result
    ]
    environment_field = (
        environment_candidates[0]
        if len(environment_candidates) == 1
        else None
    )
    reported_environment = (
        result.get(environment_field)
        if environment_field is not None
        else None
    )
    if len(environment_candidates) != 1:
        checks["environment_matches"] = False
        errors.append(
            {
                "kind": "runner_environment_field_count",
                "observed": environment_candidates,
            }
        )
    elif not isinstance(reported_environment, dict):
        checks["environment_matches"] = False
        errors.append({"kind": "runner_environment_not_object"})
    else:
        for name, value in reported_environment.items():
            if not isinstance(name, str) or not (
                value is None or isinstance(value, str)
            ):
                checks["environment_matches"] = False
                errors.append(
                    {
                        "kind": "invalid_runner_environment_entry",
                        "name": str(name),
                    }
                )
                continue
            if value != environment.get(name):
                checks["environment_matches"] = False
                errors.append(
                    {
                        "kind": "runner_environment_mismatch",
                        "name": name,
                        "expected": environment.get(name),
                        "observed": value,
                    }
                )
        for name in REQUIRED_RESULT_ENVIRONMENT:
            expected = environment.get(name)
            observed = reported_environment.get(name)
            if name not in reported_environment or observed != expected:
                checks["environment_matches"] = False
                errors.append(
                    {
                        "kind": "required_runner_environment_mismatch",
                        "name": name,
                        "expected": expected,
                        "observed": observed,
                    }
                )
        comparison_environment = (
            comparison_config.get("triangle_environment_except_enable")
            if isinstance(comparison_config, dict)
            else None
        )
        checks["comparison_environment_excludes_only_enable"] = (
            comparison_environment
            == {
                name: value
                for name, value in reported_environment.items()
                if name != "VLLM_ASCEND_ENABLE_TRIANGLE_MIX"
            }
        )
        if not checks["comparison_environment_excludes_only_enable"]:
            errors.append(
                {
                    "kind": (
                        "comparison_environment_excludes_only_enable"
                    )
                }
            )

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "environment_field": environment_field,
        "errors": errors,
    }


def _counter_contract(
    counters: dict[str, Any],
    *,
    variant: str,
) -> dict[str, Any]:
    """Fail closed on malformed counters and the custom-launch invariant."""

    error_fields_well_formed = all(
        isinstance(counters.get(name), list)
        for name in COUNTER_ERROR_FIELDS
    )
    counter_errors_absent = error_fields_well_formed and all(
        not counters[name] for name in COUNTER_ERROR_FIELDS
    )
    event_count = counters.get("event_count")
    worker_count = counters.get("worker_count")
    counts_well_formed = (
        isinstance(event_count, int)
        and not isinstance(event_count, bool)
        and event_count >= 0
        and isinstance(worker_count, int)
        and not isinstance(worker_count, bool)
        and worker_count >= 0
        and (event_count > 0 or worker_count == 0)
        and (event_count == 0 or worker_count > 0)
    )
    aggregate = counters.get("aggregate")
    aggregate = aggregate if isinstance(aggregate, dict) else {}
    aggregate_counters = aggregate.get("counters")
    aggregate_counters = (
        aggregate_counters if isinstance(aggregate_counters, dict) else {}
    )
    single_launch = aggregate_counters.get("single_launch", 0)
    launch_well_formed = (
        isinstance(single_launch, (int, float))
        and not isinstance(single_launch, bool)
        and single_launch >= 0
    )
    launch_matches_variant = launch_well_formed and (
        single_launch == 0 if variant == "dense" else single_launch > 0
    )
    sparse_has_counter_evidence = variant == "dense" or (
        counts_well_formed and event_count > 0 and worker_count > 0
    )
    checks = {
        "counter_report_schema": (
            counters.get("schema_version") == 1
            and counters.get("event") == "trianglemix_runtime_stats"
            and error_fields_well_formed
            and counts_well_formed
        ),
        "counter_schema_and_invariants": counter_errors_absent,
        "sparse_has_counter_evidence": sparse_has_counter_evidence,
        "custom_launch_matches_variant": launch_matches_variant,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "single_launch": single_launch if launch_well_formed else None,
        "event_count": event_count,
        "worker_count": worker_count,
        "error_fields": {
            name: counters.get(name) for name in COUNTER_ERROR_FIELDS
        },
    }


def _run_once(
    *,
    args: argparse.Namespace,
    cycle: int,
    position: int,
    variant: str,
    artifact_dir: Path,
    runner_args: list[str],
) -> dict[str, Any]:
    label = f"cycle-{cycle:02d}-{position:02d}-{variant}"
    result_dir = artifact_dir / "runner_results" / label
    result_dir.mkdir(parents=True)
    log_path = artifact_dir / "logs" / f"{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_id = f"{args.experiment_id}-{label}"
    runner_path = args.runner.resolve()
    bundled_runner = runner_path == BUNDLED_RUNNER
    runner_cwd = PROJECT_ROOT if bundled_runner else runner_path.parent
    command = [
        str(args.python.absolute()),
        *(
            ["-m", "release_validation.installed_wheel_ttft_runner"]
            if bundled_runner
            else [str(runner_path)]
        ),
        "--wheel",
        str(args.wheel.resolve()),
        "--model",
        str(args.model.resolve()),
        "--legacy-script",
        str(args.legacy_script.resolve()),
        "--results-dir",
        str(result_dir),
        "--variant",
        variant,
        "--experiment-id",
        experiment_id,
        *runner_args,
    ]
    environment = os.environ.copy()
    # The release wheel owns both native payloads.  Do not let a historical
    # site-packages override or an external OPP tree contaminate the run.
    environment.pop("VLLM_ASCEND_TRIANGLE_MIX_ADAPTER_PATH", None)
    environment.pop("ASCEND_CUSTOM_OPP_PATH", None)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "VLLM_PLUGINS": "ascend,trianglemix",
            "VLLM_ASCEND_ENABLE_TRIANGLE_MIX": (
                "1" if variant == "sparse" else "0"
            ),
            "VLLM_ASCEND_TRIANGLE_MIX_LAYERS": args.layers,
            "VLLM_ASCEND_TRIANGLE_MIX_STRICT": "1",
            "VLLM_ASCEND_TRIANGLE_MIX_STATS_LOG_INTERVAL": "1",
        }
    )
    started = time.time()
    process: subprocess.Popen[str] | None = None
    process_id = None
    log_text = ""
    return_code = None
    process_error = None
    try:
        process = subprocess.Popen(
            command,
            cwd=runner_cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        process_id = process.pid
        try:
            log_text, _ = process.communicate(timeout=args.process_timeout)
        except subprocess.TimeoutExpired as error:
            process.kill()
            log_text, _ = process.communicate()
            process_error = serialise_error(error)
        return_code = process.returncode
    except OSError as error:
        if process is not None and process.poll() is None:
            process.kill()
            drained, _ = process.communicate()
            log_text = drained or log_text
        process_error = serialise_error(error)
    log_text = log_text or ""
    log_path.write_text(log_text, encoding="utf-8")
    result_path = None
    result = None
    parse_error = None
    try:
        result_path = _runner_result_path(
            log_text,
            runner_cwd,
            result_dir=result_dir,
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as error:
        parse_error = serialise_error(error)
    events = extract_stats_events(log_text, source=str(log_path))
    counters = aggregate_worker_events(events)
    result_contract = _runner_result_contract(
        result,
        experiment_id=experiment_id,
        model=args.model.resolve(),
        legacy_script=args.legacy_script.resolve(),
        wheel=args.wheel.resolve(),
        variant=variant,
        environment=environment,
    )
    counter_contract = _counter_contract(counters, variant=variant)
    checks = {
        "process_started_with_pid": (
            isinstance(process_id, int) and process_id > 0
        ),
        "process_exit_zero": (
            return_code == 0 and process_error is None
        ),
        "runner_result_in_dedicated_directory": (
            parse_error is None
            and result_path is not None
            and is_relative_to(result_path, result_dir.resolve())
        ),
        "runner_result_contract": result_contract["passed"],
        "worker_counter_contract": counter_contract["passed"],
    }
    valid = all(checks.values())
    return {
        "label": label,
        "cycle": cycle,
        "position": position,
        "variant": variant,
        "status": "PASS" if valid else "FAIL",
        "pid": process_id,
        "command": command,
        "return_code": return_code,
        "process_error": process_error,
        "parse_error": parse_error,
        "checks": checks,
        "result_contract": result_contract,
        "counter_contract": counter_contract,
        "log": str(log_path),
        "runner_result": None if result_path is None else str(result_path),
        "result": result,
        "worker_counters": counters,
        "started_unix_seconds": started,
        "finished_unix_seconds": time.time(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel",
        type=Path,
        required=True,
        help="The exact wheel artifact installed in the runner environment.",
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=BUNDLED_RUNNER,
        help=(
            "Wheel-aware TTFT runner; defaults to the bundled release runner."
        ),
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--legacy-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--layers", required=True)
    parser.add_argument("--experiment-id", default="trianglemix-release-abba")
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--min-gain-percent", type=float, default=10.0)
    parser.add_argument("--process-timeout", type=float, default=3600.0)
    parser.add_argument(
        "--runner-args-json",
        default="[]",
        help='Additional runner arguments, for example ["--runs","6"].',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    if args.cycles <= 0:
        raise SystemExit("--cycles must be positive")
    if args.bootstrap_samples < 0:
        raise SystemExit("--bootstrap-samples must be non-negative")
    if args.process_timeout <= 0:
        raise SystemExit("--process-timeout must be positive")
    for path_name in ("wheel", "runner", "model", "legacy_script", "python"):
        path = getattr(args, path_name)
        if not path.exists():
            raise SystemExit(f"--{path_name.replace('_', '-')} does not exist")
        if path_name != "model" and not path.is_file():
            raise SystemExit(
                f"--{path_name.replace('_', '-')} must be a file"
            )
    runner_args = parse_json_string_list(
        args.runner_args_json,
        name="--runner-args-json",
    )
    output_path = args.output.resolve()
    artifact_dir = (
        args.artifacts_dir.resolve()
        if args.artifacts_dir is not None
        else output_path.with_name(f"{output_path.stem}_artifacts")
    )
    if artifact_dir.exists():
        if not artifact_dir.is_dir():
            raise SystemExit(
                f"artifacts path must be a directory: {artifact_dir}"
            )
        if any(artifact_dir.iterdir()):
            raise SystemExit(
                f"artifacts directory must be empty: {artifact_dir}"
            )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = args.wheel.resolve()
    try:
        installed_wheel_audit = audit_installed_wheel(wheel_path)
    except Exception as error:
        installed_wheel_audit = {
            "passed": False,
            "wheel": str(wheel_path),
            "errors": [
                {
                    "kind": "installed_wheel_audit_error",
                    **serialise_error(error),
                }
            ],
        }
    started = time.time()
    runs: list[dict[str, Any]] = []
    audited_python = Path(sys.executable).absolute()
    runner_python = args.python.absolute()
    same_audited_interpreter = runner_python == audited_python
    if (
        installed_wheel_audit.get("passed") is True
        and same_audited_interpreter
    ):
        for cycle in range(1, args.cycles + 1):
            for position, variant in enumerate(ORDER, start=1):
                run = _run_once(
                    args=args,
                    cycle=cycle,
                    position=position,
                    variant=variant,
                    artifact_dir=artifact_dir,
                    runner_args=runner_args,
                )
                runs.append(run)
                if run["status"] != "PASS":
                    break
            if (
                len(runs) != cycle * 4
                or runs[-1]["status"] != "PASS"
            ):
                break

    summary = None
    summary_error = None
    try:
        summary = summarize_abba(
            runs,
            cycles=args.cycles,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
    except Exception as error:
        summary_error = {
            "type": type(error).__name__,
            "message": str(error),
        }
    fingerprints = {
        run["result"].get("comparison_fingerprint_sha256")
        for run in runs
        if isinstance(run.get("result"), dict)
    }
    process_ids = [
        run.get("pid")
        for run in runs
        if isinstance(run.get("pid"), int)
        and not isinstance(run.get("pid"), bool)
        and run["pid"] > 0
    ]
    expected_processes = args.cycles * len(ORDER)
    checks = {
        "controller_bytecode_writes_disabled": (
            os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
            and sys.dont_write_bytecode
        ),
        "exact_installed_wheel_payload": (
            installed_wheel_audit.get("passed") is True
        ),
        "runner_uses_audited_python": same_audited_interpreter,
        "all_processes_passed": all(
            run["status"] == "PASS" for run in runs
        )
        and len(runs) == expected_processes,
        "all_process_pids_unique": (
            len(process_ids) == expected_processes
            and len(set(process_ids)) == expected_processes
        ),
        "at_least_four_abba_cycles": args.cycles >= 4
        and len(runs) == expected_processes,
        "one_comparison_fingerprint": len(fingerprints) == 1
        and None not in fingerprints,
        "summary_available": summary is not None,
        "paired_gain_positive_95ci": (
            summary is not None
            and summary["bootstrap"]["gain_fraction_p2_5"] > 0
        ),
        "ttft_gain_target_met": (
            summary is not None
            and summary["overall_gain_percent"] >= args.min_gain_percent
        ),
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "suite": "trianglemix_independent_process_ttft_abba",
        "status": "PASS" if passed else "FAIL",
        "measurement": "end_to_end_ttft",
        "attention_microbenchmark_used": False,
        "runner": {
            "path": str(args.runner.resolve()),
            "sha256": sha256_file(args.runner.resolve()),
            "python": str(runner_python),
            "extra_args": runner_args,
        },
        "wheel": {
            "path": str(wheel_path),
            "sha256": sha256_file(wheel_path),
            "installed_payload_audit": installed_wheel_audit,
        },
        "runtime": environment_fingerprint(),
        "model": str(args.model.resolve()),
        "legacy_script": str(args.legacy_script.resolve()),
        "layers": args.layers,
        "cycles_requested": args.cycles,
        "min_gain_percent": args.min_gain_percent,
        "comparison_fingerprints": sorted(
            str(value) for value in fingerprints
        ),
        "checks": checks,
        "summary": summary,
        "summary_error": summary_error,
        "runs": runs,
        "process_ids": process_ids,
        "artifacts_dir": str(artifact_dir),
        "started_unix_seconds": started,
        "finished_unix_seconds": time.time(),
    }
    write_json(output_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "processes": len(runs),
                "gain_percent": (
                    None
                    if summary is None
                    else summary["overall_gain_percent"]
                ),
                "output": str(output_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
