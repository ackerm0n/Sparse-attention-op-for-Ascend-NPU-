#!/usr/bin/env python3
"""Run the legacy 8K TTFT workload from one exact self-contained wheel."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import importlib.util
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import (
    is_relative_to,
    package_version,
    sha256_file,
    sha256_json,
    write_json,
)
from .installed_artifact import (
    DISTRIBUTION_NAME,
    audit_installed_wheel,
)


EXPECTED_PROMPT_TOKEN_COUNTS = {
    "legacy_context_question": 8320,
    "legacy_chinese_long_text": 8193,
}
ENABLE_ENV = "VLLM_ASCEND_ENABLE_TRIANGLE_MIX"
LAYERS_ENV = "VLLM_ASCEND_TRIANGLE_MIX_LAYERS"
STRICT_ENV = "VLLM_ASCEND_TRIANGLE_MIX_STRICT"
STATS_INTERVAL_ENV = "VLLM_ASCEND_TRIANGLE_MIX_STATS_LOG_INTERVAL"
LEGACY_ADAPTER_ENV = "VLLM_ASCEND_TRIANGLE_MIX_ADAPTER_PATH"
EXTERNAL_OPP_ENV = "ASCEND_CUSTOM_OPP_PATH"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--legacy-script", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=("dense", "sparse"),
        required=True,
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--long-warmup-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=8500)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("wheel", "legacy_script"):
        path = getattr(args, name).resolve()
        if not path.is_file():
            raise ValueError(f"--{name.replace('_', '-')} must be a file")
    if not args.model.resolve().exists():
        raise ValueError("--model does not exist")
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if args.long_warmup_runs < 0:
        raise ValueError("--long-warmup-runs must be non-negative")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    if (
        args.max_model_len <= 0
        or args.max_num_batched_tokens <= 0
        or args.max_num_seqs <= 0
    ):
        raise ValueError("engine size limits must be positive")
    results_dir = args.results_dir.resolve()
    if results_dir.exists():
        if not results_dir.is_dir():
            raise ValueError("--results-dir must be a directory")
        if any(results_dir.iterdir()):
            raise ValueError("--results-dir must be empty")
    results_dir.mkdir(parents=True, exist_ok=True)


def _triangle_environment(variant: str) -> dict[str, str]:
    expected_enable = "1" if variant == "sparse" else "0"
    required = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "VLLM_PLUGINS": "ascend,trianglemix",
        ENABLE_ENV: expected_enable,
        STRICT_ENV: "1",
        STATS_INTERVAL_ENV: "1",
    }
    for name, expected in required.items():
        if os.getenv(name) != expected:
            raise RuntimeError(
                f"{name} must be {expected!r} for release TTFT"
            )
    if not (os.getenv(LAYERS_ENV) or "").strip():
        raise RuntimeError(f"{LAYERS_ENV} must be non-empty")
    if os.getenv(LEGACY_ADAPTER_ENV):
        raise RuntimeError(
            "legacy external adapter override is forbidden for wheel TTFT"
        )
    if os.getenv(EXTERNAL_OPP_ENV):
        raise RuntimeError(
            "external ASCEND_CUSTOM_OPP_PATH is forbidden before bundled "
            "native bootstrap"
        )
    names = {
        name
        for name in os.environ
        if (
            name == "VLLM_PLUGINS"
            or name == ENABLE_ENV
            or name.startswith("VLLM_ASCEND_TRIANGLE_MIX_")
        )
    }
    return {name: os.environ[name] for name in sorted(names)}


def _extract_legacy_strings(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted = {"chinese_cont", "context", "question"}
    values: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value_node = node.value
        else:
            continue
        if not isinstance(value_node, ast.Constant) or not isinstance(
            value_node.value,
            str,
        ):
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                values[target.id] = value_node.value
    missing = wanted - values.keys()
    if missing:
        raise RuntimeError(
            f"could not recover legacy strings: {sorted(missing)}"
        )
    return values


def _legacy_prompts(values: dict[str, str]) -> tuple[str, str]:
    english = f"""请基于以下文本回答问题：

{values["context"]}

问题：{values["question"]}

选项：
A. the democracy
B. the education of the young people who are allways considered as the sun of the nation  
C. the cooperation between various people
D. the bright future of China

请先分析每个段落中提到的共同点，然后给出答案。答案是："""
    return english, values["chinese_cont"]


def _file_record(path: Path, *, root: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file() or not is_relative_to(resolved, root):
        raise RuntimeError(f"bundled native file is invalid: {resolved}")
    return {
        "path": str(resolved),
        "relative_to_distribution": resolved.relative_to(root).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _bundled_artifact_manifest(
    inspection: object,
    *,
    distribution_root: Path,
) -> dict[str, object]:
    distribution_root = distribution_root.resolve()
    adapter = Path(str(getattr(inspection, "adapter_path", ""))).resolve()
    cust_opapi = Path(
        str(getattr(inspection, "cust_opapi_path", ""))
    ).resolve()
    opp_vendor = Path(
        str(getattr(inspection, "opp_vendor_path", ""))
    ).resolve()
    if not opp_vendor.is_dir() or not is_relative_to(
        opp_vendor,
        distribution_root,
    ):
        raise RuntimeError("bundled OPP vendor tree is outside distribution")
    files = [
        _file_record(path, root=distribution_root)
        for path in sorted(opp_vendor.rglob("*"))
        if path.is_file()
    ]
    if not files:
        raise RuntimeError("bundled OPP vendor tree is empty")
    adapter_record = _file_record(adapter, root=distribution_root)
    cust_record = _file_record(cust_opapi, root=distribution_root)
    bundle_mapping = {
        str(item["relative_to_distribution"]): str(item["sha256"])
        for item in [adapter_record, *files]
    }
    return {
        "all_files_hashed": True,
        "adapter": adapter_record,
        "cust_opapi": cust_record,
        "opp_vendor": str(opp_vendor),
        "opp_files": files,
        "opp_file_count": len(files),
        "bundle_sha256": sha256_json(bundle_mapping),
    }


def _native_provenance(
    *,
    variant: str,
    wheel: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    audit = audit_installed_wheel(wheel)
    if audit.get("passed") is not True:
        raise RuntimeError(
            "installed TriangleMix payload does not match --wheel"
        )
    distribution = importlib.metadata.distribution(DISTRIBUTION_NAME)
    distribution_root = Path(distribution.locate_file("")).resolve()
    native = importlib.import_module("vllm_ascend_trianglemix.native")
    native_path = Path(str(getattr(native, "__file__", ""))).resolve()
    if not native_path.is_file() or not is_relative_to(
        native_path,
        distribution_root,
    ) or "/package/src/" in native_path.as_posix():
        raise RuntimeError("TriangleMix native module is not installed wheel")

    inspection_object = native.inspect_native_compatibility(enabled=True)
    inspection = inspection_object.to_dict()
    if not inspection_object.compatible:
        raise RuntimeError(
            "installed bundled native runtime is incompatible: "
            + "; ".join(inspection_object.errors)
        )
    artifacts = _bundled_artifact_manifest(
        inspection_object,
        distribution_root=distribution_root,
    )
    if variant == "sparse":
        load_object = native.ensure_native_loaded(
            enabled=True,
            strict=True,
        )
        if not load_object.loaded:
            raise RuntimeError("bundled native bootstrap did not load")
        load_status: dict[str, Any] | None = load_object.to_dict()
    else:
        if native.native_status().loaded:
            raise RuntimeError("dense parent unexpectedly loaded native code")
        load_status = None

    opp_entries = [
        item
        for item in os.getenv(EXTERNAL_OPP_ENV, "").split(os.pathsep)
        if item
    ]
    ld_entries = [
        item
        for item in os.getenv("LD_LIBRARY_PATH", "").split(os.pathsep)
        if item
    ]
    expected_opp = Path(str(inspection["opp_vendor_path"])).resolve()
    expected_lib = Path(str(inspection["cust_opapi_path"])).resolve().parent
    if variant == "sparse":
        if (
            not opp_entries
            or Path(opp_entries[0]).resolve() != expected_opp
            or not ld_entries
            or Path(ld_entries[0]).resolve() != expected_lib
        ):
            raise RuntimeError(
                "bundled OPP/op-api directories were not prepended"
            )
    elif opp_entries:
        raise RuntimeError("dense parent modified ASCEND_CUSTOM_OPP_PATH")

    provenance = {
        "bundled_only": True,
        "distribution_name": distribution.metadata.get("Name"),
        "distribution_version": distribution.version,
        "distribution_root": str(distribution_root),
        "native_module": str(native_path),
        "native_module_sha256": sha256_file(native_path),
        "inspection": inspection,
        "load_status": load_status,
        "parent_bootstrap_loaded": bool(native.native_status().loaded),
        "bundled_artifacts": artifacts,
        "effective_native_environment": {
            EXTERNAL_OPP_ENV: opp_entries,
            "LD_LIBRARY_PATH_first": (
                ld_entries[0] if ld_entries else None
            ),
        },
    }
    return audit, provenance


def _module_provenance() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for module_name in ("vllm", "vllm_ascend"):
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            result[module_name] = None
            continue
        package_dir = (
            Path(next(iter(spec.submodule_search_locations))).resolve()
            if spec.submodule_search_locations
            else Path(spec.origin).resolve().parent
        )
        tracked = {
            "package_init": package_dir / "__init__.py",
        }
        if module_name == "vllm_ascend":
            tracked.update(
                {
                    "attention_v1": package_dir
                    / "attention"
                    / "attention_v1.py",
                    "model_runner_v1": package_dir
                    / "worker"
                    / "model_runner_v1.py",
                }
            )
        files = {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
            for name, path in tracked.items()
        }
        result[module_name] = {
            "origin": str(Path(spec.origin).resolve()),
            "package_dir": str(package_dir),
            "tracked_files": files,
            "tracked_bundle_sha256": sha256_json(
                {
                    name: value["sha256"]
                    for name, value in files.items()
                }
            ),
        }
    return result


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(record["latency_s"]) for record in records]
    if not latencies:
        raise ValueError("cannot summarize empty TTFT records")
    return {
        "count": len(latencies),
        "mean_s": statistics.fmean(latencies),
        "median_s": statistics.median(latencies),
        "min_s": min(latencies),
        "max_s": max(latencies),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    wheel = args.wheel.resolve()
    model = args.model.resolve()
    legacy_script = args.legacy_script.resolve()
    results_dir = args.results_dir.resolve()
    triangle_environment = _triangle_environment(args.variant)
    installed_audit, native_provenance = _native_provenance(
        variant=args.variant,
        wheel=wheel,
    )

    import torch
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams

    legacy_values = _extract_legacy_strings(legacy_script)
    english_prompt, chinese_prompt = _legacy_prompts(legacy_values)
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=True,
        local_files_only=True,
    )
    model_config = AutoConfig.from_pretrained(
        model,
        trust_remote_code=True,
        local_files_only=True,
    )
    prompt_specs = [
        {
            "name": "legacy_context_question",
            "text": english_prompt,
            "token_ids": tokenizer.encode(english_prompt),
        },
        {
            "name": "legacy_chinese_long_text",
            "text": chinese_prompt,
            "token_ids": tokenizer.encode(chinese_prompt),
        },
    ]
    for spec in prompt_specs:
        spec["token_count"] = len(spec["token_ids"])
        spec["text_sha256"] = hashlib.sha256(
            spec["text"].encode("utf-8")
        ).hexdigest()
        spec["token_ids_sha256"] = sha256_json(spec["token_ids"])
        expected = EXPECTED_PROMPT_TOKEN_COUNTS[spec["name"]]
        if spec["token_count"] != expected:
            raise RuntimeError(
                f'{spec["name"]} has {spec["token_count"]} tokens; '
                f"expected {expected}"
            )
        print(
            f'prompt={spec["name"]} tokens={spec["token_count"]} '
            f'chars={len(spec["text"])}',
            flush=True,
        )

    engine_args: dict[str, Any] = {
        "model": str(model),
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": True,
        "max_model_len": args.max_model_len,
        "block_size": 128,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "dtype": "bfloat16",
        "max_num_seqs": args.max_num_seqs,
        "enable_prefix_caching": False,
        "enforce_eager": args.enforce_eager,
        "seed": args.seed,
    }
    warmup_sampling = {
        "temperature": 0,
        "top_p": 0.9,
        "max_tokens": 10,
        "seed": args.seed,
    }
    one_token_sampling = {
        "temperature": 0,
        "top_p": 0.9,
        "max_tokens": 1,
        "seed": args.seed,
    }
    split_at = (args.runs + 1) // 2
    run_schedule = [
        (
            prompt_specs[0]["name"]
            if index < split_at
            else prompt_specs[1]["name"]
        )
        for index in range(args.runs)
    ]
    versions = {
        "python": platform.python_version(),
        "vllm": package_version("vllm"),
        "vllm_ascend": package_version("vllm-ascend"),
        "trianglemix": package_version(DISTRIBUTION_NAME),
        "torch": str(torch.__version__),
        "torch_npu": package_version("torch-npu"),
        "transformers": package_version("transformers"),
        "triton_ascend": package_version("triton-ascend"),
        "numpy": package_version("numpy"),
    }
    module_provenance = _module_provenance()
    comparison_config = {
        "benchmark_script_sha256": sha256_file(Path(__file__).resolve()),
        "wheel_sha256": sha256_file(wheel),
        "installed_payload_manifest_sha256": installed_audit.get(
            "installed_payload_manifest_sha256"
        ),
        "native_artifact_bundle_sha256": native_provenance[
            "bundled_artifacts"
        ]["bundle_sha256"],
        "legacy_script_sha256": sha256_file(legacy_script),
        "model_config_sha256": (
            sha256_file(model / "config.json")
            if (model / "config.json").is_file()
            else None
        ),
        "host": platform.node(),
        "platform": platform.platform(),
        "versions": versions,
        "model": str(model),
        "vllm_bundle_sha256": (
            module_provenance.get("vllm") or {}
        ).get("tracked_bundle_sha256"),
        "vllm_ascend_bundle_sha256": (
            module_provenance.get("vllm_ascend") or {}
        ).get("tracked_bundle_sha256"),
        "engine_args": engine_args,
        "warmup_sampling": warmup_sampling,
        "one_token_sampling": one_token_sampling,
        "short_warmup_prompt": "游泳前如何热身",
        "long_prompt_warmups_per_prompt": args.long_warmup_runs,
        "prompt_metadata": [
            {
                "name": spec["name"],
                "token_count": spec["token_count"],
                "character_count": len(spec["text"]),
                "text_sha256": spec["text_sha256"],
                "token_ids_sha256": spec["token_ids_sha256"],
            }
            for spec in prompt_specs
        ],
        "run_schedule": run_schedule,
        # This is the only intentional dense/sparse environment difference.
        "triangle_environment_except_enable": {
            name: value
            for name, value in triangle_environment.items()
            if name != ENABLE_ENV
        },
    }
    comparison_fingerprint = sha256_json(comparison_config)
    print(
        f"variant={args.variant} "
        f"comparison_fingerprint_sha256={comparison_fingerprint}",
        flush=True,
    )
    print(f"engine_args={engine_args}", flush=True)

    llm = LLM(**engine_args)
    warmup_params = SamplingParams(**warmup_sampling)
    one_token_params = SamplingParams(**one_token_sampling)
    short_warmup_started = time.perf_counter_ns()
    short_warmup = llm.generate(["游泳前如何热身"], warmup_params)
    short_warmup_seconds = (
        time.perf_counter_ns() - short_warmup_started
    ) / 1_000_000_000

    long_warmups: list[dict[str, Any]] = []
    for warmup_index in range(args.long_warmup_runs):
        for spec in prompt_specs:
            started = time.perf_counter_ns()
            output = llm.generate([spec["text"]], one_token_params)
            long_warmups.append(
                {
                    "warmup": warmup_index + 1,
                    "prompt": spec["name"],
                    "prompt_tokens": spec["token_count"],
                    "latency_s": (
                        time.perf_counter_ns() - started
                    )
                    / 1_000_000_000,
                    "generated_text": output[0].outputs[0].text,
                }
            )

    records: list[dict[str, Any]] = []
    for index, prompt_name in enumerate(run_schedule):
        spec = next(
            item for item in prompt_specs if item["name"] == prompt_name
        )
        started = time.perf_counter_ns()
        output = llm.generate([spec["text"]], one_token_params)
        record = {
            "run": index + 1,
            "prompt": spec["name"],
            "prompt_tokens": spec["token_count"],
            "latency_s": (
                time.perf_counter_ns() - started
            )
            / 1_000_000_000,
            "generated_text": output[0].outputs[0].text,
        }
        records.append(record)
        print(
            f'run={record["run"]} prompt={record["prompt"]} '
            f'tokens={record["prompt_tokens"]} '
            f'latency_s={record["latency_s"]:.6f}',
            flush=True,
        )

    by_prompt = {
        spec["name"]: {
            "prompt_tokens": spec["token_count"],
            **_summary(
                [
                    record
                    for record in records
                    if record["prompt"] == spec["name"]
                ]
            ),
        }
        for spec in prompt_specs
        if any(
            record["prompt"] == spec["name"] for record in records
        )
    }
    summary = _summary(records)
    summary["by_prompt"] = by_prompt
    result = {
        "schema_version": 1,
        "suite": "trianglemix_installed_wheel_ttft_runner",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "variant": args.variant,
        "model": str(model),
        "legacy_script": str(legacy_script),
        "wheel": {
            "path": str(wheel),
            "sha256": sha256_file(wheel),
        },
        "installed_wheel_audit": installed_audit,
        "native_provenance": native_provenance,
        "comparison_fingerprint_sha256": comparison_fingerprint,
        "comparison_config": comparison_config,
        "measurement": (
            "synchronous llm.generate latency with max_tokens=1; "
            "end-to-end TTFT methodology"
        ),
        "triangle_mix_environment": triangle_environment,
        "python_dont_write_bytecode": os.getenv(
            "PYTHONDONTWRITEBYTECODE"
        ),
        "versions": versions,
        "module_provenance": module_provenance,
        "device": {
            "name": torch.npu.get_device_name(0),
            "count": torch.npu.device_count(),
        },
        "engine_args": engine_args,
        "prompt_metadata": comparison_config["prompt_metadata"],
        "warmup": {
            "prompt": "游泳前如何热身",
            "max_tokens": 10,
            "latency_s": short_warmup_seconds,
            "generated_text": short_warmup[0].outputs[0].text,
            "long_prompt_runs_per_prompt": args.long_warmup_runs,
            "long_prompt_records": long_warmups,
        },
        "runs": records,
        "summary": summary,
    }
    output_path = results_dir / f"ttft-installed-{args.variant}.json"
    write_json(output_path, result)
    print(
        f'summary mean_s={summary["mean_s"]:.6f} '
        f'median_s={summary["median_s"]:.6f} '
        f'min_s={summary["min_s"]:.6f} '
        f'max_s={summary["max_s"]:.6f}',
        flush=True,
    )
    print(f"result_json={output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
