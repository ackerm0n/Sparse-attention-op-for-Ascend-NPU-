#!/usr/bin/env python3
"""Run one fresh vLLM engine for TriangleMix model-level smoke scenarios."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import inspect
import json
import os
import time
import traceback
from importlib import metadata
from pathlib import Path
from typing import Any

from .common import environment_fingerprint, json_safe, sha256_json, write_json
from .installed_artifact import audit_installed_wheel


START_MARKER = "TRIANGLEMIX_SCENARIO_START "
END_MARKER = "TRIANGLEMIX_SCENARIO_END "


def _cycle_tokens(pool: list[int], length: int, offset: int = 0) -> list[int]:
    if length <= 0 or not pool:
        raise ValueError("token pool and requested length must be positive")
    return [int(pool[(offset + index) % len(pool)]) for index in range(length)]


def _cached_tokens(request_output: object) -> int | None:
    candidates = (
        getattr(request_output, "num_cached_tokens", None),
        getattr(getattr(request_output, "metrics", None), "num_cached_tokens", None),
    )
    for value in candidates:
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _completion_record(request_output: object) -> dict[str, Any]:
    outputs = list(getattr(request_output, "outputs", ()))
    if not outputs:
        raise RuntimeError("vLLM request returned no completion")
    completion = outputs[0]
    token_ids = [int(value) for value in getattr(completion, "token_ids", ())]
    text = str(getattr(completion, "text", ""))
    return {
        "output_tokens": len(token_ids),
        "token_ids_sha256": sha256_json(token_ids),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text_preview": text[:160],
        "finish_reason": json_safe(getattr(completion, "finish_reason", None)),
        "stop_reason": json_safe(getattr(completion, "stop_reason", None)),
        "cached_tokens": _cached_tokens(request_output),
        "finite_runtime_output": len(token_ids) > 0,
    }


def _run_case(
    *,
    llm: object,
    sampling_params_type: type[Any],
    name: str,
    prompts: list[list[int]],
    max_tokens: int,
    seed: int,
) -> dict[str, Any]:
    print(
        START_MARKER
        + json.dumps(
            {
                "name": name,
                "prompt_lengths": [len(prompt) for prompt in prompts],
                "max_tokens": max_tokens,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    started = time.perf_counter()
    outputs = llm.generate(
        [{"prompt_token_ids": prompt} for prompt in prompts],
        sampling_params_type(
            temperature=0.0,
            min_tokens=max_tokens,
            max_tokens=max_tokens,
            seed=seed,
        ),
        use_tqdm=False,
    )
    elapsed = time.perf_counter() - started
    completions = [_completion_record(output) for output in outputs]
    result = {
        "name": name,
        "prompt_lengths": [len(prompt) for prompt in prompts],
        "prompt_token_ids_sha256": [sha256_json(prompt) for prompt in prompts],
        "request_count": len(prompts),
        "max_tokens": max_tokens,
        "elapsed_seconds": elapsed,
        "completions": completions,
        "status": (
            "PASS"
            if len(completions) == len(prompts)
            and all(item["finite_runtime_output"] for item in completions)
            else "FAIL"
        ),
    }
    print(
        END_MARKER
        + json.dumps(
            {
                "name": name,
                "status": result["status"],
                "elapsed_seconds": elapsed,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def _installed_origin(wheel: Path) -> dict[str, object]:
    spec = importlib.util.find_spec("vllm_ascend_trianglemix")
    result = {
        "distribution_version": metadata.version(
            "vllm-ascend-trianglemix"
        ),
        "module_origin": None if spec is None else spec.origin,
        "exact_wheel_payload_match": audit_installed_wheel(wheel),
    }
    if not result["exact_wheel_payload_match"]["passed"]:
        raise RuntimeError(
            "installed TriangleMix payload does not match --wheel"
        )
    return result


def _prefix_block_hashes(tokens: list[int], block_size: int) -> list[str]:
    if block_size <= 0 or len(tokens) % block_size:
        raise ValueError("prefix evidence requires full aligned blocks")
    return [
        sha256_json(tokens[start : start + block_size])
        for start in range(0, len(tokens), block_size)
    ]


def _expected_repeat_cached_tokens(
    prompt_tokens: int,
    block_size: int,
) -> int:
    if prompt_tokens <= 0 or block_size <= 0:
        raise ValueError("prompt length and block size must be positive")
    # vLLM must schedule at least one prompt token to produce the first
    # output, so a completely repeated, block-aligned prompt recomputes its
    # final cache block.
    return ((prompt_tokens - 1) // block_size) * block_size


def _validate_lengths(args: argparse.Namespace) -> None:
    requested = {
        "chunk_prompt_tokens": args.chunk_prompt_tokens,
        "decode_prompt_tokens": args.decode_prompt_tokens,
        "prefix_tokens": args.prefix_tokens,
        "prefix_suffix_tokens": args.prefix_suffix_tokens,
        "batch_prompt_tokens": args.batch_prompt_tokens,
    }
    for name, value in requested.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.chunk_prompt_tokens + 1 > args.max_model_len:
        raise ValueError("chunk prompt exceeds max model length")
    if args.decode_prompt_tokens + args.decode_tokens > args.max_model_len:
        raise ValueError("decode prompt plus completion exceeds max model length")
    if args.prefix_tokens + args.prefix_suffix_tokens + 1 > args.max_model_len:
        raise ValueError("prefix-cache prompt exceeds max model length")
    if args.prefix_tokens % 128 or args.prefix_suffix_tokens % 128:
        raise ValueError(
            "prefix-cache prefix and suffix lengths must align to 128 tokens"
        )
    if 2 * args.batch_prompt_tokens > args.max_num_batched_tokens:
        raise ValueError(
            "two batch prompts must fit in one scheduler token budget"
        )
    levels = [
        int(item)
        for item in args.concurrency_levels.split(",")
        if item.strip()
    ]
    if not levels or any(level <= 0 for level in levels):
        raise ValueError("concurrency levels must be positive")
    if max(levels) > args.concurrency_max_num_seqs:
        raise ValueError("concurrency level exceeds max_num_seqs")
    if (
        max(levels) * args.concurrency_prompt_tokens
        > args.concurrency_token_budget
    ):
        raise ValueError(
            "all concurrent prompts must fit one scheduler token budget"
        )
    if args.concurrency_prompt_tokens + 1 > args.max_model_len:
        raise ValueError("concurrency prompt exceeds max model length")


async def _drain_request_collector(collector: object) -> object:
    while True:
        output = collector.get_nowait()
        if output is None:
            output = await collector.get()
        if bool(getattr(output, "finished", False)):
            return output


async def _run_paused_request_batch(
    engine: object,
    request_specs: list[dict[str, object]],
) -> tuple[list[object], dict[str, object]]:
    """Queue a complete scheduler batch before allowing generation to step."""

    if not request_specs:
        raise ValueError("paused request batch must not be empty")
    request_ids: list[str] = []
    collectors: list[object] = []
    tasks: list[asyncio.Task[object]] = []
    paused = False
    try:
        await engine.pause_generation(mode="keep", clear_cache=False)
        paused = True
        for spec in request_specs:
            request_id = str(spec["request_id"])
            collector = await engine.add_request(
                request_id=request_id,
                prompt=spec["prompt"],
                params=spec["params"],
            )
            request_ids.append(request_id)
            collectors.append(collector)
        paused_verified = bool(await engine.is_paused())
        if not paused_verified:
            raise RuntimeError(
                "AsyncLLM resumed before the complete batch was admitted"
            )
        tasks = [
            asyncio.create_task(_drain_request_collector(collector))
            for collector in collectors
        ]
        await engine.resume_generation()
        paused = False
        outputs = list(await asyncio.gather(*tasks))
        return outputs, {
            "mode": "keep",
            "cache_cleared": False,
            "paused_verified_after_all_adds": True,
            "submitted_before_resume": len(request_ids),
            "async_collector_tasks": len(tasks),
            "all_requests_submitted_before_resume": (
                len(request_ids) == len(request_specs)
            ),
        }
    except BaseException:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        abort = getattr(engine, "abort", None)
        if callable(abort):
            await asyncio.gather(
                *(abort(request_id) for request_id in request_ids),
                return_exceptions=True,
            )
        raise
    finally:
        if paused:
            try:
                if await engine.is_paused():
                    await engine.resume_generation()
            except Exception:
                # Preserve the primary admission/collector error. The outer
                # worker shutdown remains responsible for process cleanup.
                pass


async def _run_concurrency(
    *,
    args: argparse.Namespace,
    pool_a: list[int],
    pool_b: list[int],
    installed: dict[str, object],
) -> dict[str, Any]:
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM

    levels = [
        int(item)
        for item in args.concurrency_levels.split(",")
        if item.strip()
    ]
    engine_config = {
        "model": str(args.model),
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.concurrency_max_num_seqs,
        "max_num_batched_tokens": args.concurrency_token_budget,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "seed": args.seed,
        "trust_remote_code": True,
        "block_size": 128,
    }
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_config))
    cases: list[dict[str, Any]] = []
    try:
        for level in levels:
            name = f"concurrency_{level}"
            prompts = [
                _cycle_tokens(
                    pool_a if index % 2 == 0 else pool_b,
                    args.concurrency_prompt_tokens,
                    offset=37 * index + level,
                )
                for index in range(level)
            ]
            print(
                START_MARKER
                + json.dumps(
                    {
                        "name": name,
                        "concurrency": level,
                        "prompt_lengths": [len(prompt) for prompt in prompts],
                        "async_collector_tasks": level,
                        "admission": (
                            "paused_ordered_add_then_concurrent_collect"
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            started = time.perf_counter()
            request_specs = [
                {
                    "request_id": f"{name}-{index}",
                    "prompt": {"prompt_token_ids": prompt},
                    "params": SamplingParams(
                        temperature=0.0,
                        ignore_eos=True,
                        max_tokens=1,
                        seed=args.seed + index,
                    ),
                }
                for index, prompt in enumerate(prompts)
            ]
            outputs, admission = await _run_paused_request_batch(
                engine,
                request_specs,
            )
            completions = [
                _completion_record(output) for output in outputs
            ]
            elapsed = time.perf_counter() - started
            case = {
                "name": name,
                "status": (
                    "PASS"
                    if len(completions) == level
                    and all(
                        item["finite_runtime_output"]
                        for item in completions
                    )
                    else "FAIL"
                ),
                "concurrency": level,
                "true_async_tasks": level,
                "async_collector_tasks": level,
                "admission": admission,
                "request_count": level,
                "prompt_lengths": [len(prompt) for prompt in prompts],
                "prompt_token_ids_sha256": [
                    sha256_json(prompt) for prompt in prompts
                ],
                "max_tokens": 1,
                "elapsed_seconds": elapsed,
                "completions": completions,
            }
            cases.append(case)
            print(
                END_MARKER
                + json.dumps(
                    {
                        "name": name,
                        "status": case["status"],
                        "elapsed_seconds": elapsed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        shutdown_result = engine.shutdown()
        if inspect.isawaitable(shutdown_result):
            await shutdown_result
    return {
        "schema_version": 1,
        "suite": "trianglemix_model_smoke_worker",
        "status": (
            "PASS"
            if all(case["status"] == "PASS" for case in cases)
            else "FAIL"
        ),
        "mode": "concurrency",
        "installed": installed,
        "runtime": environment_fingerprint(),
        "engine_args": engine_config,
        "model": str(args.model.resolve()),
        "concurrency_levels": levels,
        "cases": cases,
        "counter_flush": None,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_lengths(args)
    installed = _installed_origin(args.wheel.resolve())
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )
    pool_a = tokenizer.encode(
        "TriangleMix release validation checks paged attention scheduling. ",
        add_special_tokens=False,
    )
    pool_b = tokenizer.encode(
        "Independent suffix tokens exercise cache sharing and dense decode. ",
        add_special_tokens=False,
    )
    if not pool_a or not pool_b:
        raise RuntimeError("tokenizer produced an empty validation token pool")
    if args.mode == "concurrency":
        return asyncio.run(
            _run_concurrency(
                args=args,
                pool_a=pool_a,
                pool_b=pool_b,
                installed=installed,
            )
        )

    from vllm import LLM, SamplingParams

    engine_args = {
        "model": str(args.model),
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": args.max_model_len,
        "max_num_seqs": 2,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.mode == "eager",
        "enable_prefix_caching": True,
        "seed": args.seed,
        "trust_remote_code": True,
        "block_size": 128,
    }
    llm = LLM(**engine_args)
    cases: list[dict[str, Any]] = []

    if args.mode == "eager":
        chunk_prompt = _cycle_tokens(pool_a, args.chunk_prompt_tokens)
        shared_prefix = _cycle_tokens(pool_a, args.prefix_tokens, 3)
        suffix_a = _cycle_tokens(pool_a, args.prefix_suffix_tokens, 11)
        suffix_b = _cycle_tokens(pool_b, args.prefix_suffix_tokens, 17)
        batch_a = _cycle_tokens(pool_a, args.batch_prompt_tokens, 5)
        batch_b = _cycle_tokens(pool_b, args.batch_prompt_tokens, 9)
        decode_prompt = _cycle_tokens(pool_b, args.decode_prompt_tokens, 23)

        cases.append(
            _run_case(
                llm=llm,
                sampling_params_type=SamplingParams,
                name="chunked_prefill",
                prompts=[chunk_prompt],
                max_tokens=1,
                seed=args.seed,
            )
        )
        reset_prefix_cache = getattr(llm, "reset_prefix_cache", None)
        if not callable(reset_prefix_cache):
            raise RuntimeError("vLLM engine does not expose reset_prefix_cache")
        prefix_cache_reset_succeeded = bool(reset_prefix_cache())
        if not prefix_cache_reset_succeeded:
            raise RuntimeError("vLLM prefix cache reset was not acknowledged")
        seed_prompt = shared_prefix + suffix_a
        cases.append(
            _run_case(
                llm=llm,
                sampling_params_type=SamplingParams,
                name="prefix_cache_seed",
                prompts=[seed_prompt],
                max_tokens=1,
                seed=args.seed,
            )
        )
        cases.append(
            _run_case(
                llm=llm,
                sampling_params_type=SamplingParams,
                name="prefix_cache_repeat",
                prompts=[seed_prompt],
                max_tokens=1,
                seed=args.seed,
            )
        )
        prefix_cache_evidence = {
            "block_size": 128,
            "prefix_cache_reset_succeeded": (
                prefix_cache_reset_succeeded
            ),
            "shared_prefix_tokens": len(shared_prefix),
            "shared_prefix_blocks": len(shared_prefix) // 128,
            "expected_repeat_cached_tokens": _expected_repeat_cached_tokens(
                len(seed_prompt),
                128,
            ),
            "expected_shared_cached_tokens": (
                len(shared_prefix) // 128 * 128
            ),
            "seed_prefix_block_hashes": _prefix_block_hashes(
                seed_prompt[: len(shared_prefix)],
                128,
            ),
            "shared_request_prefix_block_hashes": _prefix_block_hashes(
                (shared_prefix + suffix_b)[: len(shared_prefix)],
                128,
            ),
            "seed_suffix_sha256": sha256_json(suffix_a),
            "shared_request_suffix_sha256": sha256_json(suffix_b),
            "seed_suffix_block_hashes": _prefix_block_hashes(
                suffix_a,
                128,
            ),
            "shared_request_suffix_block_hashes": _prefix_block_hashes(
                suffix_b,
                128,
            ),
            "suffixes_are_distinct": suffix_a != suffix_b,
            "shared_hit_is_partial": len(shared_prefix)
            < len(shared_prefix + suffix_b),
            "evidence_scope": (
                "scheduler-reported cached tokens plus identical aligned "
                "prefix-block hashes and distinct suffix hashes"
            ),
            "physical_block_id_identity": (
                "NOT_PROVEN_public_request_output_has_no_block_table_ids"
            ),
            "eviction_reallocation": (
                "NOT_PROVEN_not_exercised_by_this_smoke"
            ),
        }
        cases.append(
            _run_case(
                llm=llm,
                sampling_params_type=SamplingParams,
                name="prefix_cache_shared",
                prompts=[shared_prefix + suffix_b],
                max_tokens=1,
                seed=args.seed,
            )
        )
        cases.append(
            _run_case(
                llm=llm,
                sampling_params_type=SamplingParams,
                name="batch2_fallback",
                prompts=[batch_a, batch_b],
                max_tokens=1,
                seed=args.seed,
            )
        )
        cases.append(
            _run_case(
                llm=llm,
                sampling_params_type=SamplingParams,
                name="sustained_decode",
                prompts=[decode_prompt],
                max_tokens=args.decode_tokens,
                seed=args.seed,
            )
        )
    else:
        graph_prompt = _cycle_tokens(pool_a, args.graph_prompt_tokens, 7)
        cases.append(
            _run_case(
                llm=llm,
                sampling_params_type=SamplingParams,
                name="graph_capture_replay",
                prompts=[graph_prompt],
                max_tokens=args.graph_decode_tokens,
                seed=args.seed,
            )
        )

    # A final scheduler call flushes the cumulative worker snapshot after the
    # preceding scenario. It is reported but excluded from scenario gates.
    flush = _run_case(
        llm=llm,
        sampling_params_type=SamplingParams,
        name="counter_flush",
        prompts=[_cycle_tokens(pool_b, 64, 31)],
        max_tokens=1,
        seed=args.seed,
    )
    return {
        "schema_version": 1,
        "suite": "trianglemix_model_smoke_worker",
        "status": (
            "PASS"
            if all(case["status"] == "PASS" for case in cases)
            and flush["status"] == "PASS"
            else "FAIL"
        ),
        "mode": args.mode,
        "installed": installed,
        "runtime": environment_fingerprint(),
        "engine_args": engine_args,
        "model": str(args.model.resolve()),
        "model_config_sha256": (
            hashlib.sha256(
                (args.model / "config.json").read_bytes()
            ).hexdigest()
            if (args.model / "config.json").is_file()
            else None
        ),
        "triangle_environment": {
            name: os.getenv(name)
            for name in (
                "VLLM_PLUGINS",
                "VLLM_ASCEND_ENABLE_TRIANGLE_MIX",
                "VLLM_ASCEND_TRIANGLE_MIX_LAYERS",
                "VLLM_ASCEND_TRIANGLE_MIX_STRICT",
                "VLLM_ASCEND_TRIANGLE_MIX_STATS_LOG_INTERVAL",
            )
        },
        "cases": cases,
        "counter_flush": flush,
        "prefix_cache_evidence": (
            prefix_cache_evidence if args.mode == "eager" else None
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("eager", "graph", "concurrency"),
        required=True,
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=9216)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--chunk-prompt-tokens", type=int, default=8320)
    parser.add_argument("--decode-prompt-tokens", type=int, default=8193)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--prefix-tokens", type=int, default=4096)
    parser.add_argument("--prefix-suffix-tokens", type=int, default=256)
    parser.add_argument("--batch-prompt-tokens", type=int, default=768)
    parser.add_argument("--graph-prompt-tokens", type=int, default=2048)
    parser.add_argument("--graph-decode-tokens", type=int, default=16)
    parser.add_argument(
        "--concurrency-levels",
        default="1,2,4,8,16",
    )
    parser.add_argument("--concurrency-prompt-tokens", type=int, default=768)
    parser.add_argument(
        "--concurrency-token-budget",
        type=int,
        default=16384,
    )
    parser.add_argument(
        "--concurrency-max-num-seqs",
        type=int,
        default=16,
    )
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = args.output.resolve()
    started = time.time()
    try:
        report = _run(args)
        report["error"] = None
    except Exception as error:
        report = {
            "schema_version": 1,
            "suite": "trianglemix_model_smoke_worker",
            "status": "FAIL",
            "mode": args.mode,
            "model": str(args.model.resolve()),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        }
    report["started_unix_seconds"] = started
    report["finished_unix_seconds"] = time.time()
    write_json(output_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "mode": args.mode,
                "output": str(output_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
