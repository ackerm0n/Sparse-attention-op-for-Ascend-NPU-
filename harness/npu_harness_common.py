#!/usr/bin/env python3
"""Shared runtime support for direct-wrapper NPU validation."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib.util
import math
import os
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator

import torch
import torch_npu  # noqa: F401

from harness_spec import (
    HEAD_DIM,
    KV_HEADS,
    PAGE_SIZE,
    QUERY_HEADS,
    SpanPlan,
    WrapperCase,
    plan_for_case,
)

DEFAULT_NPU_LOCK = Path(
    os.environ.get(
        "TRIANGLEMIX_NPU_LOCK",
        str(Path(tempfile.gettempdir()) / "trianglemix_npu.lock"),
    )
)
SCALE = HEAD_DIM**-0.5


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def exclusive_npu_lock(
    path: Path = DEFAULT_NPU_LOCK,
    timeout_seconds: float = 300.0,
) -> Iterator[None]:
    """Acquire the project-wide NPU lock with a bounded wait."""
    if timeout_seconds < 0:
        raise ValueError("lock timeout must be non-negative")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            try:
                fcntl.flock(
                    descriptor,
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out waiting for NPU lock {path}"
                    )
                time.sleep(
                    min(1.0, max(0.05, deadline - time.monotonic()))
                )
        os.ftruncate(descriptor, 0)
        os.write(
            descriptor,
            f"pid={os.getpid()} acquired={time.time():.6f}\n".encode(),
        )
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def load_candidate_module(
    wrapper_path: Path,
    adapter_path: Path,
) -> tuple[ModuleType, object]:
    """Load the exact candidate source and its adapter, without vLLM import."""
    wrapper_path = wrapper_path.resolve()
    adapter_path = adapter_path.resolve()
    if not wrapper_path.is_file():
        raise FileNotFoundError(wrapper_path)
    if not adapter_path.is_file():
        raise FileNotFoundError(adapter_path)

    module_name = (
        "trianglemix_direct_candidate_"
        + file_sha256(wrapper_path)[:16]
    )
    spec = importlib.util.spec_from_file_location(
        module_name,
        wrapper_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load wrapper source {wrapper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    required = (
        "TriangleMixConfig",
        "load_triangle_mix_adapter",
        "triangle_direct_eligible",
        "triangle_direct_paged_attention",
        "triangle_sparse_span",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            f"candidate wrapper lacks symbols: {', '.join(missing)}"
        )
    loaded_path = module.load_triangle_mix_adapter(str(adapter_path))
    if Path(loaded_path).resolve() != adapter_path:
        raise RuntimeError(
            f"candidate loader returned unexpected adapter {loaded_path}"
        )
    config = module.TriangleMixConfig(
        enabled=True,
        layer_indices=frozenset({0}),
        sink_tokens=8,
        local_window=512,
        last_rows=128,
        direct_min_seq_len=0,
        direct_min_sparse_rows=0,
        direct_min_saved_qk=0,
        direct_split_min_sparse_rows=0,
        direct_split_min_saved_qk=0,
    )
    return module, config


def noncontiguous_block_table(
    logical_pages: int,
    physical_pages: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return a deterministic randomized logical-to-physical page map."""
    if logical_pages <= 0 or physical_pages < logical_pages:
        raise ValueError("invalid logical/physical page counts")
    permutation = torch.randperm(
        physical_pages,
        generator=generator,
        dtype=torch.int64,
    )
    table = permutation[:logical_pages].clone()
    if (
        logical_pages > 1
        and bool(torch.all(table[1:] == table[:-1] + 1))
    ):
        first = table[0].clone()
        table[0] = table[-1]
        table[-1] = first
    return table.to(torch.int32).view(1, -1).contiguous()


@dataclass
class NpuInputs:
    query_bank_cpu: torch.Tensor
    key_cache_cpu: torch.Tensor
    value_cache_cpu: torch.Tensor
    block_table_cpu: torch.Tensor
    query_bank: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_table: torch.Tensor
    causal_mask: torch.Tensor
    logical_pages: int
    physical_pages: int

    def query(self, query_tokens: int) -> torch.Tensor:
        query = self.query_bank[:query_tokens]
        if not query.is_contiguous():
            raise RuntimeError("query-bank slice must remain contiguous")
        return query


def make_npu_inputs(
    *,
    maximum_seq_len: int,
    maximum_query_tokens: int,
    seed: int,
) -> NpuInputs:
    """Create one BF16 Q32/KV8/D128/page128 randomized paged input set."""
    if maximum_seq_len <= 0 or maximum_query_tokens <= 0:
        raise ValueError("maximum input sizes must be positive")
    if maximum_query_tokens > 2048:
        raise ValueError(
            "This harness models the production 2048-token scheduler "
            "budget; larger query calls require a separately validated "
            "CANN attention-mask contract"
        )
    generator = torch.Generator().manual_seed(seed)
    logical_pages = math.ceil(maximum_seq_len / PAGE_SIZE)
    physical_pages = logical_pages + 17
    block_table = noncontiguous_block_table(
        logical_pages,
        physical_pages,
        generator,
    )
    cache_shape = (
        physical_pages,
        PAGE_SIZE,
        KV_HEADS,
        HEAD_DIM,
    )
    key_cache_cpu = (
        torch.randn(cache_shape, generator=generator)
        .mul_(0.5)
        .to(torch.bfloat16)
    )
    value_cache_cpu = (
        torch.randn(cache_shape, generator=generator)
        .mul_(0.5)
        .to(torch.bfloat16)
    )
    query_bank_cpu = (
        torch.randn(
            maximum_query_tokens,
            QUERY_HEADS,
            HEAD_DIM,
            generator=generator,
        )
        .mul_(0.5)
        .to(torch.bfloat16)
    )
    # CANN 9.0.1's split-fuse paged FIA path requires exactly a
    # 2048-column two-dimensional causal mask even when Tq is smaller.
    mask_size = 2048
    causal_mask = (
        torch.triu(
            torch.ones(
                mask_size,
                mask_size,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        .to(torch.int8)
        .npu()
    )
    return NpuInputs(
        query_bank_cpu=query_bank_cpu,
        key_cache_cpu=key_cache_cpu,
        value_cache_cpu=value_cache_cpu,
        block_table_cpu=block_table,
        query_bank=query_bank_cpu.npu(),
        key_cache=key_cache_cpu.npu(),
        value_cache=value_cache_cpu.npu(),
        block_table=block_table.npu(),
        causal_mask=causal_mask,
        logical_pages=logical_pages,
        physical_pages=physical_pages,
    )


def dense_paged_fia_out(
    *,
    query: torch.Tensor,
    inputs: NpuInputs,
    seq_len: int,
    output: torch.Tensor,
    softmax_lse_scratch: torch.Tensor,
) -> torch.Tensor:
    """Run the single-call official dense paged FIA baseline in-place."""
    key = inputs.key_cache.view(
        inputs.key_cache.shape[0],
        PAGE_SIZE,
        -1,
    )
    value = inputs.value_cache.view(
        inputs.value_cache.shape[0],
        PAGE_SIZE,
        -1,
    )
    torch.ops.npu.npu_fused_infer_attention_score.out(
        query,
        key,
        value,
        atten_mask=inputs.causal_mask,
        block_table=inputs.block_table,
        input_layout="TND",
        block_size=PAGE_SIZE,
        actual_seq_lengths=[query.shape[0]],
        actual_seq_lengths_kv=[seq_len],
        num_key_value_heads=KV_HEADS,
        num_heads=QUERY_HEADS,
        scale=SCALE,
        pre_tokens=2147483647,
        next_tokens=0,
        sparse_mode=3,
        workspace=None,
        out=[output, softmax_lse_scratch],
    )
    return output


def run_wrapper_or_dense_fallback(
    *,
    wrapper: ModuleType,
    config: object,
    case: WrapperCase,
    inputs: NpuInputs,
    output: torch.Tensor,
    softmax_lse_scratch: torch.Tensor,
) -> tuple[torch.Tensor, str]:
    """Run the production wrapper decision with the official dense fallback."""
    query = inputs.query(case.query_tokens)
    eligible = wrapper.triangle_direct_eligible(
        query_len=case.query_tokens,
        seq_len=case.seq_len,
        prompt_len=case.prompt_len,
        config=config,
    )
    if not eligible:
        returned = dense_paged_fia_out(
            query=query,
            inputs=inputs,
            seq_len=case.seq_len,
            output=output,
            softmax_lse_scratch=softmax_lse_scratch,
        )
        return returned, "dense_fallback"

    returned = wrapper.triangle_direct_paged_attention(
        query=query,
        key_cache=inputs.key_cache,
        value_cache=inputs.value_cache,
        block_table=inputs.block_table,
        seq_len=case.seq_len,
        prompt_len=case.prompt_len,
        output=output,
        softmax_scale=SCALE,
        config=config,
    )
    return returned, "direct"


def standalone_single_launch_out(
    *,
    case: WrapperCase,
    plan: SpanPlan,
    inputs: NpuInputs,
) -> torch.Tensor | None:
    """Run the complete adapter launch alone to audit wrapper attributes."""
    if not plan.has_sparse_middle:
        return None
    query = inputs.query(case.query_tokens)
    single_launch_output = torch.empty_like(query)
    torch.ops.trianglemix.triangle_paged_sparse_attention.out(
        query,
        inputs.key_cache,
        inputs.value_cache,
        inputs.block_table,
        plan.q0,
        plan.q1,
        case.prompt_len,
        SCALE,
        out=single_launch_output,
    )
    return single_launch_output


def fp32_triangle_middle_reference(
    *,
    case: WrapperCase,
    plan: SpanPlan,
    inputs: NpuInputs,
) -> torch.Tensor | None:
    """Independent CPU FP32 sink+local reference for every sparse row."""
    if not plan.has_sparse_middle:
        return None
    positions = torch.arange(case.seq_len, dtype=torch.int64)
    logical_pages = positions // PAGE_SIZE
    page_offsets = positions % PAGE_SIZE
    physical_pages = inputs.block_table_cpu[0, logical_pages].long()
    logical_key = inputs.key_cache_cpu[
        physical_pages,
        page_offsets,
    ]
    logical_value = inputs.value_cache_cpu[
        physical_pages,
        page_offsets,
    ]
    query_cpu = inputs.query_bank_cpu[: case.query_tokens]
    sink_indices = torch.arange(8, dtype=torch.int64)
    rows: list[torch.Tensor] = []
    queries_per_kv = QUERY_HEADS // KV_HEADS
    for query_position in range(plan.s0, plan.s1):
        local_indices = torch.arange(
            query_position - 512,
            query_position + 1,
            dtype=torch.int64,
        )
        selected = torch.cat((sink_indices, local_indices))
        selected_key = logical_key[selected].float().repeat_interleave(
            queries_per_kv,
            dim=1,
        )
        selected_value = logical_value[selected].float().repeat_interleave(
            queries_per_kv,
            dim=1,
        )
        query_row = query_cpu[
            query_position - case.query_start
        ].float()
        scores = torch.einsum(
            "hd,khd->hk",
            query_row,
            selected_key,
        ) * SCALE
        probabilities = torch.softmax(scores, dim=-1)
        rows.append(
            torch.einsum(
                "hk,khd->hd",
                probabilities,
                selected_value,
            )
        )
    return torch.stack(rows)


def tensor_error_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float | bool]:
    if actual.shape != expected.shape or actual.numel() == 0:
        raise ValueError("metric tensors must have equal non-empty shapes")
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    difference = (actual_fp32 - expected_fp32).abs()
    difference_norm = float(torch.linalg.vector_norm(difference).cpu())
    expected_norm = float(torch.linalg.vector_norm(expected_fp32).cpu())
    return {
        "finite": bool(
            torch.isfinite(actual_fp32).all().cpu()
            and torch.isfinite(expected_fp32).all().cpu()
        ),
        "max_abs": float(difference.max().cpu()),
        "mean_abs": float(difference.mean().cpu()),
        "relative_l2": difference_norm / max(expected_norm, 1.0e-12),
    }


def metrics_within(
    metrics: dict[str, float | bool],
    *,
    max_abs: float,
    mean_abs: float,
) -> bool:
    return bool(
        metrics["finite"]
        and float(metrics["max_abs"]) <= max_abs
        and float(metrics["mean_abs"]) <= mean_abs
    )


def evaluate_wrapper_case(
    *,
    wrapper: ModuleType,
    config: object,
    case: WrapperCase,
    inputs: NpuInputs,
    dense_region_max_abs: float,
    dense_region_mean_abs: float,
    sparse_stage_max_abs: float,
    triangle_reference_max_abs: float,
    triangle_reference_mean_abs: float,
) -> dict[str, Any]:
    """Validate dispatch, dense/sparse regions, and the one-launch call."""
    query = inputs.query(case.query_tokens)
    plan = plan_for_case(case)
    wrapper_span = wrapper.triangle_sparse_span(
        query_len=case.query_tokens,
        seq_len=case.seq_len,
        prompt_len=case.prompt_len,
        config=config,
    )
    wrapper_coordinates = (
        int(wrapper_span.q0),
        int(wrapper_span.q1),
        int(wrapper_span.s0),
        int(wrapper_span.s1),
    )
    expected_coordinates = (plan.q0, plan.q1, plan.s0, plan.s1)
    if wrapper_coordinates != expected_coordinates:
        raise AssertionError(
            f"wrapper span {wrapper_coordinates} != {expected_coordinates}"
        )

    candidate_output = torch.empty_like(query)
    dense_output = torch.empty_like(query)
    lse_scratch = torch.empty(
        0,
        dtype=torch.float32,
        device=query.device,
    )
    started = time.perf_counter()
    dense_paged_fia_out(
        query=query,
        inputs=inputs,
        seq_len=case.seq_len,
        output=dense_output,
        softmax_lse_scratch=lse_scratch,
    )
    returned, mode = run_wrapper_or_dense_fallback(
        wrapper=wrapper,
        config=config,
        case=case,
        inputs=inputs,
        output=candidate_output,
        softmax_lse_scratch=lse_scratch,
    )
    standalone_single_launch = standalone_single_launch_out(
        case=case,
        plan=plan,
        inputs=inputs,
    )
    triangle_reference = fp32_triangle_middle_reference(
        case=case,
        plan=plan,
        inputs=inputs,
    )
    torch.npu.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    full_vs_dense = tensor_error_metrics(
        candidate_output,
        dense_output,
    )
    region_metrics: dict[str, dict[str, float | bool]] = {}
    checks: list[bool] = [
        returned is candidate_output,
        bool(full_vs_dense["finite"]),
        mode == ("direct" if plan.has_sparse_middle else "dense_fallback"),
    ]

    if not plan.has_sparse_middle:
        checks.append(
            metrics_within(
                full_vs_dense,
                max_abs=dense_region_max_abs,
                mean_abs=dense_region_mean_abs,
            )
        )
    else:
        if plan.q0 < plan.s0:
            prefix_rows = plan.s0 - plan.q0
            prefix_metrics = tensor_error_metrics(
                candidate_output[:prefix_rows],
                dense_output[:prefix_rows],
            )
            region_metrics["dense_prefix_vs_single_dense"] = prefix_metrics
            checks.append(
                metrics_within(
                    prefix_metrics,
                    max_abs=dense_region_max_abs,
                    mean_abs=dense_region_mean_abs,
                )
            )
        if plan.s1 < plan.q1:
            tail_start = plan.s1 - plan.q0
            tail_metrics = tensor_error_metrics(
                candidate_output[tail_start:],
                dense_output[tail_start:],
            )
            region_metrics["dense_tail_vs_single_dense"] = tail_metrics
            checks.append(
                metrics_within(
                    tail_metrics,
                    max_abs=dense_region_max_abs,
                    mean_abs=dense_region_mean_abs,
                )
            )
        assert standalone_single_launch is not None
        assert triangle_reference is not None
        sparse_start = plan.s0 - plan.q0
        sparse_end = plan.s1 - plan.q0
        launch_metrics = tensor_error_metrics(
            candidate_output,
            standalone_single_launch,
        )
        region_metrics["full_vs_standalone_single_launch"] = launch_metrics
        checks.append(
            bool(launch_metrics["finite"])
            and float(launch_metrics["max_abs"])
            <= sparse_stage_max_abs
        )
        reference_metrics = tensor_error_metrics(
            candidate_output[sparse_start:sparse_end].cpu(),
            triangle_reference,
        )
        region_metrics["middle_vs_cpu_fp32_triangle"] = (
            reference_metrics
        )
        checks.append(
            metrics_within(
                reference_metrics,
                max_abs=triangle_reference_max_abs,
                mean_abs=triangle_reference_mean_abs,
            )
        )

    table = (
        inputs.block_table[0, : inputs.logical_pages]
        .cpu()
        .tolist()
    )
    return {
        **case.to_dict(),
        "status": "PASS" if all(checks) else "FAIL",
        "mode": mode,
        "plan": plan.to_dict(),
        "returned_original_output": returned is candidate_output,
        "full_candidate_vs_single_dense": full_vs_dense,
        "region_metrics": region_metrics,
        "wall_ms_including_first_calls": elapsed_ms,
        "logical_pages": inputs.logical_pages,
        "physical_pages": inputs.physical_pages,
        "physical_table": table,
        "table_has_nonconsecutive_transition": any(
            right != left + 1
            for left, right in zip(table, table[1:])
        ),
    }


def event_sample_ms(
    function: Callable[[], object],
    iterations: int,
) -> float:
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / iterations


def _linear_percentile(sorted_samples: list[float], fraction: float) -> float:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must be in [0, 1]")
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    position = fraction * (len(sorted_samples) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_samples[lower]
    weight = position - lower
    return (
        sorted_samples[lower] * (1.0 - weight)
        + sorted_samples[upper] * weight
    )


def sample_summary(samples: list[float]) -> dict[str, Any]:
    if not samples:
        raise ValueError("samples must be non-empty")
    ordered = sorted(float(sample) for sample in samples)
    median = statistics.median(ordered)
    pstdev = statistics.pstdev(ordered)
    q1 = _linear_percentile(ordered, 0.25)
    q3 = _linear_percentile(ordered, 0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outlier_indices = [
        index
        for index, sample in enumerate(samples)
        if sample < lower_fence or sample > upper_fence
    ]
    mean = statistics.fmean(ordered)
    return {
        "median_ms": median,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "mean_ms": mean,
        "pstdev_ms": pstdev,
        "variance_ms2": statistics.pvariance(ordered),
        "coefficient_of_variation": (
            pstdev / abs(mean) if mean != 0.0 else 0.0
        ),
        "p10_ms": _linear_percentile(ordered, 0.10),
        "p25_ms": q1,
        "p75_ms": q3,
        "p90_ms": _linear_percentile(ordered, 0.90),
        "iqr_ms": iqr,
        "median_absolute_deviation_ms": statistics.median(
            abs(sample - median) for sample in ordered
        ),
        "tukey_lower_fence_ms": lower_fence,
        "tukey_upper_fence_ms": upper_fence,
        "tukey_outlier_count": len(outlier_indices),
        "tukey_outlier_indices": outlier_indices,
        "tukey_outlier_samples_ms": [
            samples[index] for index in outlier_indices
        ],
    }


def memory_info() -> dict[str, int] | None:
    try:
        free_bytes, total_bytes = torch.npu.mem_get_info()
    except Exception:
        return None
    return {
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "used_bytes": int(total_bytes - free_bytes),
    }
