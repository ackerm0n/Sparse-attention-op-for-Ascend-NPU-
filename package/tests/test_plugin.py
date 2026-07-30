from __future__ import annotations

import json
import os
import sys
import types
import unittest
from enum import Enum
from pathlib import Path
from unittest.mock import Mock, patch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")

from vllm_ascend_trianglemix import plugin
from vllm_ascend_trianglemix.planning import FallbackReason
from vllm_ascend_trianglemix.stats import runtime_stats


class State(Enum):
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3


class FakeTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        dtype: str = "torch.bfloat16",
        device: str = "npu:0",
        contiguous: bool = True,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self._contiguous = contiguous
        self.contiguous_checks = 0

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def stride(self) -> tuple[int, ...]:
        running = 1
        result: list[int] = []
        for size in reversed(self.shape):
            result.append(running)
            running *= max(1, size)
        return tuple(reversed(result))

    def is_contiguous(self) -> bool:
        self.contiguous_checks += 1
        return self._contiguous

    def __getitem__(self, item: object) -> "FakeTensor":
        if not isinstance(item, slice):
            raise TypeError(item)
        start = 0 if item.start is None else int(item.start)
        stop = self.shape[0] if item.stop is None else int(item.stop)
        first = max(0, min(self.shape[0], stop) - start)
        return FakeTensor(
            (first, *self.shape[1:]),
            dtype=self.dtype,
            device=self.device,
            contiguous=self._contiguous,
        )

    def view(self, *shape: int) -> "FakeTensor":
        return FakeTensor(
            tuple(shape),
            dtype=self.dtype,
            device=self.device,
            contiguous=self._contiguous,
        )


def plugin_config(
    *,
    enabled: bool = True,
    strict: bool = False,
    layers: str = "3",
    **overrides: object,
) -> types.SimpleNamespace:
    section: dict[str, object] = {
        "enabled": enabled,
        "strict": strict,
        "layers": layers,
    }
    section.update(overrides)
    return types.SimpleNamespace(
        additional_config={"trianglemix": section}
    )


def make_common(
    *,
    query_len: int = 4096,
    seq_len: int = 4096,
    prompt_len: int | None = 4096,
    state: State = State.PrefillNoCache,
    causal: bool = True,
    block_columns: int = 64,
) -> types.SimpleNamespace:
    metadata = types.SimpleNamespace(
        actual_seq_lengths_q=[query_len],
        seq_lens_list=[seq_len],
        num_decodes=1 if state is State.DecodeOnly else 0,
        num_prefills=0 if state is State.DecodeOnly else 1,
        attn_state=state,
        causal=causal,
        block_tables=FakeTensor(
            (1, block_columns),
            dtype="torch.int32",
        ),
    )
    return types.SimpleNamespace(
        metadata=metadata,
        num_prompt_tokens_cpu=(
            None if prompt_len is None else [prompt_len]
        ),
        attn_state=state,
    )


def make_attention_module(
    *,
    legacy_hooks: bool,
    patch_module: bool = True,
) -> tuple[types.SimpleNamespace, type, type]:
    class Builder:
        def __init__(
            self,
            kv_cache_spec: object,
            layer_names: list[str],
            vllm_config: object,
            device: object,
        ) -> None:
            del kv_cache_spec, layer_names, device
            self.vllm_config = vllm_config

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: object,
            fast_build: bool = False,
        ) -> object:
            del common_prefix_len, fast_build
            return common_attn_metadata.metadata

        def build_for_graph_capture(
            self,
            common_attn_metadata: object,
            attn_state: State = State.DecodeOnly,
        ) -> object:
            metadata = self.build(0, common_attn_metadata)
            metadata.attn_state = attn_state
            return metadata

    class Backend:
        def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: list[float] | None,
            sliding_window: int | None,
            kv_cache_dtype: str,
            logits_soft_cap: float | None,
            attn_type: str,
            kv_sharing_target_layer_name: str | None,
            sinks: object = None,
            **kwargs: object,
        ) -> None:
            del (
                kv_cache_dtype,
                logits_soft_cap,
                kv_sharing_target_layer_name,
            )
            vllm_config = kwargs.pop("vllm_config")
            if legacy_hooks:
                module.load_triangle_mix_adapter_if_enabled(
                    True,
                    "/legacy/adapter.so",
                )
                self._triangle_mix_layer_enabled = None
            self.vllm_config = vllm_config
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.head_size = head_size
            self.scale = scale
            self.sliding_window = sliding_window
            self.sinks = sinks
            self.alibi_slopes = alibi_slopes
            self.enable_hamming_sparse = False
            self.enable_c8_quant = False
            self.attn_type = attn_type
            self.key_cache = None
            self.value_cache = None
            self.official_calls = 0
            self.legacy_calls = 0

        def forward(
            self,
            layer: object,
            query: object,
            key: object,
            value: object,
            kv_cache: tuple[object, object],
            attn_metadata: object,
            output: object | None = None,
            output_scale: object | None = None,
            output_block_scale: object | None = None,
        ) -> object:
            del layer, output_scale, output_block_scale
            assert output is not None
            self.key_cache, self.value_cache = kv_cache
            return self.forward_fused_infer_attention(
                query,
                key,
                value,
                attn_metadata,
                output,
                kv_cache,
            )

        def forward_fused_infer_attention(
            self,
            query: object,
            key: object,
            value: object,
            attn_metadata: object,
            output: object,
            kv_cache: object = None,
        ) -> object:
            del query, key, value, attn_metadata, kv_cache
            predicate = getattr(self, "_can_use_triangle_mix", None)
            if callable(predicate) and predicate():
                return self._forward_triangle_mix(output)
            self.official_calls += 1
            return output

    if legacy_hooks:

        def can_use(self: object, *args: object) -> bool:
            del self, args
            return True

        def forward_triangle(self: object, output: object) -> object:
            self.legacy_calls += 1
            return output

        Backend._can_use_triangle_mix = can_use
        Backend._forward_triangle_mix = forward_triangle

    legacy_loader = Mock(return_value=True)
    module = types.SimpleNamespace(
        AscendAttentionMetadataBuilder=Builder,
        AscendAttentionBackendImpl=Backend,
        get_tensor_model_parallel_world_size=lambda: 1,
        enable_cp=lambda: False,
        _EXTRA_CTX=types.SimpleNamespace(capturing=False),
        load_triangle_mix_adapter_if_enabled=legacy_loader,
    )
    if patch_module:
        plugin._patch_attention_module(
            module,
            versions=("0.23.0", "0.23.0rc1"),
        )
    return module, Builder, Backend


def new_builder(
    builder_cls: type,
    config: object,
) -> object:
    return builder_cls(None, [], config, "npu:0")


def new_backend(
    backend_cls: type,
    config: object,
) -> object:
    return backend_cls(
        32,
        128,
        0.088,
        8,
        None,
        None,
        "auto",
        None,
        "decoder",
        None,
        vllm_config=config,
    )


def make_runner_module(
    result_factory: object,
    *,
    prompt_lens: object = (8320,),
    patch_module: bool = True,
) -> tuple[types.SimpleNamespace, type]:
    class Runner:
        def __init__(self) -> None:
            self.input_batch = types.SimpleNamespace(
                num_prompt_tokens_cpu_tensor=prompt_lens
            )
            self.speculative_config = None
            self.use_async_spec_decode = False
            self.result_factory = result_factory

        def _build_attention_metadata(
            self,
            num_tokens: int,
            num_reqs: int,
            max_query_len: int,
            num_tokens_padded: int | None = None,
            num_reqs_padded: int | None = None,
            ubatch_slices: object | None = None,
            logits_indices: object | None = None,
            use_spec_decode: bool = False,
            for_cudagraph_capture: bool = False,
            num_scheduled_tokens: dict[str, int] | None = None,
            num_scheduled_tokens_np: object | None = None,
            cascade_attn_prefix_lens: list[list[int]] | None = None,
        ) -> object:
            del (
                num_tokens,
                num_reqs,
                max_query_len,
                num_tokens_padded,
                num_reqs_padded,
                ubatch_slices,
                logits_indices,
                use_spec_decode,
                for_cudagraph_capture,
                num_scheduled_tokens,
                num_scheduled_tokens_np,
                cascade_attn_prefix_lens,
            )
            return self.result_factory()

    module = types.SimpleNamespace(NPUModelRunner=Runner)
    if patch_module:
        plugin._patch_model_runner_module(
            module,
            versions=("0.23.0", "0.23.0rc1"),
        )
    return module, Runner


def tensors(
    query_len: int = 4096,
) -> tuple[FakeTensor, FakeTensor, FakeTensor, FakeTensor]:
    return (
        FakeTensor((query_len, 32, 128)),
        FakeTensor((80, 128, 8, 128)),
        FakeTensor((80, 128, 8, 128)),
        FakeTensor((query_len, 4096)),
    )


class PluginTests(unittest.TestCase):
    def test_runtime_stats_logger_uses_vllm_namespace(self) -> None:
        self.assertTrue(plugin.logger.name.startswith("vllm."))

    def setUp(self) -> None:
        runtime_stats().snapshot(reset=True)

    def tearDown(self) -> None:
        runtime_stats().snapshot(reset=True)

    def test_feature_off_does_not_touch_native_runtime(self) -> None:
        _, _, Backend = make_attention_module(legacy_hooks=False)
        ensure_native = Mock(return_value=True)
        ensure_runner = Mock(return_value=True)
        with patch.object(
            plugin,
            "_ensure_native_ready",
            ensure_native,
        ), patch.object(
            plugin,
            "_ensure_model_runner_patch",
            ensure_runner,
        ):
            backend = new_backend(
                Backend,
                plugin_config(enabled=False),
            )
        ensure_native.assert_not_called()
        ensure_runner.assert_not_called()
        self.assertFalse(
            getattr(backend, plugin._NATIVE_READY_ATTR)
        )

    def test_framework_version_gate_is_exact_except_local_suffix(
        self,
    ) -> None:
        plugin._validate_versions(
            ("0.23.0+vendor.1", "0.23.0rc1+build.7")
        )
        for versions in (
            ("0.23.1", "0.23.0rc1"),
            ("0.23.0", "0.23.0"),
            ("0.23.0.dev1", "0.23.0rc1"),
        ):
            with self.subTest(versions=versions):
                with self.assertRaises(
                    plugin._UnsupportedCompatibilityError
                ):
                    plugin._validate_versions(versions)

    def test_attention_preflight_never_leaves_a_half_patch(
        self,
    ) -> None:
        module, Builder, Backend = make_attention_module(
            legacy_hooks=False,
            patch_module=False,
        )
        originals = {
            "builder_build": Builder.build,
            "backend_init": Backend.__init__,
            "backend_forward": Backend.forward,
            "backend_fia": Backend.forward_fused_infer_attention,
        }
        with self.assertRaises(
            plugin._UnsupportedCompatibilityError
        ):
            plugin._patch_attention_module(
                module,
                versions=("0.23.1", "0.23.0rc1"),
            )
        self.assertIs(Builder.build, originals["builder_build"])
        self.assertIs(Backend.forward, originals["backend_forward"])

        with (
            patch.object(
                plugin,
                "_backend_patch",
                side_effect=RuntimeError("late patch failure"),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "late patch failure",
            ),
        ):
            plugin._patch_attention_module(
                module,
                versions=("0.23.0", "0.23.0rc1"),
            )
        self.assertIs(Builder.build, originals["builder_build"])
        self.assertIs(Backend.__init__, originals["backend_init"])
        self.assertIs(Backend.forward, originals["backend_forward"])
        self.assertIs(
            Backend.forward_fused_infer_attention,
            originals["backend_fia"],
        )
        self.assertFalse(getattr(Builder, plugin._PATCH_MARKER, False))
        self.assertFalse(getattr(Backend, plugin._PATCH_MARKER, False))

    def test_attention_signature_mismatch_is_rejected_before_patch(
        self,
    ) -> None:
        module, Builder, Backend = make_attention_module(
            legacy_hooks=False,
            patch_module=False,
        )
        original_build = Builder.build

        def incompatible_forward(self: object) -> object:
            return self

        Backend.forward = incompatible_forward
        with self.assertRaises(
            plugin._UnsupportedCompatibilityError
        ):
            plugin._patch_attention_module(
                module,
                versions=("0.23.0", "0.23.0rc1"),
            )
        self.assertIs(Builder.build, original_build)
        self.assertIs(Backend.forward, incompatible_forward)
        self.assertFalse(getattr(Builder, plugin._PATCH_MARKER, False))
        self.assertFalse(getattr(Backend, plugin._PATCH_MARKER, False))

    def test_runner_gate_records_non_strict_and_raises_strict(
        self,
    ) -> None:
        module = types.SimpleNamespace(
            NPUModelRunner=type(
                "BadRunner",
                (),
                {
                    "_build_attention_metadata": (
                        lambda self: ({}, None)
                    )
                },
            )
        )
        with (
            patch.dict(
                sys.modules,
                {plugin._RUNNER_MODULE: module},
            ),
            patch.object(
                plugin,
                "_installed_versions",
                return_value=("0.23.0", "0.23.0rc1"),
            ),
            patch.object(plugin.logger, "warning"),
        ):
            self.assertFalse(
                plugin._ensure_model_runner_patch(strict=False)
            )
            snapshot = runtime_stats().snapshot()
            self.assertEqual(
                snapshot["counters"][
                    "runtime_error_stage:unsupported_version"
                ],
                1,
            )
            with self.assertRaises(
                plugin._UnsupportedCompatibilityError
            ):
                plugin._ensure_model_runner_patch(strict=True)

    def test_register_version_mismatch_keeps_official_backend(
        self,
    ) -> None:
        ascend_package = types.ModuleType("vllm_ascend")
        ascend_package.__path__ = []
        ops_module = types.ModuleType("vllm_ascend.ops")
        attention_package = types.ModuleType(
            "vllm_ascend.attention"
        )
        attention_package.__path__ = []
        attention_module = types.ModuleType(
            "vllm_ascend.attention.attention_v1"
        )
        attention_package.attention_v1 = attention_module
        ascend_package.attention = attention_package
        modules = {
            "vllm_ascend": ascend_package,
            "vllm_ascend.ops": ops_module,
            "vllm_ascend.attention": attention_package,
            "vllm_ascend.attention.attention_v1": (
                attention_module
            ),
        }
        previous_registered = plugin._REGISTERED
        rejection = plugin._UnsupportedCompatibilityError(
            "unsupported_version"
        )
        try:
            plugin._REGISTERED = False
            with (
                patch.dict(sys.modules, modules),
                patch.dict(
                    os.environ,
                    {
                        "VLLM_ASCEND_TRIANGLE_MIX_STRICT": (
                            "false"
                        )
                    },
                ),
                patch.object(
                    plugin,
                    "_patch_attention_module",
                    side_effect=rejection,
                ),
                patch.object(plugin.logger, "warning"),
            ):
                plugin.register()
            self.assertFalse(plugin._REGISTERED)
            snapshot = runtime_stats().snapshot()
            self.assertEqual(
                snapshot["counters"][
                    "runtime_error_stage:unsupported_version"
                ],
                1,
            )

            runtime_stats().snapshot(reset=True)
            plugin._REGISTERED = False
            with (
                patch.dict(sys.modules, modules),
                patch.dict(
                    os.environ,
                    {
                        "VLLM_ASCEND_TRIANGLE_MIX_STRICT": "true"
                    },
                ),
                patch.object(
                    plugin,
                    "_patch_attention_module",
                    side_effect=rejection,
                ),
                self.assertRaises(
                    plugin._UnsupportedCompatibilityError
                ),
            ):
                plugin.register()
            self.assertFalse(plugin._REGISTERED)
        finally:
            plugin._REGISTERED = previous_registered

    def test_metadata_plan_is_built_once_and_reused_across_layers(self) -> None:
        _, Builder, Backend = make_attention_module(legacy_hooks=False)
        config = plugin_config()
        common = make_common()
        query, key_cache, value_cache, output = tensors()
        layer = types.SimpleNamespace(
            layer_name="model.layers.3.self_attn"
        )
        direct = Mock(side_effect=lambda **kwargs: kwargs["output"])
        with (
            patch.object(plugin, "_ensure_native_ready", return_value=True),
            patch.object(
                plugin,
                "build_batch_plan",
                wraps=plugin.build_batch_plan,
            ) as planner,
            patch.object(
                plugin,
                "triangle_direct_paged_attention",
                direct,
            ),
        ):
            metadata = new_builder(Builder, config).build(0, common)
            backend = new_backend(Backend, config)
            first = backend.forward(
                layer,
                query,
                None,
                None,
                (key_cache, value_cache),
                metadata,
                output,
            )
            second = backend.forward(
                layer,
                query,
                None,
                None,
                (key_cache, value_cache),
                metadata,
                output,
            )

        self.assertIs(first, output)
        self.assertIs(second, output)
        self.assertEqual(planner.call_count, 1)
        self.assertEqual(direct.call_count, 2)
        self.assertEqual(backend.official_calls, 0)
        self.assertEqual(key_cache.contiguous_checks, 1)
        self.assertEqual(value_cache.contiguous_checks, 1)
        snapshot = runtime_stats().snapshot()
        self.assertEqual(snapshot["counters"]["request_total"], 1)
        self.assertEqual(
            snapshot["counters"]["request_planner_eligible"],
            1,
        )
        self.assertEqual(snapshot["counters"]["single_launch"], 2)

    def test_final_prompt_length_comes_from_scheduler_metadata(self) -> None:
        _, Builder, _ = make_attention_module(legacy_hooks=False)
        common = make_common(
            query_len=2048,
            seq_len=4096,
            prompt_len=8320,
            state=State.ChunkedPrefill,
        )
        metadata = new_builder(
            Builder,
            plugin_config(),
        ).build(0, common)
        self.assertFalse(hasattr(metadata, "prompt_lens_list"))
        plan = getattr(metadata, plugin._PLAN_ATTR)
        self.assertEqual(plan.requests[0].prompt_len, 8320)
        self.assertEqual(plan.requests[0].q_begin, 2048)
        self.assertTrue(plan.direct)

    def test_common_prompt_cannot_cross_a_new_builder_generation(
        self,
    ) -> None:
        _, Builder, Backend = make_attention_module(legacy_hooks=False)
        config = plugin_config()
        common = make_common(
            query_len=2048,
            seq_len=4096,
            prompt_len=8320,
            state=State.ChunkedPrefill,
        )
        query, key_cache, value_cache, output = tensors(
            query_len=2048
        )
        layer = types.SimpleNamespace(
            layer_name="model.layers.3.self_attn"
        )
        with (
            patch.object(plugin, "_ensure_native_ready", return_value=True),
            patch.object(
                plugin,
                "build_batch_plan",
                wraps=plugin.build_batch_plan,
            ) as planner,
            patch.object(
                plugin,
                "triangle_direct_paged_attention",
            ) as direct,
        ):
            builder = new_builder(Builder, config)
            metadata = builder.build(0, common)
            first_generation = getattr(
                metadata,
                plugin._GENERATION_ATTR,
            )
            first_plan = getattr(metadata, plugin._PLAN_ATTR)
            backend = new_backend(Backend, config)
            backend.forward(
                layer,
                query,
                None,
                None,
                (key_cache, value_cache),
                metadata,
                output,
            )

            common.num_prompt_tokens_cpu = None
            reused = builder.build(0, common)
            second_generation = getattr(
                reused,
                plugin._GENERATION_ATTR,
            )
            backend.forward(
                layer,
                query,
                None,
                None,
                (key_cache, value_cache),
                reused,
                output,
            )

        self.assertIs(reused, metadata)
        self.assertNotEqual(first_generation, second_generation)
        self.assertTrue(first_plan.direct)
        self.assertEqual(first_plan.requests[0].prompt_len, 8320)
        self.assertEqual(planner.call_count, 2)
        self.assertEqual(direct.call_count, 1)
        self.assertEqual(backend.official_calls, 1)
        second_plan = getattr(metadata, plugin._PLAN_ATTR)
        self.assertFalse(second_plan.direct)
        self.assertEqual(
            second_plan.primary_reason,
            FallbackReason.MISSING_METADATA,
        )

    def test_common_prompt_reuses_one_plan_within_one_generation(
        self,
    ) -> None:
        _, Builder, Backend = make_attention_module(legacy_hooks=False)
        config = plugin_config(layers="3,5")
        common = make_common(
            query_len=2048,
            seq_len=4096,
            prompt_len=8320,
            state=State.ChunkedPrefill,
        )
        query, key_cache, value_cache, output = tensors(
            query_len=2048
        )
        with (
            patch.object(plugin, "_ensure_native_ready", return_value=True),
            patch.object(
                plugin,
                "build_batch_plan",
                wraps=plugin.build_batch_plan,
            ) as planner,
            patch.object(
                plugin,
                "triangle_direct_paged_attention",
            ) as direct,
        ):
            metadata = new_builder(Builder, config).build(0, common)
            generation = getattr(metadata, plugin._GENERATION_ATTR)
            for index in (3, 5):
                new_backend(Backend, config).forward(
                    types.SimpleNamespace(
                        layer_name=(
                            f"model.layers.{index}.self_attn"
                        )
                    ),
                    query,
                    None,
                    None,
                    (key_cache, value_cache),
                    metadata,
                    output,
                )

        self.assertEqual(
            getattr(metadata, plugin._GENERATION_ATTR),
            generation,
        )
        self.assertEqual(planner.call_count, 1)
        self.assertEqual(direct.call_count, 2)

    def test_official_metadata_prompt_lengths_take_precedence(self) -> None:
        _, Builder, _ = make_attention_module(legacy_hooks=False)
        common = make_common(
            query_len=2048,
            seq_len=4096,
            prompt_len=8320,
            state=State.ChunkedPrefill,
        )
        common.metadata.prompt_lens_list = [4096]
        metadata = new_builder(
            Builder,
            plugin_config(),
        ).build(0, common)
        plan = getattr(metadata, plugin._PLAN_ATTR)
        self.assertEqual(plan.requests[0].prompt_len, 4096)

    def test_metadata_fingerprint_rebuilds_a_reused_object(self) -> None:
        _, Builder, Backend = make_attention_module(legacy_hooks=False)
        config = plugin_config()
        common = make_common()
        first_query, key_cache, value_cache, first_output = tensors()
        second_query, _, _, second_output = tensors(query_len=1024)
        layer = types.SimpleNamespace(
            layer_name="model.layers.3.self_attn"
        )
        with (
            patch.object(plugin, "_ensure_native_ready", return_value=True),
            patch.object(
                plugin,
                "build_batch_plan",
                wraps=plugin.build_batch_plan,
            ) as planner,
            patch.object(
                plugin,
                "triangle_direct_paged_attention",
            ) as direct,
        ):
            metadata = new_builder(Builder, config).build(0, common)
            backend = new_backend(Backend, config)
            backend.forward(
                layer,
                first_query,
                None,
                None,
                (key_cache, value_cache),
                metadata,
                first_output,
            )
            metadata.actual_seq_lengths_q = [1024]
            metadata.seq_lens_list = [1024]
            backend.forward(
                layer,
                second_query,
                None,
                None,
                (key_cache, value_cache),
                metadata,
                second_output,
            )

        self.assertEqual(planner.call_count, 2)
        self.assertEqual(direct.call_count, 1)
        self.assertEqual(backend.official_calls, 1)
        refreshed = getattr(metadata, plugin._PLAN_ATTR)
        self.assertEqual(refreshed.requests[0].query_len, 1024)
        self.assertEqual(refreshed.requests[0].seq_len, 1024)
        self.assertFalse(refreshed.direct)
        self.assertEqual(
            refreshed.primary_reason,
            FallbackReason.MISSING_METADATA,
        )

    def test_cleared_metadata_prompt_source_fails_closed(self) -> None:
        _, Builder, Backend = make_attention_module(legacy_hooks=False)
        config = plugin_config()
        common = make_common()
        common.metadata.prompt_lens_list = [4096]
        query, key_cache, value_cache, output = tensors()
        layer = types.SimpleNamespace(
            layer_name="model.layers.3.self_attn"
        )
        with (
            patch.object(plugin, "_ensure_native_ready", return_value=True),
            patch.object(
                plugin,
                "build_batch_plan",
                wraps=plugin.build_batch_plan,
            ) as planner,
            patch.object(
                plugin,
                "triangle_direct_paged_attention",
            ) as direct,
        ):
            metadata = new_builder(Builder, config).build(0, common)
            backend = new_backend(Backend, config)
            backend.forward(
                layer,
                query,
                None,
                None,
                (key_cache, value_cache),
                metadata,
                output,
            )
            metadata.prompt_lens_list = None
            backend.forward(
                layer,
                query,
                None,
                None,
                (key_cache, value_cache),
                metadata,
                output,
            )

        self.assertEqual(planner.call_count, 2)
        self.assertEqual(direct.call_count, 1)
        self.assertEqual(backend.official_calls, 1)
        refreshed = getattr(metadata, plugin._PLAN_ATTR)
        self.assertFalse(refreshed.direct)
        self.assertEqual(
            refreshed.primary_reason,
            FallbackReason.MISSING_METADATA,
        )

    def test_clean_upstream_runner_prompt_enables_direct_prefill(
        self,
    ) -> None:
        _, Builder, Backend = make_attention_module(legacy_hooks=False)
        config = plugin_config()
        common = make_common(
            query_len=2048,
            seq_len=4096,
            prompt_len=None,
            state=State.ChunkedPrefill,
        )

        def result_factory() -> object:
            metadata = new_builder(Builder, config).build(0, common)
            return {"model.layers.3.self_attn": metadata}, None

        _, Runner = make_runner_module(result_factory)
        query, key_cache, value_cache, output = tensors(
            query_len=2048
        )
        with (
            patch.object(plugin, "_ensure_native_ready", return_value=True),
            patch.object(
                plugin,
                "build_batch_plan",
                wraps=plugin.build_batch_plan,
            ) as planner,
            patch.object(
                plugin,
                "triangle_direct_paged_attention",
            ) as direct,
        ):
            runner = Runner()
            result = runner._build_attention_metadata(
                2048,
                1,
                2048,
            )
            metadata = result[0]["model.layers.3.self_attn"]
            injected_prompt = getattr(
                metadata,
                plugin._RUNNER_PROMPT_ATTR,
            )
            injected_step = getattr(
                metadata,
                plugin._RUNNER_STEP_ATTR,
            )
            backend = new_backend(Backend, config)
            backend.forward(
                types.SimpleNamespace(
                    layer_name="model.layers.3.self_attn"
                ),
                query,
                None,
                None,
                (key_cache, value_cache),
                metadata,
                output,
            )
            direct_plan = getattr(metadata, plugin._PLAN_ATTR)

            runner.input_batch.num_prompt_tokens_cpu_tensor = None
            repeated = runner._build_attention_metadata(
                2048,
                1,
                2048,
            )
            repeated_metadata = repeated[0][
                "model.layers.3.self_attn"
            ]
            backend.forward(
                types.SimpleNamespace(
                    layer_name="model.layers.3.self_attn"
                ),
                query,
                None,
                None,
                (key_cache, value_cache),
                repeated_metadata,
                output,
            )

        self.assertFalse(hasattr(metadata, "prompt_lens_list"))
        self.assertEqual(injected_prompt, (8320,))
        self.assertIsInstance(injected_step, int)
        self.assertEqual(planner.call_count, 4)
        direct.assert_called_once()
        self.assertEqual(backend.official_calls, 1)
        self.assertTrue(direct_plan.direct)
        self.assertEqual(
            direct_plan.requests[0].prompt_len,
            8320,
        )
        self.assertIs(repeated_metadata, metadata)
        self.assertIsNone(
            getattr(metadata, plugin._RUNNER_PROMPT_ATTR, None)
        )
        fallback_plan = getattr(metadata, plugin._PLAN_ATTR)
        self.assertFalse(fallback_plan.direct)
        self.assertEqual(
            fallback_plan.primary_reason,
            FallbackReason.MISSING_METADATA,
        )

    def test_runner_rejects_batch_ubatch_spec_and_graph(self) -> None:
        metadata = types.SimpleNamespace(
            attn_state=State.ChunkedPrefill,
            actual_seq_lengths_q=[2048],
            seq_lens_list=[4096],
        )
        _, Runner = make_runner_module(
            lambda: ({"layer": metadata}, None)
        )
        runner = Runner()

        def invoke(**overrides: object) -> None:
            arguments: dict[str, object] = {
                "num_tokens": 2048,
                "num_reqs": 1,
                "max_query_len": 2048,
            }
            arguments.update(overrides)
            runner._build_attention_metadata(**arguments)

        cases = (
            {"num_reqs": 2},
            {"ubatch_slices": [slice(0, 1)]},
            {"use_spec_decode": True},
            {"for_cudagraph_capture": True},
        )
        for invalid in cases:
            with self.subTest(invalid=invalid):
                invoke()
                self.assertEqual(
                    getattr(metadata, plugin._RUNNER_PROMPT_ATTR),
                    (8320,),
                )
                invoke(**invalid)
                self.assertIsNone(
                    getattr(
                        metadata,
                        plugin._RUNNER_PROMPT_ATTR,
                        None,
                    )
                )
                self.assertIsNone(
                    getattr(
                        metadata,
                        plugin._RUNNER_STEP_ATTR,
                        None,
                    )
                )

    def test_runner_clears_stale_prompt_when_cpu_field_disappears(
        self,
    ) -> None:
        metadata = types.SimpleNamespace(
            attn_state=State.ChunkedPrefill,
            actual_seq_lengths_q=[2048],
            seq_lens_list=[4096],
        )
        _, Runner = make_runner_module(
            lambda: ({"layer": metadata}, None)
        )
        runner = Runner()
        runner._build_attention_metadata(2048, 1, 2048)
        first_step = getattr(metadata, plugin._RUNNER_STEP_ATTR)
        self.assertEqual(
            getattr(metadata, plugin._RUNNER_PROMPT_ATTR),
            (8320,),
        )

        runner.input_batch.num_prompt_tokens_cpu_tensor = None
        runner._build_attention_metadata(2048, 1, 2048)

        self.assertIsNone(
            getattr(metadata, plugin._RUNNER_PROMPT_ATTR, None)
        )
        self.assertIsNone(
            getattr(metadata, plugin._RUNNER_STEP_ATTR, None)
        )
        self.assertIsInstance(first_step, int)

    def test_request_planner_stats_do_not_depend_on_layer_order(
        self,
    ) -> None:
        def run(order: tuple[str, str]) -> dict[str, object]:
            runtime_stats().snapshot(reset=True)
            _, Builder, Backend = make_attention_module(
                legacy_hooks=False
            )
            config = plugin_config()
            common = make_common()
            query, key_cache, value_cache, output = tensors()
            layer = types.SimpleNamespace(
                layer_name="model.layers.3.self_attn"
            )
            with (
                patch.object(
                    plugin,
                    "_ensure_native_ready",
                    side_effect=[False, True],
                ),
                patch.object(
                    plugin,
                    "triangle_direct_paged_attention",
                ),
            ):
                metadata = new_builder(Builder, config).build(0, common)
                backends = {
                    "fallback": new_backend(Backend, config),
                    "direct": new_backend(Backend, config),
                }
                for name in order:
                    backends[name].forward(
                        layer,
                        query,
                        None,
                        None,
                        (key_cache, value_cache),
                        metadata,
                        output,
                    )
            snapshot = runtime_stats().snapshot()
            counters = snapshot["counters"]
            recent = snapshot["recent"][0]
            execution = recent["execution"]
            request_execution = recent["requests"][0]["execution"]
            return {
                "request": {
                    key: value
                    for key, value in counters.items()
                    if key.startswith("request_")
                },
                "fallback_reasons": snapshot["fallback_reasons"],
                "layer_direct": counters["layer_direct"],
                "layer_fia": counters["layer_fia"],
                "adapter_fia": counters[
                    "layer_fia_reason:adapter_unavailable"
                ],
                "plan_execution": {
                    "observed_route": execution["observed_route"],
                    "layer_direct": execution["layer_direct"],
                    "layer_fia": execution["layer_fia"],
                    "fallback_reasons": execution[
                        "fallback_reasons"
                    ],
                    "single_launch": execution["single_launch"],
                    "estimated_saved_qk": execution[
                        "estimated_saved_qk"
                    ],
                },
                "request_execution": {
                    "observed_route": request_execution[
                        "observed_route"
                    ],
                    "layer_direct": request_execution[
                        "layer_direct"
                    ],
                    "layer_fia": request_execution["layer_fia"],
                    "fallback_reasons": request_execution[
                        "fallback_reasons"
                    ],
                },
                "layer_events": [
                    (
                        event["layer_index"],
                        event["route"],
                        event["reason"],
                        event["request_slots"],
                    )
                    for event in recent["layer_events"]
                ],
            }

        fallback_first = run(("fallback", "direct"))
        direct_first = run(("direct", "fallback"))
        self.assertEqual(fallback_first, direct_first)
        self.assertEqual(
            fallback_first["request"],
            {
                "request_total": 1,
                "request_planner_eligible": 1,
                "request_planner_ineligible": 0,
            },
        )
        self.assertEqual(fallback_first["fallback_reasons"], {})
        self.assertEqual(fallback_first["layer_direct"], 1)
        self.assertEqual(fallback_first["layer_fia"], 1)
        self.assertEqual(fallback_first["adapter_fia"], 1)
        self.assertEqual(
            fallback_first["plan_execution"]["observed_route"],
            "mixed",
        )
        self.assertEqual(
            fallback_first["request_execution"]["fallback_reasons"],
            {"adapter_unavailable": 1},
        )
        self.assertEqual(
            fallback_first["layer_events"],
            [
                (3, "direct", "none", [0]),
                (3, "fia", "adapter_unavailable", [0]),
            ],
        )

    def test_graph_capture_build_reuses_plan_and_forces_fia(self) -> None:
        _, Builder, Backend = make_attention_module(legacy_hooks=False)
        config = plugin_config()
        common = make_common(state=State.ChunkedPrefill)
        query, key_cache, value_cache, output = tensors()
        layer = types.SimpleNamespace(
            layer_name="model.layers.3.self_attn"
        )
        direct = Mock()
        with (
            patch.object(plugin, "_ensure_native_ready", return_value=True),
            patch.object(
                plugin,
                "build_batch_plan",
                wraps=plugin.build_batch_plan,
            ) as planner,
            patch.object(
                plugin,
                "triangle_direct_paged_attention",
                direct,
            ),
        ):
            metadata = new_builder(
                Builder,
                config,
            ).build_for_graph_capture(
                common,
                State.ChunkedPrefill,
            )
            backend = new_backend(Backend, config)
            backend.forward(
                layer,
                query,
                None,
                None,
                (key_cache, value_cache),
                metadata,
                output,
            )

        self.assertEqual(planner.call_count, 1)
        direct.assert_not_called()
        self.assertEqual(backend.official_calls, 1)
        snapshot = runtime_stats().snapshot()
        self.assertEqual(
            snapshot["counters"].get("request_total", 0),
            0,
        )
        self.assertEqual(
            snapshot["counters"]["layer_fia_reason:graph_capture"],
            1,
        )
        self.assertEqual(snapshot["recent"], [])

    def test_prefix_cache_prefill_can_use_direct_paged_kv(self) -> None:
        _, Builder, Backend = make_attention_module(legacy_hooks=False)
        config = plugin_config()
        common = make_common(
            query_len=2048,
            seq_len=4096,
            prompt_len=8320,
            state=State.PrefillCacheHit,
        )
        query, key_cache, value_cache, output = tensors(
            query_len=2048
        )
        direct = Mock(side_effect=lambda **kwargs: kwargs["output"])
        with (
            patch.object(plugin, "_ensure_native_ready", return_value=True),
            patch.object(
                plugin,
                "triangle_direct_paged_attention",
                direct,
            ),
        ):
            metadata = new_builder(Builder, config).build(0, common)
            backend = new_backend(Backend, config)
            backend.forward(
                types.SimpleNamespace(
                    layer_name="model.layers.3.self_attn"
                ),
                query,
                None,
                None,
                (key_cache, value_cache),
                metadata,
                output,
            )

        direct.assert_called_once()
        self.assertEqual(backend.official_calls, 0)
        recent = runtime_stats().snapshot()["recent"][0]
        self.assertEqual(recent["state"], "PrefillCacheHit")
        self.assertEqual(
            recent["requests"][0]["q_begin"],
            2048,
        )
        self.assertEqual(
            recent["requests"][0]["execution"]["observed_route"],
            "all_direct",
        )

    def test_batch_greater_than_one_falls_back_for_every_slot(
        self,
    ) -> None:
        _, Builder, Backend = make_attention_module(legacy_hooks=False)
        config = plugin_config()
        common = make_common(
            query_len=4096,
            seq_len=4096,
            prompt_len=None,
        )
        common.metadata.actual_seq_lengths_q = [2048, 4096]
        common.metadata.seq_lens_list = [2048, 4096]
        common.metadata.num_prefills = 2
        common.metadata.block_tables = FakeTensor(
            (2, 64),
            dtype="torch.int32",
        )
        common.num_prompt_tokens_cpu = [4096, 4096]
        query, key_cache, value_cache, output = tensors()
        with (
            patch.object(plugin, "_ensure_native_ready", return_value=True),
            patch.object(
                plugin,
                "triangle_direct_paged_attention",
            ) as direct,
        ):
            metadata = new_builder(Builder, config).build(0, common)
            backend = new_backend(Backend, config)
            backend.forward(
                types.SimpleNamespace(
                    layer_name="model.layers.3.self_attn"
                ),
                query,
                None,
                None,
                (key_cache, value_cache),
                metadata,
                output,
            )

        direct.assert_not_called()
        self.assertEqual(backend.official_calls, 1)
        snapshot = runtime_stats().snapshot()
        self.assertEqual(snapshot["counters"]["request_total"], 2)
        self.assertEqual(
            snapshot["counters"]["request_planner_ineligible"],
            2,
        )
        recent = snapshot["recent"][0]
        self.assertEqual(
            recent["layer_events"][0]["request_slots"],
            [0, 1],
        )
        for request in recent["requests"]:
            self.assertEqual(
                request["execution"]["observed_route"],
                "all_fia",
            )
            self.assertEqual(
                request["execution"]["fallback_reasons"],
                {"batch_unsupported": 1},
            )

    def test_decode_and_legacy_hook_fallback_to_official_path(self) -> None:
        module, Builder, Backend = make_attention_module(
            legacy_hooks=True
        )
        config = plugin_config()
        common = make_common(
            query_len=1,
            seq_len=4096,
            state=State.DecodeOnly,
        )
        query, key_cache, value_cache, output = tensors(query_len=1)
        with (
            patch.object(
                plugin,
                "_ensure_native_ready",
                return_value=True,
            ),
            patch.object(
                plugin,
                "triangle_direct_paged_attention",
            ) as direct,
        ):
            metadata = new_builder(Builder, config).build(0, common)
            backend = new_backend(Backend, config)
            backend.forward(
                types.SimpleNamespace(
                    layer_name="model.layers.3.self_attn"
                ),
                query,
                None,
                None,
                (key_cache, value_cache),
                metadata,
                output,
            )
        direct.assert_not_called()
        self.assertEqual(backend.official_calls, 1)
        self.assertEqual(backend.legacy_calls, 0)
        self.assertFalse(backend._triangle_mix_layer_enabled)
        module_loader = getattr(
            module,
            "_vllm_ascend_trianglemix_legacy_loader",
            None,
        )
        self.assertIsNotNone(module_loader)
        module_loader.assert_not_called()
        recent = runtime_stats().snapshot()["recent"][0]
        self.assertEqual(
            recent["requests"][0]["execution"]["observed_route"],
            "all_fia",
        )
        self.assertEqual(
            recent["requests"][0]["execution"]["fallback_reasons"],
            {"state_unsupported": 1},
        )

    def test_unsupported_runtime_modes_never_dispatch(self) -> None:
        cases = {
            "tp": FallbackReason.TENSOR_PARALLEL,
            "cp": FallbackReason.CONTEXT_PARALLEL,
            "c8": FallbackReason.MODEL_UNSUPPORTED,
            "noncausal": FallbackReason.NON_CAUSAL,
            "sliding": FallbackReason.MODEL_UNSUPPORTED,
            "geometry": FallbackReason.GEOMETRY_UNSUPPORTED,
        }
        for case, expected_reason in cases.items():
            with self.subTest(case=case):
                runtime_stats().snapshot(reset=True)
                module, Builder, Backend = make_attention_module(
                    legacy_hooks=False
                )
                config = plugin_config(
                    **(
                        {"local_window": 256}
                        if case == "geometry"
                        else {}
                    )
                )
                common = make_common(causal=case != "noncausal")
                query, key_cache, value_cache, output = tensors()
                if case == "tp":
                    module.get_tensor_model_parallel_world_size = (
                        lambda: 2
                    )
                if case == "cp":
                    module.enable_cp = lambda: True
                with (
                    patch.object(
                        plugin,
                        "_ensure_native_ready",
                        return_value=True,
                    ),
                    patch.object(
                        plugin,
                        "triangle_direct_paged_attention",
                    ) as direct,
                ):
                    metadata = new_builder(
                        Builder,
                        config,
                    ).build(0, common)
                    backend = new_backend(Backend, config)
                    if case == "c8":
                        backend.enable_c8_quant = True
                    if case == "sliding":
                        backend.sliding_window = 4096
                    backend.forward(
                        types.SimpleNamespace(
                            layer_name="model.layers.3.self_attn"
                        ),
                        query,
                        None,
                        None,
                        (key_cache, value_cache),
                        metadata,
                        output,
                    )
                direct.assert_not_called()
                self.assertEqual(backend.official_calls, 1)
                snapshot = runtime_stats().snapshot()
                self.assertEqual(
                    snapshot["counters"][
                        f"layer_fia_reason:{expected_reason.value}"
                    ],
                    1,
                )

    def test_block_table_is_revalidated_instead_of_cached(self) -> None:
        _, Builder, Backend = make_attention_module(legacy_hooks=False)
        config = plugin_config()
        common = make_common()
        query, key_cache, value_cache, output = tensors()
        layer = types.SimpleNamespace(
            layer_name="model.layers.3.self_attn"
        )
        with (
            patch.object(plugin, "_ensure_native_ready", return_value=True),
            patch.object(
                plugin,
                "triangle_direct_paged_attention",
            ) as direct,
        ):
            metadata = new_builder(Builder, config).build(0, common)
            backend = new_backend(Backend, config)
            backend.forward(
                layer,
                query,
                None,
                None,
                (key_cache, value_cache),
                metadata,
                output,
            )
            metadata.block_tables.shape = (1, 1)
            backend.forward(
                layer,
                query,
                None,
                None,
                (key_cache, value_cache),
                metadata,
                output,
            )
        self.assertEqual(direct.call_count, 1)
        self.assertEqual(backend.official_calls, 1)
        snapshot = runtime_stats().snapshot()
        self.assertEqual(
            snapshot["counters"][
                "layer_fia_reason:block_table_unsupported"
            ],
            1,
        )

    def test_structured_stats_log_is_worker_local(self) -> None:
        _, Builder, Backend = make_attention_module(legacy_hooks=False)
        config = plugin_config(stats_log_interval=1)
        common = make_common()
        query, key_cache, value_cache, output = tensors()
        with (
            patch.object(plugin, "_ensure_native_ready", return_value=True),
            patch.object(
                plugin,
                "triangle_direct_paged_attention",
            ),
            patch.object(plugin.logger, "info") as info,
        ):
            metadata = new_builder(Builder, config).build(0, common)
            backend = new_backend(Backend, config)
            backend.forward(
                types.SimpleNamespace(
                    layer_name="model.layers.3.self_attn"
                ),
                query,
                None,
                None,
                (key_cache, value_cache),
                metadata,
                output,
            )
        info.assert_called_once()
        self.assertEqual(info.call_args.args[0], "%s")
        payload = json.loads(info.call_args.args[1])
        self.assertEqual(payload["event"], "trianglemix_runtime_stats")
        self.assertEqual(payload["scope"], "worker_local")
        self.assertEqual(payload["request_boundary"], 1)
        self.assertEqual(
            payload["stats"]["counters"]["single_launch"],
            1,
        )
        self.assertNotIn("recent", payload["stats"])

    def test_direct_launch_failure_is_fail_open_by_default(self) -> None:
        _, Builder, Backend = make_attention_module(legacy_hooks=False)
        config = plugin_config()
        common = make_common()
        query, key_cache, value_cache, output = tensors()
        with (
            patch.object(plugin, "_ensure_native_ready", return_value=True),
            patch.object(
                plugin,
                "triangle_direct_paged_attention",
                side_effect=RuntimeError("launch failed"),
            ),
            patch.object(plugin.logger, "exception"),
        ):
            metadata = new_builder(Builder, config).build(0, common)
            backend = new_backend(Backend, config)
            result = backend.forward(
                types.SimpleNamespace(
                    layer_name="model.layers.3.self_attn"
                ),
                query,
                None,
                None,
                (key_cache, value_cache),
                metadata,
                output,
            )
        self.assertIs(result, output)
        self.assertEqual(backend.official_calls, 1)
        snapshot = runtime_stats().snapshot()
        self.assertEqual(
            snapshot["counters"]["runtime_error_stage:direct_launch"],
            1,
        )
        self.assertEqual(
            snapshot["counters"][
                "layer_fia_reason:direct_launch_error"
            ],
            1,
        )
        self.assertEqual(snapshot["fallback_reasons"], {})
        self.assertEqual(
            snapshot["counters"]["request_planner_eligible"],
            1,
        )

    def test_strict_mode_propagates_direct_launch_failure(self) -> None:
        _, Builder, Backend = make_attention_module(legacy_hooks=False)
        config = plugin_config(strict=True)
        common = make_common()
        query, key_cache, value_cache, output = tensors()
        with (
            patch.object(plugin, "_ensure_native_ready", return_value=True),
            patch.object(
                plugin,
                "triangle_direct_paged_attention",
                side_effect=RuntimeError("launch failed"),
            ),
        ):
            metadata = new_builder(Builder, config).build(0, common)
            backend = new_backend(Backend, config)
            with self.assertRaisesRegex(RuntimeError, "launch failed"):
                backend.forward(
                    types.SimpleNamespace(
                        layer_name="model.layers.3.self_attn"
                    ),
                    query,
                    None,
                    None,
                    (key_cache, value_cache),
                    metadata,
                    output,
                )
        self.assertEqual(backend.official_calls, 0)


if __name__ == "__main__":
    unittest.main()
