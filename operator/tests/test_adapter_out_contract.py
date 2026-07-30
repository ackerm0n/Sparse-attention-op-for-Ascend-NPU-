#!/usr/bin/env python3
"""Static contracts for the caller-owned Torch adapter overload."""

from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class AdapterOutContractTest(unittest.TestCase):
    def test_mutating_out_schema_and_dispatch_are_registered(self) -> None:
        source = (
            ROOT / "torch_adapter/triangle_paged_attention_torch.cpp"
        ).read_text()
        self.assertIn(
            "TORCH_LIBRARY_FRAGMENT(trianglemix, ops)",
            source,
        )
        self.assertIn(
            '"int prompt_len, float scale, *, Tensor(a!) out) -> '
            'Tensor(a!)"',
            source,
        )
        self.assertIn(
            '"triangle_paged_sparse_attention.out",',
            source,
        )

    def test_out_path_uses_caller_storage_without_allocating(self) -> None:
        source = (
            ROOT / "torch_adapter/triangle_paged_attention_torch.cpp"
        ).read_text()
        begin = source.index(
            "at::Tensor& triangle_paged_sparse_attention_out("
        )
        end = source.index(
            "int64_t triangle_paged_sparse_attention_workspace_size(",
            begin,
        )
        function = source[begin:end]
        self.assertIn(
            "check_output(query, key_cache, value_cache, block_table, output);",
            function,
        )
        self.assertIn(
            "c10::OptionalDeviceGuard device_guard(query.device());",
            function,
        )
        self.assertIn("EXEC_NPU_CMD(", function)
        self.assertIn("return output;", function)
        self.assertNotIn("empty_like", function)
        self.assertNotIn("TensorMove", function)

    def test_adapter_rejects_cross_device_and_bad_output_geometry(self) -> None:
        source = (
            ROOT / "torch_adapter/triangle_paged_attention_torch.cpp"
        ).read_text()
        for contract in (
            "all inputs must be on the same NPU device",
            "out must be on the same NPU device as query",
            "out shape must exactly match query",
            "out must be contiguous",
        ):
            self.assertIn(contract, source)

    def test_adapter_rejects_output_storage_overlap(self) -> None:
        source = (
            ROOT / "torch_adapter/triangle_paged_attention_torch.cpp"
        ).read_text()
        self.assertIn("at::assert_no_internal_overlap(output);", source)
        for read_only_input in (
            "query",
            "key_cache",
            "value_cache",
            "block_table",
        ):
            self.assertIn(
                f"at::assert_no_overlap(output, {read_only_input});",
                source,
            )


if __name__ == "__main__":
    unittest.main()
