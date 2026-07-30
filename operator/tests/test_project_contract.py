#!/usr/bin/env python3
"""CPU-only contract checks for the direct-paged production operator."""

from __future__ import annotations

import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def build_schedule(
    q_begin: int,
    q_end: int,
    kv_length: int,
    sparse_begin: int,
    sparse_end: int,
    sink_tokens: int = 8,
    window_tokens: int = 512,
) -> tuple[bool, list[tuple[int, int]]]:
    q_end = min(q_end, kv_length)
    if not (q_begin >= sparse_begin and q_end <= sparse_end):
        return True, [] if q_end == 0 else [(0, q_end)]

    sink_end = min(sink_tokens, kv_length)
    local_begin = min(max(0, q_begin - window_tokens), q_end)
    if local_begin <= sink_end:
        return False, [] if q_end == 0 else [(0, q_end)]

    intervals: list[tuple[int, int]] = []
    if sink_end:
        intervals.append((0, sink_end))
    if local_begin < q_end:
        intervals.append((local_begin, q_end))
    return False, intervals


def build_row_schedule(
    query_position: int,
    seq_len: int,
    sparse_begin: int,
    sparse_end: int,
    sink_tokens: int = 8,
    window_tokens: int = 512,
) -> list[tuple[int, int]]:
    causal_end = min(query_position + 1, seq_len)
    if not (sparse_begin <= query_position < sparse_end):
        return [] if causal_end == 0 else [(0, causal_end)]

    sink_end = min(sink_tokens, causal_end)
    local_begin = max(0, query_position - window_tokens)
    if local_begin <= sink_end:
        return [] if causal_end == 0 else [(0, causal_end)]

    intervals: list[tuple[int, int]] = []
    if sink_end:
        intervals.append((0, sink_end))
    if local_begin < causal_end:
        intervals.append((local_begin, causal_end))
    return intervals


class ProjectContractTest(unittest.TestCase):
    def test_schema_is_fixed_qwen3_fast_path(self) -> None:
        schema = json.loads(
            (ROOT / "schema/triangle_paged_sparse_attention.json").read_text()
        )[0]
        self.assertEqual(schema["op"], "TrianglePagedSparseAttention")
        self.assertEqual(
            [(item["name"], item["type"][0]) for item in schema["input_desc"]],
            [
                ("query", "bf16"),
                ("key_cache", "bf16"),
                ("value_cache", "bf16"),
                ("block_table", "int32"),
            ],
        )
        defaults = {
            item["name"]: item.get("default_value") for item in schema["attr"]
        }
        self.assertEqual(defaults["q_tile"], 32)
        self.assertEqual(defaults["page_size"], 128)
        self.assertEqual(defaults["sink_tokens"], 8)
        self.assertEqual(defaults["local_window"], 512)
        self.assertEqual(defaults["dense_tail"], 128)

    def test_triangle_intervals_and_dense_boundaries(self) -> None:
        self.assertEqual(
            build_schedule(0, 32, 8320, 521, 8192),
            (True, [(0, 32)]),
        )
        self.assertEqual(
            build_schedule(544, 576, 8320, 521, 8192),
            (False, [(0, 8), (32, 576)]),
        )
        self.assertEqual(
            build_schedule(1024, 1056, 8320, 521, 8192),
            (False, [(0, 8), (512, 1056)]),
        )
        self.assertEqual(
            build_schedule(8192, 8224, 8320, 521, 8192),
            (True, [(0, 8224)]),
        )

    def test_paged_bsnd_address_formula(self) -> None:
        block_table = [7, 2, 19]
        logical_token = 128 + 11
        kv_head = 3
        page_size = 128
        kv_heads = 8
        head_dim = 128
        logical_page, offset = divmod(logical_token, page_size)
        physical_page = block_table[logical_page]
        element_offset = (
            ((physical_page * page_size + offset) * kv_heads + kv_head)
            * head_dim
        )
        self.assertEqual(logical_page, 1)
        self.assertEqual(physical_page, 2)
        self.assertEqual(element_offset, 273792)

    def test_per_row_sparse_boundary_and_overlap(self) -> None:
        self.assertEqual(
            build_row_schedule(520, 524, 521, 896),
            [(0, 521)],
        )
        self.assertEqual(
            build_row_schedule(521, 524, 521, 896),
            [(0, 8), (9, 522)],
        )
        self.assertEqual(
            build_row_schedule(522, 524, 521, 896),
            [(0, 8), (10, 523)],
        )
        self.assertEqual(
            build_row_schedule(895, 896, 521, 896),
            [(0, 8), (383, 896)],
        )
        self.assertEqual(
            build_row_schedule(896, 897, 521, 896),
            [(0, 897)],
        )

    def test_virtual_axis_and_cube_work_are_exact(self) -> None:
        dense, intervals = build_schedule(544, 576, 8320, 521, 8192)
        self.assertFalse(dense)
        self.assertEqual(intervals, [(0, 8), (32, 576)])
        virtual_tokens = sum(end - begin for begin, end in intervals)
        cube_tokens = 16 + (576 - 32)
        self.assertEqual(virtual_tokens, 552)
        self.assertEqual(cube_tokens, 560)
        self.assertEqual(8 + 512 + 1, 521)

    def test_workspace_v2_fixed_shape_budget(self) -> None:
        rows = 32 * 4
        score_bytes = rows * 512 * 4
        probability_bytes = rows * 512 * 2
        output_tmp_bytes = rows * 128 * 4
        output_update_bytes = output_tmp_bytes
        lse_bytes = rows * 4
        per_core = (
            score_bytes
            + probability_bytes
            + output_tmp_bytes
            + output_update_bytes
            + lse_bytes
        )
        per_core = (per_core + 511) // 512 * 512
        self.assertEqual(rows, 128)
        self.assertEqual(per_core, 524800)

    def test_outer_512_inner_128_memory_and_schedule_budget(self) -> None:
        qk_l1 = 128 * 128 * 2 + 2 * (128 * 128 * 2)
        pv_l1 = 2 * (128 * 512 * 2) + 128 * 512 * 2
        self.assertEqual(qk_l1, 96 * 1024)
        self.assertEqual(pv_l1, 384 * 1024)
        self.assertEqual(qk_l1 + pv_l1, 480 * 1024)
        self.assertLessEqual(qk_l1 + pv_l1, 512 * 1024)

        l0_a = 128 * 128 * 2
        l0_b = 128 * 128 * 2
        l0_c = 128 * 128 * 4
        self.assertLessEqual(l0_a, 64 * 1024 // 2)
        self.assertLessEqual(l0_b, 64 * 1024 // 2)
        self.assertLessEqual(l0_c, 128 * 1024 // 2)

        _, intervals = build_schedule(1024, 1056, 8320, 521, 8192)
        outer_512_tiles = sum(
            (end - begin + 511) // 512 for begin, end in intervals
        )
        outer_256_tiles = sum(
            (end - begin + 255) // 256 for begin, end in intervals
        )
        outer_128_tiles = sum(
            (end - begin + 127) // 128 for begin, end in intervals
        )
        self.assertEqual(outer_128_tiles, 6)
        self.assertEqual(outer_256_tiles, 4)
        self.assertEqual(outer_512_tiles, 3)

    def test_outer_512_exact_sequence_boundaries(self) -> None:
        def split(length: int) -> list[tuple[int, int]]:
            return [
                (begin, min(begin + 512, length))
                for begin in range(0, length, 512)
            ]

        expected = {
            255: [(0, 255)],
            256: [(0, 256)],
            257: [(0, 257)],
            511: [(0, 511)],
            512: [(0, 512)],
            513: [(0, 512), (512, 513)],
            1023: [(0, 512), (512, 1023)],
            1024: [(0, 512), (512, 1024)],
            1025: [(0, 512), (512, 1024), (1024, 1025)],
        }
        for length, tiles in expected.items():
            self.assertEqual(split(length), tiles)

    def test_runtime_is_direct_paged_mix_fast_path(self) -> None:
        host = (ROOT / "op_host/triangle_paged_sparse_attention.cpp").read_text()
        kernel = (
            ROOT / "op_kernel/triangle_paged_sparse_attention.cpp"
        ).read_text()
        fast_path = (
            ROOT / "op_kernel/triangle_paged_fia_fast_path.h"
        ).read_text()
        softmax = (
            ROOT
            / "op_kernel/vendor/cann_9_0_1/block_sparse_attention"
            / "attn_infra/epilogue/block"
            / "block_epilogue_online_softmax.hpp"
        ).read_text()
        paged_mmad = (
            ROOT / "op_kernel/triangle_paged_block_mmad.h"
        ).read_text()

        self.assertIn("kAbiVersion = 2", host)
        self.assertIn("kFastImplementation = 2", host)
        self.assertIn("GetCoreNumAic()", host)
        self.assertNotIn("kReferenceMaxSequenceLength", host)
        self.assertIn("workspacePerCoreBytes", host)
        self.assertIn("platform.GetLibApiWorkSpaceSize()", host)
        self.assertIn(
            "std::numeric_limits<size_t>::max() - userWorkspaceBytes",
            host,
        )
        self.assertIn(
            "workspaceSizes[0] = "
            "libApiWorkspaceBytes + userWorkspaceBytes",
            host,
        )
        self.assertNotIn(
            "workspaceSizes[0] = tiling->workspaceBytes",
            host,
        )
        self.assertIn("return ge::GRAPH_SUCCESS", host)
        self.assertIn("KERNEL_TYPE_MIX_AIC_1_2", kernel)
        self.assertIn("TrianglePagedFiaFastPath", kernel)
        self.assertIn(
            "__gm__ uint8_t *userWorkspace = GetUserWorkspace(workspace);",
            kernel,
        )
        self.assertIn(
            "attention_out,\n        userWorkspace,\n        tilingData",
            kernel,
        )
        self.assertNotIn(
            "attention_out,\n        workspace,\n        tilingData",
            kernel,
        )

        self.assertIn("DirectPagedQkMmad", paged_mmad)
        self.assertIn("DirectPagedPvMmad", paged_mmad)
        self.assertIn("gBlockTable.GetValue(logicalPage)", paged_mmad)
        self.assertIn("pageSize - pageOffset", paged_mmad)
        self.assertIn(
            "CeilDiv<L1TileShape::N>(tokenCount)",
            paged_mmad,
        )
        self.assertIn(
            "nLoop * L1TileShape::N",
            paged_mmad,
        )
        self.assertIn(
            "CeilDiv<L0TileShape::K>(kActual)",
            paged_mmad,
        )
        self.assertIn(
            "kLoop == 0U && kL0Loop == 0U",
            paged_mmad,
        )
        self.assertNotIn("gSelectIdx", paged_mmad)
        self.assertNotIn("Mask2Idx", paged_mmad)

        self.assertIn("queryTileIndex = task / kKvHeads", fast_path)
        self.assertIn("kvHead = task % kKvHeads", fast_path)
        self.assertIn("rows = queryTokens * kGroupSize", fast_path)
        self.assertIn("constexpr uint32_t kKvTile = 512", fast_path)
        self.assertIn(
            "constexpr uint32_t kCubeInnerKvTile = 128",
            fast_path,
        )
        self.assertIn(
            "tiling->kvTile = static_cast<uint32_t>(kKvTile)",
            host,
        )
        self.assertIn("tiling_.kvTile != kKvTile", fast_path)
        self.assertIn("kQkL1Bytes + kPvL1Bytes", fast_path)
        self.assertIn("L0A_PINGPONG_BUF_SIZE", fast_path)
        self.assertIn("L0B_PINGPONG_BUF_SIZE", fast_path)
        self.assertIn("L0C_PINGPONG_BUF_SIZE", fast_path)
        self.assertIn("columnNum == 512", softmax)
        self.assertIn("RowmaxSPECTILE512", softmax)
        self.assertIn("RowsumSPECTILE512", softmax)
        self.assertIn("onlineSoftmax(", fast_path)
        self.assertIn("rescaleOutput(", fast_path)
        self.assertIn("tileOrdinal == 0U", fast_path)
        self.assertNotIn("gSelectIdx", fast_path)
        self.assertNotIn("blockSparseMask", fast_path)
        self.assertNotIn("FULL_MASK", fast_path)


if __name__ == "__main__":
    unittest.main()
