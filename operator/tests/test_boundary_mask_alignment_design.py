from __future__ import annotations

import math
import unittest
from dataclasses import dataclass
from pathlib import Path


FLOATS_PER_DATA_BLOCK = 8
ROOT = Path(__file__).resolve().parents[1]
EXACT_BOUNDARY_LENGTHS = (
    255,
    256,
    257,
    511,
    512,
    513,
    1023,
    1024,
    1025,
)


@dataclass(frozen=True)
class MaskedBlock:
    base: int
    mask: int


@dataclass(frozen=True)
class RangePlan:
    masked_blocks: tuple[MaskedBlock, ...]
    bulk_ranges: tuple[tuple[int, int], ...]

    def covered_columns(self) -> set[int]:
        covered: set[int] = set()
        for operation in self.masked_blocks:
            for lane in range(FLOATS_PER_DATA_BLOCK):
                if operation.mask & (1 << lane):
                    covered.add(operation.base + lane)
        for begin, end in self.bulk_ranges:
            covered.update(range(begin, end))
        return covered

    def vector_bases(self) -> tuple[int, ...]:
        return tuple(
            operation.base for operation in self.masked_blocks
        ) + tuple(begin for begin, _ in self.bulk_ranges)

    def touched_blocks(self) -> tuple[int, ...]:
        blocks = [
            operation.base for operation in self.masked_blocks
        ]
        for begin, end in self.bulk_ranges:
            blocks.extend(
                range(begin, end, FLOATS_PER_DATA_BLOCK)
            )
        return tuple(blocks)


def align_up_16(value: int) -> int:
    return (value + 15) // 16 * 16


def representative_positions(limit: int) -> tuple[int, ...]:
    """Return edge, alignment, and every 8-lane residue probe."""
    if limit < 0:
        raise ValueError(limit)

    points = {0, limit}
    for base in (
        0,
        (limit // 2) // FLOATS_PER_DATA_BLOCK * FLOATS_PER_DATA_BLOCK,
        max(0, limit - FLOATS_PER_DATA_BLOCK),
    ):
        points.update(
            base + residue
            for residue in range(FLOATS_PER_DATA_BLOCK)
            if base + residue <= limit
        )
    for anchor in (16, 32, 64, 128, 256, 512):
        points.update(
            value
            for value in (anchor - 1, anchor, anchor + 1)
            if 0 <= value <= limit
        )
    return tuple(sorted(points))


def representative_key_counts() -> tuple[int, ...]:
    """Keep all short tails plus page/inner/outer boundary neighborhoods."""
    counts = set(range(1, 33))
    for anchor in (64, 128, 256, 512):
        counts.update(
            value
            for value in range(anchor - 8, anchor + 9)
            if 1 <= value <= 512
        )
    return tuple(sorted(counts))


def contiguous_lane_mask(first_lane: int, lane_count: int) -> int:
    if not 0 <= first_lane < FLOATS_PER_DATA_BLOCK:
        raise ValueError(first_lane)
    if not 1 <= lane_count <= FLOATS_PER_DATA_BLOCK - first_lane:
        raise ValueError((first_lane, lane_count))
    return ((1 << lane_count) - 1) << first_lane


def build_range_plan(begin: int, end: int) -> RangePlan:
    if not 0 <= begin <= end <= 512:
        raise ValueError((begin, end))
    if begin == end:
        return RangePlan((), ())

    first_block = begin // FLOATS_PER_DATA_BLOCK * FLOATS_PER_DATA_BLOCK
    last_block = (
        (end - 1) // FLOATS_PER_DATA_BLOCK * FLOATS_PER_DATA_BLOCK
    )
    if first_block == last_block:
        if (
            begin == first_block
            and end == first_block + FLOATS_PER_DATA_BLOCK
        ):
            return RangePlan((), ((begin, end),))
        return RangePlan(
            (
                MaskedBlock(
                    first_block,
                    contiguous_lane_mask(begin - first_block, end - begin),
                ),
            ),
            (),
        )

    masked: list[MaskedBlock] = []
    middle_begin = begin
    first_lane = begin - first_block
    if first_lane:
        masked.append(
            MaskedBlock(
                first_block,
                contiguous_lane_mask(
                    first_lane,
                    FLOATS_PER_DATA_BLOCK - first_lane,
                ),
            )
        )
        middle_begin = first_block + FLOATS_PER_DATA_BLOCK

    middle_end = end
    last_lane_count = end - last_block
    if last_lane_count != FLOATS_PER_DATA_BLOCK:
        middle_end = last_block
        masked.append(
            MaskedBlock(
                last_block,
                contiguous_lane_mask(0, last_lane_count),
            )
        )

    bulk = (
        ((middle_begin, middle_end),)
        if middle_begin < middle_end
        else ()
    )
    return RangePlan(tuple(masked), bulk)


def build_boundary_plans(
    valid_begin: int,
    valid_end: int,
    key_count: int,
) -> tuple[int, tuple[RangePlan, ...]]:
    if not 0 <= key_count <= 512:
        raise ValueError(key_count)
    copy_columns = align_up_16(key_count)
    valid_begin = min(max(valid_begin, 0), key_count)
    valid_end = min(max(valid_end, 0), key_count)
    if valid_begin >= valid_end:
        return copy_columns, (build_range_plan(0, copy_columns),)
    return copy_columns, (
        build_range_plan(0, valid_begin),
        build_range_plan(valid_end, copy_columns),
    )


def enumerate_softmax_rows(
    query_tokens: int,
    query_heads_in_group: int,
    column_num: int,
) -> list[tuple[int, int, int, int, int]]:
    if not 1 <= query_tokens <= 32:
        raise ValueError(query_tokens)
    if not 1 <= query_heads_in_group <= 4:
        raise ValueError(query_heads_in_group)
    if not 1 <= column_num <= 512:
        raise ValueError(column_num)

    column_num_round = align_up_16(column_num)
    row_num = query_tokens * query_heads_in_group
    subblock_count = 2
    query_heads_first_subblock = (
        query_heads_in_group // subblock_count
    )
    row_split = (
        query_tokens // 2
        if query_heads_in_group == 1
        else query_tokens * query_heads_first_subblock
    )
    max_rows_per_loop = 8192 // column_num_round
    row_num_tile = min(
        max_rows_per_loop // FLOATS_PER_DATA_BLOCK
        * FLOATS_PER_DATA_BLOCK,
        64,
    )

    mapping: list[tuple[int, int, int, int, int]] = []
    for subblock in range(subblock_count):
        row_offset_this_subblock = subblock * row_split
        row_actual_this_subblock = (
            row_num - row_split if subblock == 1 else row_split
        )
        row_loop_count = (
            row_actual_this_subblock + row_num_tile - 1
        ) // row_num_tile
        for row_loop in range(row_loop_count):
            row_offset_cur_loop = row_loop * row_num_tile
            row_num_cur_loop = min(
                row_num_tile,
                row_actual_this_subblock - row_offset_cur_loop,
            )
            for local_row in range(row_num_cur_loop):
                global_row = (
                    row_offset_this_subblock
                    + row_offset_cur_loop
                    + local_row
                )
                mapping.append(
                    (
                        subblock,
                        row_loop,
                        local_row,
                        global_row,
                        global_row % query_tokens,
                    )
                )
    return mapping


def procedural_valid_range(
    *,
    mode: int,
    query_position: int,
    key_begin: int,
    key_count: int,
    window_tokens: int,
) -> tuple[int, int]:
    lower = (
        query_position - window_tokens
        if mode == 2 and query_position > window_tokens
        else 0
    )
    upper = query_position + 1
    valid_begin = min(
        max(lower - key_begin, 0),
        key_count,
    )
    valid_end = min(
        max(upper - key_begin, 0),
        key_count,
    )
    return valid_begin, valid_end


def shared_online_attention(
    tiles: list[list[tuple[float, float]]],
) -> float:
    global_max = -math.inf
    global_sum = 0.0
    global_output = 0.0
    for tile in tiles:
        if not tile:
            continue
        tile_max = max(score for score, _ in tile)
        next_max = max(global_max, tile_max)
        previous_scale = (
            0.0
            if global_max == -math.inf
            else math.exp(global_max - next_max)
        )
        global_sum *= previous_scale
        global_output *= previous_scale
        for score, value in tile:
            probability = math.exp(score - next_max)
            global_sum += probability
            global_output += probability * value
        global_max = next_max
    if global_sum == 0.0:
        raise ValueError("no valid attention entries")
    return global_output / global_sum


class BoundaryMaskAlignmentDesignTest(unittest.TestCase):
    def assert_plan_exact(
        self,
        plan: RangePlan,
        begin: int,
        end: int,
    ) -> None:
        self.assertEqual(
            plan.covered_columns(),
            set(range(begin, end)),
            (begin, end, plan),
        )
        for base in plan.vector_bases():
            self.assertEqual(
                base % FLOATS_PER_DATA_BLOCK,
                0,
                (begin, end, plan),
            )
        touched = plan.touched_blocks()
        self.assertEqual(
            len(touched),
            len(set(touched)),
            (begin, end, plan),
        )

    def test_exhaustive_range_coverage_alignment_and_neighbor_lanes(
        self,
    ) -> None:
        residues: set[int] = set()
        for copy_columns in range(16, 513, 16):
            positions = (
                tuple(range(copy_columns + 1))
                if copy_columns <= 64
                else representative_positions(copy_columns)
            )
            for begin_index, begin in enumerate(positions):
                for end in positions[begin_index:]:
                    plan = build_range_plan(begin, end)
                    residues.add(begin % FLOATS_PER_DATA_BLOCK)
                    residues.add(end % FLOATS_PER_DATA_BLOCK)
                    self.assert_plan_exact(plan, begin, end)

                    original = list(range(copy_columns))
                    actual = original.copy()
                    for column in plan.covered_columns():
                        actual[column] = -1
                    expected = original.copy()
                    expected[begin:end] = [-1] * (end - begin)
                    self.assertEqual(actual, expected)
        self.assertEqual(residues, set(range(FLOATS_PER_DATA_BLOCK)))

    def test_every_key_count_masks_suffix_through_copy_columns(self) -> None:
        for key_count in range(1, 513):
            copy_columns = align_up_16(key_count)
            for valid_end in representative_positions(key_count):
                plan = build_range_plan(valid_end, copy_columns)
                self.assert_plan_exact(
                    plan,
                    valid_end,
                    copy_columns,
                )

    def test_prefix_suffix_and_padding_preserve_only_valid_region(
        self,
    ) -> None:
        key_counts = representative_key_counts()
        self.assertTrue(
            {255, 256, 257, 511, 512}.issubset(key_counts)
        )
        for key_count in key_counts:
            positions = representative_positions(key_count)
            for begin_index, valid_begin in enumerate(positions[:-1]):
                for valid_end in positions[begin_index + 1 :]:
                    copy_columns, plans = build_boundary_plans(
                        valid_begin,
                        valid_end,
                        key_count,
                    )
                    masked: set[int] = set()
                    for plan in plans:
                        masked.update(plan.covered_columns())
                    expected = set(range(valid_begin))
                    expected.update(range(valid_end, copy_columns))
                    self.assertEqual(masked, expected)
                    self.assertTrue(
                        set(range(valid_begin, valid_end)).isdisjoint(
                            masked
                        )
                    )

    def test_empty_or_inverted_valid_range_masks_full_padded_row(self) -> None:
        for key_count in range(1, 513):
            for valid_end in representative_positions(key_count):
                for valid_begin in (
                    valid_end,
                    key_count,
                    key_count + 7,
                ):
                    copy_columns, plans = build_boundary_plans(
                        valid_begin,
                        valid_end,
                        key_count,
                    )
                    masked: set[int] = set()
                    for plan in plans:
                        masked.update(plan.covered_columns())
                    self.assertEqual(
                        masked,
                        set(range(copy_columns)),
                    )

    def test_short_tail_padding_and_cross_page_regressions(self) -> None:
        for key_count in (1, 3):
            copy_columns, plans = build_boundary_plans(
                0,
                key_count,
                key_count,
            )
            self.assertEqual(copy_columns, 16)
            masked: set[int] = set()
            for plan in plans:
                masked.update(plan.covered_columns())
            self.assertEqual(
                masked,
                set(range(key_count, 16)),
            )

        copy_columns, plans = build_boundary_plans(
            valid_begin=0,
            valid_end=127,
            key_count=128,
        )
        self.assertEqual(copy_columns, 128)
        suffix_plan = plans[1]
        self.assertEqual(
            suffix_plan.masked_blocks,
            (MaskedBlock(120, 1 << 7),),
        )
        self.assertEqual(suffix_plan.bulk_ranges, ())
        self.assertEqual(suffix_plan.covered_columns(), {127})

    def test_all_nine_sequence_boundaries_map_to_exact_outer_tiles(
        self,
    ) -> None:
        expected = {
            255: (255,),
            256: (256,),
            257: (257,),
            511: (511,),
            512: (512,),
            513: (512, 1),
            1023: (512, 511),
            1024: (512, 512),
            1025: (512, 512, 1),
        }
        self.assertEqual(
            tuple(expected),
            EXACT_BOUNDARY_LENGTHS,
        )
        for sequence_length, expected_counts in expected.items():
            key_counts = tuple(
                min(512, sequence_length - key_begin)
                for key_begin in range(0, sequence_length, 512)
            )
            self.assertEqual(key_counts, expected_counts)
            self.assertEqual(sum(key_counts), sequence_length)
            for key_count in key_counts:
                copy_columns, plans = build_boundary_plans(
                    0,
                    key_count,
                    key_count,
                )
                masked: set[int] = set()
                for plan in plans:
                    masked.update(plan.covered_columns())
                self.assertEqual(
                    masked,
                    set(range(key_count, copy_columns)),
                )

    def test_runtime_source_fuses_boundary_mask_into_softmax_ub(
        self,
    ) -> None:
        fast_source = (
            ROOT / "op_kernel/triangle_paged_fia_fast_path.h"
        ).read_text()
        softmax_source = (
            ROOT
            / "op_kernel/vendor/cann_9_0_1/block_sparse_attention"
            / "attn_infra/epilogue/block"
            / "block_epilogue_online_softmax.hpp"
        ).read_text()

        self.assertIn(
            "struct ProceduralBoundaryMaskParams",
            softmax_source,
        )
        for field in (
            "uint32_t mode;",
            "uint32_t queryBegin;",
            "uint32_t keyBegin;",
            "uint32_t windowTokens;",
        ):
            self.assertIn(field, softmax_source)
        self.assertIn(
            "const ProceduralBoundaryMaskParams noProceduralMask{",
            softmax_source,
        )
        self.assertIn(
            "softmaxFlag,\n"
            "            noProceduralMask);",
            softmax_source,
        )

        helper_begin = softmax_source.index(
            "void SetNegativeInfinityRangeUb("
        )
        apply_begin = softmax_source.index(
            "void ApplyProceduralBoundaryMaskUb("
        )
        helper_source = softmax_source[helper_begin:apply_begin]
        apply_end = softmax_source.index(
            "void SetBlockReduceMask(", apply_begin
        )
        apply_source = softmax_source[apply_begin:apply_end]
        self.assertIn(
            "firstBlockBegin == lastBlockBegin",
            helper_source,
        )
        self.assertEqual(
            helper_source.count(
                "uint64_t rangeMask[2] = {laneBits, 0ULL};"
            ),
            3,
        )
        for aligned_base in (
            "rowBuffer[firstBlockBegin]",
            "rowBuffer[lastBlockBegin]",
            "rowBuffer[alignedMiddleBegin]",
        ):
            self.assertIn(aligned_base, helper_source)
        self.assertNotIn("SetVectorMask<int8_t>(", helper_source)
        self.assertIn(
            "sUbOffset + localRow * columnNumRound",
            apply_source,
        )
        self.assertIn(
            "rowOffsetThisSubBlock + rowOffsetCurLoop + localRow",
            apply_source,
        )
        self.assertIn(
            "const uint32_t tokenInTile = globalRow % qSBlockSize;",
            apply_source,
        )
        self.assertIn(
            "rowBuffer, validEnd, columnNumRound",
            apply_source,
        )
        self.assertEqual(
            apply_source.count("SetVectorMask<int8_t>("),
            1,
        )
        self.assertLess(
            apply_source.index("SetNegativeInfinityRangeUb("),
            apply_source.index("SetVectorMask<int8_t>("),
        )
        self.assertLess(
            apply_source.index("SetVectorMask<int8_t>("),
            apply_source.index("PipeBarrier<PIPE_V>();"),
        )
        self.assertNotIn("DataCopy", apply_source)
        self.assertNotIn("SetFlag<", apply_source)
        self.assertNotIn("WaitFlag<", apply_source)
        self.assertNotIn(".SetValue(", apply_source)

        fused_begin = softmax_source.index(
            "const ProceduralBoundaryMaskParams &proceduralMask)"
        )
        gmask_begin = softmax_source.index(
            "AscendC::GlobalTensor<ElementMask> gMask",
            fused_begin,
        )
        fused_operator = softmax_source[fused_begin:gmask_begin]
        delayed_begin = fused_operator.index(
            "if (rowLoopIdx >= preLoad)"
        )
        delayed_body = fused_operator[delayed_begin:]
        wait_index = delayed_body.index(
            "WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongFlag);"
        )
        scale_index = delayed_body.index("ScaleS(")
        mask_index = delayed_body.index(
            "ApplyProceduralBoundaryMaskUb("
        )
        compute_index = delayed_body.index(
            "SubCoreCompute<false>("
        )
        self.assertLess(wait_index, scale_index)
        self.assertLess(scale_index, mask_index)
        self.assertLess(mask_index, compute_index)

        gmask_operator = softmax_source[gmask_begin:]
        self.assertIn("SubCoreCompute<true>(", gmask_operator)
        self.assertNotIn(
            "ApplyProceduralBoundaryMaskUb(",
            gmask_operator,
        )

        self.assertNotIn(
            "void SetNegativeInfinityRange(",
            fast_source,
        )
        self.assertNotIn(
            "void ApplyProceduralBoundaryMask(",
            fast_source,
        )
        self.assertNotIn("gScore", fast_source)
        self.assertNotIn(
            "MTE3_MTE2>(EVENT_ID1)",
            fast_source,
        )
        self.assertIn(
            "ProceduralBoundaryMaskParams proceduralMask{",
            fast_source,
        )
        self.assertIn(
            "softmaxReady,\n"
            "                        proceduralMask);",
            fast_source,
        )
        needs_begin = fast_source.index(
            "__aicore__ inline bool NeedsBoundaryMask("
        )
        needs_end = fast_source.index(
            "class TrianglePagedFiaFastPath", needs_begin
        )
        needs_source = fast_source[needs_begin:needs_end]
        self.assertIn("uint32_t windowTokens)", needs_source)
        self.assertNotIn("kWindowTokens", needs_source)
        self.assertIn(
            "qEnd,\n"
            "                        tiling_.localWindow);",
            fast_source,
        )
        self.assertIn(
            "keyBegin,\n"
            "                            tiling_.localWindow};",
            fast_source,
        )

    def test_subblock_and_row_loop_mapping_is_exact(self) -> None:
        for query_tokens in range(1, 33):
            for query_heads_in_group in range(1, 5):
                for column_num in (
                    1,
                    16,
                    17,
                    63,
                    64,
                    65,
                    127,
                    128,
                    129,
                    131,
                    255,
                    256,
                    511,
                    512,
                ):
                    mapping = enumerate_softmax_rows(
                        query_tokens,
                        query_heads_in_group,
                        column_num,
                    )
                    global_rows = [entry[3] for entry in mapping]
                    self.assertEqual(
                        sorted(global_rows),
                        list(
                            range(
                                query_tokens
                                * query_heads_in_group
                            )
                        ),
                    )
                    self.assertEqual(
                        len(global_rows),
                        len(set(global_rows)),
                    )
                    for _, _, _, global_row, token in mapping:
                        self.assertEqual(
                            token,
                            global_row % query_tokens,
                        )

    def test_tq5_cross_page_single_outer_tile_preserves_exact_rows(
        self,
    ) -> None:
        query_begin = 126
        query_tokens = 5
        mapping = enumerate_softmax_rows(
            query_tokens,
            query_heads_in_group=4,
            column_num=131,
        )
        self.assertEqual(
            [entry[4] for entry in mapping],
            list(range(query_tokens)) * 4,
        )

        tiles = ((0, 131),)
        for _, _, _, _, token_in_tile in mapping:
            query_position = query_begin + token_in_tile
            for key_begin, key_count in tiles:
                valid_begin, valid_end = procedural_valid_range(
                    mode=1,
                    query_position=query_position,
                    key_begin=key_begin,
                    key_count=key_count,
                    window_tokens=512,
                )
                column_num_round, plans = build_boundary_plans(
                    valid_begin,
                    valid_end,
                    key_count,
                )
                masked: set[int] = set()
                for plan in plans:
                    masked.update(plan.covered_columns())
                kept = set(range(column_num_round)) - masked
                expected_kept = {
                    key - key_begin
                    for key in range(
                        key_begin,
                        key_begin + key_count,
                    )
                    if key <= query_position
                }
                self.assertEqual(kept, expected_kept)

    def test_nonfirst_fully_masked_tile_leaves_online_state_unchanged(
        self,
    ) -> None:
        first_tile = [
            (-0.75, 1.5),
            (0.25, -2.0),
            (1.0, 0.5),
        ]
        fully_masked_second_tile = [
            (-3.402823466e38, 1000.0),
            (-3.402823466e38, -2000.0),
            (-3.402823466e38, 3000.0),
        ]
        first_only = shared_online_attention([first_tile])
        with_fully_masked_tail = shared_online_attention(
            [first_tile, fully_masked_second_tile]
        )
        self.assertEqual(with_fully_masked_tail, first_only)

    def test_shared_online_softmax_matches_direct_reference(self) -> None:
        query_begin = 766
        query_end = 771
        window_tokens = 512
        intervals = (
            (0, 8, 0),
            (query_begin - window_tokens, query_end, 2),
        )

        for query_position in range(query_begin, query_end):
            tiles: list[list[tuple[float, float]]] = []
            direct_entries: list[tuple[float, float]] = []
            for interval_begin, interval_end, mode in intervals:
                for key_begin in range(
                    interval_begin,
                    interval_end,
                    512,
                ):
                    key_end = min(key_begin + 512, interval_end)
                    tile_entries: list[tuple[float, float]] = []
                    for key_position in range(key_begin, key_end):
                        valid = (
                            key_position <= query_position
                            if mode == 0
                            else (
                                query_position - window_tokens
                                <= key_position
                                <= query_position
                            )
                        )
                        if not valid:
                            continue
                        score = math.sin(
                            (query_position + 1) * 0.013
                            + (key_position + 1) * 0.017
                        )
                        value = math.cos(
                            (query_position + 3) * 0.011
                            - (key_position + 5) * 0.019
                        )
                        tile_entries.append((score, value))
                        direct_entries.append((score, value))
                    tiles.append(tile_entries)

            direct_max = max(score for score, _ in direct_entries)
            direct_weights = [
                math.exp(score - direct_max)
                for score, _ in direct_entries
            ]
            direct = sum(
                weight * value
                for weight, (_, value) in zip(
                    direct_weights,
                    direct_entries,
                )
            ) / sum(direct_weights)
            self.assertAlmostEqual(
                shared_online_attention(tiles),
                direct,
                places=12,
            )


if __name__ == "__main__":
    unittest.main()
