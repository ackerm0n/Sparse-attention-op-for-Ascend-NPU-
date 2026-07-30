#!/usr/bin/env python3
"""CPU-only regression matrix for the v2 validation oracle."""

from __future__ import annotations

import math
import random
import unittest

from triangle_v2_reference import (
    PRODUCTION_GEOMETRY,
    TriangleGeometry,
    joint_softmax_weighted_sum,
    online_softmax_weighted_sum,
    paged_vector,
    reference_attention_python,
    reference_attention_ragged_python,
    selected_intervals,
    selected_token_indices,
    sparse_end,
)
from v2_crossover_utils import summarize_crossover


def empty_cache(
    pages: int,
    geometry: TriangleGeometry,
) -> list[list[list[list[float]]]]:
    return [
        [
            [
                [0.0 for _ in range(geometry.head_dim)]
                for _ in range(geometry.kv_heads)
            ]
            for _ in range(geometry.page_size)
        ]
        for _ in range(pages)
    ]


def set_logical_vector(
    cache: list[list[list[list[float]]]],
    block_table: list[int],
    logical_token: int,
    kv_head: int,
    vector: list[float],
    geometry: TriangleGeometry,
) -> None:
    logical_page, offset = divmod(logical_token, geometry.page_size)
    cache[block_table[logical_page]][offset][kv_head] = list(vector)


def deterministic_query(
    rows: int,
    geometry: TriangleGeometry,
    seed: int,
) -> list[list[list[float]]]:
    generator = random.Random(seed)
    return [
        [
            [
                generator.uniform(-0.4, 0.4)
                for _ in range(geometry.head_dim)
            ]
            for _ in range(geometry.query_heads)
        ]
        for _ in range(rows)
    ]


def deterministic_cache(
    pages: int,
    geometry: TriangleGeometry,
    seed: int,
) -> list[list[list[list[float]]]]:
    generator = random.Random(seed)
    cache = empty_cache(pages, geometry)
    for page in range(pages):
        for token in range(geometry.page_size):
            for head in range(geometry.kv_heads):
                cache[page][token][head] = [
                    generator.uniform(-0.5, 0.5)
                    for _ in range(geometry.head_dim)
                ]
    return cache


class TriangleV2ReferenceTest(unittest.TestCase):
    def assert_nested_close(
        self,
        left: object,
        right: object,
        *,
        tolerance: float = 1.0e-10,
    ) -> None:
        if isinstance(left, list):
            self.assertIsInstance(right, list)
            assert isinstance(right, list)
            self.assertEqual(len(left), len(right))
            for left_item, right_item in zip(left, right):
                self.assert_nested_close(
                    left_item,
                    right_item,
                    tolerance=tolerance,
                )
            return
        assert isinstance(left, float)
        assert isinstance(right, float)
        self.assertAlmostEqual(left, right, delta=tolerance)

    def test_fixed_sparse_begin_overlap_and_separation(self) -> None:
        geometry = PRODUCTION_GEOMETRY
        self.assertEqual(geometry.sparse_begin, 521)

        # q=520: sink [0,8) and local [8,521) are adjacent and must merge.
        overlap = selected_token_indices(520, 521, 2048, geometry)
        self.assertEqual(overlap, tuple(range(521)))

        # q=521: token 8 is the first genuine gap.  Exactly 8 + 513 keys
        # remain, preserving the paper's fixed long-row workload.
        separated = selected_token_indices(521, 522, 2048, geometry)
        self.assertEqual(separated[:8], tuple(range(8)))
        self.assertEqual(separated[8:], tuple(range(9, 522)))
        self.assertEqual(len(separated), 521)
        self.assertNotIn(8, separated)

    def test_final_128_rows_are_dense(self) -> None:
        geometry = PRODUCTION_GEOMETRY
        prompt_len = 1024
        self.assertEqual(sparse_end(prompt_len, geometry), 896)
        self.assertEqual(
            selected_intervals(895, 1024, prompt_len, geometry),
            ((0, 8), (383, 896)),
        )
        self.assertEqual(
            selected_intervals(896, 1024, prompt_len, geometry),
            ((0, 897),),
        )
        for query_position in range(896, 1024):
            self.assertEqual(
                selected_intervals(
                    query_position,
                    1024,
                    prompt_len,
                    geometry,
                ),
                ((0, query_position + 1),),
            )

    def test_random_noncontiguous_table_and_cross_page_reads(self) -> None:
        geometry = TriangleGeometry(
            query_heads=4,
            kv_heads=1,
            head_dim=2,
            page_size=4,
            q_tile=2,
            sink_tokens=1,
            local_window=2,
            dense_tail=1,
        )
        generator = random.Random(20260728)
        block_table = generator.sample(range(7), 3)
        if block_table == sorted(block_table):
            block_table[0], block_table[-1] = block_table[-1], block_table[0]
        self.assertNotEqual(block_table, sorted(block_table))
        cache = empty_cache(7, geometry)
        for logical_token in range(12):
            vector = [float(logical_token), float(-logical_token)]
            set_logical_vector(
                cache,
                block_table,
                logical_token,
                0,
                vector,
                geometry,
            )

        # These adjacent logical tokens live on different, non-adjacent
        # physical pages.
        self.assertEqual(
            list(paged_vector(cache, block_table, 3, 0, geometry)),
            [3.0, -3.0],
        )
        self.assertEqual(
            list(paged_vector(cache, block_table, 4, 0, geometry)),
            [4.0, -4.0],
        )
        self.assertNotEqual(block_table[0], block_table[1])

        # Repack the same logical vectors into identity pages.  The complete
        # attention result must be invariant to physical page placement.
        identity_table = [0, 1, 2]
        identity_cache = empty_cache(3, geometry)
        for logical_token in range(12):
            set_logical_vector(
                identity_cache,
                identity_table,
                logical_token,
                0,
                [float(logical_token), float(-logical_token)],
                geometry,
            )
        query = [[[0.25, -0.125] for _ in range(geometry.query_heads)]]
        randomized_output = reference_attention_python(
            query,
            cache,
            cache,
            block_table,
            query_start=8,
            seq_len=9,
            prompt_len=16,
            scale=geometry.head_dim**-0.5,
            geometry=geometry,
        )
        identity_output = reference_attention_python(
            query,
            identity_cache,
            identity_cache,
            identity_table,
            query_start=8,
            seq_len=9,
            prompt_len=16,
            scale=geometry.head_dim**-0.5,
            geometry=geometry,
        )
        self.assert_nested_close(randomized_output, identity_output)

    def test_gqa_four_query_heads_share_one_kv_head(self) -> None:
        geometry = TriangleGeometry(
            query_heads=8,
            kv_heads=2,
            head_dim=1,
            page_size=2,
            q_tile=2,
            sink_tokens=1,
            local_window=1,
            dense_tail=1,
        )
        table = [1]
        key = empty_cache(2, geometry)
        value = empty_cache(2, geometry)
        set_logical_vector(value, table, 0, 0, [3.0], geometry)
        set_logical_vector(value, table, 0, 1, [7.0], geometry)
        query = [[[0.0] for _ in range(geometry.query_heads)]]
        output = reference_attention_python(
            query,
            key,
            value,
            table,
            query_start=0,
            seq_len=1,
            prompt_len=4,
            scale=1.0,
            geometry=geometry,
        )
        self.assertEqual(
            [head[0] for head in output[0]],
            [3.0, 3.0, 3.0, 3.0, 7.0, 7.0, 7.0, 7.0],
        )

    def test_sink_and_local_use_one_shared_online_softmax(self) -> None:
        geometry = TriangleGeometry(
            query_heads=4,
            kv_heads=1,
            head_dim=1,
            page_size=4,
            q_tile=2,
            sink_tokens=2,
            local_window=2,
            dense_tail=0,
        )
        self.assertEqual(geometry.sparse_begin, 5)
        table = [1, 0]
        key = empty_cache(2, geometry)
        value = empty_cache(2, geometry)
        for token in (0, 1):
            set_logical_vector(value, table, token, 0, [10.0], geometry)
        # q=5 selects sink [0,2) and local [3,6). Token 2 is poison and must
        # neither be loaded nor affect normalization.
        set_logical_vector(value, table, 2, 0, [1000.0], geometry)
        for token in (3, 4, 5):
            set_logical_vector(value, table, token, 0, [1.0], geometry)
        query = [[[0.0] for _ in range(geometry.query_heads)]]
        output = reference_attention_python(
            query,
            key,
            value,
            table,
            query_start=5,
            seq_len=6,
            prompt_len=20,
            scale=1.0,
            geometry=geometry,
        )
        joint_expected = (2.0 * 10.0 + 3.0 * 1.0) / 5.0
        for head in output[0]:
            self.assertAlmostEqual(head[0], joint_expected)

        online = online_softmax_weighted_sum(
            [
                ([0.0, 0.0], [[10.0], [10.0]]),
                ([0.0, 0.0, 0.0], [[1.0], [1.0], [1.0]]),
            ]
        )
        concatenated = joint_softmax_weighted_sum(
            [0.0] * 5,
            [[10.0], [10.0], [1.0], [1.0], [1.0]],
        )
        self.assert_nested_close(online, concatenated)

        # The forbidden implementation (two independently normalized
        # attentions added together) would produce 10 + 1 == 11.
        independently_normalized = (
            joint_softmax_weighted_sum(
                [0.0, 0.0],
                [[10.0], [10.0]],
            )[0]
            + joint_softmax_weighted_sum(
                [0.0, 0.0, 0.0],
                [[1.0], [1.0], [1.0]],
            )[0]
        )
        self.assertEqual(independently_normalized, 11.0)
        self.assertNotAlmostEqual(independently_normalized, joint_expected)

    def test_query_tile_boundaries_and_ragged_tail_are_row_exact(self) -> None:
        geometry = TriangleGeometry(
            query_heads=4,
            kv_heads=1,
            head_dim=2,
            page_size=4,
            q_tile=4,
            sink_tokens=1,
            local_window=2,
            dense_tail=2,
        )
        table = [2, 0, 3]
        key = deterministic_cache(4, geometry, 11)
        value = deterministic_cache(4, geometry, 12)
        query = deterministic_query(9, geometry, 13)
        full = reference_attention_python(
            query,
            key,
            value,
            table,
            query_start=3,
            seq_len=12,
            prompt_len=16,
            scale=geometry.head_dim**-0.5,
            geometry=geometry,
        )

        # 4 + 4 + 1 covers two exact q_tile boundaries and one ragged tile.
        tiled: list[list[list[float]]] = []
        offset = 0
        for length in (4, 4, 1):
            tiled.extend(
                reference_attention_python(
                    query[offset : offset + length],
                    key,
                    value,
                    table,
                    query_start=3 + offset,
                    seq_len=12,
                    prompt_len=16,
                    scale=geometry.head_dim**-0.5,
                    geometry=geometry,
                )
            )
            offset += length
        self.assert_nested_close(full, tiled)

    def test_packed_ragged_queries_match_batch_one_calls(self) -> None:
        geometry = TriangleGeometry(
            query_heads=4,
            kv_heads=1,
            head_dim=2,
            page_size=4,
            q_tile=4,
            sink_tokens=1,
            local_window=2,
            dense_tail=2,
        )
        key = deterministic_cache(5, geometry, 21)
        value = deterministic_cache(5, geometry, 22)
        query_lengths = [1, 3, 5]
        query_starts = [0, 3, 7]
        seq_lens = [1, 6, 12]
        prompt_lens = [16, 16, 16]
        block_tables = [
            [4],
            [1, 3],
            [2, 0, 4],
        ]
        packed_query = deterministic_query(
            sum(query_lengths),
            geometry,
            23,
        )
        packed_output = reference_attention_ragged_python(
            packed_query,
            key,
            value,
            block_tables,
            query_starts,
            seq_lens,
            prompt_lens,
            query_lengths,
            geometry.head_dim**-0.5,
            geometry,
        )

        separate_output: list[list[list[float]]] = []
        offset = 0
        for index, length in enumerate(query_lengths):
            separate_output.extend(
                reference_attention_python(
                    packed_query[offset : offset + length],
                    key,
                    value,
                    block_tables[index],
                    query_starts[index],
                    seq_lens[index],
                    prompt_lens[index],
                    geometry.head_dim**-0.5,
                    geometry,
                )
            )
            offset += length
        self.assert_nested_close(packed_output, separate_output)

    def test_crossover_requires_entire_larger_query_suffix(self) -> None:
        records = []
        for seq_end in (4096, 8192):
            for query_len, median_win, robust_win in (
                (32, False, False),
                (64, True, False),
                (128, True, True),
                (256, True, True),
            ):
                records.append(
                    {
                        "status": "ok",
                        "seq_end": seq_end,
                        "query_len": query_len,
                        "sparse_faster": median_win,
                        "robust_sparse_faster": robust_win,
                    }
                )
        summary = summarize_crossover(records, [4096, 8192])
        self.assertEqual(
            summary["global_median_continuous_crossover_query_len"],
            64,
        )
        self.assertEqual(
            summary["global_robust_continuous_crossover_query_len"],
            128,
        )
        for seq_end in ("4096", "8192"):
            self.assertEqual(
                summary["by_seq_end"][seq_end][
                    "median_continuous_crossover_query_len"
                ],
                64,
            )
            self.assertEqual(
                summary["by_seq_end"][seq_end][
                    "robust_continuous_crossover_query_len"
                ],
                128,
            )


if __name__ == "__main__":
    unittest.main()
