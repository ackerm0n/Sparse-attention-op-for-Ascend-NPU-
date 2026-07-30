#!/usr/bin/env python3
"""Pure-Python summary helpers for the v2 NPU crossover harness."""

from __future__ import annotations

from typing import Any, Iterable


def continuous_crossover(
    records: Iterable[dict[str, Any]],
    predicate: str,
) -> int | None:
    """First sampled Q length whose entire larger-Q suffix passes."""

    supported = sorted(
        (record for record in records if record.get("status") == "ok"),
        key=lambda record: int(record["query_len"]),
    )
    for index, record in enumerate(supported):
        if all(bool(item[predicate]) for item in supported[index:]):
            return int(record["query_len"])
    return None


def summarize_crossover(
    records: list[dict[str, Any]],
    seq_ends: list[int],
) -> dict[str, Any]:
    """Summarize median and strict repeat-envelope crossovers."""

    by_seq_end: dict[str, dict[str, Any]] = {}
    for seq_end in seq_ends:
        row = [
            record
            for record in records
            if int(record["seq_end"]) == seq_end
        ]
        by_seq_end[str(seq_end)] = {
            "median_continuous_crossover_query_len": continuous_crossover(
                row,
                "sparse_faster",
            ),
            "robust_continuous_crossover_query_len": continuous_crossover(
                row,
                "robust_sparse_faster",
            ),
            "failed_query_lengths": [
                int(record["query_len"])
                for record in row
                if record.get("status") != "ok"
            ],
        }

    supported_lengths = sorted(
        {
            int(record["query_len"])
            for record in records
            if record.get("status") == "ok"
        }
    )

    def global_crossover(predicate: str) -> int | None:
        for index, query_len in enumerate(supported_lengths):
            suffix = set(supported_lengths[index:])
            relevant = [
                record
                for record in records
                if (
                    record.get("status") == "ok"
                    and int(record["query_len"]) in suffix
                )
            ]
            expected = len(seq_ends) * len(suffix)
            if (
                len(relevant) == expected
                and all(bool(record[predicate]) for record in relevant)
            ):
                return query_len
        return None

    return {
        "definition": (
            "The continuous crossover is the first sampled query length that "
            "wins at that point and every larger sampled point. The robust "
            "variant requires min(dense repeats) > max(sparse repeats)."
        ),
        "by_seq_end": by_seq_end,
        "global_median_continuous_crossover_query_len": global_crossover(
            "sparse_faster"
        ),
        "global_robust_continuous_crossover_query_len": global_crossover(
            "robust_sparse_faster"
        ),
    }

