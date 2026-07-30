#!/usr/bin/env python3
"""Run fresh eager/graph engines and gate model-level TriangleMix behavior."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .common import environment_fingerprint, write_json
from .counters import (
    LocatedEvent,
    aggregate_worker_events,
    counter_value,
    extract_stats_events,
)
from .model_smoke_worker import END_MARKER, START_MARKER


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _case(report: dict[str, Any], name: str) -> dict[str, Any] | None:
    for value in report.get("cases", ()):
        if isinstance(value, dict) and value.get("name") == name:
            return value
    return None


def _first_completion(case: dict[str, Any] | None) -> dict[str, Any]:
    if case is None:
        return {}
    value = case.get("completions")
    if not isinstance(value, list) or not value:
        return {}
    first = value[0]
    return first if isinstance(first, dict) else {}


def _integer_list(case: dict[str, Any] | None, name: str) -> list[int]:
    if case is None:
        return []
    value = case.get(name)
    if not isinstance(value, list):
        return []
    return [
        item
        for item in value
        if isinstance(item, int) and not isinstance(item, bool)
    ]


def _string_list(case: dict[str, Any] | None, name: str) -> list[str]:
    if case is None:
        return []
    value = case.get(name)
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        return []
    return list(value)


def _marker_analysis(
    text: str,
) -> tuple[
    dict[str, tuple[int, int]],
    list[dict[str, object]],
    dict[str, dict[str, int]],
]:
    ranges: dict[str, tuple[int, int]] = {}
    errors: list[dict[str, object]] = []
    counts: dict[str, dict[str, int]] = {}
    active: tuple[str, int] | None = None

    def parse_marker(
        line: str,
        prefix: str,
        line_number: int,
        kind: str,
    ) -> str | None:
        position = line.find(prefix)
        if position < 0:
            return None
        try:
            value = json.loads(line[position + len(prefix) :])
            name = value["name"]
            if not isinstance(name, str) or not name:
                raise TypeError("marker name must be a non-empty string")
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            errors.append(
                {
                    "kind": "malformed_marker",
                    "marker": kind,
                    "line": line_number,
                    "error": str(error),
                }
            )
            return ""
        counts.setdefault(name, {"start": 0, "end": 0})[kind] += 1
        return name

    for line_number, line in enumerate(text.splitlines(), start=1):
        start_name = parse_marker(
            line,
            START_MARKER,
            line_number,
            "start",
        )
        if start_name:
            if active is not None:
                errors.append(
                    {
                        "kind": "nested_start_marker",
                        "line": line_number,
                        "name": start_name,
                        "active": active[0],
                    }
                )
            elif counts[start_name]["start"] != 1:
                errors.append(
                    {
                        "kind": "duplicate_start_marker",
                        "line": line_number,
                        "name": start_name,
                    }
                )
            else:
                active = (start_name, line_number)
        end_name = parse_marker(
            line,
            END_MARKER,
            line_number,
            "end",
        )
        if end_name:
            if active is None:
                errors.append(
                    {
                        "kind": "unmatched_end_marker",
                        "line": line_number,
                        "name": end_name,
                    }
                )
            elif active[0] != end_name:
                errors.append(
                    {
                        "kind": "mismatched_end_marker",
                        "line": line_number,
                        "name": end_name,
                        "active": active[0],
                    }
                )
                active = None
            elif counts[end_name]["end"] != 1:
                errors.append(
                    {
                        "kind": "duplicate_end_marker",
                        "line": line_number,
                        "name": end_name,
                    }
                )
                active = None
            elif line_number <= active[1]:
                errors.append(
                    {
                        "kind": "empty_marker_range",
                        "line": line_number,
                        "name": end_name,
                    }
                )
                active = None
            else:
                ranges[end_name] = (active[1], line_number)
                active = None
    if active is not None:
        errors.append(
            {
                "kind": "unmatched_start_marker",
                "line": active[1],
                "name": active[0],
            }
        )
    return ranges, errors, counts


def _reason_count(
    aggregation: dict[str, Any],
    reason: str,
) -> int | float:
    counters = aggregation["aggregate"]["counters"]
    return counters.get(f"layer_fia_reason:{reason}", 0)


def _event_mapping_value(
    event: LocatedEvent | None,
    mapping: str,
    name: str,
) -> int | float:
    if event is None:
        return 0
    stats = event.value.get("stats")
    stats = stats if isinstance(stats, dict) else {}
    values = stats.get(mapping)
    values = values if isinstance(values, dict) else {}
    value = values.get(name, 0)
    return value if isinstance(value, (int, float)) else 0


def _events_by_worker(
    events: list[LocatedEvent],
) -> dict[tuple[str, str, str, str], list[LocatedEvent]]:
    result: dict[tuple[str, str, str, str], list[LocatedEvent]] = {}
    for event in events:
        result.setdefault(event.worker_key, []).append(event)
    return result


def _scheduler_batch_gate(
    events: list[LocatedEvent],
    ranges: dict[str, tuple[int, int]],
    *,
    scenario: str,
    expected_batch: int,
) -> dict[str, Any]:
    bounds = ranges.get(scenario)
    if bounds is None:
        return {
            "passed": False,
            "reason": f"{scenario} markers missing",
            "expected_batch": expected_batch,
        }
    start, end = bounds
    evidence: list[dict[str, Any]] = []
    for worker_key, worker_events in _events_by_worker(events).items():
        before = [
            event for event in worker_events if event.line_number < start
        ]
        baseline = before[-1] if before else None
        baseline_requests = counter_value(baseline, "request_total")
        progressed = [
            event
            for event in worker_events
            if start < event.line_number < end
            and counter_value(event, "request_total") > baseline_requests
        ]
        first = progressed[0] if progressed else None
        request_delta = (
            counter_value(first, "request_total") - baseline_requests
        )
        first_launch_delta = (
            counter_value(first, "single_launch")
            - counter_value(baseline, "single_launch")
        )
        segment_launch_delta = (
            max(
                (
                    counter_value(event, "single_launch")
                    for event in worker_events
                    if start < event.line_number < end
                ),
                default=counter_value(baseline, "single_launch"),
            )
            - counter_value(baseline, "single_launch")
        )
        reason_delta = (
            _event_mapping_value(
                first,
                "fallback_reasons",
                "batch_unsupported",
            )
            - _event_mapping_value(
                baseline,
                "fallback_reasons",
                "batch_unsupported",
            )
        )
        passed = (
            first is not None
            and request_delta == expected_batch
            and first_launch_delta == 0
            and segment_launch_delta == 0
            and (
                reason_delta == expected_batch
                if expected_batch > 1
                else reason_delta == 0
            )
        )
        evidence.append(
            {
                "worker_key": list(worker_key),
                "first_scheduler_event_line": (
                    None if first is None else first.line_number
                ),
                "request_total_delta": request_delta,
                "first_scheduler_single_launch_delta": first_launch_delta,
                "segment_single_launch_delta": segment_launch_delta,
                "fallback_reason": (
                    "batch_unsupported"
                    if expected_batch > 1
                    else None
                ),
                "fallback_reason_request_delta": reason_delta,
                "passed": passed,
            }
        )
    return {
        "passed": len(evidence) == 1 and bool(evidence[0]["passed"]),
        "worker_count": len(evidence),
        "expected_batch": expected_batch,
        "requires_one_scheduler_step": True,
        "expected_custom_launch_delta": 0,
        "expected_fallback_reason": (
            "batch_unsupported" if expected_batch > 1 else None
        ),
        "evidence": evidence,
    }


def _split_scheduler_fallback_gate(
    events: list[LocatedEvent],
    ranges: dict[str, tuple[int, int]],
    *,
    scenario: str,
    expected_batch: int,
) -> dict[str, Any]:
    """Accept a synchronous API that admits a logical batch in steps.

    ``LLM.generate([prompt_a, prompt_b])`` is allowed to submit the two
    requests in adjacent scheduler steps, while ``AsyncLLM`` (used by the
    concurrency suite) proves the stronger one-step ``batch_unsupported``
    contract.  This gate still requires exactly ``expected_batch`` planner
    requests, no custom launch, and an official FIA fallback for every
    request; it never treats a partial or malformed batch as a pass.
    """

    bounds = ranges.get(scenario)
    if bounds is None:
        return {
            "passed": False,
            "reason": f"{scenario} markers missing",
            "expected_batch": expected_batch,
        }
    start, end = bounds
    evidence: list[dict[str, Any]] = []
    for worker_key, worker_events in _events_by_worker(events).items():
        before = [
            event for event in worker_events if event.line_number < start
        ]
        baseline = before[-1] if before else None
        baseline_requests = counter_value(baseline, "request_total")
        inside = [
            event
            for event in worker_events
            if start < event.line_number < end
        ]
        progressed = [
            event
            for event in inside
            if counter_value(event, "request_total") > baseline_requests
        ]
        request_deltas = sorted(
            {
                int(counter_value(event, "request_total"))
                - int(baseline_requests)
                for event in progressed
            }
        )
        final_request_delta = (
            request_deltas[-1] if request_deltas else 0
        )
        final_event = progressed[-1] if progressed else None
        segment_launch_delta = (
            max(
                (
                    counter_value(event, "single_launch")
                    for event in inside
                ),
                default=counter_value(baseline, "single_launch"),
            )
            - counter_value(baseline, "single_launch")
        )
        fallback_total_delta = 0
        if final_event is not None:
            # Fallback reasons live under the cumulative ``stats`` snapshot,
            # not at the top-level event object.  Read both snapshots through
            # the same typed helper used by the other gates so a split
            # scheduler submission still proves one official reason per
            # request.
            final_stats = final_event.value.get("stats")
            final_stats = final_stats if isinstance(final_stats, dict) else {}
            final_fallbacks = final_stats.get("fallback_reasons")
            final_fallbacks = (
                final_fallbacks if isinstance(final_fallbacks, dict) else {}
            )
            baseline_fallbacks = {}
            if baseline is not None:
                baseline_stats = baseline.value.get("stats")
                baseline_stats = (
                    baseline_stats
                    if isinstance(baseline_stats, dict)
                    else {}
                )
                value = baseline_stats.get("fallback_reasons")
                baseline_fallbacks = value if isinstance(value, dict) else {}
            fallback_total_delta = sum(
                max(
                    0,
                    int(value) - int(baseline_fallbacks.get(name, 0)),
                )
                for name, value in final_fallbacks.items()
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
            )
        passed = (
            final_event is not None
            and final_request_delta == expected_batch
            and len(request_deltas) >= expected_batch
            and segment_launch_delta == 0
            and fallback_total_delta >= expected_batch
        )
        evidence.append(
            {
                "worker_key": list(worker_key),
                "request_progress_deltas": request_deltas,
                "request_total_delta": final_request_delta,
                "segment_single_launch_delta": segment_launch_delta,
                "official_fallback_reason_delta": fallback_total_delta,
                "passed": passed,
            }
        )
    return {
        "passed": len(evidence) == 1 and bool(evidence[0]["passed"]),
        "worker_count": len(evidence),
        "expected_batch": expected_batch,
        "requires_one_scheduler_step": False,
        "expected_custom_launch_delta": 0,
        "expected_fallback_reason": "official_fia_any_reason",
        "evidence": evidence,
    }


def _segment_direct_delta_gate(
    events: list[LocatedEvent],
    ranges: dict[str, tuple[int, int]],
    *,
    scenario: str,
) -> dict[str, Any]:
    bounds = ranges.get(scenario)
    if bounds is None:
        return {"passed": False, "reason": f"{scenario} markers missing"}
    start, end = bounds
    evidence: list[dict[str, Any]] = []
    for worker_key, worker_events in _events_by_worker(events).items():
        before = [
            event for event in worker_events if event.line_number < start
        ]
        baseline = before[-1] if before else None
        inside = [
            event
            for event in worker_events
            if start < event.line_number < end
        ]
        maximum = max(
            (
                counter_value(event, "single_launch")
                for event in inside
            ),
            default=counter_value(baseline, "single_launch"),
        )
        delta = maximum - counter_value(baseline, "single_launch")
        evidence.append(
            {
                "worker_key": list(worker_key),
                "single_launch_delta": delta,
                "passed": delta > 0,
            }
        )
    return {
        "passed": len(evidence) == 1 and bool(evidence[0]["passed"]),
        "worker_count": len(evidence),
        "evidence": evidence,
    }


def _decode_official_gate(
    events: list[LocatedEvent],
    ranges: dict[str, tuple[int, int]],
    *,
    expected_decode_steps: int,
) -> dict[str, Any]:
    bounds = ranges.get("sustained_decode")
    if bounds is None:
        return {
            "passed": False,
            "reason": "sustained_decode markers missing",
        }
    start, end = bounds
    evidence: list[dict[str, Any]] = []
    for worker_key, worker_events in _events_by_worker(events).items():
        before = [
            event for event in worker_events if event.line_number < start
        ]
        baseline = before[-1] if before else None
        baseline_reason = _event_mapping_value(
            baseline,
            "fallback_reasons",
            "state_unsupported",
        )
        relevant = [
            event
            for event in worker_events
            if start < event.line_number < end
            and _event_mapping_value(
                event,
                "fallback_reasons",
                "state_unsupported",
            )
            > baseline_reason
        ]
        state_values = sorted(
            {
                int(
                    _event_mapping_value(
                        event,
                        "fallback_reasons",
                        "state_unsupported",
                    )
                )
                for event in relevant
            }
        )
        request_totals = sorted(
            {
                int(counter_value(event, "request_total"))
                for event in relevant
            }
        )
        launches = [
            counter_value(event, "single_launch") for event in relevant
        ]
        final_state = state_values[-1] if state_values else baseline_reason
        state_delta = final_state - baseline_reason
        worker_passed = (
            expected_decode_steps > 0
            and state_delta == expected_decode_steps
            and len(state_values) == expected_decode_steps
            and len(request_totals) == expected_decode_steps
            and bool(launches)
            and len(set(launches)) == 1
        )
        evidence.append(
            {
                "worker_key": list(worker_key),
                "baseline_state_unsupported": baseline_reason,
                "expected_decode_steps": expected_decode_steps,
                "observed_decode_steps": state_delta,
                "distinct_state_unsupported_values": state_values,
                "distinct_request_totals": request_totals,
                "layer_log_event_count": len(relevant),
                "single_launch_values_during_decode": launches,
                "passed": worker_passed,
            }
        )
    return {
        "passed": len(evidence) == 1 and bool(evidence[0]["passed"]),
        "worker_count": len(evidence),
        "definition": (
            "request-level state_unsupported must advance exactly once per "
            "DecodeOnly scheduler step, and cumulative single_launch must "
            "stay constant from the first such step through the scenario"
        ),
        "evidence": evidence,
    }


def _evaluate_mode_impl(
    mode: str,
    worker_report: dict[str, Any],
    log_text: str,
    *,
    chunk_budget: int,
) -> dict[str, Any]:
    events = extract_stats_events(log_text, source=f"{mode}.log")
    aggregation = aggregate_worker_events(events)
    ranges, marker_errors, marker_counts = _marker_analysis(log_text)
    if mode == "eager":
        required_markers = {
            "chunked_prefill",
            "batch2_fallback",
            "sustained_decode",
        }
    elif mode == "graph":
        required_markers = {"graph_capture_replay"}
    else:
        raw_levels = worker_report.get("concurrency_levels")
        marker_levels = raw_levels if isinstance(raw_levels, list) else []
        required_markers = {
            f"concurrency_{int(value)}"
            for value in marker_levels
            if isinstance(value, int)
            and not isinstance(value, bool)
        }
    missing_or_duplicate_markers = [
        {
            "name": name,
            "counts": marker_counts.get(name, {"start": 0, "end": 0}),
            "range_present": name in ranges,
        }
        for name in sorted(required_markers)
        if marker_counts.get(name) != {"start": 1, "end": 1}
        or name not in ranges
    ]
    checks: dict[str, dict[str, Any]] = {
        "worker_completed": {
            "passed": worker_report.get("status") == "PASS",
            "evidence": worker_report.get("status"),
        },
        "worker_stats_observed": {
            "passed": bool(events),
            "evidence": len(events),
        },
        "exactly_one_worker_observed": {
            "passed": aggregation["worker_count"] == 1,
            "evidence": aggregation["worker_count"],
        },
        "scenario_marker_contract": {
            "passed": (
                not marker_errors
                and not missing_or_duplicate_markers
            ),
            "evidence": {
                "parser_errors": marker_errors,
                "required_marker_errors": missing_or_duplicate_markers,
                "counts": marker_counts,
            },
        },
        "counter_snapshots_monotonic": {
            "passed": (
                not aggregation["schema_errors"]
                and not aggregation["counter_regressions"]
                and not aggregation["snapshot_invariant_errors"]
                and not aggregation["aggregate_invariant_errors"]
            ),
            "evidence": {
                "schema_errors": aggregation["schema_errors"],
                "counter_regressions": aggregation[
                    "counter_regressions"
                ],
                "snapshot_invariant_errors": aggregation[
                    "snapshot_invariant_errors"
                ],
                "aggregate_invariant_errors": aggregation[
                    "aggregate_invariant_errors"
                ],
            },
        },
    }
    if mode == "eager":
        chunk = _case(worker_report, "chunked_prefill")
        seed_case = _case(worker_report, "prefix_cache_seed")
        repeated = _case(worker_report, "prefix_cache_repeat")
        shared = _case(worker_report, "prefix_cache_shared")
        batch = _case(worker_report, "batch2_fallback")
        decode = _case(worker_report, "sustained_decode")
        seed_completion = _first_completion(seed_case)
        repeated_completion = _first_completion(repeated)
        shared_completion = _first_completion(shared)
        decode_completion = _first_completion(decode)
        seed_cached = seed_completion.get("cached_tokens")
        repeated_cached = repeated_completion.get("cached_tokens")
        shared_cached = shared_completion.get("cached_tokens")
        prefix_evidence = worker_report.get("prefix_cache_evidence")
        prefix_evidence = (
            prefix_evidence
            if isinstance(prefix_evidence, dict)
            else {}
        )
        block_size = prefix_evidence.get("block_size")
        valid_block_size = (
            isinstance(block_size, int)
            and not isinstance(block_size, bool)
            and block_size > 0
        )
        seed_lengths = _integer_list(seed_case, "prompt_lengths")
        repeated_lengths = _integer_list(repeated, "prompt_lengths")
        shared_lengths = _integer_list(shared, "prompt_lengths")
        seed_prompt_hashes = _string_list(
            seed_case,
            "prompt_token_ids_sha256",
        )
        repeated_prompt_hashes = _string_list(
            repeated,
            "prompt_token_ids_sha256",
        )
        shared_prefix_tokens = prefix_evidence.get(
            "shared_prefix_tokens"
        )
        valid_shared_prefix = (
            valid_block_size
            and isinstance(shared_prefix_tokens, int)
            and not isinstance(shared_prefix_tokens, bool)
            and shared_prefix_tokens > 0
            and shared_prefix_tokens % block_size == 0
        )
        expected_repeat = (
            ((repeated_lengths[0] - 1) // block_size) * block_size
            if valid_block_size
            and len(repeated_lengths) == 1
            and repeated_lengths[0] > 0
            else None
        )
        expected_shared = (
            (shared_prefix_tokens // block_size) * block_size
            if valid_shared_prefix
            else None
        )
        prefix_seed_hashes = prefix_evidence.get(
            "seed_prefix_block_hashes"
        )
        prefix_shared_hashes = prefix_evidence.get(
            "shared_request_prefix_block_hashes"
        )
        seed_suffix_hashes = prefix_evidence.get(
            "seed_suffix_block_hashes"
        )
        shared_suffix_hashes = prefix_evidence.get(
            "shared_request_suffix_block_hashes"
        )
        expected_prefix_blocks = (
            shared_prefix_tokens // block_size
            if valid_shared_prefix
            else None
        )
        prefix_hash_contract = (
            isinstance(prefix_seed_hashes, list)
            and isinstance(prefix_shared_hashes, list)
            and all(isinstance(item, str) for item in prefix_seed_hashes)
            and all(isinstance(item, str) for item in prefix_shared_hashes)
            and prefix_seed_hashes == prefix_shared_hashes
            and len(prefix_seed_hashes) == expected_prefix_blocks
            and prefix_evidence.get("shared_prefix_blocks")
            == expected_prefix_blocks
        )
        suffix_hash_contract = (
            isinstance(seed_suffix_hashes, list)
            and isinstance(shared_suffix_hashes, list)
            and bool(seed_suffix_hashes)
            and bool(shared_suffix_hashes)
            and all(isinstance(item, str) for item in seed_suffix_hashes)
            and all(isinstance(item, str) for item in shared_suffix_hashes)
            and seed_suffix_hashes[0] != shared_suffix_hashes[0]
        )
        seed_repeat_prompt_match = (
            len(seed_lengths) == 1
            and seed_lengths == repeated_lengths
            and len(seed_prompt_hashes) == 1
            and seed_prompt_hashes == repeated_prompt_hashes
        )
        reported_expected_matches = (
            prefix_evidence.get("expected_repeat_cached_tokens")
            == expected_repeat
            and prefix_evidence.get("expected_shared_cached_tokens")
            == expected_shared
        )
        chunk_lengths = _integer_list(chunk, "prompt_lengths")
        chunk_gate = _segment_direct_delta_gate(
            events,
            ranges,
            scenario="chunked_prefill",
        )
        # The synchronous ``LLM.generate([a, b])`` API may admit requests in
        # adjacent scheduler steps even though it represents one logical
        # batch.  The asynchronous concurrency suite separately proves the
        # stronger one-step batch contract.  For this model smoke case,
        # require the split-step-safe gate: exactly two requests, no custom
        # launches, and an official fallback reason for each request.
        batch_gate = _split_scheduler_fallback_gate(
            events,
            ranges,
            scenario="batch2_fallback",
            expected_batch=2,
        )
        decode_tokens = (
            decode.get("max_tokens") if decode is not None else None
        )
        valid_decode_tokens = (
            isinstance(decode_tokens, int)
            and not isinstance(decode_tokens, bool)
            and decode_tokens > 1
        )
        decode_gate = _decode_official_gate(
            events,
            ranges,
            expected_decode_steps=(
                decode_tokens if valid_decode_tokens else -1
            ),
        )
        checks.update(
            {
                "chunked_prefill_exercised": {
                    "passed": (
                        chunk is not None
                        and bool(chunk_lengths)
                        and max(chunk_lengths) > chunk_budget
                        and chunk.get("status") == "PASS"
                        and chunk_gate["passed"]
                    ),
                    "evidence": {
                        "prompt_lengths": chunk_lengths,
                        "chunk_budget": chunk_budget,
                        "segment_gate": chunk_gate,
                    },
                },
                "prefix_cache_seed_is_cold": {
                    "passed": (
                        seed_case is not None
                        and seed_case.get("status") == "PASS"
                        and prefix_evidence.get(
                            "prefix_cache_reset_succeeded"
                        )
                        is True
                        and seed_cached == 0
                        and seed_repeat_prompt_match
                    ),
                    "evidence": {
                        "reset_succeeded": prefix_evidence.get(
                            "prefix_cache_reset_succeeded"
                        ),
                        "seed_cached_tokens": seed_cached,
                        "seed_repeat_prompt_match": (
                            seed_repeat_prompt_match
                        ),
                    },
                },
                "prefix_cache_repeat_hit": {
                    "passed": (
                        repeated is not None
                        and repeated.get("status") == "PASS"
                        and isinstance(repeated_cached, int)
                        and not isinstance(repeated_cached, bool)
                        and expected_repeat is not None
                        and expected_repeat > 0
                        and repeated_cached == expected_repeat
                        and reported_expected_matches
                    ),
                    "evidence": {
                        "observed_cached_tokens": repeated_cached,
                        "expected_cached_tokens": expected_repeat,
                        "expected_derived_by_controller": True,
                        "worker_reported_expected_matches": (
                            reported_expected_matches
                        ),
                        "block_aligned": (
                            isinstance(repeated_cached, int)
                            and not isinstance(repeated_cached, bool)
                            and valid_block_size
                            and repeated_cached % block_size == 0
                        ),
                    },
                },
                "prefix_cache_shared_partial_hit": {
                    "passed": (
                        shared is not None
                        and shared.get("status") == "PASS"
                        and isinstance(shared_cached, int)
                        and not isinstance(shared_cached, bool)
                        and expected_shared is not None
                        and expected_shared > 0
                        and shared_cached == expected_shared
                        and len(shared_lengths) == 1
                        and 0 < shared_cached < shared_lengths[0]
                        and prefix_hash_contract
                        and suffix_hash_contract
                        and reported_expected_matches
                        and prefix_evidence.get("suffixes_are_distinct")
                        is True
                        and prefix_evidence.get("shared_hit_is_partial")
                        is True
                    ),
                    "evidence": {
                        "observed_cached_tokens": shared_cached,
                        "expected_shared_cached_tokens": expected_shared,
                        "expected_derived_by_controller": True,
                        "identical_prefix_block_hashes": prefix_hash_contract,
                        "first_suffix_block_diverged": suffix_hash_contract,
                        "shared_prefix_blocks": prefix_evidence.get(
                            "shared_prefix_blocks"
                        ),
                        "distinct_suffixes": prefix_evidence.get(
                            "suffixes_are_distinct"
                        ),
                        "partial_hit": prefix_evidence.get(
                            "shared_hit_is_partial"
                        ),
                        "scope": prefix_evidence.get(
                            "evidence_scope"
                        ),
                        "physical_block_id_identity": prefix_evidence.get(
                            "physical_block_id_identity"
                        ),
                        "eviction_reallocation": prefix_evidence.get(
                            "eviction_reallocation"
                        ),
                    },
                },
                "batch2_official_fallback": {
                    "passed": (
                        batch is not None
                        and batch.get("request_count") == 2
                        and batch.get("status") == "PASS"
                        and batch_gate["passed"]
                    ),
                    "evidence": batch_gate,
                },
                "sustained_decode_completed": {
                    "passed": (
                        decode is not None
                        and decode.get("status") == "PASS"
                        and valid_decode_tokens
                        and decode_completion.get("output_tokens")
                        == decode_tokens
                    ),
                    "evidence": (
                        None
                        if decode is None
                        else {
                            "requested": decode_tokens,
                            "observed": decode_completion.get(
                                "output_tokens"
                            ),
                        }
                    ),
                },
                "decode_remained_official": decode_gate,
            }
        )
    elif mode == "graph":
        graph_case = _case(worker_report, "graph_capture_replay")
        checks.update(
            {
                "graph_model_request_completed": {
                    "passed": (
                        graph_case is not None
                        and graph_case.get("status") == "PASS"
                    ),
                    "evidence": (
                        None
                        if graph_case is None
                        else graph_case.get("status")
                    ),
                },
                "graph_capture_bypassed_custom_kernel": {
                    "passed": (
                        _reason_count(aggregation, "graph_capture") > 0
                        and aggregation["aggregate"]["counters"].get(
                            "single_launch",
                            0,
                        )
                        == 0
                    ),
                    "evidence": {
                        "layer_fia_reason:graph_capture": _reason_count(
                            aggregation, "graph_capture"
                        ),
                        "single_launch": aggregation["aggregate"][
                            "counters"
                        ].get("single_launch", 0),
                        "meaning": (
                            "capture metadata reached the explicit official "
                            "FIA graph gate"
                        ),
                    },
                },
            }
        )
    else:
        raw_levels = worker_report.get("concurrency_levels")
        levels = (
            list(raw_levels)
            if isinstance(raw_levels, list)
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in raw_levels
            )
            else []
        )
        required_levels = {1, 2, 4, 8, 16}
        matrix: list[dict[str, Any]] = []
        for level in levels:
            case = _case(worker_report, f"concurrency_{level}")
            admission = (
                case.get("admission")
                if isinstance(case, dict)
                and isinstance(case.get("admission"), dict)
                else {}
            )
            admission_passed = (
                admission.get("mode") == "keep"
                and admission.get("cache_cleared") is False
                and admission.get("paused_verified_after_all_adds") is True
                and admission.get("submitted_before_resume") == level
                and admission.get("async_collector_tasks") == level
                and admission.get(
                    "all_requests_submitted_before_resume"
                )
                is True
            )
            gate = _scheduler_batch_gate(
                events,
                ranges,
                scenario=f"concurrency_{level}",
                expected_batch=level,
            )
            case_passed = (
                case is not None
                and case.get("status") == "PASS"
                and case.get("true_async_tasks") == level
                and case.get("async_collector_tasks") == level
                and case.get("request_count") == level
                and admission_passed
                and gate["passed"]
            )
            matrix.append(
                {
                    "concurrency": level,
                    "status": "PASS" if case_passed else "FAIL",
                    "true_async_tasks": (
                        None
                        if case is None
                        else case.get("true_async_tasks")
                    ),
                    "admission": admission,
                    "admission_passed": admission_passed,
                    "scheduler_gate": gate,
                    "crossover_status": (
                        "not_applicable_abi_batch1"
                        if level > 1
                        else "not_measured_in_concurrency_suite"
                    ),
                }
            )
        checks["required_concurrency_levels_present"] = {
            "passed": (
                set(levels) == required_levels
                and len(levels) == len(required_levels)
            ),
            "evidence": {
                "required": sorted(required_levels),
                "observed": levels,
            },
        }
        checks["concurrency_scheduler_matrix"] = {
            "passed": bool(matrix)
            and all(item["status"] == "PASS" for item in matrix),
            "evidence": matrix,
        }
        expected_request_total = sum(levels)
        expected_batch_fallback = sum(
            level for level in levels if level > 1
        )
        checks["concurrency_final_counters_complete"] = {
            "passed": (
                aggregation["aggregate"]["counters"].get(
                    "request_total",
                    0,
                )
                == expected_request_total
                and aggregation["aggregate"]["fallback_reasons"].get(
                    "batch_unsupported",
                    0,
                )
                == expected_batch_fallback
            ),
            "evidence": {
                "expected_request_total": expected_request_total,
                "observed_request_total": aggregation["aggregate"][
                    "counters"
                ].get("request_total", 0),
                "expected_batch_unsupported_requests": (
                    expected_batch_fallback
                ),
                "observed_batch_unsupported_requests": aggregation[
                    "aggregate"
                ]["fallback_reasons"].get("batch_unsupported", 0),
            },
        }
    return {
        "mode": mode,
        "status": (
            "PASS"
            if all(item["passed"] for item in checks.values())
            else "FAIL"
        ),
        "checks": checks,
        "markers": {
            name: list(bounds) for name, bounds in ranges.items()
        },
        "worker_counters": aggregation,
        "concurrency_matrix": (
            matrix if mode == "concurrency" else None
        ),
    }


def evaluate_mode(
    mode: str,
    worker_report: dict[str, Any],
    log_text: str,
    *,
    chunk_budget: int,
) -> dict[str, Any]:
    """Return a structured FAIL instead of propagating malformed evidence."""

    try:
        if not isinstance(worker_report, dict):
            raise TypeError("worker report root must be an object")
        if not isinstance(log_text, str):
            raise TypeError("worker log must be text")
        return _evaluate_mode_impl(
            mode,
            worker_report,
            log_text,
            chunk_budget=chunk_budget,
        )
    except Exception as error:
        return {
            "mode": mode,
            "status": "FAIL",
            "checks": {
                "evaluation_completed": {
                    "passed": False,
                    "evidence": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                }
            },
            "markers": {},
            "worker_counters": None,
            "concurrency_matrix": None,
        }


def _run_worker(
    *,
    args: argparse.Namespace,
    mode: str,
    artifact_dir: Path,
) -> dict[str, Any]:
    worker_output = artifact_dir / f"{mode}.json"
    log_path = artifact_dir / f"{mode}.log"
    command = [
        str(args.python),
        "-m",
        "release_validation.model_smoke_worker",
        "--mode",
        mode,
        "--model",
        str(args.model.resolve()),
        "--wheel",
        str(args.wheel.resolve()),
        "--output",
        str(worker_output),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--chunk-prompt-tokens",
        str(args.chunk_prompt_tokens),
        "--decode-prompt-tokens",
        str(args.decode_prompt_tokens),
        "--decode-tokens",
        str(args.decode_tokens),
        "--prefix-tokens",
        str(args.prefix_tokens),
        "--prefix-suffix-tokens",
        str(args.prefix_suffix_tokens),
        "--batch-prompt-tokens",
        str(args.batch_prompt_tokens),
        "--graph-prompt-tokens",
        str(args.graph_prompt_tokens),
        "--graph-decode-tokens",
        str(args.graph_decode_tokens),
        "--concurrency-levels",
        args.concurrency_levels,
        "--concurrency-prompt-tokens",
        str(args.concurrency_prompt_tokens),
        "--concurrency-token-budget",
        str(args.concurrency_token_budget),
        "--concurrency-max-num-seqs",
        str(args.concurrency_max_num_seqs),
        "--seed",
        str(args.seed),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "VLLM_PLUGINS": "ascend,trianglemix",
            "VLLM_ASCEND_ENABLE_TRIANGLE_MIX": "1",
            "VLLM_ASCEND_TRIANGLE_MIX_LAYERS": args.layers,
            "VLLM_ASCEND_TRIANGLE_MIX_STRICT": "1",
            "VLLM_ASCEND_TRIANGLE_MIX_STATS_LOG_INTERVAL": "1",
        }
    )
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.process_timeout,
            check=False,
        )
        log_text = completed.stdout
        return_code = completed.returncode
        process_error = None
    except subprocess.TimeoutExpired as error:
        value = error.stdout
        log_text = (
            value.decode("utf-8", errors="replace")
            if isinstance(value, bytes)
            else (value or "")
        )
        return_code = None
        process_error = {
            "type": type(error).__name__,
            "message": str(error),
        }
    log_path.write_text(log_text, encoding="utf-8")
    try:
        worker_report = json.loads(
            worker_output.read_text(encoding="utf-8")
        )
        if not isinstance(worker_report, dict):
            raise TypeError("worker report root must be an object")
    except Exception as error:
        worker_report = {
            "status": "FAIL",
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "worker_output": str(worker_output),
            },
        }
    evaluation = evaluate_mode(
        mode,
        worker_report,
        log_text,
        chunk_budget=args.max_num_batched_tokens,
    )
    if return_code != 0:
        evaluation["status"] = "FAIL"
    return {
        "mode": mode,
        "return_code": return_code,
        "command": command,
        "started_unix_seconds": started,
        "finished_unix_seconds": time.time(),
        "worker_output": str(worker_output),
        "log": str(log_path),
        "process_error": process_error,
        "worker_report": worker_report,
        "evaluation": evaluation,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--layers", required=True)
    parser.add_argument(
        "--modes",
        default="eager,graph,concurrency",
        help=(
            "Comma-separated fresh-engine modes; release gate requires "
            "eager, graph, and concurrency."
        ),
    )
    parser.add_argument("--process-timeout", type=float, default=3600.0)
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
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    if set(modes) - {"eager", "graph", "concurrency"} or not modes:
        raise SystemExit(
            "--modes must contain eager, graph, and/or concurrency"
        )
    output_path = args.output.resolve()
    artifact_dir = (
        args.artifacts_dir.resolve()
        if args.artifacts_dir is not None
        else output_path.with_name(f"{output_path.stem}_artifacts")
    )
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise SystemExit(
            f"artifacts directory must be empty: {artifact_dir}"
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    runs = [
        _run_worker(args=args, mode=mode, artifact_dir=artifact_dir)
        for mode in modes
    ]
    release_modes_present = set(modes) == {
        "eager",
        "graph",
        "concurrency",
    }
    passed = (
        release_modes_present
        and all(run["evaluation"]["status"] == "PASS" for run in runs)
    )
    report = {
        "schema_version": 1,
        "suite": "trianglemix_model_release_smoke",
        "status": "PASS" if passed else "FAIL",
        "release_modes_present": release_modes_present,
        "runtime": environment_fingerprint(),
        "model": str(args.model.resolve()),
        "wheel": str(args.wheel.resolve()),
        "layers": args.layers,
        "artifacts_dir": str(artifact_dir),
        "runs": runs,
        "started_unix_seconds": started,
        "finished_unix_seconds": time.time(),
    }
    write_json(output_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "modes": modes,
                "output": str(output_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
