"""Parse cumulative TriangleMix worker statistics from JSONL or mixed logs."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .common import add_numeric_mapping, numeric_mapping, write_json


EVENT_NAME = "trianglemix_runtime_stats"


@dataclass(frozen=True)
class LocatedEvent:
    source: str
    line_number: int
    value: dict[str, Any]

    @property
    def worker_key(self) -> tuple[str, str, str, str]:
        worker = self.value.get("worker")
        worker = worker if isinstance(worker, dict) else {}
        return (
            self.source,
            str(worker.get("pid")),
            str(worker.get("rank")),
            str(worker.get("local_rank")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "line_number": self.line_number,
            "worker_key": list(self.worker_key),
            "value": self.value,
        }


def _objects_in_line(line: str) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(line):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(line[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value
            message = value.get("message")
            if isinstance(message, str) and message != line:
                yield from _objects_in_line(message)


def extract_stats_events(
    text: str,
    *,
    source: str = "<memory>",
) -> list[LocatedEvent]:
    events: list[LocatedEvent] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        seen: set[str] = set()
        for value in _objects_in_line(line):
            if value.get("event") != EVENT_NAME:
                continue
            canonical = json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            if canonical in seen:
                continue
            seen.add(canonical)
            events.append(
                LocatedEvent(
                    source=source,
                    line_number=line_number,
                    value=value,
                )
            )
    return events


def read_stats_events(paths: Iterable[Path]) -> list[LocatedEvent]:
    events: list[LocatedEvent] = []
    for path in paths:
        events.extend(
            extract_stats_events(
                path.read_text(encoding="utf-8", errors="replace"),
                source=str(path.resolve()),
            )
        )
    return events


def _snapshot(event: LocatedEvent) -> dict[str, Any]:
    value = event.value.get("stats")
    return value if isinstance(value, dict) else {}


def _mapping_sum(value: object) -> int | float:
    return sum(numeric_mapping(value).values())


def _snapshot_invariant_errors(
    snapshot: dict[str, Any],
    *,
    location: str,
) -> list[dict[str, object]]:
    counters = numeric_mapping(snapshot.get("counters"))
    reasons = numeric_mapping(snapshot.get("fallback_reasons"))
    layers = snapshot.get("layers")
    layers = layers if isinstance(layers, dict) else {}
    errors: list[dict[str, object]] = []
    mappings = {
        "counters": counters,
        "fallback_reasons": reasons,
        "layers.direct": numeric_mapping(layers.get("direct")),
        "layers.fia": numeric_mapping(layers.get("fia")),
    }
    for mapping_name, values in mappings.items():
        for name, value in values.items():
            if value < 0:
                errors.append(
                    {
                        "kind": "negative_counter",
                        "location": location,
                        "mapping": mapping_name,
                        "counter": name,
                        "value": value,
                    }
                )
    expected = (
        counters.get("request_planner_eligible", 0)
        + counters.get("request_planner_ineligible", 0)
    )
    if counters.get("request_total", 0) != expected:
        errors.append(
            {
                "kind": "request_total_invariant",
                "location": location,
                "request_total": counters.get("request_total", 0),
                "planner_sum": expected,
            }
        )
    if _mapping_sum(reasons) != counters.get(
        "request_planner_ineligible", 0
    ):
        errors.append(
            {
                "kind": "fallback_reason_invariant",
                "location": location,
                "reason_sum": _mapping_sum(reasons),
                "request_planner_ineligible": counters.get(
                    "request_planner_ineligible", 0
                ),
            }
        )
    if counters.get("layer_direct", 0) != counters.get(
        "single_launch", 0
    ):
        errors.append(
            {
                "kind": "single_launch_invariant",
                "location": location,
                "layer_direct": counters.get("layer_direct", 0),
                "single_launch": counters.get("single_launch", 0),
            }
        )
    if _mapping_sum(layers.get("direct")) != counters.get(
        "layer_direct", 0
    ):
        errors.append(
            {
                "kind": "direct_layer_sum_invariant",
                "location": location,
            }
        )
    if _mapping_sum(layers.get("fia")) != counters.get("layer_fia", 0):
        errors.append(
            {
                "kind": "fia_layer_sum_invariant",
                "location": location,
            }
        )
    fia_reason_sum = sum(
        value
        for name, value in counters.items()
        if name.startswith("layer_fia_reason:")
    )
    if fia_reason_sum != counters.get("layer_fia", 0):
        errors.append(
            {
                "kind": "fia_reason_sum_invariant",
                "location": location,
                "reason_sum": fia_reason_sum,
                "layer_fia": counters.get("layer_fia", 0),
            }
        )
    return errors


def _event_schema_errors(event: LocatedEvent) -> list[dict[str, object]]:
    value = event.value
    errors: list[dict[str, object]] = []
    location = f"{event.source}:{event.line_number}"
    if value.get("scope") != "worker_local":
        errors.append(
            {
                "kind": "invalid_scope",
                "location": location,
                "value": value.get("scope"),
            }
        )
    worker = value.get("worker")
    if not isinstance(worker, dict) or not isinstance(
        worker.get("pid") if isinstance(worker, dict) else None,
        int,
    ):
        errors.append(
            {
                "kind": "missing_worker_pid",
                "location": location,
            }
        )
    request_boundary = value.get("request_boundary")
    if (
        not isinstance(request_boundary, int)
        or isinstance(request_boundary, bool)
        or request_boundary < 0
    ):
        errors.append(
            {
                "kind": "invalid_request_boundary",
                "location": location,
            }
        )
    snapshot = value.get("stats")
    if not isinstance(snapshot, dict):
        errors.append(
            {
                "kind": "missing_stats_snapshot",
                "location": location,
            }
        )
        return errors
    for name in ("counters", "fallback_reasons", "layers", "performance"):
        if not isinstance(snapshot.get(name), dict):
            errors.append(
                {
                    "kind": "invalid_snapshot_mapping",
                    "location": location,
                    "mapping": name,
                }
            )
    layers = snapshot.get("layers")
    if isinstance(layers, dict):
        for route in ("direct", "fia"):
            if not isinstance(layers.get(route), dict):
                errors.append(
                    {
                        "kind": "invalid_layer_mapping",
                        "location": location,
                        "mapping": f"layers.{route}",
                    }
                )
    numeric_mappings = {
        "counters": snapshot.get("counters"),
        "fallback_reasons": snapshot.get("fallback_reasons"),
        "layers.direct": (
            layers.get("direct") if isinstance(layers, dict) else None
        ),
        "layers.fia": (
            layers.get("fia") if isinstance(layers, dict) else None
        ),
    }
    for mapping_name, mapping in numeric_mappings.items():
        if not isinstance(mapping, dict):
            continue
        for key, item in mapping.items():
            if (
                not isinstance(key, str)
                or not isinstance(item, (int, float))
                or isinstance(item, bool)
                or not math.isfinite(float(item))
            ):
                errors.append(
                    {
                        "kind": "invalid_counter_value",
                        "location": location,
                        "mapping": mapping_name,
                        "counter": str(key),
                        "value_type": type(item).__name__,
                    }
                )
    return errors


def _counter_regressions(
    events: Iterable[LocatedEvent],
) -> list[dict[str, object]]:
    prior: dict[tuple[str, str, str, str], dict[str, int | float]] = {}
    prior_boundaries: dict[tuple[str, str, str, str], int] = {}
    regressions: list[dict[str, object]] = []
    for event in events:
        current = numeric_mapping(_snapshot(event).get("counters"))
        previous = prior.get(event.worker_key, {})
        boundary = event.value.get("request_boundary")
        previous_boundary = prior_boundaries.get(event.worker_key)
        if (
            isinstance(boundary, int)
            and not isinstance(boundary, bool)
            and previous_boundary is not None
            and boundary < previous_boundary
        ):
            regressions.append(
                {
                    "worker_key": list(event.worker_key),
                    "source": event.source,
                    "line_number": event.line_number,
                    "counter": "request_boundary",
                    "previous": previous_boundary,
                    "current": boundary,
                }
            )
        for name in set(previous) | set(current):
            value = current.get(name, 0)
            if value < previous.get(name, 0):
                regressions.append(
                    {
                        "worker_key": list(event.worker_key),
                        "source": event.source,
                        "line_number": event.line_number,
                        "counter": name,
                        "previous": previous.get(name, 0),
                        "current": value,
                    }
                )
        prior[event.worker_key] = current
        if isinstance(boundary, int) and not isinstance(boundary, bool):
            prior_boundaries[event.worker_key] = boundary
    return regressions


def aggregate_worker_events(
    events: Iterable[LocatedEvent],
) -> dict[str, Any]:
    ordered = list(events)
    latest: dict[tuple[str, str, str, str], LocatedEvent] = {}
    for event in ordered:
        latest[event.worker_key] = event
    schema_errors = [
        error
        for event in ordered
        for error in _event_schema_errors(event)
    ]
    snapshot_invariant_errors = [
        error
        for event in ordered
        for error in _snapshot_invariant_errors(
            _snapshot(event),
            location=f"{event.source}:{event.line_number}",
        )
    ]

    counters: dict[str, int | float] = {}
    fallback_reasons: dict[str, int | float] = {}
    direct_layers: dict[str, int | float] = {}
    fia_layers: dict[str, int | float] = {}
    workers: list[dict[str, Any]] = []
    for key, event in sorted(latest.items()):
        snapshot = _snapshot(event)
        layers = snapshot.get("layers")
        layers = layers if isinstance(layers, dict) else {}
        add_numeric_mapping(counters, snapshot.get("counters"))
        add_numeric_mapping(
            fallback_reasons,
            snapshot.get("fallback_reasons"),
        )
        add_numeric_mapping(direct_layers, layers.get("direct"))
        add_numeric_mapping(fia_layers, layers.get("fia"))
        workers.append(
            {
                "worker_key": list(key),
                "source": event.source,
                "line_number": event.line_number,
                "worker": event.value.get("worker"),
                "request_boundary": event.value.get("request_boundary"),
                "stats": snapshot,
            }
        )

    request_total = float(counters.get("request_total", 0))
    layer_direct = float(counters.get("layer_direct", 0))
    routing_ns = float(counters.get("routing_ns", 0))
    enqueue_ns = float(counters.get("host_enqueue_ns", 0))
    aggregate = {
        "counters": counters,
        "fallback_reasons": fallback_reasons,
        "layers": {
            "direct": direct_layers,
            "fia": fia_layers,
        },
        "performance": {
            "planner_eligibility_rate": (
                float(counters.get("request_planner_eligible", 0))
                / request_total
                if request_total
                else 0.0
            ),
            "routing_us_per_request": (
                routing_ns / request_total / 1_000
                if request_total
                else 0.0
            ),
            "host_enqueue_us_per_launch": (
                enqueue_ns / layer_direct / 1_000
                if layer_direct
                else 0.0
            ),
        },
    }
    aggregate_invariant_errors = _snapshot_invariant_errors(
        aggregate,
        location="<aggregate>",
    )
    return {
        "schema_version": 1,
        "event": EVENT_NAME,
        "scope": "sum_of_latest_worker_local_snapshots",
        "event_count": len(ordered),
        "worker_count": len(latest),
        "schema_errors": schema_errors,
        "counter_regressions": _counter_regressions(ordered),
        "snapshot_invariant_errors": snapshot_invariant_errors,
        "aggregate_invariant_errors": aggregate_invariant_errors,
        "aggregate": aggregate,
        "workers": workers,
    }


def counter_value(
    event: LocatedEvent | None,
    name: str,
) -> int | float:
    if event is None:
        return 0
    return numeric_mapping(_snapshot(event).get("counters")).get(name, 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    events = read_stats_events(args.logs)
    report = aggregate_worker_events(events)
    report["sources"] = [str(path.resolve()) for path in args.logs]
    report["status"] = (
        "PASS"
        if (
            events
            and not report["schema_errors"]
            and not report["counter_regressions"]
            and not report["snapshot_invariant_errors"]
            and not report["aggregate_invariant_errors"]
        )
        else "FAIL"
    )
    write_json(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "events": report["event_count"],
                "workers": report["worker_count"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
