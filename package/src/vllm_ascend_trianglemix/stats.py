"""Low-overhead process counters for TriangleMix routing and launches."""

from __future__ import annotations

import os
import threading
from collections import Counter, OrderedDict, deque
from copy import deepcopy
from typing import Any

from .planning import FallbackReason, TriangleBatchPlan


class TriangleMixRuntimeStats:
    _LAYER_EVENT_CAPACITY = 256
    _LOG_DELTA_CAPACITY = 8192

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Counter[str] = Counter()
        self._fallback_reasons: Counter[str] = Counter()
        self._layer_direct: Counter[str] = Counter()
        self._layer_fia: Counter[str] = Counter()
        self._seen_plans: OrderedDict[int, None] = OrderedDict()
        self._recent: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._recent_capacity = 256
        self._seen_capacity = 8192
        self._log_deltas: deque[dict[str, Any]] = deque()
        self._due_log_plan_ids: set[int] = set()
        self._active_log_plan_ids: set[int] = set()
        self._log_interval = 0
        self._next_log_request = 0
        self._last_log_boundary = 0

    def configure(
        self,
        recent_capacity: int,
        log_interval: int = 0,
    ) -> None:
        if recent_capacity < 0 or log_interval < 0:
            raise ValueError(
                "TriangleMix stats configuration must be non-negative"
            )
        with self._lock:
            self._recent_capacity = recent_capacity
            while len(self._recent) > recent_capacity:
                self._recent.popitem(last=False)
            if self._log_interval != log_interval:
                self._log_interval = log_interval
                self._log_deltas.clear()
                self._due_log_plan_ids.clear()
                self._active_log_plan_ids.clear()
                self._last_log_boundary = 0
                request_total = self._counters["request_total"]
                self._next_log_request = (
                    (
                        request_total // log_interval + 1
                    )
                    * log_interval
                    if log_interval
                    else 0
                )

    def _mark_plan(self, plan_id: int) -> bool:
        if plan_id in self._seen_plans:
            return False
        self._seen_plans[plan_id] = None
        if len(self._seen_plans) > self._seen_capacity:
            self._seen_plans.popitem(last=False)
        return True

    def record_planner_decision(
        self,
        plan: TriangleBatchPlan,
    ) -> None:
        """Record the deterministic request-level planner result once.

        Runtime capability and launch outcomes are deliberately excluded:
        those vary by layer/backend instance and belong in the layer
        counters.  Keeping request counters planner-only makes the exported
        result independent of selected-layer traversal order.
        """
        with self._lock:
            if not self._mark_plan(plan.plan_id):
                return
            self._counters["routing_ns"] += plan.routing_ns
            # The planner keeps one synthetic request solely to preserve a
            # reason when all request metadata is absent. It must not become
            # a user-request hit/fallback in exported counters.
            request_count = plan.batch_size
            self._counters["request_total"] += request_count
            requests = plan.requests[:request_count]
            eligible_count = sum(
                int(request.eligible) for request in requests
            )
            ineligible_count = request_count - eligible_count
            self._counters["request_planner_eligible"] += eligible_count
            self._counters["request_planner_ineligible"] += (
                ineligible_count
            )
            for request in requests:
                if not request.eligible:
                    self._fallback_reasons[request.reason.value] += 1
            if self._log_interval:
                self._append_log_delta_locked(
                    {
                        "kind": "planner",
                        "plan_id": plan.plan_id,
                        "state": plan.state_name,
                        "batch_size": plan.batch_size,
                        "route": (
                            "planner_eligible"
                            if plan.direct
                            else "planner_ineligible"
                        ),
                        "reason": plan.primary_reason.value,
                        "requests": [
                            {
                                "slot": request.request_index,
                                "eligible": request.eligible,
                                "reason": request.reason.value,
                            }
                            for request in requests
                        ],
                    }
                )
                if (
                    self._next_log_request
                    and self._counters["request_total"]
                    >= self._next_log_request
                ):
                    self._due_log_plan_ids.add(plan.plan_id)
            if self._recent_capacity:
                recent_requests = [
                    {
                        "slot": request.request_index,
                        "q_begin": request.q_begin,
                        "q_end": request.q_end,
                        "seq_len": request.seq_len,
                        "prompt_len": request.prompt_len,
                        "saved_qk": request.saved_qk,
                        "eligible": request.eligible,
                        "reason": request.reason.value,
                        "execution": self._empty_execution(),
                    }
                    for request in requests
                ]
                self._recent[plan.plan_id] = {
                    "plan_id": plan.plan_id,
                    "state": plan.state_name,
                    "batch_size": plan.batch_size,
                    "route": (
                        "planner_eligible"
                        if plan.direct
                        else "planner_ineligible"
                    ),
                    "reason": plan.primary_reason.value,
                    "execution": self._empty_execution(),
                    "layer_events": [],
                    "requests": recent_requests,
                    "_request_positions": {
                        request["slot"]: index
                        for index, request in enumerate(recent_requests)
                    },
                }
                while len(self._recent) > self._recent_capacity:
                    self._recent.popitem(last=False)

    def record_request_decision(
        self,
        plan: TriangleBatchPlan,
        *,
        direct: bool | None = None,
        reason: FallbackReason | None = None,
    ) -> None:
        """Compatibility alias for callers from pre-release snapshots."""
        del direct, reason
        self.record_planner_decision(plan)

    def record_layer_dispatch(
        self,
        *,
        plan_id: int,
        layer_index: int | None,
        saved_qk: int,
        host_enqueue_ns: int,
        request_index: int = 0,
    ) -> None:
        with self._lock:
            self._counters["layer_direct"] += 1
            self._counters["single_launch"] += 1
            self._counters["estimated_saved_qk"] += saved_qk
            self._counters["host_enqueue_ns"] += host_enqueue_ns
            self._counters["host_enqueue_ns_max"] = max(
                self._counters["host_enqueue_ns_max"],
                host_enqueue_ns,
            )
            self._layer_direct[str(layer_index)] += 1
            self._record_layer_event_locked(
                plan_id=plan_id,
                layer_index=layer_index,
                route="direct",
                reason=FallbackReason.NONE,
                request_indices=(request_index,),
                single_launch=1,
                estimated_saved_qk=saved_qk,
                host_enqueue_ns=host_enqueue_ns,
            )

    def record_layer_fallback(
        self,
        *,
        plan_id: int | None = None,
        layer_index: int | None,
        reason: FallbackReason,
        request_indices: tuple[int, ...] = (),
    ) -> None:
        """Record a selected layer that stayed on the official FIA path."""
        with self._lock:
            self._counters["layer_fia"] += 1
            self._layer_fia[str(layer_index)] += 1
            self._counters[f"layer_fia_reason:{reason.value}"] += 1
            if plan_id is not None:
                self._record_layer_event_locked(
                    plan_id=plan_id,
                    layer_index=layer_index,
                    route="fia",
                    reason=reason,
                    request_indices=request_indices,
                    single_launch=0,
                    estimated_saved_qk=0,
                    host_enqueue_ns=0,
                )

    @staticmethod
    def _empty_execution() -> dict[str, Any]:
        return {
            "observed_route": "unobserved",
            "layer_direct": 0,
            "layer_fia": 0,
            "layer_hit_rate": 0.0,
            "fallback_reasons": {},
            "single_launch": 0,
            "estimated_saved_qk": 0,
            "host_enqueue_ns_total": 0,
            "host_enqueue_ns_max": 0,
            "timing_scope": "host_enqueue_not_device_latency",
            "layer_events_dropped": 0,
        }

    @staticmethod
    def _update_execution(
        execution: dict[str, Any],
        *,
        route: str,
        reason: FallbackReason,
        single_launch: int,
        estimated_saved_qk: int,
        host_enqueue_ns: int,
    ) -> None:
        if route == "direct":
            execution["layer_direct"] += 1
        else:
            execution["layer_fia"] += 1
            reasons = execution["fallback_reasons"]
            reasons[reason.value] = reasons.get(reason.value, 0) + 1
        direct = execution["layer_direct"]
        fia = execution["layer_fia"]
        observed = direct + fia
        execution["layer_hit_rate"] = (
            direct / observed if observed else 0.0
        )
        if direct and fia:
            execution["observed_route"] = "mixed"
        elif direct:
            execution["observed_route"] = "all_direct"
        elif fia:
            execution["observed_route"] = "all_fia"
        execution["single_launch"] += single_launch
        execution["estimated_saved_qk"] += estimated_saved_qk
        execution["host_enqueue_ns_total"] += host_enqueue_ns
        execution["host_enqueue_ns_max"] = max(
            execution["host_enqueue_ns_max"],
            host_enqueue_ns,
        )

    def _record_layer_event_locked(
        self,
        *,
        plan_id: int,
        layer_index: int | None,
        route: str,
        reason: FallbackReason,
        request_indices: tuple[int, ...],
        single_launch: int,
        estimated_saved_qk: int,
        host_enqueue_ns: int,
    ) -> None:
        slots = tuple(sorted(set(request_indices)))
        event = {
            "layer_index": layer_index,
            "route": route,
            "reason": reason.value,
            "request_slots": list(slots),
            "single_launch": single_launch,
            "estimated_saved_qk": estimated_saved_qk,
            "host_enqueue_ns": host_enqueue_ns,
            "timing_scope": "host_enqueue_not_device_latency",
        }
        if self._log_interval and plan_id in self._seen_plans:
            self._append_log_delta_locked(
                {
                    "kind": "layer_execution",
                    "plan_id": plan_id,
                    **event,
                }
            )
        # Structured logging is independent of the in-memory recent trace.
        # Operators commonly disable recent retention in production, and a
        # plan can also be evicted before its later selected layers execute.
        # In both cases the incremental layer event must still be exported.
        entry = self._recent.get(plan_id)
        if entry is None:
            return
        event_dropped = (
            len(entry["layer_events"]) >= self._LAYER_EVENT_CAPACITY
        )
        if not event_dropped:
            entry["layer_events"].append(event)
        else:
            entry["execution"]["layer_events_dropped"] += 1
            self._counters["recent_layer_events_dropped"] += 1
        self._update_execution(
            entry["execution"],
            route=route,
            reason=reason,
            single_launch=single_launch,
            estimated_saved_qk=estimated_saved_qk,
            host_enqueue_ns=host_enqueue_ns,
        )
        request_positions = entry["_request_positions"]
        for slot in slots:
            position = request_positions.get(slot)
            if position is None:
                continue
            request = entry["requests"][position]
            if event_dropped:
                request["execution"]["layer_events_dropped"] += 1
            self._update_execution(
                request["execution"],
                route=route,
                reason=reason,
                single_launch=single_launch,
                estimated_saved_qk=estimated_saved_qk,
                host_enqueue_ns=host_enqueue_ns,
            )

    def _append_log_delta_locked(
        self,
        delta: dict[str, Any],
    ) -> None:
        if len(self._log_deltas) >= self._LOG_DELTA_CAPACITY:
            self._log_deltas.popleft()
            self._counters["structured_log_deltas_dropped"] += 1
        self._log_deltas.append(delta)

    def record_runtime_error(self, *, stage: str) -> None:
        """Count a fail-open error without storing exception text."""
        with self._lock:
            self._counters["runtime_error"] += 1
            self._counters[f"runtime_error_stage:{stage}"] += 1

    def _snapshot_locked(
        self,
        *,
        include_recent: bool,
    ) -> dict[str, Any]:
        request_total = self._counters["request_total"]
        layer_direct = self._counters["layer_direct"]
        result = {
            "counters": dict(self._counters),
            "fallback_reasons": dict(self._fallback_reasons),
            "layers": {
                "direct": dict(self._layer_direct),
                "fia": dict(self._layer_fia),
            },
            "performance": {
                "planner_eligibility_rate": (
                    self._counters["request_planner_eligible"]
                    / request_total
                    if request_total
                    else 0.0
                ),
                "routing_us_per_request": (
                    self._counters["routing_ns"]
                    / request_total
                    / 1_000
                    if request_total
                    else 0.0
                ),
                "host_enqueue_us_per_launch": (
                    self._counters["host_enqueue_ns"]
                    / layer_direct
                    / 1_000
                    if layer_direct
                    else 0.0
                ),
                "host_enqueue_us_max": (
                    self._counters["host_enqueue_ns_max"] / 1_000
                ),
                "host_enqueue_timing_scope": (
                    "host_wall_time_not_device_latency"
                ),
            },
        }
        if include_recent:
            recent = deepcopy(list(self._recent.values()))
            for entry in recent:
                entry.pop("_request_positions", None)
                entry["layer_events"].sort(
                    key=lambda event: (
                        (
                            -1
                            if event["layer_index"] is None
                            else event["layer_index"]
                        ),
                        event["route"],
                        event["reason"],
                        tuple(event["request_slots"]),
                        event["host_enqueue_ns"],
                    )
                )
            result["recent"] = recent
        return result

    def structured_log_if_due(self) -> dict[str, Any] | None:
        """Return compact worker-local counters plus incremental plan events."""
        with self._lock:
            request_total = self._counters["request_total"]
            if not self._log_interval or not self._log_deltas:
                return None
            due = request_total >= self._next_log_request
            pending_plan_ids = {
                int(delta["plan_id"])
                for delta in self._log_deltas
            }
            supplemental = (
                not due
                and bool(self._active_log_plan_ids)
                and pending_plan_ids.issubset(
                    self._active_log_plan_ids
                )
            )
            if not due and not supplemental:
                return None
            if due:
                boundary = (
                    request_total // self._log_interval
                ) * self._log_interval
                self._next_log_request = (
                    boundary + self._log_interval
                )
                self._last_log_boundary = boundary
                self._active_log_plan_ids = set(
                    self._due_log_plan_ids
                )
                self._due_log_plan_ids.clear()
            else:
                boundary = self._last_log_boundary
            recent_delta = deepcopy(list(self._log_deltas))
            self._log_deltas.clear()
            return {
                "event": "trianglemix_runtime_stats",
                "scope": "worker_local",
                "worker": {
                    "pid": os.getpid(),
                    "rank": os.getenv("RANK"),
                    "local_rank": os.getenv("LOCAL_RANK"),
                },
                "request_boundary": boundary,
                "delta_mode": "incremental_not_final_plan_snapshot",
                "recent_delta": recent_delta,
                "stats": self._snapshot_locked(
                    include_recent=False
                ),
            }

    def snapshot(self, reset: bool = False) -> dict[str, Any]:
        with self._lock:
            result = self._snapshot_locked(include_recent=True)
            if reset:
                self._counters.clear()
                self._fallback_reasons.clear()
                self._layer_direct.clear()
                self._layer_fia.clear()
                self._seen_plans.clear()
                self._recent.clear()
                self._log_deltas.clear()
                self._due_log_plan_ids.clear()
                self._active_log_plan_ids.clear()
                self._next_log_request = self._log_interval
                self._last_log_boundary = 0
            return result


_STATS = TriangleMixRuntimeStats()


def stats_snapshot(reset: bool = False) -> dict[str, Any]:
    return _STATS.snapshot(reset=reset)


def runtime_stats() -> TriangleMixRuntimeStats:
    return _STATS
