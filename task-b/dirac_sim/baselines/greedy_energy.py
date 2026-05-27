"""
dirac_sim.baselines.greedy_energy
==================================
Greedy energy-aware scheduler.

Same logic as GreedyCarbonScheduler but optimises energy_wh instead of
cfp_g.  Dispatch immediately to the lowest-energy site, or hold up to
`max_hold_hours` if a significantly lower-energy window exists.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from dirac_sim.core.job_queue import Job, JobQueue
from dirac_sim.core.scheduler import (
    DispatchDecision, DispatchPlan, ForecastBundle, Objective, Scheduler,
)
from dirac_sim.core.site_model import Site, SiteRegistry

logger = logging.getLogger(__name__)


class GreedyEnergyScheduler(Scheduler):
    """
    Greedy energy-aware scheduler.

    Parameters
    ----------
    declared_objective      : Primary objective (default: ENERGY).
    max_hold_hours          : Max hours to defer a job for a lower-energy window.
    energy_saving_threshold : Minimum fractional energy saving to justify hold.
    min_deadline_slack_h    : Minimum deadline slack (hours) required to hold.
    """

    def __init__(
        self,
        declared_objective: str = "energy",
        max_hold_hours: float = 6.0,
        energy_saving_threshold: float = 0.15,
        min_deadline_slack_h: float = 2.0,
    ) -> None:
        obj = Objective(declared_objective) if isinstance(declared_objective, str) else declared_objective
        super().__init__(declared_objective=obj)
        self.max_hold_hours = max_hold_hours
        self.energy_saving_threshold = energy_saving_threshold
        self.min_deadline_slack_h = min_deadline_slack_h
        self._energy_now: Dict[str, float] = {}
        self._energy_future: Dict[str, List[tuple]] = {}

    def on_forecast_received(self, bundle: ForecastBundle) -> None:
        for rec in bundle.horizon_1h:
            sid = rec.get("series_id", "")
            self._energy_now[sid] = float(rec.get("energy_wh_pred", 9999))

        by_series: Dict[str, List] = {}
        for rec in bundle.horizon_24h:
            sid = rec.get("series_id", "")
            ts_str = rec.get("forecast_timestamp_utc", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except ValueError:
                continue
            energy = float(rec.get("energy_wh_pred", 9999))
            by_series.setdefault(sid, []).append((ts, energy))
        for sid, entries in by_series.items():
            self._energy_future[sid] = sorted(entries, key=lambda x: x[0])

    def schedule(
        self,
        queue: JobQueue,
        registry: SiteRegistry,
        forecast: ForecastBundle,
        now: datetime,
    ) -> DispatchPlan:
        plan = DispatchPlan(declared_objective=self.declared_objective)
        for job in queue.ready_jobs(now):
            decision = self._decide(job, registry, now)
            if decision:
                plan.add(decision)
        return plan

    def _decide(self, job: Job, registry: SiteRegistry, now: datetime) -> Optional[DispatchDecision]:
        candidates = [s for s in registry.available_sites(job.site_whitelist)
                      if s.capacity.available_slots > 0]
        if not candidates:
            return None

        best_now = min(candidates, key=lambda s: self._energy_now.get(s.site_id, 9999))
        energy_now = self._energy_now.get(best_now.site_id, 9999)

        slack_h = job.deadline_slack_minutes(now) / 60.0
        if slack_h >= self.min_deadline_slack_h:
            future = self._find_low_energy_window(job, candidates, now, energy_now)
            if future is not None:
                future_ts, future_site, future_energy = future
                saving = (energy_now - future_energy) / (energy_now + 1e-9)
                if saving >= self.energy_saving_threshold:
                    logger.debug("Job %s: holding until %s (save %.1f%% energy)",
                                 job.job_id, future_ts.isoformat(), saving * 100)
                    return DispatchDecision(
                        job_id=job.job_id,
                        site_id=future_site.site_id,
                        dispatch_at=future_ts,
                        rationale=(f"greedy_energy: deferred to low-energy window "
                                   f"(save {saving:.1%}; energy_now={energy_now:.1f} "
                                   f"energy_future={future_energy:.1f})"),
                    )

        return DispatchDecision(
            job_id=job.job_id,
            site_id=best_now.site_id,
            dispatch_at=now,
            rationale=f"greedy_energy: immediate dispatch (energy={energy_now:.1f} Wh)",
        )

    def _find_low_energy_window(
        self, job: Job, candidates: List[Site], now: datetime, energy_now: float
    ) -> Optional[tuple]:
        max_hold = timedelta(hours=self.max_hold_hours)
        best_future: Optional[tuple] = None
        best_energy = energy_now
        for site in candidates:
            for ts, energy in self._energy_future.get(site.site_id, []):
                if ts <= now:
                    continue
                if ts > now + max_hold or ts > job.deadline:
                    break
                if energy < best_energy:
                    best_energy = energy
                    best_future = (ts, site, energy)
        return best_future
