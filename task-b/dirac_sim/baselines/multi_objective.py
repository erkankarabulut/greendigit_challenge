"""
dirac_sim.baselines.multi_objective
=====================================
Multi-objective weighted scheduler.

Scores each (site, time) candidate as a weighted sum of normalised energy
and carbon forecasts.  The declared primary objective drives which weight
dominates; the secondary objectives act as soft penalties.

Strategy
--------
At each tick, for every ready job:
  1. Build a list of candidate (site, timestamp) pairs over a look-ahead
     window (default 6 h) at 15-min resolution.
  2. Score each candidate:
       score = w_energy * norm_energy(site, t)
             + w_carbon * norm_carbon(site, t)
             + w_makespan * norm_delay(t - now)
  3. Pick the candidate with the lowest score.

Weights are set automatically from `declared_objective` but can be
overridden explicitly.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from dirac_sim.core.job_queue import Job, JobQueue
from dirac_sim.core.scheduler import (
    DispatchDecision, DispatchPlan, ForecastBundle, Objective, Scheduler,
)
from dirac_sim.core.site_model import Site, SiteRegistry

logger = logging.getLogger(__name__)

_TICK = timedelta(minutes=15)


class MultiObjectiveScheduler(Scheduler):
    """
    Weighted multi-objective scheduler.

    Parameters
    ----------
    declared_objective  : Primary objective — drives default weights.
    w_energy            : Weight for normalised energy cost (override).
    w_carbon            : Weight for normalised carbon cost (override).
    w_makespan          : Weight for normalised dispatch delay.
    lookahead_hours     : How far ahead to scan for better windows.
    min_deadline_slack_h: Minimum slack before holding a job.
    """

    _DEFAULT_WEIGHTS = {
        Objective.ENERGY:   (0.7, 0.2, 0.1),
        Objective.CARBON:   (0.2, 0.7, 0.1),
        Objective.MAKESPAN: (0.1, 0.1, 0.8),
    }

    def __init__(
        self,
        declared_objective: str = "carbon",
        w_energy: float | None = None,
        w_carbon: float | None = None,
        w_makespan: float | None = None,
        lookahead_hours: float = 6.0,
        min_deadline_slack_h: float = 1.0,
    ) -> None:
        obj = Objective(declared_objective) if isinstance(declared_objective, str) else declared_objective
        super().__init__(declared_objective=obj)
        defaults = self._DEFAULT_WEIGHTS[obj]
        self.w_energy = w_energy if w_energy is not None else defaults[0]
        self.w_carbon = w_carbon if w_carbon is not None else defaults[1]
        self.w_makespan = w_makespan if w_makespan is not None else defaults[2]
        self.lookahead_hours = lookahead_hours
        self.min_deadline_slack_h = min_deadline_slack_h

        # forecast caches: series_id -> list of (ts, energy, carbon) sorted by ts
        self._forecast: Dict[str, List[Tuple[datetime, float, float]]] = {}
        # per-series range for normalisation
        self._energy_range: Dict[str, Tuple[float, float]] = {}
        self._carbon_range: Dict[str, Tuple[float, float]] = {}

    def on_forecast_received(self, bundle: ForecastBundle) -> None:
        by_series: Dict[str, List[Tuple[datetime, float, float]]] = {}

        for rec in bundle.horizon_1h + bundle.horizon_24h:
            sid = rec.get("series_id", "")
            ts_str = rec.get("forecast_timestamp_utc", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except ValueError:
                continue
            energy = float(rec.get("energy_wh_pred", 0.0))
            carbon = float(rec.get("cfp_g_pred", 0.0))
            by_series.setdefault(sid, []).append((ts, energy, carbon))

        for sid, entries in by_series.items():
            self._forecast[sid] = sorted(entries, key=lambda x: x[0])
            energies = [e for _, e, _ in entries]
            carbons  = [c for _, _, c in entries]
            self._energy_range[sid] = (min(energies, default=0.0), max(energies, default=1.0))
            self._carbon_range[sid] = (min(carbons,  default=0.0), max(carbons,  default=1.0))

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

        lookahead = timedelta(hours=self.lookahead_hours)
        slack_h = job.deadline_slack_minutes(now) / 60.0
        max_delay = min(lookahead, timedelta(hours=slack_h - self.min_deadline_slack_h)
                        if slack_h > self.min_deadline_slack_h else timedelta(0))

        best_score = float("inf")
        best_decision: Optional[DispatchDecision] = None
        max_delay_minutes = max_delay.total_seconds() / 60

        # Build candidate timestamps at 15-min resolution within the allowed window
        n_steps = max(1, int(max_delay_minutes / 15))
        for step in range(n_steps):
            candidate_ts = now + step * _TICK
            if candidate_ts > job.deadline:
                break
            delay_fraction = step / max(1, n_steps - 1)

            for site in candidates:
                e_pred, c_pred = self._lookup_forecast(site.site_id, candidate_ts)
                e_norm = self._normalise(e_pred, self._energy_range.get(site.site_id, (0.0, 1.0)))
                c_norm = self._normalise(c_pred, self._carbon_range.get(site.site_id, (0.0, 1.0)))
                score = (self.w_energy * e_norm
                         + self.w_carbon * c_norm
                         + self.w_makespan * delay_fraction)
                if score < best_score:
                    best_score = score
                    best_decision = DispatchDecision(
                        job_id=job.job_id,
                        site_id=site.site_id,
                        dispatch_at=candidate_ts,
                        rationale=(
                            f"multi_obj({self.declared_objective.value}): "
                            f"score={score:.3f} "
                            f"e_norm={e_norm:.2f} c_norm={c_norm:.2f} "
                            f"delay_frac={delay_fraction:.2f}"
                        ),
                    )

        return best_decision

    def _lookup_forecast(self, sid: str, ts: datetime) -> Tuple[float, float]:
        entries = self._forecast.get(sid, [])
        best = None
        best_delta = timedelta(days=999)
        for entry_ts, e, c in entries:
            delta = abs(entry_ts - ts)
            if delta < best_delta:
                best_delta = delta
                best = (e, c)
        return best if best is not None else (0.0, 0.0)

    @staticmethod
    def _normalise(value: float, range_: Tuple[float, float]) -> float:
        lo, hi = range_
        if hi <= lo:
            return 0.5
        return max(0.0, min(1.0, (value - lo) / (hi - lo)))
