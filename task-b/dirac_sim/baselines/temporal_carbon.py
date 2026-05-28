"""
dirac_sim.baselines.temporal_carbon
=====================================
Carbon-aware scheduler that queries the site model's accumulated forecast
data directly via site.get_carbon(t), scanning 15-min steps within a
max_hold_hours window.

Unlike GreedyCarbonScheduler, this does not rely on bundle.horizon_24h
(a single point 24h ahead, always outside the 6h deferral window).
Instead it reads site.get_carbon(t) for every 15-min slot in [now, now+6h],
which is populated by the WMS from horizon_1h records on each tick.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from dirac_sim.core.scheduler import (
    DispatchDecision, DispatchPlan, ForecastBundle, Objective, Scheduler,
)
from dirac_sim.core.job_queue import JobQueue
from dirac_sim.core.site_model import SiteRegistry

logger = logging.getLogger(__name__)


class TemporalCarbonScheduler(Scheduler):
    """
    Temporal carbon-aware scheduler.

    For each ready job, scans site.get_carbon(t) in 15-min increments over
    [now, now + max_hold_hours] to find the lowest-carbon dispatch slot.
    Defers if the saving exceeds carbon_saving_threshold and the job has
    enough deadline slack.

    Parameters
    ----------
    declared_objective      : Primary objective string (default: "carbon").
    max_hold_hours          : Lookahead window in hours (default: 6).
    carbon_saving_threshold : Minimum fractional saving to justify deferral
                              (default: 0.15 = 15%).
    min_deadline_slack_h    : Minimum remaining slack required to defer
                              (default: 2 h).
    """

    def __init__(
        self,
        declared_objective: str = "carbon",
        max_hold_hours: float = 6.0,
        carbon_saving_threshold: float = 0.15,
        min_deadline_slack_h: float = 2.0,
    ) -> None:
        obj = Objective(declared_objective) if isinstance(
            declared_objective, str) else declared_objective
        super().__init__(declared_objective=obj)
        self.max_hold_hours = max_hold_hours
        self.carbon_saving_threshold = carbon_saving_threshold
        self.min_deadline_slack_h = min_deadline_slack_h

    def schedule(
        self,
        queue: JobQueue,
        registry: SiteRegistry,
        forecast: ForecastBundle,
        now: datetime,
    ) -> DispatchPlan:
        plan = DispatchPlan(declared_objective=self.declared_objective)

        for job in queue.ready_jobs(now):
            candidates = [s for s in registry.available_sites(job.site_whitelist)
                          if s.capacity.available_slots > 0]
            if not candidates:
                continue

            # Pick lowest-carbon site right now
            site = min(candidates, key=lambda s: s.get_carbon(now))
            carbon_now = site.get_carbon(now)

            dispatch_at = now
            best_carbon = carbon_now

            slack_h = job.deadline_slack_minutes(now) / 60.0
            if slack_h >= self.min_deadline_slack_h:
                horizon = now + timedelta(hours=self.max_hold_hours)
                t = now + timedelta(minutes=15)
                while t <= horizon and t <= job.deadline:
                    c = site.get_carbon(t)
                    if c < best_carbon:
                        best_carbon = c
                        dispatch_at = t
                    t += timedelta(minutes=15)

                saving = (carbon_now - best_carbon) / (carbon_now + 1e-9)
                if saving < self.carbon_saving_threshold:
                    dispatch_at = now

            plan.add(DispatchDecision(
                job_id=job.job_id,
                site_id=site.site_id,
                dispatch_at=dispatch_at,
                rationale=(
                    f"temporal_carbon: dispatch at {dispatch_at.isoformat()} "
                    f"(cfp={best_carbon:.2f} vs now={carbon_now:.2f})"
                ),
            ))

        return plan
