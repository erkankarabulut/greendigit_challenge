"""
Task B scheduling experiments.

For each (scheduler × objective) combination, runs the WMS simulator
and scores against the FCFS baseline.

Results are appended to --output CSV.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
TASK_B_SRC = ROOT / "task-b"
if str(TASK_B_SRC) not in sys.path:
    sys.path.insert(0, str(TASK_B_SRC))

from dirac_sim.core.job_queue import JobQueue
from dirac_sim.core.site_model import SiteRegistry
from dirac_sim.core.wms import WMSSimulator
from dirac_sim.core.evaluator import Evaluator
from dirac_sim.api.forecast_client import ForecastClient
from dirac_sim.baselines.fcfs import FCFSScheduler
from dirac_sim.baselines.greedy_carbon import GreedyCarbonScheduler
from dirac_sim.baselines.greedy_energy import GreedyEnergyScheduler
from dirac_sim.baselines.multi_objective import MultiObjectiveScheduler

DATA_DIR = ROOT / "data"

SCHEDULERS = ["greedy_carbon", "greedy_energy", "multi_objective"]
OBJECTIVES = ["carbon", "energy", "makespan"]

SCHEDULER_MAP = {
    "fcfs": FCFSScheduler,
    "greedy_carbon": GreedyCarbonScheduler,
    "greedy_energy": GreedyEnergyScheduler,
    "multi_objective": MultiObjectiveScheduler,
}

CSV_FIELDS = [
    "scheduler", "objective",
    "energy_wh_total", "cfp_g_total", "makespan_total", "deadline_rate",
    "delta_energy_norm", "delta_carbon_norm", "delta_makespan_norm",
    "pareto_score", "declaration_score", "final_score",
    "elapsed_s",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=Path, default=DATA_DIR / "job_trace.csv")
    p.add_argument("--sites", type=Path, default=DATA_DIR / "site_config.json")
    p.add_argument("--forecast-csv", type=Path, default=DATA_DIR / "forecast_baseline.csv")
    p.add_argument("--start", default="2025-11-19T23:00:00")
    p.add_argument("--end", default="2025-11-20T23:00:00")
    p.add_argument("--output", type=Path, default=ROOT / "task-b" / "output" / "exp_scheduling.csv")
    return p.parse_args()


def _parse_utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def run_once(scheduler, jobs_path, sites_path, client, start, end):
    queue = JobQueue.from_csv(str(jobs_path))
    registry = SiteRegistry.from_json(str(sites_path))
    sim = WMSSimulator(
        queue=queue,
        registry=registry,
        scheduler=scheduler,
        forecast_client=client,
        start_time=start,
        end_time=end,
    )
    return sim.run()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    start = _parse_utc(args.start)
    end = _parse_utc(args.end)

    sites_cfg = SiteRegistry.from_json(str(args.sites))
    series_ids = [s.site_id for s in sites_cfg.all_sites()]
    client = ForecastClient(series_ids=series_ids, offline_csv=str(args.forecast_csv))

    # Run FCFS baseline once (shared across all runs)
    print("Running FCFS baseline ...")
    baseline_report = run_once(
        FCFSScheduler(), args.jobs, args.sites, client, start, end
    )
    print("  FCFS baseline done")

    write_header = not args.output.exists()
    fh = args.output.open("a", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()

    for scheduler_name in SCHEDULERS:
        for objective in OBJECTIVES:
            print(f"\n[{scheduler_name}] objective={objective}")
            t0 = time.time()
            try:
                SchedulerClass = SCHEDULER_MAP[scheduler_name]
                scheduler = SchedulerClass(declared_objective=objective)
                report = run_once(scheduler, args.jobs, args.sites, client, start, end)
                result = Evaluator.score(report, baseline_report, declared_objective=objective)
                elapsed = time.time() - t0

                sub = result.submission
                row = {
                    "scheduler": scheduler_name,
                    "objective": objective,
                    "energy_wh_total": sub.energy_wh_total,
                    "cfp_g_total": sub.cfp_g_total,
                    "makespan_total": sub.makespan_total,
                    "deadline_rate": sub.deadline_rate,
                    "delta_energy_norm": result.delta_energy_norm,
                    "delta_carbon_norm": result.delta_carbon_norm,
                    "delta_makespan_norm": result.delta_makespan_norm,
                    "pareto_score": result.pareto_score,
                    "declaration_score": result.declaration_score,
                    "final_score": result.final_score,
                    "elapsed_s": round(elapsed, 1),
                }
                writer.writerow(row)
                fh.flush()
                print(f"  final_score={result.final_score:.4f}  ({elapsed:.1f}s)")
            except Exception as exc:
                print(f"  ERROR: {exc}")

    fh.close()
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
