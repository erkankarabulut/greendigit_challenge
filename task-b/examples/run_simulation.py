"""
examples/run_simulation.py
==========================
End-to-end Task B simulation example.

Demonstrates:
  1. Loading job trace and site config
  2. Connecting to a forecast REST API (or offline CSV fallback)
  3. Running the simulator with FCFS and GreedyCarbon schedulers
  4. Evaluating both against the official Task B scorer
  5. Printing and saving the comparison report

Usage
-----
Offline mode (no model server needed):
    python examples/run_simulation.py --offline

Live REST API mode (model server must be running on localhost:8000):
    python examples/run_simulation.py --api-url http://localhost:8000

With a custom scheduler (implement Scheduler and pass as --scheduler):
    python examples/run_simulation.py --offline --scheduler my_pkg.MyScheduler

Options
-------
    --offline           Use local CSV forecasts (no HTTP required)
    --api-url URL       Forecast REST API base URL
    --jobs PATH         Path to job trace CSV
    --sites PATH        Path to site config JSON
    --forecast-csv PATH Path to offline forecast CSV
    --output-dir PATH   Directory to write result CSVs
    --objective         Declared objective: energy|carbon|makespan
    --start             Simulation start (ISO timestamp)
    --end               Simulation end   (ISO timestamp)
"""
import argparse
import importlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from dirac_sim.core.job_queue import JobQueue
from dirac_sim.core.site_model import SiteRegistry
from dirac_sim.core.wms import WMSSimulator
from dirac_sim.core.evaluator import Evaluator
from dirac_sim.api.forecast_client import ForecastClient
from dirac_sim.baselines.fcfs import FCFSScheduler
from dirac_sim.baselines.greedy_carbon import GreedyCarbonScheduler
from dirac_sim.baselines.greedy_energy import GreedyEnergyScheduler
from dirac_sim.baselines.multi_objective import MultiObjectiveScheduler
from examples.test_policy import LowestCarbonTestPolicy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("run_simulation")

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_scheduler(spec: str):
    """Load scheduler class from 'module.ClassName' spec or built-in name."""
    builtins = {
        "fcfs": FCFSScheduler,
        "greedy_carbon": GreedyCarbonScheduler,
        "greedy_energy": GreedyEnergyScheduler,
        "multi_objective": MultiObjectiveScheduler,
        "test_policy": LowestCarbonTestPolicy,
    }
    if spec in builtins:
        return builtins[spec]
    module_path, class_name = spec.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def run_once(scheduler, queue_path, sites_path, client, start, end, label):
    """Run one simulation and return its report."""
    queue = JobQueue.from_csv(queue_path)
    registry = SiteRegistry.from_json(sites_path)

    sim = WMSSimulator(
        queue=queue,
        registry=registry,
        scheduler=scheduler,
        forecast_client=client,
        start_time=start,
        end_time=end,
    )
    logger.info("Running '%s' scheduler...", label)
    report = sim.run()
    logger.info("[%s] %s", label, report.summary().replace("\n", "  |  "))
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Task B simulation runner")
    parser.add_argument("--offline", action="store_true",
                        help="Use offline CSV forecasts")
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="Forecast REST API base URL")
    parser.add_argument("--jobs",
                        default=str(DATA_DIR / "job_trace.csv"))
    parser.add_argument("--sites",
                        default=str(DATA_DIR / "site_config.json"))
    parser.add_argument("--forecast-csv",
                        default=str(DATA_DIR / "forecast_baseline.csv"))
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--objective", default="carbon",
                        choices=["energy", "carbon", "makespan"])
    parser.add_argument("--start", default="2025-11-19T23:00:00")
    parser.add_argument("--end",   default="2025-11-20T23:00:00",
                        help="Default: 24h real-data smoke window")
    parser.add_argument("--scheduler", default="greedy_carbon",
                        help="Built-in name or 'my.module.MyClass'")
    args = parser.parse_args()

    start = _parse_utc(args.start)
    end = _parse_utc(args.end)
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Forecast client                                                      #
    # ------------------------------------------------------------------ #
    sites_cfg = SiteRegistry.from_json(args.sites)
    series_ids = [s.site_id for s in sites_cfg.all_sites()]

    if args.offline:
        logger.info("Offline mode: reading forecasts from %s", args.forecast_csv)
        client = ForecastClient(
            series_ids=series_ids,
            offline_csv=args.forecast_csv,
        )
    else:
        logger.info("Live mode: forecast API at %s", args.api_url)
        client = ForecastClient(
            base_url=args.api_url,
            series_ids=series_ids,
        )

    # ------------------------------------------------------------------ #
    # Run FCFS baseline                                                    #
    # ------------------------------------------------------------------ #
    baseline_report = run_once(
        FCFSScheduler(), args.jobs, args.sites, client, start, end, "FCFS"
    )
    baseline_report.to_csv(f"{args.output_dir}/baseline_execution_report.csv")

    # ------------------------------------------------------------------ #
    # Run participant scheduler                                            #
    # ------------------------------------------------------------------ #
    SchedulerClass = load_scheduler(args.scheduler)
    scheduler = SchedulerClass(declared_objective=args.objective) \
        if args.scheduler != "fcfs" else SchedulerClass()

    submission_report = run_once(
        scheduler, args.jobs, args.sites, client, start, end, args.scheduler
    )
    submission_report.to_csv(f"{args.output_dir}/execution_report.csv")
    submission_report.to_dispatch_csv(
        f"{args.output_dir}/dispatch_log.csv",
        declared_objective=args.objective,
    )

    # ------------------------------------------------------------------ #
    # Evaluate                                                            #
    # ------------------------------------------------------------------ #
    result = Evaluator.score(
        submission=submission_report,
        baseline=baseline_report,
        declared_objective=args.objective,
    )
    print("\n" + "=" * 55)
    print(result)
    print("=" * 55)

    # Write score summary
    summary_path = f"{args.output_dir}/score_summary.txt"
    with open(summary_path, "w") as fh:
        fh.write(str(result))
    logger.info("Score summary written to %s", summary_path)

    metrics_path = f"{args.output_dir}/score_metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump({
            "run": {
                "scheduler": args.scheduler,
                "objective": args.objective,
                "offline": args.offline,
                "api_url": None if args.offline else args.api_url,
                "jobs": args.jobs,
                "sites": args.sites,
                "forecast_csv": args.forecast_csv if args.offline else None,
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
            "evaluation": result.to_dict(),
        }, fh, indent=2)
        fh.write("\n")
    logger.info("Score metrics JSON written to %s", metrics_path)

    return result


def _parse_utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


if __name__ == "__main__":
    main()
