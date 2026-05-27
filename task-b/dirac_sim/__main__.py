"""
dirac_sim.__main__
==================
CLI entry point.

Usage:
  python -m dirac_sim simulate [options]
  python -m dirac_sim serve    [options]
  python -m dirac_sim prepare-data [options]
"""
import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


def main():
    parser = argparse.ArgumentParser(
        prog="dirac-sim",
        description="DiracSim — ECML PKDD Task B starter kit CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # simulate
    sim_p = sub.add_parser("simulate", help="Run the WMS simulator")
    sim_p.add_argument("--offline", action="store_true")
    sim_p.add_argument("--api-url", default="http://localhost:8000")
    sim_p.add_argument("--jobs", default=str(DATA_DIR / "job_trace.csv"))
    sim_p.add_argument("--sites", default=str(DATA_DIR / "site_config.json"))
    sim_p.add_argument("--forecast-csv",
                       default=str(DATA_DIR / "forecast_baseline.csv"))
    sim_p.add_argument("--objective", default="carbon")
    sim_p.add_argument(
        "--scheduler", default="greedy_carbon",
        help="Built-in: fcfs, greedy_carbon, greedy_energy, multi_objective, "
             "or 'my.module.MyClass' for custom schedulers.",
    )
    sim_p.add_argument("--output-dir", default="results")
    sim_p.add_argument("--start", default="2025-11-19T23:00:00")
    sim_p.add_argument("--end",   default="2025-11-20T23:00:00")

    # serve
    srv_p = sub.add_parser("serve", help="Start the forecast REST API server")
    srv_p.add_argument("--host", default="0.0.0.0")
    srv_p.add_argument("--port", type=int, default=8000)
    srv_p.add_argument("--reload", action="store_true")

    # prepare-data
    prep_p = sub.add_parser(
        "prepare-data",
        help="Prepare simulator inputs from raw metric summaries",
    )
    prep_p.add_argument("--raw-metrics-dir",
                        default=str(DATA_DIR / "raw_metrics"))
    prep_p.add_argument("--output-dir", default=str(DATA_DIR))

    args = parser.parse_args()

    if args.command == "simulate":
        _cmd_simulate(args)
    elif args.command == "serve":
        _cmd_serve(args)
    elif args.command == "prepare-data":
        _cmd_prepare_data(args)
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_simulate(args):
    # Delegate to the example runner
    import importlib.util, os, sys
    spec = importlib.util.spec_from_file_location(
        "run_simulation",
        os.path.join(os.path.dirname(__file__),
                     "..", "examples", "run_simulation.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    # Patch sys.argv
    sys.argv = [
        "run_simulation",
        *(["--offline"] if args.offline else []),
        "--api-url", args.api_url,
        "--jobs", args.jobs,
        "--sites", args.sites,
        "--forecast-csv", args.forecast_csv,
        "--objective", args.objective,
        "--scheduler", args.scheduler,
        "--output-dir", args.output_dir,
        "--start", args.start,
        "--end", args.end,
    ]
    spec.loader.exec_module(mod)
    mod.main()


def _cmd_serve(args):
    try:
        import uvicorn
        from dirac_sim.api.server import app
        uvicorn.run(
            "dirac_sim.api.server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    except ImportError:
        print("FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn")
        sys.exit(1)


def _cmd_prepare_data(args):
    import importlib.util
    from pathlib import Path

    path = DATA_DIR / "prepare_data.py"
    spec = importlib.util.spec_from_file_location("prepare_data", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.prepare(Path(args.raw_metrics_dir), Path(args.output_dir))
    print(f"Data files written to {args.output_dir}/")


if __name__ == "__main__":
    main()
