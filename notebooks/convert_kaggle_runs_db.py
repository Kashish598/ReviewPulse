"""Copy the Kaggle notebook's MLflow runs into a database the pinned MLflow can open.

The notebook (notebooks/bert_candidate_selection.ipynb) ran on Kaggle with MLflow 3.18.0,
whose database schema MLflow 2.19.0 (our pin) cannot read. This script reads the runs
straight from that database with sqlite and re-creates them, with the same names,
parameters, metrics and timestamps, in a new database created by the pinned MLflow.
Nothing is typed in by hand. Runs the notebook logged twice are kept once.

Usage:  python notebooks/convert_kaggle_runs_db.py notebooks/kaggle_mlflow.db mlflow.db
"""

import sqlite3
import sys
from pathlib import Path

from mlflow import MlflowClient, set_tracking_uri
from mlflow.entities import Metric, Param, RunTag

EXPERIMENT = "reviewpulse-bert-candidates"


def read_runs(source: Path) -> list[dict]:
    con = sqlite3.connect(source)
    rows = con.execute(
        "select r.run_uuid, r.name, r.start_time, r.end_time from runs r "
        "join experiments e on e.experiment_id = r.experiment_id "
        "where e.name = ? order by r.start_time",
        (EXPERIMENT,),
    ).fetchall()
    latest = {}  # the notebook ran twice: keep the last run of each name
    for run_uuid, name, start, end in rows:
        latest[name] = {"run_uuid": run_uuid, "name": name, "start": start, "end": end}
    for run in latest.values():
        run["params"] = con.execute(
            "select key, value from params where run_uuid = ?", (run["run_uuid"],)
        ).fetchall()
        run["metrics"] = con.execute(
            "select key, value, timestamp, step from metrics where run_uuid = ?",
            (run["run_uuid"],),
        ).fetchall()
    con.close()
    return list(latest.values())


def main(source: Path, target: Path) -> None:
    runs = read_runs(source)
    if not runs:
        raise SystemExit(f"No runs found in experiment {EXPERIMENT!r} of {source}.")
    set_tracking_uri(f"sqlite:///{target}")
    client = MlflowClient()
    experiment_id = client.create_experiment(EXPERIMENT)
    for run in runs:
        created = client.create_run(
            experiment_id, start_time=run["start"], run_name=run["name"]
        )
        client.log_batch(
            created.info.run_id,
            metrics=[Metric(k, v, ts, step) for k, v, ts, step in run["metrics"]],
            params=[Param(k, v) for k, v in run["params"]],
            tags=[
                RunTag("provenance", "kaggle notebook, validation split"),
                RunTag("original_mlflow_version", "3.18.0"),
            ],
        )
        client.set_terminated(created.info.run_id, "FINISHED", run["end"])
        print(f"copied {run['name']}: {len(run['params'])} params, {len(run['metrics'])} metrics")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]).resolve())
