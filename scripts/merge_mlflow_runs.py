"""Copy runs from one MLflow sqlite database into another, keeping their full history.

Use it to bring the Kaggle fine-tuning runs (`finetune_mlflow.db`, `scaling_mlflow.db`) into
your own `reviewpulse_mlflow.db`, so that `mlflow ui` shows everything in one place.

What is copied, per run: name, start and end time, params, tags, and the full metric history
(every epoch and step, not only the last value). What is not copied: artifacts (the small JSON
and CSV files logged with a run). Those are not stored in the database; the same files are in
`scaling_results.zip`.

The source database is opened read-only and is never changed. It is read with plain sqlite,
so the MLflow version that wrote it does not matter.

Running it twice is safe: a run is skipped if the target already holds a run with the same
name that was merged from the same source file (tag `merged_from`).

    python merge_mlflow_runs.py scaling_mlflow.db --dry-run
    python merge_mlflow_runs.py scaling_mlflow.db
    python merge_mlflow_runs.py finetune_mlflow.db
    python merge_mlflow_runs.py scaling_mlflow.db --tracking-uri sqlite:///reviewpulse_mlflow.db

BACK UP THE TARGET DATABASE FIRST (copy reviewpulse_mlflow.db somewhere outside the repo).
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

DEFAULT_TARGET_URI = "sqlite:///reviewpulse_mlflow.db"
DEFAULT_EXPERIMENT = "reviewpulse-bert"
BATCH_LIMIT = 900  # MLflow accepts at most 1000 metrics per log_batch call


def read_runs(db_path: str | Path) -> list[dict]:
    """Finished, active runs of an MLflow sqlite database, with full metric history."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"MLflow database not found: {db_path}")
    connection = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT r.run_uuid, r.name, r.start_time, r.end_time, e.name "
            "FROM runs r JOIN experiments e ON e.experiment_id = r.experiment_id "
            "WHERE r.lifecycle_stage = 'active' AND r.status = 'FINISHED' "
            "ORDER BY r.start_time"
        ).fetchall()
        runs = []
        for run_id, name, start, end, experiment in rows:
            params = dict(
                connection.execute("SELECT key, value FROM params WHERE run_uuid = ?", (run_id,))
            )
            tags = {
                key: value
                for key, value in connection.execute(
                    "SELECT key, value FROM tags WHERE run_uuid = ?", (run_id,)
                )
                if not key.startswith("mlflow.")
            }
            metrics = [
                (key, float(value), int(timestamp), int(step))
                for key, value, timestamp, step in connection.execute(
                    "SELECT key, value, timestamp, step FROM metrics "
                    "WHERE run_uuid = ? AND is_nan = 0 AND value IS NOT NULL "
                    "ORDER BY timestamp, step",
                    (run_id,),
                )
            ]
            runs.append(
                {
                    "name": name or run_id[:8],
                    "experiment": experiment,
                    "start_time": start,
                    "end_time": end,
                    "params": params,
                    "tags": tags,
                    "metrics": metrics,
                }
            )
        return runs
    finally:
        connection.close()


def chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def merge(
    runs: list[dict], client, experiment_id: str, source_name: str
) -> tuple[list[str], list[str]]:
    """Log `runs` with `client` (an `mlflow.MlflowClient`). Returns (merged, skipped) names."""
    from mlflow.entities import Metric, Param

    existing = {
        run.data.tags.get("mlflow.runName")
        for run in client.search_runs(
            [experiment_id],
            filter_string=f"tags.merged_from = '{source_name}'",
            max_results=1000,
        )
    }
    merged, skipped = [], []
    for run in runs:
        if run["name"] in existing:
            skipped.append(run["name"])
            continue
        tags = {**run["tags"], "merged_from": source_name, "source_experiment": run["experiment"]}
        created = client.create_run(
            experiment_id, start_time=run["start_time"], tags=tags, run_name=run["name"]
        )
        run_id = created.info.run_id
        params = [Param(k, str(v)) for k, v in run["params"].items()]
        client.log_batch(run_id, params=params)
        metrics = [Metric(k, v, t, s) for k, v, t, s in run["metrics"]]
        for part in chunks(metrics, BATCH_LIMIT):
            client.log_batch(run_id, metrics=part)
        client.set_terminated(run_id, status="FINISHED", end_time=run["end_time"])
        merged.append(run["name"])
    return merged, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy MLflow runs between sqlite databases.")
    parser.add_argument("source", help="the database to copy from, e.g. scaling_mlflow.db")
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TARGET_URI),
        help="the database to copy into (default: $MLFLOW_TRACKING_URI, else "
        f"{DEFAULT_TARGET_URI})",
    )
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="target experiment name")
    parser.add_argument("--dry-run", action="store_true", help="list the runs, change nothing")
    args = parser.parse_args()

    runs = read_runs(args.source)
    print(f"{len(runs)} finished run(s) in {args.source}:")
    for run in runs:
        points, count = len(run["metrics"]), len(run["params"])
        print(f"  {run['name']:<40} {points:>4} metric points, {count} params")
    if args.dry_run:
        print("Dry run: nothing was changed.")
        return

    target = args.tracking_uri
    target_file = Path(target.removeprefix("sqlite:///")).resolve()
    if target.startswith("sqlite:///") and target_file == Path(args.source).resolve():
        raise SystemExit("The target must not be the source database.")

    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=target)
    found = client.get_experiment_by_name(args.experiment)
    experiment_id = found.experiment_id if found else client.create_experiment(args.experiment)
    merged, skipped = merge(runs, client, experiment_id, Path(args.source).name)
    print(f"Merged {len(merged)} run(s) into '{args.experiment}' at {target}")
    if skipped:
        print(f"Skipped {len(skipped)} already merged: {skipped}")
    print(f"View with: mlflow ui --backend-store-uri {target}")


if __name__ == "__main__":
    main()
