import pytest

from src.train_tfidf import MLFLOW_TFIDF_EXPERIMENT, log_to_mlflow


def test_run_is_logged_in_the_tfidf_experiment(tmp_path):
    mlflow = pytest.importorskip("mlflow")

    previous_uri = mlflow.get_tracking_uri()
    # MLflow 3 no longer accepts a plain folder, so track into a temporary sqlite database
    # and keep the artifacts in the temporary folder too.
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.create_experiment(
        MLFLOW_TFIDF_EXPERIMENT, artifact_location=(tmp_path / "artifacts").as_uri()
    )
    try:
        log_to_mlflow(
            run_name="tfidf_test_run",
            params={"classifier": "multinomial_nb"},
            metrics={"val_macro_f1": 0.9},
            tags={"stage": "candidate"},
            dicts={"validation_metrics.json": {"macro_f1": 0.9}},
        )

        experiment = mlflow.get_experiment_by_name(MLFLOW_TFIDF_EXPERIMENT)
        assert experiment is not None

        runs = mlflow.search_runs([experiment.experiment_id])
        assert len(runs) == 1
        assert runs.iloc[0]["params.classifier"] == "multinomial_nb"
        assert runs.iloc[0]["metrics.val_macro_f1"] == pytest.approx(0.9)
        assert runs.iloc[0]["tags.stage"] == "candidate"
    finally:
        mlflow.set_tracking_uri(previous_uri)