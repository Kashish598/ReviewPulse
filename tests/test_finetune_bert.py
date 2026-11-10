import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

from src.finetune_bert import (
    CONFIGS,
    STAR_TO_BINARY,
    Tracker,
    decide,
    decide_by_sum,
    evaluate_probabilities,
    pick_best,
)


def test_three_distinct_recipes():
    assert len(CONFIGS) == 3
    assert len({(c.learning_rate, c.epochs, c.freeze_layers) for c in CONFIGS.values()}) == 3
    assert all(name == config.name for name, config in CONFIGS.items())
    assert all(0 <= c.freeze_layers < 12 for c in CONFIGS.values())


def test_star_mapping_matches_the_api_policy():
    # 1 to 3 stars negative, 4 to 5 stars positive (same as BertPredictor.map_stars_to_label)
    assert STAR_TO_BINARY == [0, 0, 0, 1, 1]


def test_pick_best_uses_macro_f1_then_accuracy():
    results = {
        "a": {"macro_f1": 0.94, "accuracy": 0.94},
        "b": {"macro_f1": 0.95, "accuracy": 0.94},
        "c": {"macro_f1": 0.95, "accuracy": 0.95},
    }
    assert pick_best(results) == "c"


def test_pick_best_keeps_the_first_on_a_full_tie():
    same = {"macro_f1": 0.9, "accuracy": 0.9}
    assert pick_best({"first": same, "second": dict(same)}) == "first"


def test_pick_best_rejects_empty_results():
    with pytest.raises(ValueError):
        pick_best({})


def test_decide_takes_the_most_likely_star_then_maps_it():
    probabilities = np.array(
        [
            [0.70, 0.10, 0.10, 0.05, 0.05],  # 1 star -> negative
            [0.05, 0.05, 0.10, 0.10, 0.70],  # 5 stars -> positive
            [0.10, 0.10, 0.50, 0.20, 0.10],  # 3 stars -> negative
        ]
    )

    labels, scores = decide(probabilities)

    assert labels == [0, 1, 0]
    assert scores == pytest.approx([0.70, 0.70, 0.50])


def test_serving_rule_and_sum_rule_can_disagree():
    # Most likely star is 3 (negative), but 4 and 5 stars together are 65% (positive).
    probabilities = np.array([[0.0, 0.0, 0.35, 0.33, 0.32]])

    assert decide(probabilities)[0] == [0]
    assert decide_by_sum(probabilities) == [1]


def test_evaluate_probabilities_reports_both_rules():
    probabilities = np.array(
        [
            [0.7, 0.1, 0.1, 0.05, 0.05],
            [0.05, 0.05, 0.1, 0.1, 0.7],
            [0.0, 0.0, 0.35, 0.33, 0.32],
            [0.05, 0.05, 0.1, 0.1, 0.7],
        ]
    )

    metrics = evaluate_probabilities(probabilities, [0, 1, 1, 1])

    assert metrics["accuracy"] == pytest.approx(0.75)  # serving rule misses the third row
    assert metrics["accuracy_sum_rule"] == pytest.approx(1.0)
    assert metrics["confusion_matrix"] == [[1, 0], [1, 2]]


def test_disabled_tracker_accepts_every_call():
    tracker = Tracker(enabled=False)

    with tracker.run("anything", {"stage": "x"}) as run:
        run.params({"a": 1})
        run.metrics({"m": 0.5}, step=1)
        run.dict({"k": "v"}, "file.json")
        run.artifact("missing.txt")


def test_tracker_logs_runs_when_enabled(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    previous_uri = mlflow.get_tracking_uri()
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    try:
        tracker = Tracker(enabled=True, tracking_uri=uri)
        with tracker.run("finetune_demo", {"stage": "finetune_candidate"}) as run:
            run.params({"learning_rate": 2e-5})
            run.metrics({"val_macro_f1": 0.95}, step=1)
            run.metrics({"val_macro_f1": 0.96}, step=2)

        runs = mlflow.search_runs(experiment_names=["reviewpulse-bert"])
        assert len(runs) == 1
        assert runs.iloc[0]["params.learning_rate"] == "2e-05"
        assert runs.iloc[0]["metrics.val_macro_f1"] == pytest.approx(0.96)
        assert runs.iloc[0]["tags.stage"] == "finetune_candidate"
    finally:
        mlflow.set_tracking_uri(previous_uri)


def test_mapped_loss_is_a_proper_binary_likelihood():
    torch = pytest.importorskip("torch")
    from src.finetune_bert import binary_log_probs, training_loss

    logits = torch.zeros(2, 5)  # all stars equally likely: P(negative) = 0.6, P(positive) = 0.4
    log_probs = binary_log_probs(logits)

    assert torch.allclose(log_probs.exp().sum(dim=1), torch.ones(2))
    assert torch.allclose(log_probs.exp()[:, 0], torch.tensor(0.6))
    loss = training_loss(logits, torch.tensor([0, 1]))
    assert float(loss) == pytest.approx(-(np.log(0.6) + np.log(0.4)) / 2, rel=1e-5)


def test_loss_is_lower_for_a_confident_correct_prediction():
    torch = pytest.importorskip("torch")
    from src.finetune_bert import training_loss

    positive = torch.tensor([[-4.0, -4.0, -4.0, 1.0, 4.0]])
    label = torch.tensor([1])

    assert training_loss(positive, label) < training_loss(positive.flip(1), label)


def test_freeze_lower_layers():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from src.finetune_bert import count_parameters, freeze_lower_layers

    layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(4)])
    model = torch.nn.Module()
    model.bert = SimpleNamespace(
        embeddings=torch.nn.Embedding(5, 2), encoder=SimpleNamespace(layer=layers)
    )
    model.head = torch.nn.Linear(2, 5)
    model.embeddings = model.bert.embeddings
    model.layers = layers

    freeze_lower_layers(model, 2)
    total, trainable = count_parameters(model)

    assert trainable < total
    assert not any(p.requires_grad for p in layers[0].parameters())
    assert not any(p.requires_grad for p in layers[1].parameters())
    assert all(p.requires_grad for p in layers[2].parameters())
    assert not model.bert.embeddings.weight.requires_grad


class FakeModel:
    def to(self, device):
        return self

    def save_pretrained(self, path):
        (path / "model.safetensors").write_text("weights")


class FakeTokenizer:
    def save_pretrained(self, path):
        (path / "tokenizer.json").write_text("{}")


def test_main_selects_saves_and_scores_test_once(monkeypatch, tmp_path):
    """Run the whole flow with stand-ins for torch, transformers and the training loop."""
    import src.finetune_bert as ft

    fake_torch = types.SimpleNamespace(
        __version__="0",
        cuda=types.SimpleNamespace(is_available=lambda: False),
        device=lambda name: types.SimpleNamespace(type=name),
    )
    fake_transformers = types.SimpleNamespace(
        __version__="0",
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda name: FakeTokenizer()),
        AutoModelForSequenceClassification=types.SimpleNamespace(
            from_pretrained=lambda name: FakeModel()
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    def frame(n):
        labels = [i % 2 for i in range(n)]
        texts = [("good " if label else "bad ") + str(i) for i, label in enumerate(labels)]
        return pd.DataFrame({"row_id": range(n), "review_text": texts, "label": labels})

    splits = {"train": frame(40), "validation": frame(20), "test": frame(30)}
    monkeypatch.setattr(ft, "load_split", lambda name, sample=False: splits[name])

    def fake_predict(model, tokenizer, texts, device, use_amp, batch_size=64):
        probabilities = np.zeros((len(texts), 5))
        for i, text in enumerate(texts):
            probabilities[i, 4 if text.startswith("good") else 0] = 1.0
        return probabilities

    trained = []

    def fake_train(config, train, validation, tokenizer, device, use_amp, seed, tracker, zero_shot):
        trained.append(config.name)
        score = 0.9 + config.learning_rate * 1000  # the last recipe has the best validation score
        metrics = {"macro_f1": score, "accuracy": score}
        summary = {
            "macro_f1": score,
            "accuracy": score,
            "best_epoch": 1,
            "metrics": metrics,
            "train_seconds": 1.0,
        }
        return FakeModel(), summary

    monkeypatch.setattr(ft, "predict_probabilities", fake_predict)
    monkeypatch.setattr(ft, "train_one", fake_train)
    monkeypatch.setattr(ft, "FINETUNED_DIR", tmp_path / "models" / "nlptown_finetuned")
    monkeypatch.setattr(ft, "PREDICTIONS_DIR", tmp_path / "predictions")
    monkeypatch.setattr(sys, "argv", ["finetune_bert", "--no-mlflow"])

    ft.main()

    assert trained == list(CONFIGS)
    model_dir = tmp_path / "models" / "nlptown_finetuned"
    assert (model_dir / "model.safetensors").exists()
    assert (model_dir / "tokenizer.json").exists()
    metadata = json.loads((model_dir / "metadata.json").read_text())
    assert metadata["recipe"]["name"] == "lr3e5_ep3_freeze6"
    assert metadata["fine_tuned"] is True
    assert metadata["test_accuracy"] == 1.0
    predictions = pd.read_csv(tmp_path / "predictions" / "bert_finetuned_test.csv")
    assert len(predictions) == 30
    assert list(predictions.columns) == ["row_id", "label_true", "pred", "score"]
    metrics = json.loads(
        (tmp_path / "predictions" / "bert_finetuned_test_metrics.json").read_text()
    )
    assert metrics["n"] == 30
