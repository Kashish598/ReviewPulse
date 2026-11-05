# Contribution: ReviewPulse (Milestone 1)

Tag: `v1.0-milestone1`
Repo: https://github.com/Kashish598/ReviewPulse

## Team

| Name | ERP ID |
|---|---|
| Kashish Anil Kumar | 29105 |
| Abdullah Khalid | 28084 |
| Muskan Pawan | 28394 |

## Member-to-task table (Milestone 1)

| Task | Owner |
|---|---|
| Repo setup, branch rules, GitHub Actions CI, ruff config, LICENSE | Kashish |
| Shared contracts (`src/common.py`), data preparation (`src/prepare_data.py`), data tests, sample fixture | Kashish |
| FastAPI service: `/predict`, `/health`, API tests | Kashish |
| README, `CONTRIBUTION.md`, `AI_USAGE.md`, release (tag and zip) | Kashish |
| BERT candidate comparison on the validation set (Kaggle notebook), MLflow candidate runs | Muskan |
| BERT model adapter (`src/bert_model.py`), `src/train_bert.py`, `src/evaluate_bert.py`, BERT tests, test predictions | Muskan |
| TF-IDF baseline: training script (`train_tfidf.py`), model module (`tfidf_model.py`), tests (`test_tfidf_model.py`, `test_tfidf_mlflow.py`) | Abdullah |

## Reflections

**Kashish:** Set up the repository, CI and the data pipeline, with validation checks, a fixed seed and no overlap between splits, and built the FastAPI service with request validation and tests. The work showed how much easier later steps are when the data splits, shared interfaces and CI are agreed and working first.

**Abdullah:** Built the TF-IDF baseline: the training script, the model module and their tests, including the MLflow logging test. The baseline gives the team a simple reference point to judge whether BERT is worth its extra cost, and showed the need to evaluate both models the same way.

**Muskan:** Compared the candidate BERT sentiment models on the validation set, logged the runs in MLflow, and wrote the BERT adapter, the training and evaluation scripts, and their tests. The work showed the importance of choosing a model on validation data only and keeping the test set for the final evaluation.
