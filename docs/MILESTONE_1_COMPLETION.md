# ReviewPulse: Milestone 1 Completion

ReviewPulse is a sentiment analysis service for Amazon product reviews. Milestone 1 delivers a reproducible data pipeline, a comparison of three pretrained BERT models, a TF-IDF baseline, a fine-tuned BERT model, MLflow experiment tracking, a FastAPI service, and CI.

## 1. Summary

| Item | Result |
|---|---|
| Task | Binary sentiment (negative / positive) on Amazon Polarity reviews |
| Data | 10,000 train / 1,000 validation / 600 test reviews, seed 42, pinned dataset revision |
| Selected pretrained model | nlptown BERT (`nlptown/bert-base-multilingual-uncased-sentiment`), validation accuracy 0.9300 |
| Pretrained BERT on test | accuracy 0.9283, macro-F1 0.9282 (557 / 600) |
| TF-IDF baseline (linear SVM) on test | accuracy 0.8683, macro-F1 0.8683 (521 / 600) |
| Fine-tuned BERT, mean of 9 runs on test | accuracy 0.9420 |
| Fine-tuned BERT, selected run on test | accuracy 0.9517, macro-F1 0.9517 (571 / 600) |
| Experiment tracking | MLflow: `reviewpulse-bert` and `reviewpulse-tfidf` |
| Service | FastAPI `POST /predict` and `GET /health` serving the saved BERT model |
| Quality | ruff and pytest in GitHub Actions; changes merged through reviewed pull requests |

The main conclusions are:

1. Pretrained BERT is 6.0 points more accurate than the TF-IDF baseline on the same 600 test reviews.
2. Fine-tuning improves on the pretrained model by about 1.4 points on average (mean over nine runs). The best single run gains 2.3 points, but it is the best of nine, so the mean is the fair figure.
3. About three quarters of the best model's remaining errors are shared with the earlier models, which points to ambiguous or mislabeled reviews rather than a lack of data.

## 2. What was delivered

| Area | Deliverable | Where |
|---|---|---|
| Data | Cleaning, split rules, validation checks, data report | `src/prepare_data.py`, `docs/data_report.json` |
| Pretrained models | Three candidates compared on validation only | `notebooks/bert_candidate_selection.ipynb`, `docs/mlflow_runs_summary.csv` |
| Model saving | The selected model saved locally with metadata | `src/train_bert.py`, `models/nlptown_sentiment/` |
| Test evaluation | Predictions and metrics on the 600 held-out reviews | `src/evaluate_bert.py`, `data/predictions/bert_test*.{csv,json}` |
| Baseline | TF-IDF with three classifiers, best chosen on validation | `src/tfidf_model.py`, `src/train_tfidf.py` |
| Fine-tuning | Three recipes, then three seeds at three training sizes | `src/finetune_bert.py`, `src/finetune_scaling.py`, Kaggle notebooks |
| Tracking | MLflow experiments with parameters, metrics and per-epoch curves, and a script to merge Kaggle runs into the local database | `reviewpulse-bert`, `reviewpulse-tfidf`, `scripts/merge_mlflow_runs.py` |
| API | Prediction and health endpoints | `src/app.py`, `src/bert_model.py` |
| Tests and CI | Unit tests, ruff, GitHub Actions job `lint-and-test` | `tests/`, `.github/workflows/ci.yml` |
| Documentation | README, this document, contribution and AI-usage notes | `README.md`, `docs/`, `CONTRIBUTION.md`, `AI_USAGE.md` |

## 3. How everything is structured

### Data flow

```text
Amazon Polarity (Hugging Face, pinned revision)
        |
   prepare_data.py ----> data/processed/{train,validation,test}.csv
        |                         |
        |        +----------------+------------------+---------------------+
        |        |                |                  |                     |
        |   pretrained BERT   train_tfidf.py   finetune_bert.py    finetune_scaling.py
        |   (3 candidates,    (3 classifiers)  (3 recipes, GPU)    (3 seeds x 3 sizes, GPU)
        |    validation only)       |                  |                     |
        |        |                  +--------> MLflow <-+---------------------+
        |        v
        |   train_bert.py ----> models/nlptown_sentiment/ ----> app.py (FastAPI)
        |
   evaluate_bert.py ----> data/predictions/bert_test.csv + metrics
```

### Repository layout

```text
ReviewPulse/
├── .github/workflows/ci.yml   # ruff + pytest on every change
├── data/                      # generated: processed splits and predictions
├── docs/                      # data report, MLflow run export, this document
├── notebooks/                 # Kaggle notebooks and the candidate MLflow database
├── scripts/                   # merge_mlflow_runs.py: copy runs between MLflow databases
├── models/                    # generated model weights (not committed)
├── src/
│   ├── common.py              # shared constants, metrics, prediction IO, Predictor interface
│   ├── prepare_data.py        # download, clean, split
│   ├── bert_model.py          # BERT adapter used by the API and evaluation
│   ├── train_bert.py          # save the selected pretrained model (no training)
│   ├── evaluate_bert.py       # score the 600-review test set once
│   ├── tfidf_model.py         # TF-IDF adapter
│   ├── train_tfidf.py         # train, select, score the baseline
│   ├── finetune_bert.py       # three fine-tuning recipes
│   ├── finetune_scaling.py    # seeds and training-size study
│   └── app.py                 # FastAPI service
└── tests/
```

`src/common.py` is the shared contract: every model adapter implements the same `Predictor` interface, and every prediction file has the columns `row_id`, `label_true`, `pred`, `score`. That is what allows any two models to be compared row by row.

### Design decisions

- **Selection on validation only.** Every choice (candidate, classifier, recipe, epoch, best seed) is made on the 1,000 validation reviews. The 600 test reviews are only used to report results.
- **Same splits for every model.** All models use the same seed, dataset revision and rows, so the comparison is like for like.
- **Label mapping.** The nlptown model predicts 1 to 5 stars. 1 to 3 stars map to negative and 4 to 5 stars to positive, so 3-star reviews count as negative.
- **Fine-tuning keeps the 5-star head.** The loss uses the mapped binary probability, so the saved model is a drop-in replacement for the API.
- **Reproducibility.** Pinned dataset revision, fixed seeds, MD5 hashes of the splits compared between Kaggle and the laptop, and pinned dependency versions.

## 4. Data

- **Dataset:** `fancyzhx/amazon_polarity`, revision `9d9c45c18f8c3cf1b23a3c27917b60cbf28f3289`. Input is the `content` field; the title is not used.
- **Cleaning:** missing or empty reviews removed; 5,217 exact duplicates removed from the official training split and 83 from the official test split; 970 test-pool reviews that also occur in the training pool excluded.
- **Splits:** training and validation sampled from the official training split, test sampled from the official test split; plain random sampling with seed 42.

| Split | Reviews | Negative | Positive |
|---|---:|---:|---:|
| Training | 10,000 | 4,895 | 5,105 |
| Validation | 1,000 | 509 | 491 |
| Test | 600 | 302 | 298 |

For the data-size study, up to 30,000 more training reviews (40,000 in total) come from the rest of the official training split, with nothing that appears in validation or test. The original 10,000 are always the first rows, so the three training sets are nested.

## 5. Models and results

### 5.1 Pretrained candidates (validation, 1,000 reviews)

| Candidate | Accuracy | Macro-F1 |
|---|---:|---:|
| DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`) | 0.8910 | 0.8909 |
| RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment`) | 0.8970 | 0.8969 |
| **nlptown BERT** | **0.9300** | **0.9298** |

### 5.2 Held-out test, pretrained nlptown BERT (600 reviews)

Accuracy 0.9283, macro-F1 0.9282, 557 of 600 correct (95% interval about 90.5% to 94.6%).

| Actual / Predicted | Negative | Positive |
|---|---:|---:|
| Negative | 289 | 13 |
| Positive | 30 | 268 |

Positives are missed more often than negatives, which is consistent with treating 3-star reviews as negative.

### 5.3 TF-IDF baseline

Three classifiers on TF-IDF features were fitted on the 10,000 training reviews; the linear SVM was best on validation (accuracy 0.8730) and was scored once on test: accuracy 0.8683, macro-F1 0.8683, confusion matrix `[[266, 36], [43, 255]]`, 79 errors.

### 5.4 Fine-tuning

**Recipes on 10,000 reviews (validation).**

| Recipe | Learning rate | Epochs | Best epoch | Accuracy |
|---|---:|---:|---:|---:|
| Pretrained | – | – | – | 0.9300 |
| **`lr1e5_ep2`** | 1e-5 | 2 | 1 | **0.9440** |
| `lr2e5_ep3` | 2e-5 | 3 | 2 | 0.9380 |
| `lr3e5_ep3_freeze6` | 3e-5 | 3 | 3 | 0.9330 |

The recipes are within 1.1 points, which is inside the noise of 1,000 reviews.

**Seeds and training size** (recipe `lr1e5_ep2`, three seeds per size).

| Training reviews | Validation accuracy | Test accuracy | Test range | Minutes per run |
|---:|---:|---:|---:|---:|
| Pretrained | 0.9300 | 0.9283 | – | – |
| 10,000 | 0.9383 ± 0.0049 | 0.9367 ± 0.0076 | 0.9283–0.9433 | 4.7 |
| 20,000 | 0.9450 ± 0.0026 | 0.9456 ± 0.0025 | 0.9433–0.9483 | 9.3 |
| 40,000 | 0.9507 ± 0.0015 | 0.9439 ± 0.0069 | 0.9383–0.9517 | 18.4 |

- Mean test accuracy over all nine runs: **0.9420**, against 0.9283 for the pretrained model.
- More data lifts validation accuracy steadily. On the 600-review test set, 20,000 and 40,000 cannot be told apart; the standard error of one accuracy is about 1 point, as large as the spread between seeds.
- The first single-seed result (0.9433 at 10,000) was the top of the seed range, so the mean over seeds is the honest headline.
- Averaging the three seeds' probabilities did not help on test (0.943 to 0.947).

**Selected model:** `scale_n40000_seed43`, chosen on validation (accuracy 0.9520). Test accuracy 0.9517, macro-F1 0.9517, 571 of 600 correct (95% interval 0.931 to 0.966).

| Actual / Predicted | Negative | Positive |
|---|---:|---:|
| Negative | 288 | 14 |
| Positive | 15 | 283 |

| Paired comparison (same 600 reviews) | Fixed | Broken | Exact McNemar p |
|---|---:|---:|---:|
| Pretrained to selected | 19 | 5 | 0.0066 |
| First fine-tuned (10,000, 1 seed) to selected | 10 | 5 | 0.30 |

The first p-value is for the best of nine runs and is therefore optimistic. Of the selected model's 29 errors, 22 were also wrong for the pretrained and the first fine-tuned model, with a mean confidence of 0.71.

### 5.5 Three models on the same test reviews

| Model | Accuracy | Macro-F1 | Errors | Negative recall | Positive recall |
|---|---:|---:|---:|---:|---:|
| TF-IDF + linear SVM | 0.8683 | 0.8683 | 79 | 0.881 | 0.856 |
| nlptown BERT, pretrained | 0.9283 | 0.9282 | 43 | 0.957 | 0.899 |
| nlptown BERT, fine-tuned (selected run) | 0.9517 | 0.9517 | 29 | 0.954 | 0.950 |

BERT is clearly better than TF-IDF. Even in the least favourable split of disagreements, a McNemar test of the fine-tuned model against TF-IDF gives p below 0.0001; the exact value needs the row-by-row TF-IDF predictions (see open items).

## 6. Experiment tracking

Experiments are tracked in MLflow 3.18.0 with a SQLite backend (`sqlite:///reviewpulse_mlflow.db`); the plain `./mlruns` folder is no longer accepted by MLflow 3.

| Experiment | Runs |
|---|---|
| `reviewpulse-bert` | Three pretrained candidates, the saved-model run from `train_bert.py`, the fine-tuning recipes (`finetune_<recipe>`, `finetune_selected_<recipe>`), and the nine seed and size runs (`scale_n<size>_seed<seed>`) with a `scaling_selected_<run>` summary |
| `reviewpulse-tfidf` | One run per TF-IDF classifier and one for the selected model |

Each run records its parameters, the validation metrics per epoch where relevant, training time, and tags (model family, stage). The fine-tuning runs were logged on Kaggle, in `finetune_mlflow.db` and `scaling_mlflow.db`, and are copied into the local database with `scripts/merge_mlflow_runs.py`. The script keeps each run's name, times, params and full epoch-by-epoch metrics, skips runs that were already merged, and only reads the source database. MLflow databases are local and not committed; `docs/mlflow_runs_summary.csv` is a snapshot of the main runs and should be re-exported after merging.

## 7. Service, testing and CI

- **API.** `GET /health` returns the status and the served model's name. `POST /predict` takes `{"text": "..."}` and returns a sentiment and a score. The score is the model's confidence in its original star prediction; it is not a calibrated probability of the binary sentiment. Interactive docs are at `/docs`.
- **Tests.** pytest covers data preparation, shared code, the BERT and TF-IDF adapters, MLflow logging, the API endpoints, and the pure logic of the two fine-tuning scripts. Tests that need PyTorch are skipped when it is not installed; the training itself is checked by the `--smoke` run in the Kaggle notebooks.
- **CI.** GitHub Actions job `lint-and-test` runs `ruff check` and `pytest` on every change. One CI run caught a real problem: a TF-IDF test pointed MLflow at a plain folder, which MLflow 3 rejects but MLflow 2.19 on a teammate's laptop accepted. The test now uses a temporary SQLite database.

## 8. How to reproduce

```bash
python -m pip install -r requirements.txt
python -m src.prepare_data                 # splits and data report
python -m src.train_bert                   # save the pretrained model
python -m src.evaluate_bert                # test predictions and metrics (run once)
python -m src.train_tfidf                  # TF-IDF baseline
python -m uvicorn src.app:app --reload     # API at http://127.0.0.1:8000/docs
python -m pytest -v && python -m ruff check .
```

Fine-tuning needs a GPU. Use the Kaggle notebooks in `notebooks/` (GPU T4 x2): they clone the repository, build the splits, run a smoke test, train, log to MLflow and package the results. The seeds and size study takes about 50 minutes on two T4s.

## 9. Team workflow

Work is done on short-lived branches (`feat/...`, `fix/...`, `infra/...`), merged into `main` through pull requests that CI must pass and a teammate must review. Dependencies are pinned in `requirements.txt`. Weights, databases, virtual environments and keys are not committed. AI-assisted development is disclosed in `AI_USAGE.md`, and individual contributions are in `CONTRIBUTION.md`.

## 10. Limitations and open items

**Limitations**

- The test set has 600 reviews, so one accuracy has a standard error of about 1 point. Differences of one or two points between models are often not distinguishable.
- The selected fine-tuned run is the best of nine on validation, so its test score is a slightly optimistic pick; the mean over seeds (0.944 at 40,000 reviews) is the fair figure.
- Treating 3-star reviews as negative is a design choice that lowers recall on positives for the pretrained model.
- The score returned by the API is not calibrated.
- Amazon Polarity labels are noisy and ambiguous for some reviews, which probably limits any model to about 95 to 96%.

**Open items**

- Merge `src/finetune_bert.py`, `src/finetune_scaling.py`, their tests, `scripts/merge_mlflow_runs.py` and the two Kaggle notebooks into `main`.
- Serve the fine-tuned model from the API by pointing the BERT adapter at `models/nlptown_finetuned_v2/`; the API currently serves the pretrained model.
- Commit `data/predictions/tfidf_test.csv` so the exact paired test against TF-IDF can be computed.
- Release steps: tag `v1.0-milestone1`, create the archive, and test a clean clone.
- Back up `reviewpulse_mlflow.db` outside the repository (an earlier `git pull` removed a local copy).
