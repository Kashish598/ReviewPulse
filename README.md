# ReviewPulse

A sentiment analysis service for Amazon product reviews. ReviewPulse prepares a reproducible dataset, evaluates pretrained BERT sentiment models, fine-tunes the selected one, compares it with a TF-IDF baseline, tracks every experiment with MLflow, and serves the selected model through a FastAPI REST API. The service runs in Docker, is checked by GitHub Actions on every push, is deployed on Render, and watches its own traffic: it counts and times requests, saves the reviews it receives, labels them in the background with an LLM, and sends an email alert when incoming reviews stop looking like the training data (data drift).

**Status:** Milestone 2: containerisation, CI/CD, deployment and monitoring are in place, and the Project Focus (drift detection and a retraining loop) is under way (see the last section). The Milestone 1 work (model evaluation, TF-IDF baseline, fine-tuning study and API) is documented below; see `docs/MILESTONE_1_COMPLETION.md` for its summary.

- **Live API:** https://reviewpulse-0j1b.onrender.com (interactive docs at `/docs`). *The service is suspended outside grading hours to save cost. The first request after a pause can take a minute or more: call `/health` first and wait until `/ready` answers 200.*
- **Container image:** [`ghcr.io/kashish598/reviewpulse`](https://github.com/Kashish598/ReviewPulse/pkgs/container/reviewpulse)
- **Image running in production:** `sha-dd0305d`
- **Redeploy, roll back, logs:** [RUNBOOK.md](RUNBOOK.md)

## Architecture

```mermaid
flowchart LR
    A[Amazon Polarity Dataset] --> B[prepare_data.py]
    B --> C[Train / Validation / Test CSVs]
    C --> D[Candidate Model Evaluation]
    D --> E[MLflow Experiment Tracking]
    D --> F[Selected nlptown BERT]
    F --> G[train_bert.py: Save Model Locally]
    G --> H[FastAPI: app.py]
    C --> I[evaluate_bert.py]
    H --> J[POST /predict]
    I --> K[Test Metrics and Predictions]
    C --> L[train_tfidf.py: TF-IDF Baseline]
    C --> M[finetune_bert.py / finetune_scaling.py: Fine-tuning on GPU]
    L --> E
    M --> E
    M --> N[Fine-tuned nlptown model]
    H --> P["/stats and log file"]
    H --> Q[Drift detector]
    Q --> R[Email alert]
    H --> S[Request store]
    S --> T[LLM labeller]
    T --> U[Retraining]
```

The API serves the fine-tuned BERT model when it is present locally (the pretrained model otherwise). Model selection is based on validation-set performance. The held-out test set is reserved for final evaluation. Fine-tuning runs on a GPU (Kaggle); everything else runs on a CPU. In production the API runs as a Docker image on Render (see CI/CD and Deployment).

## Technology Stack

- **Language:** Python 3.11
- **Data processing:** pandas, NumPy, Hugging Face Datasets
- **Machine learning:** scikit-learn, PyTorch, Transformers (fine-tuning needs a GPU; the Kaggle notebooks install what they need)
- **Experiment tracking:** MLflow
- **API:** FastAPI, Pydantic, Uvicorn
- **Testing and code quality:** pytest, Ruff
- **Packaging and delivery:** Docker, GitHub Actions, GitHub Container Registry, Render
- **Data loop and monitoring:** Neon (PostgreSQL) with SQLAlchemy for saved requests, Groq for LLM labelling, Python `logging`, drift detection (PSI), SMTP email alerts

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Kashish598/ReviewPulse.git
cd ReviewPulse
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the dataset

```bash
python -m src.prepare_data
```

This downloads the Amazon Polarity dataset from Hugging Face, cleans the review text, creates the train, validation, and test splits, and writes the processed files to `data/processed/`.

### 5. Save the selected model locally

```bash
python -m src.train_bert
```

This downloads the selected pretrained model and tokenizer and saves them under `models/nlptown_sentiment/`. This step does **not** fine-tune the model.

### 6. Run the API

```bash
python -m uvicorn src.app:app --reload
```

Open the interactive API documentation at:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 7. Run tests and lint checks

```bash
python -m pytest -v
python -m ruff check .
python -m ruff format --check src tests scripts
```

### 8. Run it in Docker

One command builds the image and one runs it. The build downloads the fine-tuned model (about 640 MB) into the image, so allow some minutes the first time.

```bash
docker build --build-arg MODEL_HF_REPO=Musknnnn/nlptown-sentiment -t reviewpulse:local .
docker run --rm -p 10000:10000 -e PORT=10000 reviewpulse:local
```

Open http://localhost:10000/docs. In a second terminal, `docker ps` shows `(healthy)` in the STATUS column once the model has loaded.

Without settings the container saves requests to a local SQLite file inside the container, does not label them and sends no alerts. To switch those on, pass your settings from a `.env` file (copy `.env.example`; write values without quotes and without spaces around `=`):

```bash
docker run --rm -p 10000:10000 -e PORT=10000 --env-file .env reviewpulse:local
```

To run the published image instead of building it, use a pinned `sha-` tag (the tag running in production is listed at the top of this file):

```bash
docker run --rm -p 10000:10000 -e PORT=10000 ghcr.io/kashish598/reviewpulse:sha-<SHORT_COMMIT>
```

Add `--env-file .env` as above to switch on the database, labeller and alerts.

## Dataset and Task

- **Dataset:** [`fancyzhx/amazon_polarity`](https://huggingface.co/datasets/fancyzhx/amazon_polarity)
- **Task:** Binary sentiment classification.
- **Labels:** `0` = negative, `1` = positive.
- **Model input:** The review `content` field. The `title` field is not used.
- **Random seed:** `42`.
- **Dataset revision:** Pinned in `src/common.py` to support reproducibility.

### Dataset Splits

| Split | Number of reviews | Purpose |
|---|---:|---|
| Training | 10,000 (up to 40,000 for the data-size study) | Trains the TF-IDF baseline and the fine-tuned BERT; the pretrained BERT candidates do not use it |
| Validation | 1,000 | Candidate model selection |
| Test | 600 | Final held-out evaluation |

Training and validation examples are sampled from the official training split. Test examples are sampled from the official test split.

Sampling is not stratified. The data preparation pipeline removes missing or empty reviews, removes exact duplicates before sampling, and excludes test-pool reviews that also occur in the training pool. Text is otherwise preserved, including casing, punctuation, stop words, and emojis.

For the data-size study, `python -m src.finetune_scaling prepare` adds up to 30,000 more training reviews from the rest of the official training split (duplicates removed, nothing that occurs in validation or test). The original 10,000 reviews are always the first rows, so the 10,000, 20,000 and 40,000 training sets are nested. Validation and test never change.

The dataset files are generated locally and are not required to be committed to the repository. The pinned dataset revision and random seed are used to make the preparation process reproducible.

## Model Selection

Three pretrained sentiment-analysis checkpoints were evaluated on the same validation set. Macro-F1 was the primary selection metric; accuracy was used as a secondary metric.

| Candidate | Validation Accuracy | Validation Macro-F1 |
|---|---:|---:|
| DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`) | 0.8910 | 0.8909 |
| RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment`) | 0.8970 | 0.8969 |
| **nlptown BERT (`nlptown/bert-base-multilingual-uncased-sentiment`)** | **0.9300** | **0.9298** |

The nlptown checkpoint was selected because it achieved the highest validation accuracy and macro-F1 among the evaluated candidates.

The selected checkpoint is a pretrained five-star sentiment model. ReviewPulse maps its output to binary sentiment labels as follows:

- 1, 2, and 3 stars → `negative`
- 4 and 5 stars → `positive`

The pretrained model is used as is for the candidate comparison and the first test evaluation. It is then fine-tuned (see Fine-Tuning below).

### MLflow Tracking

Experiments are tracked with MLflow:

| Experiment | Contents |
|---|---|
| `reviewpulse-bert-candidates` | The three candidate runs from `notebooks/bert_candidate_selection.ipynb` (Kaggle GPU), kept in the original Kaggle database `notebooks/kaggle_mlflow.db`. |
| `reviewpulse-bert` | The same three candidate runs copied into the repository's experiment (tag `source=kaggle_mlflow_db`), the run logged by `python -m src.train_bert`, and the fine-tuning runs (`finetune_<recipe>`, `finetune_selected_<recipe>`, `scale_n<size>_seed<seed>`, `scaling_selected_<run>`). |
| `reviewpulse-tfidf` | One run per TF-IDF classifier plus one for the selected model. |

Each candidate run records the model checkpoint, label-mapping policy, truncation setting, maximum sequence length (512) and the validation accuracy and macro-F1. `docs/mlflow_runs_summary.csv` is an export of the runs in `reviewpulse-bert`, including the three candidates, with their parameters, metrics and run IDs.

MLflow 3 no longer accepts the old `./mlruns` folder as a tracking location, so use a database backend. Set the tracking URI before running any script that logs to MLflow, and use the same value for `mlflow ui`.

Windows (PowerShell):

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///reviewpulse_mlflow.db"
mlflow ui --backend-store-uri sqlite:///reviewpulse_mlflow.db
```

macOS/Linux:

```bash
export MLFLOW_TRACKING_URI=sqlite:///reviewpulse_mlflow.db
mlflow ui --backend-store-uri sqlite:///reviewpulse_mlflow.db
```

The candidate runs were logged with MLflow 3.18.0, the version used by the project. The fine-tuning runs were logged on Kaggle, in `finetune_mlflow.db` and `scaling_mlflow.db`. Copy them into your local database with `scripts/merge_mlflow_runs.py`, which keeps each run's name, times, params and full epoch-by-epoch metric history (it does not copy the small files logged as artifacts, which are in `scaling_results.zip`). The source database is only read, and a run that was already merged is skipped, so running it twice is safe.

```bash
cp reviewpulse_mlflow.db ../reviewpulse_mlflow.backup.db       # back up first (PowerShell: Copy-Item)
python scripts/merge_mlflow_runs.py scaling_mlflow.db --dry-run  # list the runs, change nothing
python scripts/merge_mlflow_runs.py scaling_mlflow.db            # merge into reviewpulse-bert
python scripts/merge_mlflow_runs.py finetune_mlflow.db           # the first study's three recipes
mlflow ui --backend-store-uri sqlite:///reviewpulse_mlflow.db
```

MLflow databases and run artifacts are generated locally and are not committed. `docs/mlflow_runs_summary.csv` is a snapshot; re-export it after merging if it should include the fine-tuning runs. The validation metrics above and the CSV files in `docs/` are the reported comparison results.

## Held-Out Test Evaluation

The selected nlptown BERT model was evaluated once on the held-out test set of 600 reviews. These examples were not used for candidate selection or model tuning.

| Metric | Result |
|---|---:|
| Accuracy | 0.9283 |
| Macro-F1 | 0.9282 |
| Correct predictions | 557 / 600 |

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Negative | 0.91 | 0.96 | 0.93 | 302 |
| Positive | 0.95 | 0.90 | 0.93 | 298 |

### Confusion Matrix

Rows represent actual labels; columns represent predicted labels.

| Actual / Predicted | Negative | Positive |
|---|---:|---:|
| Negative | 289 | 13 |
| Positive | 30 | 268 |

The model correctly classified 557 of the 600 test reviews. The final test accuracy was 92.8%, with a macro-F1 score of approximately 0.9282. A 95% confidence interval for the accuracy is roughly 90.8% to 94.9%. The model misses positive reviews (30) more often than negative ones (13), which is consistent with treating 3-star reviews as negative.

### Reproduce the Test Evaluation

After preparing the data and saving the model locally, run:

```bash
python -m src.evaluate_bert
```

This scores the pretrained model in `models/nlptown_sentiment/`; it is the baseline that the fine-tuned model is compared with. The fine-tuned model's test predictions come from `src/finetune_scaling.py` (see Fine-Tuning).

The evaluation script runs inference on the held-out test split and saves `bert_test.csv` (predictions) and `bert_test_metrics.json` (accuracy, macro-F1 and confusion matrix) under `data/predictions/`. The test split should be evaluated once, with the final configuration.

## TF-IDF Baseline

A classical baseline is trained on the same data splits as the BERT models, so the two can be compared fairly. It uses TF-IDF features with one of three classifiers (linear SVM with calibrated probabilities, multinomial Naive Bayes, random forest). All three are fitted on the 10,000 training reviews, the best one is chosen on the 1,000 validation reviews (macro-F1, then accuracy), and only that model is scored on the 600 test reviews, once. It runs on a CPU and needs no GPU.

```bash
python -m src.train_tfidf               # trains, selects, saves the model, scores the test set
python -m src.train_tfidf --no-mlflow   # same, without MLflow tracking
```

Outputs:

- `models/tfidf_v1/`: `model.joblib` and `metadata.json` (not committed).
- `data/predictions/tfidf_test.csv`: row-by-row test predictions, in the same format as `bert_test.csv`.
- `data/predictions/tfidf_test_metrics.json`: test metrics of the selected model.
- MLflow experiment `reviewpulse-tfidf`: one run per candidate classifier (validation metrics, fit time, tags) plus one run for the selected model.

Selected model: linear SVM.

| Model | Validation accuracy | Validation macro-F1 | Test accuracy | Test macro-F1 | Test errors (of 600) |
|---|---:|---:|---:|---:|---:|
| TF-IDF + linear SVM | 0.8730 | 0.8730 | 0.8683 | 0.8683 | 79 |
| nlptown BERT (no fine-tuning) | 0.9300 | 0.9298 | 0.9283 | 0.9282 | 43 |
| nlptown BERT, fine-tuned (40,000 reviews, seed 43) | 0.9520 | 0.9520 | 0.9517 | 0.9517 | 29 |

TF-IDF confusion matrix on the test set (rows are the true label, columns the prediction): `[[266, 36], [43, 255]]`. On the same 600 reviews, the pretrained BERT made 36 fewer errors than the baseline and the fine-tuned model 50 fewer. Both models use the same split and seed, so the comparison is like for like. A row-by-row paired test needs `tfidf_test.csv`.

`MLFLOW_TRACKING_URI` should be set the same way as for BERT (see MLflow Tracking above), so both experiments are stored in the same database.

## Fine-Tuning

The selected nlptown model was fine-tuned on the training split, on a GPU (Kaggle, 2 × T4). Both scripts are in `src/`; the Kaggle notebooks in `notebooks/` run them end to end and log to MLflow.

**Method.** The 5-star output layer is kept. The training loss is the negative log-likelihood of the *mapped binary* probability (1–3 stars against 4–5 stars, computed with a log-sum-exp over the stars), so training matches how the model is scored and the saved model stays a drop-in replacement. Serving still takes the most likely star and maps it. Training uses AdamW (no weight decay on bias and LayerNorm), a linear warm-up (10%) and decay, gradient clipping at 1.0, fp16 on the GPU, batch size 16, maximum length 256 for training and 512 for evaluation. The best epoch is kept, chosen on validation. The test split is never used to choose anything.

### 1. Three recipes on 10,000 reviews

```bash
python -m src.finetune_bert                 # all three recipes, picks the best on validation
python -m src.finetune_bert --smoke         # tiny end-to-end check
```

| Recipe | Learning rate | Epochs | Best epoch | Validation accuracy | Validation macro-F1 |
|---|---:|---:|---:|---:|---:|
| Pretrained, no fine-tuning | – | – | – | 0.9300 | 0.9298 |
| **`lr1e5_ep2`** | 1e-5 | 2 | 1 | **0.9440** | **0.9440** |
| `lr2e5_ep3` | 2e-5 | 3 | 2 | 0.9380 | 0.9380 |
| `lr3e5_ep3_freeze6` (lower 6 layers frozen) | 3e-5 | 3 | 3 | 0.9330 | 0.9330 |

The three recipes are within 1.1 points of each other, which is inside the noise of a 1,000-review validation set, and one seed cannot say whether 0.944 is typical. The next study checks both.

### 2. Seeds and training size

```bash
python -m src.finetune_scaling prepare                         # builds data/processed/train_extended.csv (40,000 rows)
python -m src.finetune_scaling train --worker 0 --workers 2    # GPU 0
python -m src.finetune_scaling train --worker 1 --workers 2    # GPU 1, at the same time
python -m src.finetune_scaling summarize                       # select on validation, re-score, log to MLflow
```

Recipe `lr1e5_ep2`, three seeds (42, 43, 44) at 10,000, 20,000 and 40,000 training reviews: nine runs. The workers record what they would log; `summarize` replays it into MLflow from one process and keeps the best run (best validation macro-F1) as `models/nlptown_finetuned_v2/`. The test set is scored for every run so the spread between seeds can be reported, and is never used to choose a model.

| Training reviews | Validation accuracy (mean ± std) | Test accuracy (mean ± std) | Test range | Minutes per run |
|---:|---:|---:|---:|---:|
| Pretrained | 0.9300 | 0.9283 | – | – |
| 10,000 | 0.9383 ± 0.0049 | 0.9367 ± 0.0076 | 0.9283–0.9433 | 4.7 |
| 20,000 | 0.9450 ± 0.0026 | 0.9456 ± 0.0025 | 0.9433–0.9483 | 9.3 |
| 40,000 | 0.9507 ± 0.0015 | 0.9439 ± 0.0069 | 0.9383–0.9517 | 18.4 |

**How to read this**

- The mean test accuracy over all nine runs is **0.9420**, 1.4 points above the pretrained model (0.9283). Eight runs beat it and one ties it.
- More data raises validation accuracy steadily (0.938, 0.945, 0.951). On a 600-review test set, 20,000 and 40,000 reviews cannot be told apart: the standard error of one accuracy is about 1 point, as large as the spread between seeds.
- The first single-seed result (0.9433 at 10,000 reviews) is the top of the seed range at that size, so quote the mean over seeds as the headline.
- Averaging the three seeds' probabilities did not help on test (0.943–0.947).

### Selected fine-tuned model

`scale_n40000_seed43` (chosen on validation: accuracy 0.9520, macro-F1 0.9520). Test accuracy **0.9517** (571 / 600; 95% Wilson interval 0.931–0.966), macro-F1 0.9517.

| Actual / Predicted | Negative | Positive |
|---|---:|---:|
| Negative | 288 | 14 |
| Positive | 15 | 283 |

Paired against the other models on the same 600 reviews (exact McNemar test):

| Comparison | Fixed | Broken | p |
|---|---:|---:|---:|
| Pretrained → selected | 19 | 5 | 0.0066 |
| First fine-tuned (10,000, 1 seed) → selected | 10 | 5 | 0.30 |

This p-value is for the best of nine runs, so it is optimistic; the mean over seeds above is the fairer summary. Of the selected model's 29 errors, 22 were also wrong for the pretrained and the first fine-tuned model (mean confidence 0.71), which suggests ambiguous or mislabeled reviews rather than a lack of data.

Outputs: `models/nlptown_finetuned_v2/` (weights, tokenizer, `metadata.json`), `data/predictions/bert_finetuned_v2_test.csv`, `data/predictions/bert_finetuned_v2_test_metrics.json`, and `data/predictions/scaling/` (per-run predictions and probabilities, `runs_table.csv`, `by_size.csv`, `report.json`). Model weights are not committed.

**Serving it.** The API serves the fine-tuned model when `models/nlptown_finetuned_v2/` exists, and otherwise falls back to the pretrained model in `models/nlptown_sentiment/` (with a warning in the log). The Kaggle run saves the model as `nlptown_finetuned_v2.zip`; unzip it so the folder looks like this, then start the API as usual:

```text
models/nlptown_finetuned_v2/
├── config.json, model.safetensors (or pytorch_model.bin), tokenizer files
└── metadata.json
```

`GET /health` shows which model was loaded (`"model": "nlptown-bert-finetuned", "fine_tuned": true`). To serve a different folder, set `REVIEWPULSE_MODEL_DIR` before starting the API. Model weights are not committed, so every teammate unzips the file locally.

## API Usage

Start the API:

```bash
python -m uvicorn src.app:app --reload
```

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness: the process is up. Answers at once and never loads the model. Names the model (and whether it is fine-tuned) once it is loaded. |
| GET | `/ready` | Readiness: 200 when the model is loaded, 503 (`loading` or `error`) otherwise. |
| POST | `/predict` | Sentiment for `{"text": "..."}` (1 to 5000 characters). Empty, blank, missing or over-long text gets HTTP 422. |
| GET | `/stats`, `/metrics` | Request count, average API latency and average model latency, in total and per model with tokenize, inference and label-mapping times. Counts successful `/predict` calls; resets on restart. |
| GET | `/drift` | Drift status of recent traffic against the training data: `ok`, `drift`, `insufficient_data` (window still filling) or `disabled`. |
| GET | `/docs` | Interactive Swagger UI. |

### Sentiment prediction

```http
POST /predict
Content-Type: application/json
```

Example request:

```json
{
  "text": "I absolutely love this product. It works perfectly!"
}
```

The endpoint returns a sentiment prediction and its associated model score. The score is the confidence associated with the model's original star prediction; it should not be interpreted as a calibrated probability of the mapped binary sentiment.

> **Privacy.** Reviews sent to `/predict` are stored, together with the model's answer, and may be used to retrain the model. Nothing else about the caller is stored. Do not send personal data.

Interactive request and response schemas are available through `/docs`.

### Example requests against the live service

```bash
curl https://reviewpulse-0j1b.onrender.com/health
curl https://reviewpulse-0j1b.onrender.com/ready
curl -X POST https://reviewpulse-0j1b.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Great product, works well."}'
curl https://reviewpulse-0j1b.onrender.com/stats
curl https://reviewpulse-0j1b.onrender.com/drift
```

PowerShell:

```powershell
$H = "https://reviewpulse-0j1b.onrender.com"
Invoke-RestMethod "$H/health"
Invoke-RestMethod -Method Post -Uri "$H/predict" -ContentType "application/json" -Body '{"text":"Great product, works well."}'
Invoke-RestMethod "$H/stats" | ConvertTo-Json -Depth 5
```

## Monitoring

- **Request count and latency:** `/stats` (and `/metrics`, the same JSON) returns `request_count`, `avg_api_latency_ms`, `avg_model_latency_ms` and a `per_model` block. The API time includes the model time, so it is always the larger.
- **Logs:** every component logs through Python's `logging` module to the console and to `logs/reviewpulse.log` (rotating, 5 MB, 5 backups). `LOG_LEVEL=DEBUG` also records what each stage received and returned. On Render the console output is in the service's Logs tab.
- **Data drift:** after each prediction the review's simple features are added to a rolling window of the last 200 requests and compared with a reference snapshot of the 10,000 training reviews (`data/reference/reference.json`) using the Population Stability Index. When the window drifts, the log records an alert and an email is sent, at most once per hour. `scripts/simulate_drift.py` sends normal or deliberately shifted traffic to a running API for a demonstration.

## CI/CD and Deployment

- **CI** (`.github/workflows/ci.yml`) runs on pushes to `main`, on pull requests, on version tags (`v*`) and manually: a **lint** job (`ruff check`, `ruff format --check`), a **test** job (`pytest`), and a **Docker build** job that needs both. On `main` and on tags the image is published to GitHub Container Registry as `sha-<short commit>`, `latest`, and the git tag.
- **Deployment:** Render runs the published image as a web service with health check path `/health`. The model needs about 1 GB of memory, so the service uses a 2 GB instance (Render's 512 MB free plan is too small). Image-backed services do not redeploy by themselves: a new image is deployed from the Render dashboard or with a deploy hook.
- **Redeploy and roll back:** see [RUNBOOK.md](RUNBOOK.md). Every image keeps its `sha-` tag, so an earlier version can be deployed again or run locally.

## Settings (environment variables)

Everything is optional locally. Copy `.env.example` to `.env` for a Docker run; on Render set them in the service's Environment tab. **Never commit `.env` or any key.**

| Variable | Purpose |
|---|---|
| `PORT` | Port to listen on (default 8000; Render sets its own; the Docker examples above use 10000). |
| `LOG_LEVEL`, `LIB_LOG_LEVEL`, `LOG_DIR` | Logging level for the app and for libraries (default `INFO`), log folder. |
| `MODEL_HF_REPO` | Hugging Face repo to fetch the fine-tuned model from (`Musknnnn/nlptown-sentiment`). Without it the public pretrained model is used. |
| `MODEL_URL`, `MODEL_SHA256` | Alternative source: a zip of the model folder and its checksum. |
| `REVIEWPULSE_MODEL_DIR`, `MODEL_PRELOAD` | Serve a model from another folder; `off` loads the model on the first request instead of at startup. |
| `DATABASE_URL`, `FEEDBACK_STORE` | Neon PostgreSQL connection string for saved requests (a SQLite file if unset); `off` stops saving. |
| `GROQ_API_KEY`, `LABEL_ENABLED`, `LABEL_DAILY_CAP`, `LABEL_MIN_INTERVAL_SECONDS`, `LABEL_POLL_SECONDS`, `LABEL_MODEL`, `LABEL_MAX_TOKENS` | LLM labeller: key (it stays off without one), switch, daily call limit, pacing, model. |
| `DRIFT_REFERENCE`, `DRIFT_WINDOW_SIZE`, `DRIFT_MIN_WINDOW`, `DRIFT_PSI_THRESHOLD` | Drift detector: reference file, window size (200), minimum requests before judging (100), PSI threshold (0.25). |
| `SMTP_USER`, `SMTP_PASSWORD`, `ALERT_TO`, `SMTP_HOST`, `SMTP_PORT`, `ALERT_FROM`, `ALERT_COOLDOWN_SECONDS` | Email alerts. For Gmail use an app password. Alerts are only logged when these are not set. |

## Shared Code (`src/common.py`)

| Component | Purpose |
|---|---|
| `load_split(name, sample=False)` | Loads a prepared dataset split. |
| `compute_metrics(y_true, y_pred)` | Computes macro-F1, accuracy, and confusion matrix. |
| `Predictor`, `Prediction` | Shared interface and prediction representation for model adapters. |
| `write_predictions(...)`, `read_predictions(...)` | Writes and reads prediction CSV files. |
| `MLFLOW_BERT_EXPERIMENT` | Shared MLflow experiment name for BERT experiments. |

Predictions use the columns `row_id`, `label_true`, `pred`, and `score`.

## Repository Layout

```text
ReviewPulse/
├── .github/
│   └── workflows/
│       └── ci.yml             # lint, test, Docker build and publish
├── data/
│   ├── sample/                # Small test fixtures, if included
│   ├── processed/             # Generated dataset splits (not committed)
│   ├── predictions/           # Evaluation predictions and metrics
│   └── reference/             # Drift reference snapshot (committed)
├── docs/                      # data report, MLflow summary, Milestone 1 report, labeller validation, retraining notes
├── notebooks/                 # Kaggle notebooks (candidate selection, fine-tuning, seeds and size) and the candidate MLflow database
├── models/                    # Locally saved model weights (not committed)
├── src/
│   ├── app.py                 # FastAPI service
│   ├── common.py
│   ├── prepare_data.py
│   ├── train_bert.py, evaluate_bert.py, bert_model.py
│   ├── train_tfidf.py, tfidf_model.py
│   ├── finetune_bert.py       # three fine-tuning recipes (GPU)
│   ├── finetune_scaling.py    # seeds x training size study (GPU)
│   ├── logging_config.py, monitoring.py
│   ├── drift.py, alerts.py
│   ├── feedback_store.py, labeler.py, retrain.py
├── scripts/                   # model fetch, healthcheck, drift reference and simulator, labeller validation, benchmark, requirements check, MLflow merge
├── tests/
├── Dockerfile, .dockerignore
├── requirements.txt, requirements-serve.txt, pyproject.toml
├── .env.example
├── README.md, RUNBOOK.md, CONTRIBUTION.md, AI_USAGE.md, LICENSE
```

Generated dataset splits, model weights, virtual environments, MLflow run artifacts, and local database files should not be committed unless explicitly required for a particular deliverable.

## Team Workflow

- Use descriptive branches such as `feat/...`, `fix/...`, and `infra/...`.
- Open a pull request for changes and obtain approval from another team member before merging into `main`.
- Run Ruff and pytest before pushing changes.
- CI runs `ruff check`, `ruff format --check src tests scripts` and `pytest`, then builds the Docker image. A pull request needs a green run and one approval.
- Add new dependencies to `requirements.txt` with pinned versions in the same change that introduces them. If the API needs them at run time, add the same pins to `requirements-serve.txt` too (`scripts/check_requirements_sync.py` checks that both files agree).
- Do not commit API keys, virtual environments, generated model weights, or local MLflow artifact directories.

## Tests

Run the complete test suite:

```bash
python -m pytest -v
```

The tests cover data preparation, shared utilities, the BERT and TF-IDF model adapters, TF-IDF MLflow logging, the FastAPI endpoints (including `/ready`, `/stats` and `/drift`), logging, request and latency monitoring, drift detection and email alerts (with a mocked mail server), the request store and the LLM labeller (without a real database or network), retraining, the model fetch script and the requirements check, and the pure logic of the two fine-tuning scripts (configs, selection rules, the extended training set, statistics, job splitting, MLflow replay). The tests that need PyTorch are skipped when it is not installed, so CI does not need a GPU; the training itself is checked by the smoke test (`--smoke`) in the Kaggle notebooks.

## Additional Documentation

- `RUNBOOK.md`: how to check, redeploy and roll back the service, run an older image locally, and find the logs.
- `CONTRIBUTION.md`: team responsibilities and contributions.
- `AI_USAGE.md`: disclosure of AI-assisted development and review.
- `docs/MILESTONE_1_COMPLETION.md`: what was delivered in Milestone 1, how it is structured, the results and the known limitations.
- `docs/labeler_eval.md`: validation of the LLM labeller on the human-labelled test reviews.
- `docs/retraining.md`: how retraining on labelled requests works.
- `LICENSE`: MIT.

## Project Focus: data drift detection and the retraining loop

**What the focus is.** ReviewPulse watches the reviews that people send to `/predict` and notices
when they stop looking like the data the model was trained on. When that happens it sends an email
alert. Every request is also saved, labelled by an LLM in the background, and kept as new training
data for a retraining step. A retrained model replaces the current one only if it clears a margin
on validation and is not worse on the original human-labelled test set.

**Why it suits this project.** A sentiment model is trained once on 10,000 product reviews, but the
reviews it sees later change: new products, other languages (Urdu, Hindi, emoji), much shorter or
longer text. Nobody labels production traffic, so the model can get worse without any error
showing. Input drift can be measured without labels, and an LLM can supply labels cheaply, which
makes drift detection plus retraining a natural fit for a text classifier behind an API.

**What we build and where it stands.**

| Part | What it does | Status |
|---|---|---|
| Drift detector (`src/drift.py`, `scripts/build_drift_reference.py`) | Compares a rolling window of requests with a snapshot of the 10,000 training reviews: text length, word count, share of non-ASCII characters, unknown-word rate and share of "positive" predictions, using PSI with a threshold of 0.25. | Built, tested and connected to `/predict`; `/drift` shows the current status. On clean data 0 of 5 windows of 200 reviews were flagged. Shifted traffic (short text, emoji, other scripts, all-positive answers) is flagged. |
| Email alert (`src/alerts.py`) | Sends one email naming the drifted features and their PSI, with a cooldown so it does not repeat. | Built and tested; a real email was received during development. |
| Request store (`src/feedback_store.py`) | Saves each request with the model's answer in Postgres (Neon) or SQLite, with a fixed train/validation split and label status. | Built, tested on SQLite and on real Postgres, and connected to `/predict`. |
| LLM labeller (`src/labeler.py`) | Labels saved requests in the background with Groq (`openai/gpt-oss-20b`, temperature 0, one word). Anything that is not exactly `negative` or `positive` is never used as a label. Daily cap, rate-limit handling, every disagreement with the model logged. | Built, tested, including real Groq calls, and running in the API. |
| Labeller validation (`scripts/eval_labeler.py`) | Runs the labeller on the 600 human-labelled test reviews to measure its accuracy before its labels are trusted. | Done: results in `docs/labeler_eval.md`. |
| Retraining (`src/retrain.py`) | Trains TF-IDF candidates on the original 10,000 plus labelled requests, logs every candidate to MLflow. | Built (see `docs/retraining.md`). |
| Promotion | Picks the best candidate on validation, scores it once on the test set, promotes it only if it clears the margin. The current model is stored as an MLflow alias. | Next |

**How we demonstrate it.**

1. Start the API and send normal traffic: `python scripts/simulate_drift.py --mode normal --count 150` (add `--url <API address>` for a deployed service).
   `/drift` reports no drift and no email is sent.
2. Send shifted traffic: `python scripts/simulate_drift.py --mode shifted --count 150`.
   The detector flags the drifted features and an alert email arrives.
3. Show the saved requests in the database and the labeller giving them an `llm_label`
   (and the log line for a disagreement with the model).
4. Run retraining and show the candidate runs in the MLflow UI. Promotion of a retrained model
   is the next step (see Next steps).

**Privacy.** Reviews sent to `/predict` are stored together with the model's answer and may be used
to retrain the model. Do not send personal data. Set `FEEDBACK_STORE=off` to stop saving.

**Next steps.**

- Promote a retrained model only when it beats the current one by a margin on validation and is not worse on the human-labelled test set (selection on validation, the test set scored once).
- Use the labeller validation numbers in `docs/labeler_eval.md` to set the retraining trigger.
- Add an `/explain` endpoint (explainability) for the final demonstration.

video demo.