# Contribution: ReviewPulse (Milestones 1 and 2)

Tags: `v1.0-milestone1`, `v2.0-milestone2`
Repo: https://github.com/Kashish598/ReviewPulse

## Team

| Name               | ERP ID |
| Kashish Anil Kumar | 29105  |
| Abdullah Khalid    | 28084  |
| Muskan Pawan       | 28394  |

## Member-to-task table (Milestone 1)

| Task                                                                                                                |  Owner  |
|---                                                                                                                  | ---     |
| Repo setup, branch rules, GitHub Actions CI, ruff config, LICENSE                                                   | Kashish |
| Shared contracts (`src/common.py`), data preparation (`src/prepare_data.py`), data tests, sample fixture            | Kashish |
| FastAPI service: `/predict`, `/health`, API tests                                                                   | Kashish |
| README, `CONTRIBUTION.md`, `AI_USAGE.md`, release (tag and zip)                                                     | Kashish |
| BERT candidate comparison on the validation set (Kaggle notebook), MLflow candidate runs                            | Muskan  |
| BERT model adapter (`src/bert_model.py`), `src/train_bert.py`, `src/evaluate_bert.py`, BERT tests, test predictions | Muskan  |
| TF-IDF baseline: training script (`train_tfidf.py`), model module (`tfidf_model.py`), tests                         | Abdullah |
(`test_tfidf_model.py`, `test_tfidf_mlflow.py`)                                                                       | Abdullah |

## Member-to-task table (Milestone 2)

The task IDs are those of the Milestone 2 task plan. Project Focus: data drift detection with an
email alert, and the retraining loop behind it (request store, LLM labeller, retraining).

### Packaging and deployment

| ID | Task | Owner | Branch (PR) |
|---|---|---|---|
| A1 | Docker image: `Dockerfile` (CPU-only torch, non-root user, healthcheck on `/health`, `PORT`), `.dockerignore`, `requirements-serve.txt` | Abdullah | `infra/dockerfile` (#__) |
| A2 | Model weights in the container: `scripts/fetch_model.py` and tests (zip URL, Hugging Face repo, or the pretrained model) | Abdullah | `infra/model-artifact` (#18) |
| A3 | CI: lint, test, and a Docker build that needs both; images published to ghcr with `sha-<short>`, `latest` and git tags | Abdullah | `infra/ci` (#__) |
| A4 | Render deployment from the published image, secrets set on the platform, health path `/health` | Abdullah | Render service (no PR) |
| A5 | `RUNBOOK.md`: check, redeploy, roll back, suspend and resume, run an older tag locally, endpoints, settings, logs | Abdullah and Muskan | (#__) |

### Observability and drift

| ID | Task | Owner | Branch (PR) |
|---|---|---|---|
| K1 | Logging: `src/logging_config.py`, console and rotating file, library logs captured | Kashish | `feat/logging` (#10) |
| K2 | Latency monitoring: request-timing middleware, `/stats` and `/metrics`, stage timings from `bert_model.py` | Kashish | (#__) |
| K3 | Health split: `/health` liveness that never loads the model, `/ready`, model loaded at start-up | Kashish | (#__) |
| K4 | Drift detector: `src/drift.py`, reference snapshot builder and `data/reference/reference.json` | Kashish | `feat/drift` (#11) |
| K5 | Email alert with cooldown: `src/alerts.py`, `scripts/simulate_drift.py`, drift-to-email flow test | Kashish | `feat/alerts` (#12), `test/drift-flow` (#14) |

### Data loop and Project Focus

| ID | Task | Owner | Branch |
|--- |---   |---    |---     |
| M1 | Request store: `src/feedback_store.py`, Postgres (Neon) or SQLite, duplicates stored once, a store failure never fails `/predict`; tested on SQLite and on a real Postgres database | Muskan | `feat/request-store` (#15) |
| M2 | LLM labeller: `src/labeler.py`, Groq, one-word answer, strict label check, daily cap, rate-limit handling, model and prompt version recorded | Muskan | `feat/labeler` (#16) |
| M3 | Labeller validation: `scripts/eval_labeler.py` run on the 600 human-labelled test reviews, results in `docs/labeler_eval.md` and `docs/labeler_eval.json` | Muskan | `feat/labeler-eval`  |
| M4 | Retraining: `src/retrain.py` trains TF-IDF candidates on the original data plus labelled requests, every candidate logged to MLflow; `scripts/fresh_reviews.py` for the demo | Muskan | `feat/retrain` , `feat/fresh-reviews` |
| M6 | Project Focus section of the README, tests for M1 to M4, and the demo script for drift, label and retrain | Muskan | `docs/project-focus`  |

Still open when this was written, to be added here when done: A6 (README deployment section and
housekeeping), K6 (the README table of inference measurements), M5 (promotion of a retrained
model) and `/explain`.

## Task delegation, team leadership and fixes (Muskan)

**Task delegation.** Prepared the Milestone 2 task plan and split the work between the three
members (packaging and deployment, observability and drift, the data loop). Set the shared
rules: branch naming (`feat/`, `fix/`, `infra/`), one reviewer per pull request, the merge order for
`src/app.py`, and the review rota (Kashish reviews Muskan's pull requests, Abdullah reviews
Kashish's, Muskan reviews Abdullah's). Prepared an instruction file for each member with the exact
branch, checks, commit message and pull request text, so the work could be pushed in the agreed
order.

**Leading the team.** Kept the open decisions moving (where the fine-tuned weights live, who owns
the Neon and Groq accounts, who owns `/explain`). Published the fine-tuned model on Hugging Face
(`Musknnnn/nlptown-sentiment`, with its `metadata.json`) so the container and the fetch script
can download it. Checked each teammate's failing checks and unblocked them.

**Fixes.**

- Labeller: Groq retired the model the labeller used (`llama-3.1-8b-instant`, HTTP 404). Switched to
  `openai/gpt-oss-20b`, and changed the labeller so that an unusable answer is kept for one retry
  before it is discarded, and so that it stops, and discards nothing, on a bad key or a missing
  model (`fix/labeler-model`, #17 and #19).
- Docker build: prepared fixes for Abdullah's image after his build timed out downloading
  packages (torch in its own layer, longer pip timeout), added the serving requirements the request
  store needs, and the copy of the drift reference.
- Runbook: checked the redeploy, rollback, suspend and resume, environment variable and local run
  instructions against Render's documentation and corrected them.
- Verification: ran the request store against a real Postgres database (52 tests, none skipped) and
  the labeller against the real Groq API before anything depended on them.

## Reflections

**Kashish (Milestone 1):** Set up the repository, CI and the data pipeline, with validation checks, a fixed seed and no overlap between splits, and built the FastAPI service with request validation and tests. The work showed how much easier later steps are when the data splits, shared interfaces and CI are agreed and working first.

**Abdullah (Milestone 1):** Built the TF-IDF baseline: the training script, the model module and their tests, including the MLflow logging test. The baseline gives the team a simple reference point to judge whether BERT is worth its extra cost, and showed the need to evaluate both models the same way.

**Muskan (Milestone 1):** Compared the candidate BERT sentiment models on the validation set, logged the runs in MLflow, and wrote the BERT adapter, the training and evaluation scripts, and their tests. The work showed the importance of choosing a model on validation data only and keeping the test set for the final evaluation.

**Kashish (Milestone 2):** The API silently served the pretrained model when the fine-tuned folder was missing; only a log warning and the fine_tuned flag in /health showed it, so I now check both before trusting a run. inference_ms is derived because the pipeline tokenizes internally, so it shows where the time goes but isn't a direct measurement.

**Abdullah (Milestone 2):** I owned packaging and deployment for Milestone 2. I delivered the model fetch script (A2), the Dockerfile with a healthcheck (A1) and the CI pipeline that publishes versioned images to ghcr (A3). I then deployed the image to Render (A4) and wrote the runbook for redeploying and rolling back (A5). The hardest part was getting the Docker build to finish on a slow connection, because the torch download kept timing out and once failed a pip hash check. I fixed it by splitting the install into separate cached steps and using BuildKit cache mounts, so a retry no longer re-downloads torch. I also learned why deploying a sha- tag, never latest, makes a rollback exact, and how to work in small branches and pull requests that a teammate reviews before they merge.

**Muskan (Milestone 2):** Built the data loop behind the project focus: the request store, the LLM labeller, its validation on the 600 human-labelled test reviews (all answers usable, 96.8% accuracy against the human labels) and the retraining script, and coordinated the team's pull requests. The work showed that an LLM labeller has to be validated before its labels are trusted, and that an outside service can change underneath you: the model first chosen for the labeller was retired, which only a real API call exposed.