# AI Usage

This file discloses how AI tools were used in ReviewPulse. Each team member has their own section. The team is responsible for everything committed, whether or not an AI helped to write it.

## Member: Kashish Anil Kumar

**Tool:** Claude (Anthropic), used through the Claude app with a coding sandbox. Sessions on 5 to 6 November 2026.

### What AI was used for

| Area | How AI helped | Files or outputs |
|---|---|---|
| Planning | Read the course project document and the Milestone 1 work plan, listed the requirements with the LLM part removed, and reviewed the repository and the BERT branch against them (split sizes, MLflow runs, leftover LLM code, missing documents). | Not committed |
| Debugging | Read the failed CI log, identified a ruff import-sorting error in the BERT candidate notebook, and suggested the fix. | Lint fix on `fix/bert-evidence` |
| Documentation | Drafted `CONTRIBUTION.md` and the structure of this file, and suggested README clean-up of leftover LLM wording. | `CONTRIBUTION.md`, `AI_USAGE.md` |

### What AI did not do

- The team made the decisions: the dataset and splits, the BERT-plus-TF-IDF-baseline plan, and which checkpoint to serve.
- AI did not produce or estimate any reported metric. Results come from the team's own runs.
- AI had no network access to the repository. It worked only from zip files and screenshots supplied by Kashish, and could not run the project's tests (the packages are not installed in its sandbox).

### How the output was checked

- Suggested fixes were pushed on a branch, and CI (ruff and pytest) passed on the pull request before it was merged by a teammate.
- AI's review findings were checked against the repository and the course document by Kashish.

## Member: Abdullah Khalid and Muskan Pawan

The text below is the account these two members provided. Split it into one section per person if each used AI differently.

**Tool:** Claude (Anthropic), used through the Claude app with a coding sandbox. Sessions on 5 to 6 November 2026.

### What AI was used for

| Area | How AI helped | Files or outputs |
|---|---|---|
| Planning | Compared the repository and notebook with the Milestone 1 work plan and listed what was done, what was left, and where the repo differed from the plan (split sizes, MLflow experiment names, leftover LLM code). | Not committed |
| TF-IDF baseline | Drafted the TF-IDF predictor, the training script (three classifiers, selection on validation, one test run) and the MLflow tracking for it, plus their unit tests. | `src/tfidf_model.py`, `src/train_tfidf.py`, `tests/test_tfidf_model.py`, `tests/test_tfidf_mlflow.py` |
| MLflow | Explained the Kaggle-notebook workflow. Wrote a one-off script to copy the three candidate runs from the Kaggle database into `reviewpulse-bert`. The script was run once and then deleted. | `docs/mlflow_runs_summary.csv` was exported after that run |
| Documentation | Rewrote the README sections for the 600-review test results and MLflow tracking. Drafted the original version of this file. | `README.md`, `AI_USAGE.md` |
| Presentation | Built the Milestone 1 status deck from the repository, the notebook output and the work plan. | Slide deck (not in the repository) |
| Code review | Suggested fixes for the API adapter, evaluation script, tests and README mismatches. | Changes made in the pull requests |

### What AI did not do

- The three-candidate BERT experiment was written and run by the team on Kaggle, and the model was selected from its validation results.
- The team decided the data splits (10,000 / 1,000 / 600) and the label-mapping policy.
- All reported metrics come from the team's own runs (`bert_test_metrics.json`, the MLflow runs). AI did not supply or estimate any result.
- AI-written code was not run against the real `transformers` and MLflow packages in its sandbox. It was run locally by the team, with `python -m pytest -v` and `python -m ruff check .`.

### How the output was checked

- Test and lint runs on the members' machines before committing.
- Numbers in the README and slides recomputed from `bert_test.csv` and the MLflow exports.
- Pull requests reviewed by a different team member before merging.

### Limits and corrections

- One early patch set conflicted with a teammate's newer code and was discarded. Patches were then redone against the current files.
- An early claim that a missing `mlruns` folder was a problem was wrong and was corrected.
- AI-written TF-IDF results are not reported until the code has been run on the project data.

## Review statement

All AI-suggested code and text was read, run and reviewed by the group before being committed. Every member can explain any part of the codebase, including parts they did not write.

# AI Usage — Milestone 2

This document records how AI tools were used by each team member during Milestone 2. It distinguishes AI-generated assistance from work carried out and checked by the team members.

---

## Member: Kashish Anil Kumar

**Tool:** Claude (Anthropic).

### What AI was used for

| Area | How AI helped | Files or outputs |
|---|---|---|
| Latency monitoring and health endpoints | Supplied a starting implementation for the latency-monitoring and health-check work. The supplied kit had not been tested against the API by its author. | `src/app.py`, `src/bert_model.py`, `src/monitoring.py`, and three test files |
| Fine-tuned model loading | Explained why the API was serving the pretrained model rather than the fine-tuned model, identified the missing model directory as the issue, and explained what to check. | Guidance for checking `models/nlptown_finetuned_v2/` and `metadata.json` |
| Email alert configuration | Explained the `SMTP_USER`, `SMTP_PASSWORD`, and `ALERT_TO` variables and how to use a Gmail app password. No code was supplied for this task. | Email alert environment-variable configuration |
| Logging, drift detection, email alerts, and inference benchmark | Supplied code for these tasks as a kit. The code in the kit was AI-written; no part of that supplied code was written by a person. | `src/logging_config.py`, `src/drift.py`, `src/alerts.py`, `scripts/build_drift_reference.py`, `scripts/simulate_drift.py`, `scripts/benchmark_inference.py`, and their tests |
| Setup and Git help | Helped troubleshoot Windows setup issues, including the PowerShell execution-policy error, packages installing into global Python instead of `.venv`, and an interrupted `pip install`. Explained Git output and helped interpret test and Ruff output. | Setup instructions, Git commands, test output, and Ruff output |

### What AI did not do

- The initial latency-monitoring and health-check kit had not been run against the API by its author. I performed the API checks myself.
- For the email alert configuration task, AI explained the required settings and Gmail app password process; it did not provide code.
- AI did not carry out the repository operations on my behalf. I created the branches, commits, and pull requests and ran the commands myself.

### How the output was checked

- Copied the supplied files into the repository and ran the full test suite: **348 passed, 22 skipped**.
- Ran the API with both the pretrained and fine-tuned models and checked `/health`, `/ready`, and `/stats`.
- Downloaded the fine-tuned weights from the team's Hugging Face repository, checked `metadata.json` (`fine_tuned: true`, `run_name: r6000_seed43`), and confirmed that `/health` reported the fine-tuned model.
- Ran Ruff and pytest on the logging, drift, alert, and benchmark work.
- Ran the inference benchmark on my machine and sent a real alert email, which arrived.
- Ran each setup and Git command myself and compared the advice with the instruction file. Where I was unsure about setup, I followed the instruction file.


---

## Member: Abdullah Khalid

**Tool:** Claude (Anthropic)

### What AI was used for

| Area | How AI helped | Files or outputs |
|---|---|---|
| Docker image | [Confirm whether AI helped draft or troubleshoot the Docker configuration, including CPU-only PyTorch, the non-root user, `/health` healthcheck, `PORT`, and the serving dependencies.] | `Dockerfile`, `.dockerignore`, `requirements-serve.txt` |
| Model weights in the container | [Confirm the AI assistance used for the model-artifact approach and implementation.] | `scripts/fetch_model.py` and its tests |
| CI pipeline | [Confirm whether AI helped with the lint, test, Docker build, and GHCR publishing workflow.] | CI workflow; GHCR image tags (`sha-<short>`, `latest`, and Git tags) |
| Render deployment | [Confirm whether AI provided guidance for deploying the published image, configuring secrets, or setting the `/health` health-check path.] | Render service configuration (no PR) |
| Runbook | [Confirm the AI assistance used to draft, review, or troubleshoot the operational instructions.] | `RUNBOOK.md` |

### What AI did not do

- The selection of hosting platform had many issues so the decision was taken as a team. Ai helped with the commands but everything was ran and verified by me locally. All the results are legit and reviewed. The docker build and deployment was ran by me. AI was used to refine things and understand errors.

---

## Member: Muskan Pawan

**Tool:** Claude (Anthropic).

### What AI was used for

| Area | How AI helped | Files or outputs |
|---|---|---|
| Planning | Drafted the Milestone 2 task plan from the course document, split the work between the three members, and wrote the branch, commit, and pull request instructions for each member. | Task plan and instruction files (not committed) |
| Request store | Drafted the request store and its tests. | `src/feedback_store.py`, `tests/test_feedback_store.py` |
| LLM labeller | Drafted the labeller, retry and error handling, and validation script. After Groq retired the first model, helped change the labeller so an unusable answer is retried once and a bad key or missing model stops it without discarding anything. | `src/labeler.py`, `scripts/eval_labeler.py`, `tests/test_labeler.py`, `tests/test_eval_labeler.py` |
| Retraining | Drafted the retraining script, fresh-reviews demo script, and their tests. | `src/retrain.py`, `scripts/fresh_reviews.py`, `tests/test_retrain.py` |
| Docker and runbook | Suggested fixes for the Docker build after it timed out, including putting torch in its own layer, increasing the pip timeout, using serving requirements, and copying the drift reference. Checked the runbook's redeploy, rollback, and environment-variable steps against Render's documentation. | `Dockerfile`, `requirements-serve.txt`, `RUNBOOK.md` |
| Documentation | Compared the repository with the Milestone 2 requirements, identified README gaps (deployment section, Docker command, stale Project Focus status, placeholders, and port in the local Docker command), and drafted the Docker, deployment, and monitoring README sections. Also drafted the Project Focus section and the Milestone 2 part of `CONTRIBUTION.md`. | `README.md`, `CONTRIBUTION.md`, `AI_USAGE.md` |

### What AI did not do

- The team made the decisions: Render as the host, drift detection with an email alert as the Project Focus, Postgres for saved requests, Groq as the labeller, and where the model weights live.
- AI did not produce or estimate any reported result. The labeller accuracy (**96.8% on the 600 human-labelled test reviews**) comes from running `scripts/eval_labeler.py`, and the figures in `docs/labeler_eval.md` come from that run.
- AI could not reach the real services. I ran the request store against a real Postgres database and the labeller against the real Groq API, not in AI's sandbox.

