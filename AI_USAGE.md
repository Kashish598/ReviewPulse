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
