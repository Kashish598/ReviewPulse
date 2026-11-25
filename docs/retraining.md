# Retraining on labelled requests (M4)

`src/retrain.py` trains new candidate models on the original 10,000 reviews plus the requests that
users sent to `/predict` and the LLM labeller has labelled. It prepares the choice of a new model;
it does not make it. Promotion is M5.

## The loop

```
/predict  ->  request store  ->  LLM labeller  ->  retrain  ->  (M5) promote or refuse
              (Postgres)         (Groq, labels)     (this)
```

## What goes in

| Set | Source | Labels | Used for |
|---|---|---|---|
| Original train (10,000) | `data/processed/train.csv` | human | training |
| New requests, split `train` | request store or CSV | LLM | training |
| Original validation (1,000) | `data/processed/validation.csv` | human | **choosing** (`val_*`) |
| New requests, split `validation` | request store or CSV | LLM | extra check (`new_val_*`), never trained on |
| Original test (600) | `data/processed/test.csv` | human | **not scored here**; its texts are only used to keep test reviews out of the new data |

The train/validation split of a request is fixed by a hash of its text (about 80/20), so a request
can never be in both. A new request is dropped, and counted in the report, when its text is empty,
its label is not exactly `negative` or `positive`, it repeats an earlier request, or it already
appears in the original train, validation or test data.

LLM labels are about 97% correct on the 600 test reviews (`docs/labeler_eval.md`), so the new rows
carry a few percent label noise. They are not down-weighted.

## Candidates

TF-IDF variants that train on a CPU in seconds. BERT candidates are trained on Kaggle by hand.

| Name | Classifier | Words |
|---|---|---|
| `svc_bigram_c0.5` | calibrated linear SVM, C = 0.5 | 1 and 2 word groups (the settings of `train_tfidf.py`) |
| `svc_bigram_c1` | calibrated linear SVM, C = 1 | 1 and 2 |
| `svc_unigram_c0.5` | calibrated linear SVM, C = 0.5 | single words |
| `nb_bigram` | multinomial naive Bayes | 1 and 2 |
| `nb_unigram` | multinomial naive Bayes | single words |

Plus `baseline_original_only`: the first candidate trained on the original data alone, so the effect
of the new data can be read from the same table.

## What comes out

- MLflow experiment `reviewpulse-retrain`: one run per candidate and one for the baseline, with
  parameters (candidate, TF-IDF and classifier settings, sizes of the original and new data, source),
  metrics (`val_accuracy`, `val_macro_f1`, `new_val_accuracy`, `new_val_macro_f1`, `fit_seconds`)
  and the saved model under the artifact path `model/`, so the winner can be registered later.
  `new_val_*` is left out when fewer than 10 new validation requests exist.
- `models/candidates/<name>/model.joblib` and `metadata.json` (loadable with `TfidfPredictor`, which
  names the model `tfidf-<name>`), and `models/candidates/summary.json` listing every candidate with
  its metrics and MLflow run id. Both are git-ignored.

The best candidate on validation (macro-F1, accuracy breaks ties) is named in the summary. It is
not promoted and the test split is not scored.

## Running it

See `INSTRUCTIONS_M4.md`. In short:

```
python -m src.retrain                          # reads the labelled requests from DATABASE_URL
python -m src.retrain --new-data requests.csv  # or from a CSV (text, llm_label, optional split)
python scripts/fresh_reviews.py --count 120 --send   # demo: send reviews never seen before
```

With only a few hundred new reviews against 10,000 original ones, expect validation numbers to move
by a fraction of a point either way. That is a real result, and M5's refusal to promote is a valid
outcome.
