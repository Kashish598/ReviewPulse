# LLM labeller validation (M3)

Run on 2026-11-25T11:29:56+00:00. Model `openai/gpt-oss-20b`, prompt version `v1`, temperature 0. Reference: the human labels of the 600 test reviews.

| Measure | Value |
|---|---|
| Reviews sent | 600 of 600 |
| Usable answers (exactly negative or positive) | 600 (100.0%) |
| Unusable answers | 0 |
| Requests the API refused | 0 |
| Tokens used | 147427 |
| Labeller accuracy vs human (usable answers) | 96.8% |
| Labeller macro-F1 vs human | 0.9683 |
| Accuracy if unusable answers count as wrong | 96.8% |

Confusion matrix (rows: human label, columns: labeller; order negative, positive): `[[295, 7], [12, 286]]`

## Agreement with the served BERT model (600 reviews)

| Measure | Value |
|---|---|
| Labeller and BERT give the same label | 95.0% |
| Reviews where they differ | 30 |
| ...the labeller matches the human label | 20 |
| ...BERT matches the human label | 10 |
| BERT accuracy on these reviews | 95.2% |
| Labeller accuracy on these reviews | 96.8% |

The labels are LLM opinions. A labeller accuracy below the served model's would mean that retraining on its labels adds noise, not signal.
