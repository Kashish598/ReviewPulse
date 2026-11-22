# Request store and LLM labeller (M1 and M2)

The Project Focus is a data loop: requests are saved, an LLM labels them in the background, and
the labelled requests become new training data when the data drifts.

```
POST /predict --> answer is returned
      |
      +--> save request + prediction  (src/feedback_store.py, Postgres)
                |
                +--> background labeller asks the LLM for one word  (src/labeler.py)
                          |
                          +--> labelled rows are the training data for retraining (M4)
```

## Privacy note (put this in the README and the /predict description)

> Reviews sent to `/predict` are stored, together with the model's answer, and may be used to
> retrain the model. Nothing else about the caller is stored. Do not send personal data.

## Request store (`src/feedback_store.py`)

- Database: `DATABASE_URL`. Any Postgres works (Neon gives `postgresql://...?sslmode=require`,
  which is rewritten for the psycopg 3 driver). With no `DATABASE_URL` it falls back to a local
  SQLite file at `data/processed/feedback.db` and logs a warning, because that file is lost when
  a host without a persistent disk redeploys. `FEEDBACK_STORE=off` turns saving off.
- Tables are created on start. Table `requests` holds the text, the served model's label and
  score, the model name, a split (`train` or `validation`, fixed from the text's hash, about 80/20),
  and the label status (`pending`, `labelled`, `discarded`) with the LLM label, LLM model and prompt
  version. Table `label_usage` counts LLM calls and tokens per UTC day.
- The same text is stored once (matched on a hash of the stripped text), which also stops one
  repeated review from dominating the training data.
- Saving never breaks a prediction: `record_prediction()` swallows and logs errors.
- Neon suspends idle databases, so the engine tests each connection before using it.

## Labeller (`src/labeler.py`)

- Groq, `llama-3.1-8b-instant`, temperature 0, one word. The answer is kept only if it is exactly
  `negative` or `positive` (case, a trailing full stop or quotes are tolerated). Anything else is
  discarded at once, with the answer saved (first 200 characters) and logged. It is not retried,
  because at temperature 0 it would answer the same way.
- The review is wrapped in `<review>` tags and the prompt says it is data, not instructions. A
  review that tries to steer the label can still succeed sometimes, which is one more reason the
  labels are validated against the 600 human-labelled test reviews (M3) before they are trusted.
- A daily cap (`LABEL_DAILY_CAP`, default 200) counted in the database, so a restart does not
  reset it. One call at most every `LABEL_MIN_INTERVAL_SECONDS` (default 4), because the tokens
  per minute limit binds first. Check the limits on your own Groq account page, they can change.
- On a 429 it waits as long as the API says. On a network error or a 5xx it backs off (30 s, then
  60 s, up to 10 minutes) and the request stays pending. On 401 or 403 it stops for good and logs
  `Check GROQ_API_KEY`.
- Every label stores the LLM model and the prompt version. Bump `PROMPT_VERSION` whenever the
  prompt changes, because the meaning of a label depends on it.
- Every disagreement between the served model and the LLM is logged at INFO, with a summary of
  the agreement rate every 25 labels.
- The key comes from `GROQ_API_KEY` only and is never logged or shown in `repr`.
- The idle poll is long (`LABEL_POLL_SECONDS`, default 360) so an idle Neon database can suspend.
  `labeler.wake()` makes it look straight away when a new request was saved.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | local SQLite | Postgres connection string (a secret, set it on the host) |
| `FEEDBACK_STORE` | on | `off` stops saving requests |
| `GROQ_API_KEY` | none | the labeller is off without it (a secret) |
| `LABEL_MODEL` | `llama-3.1-8b-instant` | Groq model name |
| `LABEL_DAILY_CAP` | 200 | most LLM calls per UTC day |
| `LABEL_MIN_INTERVAL_SECONDS` | 4 | pause between calls |
| `LABEL_POLL_SECONDS` | 360 | how long it sleeps when nothing is pending |
| `LABEL_ENABLED` | true | `false` turns the labeller off even with a key |

## Setting up the database

1. Create a Postgres database (Neon: create a project and copy its connection string). Check the
   provider's current free plan limits and idle behaviour yourself, they were not verified here.
2. Set it as `DATABASE_URL` in your shell for local runs and as a secret on the host. Never
   commit it.
3. Tables are created the first time the API starts.

Look at the data with any SQL client:

```sql
select label_status, count(*) from requests group by 1;
select pred_label, llm_label, count(*) from requests where label_status = 'labelled' group by 1, 2;
select * from label_usage order by day desc limit 7;
```

## Running the store tests against Postgres

The tests run on SQLite by default. To also run them on Postgres:

```powershell
$env:TEST_DATABASE_URL = "postgresql://user:password@host/my_test_db"
pytest tests/test_feedback_store.py -v
```

The tests drop and recreate the tables, so the database name must contain `test`. The test
refuses to run on any other name.

## Dependencies to add when this is wired into the app

Not needed for these two branches (the tests use SQLite and `SQLAlchemy` is already in
`requirements.txt`), but the running service needs a Postgres driver. Versions chosen: the newest
that is more than two weeks old and has Python 3.11 wheels.

- `requirements.txt`: add `psycopg==3.3.8` and `psycopg-binary==3.3.8`
- `requirements-serve.txt` (after the Docker branch is merged): add `SQLAlchemy==2.0.57`,
  `psycopg==3.3.8` and `psycopg-binary==3.3.8`

`psycopg-binary` is needed because the slim Python image has no `libpq`. The two files must keep
the same pins (the CI check enforces it).

## How it is wired into the API (integration step, after the logging and drift branches)

```python
# lifespan
app.state.labeler = labeler.start_labeler()      # None when there is no key or no store
yield
if app.state.labeler:
    app.state.labeler.stop()

# /predict
def save_request(text, result, model_name):
    if feedback_store.record_prediction(text, result.label, result.score, model_name) \
            and app.state.labeler:
        app.state.labeler.wake()

background_tasks.add_task(save_request, request.text, result, predictor.name)
```

Saving and labelling run after the response is sent, so they add nothing to the request latency.
