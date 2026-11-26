# ReviewPulse Runbook

How to check, redeploy, roll back and run ReviewPulse. Written for a teammate who has not touched
the deployment before. If a step here does not match what you see, fix the runbook in the same PR.

## Current deployment record

The known good state. If a change breaks the service, this is what to go back to. Update it every
time you deploy something that works.

| | |
|---|---|
| Live URL | https://reviewpulse-0j1b.onrender.com |
| Render service name | reviewpulse |
| Render region | Ohio (US East) |
| Render plan | 1 CPU / 2 GB (`1c-2g`) |
| Current image tag (Render, Settings, Deploy) | ghcr.io/kashish598/reviewpulse:sha-dd0305d |
| Last known good tag (roll back to this) | sha-521f973 |
| Deployed on | 26 / 11 / 2026 |
| Render account owner | Abdullah |

**Never delete old image tags from the GitHub Packages registry (ghcr).** A Render rollback needs
the old image to still exist. If it is gone, the rollback fails.

| | |
|---|---|
| Live API | https://reviewpulse-0j1b.onrender.com |
| Image registry | `ghcr.io/kashish598/reviewpulse` (package visibility: public) |
| Image tags | `sha-<short commit>`, `latest`, and the git tag (for example `v2.0-milestone2`) |
| Host | Render web service from the prebuilt image, 1 CPU / 2 GB plan (the model needs about 1 GB of memory) |
| Health check path on Render | `/health` |
| Database | Neon Postgres (`DATABASE_URL`). Without it the app uses a local SQLite file that is lost on redeploy |
| Labeller | Groq (`GROQ_API_KEY`) |

**Note on Live Testing and Account Access**
Due to Render's billing structure, inviting additional team members to the dashboard requires upgrading to a Pro workspace. This would add a flat monthly fee per user on top of our existing $25/month compute costs. 

Because we are actively suspending the service to conserve funds, we have opted not to upgrade. Consequently, only the single account owner has access to the Render dashboard. The live deployment, rollback, and suspend/resume steps in this runbook cannot be independently verified by other teammates. Team members should rely on Section 5 to test the deployment image locally via Docker.

## 1. Check that it is running

Replace the host to test locally (`http://127.0.0.1:10000` when you run the Docker commands in section 5).

```powershell
$H = "https://reviewpulse-0j1b.onrender.com"
Invoke-RestMethod "$H/health"      # {status: ok}; also names the model once it is loaded
Invoke-RestMethod "$H/ready"       # 200 when the model is loaded, 503 while it loads or if loading failed
Invoke-RestMethod "$H/stats" | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri "$H/predict" -ContentType "application/json" `
  -Body '{"text":"Great product, works well."}'
```

cURL equivalents:

```bash
curl https://reviewpulse-0j1b.onrender.com/health
curl https://reviewpulse-0j1b.onrender.com/ready
curl -X POST https://reviewpulse-0j1b.onrender.com/predict \
  -H "Content-Type: application/json" -d '{"text":"Great product, works well."}'
curl https://reviewpulse-0j1b.onrender.com/stats
```

### What a healthy service looks like

`/ready` once the model is loaded (HTTP 200):

```
status model                  fine_tuned
------ -----                  ----------
ready  nlptown-bert-finetuned       True
```

Things to check: `status` is `ready`, `model` is `nlptown-bert-finetuned` and `fine_tuned` is `True`.
If `fine_tuned` is `False` or the model name is `nlptown-bert-multilingual`, the service is serving the
pretrained model, not the fine-tuned one: the weights did not get into the image (see section 6, note on
`MODEL_HF_REPO`). While the model is still loading, `/ready` answers HTTP 503 with `loading`; wait and
try again. `/predict` answers `{"sentiment": "positive", "score": ...}`.

### Endpoints

| Method | Path | What it tells you |
|---|---|---|
| GET | `/health` | Liveness. The process is up. Never loads the model, answers at once. Names the model after it has loaded. |
| GET | `/ready` | Readiness. 200 when the model is loaded, 503 `loading` or `error` otherwise. |
| POST | `/predict` | Sentiment for `{"text": "..."}`. 422 for empty, blank, missing or over-long (5000 characters) text. Reviews sent here are stored and may be used for retraining. |
| GET | `/stats` and `/metrics` | Same JSON: `request_count`, `avg_api_latency_ms`, `avg_model_latency_ms`, and a `per_model` block with stage times. Counts successful `/predict` calls only. In memory, resets on restart. |
| GET | `/docs` | Swagger UI. |
| GET | `/drift` | Drift status. **Only once the drift wiring is merged into `src/app.py`; delete this row if it is not in the tagged release.** |

The service sleeps or is suspended outside grading hours to save cost. The first request after
that can take a minute or more (the container starts and the model loads). Call `/health` first,
then wait until `/ready` returns 200 before you call `/predict`.

## 2. Redeploy

Render does not redeploy an image-backed service by itself when a new image gets pushed, whatever
the tag. Redeploying is a manual step. **Do not use "Deploy latest reference" on its own to ship a
new version:** it only pulls the image behind the tag the service is currently set to. If the
service is pinned to `sha-521f973`, it reinstalls that same old image.

**Deploy a new version**

1. Make sure CI is green on the commit you want, and that the image exists: GitHub, the repo's
   Packages, `reviewpulse`, and look for the tag `sha-<short commit>`.
2. In the Render Dashboard open the service, **Settings**, **Deploy** section. Change the Image URL
   to the new tag, `ghcr.io/kashish598/reviewpulse:sha-<short commit>`, and save.
3. Go to **Manual Deploy**, **Deploy latest reference**. The service now pulls the tag you set.
4. Watch **Logs** until you see the model ready line, then run section 1.
5. Write the tag and date in the Current deployment record at the top.

Always point the service at a `sha-<short commit>` tag, not `latest`. A `sha-` tag names one exact
build, which is what makes a rollback exact (section 3).

**Deploy one exact tag with the deploy hook instead**

The deploy hook URL is on the service's Settings tab. It is a secret, do not paste it into the repo
or chat. **The service must be running: if it is suspended the hook fails. Resume it first
(section 4).** Append the tag. All parts of the image URL other than the tag or digest must match
the service's image URL:

```powershell
$hook = "<deploy hook URL from Render settings>"
Invoke-RestMethod "$hook&imgURL=ghcr.io%2Fkashish598%2Freviewpulse%3Asha-<short commit>"
```

A deploy through the hook with `imgURL` does not change the tag stored in the service's
settings. Later deploys go back to that tag.

**Redeploy the same version** (for example to restart the container): **Manual Deploy**,
**Deploy latest reference**. It pulls the tag already set in the service's settings.

## 3. Roll back

**Never delete old image tags from the GitHub Packages registry.** Every rollback below needs the
old image to still exist in ghcr.

Choose the first way that applies.

**A. Roll back in Render (fastest).** Service, **Events** (deploys), find the last good successful
deploy, **Rollback to this deploy**, confirm. Render redeploys the image of that deploy. Two things
to know: the old image must still exist in ghcr, otherwise the rollback fails, and the rollback
pulls the image again by its tag. That is exact when the deploy used a `sha-<short commit>` tag.
If the service was set to a moving tag such as `latest`, the rollback pulls whatever `latest` is
now, which may be the broken version.

**B. Deploy an earlier tag.** Find the tag you want in the repo's Packages (`sha-...` or a git tag)
and deploy that tag: change the Image URL in Settings, Deploy to it and use **Deploy latest
reference** (section 2), or use the deploy hook. Changing the Image URL makes it stay.

**C. Roll back the code.** If the bad change is in `main`, revert it with a PR (`git revert
<commit>`), wait for CI to publish the new image, and redeploy it (section 2).

After any rollback run section 1 and check `/stats` and a `/predict` call, then update the Current
deployment record.

## 4. Suspend and resume

Suspending stops the service (and the cost of the paid plan) without deleting it. Do it after
grading, resume it before a demo.

- **Suspend:** Render Dashboard, open the service, **Settings**, scroll to **Suspend** and confirm.
  Or tick the service in the dashboard's service list and click **Suspend** in the action menu.
- **Resume:** open the suspended service and click **Resume**, or tick it in the service list and
  click **Resume** in the action menu. Wait for the deploy to finish, then run section 1.

**You must resume the service before you trigger a deploy hook, or the hook fails.** Resume first,
wait until it is running, then deploy.

Resume it a few minutes before any demo or video recording: the container has to start and load the
model before `/ready` returns 200.

## 5. Run an older version locally

```powershell
docker pull ghcr.io/kashish598/reviewpulse:sha-521f973        # replace with the tag you want
docker run --rm -p 10000:10000 -e PORT=10000 --env-file .env ghcr.io/kashish598/reviewpulse:sha-521f973
```

Then `http://127.0.0.1:10000/docs`. The container listens on `PORT` (10000 here, the same as Render uses), so
the port in `-p` and in `-e PORT` must match.

**If you have no local `.env` file, remove `--env-file .env` from the command. Otherwise Docker fails
to start** (it stops with an error that it cannot open `.env`). Without a `.env` the app still
starts: it uses SQLite, no labelling, no alerts, and the model that the image has baked in (or
fetches at start). Write `.env` values without quotes and without spaces around `=` (Docker takes
them literally). `docker ps` shows `(healthy)` after the model has loaded. Stop it with Ctrl+C, or
`docker stop <name>` if you used `-d`.

To build the same image yourself from a checkout of that commit:

```powershell
git switch --detach <commit or tag>
docker build --build-arg MODEL_HF_REPO=Musknnnn/nlptown-sentiment -t reviewpulse:test .
docker run --rm -p 10000:10000 -e PORT=10000 reviewpulse:test
```

## 6. Settings (environment variables)

Set them in Render (Environment tab) or in a local `.env`. Never commit `.env` or a value.
`.env.example` in the repo lists every variable with comments.

**Changing an environment variable in Render restarts the container for it to take effect.** When
you save, Render asks how to apply the change: choose **Save and deploy**, which redeploys the
existing build with the new values. **Save only** stores the value but the running service keeps
the old one until the next deploy. After the restart, run the section 1 checks.

| Variable | Purpose |
|---|---|
| `PORT` | Port the server listens on. The image defaults to 8000, Render sets 10000, and the Docker commands in this runbook use 10000. |
| `LOG_LEVEL`, `LIB_LOG_LEVEL`, `LOG_DIR` | App and library log level (default INFO), log folder. `DEBUG` shows what each stage received, and logs the text of every submitted review: see the privacy warning in section 7. |
| `MODEL_HF_REPO` | Hugging Face repo with the fine-tuned model, `Musknnnn/nlptown-sentiment`. See the note below. |
| `MODEL_URL`, `MODEL_SHA256` | Alternative: a zip of the model folder, and its checksum. |
| `REVIEWPULSE_MODEL_DIR` | Serve a model from a different folder. |
| `MODEL_PRELOAD` | `off` skips loading the model at startup (the first `/predict` loads it). |
| `DATABASE_URL` | Neon Postgres connection string. Secret. |
| `FEEDBACK_STORE` | `off` stops saving requests. |
| `GROQ_API_KEY` | Groq key for the labeller. Secret. The labeller stays off without it. |
| `LABEL_ENABLED`, `LABEL_DAILY_CAP`, `LABEL_MIN_INTERVAL_SECONDS`, `LABEL_POLL_SECONDS`, `LABEL_MODEL`, `LABEL_MAX_TOKENS` | Labeller on/off, daily call limit, pacing and model. |
| `DRIFT_MIN_WINDOW` and the other `DRIFT_*` | Drift window size and thresholds (see `.env.example`). |
| `SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD`, `ALERT_TO` | Email alerts. `SMTP_PASSWORD` is a secret (a Gmail app password). |

**Note on `MODEL_HF_REPO`.** Earlier deployments needed `MODEL_HF_REPO` set on Render so the
container downloaded the roughly 670 MB model at startup. Images published by CI now carry the
weights baked in, so on the next deployment this variable can be removed from Render. Check first
that the image really has the weights: run it locally without the variable (section 5, no
`MODEL_HF_REPO`, no `--env-file`) and confirm that `/ready` returns 200 with `fine_tuned` `True` within
seconds, and that the log has no line about downloading from Hugging Face. If it does download, or
`fine_tuned` is `False`, keep the variable. Leaving it set is harmless: the fetch script does
nothing when a model is already in place. The weights were verified to be baked into the image
starting with tag `sha-3b02503` on 26 Nov 2026. If the service is running this tag or newer,
`MODEL_HF_REPO` can be safely removed from Render.

If a secret leaks (pushed, pasted in chat or in a screenshot): rotate it at the provider first
(Neon, Groq, Google app passwords), then update Render.

## 7. Logs

- Render: service, **Logs**. This is the place to look first. The container writes to the console.
- Inside a container the app also writes `logs/reviewpulse.log` (rotating, 5 MB, 5 backups). It
  is lost when the container is replaced.
- Locally: `logs\reviewpulse.log` in the repo folder (git-ignored). `Get-Content logs\reviewpulse.log -Tail 50`.
- Startup lines to look for: `Loading sentiment model from ...`, `Model ready after ... s`.

**WARNING: privacy.** `LOG_LEVEL=DEBUG` writes the text of every review submitted to the API into
the logs, on Render and in the log file. Use `DEBUG` only temporarily for troubleshooting, then set
it back to `INFO`.

## 8. When something is wrong

| Symptom | Check |
|---|---|
| Render shows the deploy as failed, "Image Pull Failed" | Settings, Deploy section: image URL and tag correct, the tag exists in ghcr, the package is public. |
| `/health` works but `/ready` stays 503 | Logs: the model is still loading (wait), or it failed to load (the line after `/ready` says `error` and the reason). |
| Out of memory / restarts | The plan must be 2 GB. The 512 MB free plan cannot hold the model. |
| First request very slow | The service was asleep or suspended. Wake it with `/health` a few minutes before you need it. |
| The deploy hook returns an error | The service is suspended. Resume it (section 4), then trigger the hook again. |
| The deploy hook returns a 404 error | The tag doesn't exist in ghcr, or the host/repository in the `imgURL` doesn't match the service's current Image URL exactly. |
| A new image was pushed but the service still runs the old one | Render never redeploys image services by itself. Change the Image URL to the new tag and deploy (section 2). |
| An environment variable change did nothing | It was saved with "Save only". Redeploy (**Manual Deploy**, **Deploy latest reference**) or save again with "Save and deploy". |
| `/predict` works but nothing is saved | `DATABASE_URL` missing or wrong, or `FEEDBACK_STORE=off`. A store failure never fails `/predict`; look for `reviewpulse.store` errors in the logs. |
| Saved requests stay `pending` | `GROQ_API_KEY` missing, `LABEL_ENABLED` false, daily cap reached, or Groq rate limits. Look for `reviewpulse.labeler` lines. |
| `/drift` says disabled | The reference snapshot `data/reference/reference.json` is not in the image. |
| `/stats` went back to zero | The service restarted. The counters are in memory. |
| `docker run` fails at once with an error about `.env` | There is no `.env` file. Remove `--env-file .env` from the command (section 5). |

## 9. Cost

The paid 2 GB plan is only for the grading window. Suspend the service afterwards (section 4) or
delete it, and say so in the README. Check Render pricing before you resume it.
