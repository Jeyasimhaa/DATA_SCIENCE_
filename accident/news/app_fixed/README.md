# AI Road Accident Prediction System — Fixed

This project predicts road-accident risk from speed, road/traffic conditions
and live weather, using a trained ML model served through a Flask API, with
a Streamlit dashboard as the frontend.

## What was broken and what was fixed

1. **`ImportError` on API startup** — `api/app.py` imported `send_email` and
   `send_sms` from `notification.py`, but those functions were never
   defined (only `desktop_notification` existed). Both functions are now
   implemented (`smtplib` for email, `twilio` for SMS), and fail gracefully
   with a log message instead of crashing if credentials aren't configured.

2. **Missing model files** — `api/app.py` loaded `models/best_model.pkl`
   and `models/label_encoders.pkl`, but no training had ever been run, so
   these files didn't exist and the API crashed on import. The full
   training pipeline (`src/preprocessing.py` → `src/train_random_forest.py`
   / `src/train_xgboost.py` → `src/evaluate_models.py`) has been run, and
   the generated model files are included so the app works out of the box.

3. **Weather category mismatch** — the model is trained only on
   `Sunny / Cloudy / Rain / Fog` (from `dataset/accident_data.csv`), but
   OpenWeatherMap returns categories like `Clear`, `Clouds`, `Drizzle`,
   `Mist`, `Snow`, etc. Feeding those straight into the label encoder would
   raise `"y contains previously unseen labels"` on almost every real
   request. `api/weather_service.py` now maps live weather onto the
   trained categories.

4. **Predictions were never saved** — `api/app.py` never called any
   database function, so the "Prediction History" and heatmap in the
   dashboard would always be empty. It now calls `save_prediction(...)`
   after every prediction.

5. **Inconsistent / broken database schema** — three separate scripts
   (`api/database.py`, `database/init_db.py`, `database/db_manager.py`,
   `src/database.py`, `src/history.py`) each defined a *different* table
   schema (some missing a `recommendation` column, some using
   `road_condition` vs `road`, some inserting columns like `prediction`
   that didn't exist in their own schema). All five now share one
   consistent schema. The corrupted placeholder `accident_history.db` file
   from the zip (not a valid SQLite file) has been regenerated.

6. **Broken relative paths everywhere** — almost every script used paths
   like `"../dataset/accident_data.csv"`, which only work if you happen to
   run the script from inside its own folder. All scripts now compute
   paths relative to their own file location (`os.path.dirname(__file__)`),
   so they work no matter where they're launched from.

7. **`models/*.py` used dataset columns that don't exist** —
   `accident_prediction.py`, `predict.py`, and `train_model.py` referenced
   `RoadCondition`, `TimeOfDay`, `RoadType`, and `Visibility`, none of
   which are in `dataset/accident_data.csv` (real columns: `Weather`,
   `Speed`, `Road_Condition`, `Traffic`, `Time`, `Latitude`, `Longitude`,
   `Accident`). These would fail immediately with a `KeyError`. Fixed to
   use the real columns, and their output files were renamed
   (`accident_model.pkl` / `accident_label_encoders.pkl`) so they can't
   silently overwrite the encoders the API depends on.

8. **Missing dependencies** — `requirements.txt` was missing `matplotlib`
   (used by `src/evaluate_models.py`) and `twilio` (needed for the SMS
   alert). Both have been added. Line endings were also normalized.

9. **Hardcoded secrets in source code** — the OpenWeatherMap API key was
   hardcoded directly in `api/weather_service.py`. It's now read from the
   `.env` file via `python-dotenv`. **See the security note below — this
   needs your attention.**

10. **Tiny-dataset crashes in evaluation** — with only 15 rows, a test
    split can end up with a single class, which made `precision_score` /
    `recall_score` / `roc_auc_score` raise or warn. Added `zero_division=0`
    guards and a check before computing ROC AUC.

11. **Unpinned dependency versions** — `requirements.txt` had no version
    pins, so a fresh `pip install` could pull a newer `scikit-learn`/
    `xgboost` than the one used to create `models/*.pkl`, which throws
    `InconsistentVersionWarning` (and, across a big enough version jump,
    could fail to unpickle at all). Versions are now pinned to exactly
    what was used to train the shipped model files. If you upgrade a
    package, retrain afterwards using the steps in "How to run" below.

## ⚠️ Security note — please rotate these credentials

The uploaded project contained a file named `.eve` (now renamed to `.env`,
which is what `python-dotenv` actually looks for) with what appear to be
**real, live credentials**: an OpenWeatherMap API key and a Gmail address
with a password. Because this file was included in the zip you uploaded,
treat those as compromised:

- Rotate/regenerate the OpenWeatherMap API key.
- Change the Gmail password, and if you want email alerts to work, use a
  [Gmail App Password](https://myaccount.google.com/apppasswords) instead
  of your real account password (Gmail blocks plain-password SMTP login
  anyway).
- The Twilio credentials in the file are placeholders (`your_twilio_sid`,
  etc.), so those are fine as-is until you add real ones.

`.env` is already in `.gitignore`, so it won't get committed if you push
this to Git — just don't share the zip itself with the real values inside.

## Project structure

```
app/
├── api/            Flask REST API (serves /predict)
├── dashboard/      Streamlit frontend
├── analytics/      Small script to dump prediction_history as a DataFrame
├── database/       SQLite DB + init/manager scripts
├── dataset/        Training data (accident_data.csv, processed_data.csv)
├── models/         Trained model files (.pkl) + a standalone example pipeline
├── src/            Training/inference pipeline used by api/
├── output/         Generated PDFs, confusion-matrix plots, heatmap.html
└── requirements.txt
```

## How to run

**Important: the API and the dashboard are two separate processes that
must both be running at the same time, in two separate terminal
windows.** If you only start the dashboard, it will load fine but every
prediction attempt will show "❌ Flask API is not running" — that's
expected, not a bug.

```bash
pip install -r requirements.txt
```

1. **(Already done for you, only needed if you change the dataset)**
   Retrain the model:
   ```bash
   cd src
   python preprocessing.py
   python train_random_forest.py
   python train_xgboost.py
   python evaluate_models.py
   ```

2. **Terminal 1 — start the API** and leave it running:
   ```bash
   cd api
   python app.py
   ```
   or just run `start_api.sh` (Mac/Linux) / `start_api.bat` (Windows)
   from the project root.
   You should see `Running on http://127.0.0.1:5000`. Leave this window open.

3. **Terminal 2 — start the dashboard**, in a *different* terminal window:
   ```bash
   cd dashboard
   streamlit run app.py
   ```
   or run `start_dashboard.sh` / `start_dashboard.bat`.

4. Add your own weather/email/SMS credentials to `.env` at the project
   root (see the security note above).

## Verification performed

- Every `.py` file compiles cleanly (`python -m py_compile`).
- The full training pipeline was run end-to-end and produced
  `models/best_model.pkl` and `models/label_encoders.pkl`.
- The Flask API was started and `/`, `/health`, and `/predict` were all
  exercised successfully (weather call mocked, since this sandbox has no
  internet access to openweathermap.org) — a prediction was returned, a
  PDF report was generated, and the row was correctly saved to
  `accident_history.db`.
- The Streamlit dashboard was launched headlessly and served HTTP 200
  with no import/runtime errors.
