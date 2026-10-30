# Mental Health Detection from Text: A Web App That Finds Stress in Reddit Posts

**Team TextLab** · MS in Artificial Intelligence, Saint Louis University · NLP final project (application track)

**Live demo: <https://textlab-q3ec.onrender.com>** (free hosting: the first visit after a quiet spell can take
about a minute to wake up). API docs: <https://textlab-q3ec.onrender.com/docs>

A user pastes a post; the app answers **"stress"** or **"no stress"** with a confidence score.
Behind it: three models trained and evaluated on the Dreaddit dataset, a FastAPI backend, and a
React + TypeScript frontend.

> **This is a research demo, not a medical diagnosis.** If you or someone you know is in crisis, call or text
> **988** ([988 Suicide & Crisis Lifeline](https://988lifeline.org), US, free, 24/7).

## Contents

- [Results](#results)
- [Project structure](#project-structure)
- [Install](#install)
- [Run the pipeline](#run-the-pipeline)
- [Run the app](#run-the-app)
- [Deploy (Render, free)](#deploy-render-free)
- [Train RoBERTa on Colab](#train-roberta-on-colab)
- [Screenshots](#screenshots)
- [Method](#method)
- [Limitations](#limitations)
- [Team](#team)

## Results

All numbers below were produced by the code in this repository (`results/metrics.csv`, seed 42).
F1, precision and recall are for the **stress** class. Brackets are 95% bootstrap confidence intervals
(1,000 resamples).

### Test set (715 posts)

| Model | F1 (stress) | Precision | Recall | Accuracy |
|---|---|---|---|---|
| Majority class | 0.681 [0.651, 0.711] | 0.516 | 1.000 | 0.516 |
| TF-IDF + logistic regression | **0.753** [0.719, 0.785] | 0.698 | 0.816 | 0.723 |
| RoBERTa (fine-tuned) | *pending: run the Colab notebook* | | | |

### New-topic test (leave one topic out), TF-IDF + LR

Each row is the same test posts of one topic. **Seen** = the standard model, trained on all 5 topics.
**New** = a model trained on the other 4 topics only.

| Held-out topic | Test posts | F1 seen | F1 new | Change |
|---|---|---|---|---|
| abuse | 142 | 0.758 | 0.719 [0.630, 0.792] | −0.039 |
| anxiety | 161 | 0.798 | 0.777 [0.705, 0.838] | −0.021 |
| money | 143 | 0.701 | 0.662 [0.566, 0.748] | −0.039 |
| ptsd | 127 | 0.812 | 0.800 [0.727, 0.860] | −0.012 |
| relationships | 142 | 0.642 | 0.613 [0.513, 0.703] | −0.029 |

F1 drops for every held-out topic, but by only 0.01–0.04, and the confidence intervals overlap widely.

### App checks (TF-IDF model served by the API)

| Check | Result |
|---|---|
| API vs offline model, all 715 test posts | **715 / 715 labels identical** (max probability difference 5e-5, from rounding to 4 decimals) |
| Time per prediction inside the API | mean 0.63 ms, p95 0.73 ms |
| Full HTTP round trip (localhost) | mean 1.69 ms, p95 2.02 ms (goal < 1 s: met) |

Latency varies slightly between runs. Re-run `python -m scripts.check_api` for current numbers, and again after
RoBERTa is installed, since a transformer on CPU is much slower than TF-IDF.

Charts: `results/model_comparison.png`, `results/new_topic_tfidf.png`, `results/latency.png`.

## Project structure

```
data/raw/            Dreaddit CSVs downloaded from Hugging Face
data/processed/      cleaned train.csv / test.csv with the added topic column
data/label_check.csv 50 training posts for the manual label check (team fills in)
ml/                  Python package
  config.py            paths, seed, topic mapping
  data.py              download, clean, load, summaries
  splits.py            validation split and leave-one-topic-out experiments
  baselines.py         majority class and TF-IDF + logistic regression (with C tuning)
  evaluation.py        metrics and bootstrap confidence intervals
  predictors.py        load a saved model (RoBERTa or TF-IDF) and predict; used by scripts and API
  reporting.py         prediction files -> metrics table
  charts.py            PNG charts
  annotation.py        label-check sampling and agreement (Cohen's kappa)
  api_checks.py        API consistency, latency and error sampling
scripts/             command-line entry points (one per step, see below)
notebooks/           01_data_exploration.ipynb, train_roberta_colab.ipynb
models/              saved models (git-ignored): tfidf/, roberta/
results/             metrics.csv, predictions/, charts, api_check.csv, latency.csv, errors.csv
backend/             FastAPI app (config, schemas, service, main)
frontend/            React + Vite + TypeScript + Tailwind CSS
tests/               pytest tests
docs/screenshots/    screenshots for this README
```

## Install

Requirements: Windows, Python 3.11+ (Anaconda is fine), Node.js 20+ (LTS). In PowerShell, from the project folder:

```powershell
# Python environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt

# Frontend
cd frontend
npm install
cd ..
```

## Run the pipeline

With the virtual environment active, from the project root:

```powershell
python -m scripts.prepare_data       # 1. download Dreaddit, add topics, print summary
python -m scripts.train_baselines    # 2. majority + TF-IDF on all 6 experiments, save models/tfidf, results, charts
python -m scripts.make_label_check   # 3. create data/label_check.csv (never overwrites a filled file)
python -m scripts.label_agreement    #    after the team fills in labels: agreement + Cohen's kappa
python -m scripts.evaluate           # 4. (after Colab) score all prediction files, redraw charts
python -m scripts.check_api          # 7. with the API running: consistency, latency, errors.csv
python -m pytest                     # all tests
ruff check ml backend scripts tests --line-length 120
```

## Run the app

Use two PowerShell terminals in the project folder.

**Terminal 1, backend:**
```powershell
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --reload
```
Interactive API docs: <http://localhost:8000/docs>.

| Endpoint | Description |
|---|---|
| `POST /predict` | `{"text": "..."}` → `{label, confidence, stress_probability, model_name, latency_ms}` |
| `POST /predict/batch` | `{"texts": [...]}` (max 64) → list of predictions |
| `GET /health` | status and loaded model |
| `GET /model-info` | model name, test F1 with CI, training date |
| `GET /metrics` | everything in `results/` for the Results page |

The API loads RoBERTa if `models/roberta/config.json` exists, otherwise TF-IDF. Empty text and text longer than
5,000 characters return HTTP 422 with a clear message. Settings live in `backend/config.py` and can be changed with
`TEXTLAB_*` environment variables (for example `TEXTLAB_MAX_TEXT_CHARS=8000`).

**Terminal 2, frontend:**
```powershell
cd frontend
npm run dev
```
Open <http://localhost:5173>.

## Deploy (Render, free)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/kdhamu2804/textlab)

`render.yaml` defines one free web service built from the `Dockerfile`. The image builds the website, downloads
Dreaddit, trains the TF-IDF model (seed 42, so it is the same model as locally) and serves the website and API
together on one URL. Every push to `main` redeploys automatically.

On the free plan the service sleeps after 15 minutes without visitors, so the first visit after that takes about a
minute. It has 512 MB of memory: enough for TF-IDF, but too little for RoBERTa.

## Train RoBERTa on Colab

1. Upload `notebooks/train_roberta_colab.ipynb` to <https://colab.research.google.com>.
2. **Runtime → Change runtime type → T4 GPU**, then upload `data/processed/train.csv` and `test.csv`.
3. **Runtime → Run all.** It trains 6 models (standard + 5 new-topic) with a validation split and early stopping,
   and downloads `textlab_roberta_outputs.zip`.
4. Unzip: `roberta/` → `models/roberta/`, `roberta.csv` → `results/predictions/`,
   `roberta_training.csv` → `results/`.
5. Run `python -m scripts.evaluate`, restart the API, then `python -m scripts.check_api`.
   RoBERTa now appears in the tables, charts and on the website.

The notebook code was smoke-tested locally with a tiny random model to check that it runs end to end; the real
`roberta-base` training happens on Colab.

## Screenshots

| Analyze | Results |
|---|---|
| ![Home page](docs/screenshots/home.png) | ![Results page](docs/screenshots/results.png) |

| Results (dark mode) | Phone |
|---|---|
| ![Results dark](docs/screenshots/results-dark.png) | ![Mobile](docs/screenshots/mobile.png) |

## Method

- **Data:** Dreaddit (Turcan & McKeown, 2019) from Hugging Face `andreagasparini/dreaddit`: 2,838 train and
  715 test posts, labels 1 = stress / 0 = no stress. Subreddits are grouped into five topics: abuse
  (domesticviolence, survivorsofabuse), anxiety (anxiety, stress), money (assistance, homeless, almosthomeless,
  food_pantry), ptsd, and relationships.
- **Models:** majority class; TF-IDF (word 1–2 grams, min_df 2, sublinear tf) + logistic regression with C chosen
  from {0.1, 0.3, 1, 3, 10, 30} by validation F1; `roberta-base` fine-tuned (lr 2e-5, batch 16, up to 6 epochs,
  early stopping with patience 2 on validation F1, max 256 tokens).
- **No test leakage:** every model is tuned on a stratified 80/20 validation split of its own training data
  (seed 42). Test posts are used only for the final scores.
- **New-topic test:** for each topic, train on the other four topics of the training split and test on that
  topic's test posts. The same posts are scored with the standard model for a direct comparison.
- **Uncertainty:** percentile bootstrap, 1,000 resamples of the test posts (seed 42).

## Limitations

- Labels say how stressed a post *sounds* to crowd annotators; they are not a clinical assessment.
- Ten English subreddits only; results may not transfer to other platforms, communities or languages.
- Only 127–161 test posts per topic, so per-topic confidence intervals are wide.
- One short post with no context: sarcasm, quoted speech, or stories about other people can mislead the model.
- The TF-IDF model has more false positives than false negatives among its errors (34 of the 50 sampled errors).

## Team

- Dhamodhar Konduru
- Kushvanth Thati
- Uday Kiran Thurumella
