# Mental Health Detection from Text: A Web App That Finds Stress in Reddit Posts

**Team TextLab** · MS in Artificial Intelligence, Saint Louis University · NLP final project (application track)

**Live demo: <https://textlab-q3ec.onrender.com>** (free hosting: the first visit after a quiet spell can take
about a minute to wake up). API docs: <https://textlab-q3ec.onrender.com/docs>

A user pastes a post; the app answers **"stress"** or **"no stress"** with a calibrated probability and
highlights the sentences that drove the prediction. Behind it: three models trained and evaluated on the
Dreaddit dataset, a FastAPI backend, and a React + TypeScript frontend.

**Key findings**

- Fine-tuned RoBERTa reaches **F1 0.834** on the 715 test posts, **+8.1 F1 points** over TF-IDF + logistic
  regression (95% CI of the gain [+5.1, +11.2]; paired bootstrap p < 0.001; McNemar p < 0.001). The proposal's
  success target was +3.
- On a topic it never saw in training, RoBERTa still scores F1 0.71–0.86, above TF-IDF's score on topics it
  *was* trained on.
- Raw RoBERTa probabilities are over-confident; temperature scaling (T = 1.71, fitted on validation data)
  cuts the test calibration error from 0.112 to 0.073 without changing any label.

![Analyze page](docs/screenshots/home.png)

> **This is a research demo, not a medical diagnosis.** If you or someone you know is in crisis, call or text
> **988** ([988 Suicide & Crisis Lifeline](https://988lifeline.org), US, free, 24/7).

## Contents

- [Results](#results)
- [Extension: 27 emotions (GoEmotions)](#extension-27-emotions-goemotions)
- [Project structure](#project-structure)
- [Install](#install)
- [Run the pipeline](#run-the-pipeline)
- [Run the app](#run-the-app)
- [Deploy (Render, free)](#deploy-render-free)
- [Train RoBERTa](#train-roberta)
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
| TF-IDF + logistic regression | 0.753 [0.719, 0.785] | 0.698 | 0.816 | 0.723 |
| RoBERTa (fine-tuned) | **0.834** [0.806, 0.860] | 0.789 | 0.883 | 0.818 |

RoBERTa beats TF-IDF by **8.1 F1 points** (the success target was 3), and the two confidence intervals do not
overlap.

### Is the difference significant? (`results/significance.csv`)

Paired tests on the same 715 posts: a paired bootstrap of the F1 difference (10,000 resamples, one-sided) and
McNemar's exact test on the posts where exactly one of the two models is right.

| Comparison | Δ F1 | 95% CI of Δ | Paired bootstrap | McNemar (only B right / only A right) |
|---|---|---|---|---|
| TF-IDF + LR vs majority | +0.072 | [+0.041, +0.103] | p < 0.001 | p < 0.001 (216 / 68) |
| RoBERTa vs TF-IDF + LR | **+0.081** | [+0.051, +0.112] | p < 0.001 | p < 0.001 (113 / 45) |

### Confusion matrices (`results/confusion.csv`)

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Majority class | 0 | 346 | 0 | 369 |
| TF-IDF + LR | 216 | 130 | 68 | 301 |
| RoBERTa | 259 | 87 | 43 | 326 |

RoBERTa removes 33% of TF-IDF's false positives and 37% of its false negatives.

### Calibration (`results/calibration_summary.csv`, `results/temperature.csv`)

The app shows a probability, so it should mean what it says. Expected calibration error (ECE, 10 bins) and
Brier score on the test set, lower is better:

| Model | ECE | Brier |
|---|---|---|
| TF-IDF + LR | 0.097 | 0.185 |
| RoBERTa, raw | 0.112 | – |
| RoBERTa, temperature-scaled (T = 1.71) | **0.073** | **0.133** |

The temperature was fitted by minimizing negative log-likelihood on the 568 validation posts only (validation
NLL 0.436 → 0.376). Dividing logits by T never changes the predicted label, so all F1 scores are unaffected.

### New-topic test (leave one topic out), RoBERTa

| Held-out topic | Test posts | F1 seen | F1 new | Change |
|---|---|---|---|---|
| abuse | 142 | 0.845 | 0.824 [0.756, 0.880] | −0.021 |
| anxiety | 161 | 0.871 | 0.864 [0.811, 0.911] | −0.007 |
| money | 143 | 0.796 | 0.807 [0.718, 0.878] | +0.011 |
| ptsd | 127 | 0.877 | 0.864 [0.807, 0.915] | −0.013 |
| relationships | 142 | 0.734 | 0.710 [0.615, 0.791] | −0.024 |

On a topic it never saw, RoBERTa is still above TF-IDF's *seen-topic* F1 for every topic. Relationships is the
hardest topic for both models.

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

### App checks (RoBERTa served by the API on CPU)

| Check | Result |
|---|---|
| API vs offline model, all 715 test posts | **715 / 715 labels identical** (max probability difference 5e-5, from rounding to 4 decimals) |
| Time per prediction inside the API | mean 84 ms, p95 121 ms |
| Full HTTP round trip (localhost) | mean 87 ms, p95 124 ms (goal < 1 s: met) |
| `/explain` (prediction + sentence occlusion) | about 0.2 s for a typical post, under 1 s for the longest test post |

The first request after starting the API takes about 1 s while the model warms up. Latency varies between runs;
re-run `python -m scripts.check_api` for current numbers. (With TF-IDF the API answered in about 2 ms.)

Charts: `results/model_comparison.png`, `results/new_topic_roberta.png`, `results/new_topic_tfidf.png`,
`results/confusion_matrices.png`, `results/calibration.png`, `results/latency.png`.

### Compact model for the live demo (`results/onnx_check.csv`)

Free hosting has 512 MB of RAM, too little for PyTorch plus RoBERTa. `scripts.export_onnx` exports the
fine-tuned model to ONNX with 8-bit per-channel weight quantization and checks it against the full model on
all 715 test posts:

| | Full RoBERTa (PyTorch) | 8-bit ONNX |
|---|---|---|
| Model file | 499 MB | 126 MB |
| F1 (stress), 715 test posts | 0.834 | 0.834 |
| Same label as the full model | – | 689 / 715 (96.4%) |
| Peak server memory (measured locally, no PyTorch loaded) | – | 434 MB |

The 26 posts that change are borderline cases, flipping in both directions, so F1 is unchanged. Quantizing with
one scale per tensor instead of per channel cost about 1.4 F1 points, so per-channel scales are used. All
reported results are from the full model; the live demo uses the compact one.

### Error analysis (`data/error_groups.csv`)

The 50 sampled errors (`results/errors.csv`) were read and each assigned one group. **This grouping is a
draft prepared with AI assistance and must be reviewed by the team** (fill in `reviewed_by` after checking).

| Group | False positives | False negatives |
|---|---|---|
| Stress implied, not stated | 0 | 10 |
| Calm account of a hard situation | 9 | 0 |
| Past trauma, now coping | 8 | 0 |
| Mixed or ambiguous post | 2 | 5 |
| Sad, hurt or frustrated, not stressed | 5 | 0 |
| Someone else's distress | 4 | 1 |
| Lyrics, quotes or jokes | 3 | 0 |
| Possible label noise | 2 | 1 |

Most false positives are posts about a hard *topic* told calmly, or about past or someone else's distress: the
model reacts to the subject matter. Most false negatives (10 of 17) imply stress through the
situation (no food left, a fiancée leaving) without stress words.

### Label check (`data/label_check.csv`)

Each team member labels the same 50 training posts with one rule, *"Does the writer sound stressed, worried or
upset?"*. Use the labelling page at <http://localhost:8000/label> (dataset label hidden, keyboard Y/N, saves
automatically) or fill in the CSV. Agreement with the dataset and between team members (Cohen's κ) appears on
the Results page as soon as labels exist, and `python -m scripts.label_agreement` saves it to
`results/label_agreement.csv`. Per the proposal, more than 10 disagreements with the dataset labels is reported
as a label-quality problem.

## Extension: 27 emotions (GoEmotions)

Stress is one signal; a post can also show fear, sadness, anger and more. As an extension, a **second, separate**
model detects 27 emotions plus neutral. The stress study above (Dreaddit) is unchanged.

- **Data:** GoEmotions (Demszky et al., 2020), `google-research-datasets/go_emotions` ("simplified"): 43,410 train /
  5,426 validation / 5,427 test Reddit comments. Multi-label: a comment can have several emotions (1.18 on average).
  Very uneven: neutral has 14,219 training examples, grief only 77.
- **Models:** TF-IDF (word 1–2 grams) + one logistic regression per emotion (C chosen on validation), and
  `roberta-base` fine-tuned with one sigmoid output per emotion (lr 3e-5, batch 16, max 128 tokens, best epoch by
  validation loss: epoch 3 of 4).
- **Thresholds:** each emotion gets its own probability threshold, chosen to maximize that emotion's F1 on the
  **validation** split. The test split is used once, for the scores below.

| Model | Macro F1 (mean over 28 labels) | Micro F1 (all decisions pooled) |
|---|---|---|
| TF-IDF + LR | 0.451 [0.429, 0.469] | 0.525 [0.515, 0.535] |
| **RoBERTa** | **0.509** [0.489, 0.525] | **0.602** [0.592, 0.612] |

RoBERTa vs TF-IDF macro F1: **+5.8 points** [+4.0, +7.8], paired bootstrap p < 0.001 (`results/emotions_*.csv`,
chart `results/emotions_per_label.png`). Best emotions: gratitude 0.92, amusement 0.83, love 0.81; hardest:
realization 0.21 and grief 0.00 (only 6 test comments; RoBERTa never predicted it, while TF-IDF reached 0.46 on it).

In the app, the Analyze page shows "Emotions in this post" next to the stress result (`POST /emotions`). It runs
in the **local** version only: the free 512 MB server cannot hold a second model, so online it says so.

```powershell
python -m scripts.prepare_emotions                 # download GoEmotions
python -m scripts.train_emotions --model tfidf     # minutes, CPU
python -m scripts.train_emotions --model roberta   # about 25 minutes on an RTX 3050 Ti
python -m scripts.evaluate_emotions                # test scores with CIs, per-emotion table and chart
```

Limitations: GoEmotions labels are single short comments rated by crowd workers; rare emotions have very few
examples; and the two models are trained on different data, so the emotion panel and the stress result can
disagree.

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
  analysis.py          confusion matrices, paired bootstrap + McNemar tests, calibration (ECE, Brier)
  calibration.py       temperature scaling
  predictors.py        load a saved model (RoBERTa or TF-IDF), predict and explain; used by scripts and API
  reporting.py         prediction files -> metrics table
  charts.py            PNG charts
  annotation.py        label-check sampling and agreement (Cohen's kappa)
  api_checks.py        API consistency, latency and error sampling
scripts/             command-line entry points (one per step, see below)
notebooks/           01_data_exploration.ipynb, train_roberta_colab.ipynb
models/              saved models (git-ignored): tfidf/, roberta/, roberta_onnx/
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
python -m scripts.train_roberta      # 4. fine-tune RoBERTa on a local GPU (or use the Colab notebook)
python -m scripts.calibrate_roberta  #    fit the temperature on validation data
python -m scripts.evaluate           # 5. score all prediction files with bootstrap CIs, redraw charts
python -m scripts.analyze            # 6. confusion matrices, significance tests, calibration
python -m scripts.export_onnx        #    compact 8-bit ONNX model for the live demo
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
| `POST /explain` | `{"text": "..."}` → prediction plus `segments` (sentence or word, offsets, influence score) |
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
Dreaddit, trains the TF-IDF model (seed 42, so it is the same model as locally), downloads the 8-bit ONNX
RoBERTa from the `model-v1` GitHub release, and serves the website and API together on one URL. The labelling
page is switched off there (`TEXTLAB_LABELING_ENABLED=false`). Every push to `main` redeploys automatically.

To publish a new model: run `python -m scripts.export_onnx`, zip the contents of `models/roberta_onnx/`, and
upload the zip as `roberta_onnx.zip` to a GitHub release (update `ONNX_MODEL_URL` in the `Dockerfile` if the tag
changes).

On the free plan the service sleeps after 15 minutes without visitors, so the first visit after that takes about a
minute.

## Train RoBERTa

**On a local NVIDIA GPU** (the reported results were trained this way on an RTX 3050 Ti, 4 GB, in about
20 minutes for all 6 models). Install a CUDA build of PyTorch first, for example
`pip install torch --index-url https://download.pytorch.org/whl/cu130`, then:

```powershell
python -m scripts.train_roberta      # writes models/roberta/, results/predictions/roberta.csv, results/roberta_training.csv
python -m scripts.calibrate_roberta  # temperature scaling on the validation split
python -m scripts.evaluate
python -m scripts.analyze
```

The script uses the same settings as the notebook (effective batch 16 via gradient accumulation, fp16).
Add `--standard-only` to skip the 5 new-topic models, or `--per-device-batch 4` if the GPU runs out of memory.

**On Google Colab** (no local GPU):

1. Upload `notebooks/train_roberta_colab.ipynb` to <https://colab.research.google.com>.
2. **Runtime → Change runtime type → T4 GPU**, then upload `data/processed/train.csv` and `test.csv`.
3. **Runtime → Run all.** It trains 6 models (standard + 5 new-topic) with a validation split and early stopping,
   and downloads `textlab_roberta_outputs.zip`.
4. Unzip: `roberta/` → `models/roberta/`, `roberta.csv` → `results/predictions/`,
   `roberta_training.csv` → `results/`.
5. Run `python -m scripts.calibrate_roberta`, `python -m scripts.evaluate` and `python -m scripts.analyze`,
   restart the API, then `python -m scripts.check_api`. RoBERTa now appears in the tables, charts and on the website.

## Screenshots

| Results: answer and model comparison | Results: new topics, calibration, errors |
|---|---|
| ![Results page](docs/screenshots/results.png) | ![Results analysis](docs/screenshots/results-analysis.png) |

| Results (dark mode) | Phone | About |
|---|---|---|
| ![Results dark](docs/screenshots/results-dark.png) | ![Mobile](docs/screenshots/mobile.png) | ![About](docs/screenshots/about.png) |

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
- **Significance:** paired bootstrap of the F1 difference (10,000 resamples, one-sided) and McNemar's exact
  test, both on the same test posts.
- **Calibration:** reliability diagram, ECE (10 equal-width bins) and Brier score; RoBERTa is
  temperature-scaled (Guo et al., 2017) with T fitted on its validation split.
- **Explanations:** RoBERTa uses sentence-level occlusion (P(stress) with and without each sentence; clauses
  for posts with fewer than three sentences). TF-IDF uses its exact linear contributions per word. We tried
  gradient × input and Integrated Gradients at the word level for RoBERTa, but the attributions were noisy and
  Integrated Gradients with 20 steps failed its completeness check, so they are not shown.

## Limitations

- Labels say how stressed a post *sounds* to crowd annotators; they are not a clinical assessment.
- Ten English subreddits only; results may not transfer to other platforms, communities or languages.
- Only 127–161 test posts per topic, so per-topic confidence intervals are wide.
- One short post with no context: sarcasm, quoted speech, or stories about other people can mislead the model.
- RoBERTa has more false positives than false negatives among its errors (33 of the 50 sampled errors in
  `results/errors.csv`): sad or emotional posts that annotators did not label as stressed.
- The live demo serves the 8-bit ONNX copy of RoBERTa, which gives a different label than the full model on
  26 of 715 test posts (same F1).

## Team

- Dhamodhar Konduru
- Kushvanth Thati
- Uday Kiran Thurumella
