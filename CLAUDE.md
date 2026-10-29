# CLAUDE.md — TextLab Project Plan

Team **TextLab** (MS in AI, Saint Louis University): Dhamodhar Konduru, Kushvanth Thati,
Uday Kiran Thurumella.

## Project

**Mental Health Detection from Text: A Web App That Finds Stress in Reddit Posts**
(Application track). A user pastes a post, and the app says "stress" or "no stress"
with a confidence score. Code must be clean, well organized and professional
(graduate level).

Environment: Windows, VS Code, Anaconda (Jupyter), Python 3.11+. Give all terminal
commands for Windows PowerShell.

Project root: `C:\Users\kondu\Projects\Mental Health Detection from Text`
(target of the desktop shortcut `TextLab.lnk`). Do not modify `Project.docx` or
`TextLab Project Proposal.pdf`.

## Data

- Dreaddit from Hugging Face `andreagasparini/dreaddit` (2,838 train, 715 test).
- Raw CSVs live in `data/raw/` (`dreaddit_train.csv`, `dreaddit_test.csv`).
- Columns used: `text`, `label` (1 = stress, 0 = no stress), `subreddit`, `confidence`.
- Added `topic` column:
  - abuse = domesticviolence, survivorsofabuse
  - anxiety = anxiety, stress
  - money = assistance, homeless, almosthomeless, food_pantry
  - ptsd = ptsd
  - relationships = relationships
- Processed data goes in `data/processed/`.

## Folder structure

```
TextLab/
  data/raw/, data/processed/
  ml/          -> Python package: data loading, baselines, evaluation, bootstrap CIs
  notebooks/   -> Colab notebook for RoBERTa training, data exploration notebook
  models/      -> saved models (git-ignored)
  results/     -> CSV results and charts
  backend/     -> FastAPI app
  frontend/    -> React + Vite + TypeScript + Tailwind CSS
  tests/       -> pytest tests
  scripts/     -> command-line entry points (data prep, evaluation, checks)
  README.md, requirements.txt, .gitignore
```

## Models

1. Majority class (floor).
2. TF-IDF (word 1-2 grams) + logistic regression, scikit-learn (main baseline).
3. Fine-tuned `roberta-base` (main model). Trained on Google Colab GPU, saved to `models/roberta/`.

## Evaluation

- F1 for the stress class, plus precision, recall, accuracy on the 715 test posts,
  with 95% bootstrap confidence intervals (1,000 resamples).
- Never train or tune on test data. Tune on a validation split from training data.
- New-topic test (leave-one-topic-out): train on 4 topics, test on the 5th, for all 5.
- App check: the API must give the same predictions as the offline model on all 715 posts.
- Speed: measure time per prediction in the API (goal: under 1 second).
- Save all results as CSV in `results/` and make clean charts.

## Backend (FastAPI)

- `POST /predict` -> `{text}` returns `{label, confidence, model_name, latency_ms}`
- `POST /predict/batch` -> list of texts
- `GET /health` and `GET /model-info` (model name, test F1, training date)
- `GET /metrics` -> evaluation results from `results/` for the frontend
- Load the model once at startup. Use RoBERTa if `models/roberta/` exists,
  otherwise fall back to the TF-IDF model.
- Input checks (empty text, very long text), clear error messages, CORS for the frontend.
- Pydantic models, type hints, logging, settings in a config file.

## Frontend (React + Vite + TypeScript + Tailwind)

- Clean, modern, calm design (mental health topic). Light and dark mode.
- Home page: text box, "Analyze" button, result card with label, confidence bar,
  model used. Clickable example posts.
- Results page: charts of model comparison (majority, TF-IDF, RoBERTa) and the
  new-topic test, from `GET /metrics`.
- About page: dataset, method, team members, limitations.
- On every result: "This is a research demo, not a medical diagnosis." and a link to the
  988 Suicide & Crisis Lifeline (call or text 988).
- Responsive (phone and laptop).

## Quality rules

- Clean code: small functions, docstrings, type hints, no repeated code.
- Fixed random seed **42** everywhere.
- pytest tests for data loading, API endpoints and predictions.
- README: summary, install, run backend and frontend, results table, screenshots, team.

## How to work

- Work in phases. After each phase, stop, explain how to run and check it,
  and explain every new file in simple English.
- **Do not invent results.** Only report numbers the code actually produced.

## Phases

1. Folder, data, structure, requirements, .gitignore, topic column, processed data, summary.
2. Majority + TF-IDF baselines, test + leave-one-topic-out evaluation with bootstrap CIs,
   save TF-IDF model to `models/tfidf/`, results + chart, tests.
3. `data/label_check.csv` (50 random training posts, seed 42; columns id, text,
   dataset_label, dhamodhar, kushvanth, uday) + agreement script. Never fill in labels.
4. `notebooks/train_roberta_colab.ipynb`: fine-tune roberta-base for the standard test split
   and all 5 new-topic splits, validation + early stopping, CSV results, model zip.
5. FastAPI backend + endpoint tests.
6. React frontend connected to backend.
7. API vs offline consistency check (715 posts), latency (mean + p95), 50 errors to
   `results/errors.csv`; show on Results page.
8. Strict review, cleanup, all tests pass, final README, demo checklist.
