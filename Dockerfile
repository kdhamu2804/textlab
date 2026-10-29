# Builds the whole app (website + API) for Hugging Face Spaces or any Docker host.
# The site and the API are served together by FastAPI on port 7860.

# --- Stage 1: build the React website -------------------------------------
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# A production build calls the API on the same address as the website.
RUN npm run build

# --- Stage 2: Python API ---------------------------------------------------
FROM python:3.12-slim
RUN useradd -m -u 1000 user
ENV HOME=/home/user PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
# CPU-only PyTorch keeps the image small; it is only needed once RoBERTa is added.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
COPY --from=web --chown=user /web/dist frontend/dist
USER user

# Download Dreaddit and train the TF-IDF model (seed 42 -> same model as locally),
# unless a trained model was copied in.
RUN if [ ! -f models/tfidf/model.joblib ]; then \
      python -m scripts.prepare_data && python -m scripts.train_baselines; \
    fi

EXPOSE 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
