# Builds the whole app (website + API) for Render or any Docker host.
# The site and the API are served together by FastAPI.

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
# CPU-only PyTorch keeps the image small. The server itself serves the ONNX model and never imports it.
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

# The fine-tuned RoBERTa as an 8-bit ONNX model (126 MB, made by scripts.export_onnx),
# published as a GitHub release file. It fits in 512 MB of RAM without PyTorch.
# Build with --build-arg ONNX_MODEL_URL= (empty) to serve TF-IDF instead.
ARG ONNX_MODEL_URL=https://github.com/kdhamu2804/textlab/releases/download/model-v1/roberta_onnx.zip
RUN if [ -n "$ONNX_MODEL_URL" ]; then \
      python -c "import io, sys, urllib.request, zipfile; zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(sys.argv[1]).read())).extractall('models/roberta_onnx')" "$ONNX_MODEL_URL"; \
    fi

# The public site is read-only: the labelling page is for local use by the team.
ENV TEXTLAB_LABELING_ENABLED=false

# Hosts such as Render set $PORT; default to 7860 otherwise.
EXPOSE 7860
CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
