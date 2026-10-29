"""Phase 7: test the running API on all 715 test posts.

1. Sends every test post to POST /predict and checks the answers match the
   offline predictions of the same model (results/predictions/<model>.csv).
2. Measures prediction time: inside the API (latency_ms) and the full round trip.
3. Samples 50 posts the model gets wrong into results/errors.csv.

Start the API first (in another terminal):
    uvicorn backend.main:app
Then run from the project root:
    python -m scripts.check_api            # or: --url http://localhost:8000
"""

import argparse
import time

import httpx
import numpy as np
import pandas as pd

from ml.api_checks import compare_with_offline, latency_summary, offline_standard_predictions, sample_errors
from ml.charts import plot_latency
from ml.config import RESULTS_DIR
from ml.data import load_split
from ml.reporting import load_all_predictions


def query_api(client: httpx.Client, texts: pd.Series) -> pd.DataFrame:
    """Send each text to /predict one at a time and record the answers and timings."""
    rows = []
    for text in texts:
        start = time.perf_counter()
        response = client.post("/predict", json={"text": text})
        client_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        body = response.json()
        rows.append({"api_label": body["label"], "api_stress_prob": body["stress_probability"],
                     "server_ms": body["latency_ms"], "client_ms": client_ms,
                     "model_name": body["model_name"]})
    return pd.DataFrame(rows)


def main() -> None:
    """Run the consistency check, latency measurement and error sampling."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    test = load_split("test")
    with httpx.Client(base_url=args.url, timeout=30) as client:
        client.get("/health").raise_for_status()
        client.post("/predict", json={"text": "warm-up request"})   # not timed
        answers = query_api(client, test["text"])

    model_name = answers["model_name"].iloc[0]
    offline = offline_standard_predictions(load_all_predictions()[model_name])
    api = answers.assign(id=test["id"].to_numpy(), api_pred=(answers["api_label"] == "stress").astype(int))

    check = compare_with_offline(api, offline, model_name)
    latency = latency_summary(api["server_ms"].to_numpy(), api["client_ms"].to_numpy(), model_name)
    errors = sample_errors(test, offline)

    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame([check]).to_csv(RESULTS_DIR / "api_check.csv", index=False)
    pd.DataFrame([latency]).to_csv(RESULTS_DIR / "latency.csv", index=False)
    api[["id", "api_label", "api_stress_prob", "server_ms", "client_ms"]].to_csv(
        RESULTS_DIR / "api_predictions.csv", index=False)
    errors.to_csv(RESULTS_DIR / "errors.csv", index=False)
    plot_latency(api["server_ms"].to_numpy(), api["client_ms"].to_numpy(), model_name,
                 RESULTS_DIR / "latency.png")

    print(f"Model: {model_name}")
    print(f"Label match: {check['n_label_match']}/{check['n_posts']} "
          f"({check['label_match_pct']:.1f}%), max probability difference {check['max_prob_diff']:.2e}")
    print(f"Server latency: mean {latency['server_mean_ms']:.2f} ms, p95 {latency['server_p95_ms']:.2f} ms")
    print(f"Round trip:     mean {latency['client_mean_ms']:.2f} ms, p95 {latency['client_p95_ms']:.2f} ms "
          f"(goal < 1000 ms: {'met' if latency['meets_goal'] else 'NOT met'})")
    print(f"Saved {len(errors)} misclassified posts to {RESULTS_DIR / 'errors.csv'} "
          f"({int(np.sum(errors['error_type'] == 'false positive'))} false positives)")


if __name__ == "__main__":
    main()
