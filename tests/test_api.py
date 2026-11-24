"""Tests for every API endpoint."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.config import Settings, get_settings
from backend.main import create_app
from ml.config import TFIDF_DIR
from ml.data import load_split
from ml.predictors import TfidfPredictor

pytestmark = pytest.mark.skipif(not (TFIDF_DIR / "model.joblib").exists(),
                                reason="run scripts.train_baselines first")

STRESS_TEXT = "I am terrified, I can't sleep and I have panic attacks every night."


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """API client that always uses the TF-IDF model (no RoBERTa folder)."""
    settings = Settings(roberta_dir=tmp_path_factory.mktemp("no_roberta"), onnx_dir=tmp_path_factory.mktemp("no_onnx"), emotion_roberta_dir=tmp_path_factory.mktemp("no_emo"))
    with TestClient(create_app(settings)) as c:
        yield c


def test_health(client) -> None:
    assert client.get("/health").json() == {"status": "ok", "model_name": "tfidf"}


def test_predict_returns_label_confidence_and_latency(client) -> None:
    body = client.post("/predict", json={"text": STRESS_TEXT}).json()
    assert body["label"] in {"stress", "no stress"}
    assert 0.5 <= body["confidence"] <= 1
    assert body["model_name"] == "tfidf"
    assert body["latency_ms"] >= 0


@pytest.mark.parametrize(("payload", "message"), [
    ({"text": "   "}, "must not be empty"),
    ({"text": "a" * (get_settings().max_text_chars + 1)}, "too long"),
    ({}, "Field required"),
])
def test_predict_rejects_bad_input(client, payload: dict, message: str) -> None:
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert message in response.json()["detail"]


def test_explain_returns_prediction_and_segments(client) -> None:
    body = client.post("/explain", json={"text": f"  {STRESS_TEXT}  "}).json()
    predicted = client.post("/predict", json={"text": STRESS_TEXT}).json()
    assert body["label"] == predicted["label"]
    assert body["stress_probability"] == pytest.approx(predicted["stress_probability"])
    assert body["text"] == STRESS_TEXT and body["unit"] == "word"
    assert all(body["text"][s["start"]:s["end"]] == s["text"] for s in body["segments"])
    assert max(abs(s["score"]) for s in body["segments"]) == pytest.approx(1.0)


def test_explain_rejects_empty_text(client) -> None:
    assert client.post("/explain", json={"text": " "}).status_code == 422


def test_metrics_include_analysis_tables(client) -> None:
    body = client.get("/metrics").json()
    for key in ("confusion", "significance", "calibration", "calibration_summary"):
        assert body[key], key


def test_batch_predict(client) -> None:
    body = client.post("/predict/batch", json={"texts": [STRESS_TEXT, "Lovely calm day."]}).json()
    assert len(body["predictions"]) == 2
    assert body["model_name"] == "tfidf"


@pytest.mark.parametrize(("texts", "message"), [
    ([], "at least 1"),
    (["ok"] * (get_settings().max_batch_size + 1), "Too many texts"),
    (["fine", ""], "must not be empty"),
])
def test_batch_rejects_bad_input(client, texts: list[str], message: str) -> None:
    response = client.post("/predict/batch", json={"texts": texts})
    assert response.status_code == 422
    assert message in response.json()["detail"]


def test_api_matches_offline_model(client) -> None:
    test = load_split("test").head(40)
    offline = TfidfPredictor().predict(test["text"])
    body = client.post("/predict/batch", json={"texts": test["text"].tolist()}).json()
    assert [p["label"] for p in body["predictions"]] == [p.label for p in offline]


def test_model_info_reports_test_f1(client) -> None:
    info = client.get("/model-info").json()
    assert info["model_name"] == "tfidf" and info["is_fallback"] is True
    metrics = pd.read_csv(get_settings().results_dir / "metrics.csv")
    row = metrics.query("model == 'tfidf' and setting == 'standard' and test_set == 'all'").iloc[0]
    assert info["test_f1"] == pytest.approx(row["f1"])
    assert info["trained_at"]


def test_metrics_endpoint(client) -> None:
    body = client.get("/metrics").json()
    assert {"model", "setting", "test_set", "f1", "f1_low", "f1_high"} <= set(body["metrics"][0])
    assert {row["model"] for row in body["metrics"]} >= {"majority", "tfidf"}


def test_serves_built_frontend(tmp_path) -> None:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>app</html>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    settings = Settings(roberta_dir=tmp_path / "none", onnx_dir=tmp_path / "none", emotion_roberta_dir=tmp_path / "none", frontend_dist=dist)
    with TestClient(create_app(settings)) as c:
        assert c.get("/").text == "<html>app</html>"
        assert c.get("/results").text == "<html>app</html>"        # client-side route
        assert c.get("/assets/app.js").text == "console.log(1)"
        assert c.get("/../secret.txt").text == "<html>app</html>"  # no path traversal
        assert c.get("/health").json()["status"] == "ok"           # API still wins


def test_cors_allows_frontend(client) -> None:
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


@pytest.fixture()
def label_client(tmp_path):
    """API client whose label-check file is a temporary copy (the real file is never touched)."""
    from ml.annotation import LABEL_CHECK_FILE

    copy = tmp_path / "label_check.csv"
    copy.write_bytes(LABEL_CHECK_FILE.read_bytes())
    settings = Settings(roberta_dir=tmp_path / "none", onnx_dir=tmp_path / "none", emotion_roberta_dir=tmp_path / "none", label_check_file=copy)
    with TestClient(create_app(settings)) as c:
        yield c, copy


def test_label_sheet_hides_dataset_label(label_client) -> None:
    client, _ = label_client
    body = client.get("/label-check/dhamodhar").json()
    assert body["n_total"] == 50 and len(body["items"]) == 50
    assert set(body["items"][0]) == {"id", "text", "label"}


def test_saving_a_label_updates_file_and_agreement(label_client) -> None:
    # Works whether or not the (temporary copy of the) file already has labels.
    client, path = label_client
    post = client.get("/label-check/uday").json()["items"][0]
    client.put(f"/label-check/uday/{post['id']}", json={"label": None})
    before = client.get("/label-check/uday").json()["n_done"]
    body = client.put(f"/label-check/uday/{post['id']}", json={"label": 1}).json()
    saved = pd.read_csv(path, dtype={"id": str}).set_index("id")
    assert saved.at[post["id"], "uday"] == 1
    assert body["n_done"] == before + 1
    client.put(f"/label-check/uday/{post['id']}", json={"label": None})
    assert pd.read_csv(path, dtype={"id": str}).set_index("id").isna().at[post["id"], "uday"]


@pytest.mark.parametrize(("url", "payload", "status"), [
    ("/label-check/nobody/1", {"label": 1}, 404),
    ("/label-check/uday/not-a-post", {"label": 1}, 404),
    ("/label-check/uday/1948", {"label": 7}, 422),
])
def test_label_rejects_bad_requests(label_client, url: str, payload: dict, status: int) -> None:
    client, _ = label_client
    assert client.put(url, json=payload).status_code == status


def test_labeling_can_be_switched_off(tmp_path) -> None:
    settings = Settings(roberta_dir=tmp_path / "none", onnx_dir=tmp_path / "none", emotion_roberta_dir=tmp_path / "none", labeling_enabled=False)
    with TestClient(create_app(settings)) as c:
        assert c.get("/label-check/uday").status_code == 403


def test_errors_include_manual_groups(tmp_path) -> None:
    errors = pd.read_csv(get_settings().results_dir / "errors.csv", dtype={"id": str})
    groups = tmp_path / "groups.csv"
    pd.DataFrame({"id": [errors["id"][0]], "group": ["Test group"], "reviewed_by": ["uday"]}).to_csv(groups, index=False)
    settings = Settings(roberta_dir=tmp_path / "none", onnx_dir=tmp_path / "none", emotion_roberta_dir=tmp_path / "none", error_groups_file=groups)
    with TestClient(create_app(settings)) as c:
        first = c.get("/metrics").json()["errors"][0]
    assert (first["group"], first["reviewed_by"]) == ("Test group", "uday")
