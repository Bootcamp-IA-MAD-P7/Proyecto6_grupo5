import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.api import app

SAMPLE = {
    "Area": 30000,
    "Perimeter": 620,
    "MajorAxisLength": 200,
    "MinorAxisLength": 180,
    "AspectRation": 1.1,
    "Eccentricity": 0.45,
    "ConvexArea": 30500,
    "EquivDiameter": 195,
    "Extent": 0.78,
    "Solidity": 0.99,
    "roundness": 0.94,
    "Compactness": 0.92,
    "ShapeFactor1": 0.007,
    "ShapeFactor2": 0.003,
    "ShapeFactor3": 0.85,
    "ShapeFactor4": 0.998,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── Health ────────────────────────────────────────────────────────────────────

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


# ── Model Info ────────────────────────────────────────────────────────────────

def test_model_info(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_type"] == "RandomForestClassifier"
    assert len(body["classes"]) == 7
    assert len(body["features"]) == 16


def test_model_info_classes(client):
    resp = client.get("/model/info")
    classes = set(resp.json()["classes"])
    expected = {"BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"}
    assert classes == expected


# ── Single Prediction ─────────────────────────────────────────────────────────

def test_predict_single(client):
    resp = client.post("/predict", json=SAMPLE)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_class"] in {
        "BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"
    }
    assert len(body["probabilities"]) == 7
    assert len(body["top3"]) == 3


def test_predict_probabilities_sum_to_one(client):
    resp = client.post("/predict", json=SAMPLE)
    probs = resp.json()["probabilities"]
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-4


def test_predict_top3_sorted_descending(client):
    resp = client.post("/predict", json=SAMPLE)
    top3 = resp.json()["top3"]
    probs = [t["probability"] for t in top3]
    assert probs == sorted(probs, reverse=True)


def test_predict_invalid_feature(client):
    invalid = SAMPLE.copy()
    invalid["Area"] = -1
    resp = client.post("/predict", json=invalid)
    assert resp.status_code == 422


def test_predict_missing_feature(client):
    incomplete = {k: v for k, v in SAMPLE.items() if k != "Area"}
    resp = client.post("/predict", json=incomplete)
    assert resp.status_code == 422


def test_predict_extra_feature(client):
    extra = SAMPLE.copy()
    extra["ExtraField"] = 123
    resp = client.post("/predict", json=extra)
    assert resp.status_code == 200


# ── Batch Prediction ──────────────────────────────────────────────────────────

def test_predict_batch(client):
    payload = {"samples": [SAMPLE, SAMPLE]}
    resp = client.post("/predict/batch", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_predict_batch_empty(client):
    resp = client.post("/predict/batch", json={"samples": []})
    assert resp.status_code == 400


def test_predict_batch_too_large(client):
    payload = {"samples": [SAMPLE] * 101}
    resp = client.post("/predict/batch", json=payload)
    assert resp.status_code == 400


# ── Dataset Stats ─────────────────────────────────────────────────────────────

def test_dataset_stats(client):
    resp = client.get("/dataset/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_samples"] > 0
    assert body["feature_count"] == 16
    assert len(body["classes"]) == 7


def test_dataset_stats_class_sum(client):
    resp = client.get("/dataset/stats")
    total = sum(resp.json()["classes"].values())
    assert total == resp.json()["total_samples"]


# ── Dataset Sample ────────────────────────────────────────────────────────────

def test_dataset_sample(client):
    resp = client.get("/dataset/sample?n=3")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3
    assert len(body["data"]) == 3


def test_dataset_sample_default(client):
    resp = client.get("/dataset/sample")
    assert resp.status_code == 200
    assert resp.json()["count"] == 5


def test_dataset_sample_max_capped(client):
    resp = client.get("/dataset/sample?n=100")
    assert resp.status_code == 200
    assert resp.json()["count"] == 50
