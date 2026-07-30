import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FEATURE_COLS = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
    "AspectRation", "Eccentricity", "ConvexArea", "EquivDiameter",
    "Extent", "Solidity", "roundness", "Compactness",
    "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4",
]

MODELS_DIR = ROOT / "models"


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODELS_DIR / "best_model.pkl")


@pytest.fixture(scope="module")
def scaler():
    return joblib.load(MODELS_DIR / "scaler.pkl")


@pytest.fixture(scope="module")
def label_encoder():
    return joblib.load(MODELS_DIR / "label_encoder.pkl")


@pytest.fixture(scope="module")
def sample_data():
    csv_path = ROOT / "data" / "raw" / "Dry_Bean.csv"
    df = pd.read_csv(csv_path)
    df["Class"] = df["Class"].astype(str).str.strip()
    return df


# ── Data tests ────────────────────────────────────────────────────────────────

def test_dataset_exists():
    csv_path = ROOT / "data" / "raw" / "Dry_Bean.csv"
    assert csv_path.exists(), f"Dataset not found at {csv_path}"


def test_dataset_has_expected_columns(sample_data):
    for col in FEATURE_COLS + ["Class"]:
        assert col in sample_data.columns, f"Missing column: {col}"


def test_dataset_no_null_values(sample_data):
    null_counts = sample_data[FEATURE_COLS].isnull().sum()
    assert null_counts.sum() == 0, f"Null values found:\n{null_counts[null_counts > 0]}"


def test_dataset_class_values(sample_data):
    expected_classes = {"BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"}
    actual_classes = set(sample_data["Class"].unique())
    assert actual_classes == expected_classes, f"Unexpected classes: {actual_classes - expected_classes}"


def test_dataset_no_trailing_spaces(sample_data):
    has_space = sample_data["Class"].str.contains(r"\s+$", regex=True).any()
    assert not has_space, "Class column has trailing whitespace"


# ── Model artifact tests ─────────────────────────────────────────────────────

def test_model_file_exists():
    assert (MODELS_DIR / "best_model.pkl").exists()


def test_scaler_file_exists():
    assert (MODELS_DIR / "scaler.pkl").exists()


def test_label_encoder_file_exists():
    assert (MODELS_DIR / "label_encoder.pkl").exists()


# ── Prediction pipeline tests ────────────────────────────────────────────────

def test_label_encoder_classes(label_encoder):
    classes = set(label_encoder.classes_)
    expected = {"BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"}
    assert classes == expected


def test_scaler_output_shape(scaler, sample_data):
    X = sample_data[FEATURE_COLS].iloc[:5]
    X_scaled = scaler.transform(X)
    assert X_scaled.shape == (5, 16)


def test_prediction_output(model, scaler, label_encoder, sample_data):
    X = sample_data[FEATURE_COLS].iloc[:1]
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)

    assert pred.shape == (1,)
    assert proba.shape == (1, 7)
    assert abs(proba.sum() - 1.0) < 1e-6


def test_prediction_returns_valid_class(model, scaler, label_encoder, sample_data):
    X = sample_data[FEATURE_COLS].iloc[:10]
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    decoded = label_encoder.inverse_transform(preds)

    valid_classes = set(label_encoder.classes_)
    for p in decoded:
        assert p in valid_classes, f"Invalid class predicted: {p}"


def test_prediction_confidence_above_random(model, scaler, sample_data):
    X = sample_data[FEATURE_COLS].iloc[:50]
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)
    max_probs = proba.max(axis=1)
    assert max_probs.mean() > 0.5, "Model confidence too low (worse than random)"


def test_model_handles_single_sample(model, scaler, label_encoder):
    dummy = np.array([[30000, 620, 200, 180, 1.1, 0.45, 30500, 195, 0.78, 0.99, 0.94, 0.92, 0.007, 0.003, 0.85, 0.998]])
    X_scaled = scaler.transform(dummy)
    pred = model.predict(X_scaled)
    assert pred.shape == (1,)


def test_model_loads_within_memory_limit():
    import os
    import tracemalloc

    tracemalloc.start()
    _ = joblib.load(MODELS_DIR / "best_model.pkl")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 500 * 1024 * 1024, f"Model uses {peak / 1024 / 1024:.1f} MB (>500 MB limit)"
