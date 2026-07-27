from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

FEATURE_COLS = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
    "AspectRation", "Eccentricity", "ConvexArea", "EquivDiameter",
    "Extent", "Solidity", "roundness", "Compactness",
    "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4",
]


def load_artifacts(models_dir: str | Path | None = None):
    if models_dir is None:
        models_dir = ROOT / "models"
    model = joblib.load(models_dir / "best_model.pkl")
    scaler = joblib.load(models_dir / "scaler.pkl")
    le = joblib.load(models_dir / "label_encoder.pkl")
    return model, scaler, le


def predict_single(input_dict: dict, model, scaler, le) -> dict:
    input_df = pd.DataFrame([input_dict])[FEATURE_COLS]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)
    predicted_class = le.inverse_transform(prediction)[0]

    prob_dict = {
        cls: float(prob)
        for cls, prob in zip(le.classes_, probabilities[0])
    }

    return {
        "predicted_class": predicted_class,
        "probabilities": prob_dict,
        "top3": sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3],
    }


def predict_batch(df: pd.DataFrame, model, scaler, le) -> pd.DataFrame:
    X = df[FEATURE_COLS]
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    probas = model.predict_proba(X_scaled)

    classes = le.classes_
    result = df.copy()
    result["predicted_class"] = le.inverse_transform(preds)

    for i, cls in enumerate(classes):
        result[f"prob_{cls}"] = probas[:, i]

    return result
