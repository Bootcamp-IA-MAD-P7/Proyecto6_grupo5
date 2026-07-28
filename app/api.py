from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

ROOT = Path(__file__).resolve().parent.parent

FEATURE_COLS = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
    "AspectRation", "Eccentricity", "ConvexArea", "EquivDiameter",
    "Extent", "Solidity", "roundness", "Compactness",
    "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4",
]

FEATURE_DESCRIPTIONS = {
    "Area": "Área de la semilla (px²)",
    "Perimeter": "Perímetro (px)",
    "MajorAxisLength": "Longitud del eje mayor (px)",
    "MinorAxisLength": "Longitud del eje menor (px)",
    "AspectRation": "Relación de aspecto (eje mayor / eje menor)",
    "Eccentricity": "Excentricidad de la elipse",
    "ConvexArea": "Área del casco convexo (px²)",
    "EquivDiameter": "Diámetro equivalente al área (px)",
    "Extent": "Proporción del bounding box",
    "Solidity": "Solidez (área / casco convexo)",
    "roundness": "Redondez",
    "Compactness": "Compacidad (4π × área / perímetro²)",
    "ShapeFactor1": "Shape Factor 1",
    "ShapeFactor2": "Shape Factor 2",
    "ShapeFactor3": "Shape Factor 3",
    "ShapeFactor4": "Shape Factor 4",
}

FEATURE_RANGES = {
    "Area": (1.0, 255000.0),
    "Perimeter": (50.0, 2100.0),
    "MajorAxisLength": (30.0, 750.0),
    "MinorAxisLength": (20.0, 450.0),
    "AspectRation": (0.5, 3.0),
    "Eccentricity": (0.1, 1.0),
    "ConvexArea": (1.0, 260000.0),
    "EquivDiameter": (5.0, 570.0),
    "Extent": (0.1, 1.0),
    "Solidity": (0.5, 1.0),
    "roundness": (0.1, 1.0),
    "Compactness": (0.1, 1.0),
    "ShapeFactor1": (0.0001, 0.02),
    "ShapeFactor2": (0.0001, 0.02),
    "ShapeFactor3": (0.1, 1.0),
    "ShapeFactor4": (0.5, 1.0),
}


# ── Pydantic models ───────────────────────────────────────────────────────────

class BeanFeatures(BaseModel):
    Area: float = Field(..., ge=1.0, le=255000.0, description="Área de la semilla (px²)")
    Perimeter: float = Field(..., ge=50.0, le=2100.0, description="Perímetro (px)")
    MajorAxisLength: float = Field(..., ge=30.0, le=750.0, description="Longitud del eje mayor (px)")
    MinorAxisLength: float = Field(..., ge=20.0, le=450.0, description="Longitud del eje menor (px)")
    AspectRation: float = Field(..., ge=0.5, le=3.0, description="Relación de aspecto")
    Eccentricity: float = Field(..., ge=0.1, le=1.0, description="Excentricidad de la elipse")
    ConvexArea: float = Field(..., ge=1.0, le=260000.0, description="Área del casco convexo (px²)")
    EquivDiameter: float = Field(..., ge=5.0, le=570.0, description="Diámetro equivalente (px)")
    Extent: float = Field(..., ge=0.1, le=1.0, description="Proporción del bounding box")
    Solidity: float = Field(..., ge=0.5, le=1.0, description="Solidez (área / casco convexo)")
    roundness: float = Field(..., ge=0.1, le=1.0, description="Redondez")
    Compactness: float = Field(..., ge=0.1, le=1.0, description="Compacidad")
    ShapeFactor1: float = Field(..., ge=0.0001, le=0.02, description="Shape Factor 1")
    ShapeFactor2: float = Field(..., ge=0.0001, le=0.02, description="Shape Factor 2")
    ShapeFactor3: float = Field(..., ge=0.1, le=1.0, description="Shape Factor 3")
    ShapeFactor4: float = Field(..., ge=0.5, le=1.0, description="Shape Factor 4")

    model_config = ConfigDict(json_schema_extra={
        "examples": [{
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
        }]
    })


class PredictionResponse(BaseModel):
    predicted_class: str
    probabilities: dict[str, float]
    top3: list[dict]


class BatchPredictionRequest(BaseModel):
    samples: list[BeanFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


class ModelInfo(BaseModel):
    model_type: str
    classes: list[str]
    features: list[str]
    feature_descriptions: dict[str, str]


class DatasetStats(BaseModel):
    total_samples: int
    classes: dict[str, int]
    columns: list[str]
    feature_count: int


# ── App & startup ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    models_dir = ROOT / "models"
    app.state.model = joblib.load(models_dir / "random_forest_model.pkl")
    app.state.scaler = joblib.load(models_dir / "scaler.pkl")
    app.state.le = joblib.load(models_dir / "label_encoder.pkl")
    yield


app = FastAPI(
    title="Dry Bean Classification API",
    description="API REST para la clasificación multiclase de semillas de judías (Dry Bean). "
                "Utiliza un modelo Random Forest entrenado con 16 características numéricas "
                "para predecir la variedad de judía.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _predict(features: BeanFeatures) -> PredictionResponse:
    input_dict = features.model_dump()
    input_array = np.array([[input_dict[f] for f in FEATURE_COLS]])
    input_scaled = app.state.scaler.transform(input_array)
    prediction = app.state.model.predict(input_scaled)
    probabilities = app.state.model.predict_proba(input_scaled)
    predicted_class = app.state.le.inverse_transform(prediction)[0]

    prob_dict = {
        cls: round(float(prob), 6)
        for cls, prob in zip(app.state.le.classes_, probabilities[0])
    }
    top3 = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]

    return PredictionResponse(
        predicted_class=predicted_class,
        probabilities=prob_dict,
        top3=[{"class": c, "probability": p} for c, p in top3],
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Infraestructura"])
def health_check():
    return {"status": "healthy", "model_loaded": hasattr(app.state, "model")}


@app.get("/model/info", response_model=ModelInfo, tags=["Modelo"])
def model_info():
    return ModelInfo(
        model_type="RandomForestClassifier",
        classes=list(app.state.le.classes_),
        features=FEATURE_COLS,
        feature_descriptions=FEATURE_DESCRIPTIONS,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
def predict(features: BeanFeatures):
    return _predict(features)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predicción"])
def predict_batch(request: BatchPredictionRequest):
    if len(request.samples) == 0:
        raise HTTPException(status_code=400, detail="La lista de muestras no puede estar vacía.")
    if len(request.samples) > 100:
        raise HTTPException(status_code=400, detail="Máximo 100 muestras por petición.")

    predictions = [_predict(sample) for sample in request.samples]
    return BatchPredictionResponse(predictions=predictions, count=len(predictions))


@app.get("/dataset/stats", response_model=DatasetStats, tags=["Dataset"])
def dataset_stats():
    csv_path = ROOT / "data" / "raw" / "Dry_Bean.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=500, detail="Dataset no encontrado.")

    df = pd.read_csv(csv_path)
    df["Class"] = df["Class"].astype(str).str.strip()
    class_counts = df["Class"].value_counts().to_dict()

    return DatasetStats(
        total_samples=len(df),
        classes=class_counts,
        columns=list(df.columns),
        feature_count=len(FEATURE_COLS),
    )


@app.get("/dataset/sample", tags=["Dataset"])
def dataset_sample(n: int = 5):
    csv_path = ROOT / "data" / "raw" / "Dry_Bean.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=500, detail="Dataset no encontrado.")

    df = pd.read_csv(csv_path)
    df["Class"] = df["Class"].astype(str).str.strip()
    n = min(max(n, 1), 50)
    sample = df.sample(n=n, random_state=42)
    return {"count": len(sample), "data": sample.to_dict(orient="records")}
