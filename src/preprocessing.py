from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent

FEATURE_COLS = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
    "AspectRation", "Eccentricity", "ConvexArea", "EquivDiameter",
    "Extent", "Solidity", "roundness", "Compactness",
    "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4",
]


def load_raw_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = ROOT / "data" / "raw" / "Dry_Bean.csv"
    df = pd.read_csv(csv_path)
    df["Class"] = df["Class"].astype(str).str.strip()
    return df


def load_clean_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = ROOT / "data" / "processed" / "dry_bean_clean.csv"
    return pd.read_csv(csv_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Class"] = df["Class"].astype(str).str.strip()
    df = df.drop_duplicates()
    df = df.dropna(subset=FEATURE_COLS + ["Class"])
    return df


def get_feature_target_split(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y_raw = df["Class"]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return X, y, le


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_sc, X_test_sc, scaler
