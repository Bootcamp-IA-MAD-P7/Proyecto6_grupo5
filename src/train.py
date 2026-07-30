import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RANDOM_STATE = 42


def split_data(X, y, test_size=0.2):
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )


def get_models():
    return {
        "Regresión Logística": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, multi_class="multinomial"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5, random_state=RANDOM_STATE
        ),
    }


def evaluate_model(model, X_train, y_train, X_test, y_test, use_scaled=False):
    Xtr = X_train if not use_scaled else X_train
    Xte = X_test if not use_scaled else X_test

    model.fit(Xtr, y_train)
    train_pred = model.predict(Xtr)
    test_pred = model.predict(Xte)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    diff = abs(train_acc - test_acc) * 100

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "diff_pct": diff,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "y_pred": test_pred,
    }


def train_all_models(X_train, y_train, X_test, y_test, models=None, scaled_models=None):
    if models is None:
        models = get_models()
    if scaled_models is None:
        scaled_models = {"Regresión Logística"}

    results = {}
    for name, model in models.items():
        use_scaled = name in scaled_models
        result = evaluate_model(model, X_train, y_train, X_test, y_test, use_scaled=use_scaled)
        result["model"] = model
        results[name] = result

    return results


def results_to_dataframe(results: dict) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Modelo": list(results.keys()),
            "Accuracy Train": [r["train_acc"] for r in results.values()],
            "Accuracy Test": [r["test_acc"] for r in results.values()],
            "Diferencia (%)": [r["diff_pct"] for r in results.values()],
            "CV Mean": [r["cv_mean"] for r in results.values()],
            "CV Std": [r["cv_std"] for r in results.values()],
        }
    )
    return df.sort_values("Accuracy Test", ascending=False).reset_index(drop=True)


def save_model(model, scaler, label_encoder, model_name="best_model.pkl"):
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model, models_dir / model_name)
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(label_encoder, models_dir / "label_encoder.pkl")
    return models_dir / model_name
