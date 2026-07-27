import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(ROOT / "data" / "processed" / "dry_bean_clean.csv")
    return df


def main():
    df = load_data()
    X = df.drop("Class", axis=1)
    y_raw = df["Class"]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = list(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, multi_class="multinomial"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5, random_state=RANDOM_STATE
        ),
    }

    mlflow.set_experiment("dry_bean_classification")

    best_acc = 0
    best_model_name = None

    for name, model in models.items():
        use_scaled = name == "logistic_regression"
        Xtr = X_train_sc if use_scaled else X_train
        Xte = X_test_sc if use_scaled else X_test

        with mlflow.start_run(run_name=name):
            model.fit(Xtr, y_train)

            train_pred = model.predict(Xtr)
            test_pred = model.predict(Xte)

            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            diff = abs(train_acc - test_acc) * 100

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            cv_scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

            f1_macro = f1_score(y_test, test_pred, average="macro")
            precision_macro = precision_score(y_test, test_pred, average="macro")
            recall_macro = recall_score(y_test, test_pred, average="macro")

            mlflow.log_param("model_type", name)
            mlflow.log_param("n_estimators", getattr(model, "n_estimators", "N/A"))
            mlflow.log_param("random_state", RANDOM_STATE)

            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("difference_pct", diff)
            mlflow.log_metric("cv_mean", cv_scores.mean())
            mlflow.log_metric("cv_std", cv_scores.std())
            mlflow.log_metric("f1_macro", f1_macro)
            mlflow.log_metric("precision_macro", precision_macro)
            mlflow.log_metric("recall_macro", recall_macro)

            mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"\n--- {name} ---")
            print(f"  Train: {train_acc:.4f}  |  Test: {test_acc:.4f}  |  Diff: {diff:.2f}%")
            print(f"  CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"  F1: {f1_macro:.4f}  |  Precision: {precision_macro:.4f}  |  Recall: {recall_macro:.4f}")

            if test_acc > best_acc:
                best_acc = test_acc
                best_model_name = name
                best_model = model

    print(f"\n{'='*50}")
    print(f"Mejor modelo: {best_model_name} (Test Accuracy: {best_acc:.4f})")

    joblib.dump(best_model, ROOT / "models" / "best_model.pkl")
    joblib.dump(scaler, ROOT / "models" / "scaler.pkl")
    joblib.dump(le, ROOT / "models" / "label_encoder.pkl")
    print(f"Modelos guardados en models/")

    return best_model_name, best_acc


if __name__ == "__main__":
    main()
