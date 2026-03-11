from pathlib import Path
import sys
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config.settings import PROCESSED_DIR, MODELS_DIR


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    df = pd.read_csv(PROCESSED_DIR / "model_input_dataset.csv")

    X = df.drop(columns=["target_churn"])
    y = df["target_churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model_files = {
        "logistic_regression": MODELS_DIR / "logistic_regression_model.pkl",
        "random_forest": MODELS_DIR / "random_forest_model.pkl",
        "xgboost": MODELS_DIR / "xgboost_model.pkl",
    }

    metrics_rows = []
    confusion_rows = []

    for model_name, model_path in model_files.items():
        model = load_pickle(model_path)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        metrics_rows.append({
            "model_name": model_name,
            "roc_auc": round(auc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(accuracy, 4),
        })

        confusion_rows.append({
            "model_name": model_name,
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        })

    metrics_df = pd.DataFrame(metrics_rows).sort_values("roc_auc", ascending=False)
    confusion_df = pd.DataFrame(confusion_rows)

    best_model = metrics_df.iloc[0]["model_name"]

    best_model_summary = pd.DataFrame([{
        "selected_model": best_model,
        "selection_basis": "Highest ROC AUC on holdout test set",
    }])

    metrics_path = PROCESSED_DIR / "model_performance_metrics.csv"
    confusion_path = PROCESSED_DIR / "confusion_matrix_summary.csv"
    best_model_path = PROCESSED_DIR / "best_model_selection.csv"

    metrics_df.to_csv(metrics_path, index=False)
    confusion_df.to_csv(confusion_path, index=False)
    best_model_summary.to_csv(best_model_path, index=False)

    print("Model evaluation complete.")
    print("\nPerformance metrics:")
    print(metrics_df)
    print("\nConfusion matrix summary:")
    print(confusion_df)
    print("\nBest model:")
    print(best_model_summary)
    print("\nSaved outputs:")
    print("-", metrics_path)
    print("-", confusion_path)
    print("-", best_model_path)


if __name__ == "__main__":
    main()
