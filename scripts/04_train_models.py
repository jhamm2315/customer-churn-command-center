from pathlib import Path
import sys
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config.settings import PROCESSED_DIR, MODELS_DIR


def save_pickle(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():
    model_input_path = PROCESSED_DIR / "model_input_dataset.csv"
    df = pd.read_csv(model_input_path)

    X = df.drop(columns=["target_churn"])
    y = df["target_churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Logistic Regression pipeline
    logistic_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42))
    ])

    # Random Forest pipeline
    rf_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # XGBoost pipeline
    xgb_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1
        ))
    ])

    models = {
        "logistic_regression": logistic_pipeline,
        "random_forest": rf_pipeline,
        "xgboost": xgb_pipeline,
    }

    prediction_frames = []
    training_summary = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        train_proba = model.predict_proba(X_train)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]

        train_auc = roc_auc_score(y_train, train_proba)
        test_auc = roc_auc_score(y_test, test_proba)

        training_summary.append({
            "model_name": model_name,
            "train_auc": round(train_auc, 4),
            "test_auc": round(test_auc, 4),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        })

        pred_df = pd.DataFrame({
            "model_name": model_name,
            "row_index": X_test.index,
            "actual_churn": y_test.values,
            "predicted_probability": test_proba,
        })
        prediction_frames.append(pred_df)

        model_path = MODELS_DIR / f"{model_name}_model.pkl"
        save_pickle(model, model_path)

        # Feature importance where available
        if model_name in ["random_forest", "xgboost"]:
            fitted_model = model.named_steps["model"]
            imputer = model.named_steps["imputer"]
            X_train_imputed = imputer.transform(X_train)

            importance_df = pd.DataFrame({
                "feature": X.columns,
                "importance": fitted_model.feature_importances_,
            }).sort_values("importance", ascending=False)

            importance_path = PROCESSED_DIR / f"feature_importance_{model_name}.csv"
            importance_df.to_csv(importance_path, index=False)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(PROCESSED_DIR / "test_predictions.csv", index=False)

    training_summary_df = pd.DataFrame(training_summary)
    training_summary_df.to_csv(PROCESSED_DIR / "training_auc_summary.csv", index=False)

    print("Model training complete.")
    print("\nTraining summary:")
    print(training_summary_df)
    print("\nSaved models to:", MODELS_DIR)
    print("\nSaved outputs:")
    print("-", PROCESSED_DIR / "test_predictions.csv")
    print("-", PROCESSED_DIR / "training_auc_summary.csv")
    print("-", PROCESSED_DIR / "feature_importance_random_forest.csv")
    print("-", PROCESSED_DIR / "feature_importance_xgboost.csv")


if __name__ == "__main__":
    main()
