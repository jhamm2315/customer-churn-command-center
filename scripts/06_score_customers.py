from pathlib import Path
import sys
import pickle
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config.settings import PROCESSED_DIR, MODELS_DIR


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def assign_risk_band(prob: float) -> str:
    if prob >= 0.75:
        return "Critical"
    elif prob >= 0.50:
        return "High"
    elif prob >= 0.25:
        return "Moderate"
    return "Low"


def main():
    # Load clean customer data for business fields
    clean_df = pd.read_csv(PROCESSED_DIR / "customer_churn_clean.csv")

    # Load model input data
    model_input_df = pd.read_csv(PROCESSED_DIR / "model_input_dataset.csv")
    X = model_input_df.drop(columns=["target_churn"])

    # Load best model selection
    best_model_df = pd.read_csv(PROCESSED_DIR / "best_model_selection.csv")
    best_model_name = best_model_df.loc[0, "selected_model"]

    model_path = MODELS_DIR / f"{best_model_name}_model.pkl"
    model = load_pickle(model_path)

    # Score all customers
    churn_probability = model.predict_proba(X)[:, 1]
    churn_prediction = (churn_probability >= 0.5).astype(int)

    predictions = clean_df.copy()
    predictions["predicted_churn_probability"] = churn_probability
    predictions["predicted_churn_flag"] = churn_prediction
    predictions["predicted_risk_band"] = predictions["predicted_churn_probability"].apply(assign_risk_band)

    # Revenue-based prioritization
    predictions["predicted_revenue_at_risk"] = (
        predictions["estimated_annual_revenue"] * predictions["predicted_churn_probability"]
    ).round(2)

    # Priority score combines risk and value
    predictions["retention_priority_score"] = (
        predictions["predicted_churn_probability"] * predictions["estimated_annual_revenue"]
    ).round(2)

    customer_predictions = predictions[[
        "customerID",
        "gender",
        "tenure",
        "Contract",
        "MonthlyCharges",
        "TotalCharges",
        "estimated_annual_revenue",
        "predicted_churn_probability",
        "predicted_churn_flag",
        "predicted_risk_band",
        "predicted_revenue_at_risk",
        "retention_priority_score",
    ]].copy()

    risk_segments = (
        customer_predictions
        .groupby("predicted_risk_band", as_index=False)
        .agg(
            customer_count=("customerID", "count"),
            avg_churn_probability=("predicted_churn_probability", "mean"),
            total_predicted_revenue_at_risk=("predicted_revenue_at_risk", "sum"),
            avg_monthly_charges=("MonthlyCharges", "mean"),
        )
        .sort_values("total_predicted_revenue_at_risk", ascending=False)
    )

    revenue_summary = pd.DataFrame([{
        "total_customers_scored": len(customer_predictions),
        "total_estimated_annual_revenue": round(customer_predictions["estimated_annual_revenue"].sum(), 2),
        "total_predicted_revenue_at_risk": round(customer_predictions["predicted_revenue_at_risk"].sum(), 2),
        "high_risk_customers": int(customer_predictions["predicted_risk_band"].isin(["Critical", "High"]).sum()),
        "high_risk_revenue_at_risk": round(
            customer_predictions.loc[
                customer_predictions["predicted_risk_band"].isin(["Critical", "High"]),
                "predicted_revenue_at_risk"
            ].sum(),
            2
        ),
        "selected_model": best_model_name,
    }])

    customer_predictions_path = PROCESSED_DIR / "customer_predictions.csv"
    customer_risk_segments_path = PROCESSED_DIR / "customer_risk_segments.csv"
    revenue_summary_path = PROCESSED_DIR / "revenue_at_risk_summary.csv"

    customer_predictions.to_csv(customer_predictions_path, index=False)
    risk_segments.to_csv(customer_risk_segments_path, index=False)
    revenue_summary.to_csv(revenue_summary_path, index=False)

    print("Customer scoring complete.")
    print(f"Best model used: {best_model_name}")
    print("\nSaved outputs:")
    print("-", customer_predictions_path)
    print("-", customer_risk_segments_path)
    print("-", revenue_summary_path)

    print("\nRisk segment summary:")
    print(risk_segments)

    print("\nRevenue at risk summary:")
    print(revenue_summary)

    print("\nTop 10 customers by retention priority score:")
    print(
        customer_predictions.sort_values("retention_priority_score", ascending=False).head(10)
    )


if __name__ == "__main__":
    main()
