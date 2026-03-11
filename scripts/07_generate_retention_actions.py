from pathlib import Path
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config.settings import PROCESSED_DIR


def assign_action(row: pd.Series) -> str:
    risk = row["predicted_risk_band"]
    contract = row["Contract"]
    tenure = row["tenure"]
    monthly = row["MonthlyCharges"]
    revenue_risk = row["predicted_revenue_at_risk"]

    if risk == "Critical":
        if contract == "Month-to-month" and monthly >= 80:
            return "Immediate retention outreach with contract conversion incentive"
        if tenure <= 12:
            return "High-touch onboarding recovery and service review"
        return "Executive retention call and custom retention offer"

    if risk == "High":
        if contract == "Month-to-month":
            return "Offer discount or bundle upgrade to reduce churn risk"
        if monthly >= 70:
            return "Proactive outreach with pricing/value reinforcement"
        return "Targeted retention campaign within 7 days"

    if risk == "Moderate":
        if tenure <= 12:
            return "Send onboarding reinforcement and support check-in"
        return "Monitor and enroll in nurture retention journey"

    return "Monitor only"


def assign_priority_tier(score: float) -> str:
    if score >= 750:
        return "P1"
    elif score >= 400:
        return "P2"
    elif score >= 150:
        return "P3"
    return "P4"


def assign_customer_value_band(annual_revenue: float) -> str:
    if annual_revenue >= 1000:
        return "High Value"
    elif annual_revenue >= 600:
        return "Medium Value"
    return "Standard Value"


def main():
    preds = pd.read_csv(PROCESSED_DIR / "customer_predictions.csv")

    action_df = preds.copy()

    action_df["customer_value_band"] = action_df["estimated_annual_revenue"].apply(assign_customer_value_band)
    action_df["retention_action"] = action_df.apply(assign_action, axis=1)
    action_df["priority_tier"] = action_df["retention_priority_score"].apply(assign_priority_tier)

    action_df["recommended_offer_type"] = np.select(
        [
            (action_df["predicted_risk_band"] == "Critical") & (action_df["Contract"] == "Month-to-month"),
            (action_df["predicted_risk_band"] == "High") & (action_df["MonthlyCharges"] >= 75),
            (action_df["predicted_risk_band"] == "Moderate"),
        ],
        [
            "Contract conversion incentive",
            "Discount / pricing support",
            "Customer education / support touchpoint",
        ],
        default="No immediate offer"
    )

    retention_action_plan = action_df[[
        "customerID",
        "Contract",
        "tenure",
        "MonthlyCharges",
        "estimated_annual_revenue",
        "predicted_churn_probability",
        "predicted_risk_band",
        "predicted_revenue_at_risk",
        "retention_priority_score",
        "priority_tier",
        "customer_value_band",
        "recommended_offer_type",
        "retention_action",
    ]].copy()

    high_value_at_risk_customers = (
        retention_action_plan.loc[
            retention_action_plan["predicted_risk_band"].isin(["Critical", "High"])
            & retention_action_plan["customer_value_band"].isin(["High Value", "Medium Value"])
        ]
        .sort_values("retention_priority_score", ascending=False)
        .copy()
    )

    retention_playbook_by_segment = (
        retention_action_plan
        .groupby(["predicted_risk_band", "customer_value_band", "recommended_offer_type", "retention_action"], as_index=False)
        .agg(
            customer_count=("customerID", "count"),
            total_revenue_at_risk=("predicted_revenue_at_risk", "sum"),
            avg_monthly_charges=("MonthlyCharges", "mean"),
            avg_churn_probability=("predicted_churn_probability", "mean"),
        )
        .sort_values(["predicted_risk_band", "total_revenue_at_risk"], ascending=[True, False])
    )

    retention_action_plan_path = PROCESSED_DIR / "retention_action_plan.csv"
    high_value_at_risk_path = PROCESSED_DIR / "high_value_at_risk_customers.csv"
    retention_playbook_path = PROCESSED_DIR / "retention_playbook_by_segment.csv"

    retention_action_plan.to_csv(retention_action_plan_path, index=False)
    high_value_at_risk_customers.to_csv(high_value_at_risk_path, index=False)
    retention_playbook_by_segment.to_csv(retention_playbook_path, index=False)

    print("Retention action generation complete.")
    print("\nSaved outputs:")
    print("-", retention_action_plan_path)
    print("-", high_value_at_risk_path)
    print("-", retention_playbook_path)

    print("\nTop 10 retention action priorities:")
    print(retention_action_plan.sort_values("retention_priority_score", ascending=False).head(10))

    print("\nHigh-value at-risk customers:")
    print(high_value_at_risk_customers.head(10))

    print("\nRetention playbook by segment:")
    print(retention_playbook_by_segment.head(10))


if __name__ == "__main__":
    main()
