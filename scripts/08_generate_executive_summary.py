from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config.settings import PROCESSED_DIR


def main():
    preds = pd.read_csv(PROCESSED_DIR / "customer_predictions.csv")
    segments = pd.read_csv(PROCESSED_DIR / "customer_risk_segments.csv")
    revenue_summary = pd.read_csv(PROCESSED_DIR / "revenue_at_risk_summary.csv")
    playbook = pd.read_csv(PROCESSED_DIR / "retention_playbook_by_segment.csv")
    best_model = revenue_summary.loc[0, "selected_model"]

    total_customers = int(revenue_summary.loc[0, "total_customers_scored"])
    total_annual_revenue = float(revenue_summary.loc[0, "total_estimated_annual_revenue"])
    total_revenue_at_risk = float(revenue_summary.loc[0, "total_predicted_revenue_at_risk"])
    high_risk_customers = int(revenue_summary.loc[0, "high_risk_customers"])
    high_risk_revenue = float(revenue_summary.loc[0, "high_risk_revenue_at_risk"])

    critical_count = int((preds["predicted_risk_band"] == "Critical").sum())
    high_count = int((preds["predicted_risk_band"] == "High").sum())
    moderate_count = int((preds["predicted_risk_band"] == "Moderate").sum())
    low_count = int((preds["predicted_risk_band"] == "Low").sum())

    top_playbooks = (
        playbook.sort_values("total_revenue_at_risk", ascending=False)
        .head(5)
        .copy()
    )

    strategy_summary = pd.DataFrame([
        {
            "metric": "Total customers analyzed",
            "value": total_customers
        },
        {
            "metric": "Total estimated annual revenue",
            "value": round(total_annual_revenue, 2)
        },
        {
            "metric": "Total predicted revenue at risk",
            "value": round(total_revenue_at_risk, 2)
        },
        {
            "metric": "High-risk customers",
            "value": high_risk_customers
        },
        {
            "metric": "High-risk revenue at risk",
            "value": round(high_risk_revenue, 2)
        },
        {
            "metric": "Selected production model",
            "value": best_model
        },
    ])

    summary_text = f"""
Customer Retention Executive Summary

Total customers analyzed: {total_customers}
Selected production model: {best_model}

Predicted churn risk distribution:
- Critical: {critical_count}
- High: {high_count}
- Moderate: {moderate_count}
- Low: {low_count}

Financial exposure:
- Total estimated annual revenue: ${total_annual_revenue:,.2f}
- Total predicted revenue at risk: ${total_revenue_at_risk:,.2f}
- High-risk revenue at risk: ${high_risk_revenue:,.2f}

Leadership interpretation:
The retention portfolio shows a concentrated pocket of high-value customer risk that should be addressed through targeted interventions rather than broad campaigns. Priority should be placed on customers with high churn probability, month-to-month contracts, and elevated monthly charges, as these segments represent the greatest near-term revenue exposure.

Recommended strategic actions:
1. Prioritize high-value Critical and High risk customers for immediate retention outreach.
2. Use contract conversion incentives for high-risk month-to-month customers.
3. Deploy onboarding and support interventions for newer customers with elevated churn probability.
4. Focus retention resources on the segments with the highest modeled revenue-at-risk.
5. Operationalize the best-performing model ({best_model}) as the primary scoring engine for future retention workflows.
""".strip()

    summary_txt_path = PROCESSED_DIR / "executive_retention_summary.txt"
    strategy_csv_path = PROCESSED_DIR / "retention_strategy_summary.csv"
    top_playbooks_path = PROCESSED_DIR / "top_retention_playbooks.csv"

    summary_txt_path.write_text(summary_text)
    strategy_summary.to_csv(strategy_csv_path, index=False)
    top_playbooks.to_csv(top_playbooks_path, index=False)

    print("Executive summary generation complete.")
    print("\nSaved outputs:")
    print("-", summary_txt_path)
    print("-", strategy_csv_path)
    print("-", top_playbooks_path)

    print("\nExecutive summary:")
    print(summary_text)

    print("\nStrategy summary table:")
    print(strategy_summary)

    print("\nTop retention playbooks:")
    print(top_playbooks)


if __name__ == "__main__":
    main()
