from pathlib import Path
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config.settings import RAW_DATA_PATH, PROCESSED_DIR, TARGET_COLUMN


SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

BINARY_YES_NO_COLUMNS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

YES_NO_SERVICE_COLUMNS = [
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

CATEGORICAL_COLUMNS = [
    "InternetService",
    "Contract",
    "PaymentMethod",
]

NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


def tenure_band(tenure: float) -> str:
    if tenure <= 12:
        return "0-12 Months"
    elif tenure <= 24:
        return "13-24 Months"
    elif tenure <= 48:
        return "25-48 Months"
    else:
        return "49+ Months"


def monthly_charge_band(value: float) -> str:
    if value < 35:
        return "Low"
    elif value < 70:
        return "Medium"
    elif value < 100:
        return "High"
    return "Very High"


def contract_risk(contract: str) -> str:
    if contract == "Month-to-month":
        return "High Risk"
    elif contract == "One year":
        return "Moderate Risk"
    return "Low Risk"


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    missing_mask = df["TotalCharges"].isna()
    df.loc[missing_mask, "TotalCharges"] = (
        df.loc[missing_mask, "MonthlyCharges"] * df.loc[missing_mask, "tenure"]
    )
    return df


def normalize_service_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 'No internet service' and 'No phone service' to 'No'
    so all service columns become clean Yes/No flags.
    """
    df = df.copy()

    replacements = {
        "No internet service": "No",
        "No phone service": "No",
    }

    for col in YES_NO_SERVICE_COLUMNS:
        df[col] = df[col].replace(replacements)

    return df


def build_service_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in YES_NO_SERVICE_COLUMNS:
        df[f"{col}_flag"] = np.where(df[col] == "Yes", 1, 0)

    service_flag_cols = [f"{col}_flag" for col in YES_NO_SERVICE_COLUMNS]
    df["service_count"] = df[service_flag_cols].sum(axis=1)

    df["has_fiber"] = np.where(df["InternetService"] == "Fiber optic", 1, 0)
    df["is_month_to_month"] = np.where(df["Contract"] == "Month-to-month", 1, 0)
    df["is_senior"] = np.where(df["SeniorCitizen"] == 1, 1, 0)

    return df


def build_business_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["tenure_band"] = df["tenure"].apply(tenure_band)
    df["monthly_charge_band"] = df["MonthlyCharges"].apply(monthly_charge_band)
    df["contract_risk_band"] = df["Contract"].apply(contract_risk)
    df["churn_flag"] = np.where(df[TARGET_COLUMN] == "Yes", 1, 0)
    df["avg_revenue_per_month"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )
    df["estimated_annual_revenue"] = df["MonthlyCharges"] * 12
    df["revenue_at_risk"] = np.where(df["churn_flag"] == 1, df["estimated_annual_revenue"], 0)

    return df


def build_model_input(df: pd.DataFrame) -> pd.DataFrame:
    model_df = df.copy()

    # Map simple Yes/No fields
    for col in BINARY_YES_NO_COLUMNS:
        model_df[col] = np.where(model_df[col] == "Yes", 1, 0)

    # Map service Yes/No fields after normalization
    for col in YES_NO_SERVICE_COLUMNS:
        model_df[col] = np.where(model_df[col] == "Yes", 1, 0)

    # Encode gender
    model_df["gender"] = np.where(model_df["gender"] == "Male", 1, 0)

    # One-hot encode remaining categorical business fields
    model_df = pd.get_dummies(
        model_df,
        columns=CATEGORICAL_COLUMNS + ["tenure_band", "monthly_charge_band", "contract_risk_band"],
        drop_first=True
    )

# Drop non-model and leakage fields
    drop_cols = [
        "customerID",
        TARGET_COLUMN,
        "churn_flag",
        "revenue_at_risk",
    ]


    existing_drop_cols = [c for c in drop_cols if c in model_df.columns]
    Xy = model_df.drop(columns=existing_drop_cols, errors="ignore")


    # Force bool dummies to ints
    bool_cols = Xy.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        Xy[bool_cols] = Xy[bool_cols].astype(int)


# Defensive leakage guard
    leakage_keywords = ["churn", "revenue_at_risk"]
    leakage_cols = [
        c for c in Xy.columns
        if any(keyword in c.lower() for keyword in leakage_keywords)
    ]
    if leakage_cols:
        Xy = Xy.drop(columns=leakage_cols, errors="ignore")

    Xy["target_churn"] = df["churn_flag"].values

    return Xy


def main():
    df = pd.read_csv(RAW_DATA_PATH)

    df = clean_total_charges(df)
    df = normalize_service_values(df)
    df = build_service_features(df)
    df = build_business_features(df)

    clean_output = PROCESSED_DIR / "customer_churn_clean.csv"
    feature_output = PROCESSED_DIR / "feature_matrix.csv"

    df.to_csv(clean_output, index=False)

    model_input = build_model_input(df)
    model_output = PROCESSED_DIR / "model_input_dataset.csv"
    model_input.to_csv(model_output, index=False)

    feature_cols = [
        "customerID",
        "tenure",
        "tenure_band",
        "MonthlyCharges",
        "monthly_charge_band",
        "TotalCharges",
        "Contract",
        "contract_risk_band",
        "service_count",
        "has_fiber",
        "is_month_to_month",
        "estimated_annual_revenue",
        "revenue_at_risk",
        "churn_flag",
    ]
    feature_matrix = df[feature_cols].copy()
    feature_matrix.to_csv(feature_output, index=False)

    print("Feature engineering complete.")
    print(f"Saved clean dataset to: {clean_output}")
    print(f"Saved feature matrix to: {feature_output}")
    print(f"Saved model input dataset to: {model_output}")
    print("\nClean dataset shape:", df.shape)
    print("Model input shape:", model_input.shape)
    print("\nPreview of feature matrix:")
    print(feature_matrix.head())
    print("\nNull check on TotalCharges:", df["TotalCharges"].isna().sum())
    print("\nNon-numeric columns in model input:")
    print(model_input.select_dtypes(exclude=[np.number]).columns.tolist())


if __name__ == "__main__":
    main()
