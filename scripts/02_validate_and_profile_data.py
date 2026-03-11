from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config.settings import (
    RAW_DATA_PATH,
    PROFILE_OUTPUT_PATH,
    SCHEMA_REPORT_PATH,
    EXPECTED_COLUMNS,
    TARGET_COLUMN,
)


def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    profile = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "null_count": df.isna().sum().values,
        "null_pct": (df.isna().mean() * 100).round(2).values,
        "distinct_count": df.nunique(dropna=False).values,
    })
    return profile


def main():
    df = pd.read_csv(RAW_DATA_PATH)

    actual_columns = list(df.columns)
    missing_columns = sorted(list(set(EXPECTED_COLUMNS) - set(actual_columns)))
    unexpected_columns = sorted(list(set(actual_columns) - set(EXPECTED_COLUMNS)))

    schema_report = pd.DataFrame([{
        "row_count": len(df),
        "column_count": len(df.columns),
        "target_column_present": TARGET_COLUMN in df.columns,
        "missing_columns": ", ".join(missing_columns) if missing_columns else "",
        "unexpected_columns": ", ".join(unexpected_columns) if unexpected_columns else "",
        "schema_valid": (len(missing_columns) == 0 and TARGET_COLUMN in df.columns),
    }])

    profile_df = profile_dataframe(df)

    # duplicate check
    duplicate_report = pd.DataFrame([{
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_customer_ids": int(df["customerID"].duplicated().sum()) if "customerID" in df.columns else None,
    }])

    # class balance
    if TARGET_COLUMN in df.columns:
        churn_distribution = (
            df[TARGET_COLUMN]
            .value_counts(dropna=False)
            .rename_axis("target_value")
            .reset_index(name="count")
        )
        churn_distribution["pct"] = (churn_distribution["count"] / churn_distribution["count"].sum() * 100).round(2)
    else:
        churn_distribution = pd.DataFrame()

    schema_report.to_csv(SCHEMA_REPORT_PATH, index=False)

    with pd.ExcelWriter(PROFILE_OUTPUT_PATH, engine="openpyxl") as writer:
        profile_df.to_excel(writer, sheet_name="profile", index=False)
        schema_report.to_excel(writer, sheet_name="schema_report", index=False)
        duplicate_report.to_excel(writer, sheet_name="duplicates", index=False)
        churn_distribution.to_excel(writer, sheet_name="target_distribution", index=False)

    print("Validation and profiling complete.")
    print(f"Schema report saved to: {SCHEMA_REPORT_PATH}")
    print(f"Profile workbook saved to: {PROFILE_OUTPUT_PATH}")
    print("\nSchema report:")
    print(schema_report)
    print("\nDuplicate report:")
    print(duplicate_report)
    print("\nTarget distribution:")
    print(churn_distribution)

if __name__ == "__main__":
    main()
