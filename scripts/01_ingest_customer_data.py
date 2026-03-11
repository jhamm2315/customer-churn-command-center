from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config.settings import RAW_DATA_PATH

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


def main():
    df = pd.read_csv(DATA_URL)
    df.to_csv(RAW_DATA_PATH, index=False)

    print("Customer churn dataset ingested successfully.")
    print(f"Saved to: {RAW_DATA_PATH}")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()
