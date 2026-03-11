from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DOCS_DIR = BASE_DIR / "docs"
MODELS_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

for d in [RAW_DIR, PROCESSED_DIR, DOCS_DIR, MODELS_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RAW_DATA_PATH = RAW_DIR / "telco_customer_churn.csv"
PROFILE_OUTPUT_PATH = DOCS_DIR / "data_profile.xlsx"
SCHEMA_REPORT_PATH = DOCS_DIR / "schema_validation_report.csv"

TARGET_COLUMN = "Churn"

EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]
