from pathlib import Path

# Project Root Directory
ROOT_DIR = Path(__file__).resolve().parent

# Data Directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset Paths
RAW_DATA_PATH = RAW_DATA_DIR / "phone_addiction_dataset.csv"
INGESTED_DATA_PATH = PROCESSED_DATA_DIR / "ingested_data.csv"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_data.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

# Model Directory & Artifacts
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"

# Reports Directory & Artifacts
REPORT_DIR = ROOT_DIR / "reports"
VALIDATION_REPORT_PATH = REPORT_DIR / "validation_report.json"
METRICS_PATH = REPORT_DIR / "metrics.json"
PREDICTIONS_PATH = REPORT_DIR / "predictions.csv"
PLOTS_DIR = REPORT_DIR / "plots"

# Logs Directory
LOG_DIR = ROOT_DIR / "logs"

# Params File
PARAMS_PATH = ROOT_DIR / "params.yaml"

# Target Column
TARGET_COLUMN = "Addiction_Level"
