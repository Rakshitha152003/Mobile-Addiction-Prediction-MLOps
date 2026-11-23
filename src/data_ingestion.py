import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.common import load_params, setup_logger, create_directories
import config


def data_ingestion(params_path: str = "params.yaml") -> Path:
    """Ingest raw dataset, validate presence, and save to processed directory."""
    params = load_params(params_path)
    base_cfg = params.get("base", {})
    ingestion_cfg = params.get("data_ingestion", {})

    target_column = base_cfg.get("target_column", config.TARGET_COLUMN)
    raw_path = Path(ingestion_cfg.get("raw_data_path", config.RAW_DATA_PATH))
    ingested_path = Path(ingestion_cfg.get("ingested_data_path", config.INGESTED_DATA_PATH))
    log_dir = Path(config.LOG_DIR)

    create_directories([ingested_path.parent, log_dir])
    logger = setup_logger("data_ingestion", log_dir / "data_ingestion.log")

    logger.info("=== Starting Data Ingestion Stage ===")
    logger.info("Reading raw data from: %s", raw_path)

    if not raw_path.exists():
        err_msg = f"Raw data file not found at: {raw_path}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    df = pd.read_csv(raw_path)

    if df.empty:
        err_msg = "Raw dataset is completely empty"
        logger.error(err_msg)
        raise ValueError(err_msg)

    if target_column not in df.columns:
        err_msg = f"Target column '{target_column}' missing from raw dataset"
        logger.error(err_msg)
        raise ValueError(err_msg)

    logger.info("Dataset shape: %s (Rows: %d, Cols: %d)", df.shape, df.shape[0], df.shape[1])
    logger.info("Missing values count: %d", int(df.isnull().sum().sum()))
    logger.info("Duplicate records count: %d", int(df.duplicated().sum()))

    df.to_csv(ingested_path, index=False)
    logger.info("Successfully ingested data saved to: %s", ingested_path)
    logger.info("=== Data Ingestion Completed Successfully ===")

    print(f"[DATA INGESTION] Dataset saved to {ingested_path} ({df.shape[0]} rows, {df.shape[1]} columns)")
    return ingested_path


if __name__ == "__main__":
    params_file = sys.argv[1] if len(sys.argv) > 1 else "params.yaml"
    data_ingestion(params_file)
