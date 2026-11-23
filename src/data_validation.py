import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.common import load_params, setup_logger, create_directories, save_json
import config


class DataValidation:
    """Validates the ingested data against expected schema and quality criteria."""

    def __init__(self, params_path: str = "params.yaml"):
        self.params = load_params(params_path)
        base_cfg = self.params.get("base", {})
        validation_cfg = self.params.get("data_validation", {})

        self.target_column = base_cfg.get("target_column", config.TARGET_COLUMN)
        self.data_path = Path(validation_cfg.get("ingested_data_path", config.INGESTED_DATA_PATH))
        self.report_path = Path(validation_cfg.get("report_path", config.VALIDATION_REPORT_PATH))
        self.min_target = float(validation_cfg.get("min_target_value", 1.0))
        self.max_target = float(validation_cfg.get("max_target_value", 10.0))

        create_directories([self.report_path.parent, config.LOG_DIR])
        self.logger = setup_logger("data_validation", config.LOG_DIR / "data_validation.log")

    def validate(self) -> dict:
        self.logger.info("=== Starting Data Validation Stage ===")
        self.logger.info("Validating dataset at: %s", self.data_path)

        if not self.data_path.exists():
            err_msg = f"Data file for validation does not exist at: {self.data_path}"
            self.logger.error(err_msg)
            raise FileNotFoundError(err_msg)

        df = pd.read_csv(self.data_path)

        validation_checks = {}
        report = {
            "dataset_path": str(self.data_path),
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": list(df.columns),
            "total_missing_values": int(df.isnull().sum().sum()),
            "total_duplicates": int(df.duplicated().sum()),
            "status": "PENDING"
        }

        # Check 1: Non-empty dataset
        validation_checks["non_empty"] = not df.empty
        if not validation_checks["non_empty"]:
            report["status"] = "FAILED"
            report["error"] = "Dataset is empty"
            save_json(self.report_path, report)
            raise ValueError(report["error"])

        # Check 2: Target column present
        validation_checks["target_column_present"] = self.target_column in df.columns
        if not validation_checks["target_column_present"]:
            report["status"] = "FAILED"
            report["error"] = f"Target column '{self.target_column}' not found"
            save_json(self.report_path, report)
            raise ValueError(report["error"])

        # Check 3: Target column has no missing values
        target_nulls = int(df[self.target_column].isnull().sum())
        validation_checks["target_no_nulls"] = target_nulls == 0
        if not validation_checks["target_no_nulls"]:
            report["status"] = "FAILED"
            report["error"] = f"Target column '{self.target_column}' contains {target_nulls} missing values"
            save_json(self.report_path, report)
            raise ValueError(report["error"])

        # Check 4: Target values range check
        target_min = float(df[self.target_column].min())
        target_max = float(df[self.target_column].max())
        validation_checks["target_within_range"] = (target_min >= self.min_target) and (target_max <= self.max_target)

        report["target_stats"] = {
            "column": self.target_column,
            "min": target_min,
            "max": target_max,
            "mean": float(df[self.target_column].mean()),
            "std": float(df[self.target_column].std())
        }

        if not validation_checks["target_within_range"]:
            report["status"] = "FAILED"
            report["error"] = f"Target values out of expected range [{self.min_target}, {self.max_target}]"
            save_json(self.report_path, report)
            raise ValueError(report["error"])

        # Mark passed
        report["status"] = "PASSED"
        report["checks"] = validation_checks
        save_json(self.report_path, report)

        self.logger.info("Validation report saved to: %s", self.report_path)
        self.logger.info("=== Data Validation Completed Successfully ===")
        print(f"[DATA VALIDATION] PASSED - Report saved to {self.report_path}")
        return report


if __name__ == "__main__":
    params_file = sys.argv[1] if len(sys.argv) > 1 else "params.yaml"
    validator = DataValidation(params_file)
    validator.validate()
