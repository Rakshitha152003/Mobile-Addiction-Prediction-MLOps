import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.common import load_params, setup_logger, create_directories, save_joblib
import config


class DataPreprocessor:
    """Preprocesses raw/ingested data, engineers features, and splits into train/test sets."""

    def __init__(self, params_path: str = "params.yaml"):
        self.params = load_params(params_path)
        base_cfg = self.params.get("base", {})
        prep_cfg = self.params.get("data_preprocessing", {})

        self.target_column = base_cfg.get("target_column", config.TARGET_COLUMN)
        self.ingested_path = Path(prep_cfg.get("ingested_data_path", config.INGESTED_DATA_PATH))
        self.cleaned_path = Path(prep_cfg.get("cleaned_data_path", config.CLEANED_DATA_PATH))
        self.train_path = Path(prep_cfg.get("train_data_path", config.TRAIN_DATA_PATH))
        self.test_path = Path(prep_cfg.get("test_data_path", config.TEST_DATA_PATH))
        self.preprocessor_path = Path(prep_cfg.get("preprocessor_path", config.PREPROCESSOR_PATH))

        self.test_size = float(prep_cfg.get("test_size", 0.2))
        self.random_state = int(prep_cfg.get("random_state", 42))
        self.drop_columns = prep_cfg.get("drop_columns", ["Name", "Location"])
        self.categorical_columns = prep_cfg.get("categorical_columns", ["Gender", "Phone_Usage_Purpose"])

        create_directories([self.cleaned_path.parent, self.preprocessor_path.parent, config.LOG_DIR])
        self.logger = setup_logger("data_preprocessing", config.LOG_DIR / "data_cleaning.log")

    def run(self):
        self.logger.info("=== Starting Data Preprocessing Stage ===")
        self.logger.info("Loading ingested dataset from: %s", self.ingested_path)

        if not self.ingested_path.exists():
            err_msg = f"Ingested data not found at: {self.ingested_path}"
            self.logger.error(err_msg)
            raise FileNotFoundError(err_msg)

        df = pd.read_csv(self.ingested_path)

        # 1. Clean column names
        df.columns = df.columns.str.strip().str.replace(" ", "_")

        # 2. Drop duplicates
        initial_rows = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        dropped_duplicates = initial_rows - len(df)
        self.logger.info("Removed %d duplicate rows. Remaining rows: %d", dropped_duplicates, len(df))

        # 3. Handle missing values if any
        if df.isnull().sum().sum() > 0:
            self.logger.warning("Found %d missing values. Filling numerical with median and categorical with mode.", df.isnull().sum().sum())
            for col in df.columns:
                if df[col].isnull().any():
                    if df[col].dtype in [np.float64, np.int64]:
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0])

        # 4. Save cleaned dataset for backwards compatibility / EDA
        df.to_csv(self.cleaned_path, index=False)
        self.logger.info("Cleaned dataset saved to: %s", self.cleaned_path)

        # 5. Drop uninformative / high cardinality columns
        cols_to_drop = [c for c in self.drop_columns if c in df.columns]
        if cols_to_drop:
            self.logger.info("Dropping columns: %s", cols_to_drop)
            df = df.drop(columns=cols_to_drop)

        # 6. Separate features and target
        if self.target_column not in df.columns:
            raise KeyError(f"Target column '{self.target_column}' missing after column cleaning")

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # Identify categorical and numerical features
        cat_cols = [c for c in self.categorical_columns if c in X.columns]
        num_cols = [c for c in X.columns if c not in cat_cols]

        self.logger.info("Categorical features (%d): %s", len(cat_cols), cat_cols)
        self.logger.info("Numerical features (%d): %s", len(num_cols), num_cols)

        # 7. Train / test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        self.logger.info("Train set shape: %s | Test set shape: %s", X_train.shape, X_test.shape)

        # 8. Build preprocessor pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
            ],
            remainder="passthrough"
        )

        # Fit preprocessor on train features
        preprocessor.fit(X_train)

        # Save preprocessor artifact
        save_joblib(self.preprocessor_path, preprocessor)
        self.logger.info("Preprocessor saved to: %s", self.preprocessor_path)

        # 9. Save train and test sets (keeping original raw features + target column for readability)
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        train_df.to_csv(self.train_path, index=False)
        test_df.to_csv(self.test_path, index=False)

        self.logger.info("Train dataset saved to: %s (%d rows)", self.train_path, len(train_df))
        self.logger.info("Test dataset saved to: %s (%d rows)", self.test_path, len(test_df))
        self.logger.info("=== Data Preprocessing Completed Successfully ===")

        print(f"[DATA PREPROCESSING] Cleaned: {self.cleaned_path}")
        print(f"[DATA PREPROCESSING] Train: {self.train_path} ({len(train_df)} rows)")
        print(f"[DATA PREPROCESSING] Test: {self.test_path} ({len(test_df)} rows)")
        print(f"[DATA PREPROCESSING] Preprocessor: {self.preprocessor_path}")

        return train_df, test_df


if __name__ == "__main__":
    params_file = sys.argv[1] if len(sys.argv) > 1 else "params.yaml"
    preprocessor = DataPreprocessor(params_file)
    preprocessor.run()
