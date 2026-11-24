from pathlib import Path
import pandas as pd
import pytest

from src.monitor_drift import FEATURE_COLUMNS, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

RAW_DATA_PATH = Path("data/raw/phone_addiction_dataset.csv")
TRAIN_DATA_PATH = Path("data/processed/train.csv")
TEST_DATA_PATH = Path("data/processed/test.csv")
CLEANED_DATA_PATH = Path("data/processed/cleaned_data.csv")
TARGET_COLUMN = "Addiction_Level"


def test_raw_dataset_exists_and_valid():
    """Verify raw Phone Addiction dataset exists, has 6000 rows, expected columns, and valid target."""
    assert RAW_DATA_PATH.exists(), f"Raw dataset not found at {RAW_DATA_PATH}"
    df = pd.read_csv(RAW_DATA_PATH)

    assert len(df) == 6000, f"Expected 6000 rows in raw dataset, got {len(df)}"
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        assert col in df.columns, f"Missing column {col} in raw data"

    # Verify target variable values
    assert df[TARGET_COLUMN].isnull().sum() == 0, "Null values found in target column"
    assert df[TARGET_COLUMN].between(1.0, 10.0).all(), "Target values out of [1.0, 10.0] range"


def test_preprocessed_splits_exist_and_consistent():
    """Verify train and test splits exist, have no duplicates, and match deduplicated total."""
    assert TRAIN_DATA_PATH.exists(), f"Train set not found at {TRAIN_DATA_PATH}"
    assert TEST_DATA_PATH.exists(), f"Test set not found at {TEST_DATA_PATH}"

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    # 6000 raw rows - 32 duplicates = 5968 distinct rows
    total_rows = len(train_df) + len(test_df)
    assert total_rows == 5968, f"Expected 5968 deduplicated rows across splits, got {total_rows}"

    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        assert col in train_df.columns, f"Missing {col} in train set"
        assert col in test_df.columns, f"Missing {col} in test set"

    # Verify no nulls in splits
    assert train_df.isnull().sum().sum() == 0, "Null values found in train set"
    assert test_df.isnull().sum().sum() == 0, "Null values found in test set"

    # Verify target bounds
    assert train_df[TARGET_COLUMN].between(1.0, 10.0).all()
    assert test_df[TARGET_COLUMN].between(1.0, 10.0).all()
