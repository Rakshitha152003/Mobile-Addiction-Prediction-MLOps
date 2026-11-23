import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import DataPreprocessor


class DataCleaning:
    """Wrapper class for backwards compatibility with previous data_cleaning calls."""

    def __init__(self, params_path: str = "params.yaml"):
        self.preprocessor = DataPreprocessor(params_path)

    def clean_data(self):
        train_df, test_df = self.preprocessor.run()
        return train_df


if __name__ == "__main__":
    params_file = sys.argv[1] if len(sys.argv) > 1 else "params.yaml"
    cleaner = DataCleaning(params_file)
    cleaner.clean_data()
