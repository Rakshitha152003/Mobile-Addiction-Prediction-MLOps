import os
import sys
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Silence MLflow hints and allow file store fallback
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.common import (
    load_params,
    setup_logger,
    create_directories,
    load_joblib,
    save_joblib
)
import config


class ModelTrainer:
    """Trains a machine learning regression model and logs training runs with MLflow."""

    def __init__(self, params_path: str = "params.yaml"):
        self.params = load_params(params_path)
        base_cfg = self.params.get("base", {})
        train_cfg = self.params.get("model_training", {})
        eval_cfg = self.params.get("model_evaluation", {})

        self.target_column = base_cfg.get("target_column", config.TARGET_COLUMN)
        self.train_path = Path(train_cfg.get("train_data_path", config.TRAIN_DATA_PATH))
        self.preprocessor_path = Path(train_cfg.get("preprocessor_path", config.PREPROCESSOR_PATH))
        self.model_path = Path(train_cfg.get("model_path", config.MODEL_PATH))

        self.algorithm = train_cfg.get("algorithm", "RandomForestRegressor")
        self.model_params = train_cfg.get("params", {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "random_state": 42
        })

        self.mlflow_uri = eval_cfg.get("mlflow_tracking_uri", "sqlite:///mlflow.db")
        self.experiment_name = eval_cfg.get("experiment_name", "Mobile_Addiction_Prediction")

        create_directories([self.model_path.parent, config.LOG_DIR])
        self.logger = setup_logger("model_training", config.LOG_DIR / "model_training.log")

    def _get_model(self):
        if self.algorithm == "RandomForestRegressor":
            return RandomForestRegressor(**self.model_params)
        elif self.algorithm == "GradientBoostingRegressor":
            return GradientBoostingRegressor(**self.model_params)
        else:
            self.logger.warning("Unknown algorithm '%s'. Defaulting to RandomForestRegressor.", self.algorithm)
            return RandomForestRegressor(**self.model_params)

    def train(self):
        self.logger.info("=== Starting Model Training Stage ===")
        self.logger.info("Loading training data from: %s", self.train_path)

        if not self.train_path.exists():
            err_msg = f"Training data file not found at: {self.train_path}"
            self.logger.error(err_msg)
            raise FileNotFoundError(err_msg)

        if not self.preprocessor_path.exists():
            err_msg = f"Preprocessor file not found at: {self.preprocessor_path}"
            self.logger.error(err_msg)
            raise FileNotFoundError(err_msg)

        train_df = pd.read_csv(self.train_path)
        X_train = train_df.drop(columns=[self.target_column])
        y_train = train_df[self.target_column]

        self.logger.info("Loading preprocessor and transforming features...")
        preprocessor = load_joblib(self.preprocessor_path)
        X_train_transformed = preprocessor.transform(X_train)

        self.logger.info("Instantiating %s with params: %s", self.algorithm, self.model_params)
        model = self._get_model()

        self.logger.info("Fitting model on %d samples with %d features...", len(y_train), X_train_transformed.shape[1])
        model.fit(X_train_transformed, y_train)

        train_score = model.score(X_train_transformed, y_train)
        self.logger.info("Train R^2 score: %.4f", train_score)

        # Save model artifact locally
        save_joblib(self.model_path, model)
        self.logger.info("Trained model saved to: %s", self.model_path)

        # Log with MLflow
        try:
            mlflow.set_tracking_uri(self.mlflow_uri)
            mlflow.set_experiment(self.experiment_name)
            with mlflow.start_run(run_name="train_stage"):
                mlflow.log_param("algorithm", self.algorithm)
                for k, v in self.model_params.items():
                    mlflow.log_param(k, v)
                mlflow.log_metric("train_r2_score", train_score)
                mlflow.log_artifact(str(self.model_path))
            self.logger.info("Training run logged to MLflow at %s", self.mlflow_uri)
        except Exception as e:
            self.logger.warning("MLflow logging encountered an issue: %s", e)

        self.logger.info("=== Model Training Completed Successfully ===")
        print(f"[MODEL TRAINING] Model ({self.algorithm}) saved to {self.model_path} (Train R^2: {train_score:.4f})")
        return model


if __name__ == "__main__":
    params_file = sys.argv[1] if len(sys.argv) > 1 else "params.yaml"
    trainer = ModelTrainer(params_file)
    trainer.train()
