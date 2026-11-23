import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / pipeline execution
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
    save_json,
    load_joblib
)
import config


class ModelEvaluator:
    """Evaluates trained model on the test set, writes DVC metrics and generates plots."""

    def __init__(self, params_path: str = "params.yaml"):
        self.params = load_params(params_path)
        base_cfg = self.params.get("base", {})
        eval_cfg = self.params.get("model_evaluation", {})

        self.target_column = base_cfg.get("target_column", config.TARGET_COLUMN)
        self.test_path = Path(eval_cfg.get("test_data_path", config.TEST_DATA_PATH))
        self.preprocessor_path = Path(eval_cfg.get("preprocessor_path", config.PREPROCESSOR_PATH))
        self.model_path = Path(eval_cfg.get("model_path", config.MODEL_PATH))
        self.metrics_path = Path(eval_cfg.get("metrics_path", config.METRICS_PATH))
        self.predictions_path = Path(eval_cfg.get("predictions_path", config.PREDICTIONS_PATH))
        self.plots_dir = Path(eval_cfg.get("plots_dir", config.PLOTS_DIR))

        self.mlflow_uri = eval_cfg.get("mlflow_tracking_uri", "sqlite:///mlflow.db")
        self.experiment_name = eval_cfg.get("experiment_name", "Mobile_Addiction_Prediction")

        create_directories([self.metrics_path.parent, self.predictions_path.parent, self.plots_dir, config.LOG_DIR])
        self.logger = setup_logger("model_evaluation", config.LOG_DIR / "model_evaluation.log")

    def evaluate(self):
        self.logger.info("=== Starting Model Evaluation Stage ===")

        if not self.test_path.exists():
            raise FileNotFoundError(f"Test dataset not found at: {self.test_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor file not found at: {self.preprocessor_path}")

        # Load artifacts and test dataset
        self.logger.info("Loading model, preprocessor, and test dataset...")
        model = load_joblib(self.model_path)
        preprocessor = load_joblib(self.preprocessor_path)
        test_df = pd.read_csv(self.test_path)

        X_test = test_df.drop(columns=[self.target_column])
        y_test = test_df[self.target_column].values

        X_test_transformed = preprocessor.transform(X_test)
        predictions = model.predict(X_test_transformed)

        # Compute evaluation metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, predictions))
        r2 = float(r2_score(y_test, predictions))

        metrics = {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2_score": round(r2, 4),
            "mse": round(float(mse), 4)
        }

        self.logger.info("Evaluation Metrics: %s", metrics)

        # Save metrics for DVC
        save_json(self.metrics_path, metrics)
        self.logger.info("Metrics written to: %s", self.metrics_path)

        # Save predictions table
        pred_df = pd.DataFrame({
            "actual": y_test,
            "predicted": predictions,
            "residual": y_test - predictions
        })
        pred_df.to_csv(self.predictions_path, index=False)
        self.logger.info("Predictions saved to: %s", self.predictions_path)

        # Generate evaluation plots
        self._generate_plots(pred_df, model, preprocessor)

        # Log to MLflow
        try:
            mlflow.set_tracking_uri(self.mlflow_uri)
            mlflow.set_experiment(self.experiment_name)
            with mlflow.start_run(run_name="evaluate_stage"):
                for metric_name, metric_val in metrics.items():
                    mlflow.log_metric(f"test_{metric_name}", metric_val)
                mlflow.log_artifact(str(self.metrics_path))
                mlflow.log_artifacts(str(self.plots_dir))
            self.logger.info("Evaluation run logged to MLflow at %s", self.mlflow_uri)
        except Exception as e:
            self.logger.warning("MLflow logging encountered an issue: %s", e)

        self.logger.info("=== Model Evaluation Completed Successfully ===")
        print(f"[MODEL EVALUATION] Metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
        print(f"[MODEL EVALUATION] Metrics saved to {self.metrics_path}")
        print(f"[MODEL EVALUATION] Plots saved to {self.plots_dir}")
        return metrics

    def _generate_plots(self, pred_df: pd.DataFrame, model, preprocessor):
        """Generate evaluation plots for DVC tracking."""
        sns.set_theme(style="whitegrid")

        # Plot 1: Actual vs Predicted Scatter
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(pred_df["actual"], pred_df["predicted"], alpha=0.5, color="#1f77b4", edgecolor="none")
        min_val = min(pred_df["actual"].min(), pred_df["predicted"].min())
        max_val = max(pred_df["actual"].max(), pred_df["predicted"].max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Fit (y = x)")
        ax.set_title("Actual vs Predicted Phone Addiction Level", fontsize=14, pad=10)
        ax.set_xlabel("Actual Addiction Level", fontsize=12)
        ax.set_ylabel("Predicted Addiction Level", fontsize=12)
        ax.legend()
        plt.tight_layout()
        plot1_path = self.plots_dir / "actual_vs_predicted.png"
        fig.savefig(plot1_path, dpi=150)
        plt.close(fig)

        # Plot 2: Residuals Distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(pred_df["residual"], kde=True, color="#2ca02c", ax=ax)
        ax.axvline(0, color="red", linestyle="--", lw=2)
        ax.set_title("Residuals Distribution (Actual - Predicted)", fontsize=14, pad=10)
        ax.set_xlabel("Residual Error", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        plt.tight_layout()
        plot2_path = self.plots_dir / "residuals_distribution.png"
        fig.savefig(plot2_path, dpi=150)
        plt.close(fig)

        # Plot 3: Feature Importance (if supported by the model)
        if hasattr(model, "feature_importances_"):
            try:
                feature_names = preprocessor.get_feature_names_out()
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:15]  # Top 15 features

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(range(len(indices)), importances[indices][::-1], align="center", color="#ff7f0e")
                ax.set_yticks(range(len(indices)))
                cleaned_names = [feature_names[i].replace("num__", "").replace("cat__", "") for i in indices][::-1]
                ax.set_yticklabels(cleaned_names)
                ax.set_title("Top 15 Feature Importances", fontsize=14, pad=10)
                ax.set_xlabel("Relative Importance", fontsize=12)
                plt.tight_layout()
                plot3_path = self.plots_dir / "feature_importance.png"
                fig.savefig(plot3_path, dpi=150)
                plt.close(fig)
            except Exception as e:
                self.logger.warning("Could not generate feature importance plot: %s", e)


if __name__ == "__main__":
    params_file = sys.argv[1] if len(sys.argv) > 1 else "params.yaml"
    evaluator = ModelEvaluator(params_file)
    evaluator.evaluate()
