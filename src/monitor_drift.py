import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Feature columns for Mobile Addiction dataset
NUMERICAL_FEATURES: List[str] = [
    "Age",
    "Daily_Usage_Hours",
    "Sleep_Hours",
    "Interllectual_Performance",
    "Social_Interactions",
    "Exercise_Hours",
    "Anxiety_Level",
    "Depression_Level",
    "Self_Esteem",
    "Screen_Time_Before_Bed",
    "Phone_Checks_Per_Day",
    "Apps_Used_Daily",
    "Time_on_Social_Media",
    "Time_on_Gaming",
    "Time_on_Education",
    "Family_Communication",
    "Weekend_Usage_Hours",
]

CATEGORICAL_FEATURES: List[str] = [
    "Gender",
    "Phone_Usage_Purpose",
]

FEATURE_COLUMNS: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

DEFAULT_LOG_PATH: Path = PROJECT_ROOT / "logs" / "inference_logs.csv"
INFERENCE_LOG_PATH: Path = DEFAULT_LOG_PATH
REFERENCE_DATA_PATH: Path = PROJECT_ROOT / "data" / "processed" / "train.csv"
DEFAULT_ALERT_PATH: Path = PROJECT_ROOT / "reports" / "drift_alert.md"


def load_inference_logs(log_path: Union[str, Path] = INFERENCE_LOG_PATH) -> pd.DataFrame:
    """Load logged inference records from CSV. Returns empty DataFrame if file not found."""
    path = Path(log_path)
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def log_inference(
    input_sample: pd.DataFrame,
    prediction: Any,
    probas: Optional[Dict[str, float]] = None,
    log_path: Union[str, Path] = DEFAULT_LOG_PATH,
) -> Path:
    """Log an inference request, input features, and prediction to a CSV file."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = input_sample.copy()
    if not isinstance(record, pd.DataFrame):
        record = pd.DataFrame([record])

    record["timestamp"] = datetime.datetime.now().isoformat()
    record["predicted_class"] = str(prediction)
    record["predicted_value"] = float(prediction) if isinstance(prediction, (int, float, np.number)) else str(prediction)

    # If probabilities or extra metadata provided, unpack them
    if probas:
        for k, v in probas.items():
            col_name = k if k.startswith("prob_") else f"prob_{k}"
            record[col_name] = v

    # Append to file
    header = not path.exists()
    record.to_csv(path, mode="a", header=header, index=False)
    return path


def analyze_drift(
    ref_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    p_val_threshold: float = 0.05,
    drift_share_threshold: float = 0.3,
) -> Dict[str, Any]:
    """Compare reference and current datasets to detect data drift using Kolmogorov-Smirnov test."""
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLUMNS if c in ref_df.columns and c in curr_df.columns]
        if not feature_cols:
            feature_cols = [c for c in ref_df.columns if c in curr_df.columns]

    drifted_count = 0
    feature_results: Dict[str, Dict[str, Any]] = {}

    for col in feature_cols:
        ref_series = ref_df[col].dropna()
        curr_series = curr_df[col].dropna()

        is_num = pd.api.types.is_numeric_dtype(ref_series) and pd.api.types.is_numeric_dtype(curr_series)

        ref_mean = float(ref_series.mean()) if is_num else 0.0
        curr_mean = float(curr_series.mean()) if is_num else 0.0
        mean_shift = float(curr_mean - ref_mean) if is_num else 0.0

        if is_num:
            stat, p_val = ks_2samp(ref_series, curr_series)
            test_name = "Kolmogorov-Smirnov"
            is_drifted = bool(p_val < p_val_threshold)
        else:
            test_name = "Category Frequency Shift"
            ref_dist = ref_series.value_counts(normalize=True)
            curr_dist = curr_series.value_counts(normalize=True)
            all_keys = set(ref_dist.index).union(set(curr_dist.index))
            diff = sum(abs(ref_dist.get(k, 0.0) - curr_dist.get(k, 0.0)) for k in all_keys) / 2.0
            stat = float(diff)
            p_val = 0.0 if diff > 0.15 else 1.0
            is_drifted = bool(p_val < p_val_threshold)

        if is_drifted:
            drifted_count += 1

        feature_results[col] = {
            "test_name": test_name,
            "statistic": round(float(stat), 4),
            "ks_statistic": round(float(stat), 4),
            "p_value": round(float(p_val), 5),
            "drift_detected": is_drifted,
            "reference_mean": round(ref_mean, 4),
            "current_mean": round(curr_mean, 4),
            "mean_shift": round(mean_shift, 4),
        }

    total_features = len(feature_cols)
    drift_share_pct = round((drifted_count / total_features * 100.0), 2) if total_features > 0 else 0.0
    dataset_drift_detected = bool((drift_share_pct / 100.0) >= drift_share_threshold)

    report = {
        "dataset_drift_detected": dataset_drift_detected,
        "drifted_features": drifted_count,
        "total_features": total_features,
        "drift_share_pct": drift_share_pct,
        "p_val_threshold": p_val_threshold,
        "drift_share_threshold": drift_share_threshold,
        "timestamp": datetime.datetime.now().isoformat(),
        "features": feature_results,
    }
    return report


def generate_alert_markdown(
    report: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Generate a formatted markdown report describing the drift analysis results."""
    drift_detected = report.get("dataset_drift_detected", False)
    drifted_count = report.get("drifted_features", 0)
    total_count = report.get("total_features", 0)
    drift_share = report.get("drift_share_pct", 0.0)

    header_status = "🚨 DATA DRIFT DETECTED" if drift_detected else "✅ NO DATA DRIFT DETECTED"
    action_note = "**Status:** Continuous Training Required" if drift_detected else "**Status:** Model Distribution Healthy"

    lines = [
        f"# {header_status}",
        "",
        f"- {action_note}",
        f"- **Timestamp:** `{report.get('timestamp', datetime.datetime.now().isoformat())}`",
        f"- **Drifted Features:** `{drifted_count} / {total_count}` ({drift_share}%)",
        f"- **Statistical Test:** Kolmogorov-Smirnov (Numerical) / Frequency (Categorical)",
        "",
        "## Feature Drift Analysis",
        "",
        "| Feature | Statistical Test | Statistic | p-value | Drift Detected | Reference Mean | Production Mean | Mean Shift |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    features = report.get("features", {})
    for feat, details in features.items():
        test_name = details.get("test_name", "Kolmogorov-Smirnov")
        stat = details.get("statistic", 0.0)
        p_val = details.get("p_value", 1.0)
        drift_flag = "⚠️ **YES**" if details.get("drift_detected", False) else "No"
        ref_m = details.get("reference_mean", 0.0)
        curr_m = details.get("current_mean", 0.0)
        m_shift = details.get("mean_shift", 0.0)
        lines.append(f"| `{feat}` | {test_name} | {stat} | {p_val} | {drift_flag} | {ref_m} | {curr_m} | {m_shift} |")

    lines.append("")
    if drift_detected:
        lines.extend([
            "## 📌 Action Plan",
            "- Continuous Training Required: The production data distribution has significantly shifted from training data.",
            "- Retrain the model using the latest verified samples by executing `dvc repro`.",
            "- Inspect features with `drift_detected == True` for systemic shifts in user behavior.",
        ])
    else:
        lines.extend([
            "## 📌 Action Plan",
            "- No retraining needed at this time.",
            "- Continue real-time inference logging.",
        ])

    md_content = "\n".join(lines) + "\n"

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(md_content)

    return md_content


def send_drift_alert(
    report: Dict[str, Any],
    webhook_url: Optional[str] = None,
) -> bool:
    """Send drift alert to a configured webhook (e.g. Slack/Teams/Discord). Gracefully fails if no URL."""
    if not webhook_url:
        return False

    try:
        import urllib.request
        payload = json.dumps({
            "text": f"🚨 Mobile Addiction Model Drift Alert!\nDrifted Features: {report.get('drifted_features')}/{report.get('total_features')} ({report.get('drift_share_pct')}%)"
        }).encode("utf-8")
        req = urllib.request.Request(webhook_url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False
