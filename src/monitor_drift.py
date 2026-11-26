"""Data drift monitoring module for Mobile Usage Addiction Prediction pipeline.

Compares live production inference logs against baseline training data using the
Kolmogorov-Smirnov (KS) two-sample test for numerical distributions.
Generates JSON, HTML, and Markdown reports, and alerts external webhooks.
"""
from pathlib import Path
import argparse
import datetime
import json
import sys
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INFERENCE_LOG_PATH = PROJECT_ROOT / "data" / "monitoring" / "production_inferences.csv"
REFERENCE_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "X_train.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
DRIFT_JSON_PATH = REPORTS_DIR / "drift_report.json"
DRIFT_HTML_PATH = REPORTS_DIR / "drift_report.html"
DRIFT_ALERT_MD_PATH = REPORTS_DIR / "drift_alert.md"

FEATURE_COLUMNS = [
    "daily_usage_hours",
    "sleep_hours",
    "exercise_hours",
    "phone_checks_per_day",
    "apps_used_daily",
    "time_on_social_media",
    "time_on_gaming",
    "weekend_minus_daily",
    "usage_per_check",
    "checks_per_hour",
    "social_gaming_total",
    "leisure_share",
    "apps_x_usage",
    "usage_sleep_ratio",
    "bedtime_share",
]

RAW_FEATURE_COLUMNS = [
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
    "Gender",
    "Phone_Usage_Purpose",
]

NUMERICAL_FEATURES = [c for c in RAW_FEATURE_COLUMNS if c not in ["Gender", "Phone_Usage_Purpose"]]
CATEGORICAL_FEATURES = ["Gender", "Phone_Usage_Purpose"]


def log_inference(
    input_data: pd.DataFrame,
    prediction: float,
    probas=None,
    risk_level: str = "Moderate",
    log_path: Path = INFERENCE_LOG_PATH,
) -> None:
    """Append a single or batch production inference request to monitoring logs."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    df_to_log = input_data.copy()
    df_to_log["predicted_class"] = str(prediction)
    df_to_log["predicted_addiction_level"] = round(float(prediction), 2)
    df_to_log["risk_level"] = risk_level
    df_to_log["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    if probas and isinstance(probas, dict):
        for k, v in probas.items():
            col_name = k if k.startswith("prob_") else f"prob_{k}"
            df_to_log[col_name] = v

    header = not log_path.exists()
    df_to_log.to_csv(log_path, mode="a", index=False, header=header)


def load_inference_logs(log_path: Path = INFERENCE_LOG_PATH) -> pd.DataFrame:
    """Load production inference logs if they exist."""
    path = Path(log_path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def analyze_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    p_value_threshold: float = 0.05,
    drift_share_threshold: float = 0.33,
) -> dict:
    """Run Kolmogorov-Smirnov two-sample tests across all features."""
    results = {}
    drifted_count = 0

    available_features = [f for f in FEATURE_COLUMNS if f in reference_df.columns and f in current_df.columns]
    if not available_features:
        available_features = [f for f in reference_df.columns if f in current_df.columns and pd.api.types.is_numeric_dtype(reference_df[f])]

    for feat in available_features:
        ref_values = reference_df[feat].dropna().values
        curr_values = current_df[feat].dropna().values

        if len(ref_values) < 2 or len(curr_values) < 2:
            results[feat] = {
                "drift_detected": False,
                "p_value": 1.0,
                "ks_statistic": 0.0,
                "status": "Insufficient Data",
            }
            continue

        ks_stat, p_val = stats.ks_2samp(ref_values, curr_values)
        drift_detected = bool(p_val < p_value_threshold)
        if drift_detected:
            drifted_count += 1

        ref_mean = float(np.mean(ref_values))
        curr_mean = float(np.mean(curr_values))

        results[feat] = {
            "drift_detected": drift_detected,
            "p_value": round(float(p_val), 6),
            "ks_statistic": round(float(ks_stat), 4),
            "reference_mean": round(ref_mean, 3),
            "reference_std": round(float(np.std(ref_values)), 3),
            "current_mean": round(curr_mean, 3),
            "current_std": round(float(np.std(curr_values)), 3),
            "mean_shift": round(curr_mean - ref_mean, 3),
        }

    total_feats = len(available_features)
    drift_share = (drifted_count / total_feats) if total_feats > 0 else 0.0
    dataset_drift = bool(drift_share >= drift_share_threshold)

    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_features": total_feats,
        "drifted_features": drifted_count,
        "drift_share_pct": round(drift_share * 100.0, 2),
        "dataset_drift_detected": dataset_drift,
        "p_value_threshold": p_value_threshold,
        "drift_share_threshold": drift_share_threshold,
        "reference_samples": len(reference_df),
        "current_samples": len(current_df),
        "features": results,
    }


def generate_sample_inferences(
    n_samples: int = 50,
    drift: bool = False,
    log_path: Path = INFERENCE_LOG_PATH,
) -> pd.DataFrame:
    """Generate representative synthetic production samples for monitoring demonstrations."""
    np.random.seed(42)
    daily_usage = np.random.normal(5.5, 1.5, n_samples).clip(1.0, 12.0)
    if drift:
        daily_usage += 4.0  # simulate massive shift in phone usage

    sleep = np.random.normal(6.5, 1.2, n_samples).clip(3.0, 10.0)
    exercise = np.random.normal(1.2, 0.6, n_samples).clip(0.0, 4.0)
    checks = np.random.normal(85.0, 25.0, n_samples).clip(20.0, 150.0)
    apps = np.random.normal(12.0, 4.0, n_samples).clip(5.0, 20.0)
    social = np.random.normal(2.5, 0.8, n_samples).clip(0.0, 5.0)
    gaming = np.random.normal(1.5, 0.7, n_samples).clip(0.0, 4.0)
    weekend_diff = np.random.normal(1.2, 0.8, n_samples)

    data = {
        "daily_usage_hours": daily_usage,
        "sleep_hours": sleep,
        "exercise_hours": exercise,
        "phone_checks_per_day": checks,
        "apps_used_daily": apps,
        "time_on_social_media": social,
        "time_on_gaming": gaming,
        "weekend_minus_daily": weekend_diff,
        "usage_per_check": daily_usage / (checks + 1.0),
        "checks_per_hour": checks / (daily_usage + 0.1),
        "social_gaming_total": social + gaming,
        "leisure_share": (social + gaming) / (daily_usage + 0.1),
        "apps_x_usage": apps * daily_usage,
        "usage_sleep_ratio": daily_usage / (sleep + 0.1),
        "bedtime_share": np.random.normal(1.0, 0.4, n_samples).clip(0.1, 2.5) / (daily_usage + 0.1),
    }
    df = pd.DataFrame(data)

    for i in range(len(df)):
        sample = df.iloc[[i]]
        pred = min(10.0, max(1.0, 0.5 * sample["daily_usage_hours"].values[0] + 0.3 * sample["usage_sleep_ratio"].values[0] + 1.0))
        risk = "High" if pred >= 7.0 else ("Moderate" if pred >= 4.0 else "Low")
        log_inference(sample, pred, risk_level=risk, log_path=log_path)

    return load_inference_logs(log_path)


def render_html_report(report: dict, output_path: Path = DRIFT_HTML_PATH) -> str:
    """Render comprehensive interactive HTML data drift report."""
    output_path = Path(output_path)
    status_color = "#e53e3e" if report["dataset_drift_detected"] else "#38a169"
    status_text = "DATA DRIFT DETECTED" if report["dataset_drift_detected"] else "HEALTHY DISTRIBUTION"

    rows = ""
    for feat, d in report["features"].items():
        flag_color = "#e53e3e" if d["drift_detected"] else "#38a169"
        flag_text = "DRIFT" if d["drift_detected"] else "OK"
        rows += f"""
        <tr>
            <td><strong>{feat}</strong></td>
            <td><span style="color: {flag_color}; font-weight: bold;">{flag_text}</span></td>
            <td>{d['p_value']:.4f}</td>
            <td>{d['ks_statistic']:.4f}</td>
            <td>{d.get('reference_mean', 0.0):.2f} ± {d.get('reference_std', 0.0):.2f}</td>
            <td>{d.get('current_mean', 0.0):.2f} ± {d.get('current_std', 0.0):.2f}</td>
            <td>{d.get('mean_shift', 0.0):+.2f}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Production Data Drift Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 2rem; background: #f8fafc; color: #1e293b; }}
        .header {{ background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 2rem; }}
        .status-badge {{ display: inline-block; padding: 0.5rem 1rem; border-radius: 9999px; color: white; background: {status_color}; font-weight: bold; font-size: 1.1rem; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        th, td {{ padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #f1f5f9; font-weight: 600; }}
        tr:hover {{ background: #f8fafc; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Mobile Usage Addiction Predictor - Data Drift Monitoring</h1>
        <p>Generated at: <code>{report['timestamp']}</code></p>
        <div class="status-badge">{status_text}</div>
        <p style="margin-top: 1rem;">
            Drifted features: <strong>{report['drifted_features']} / {report['total_features']}</strong>
            ({report['drift_share_pct']}%) | Reference baseline samples: <code>{report['reference_samples']}</code> | Production logs: <code>{report['current_samples']}</code>
        </p>
    </div>
    <table>
        <thead>
            <tr>
                <th>Feature Name</th>
                <th>Status</th>
                <th>p-value (KS Test)</th>
                <th>KS Statistic</th>
                <th>Baseline (Mean ± Std)</th>
                <th>Production (Mean ± Std)</th>
                <th>Shift</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
</body>
</html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return html


def generate_alert_markdown(report: dict, output_path: Path = DRIFT_ALERT_MD_PATH) -> str:
    """Generate Markdown summary suitable for GitHub Actions step summary and alerts."""
    output_path = Path(output_path) if output_path else DRIFT_ALERT_MD_PATH
    status_text = "DATA DRIFT DETECTED" if report["dataset_drift_detected"] else "HEALTHY DISTRIBUTION"
    action_note = "Continuous Training Required" if report["dataset_drift_detected"] else "Healthy Distribution"

    table_rows = []
    for feat, d in report["features"].items():
        flag = "DRIFT ⚠️" if d["drift_detected"] else "OK ✅"
        table_rows.append(
            f"| `{feat}` | {flag} | {d['p_value']:.4f} | {d['ks_statistic']:.4f} | "
            f"{d.get('reference_mean', 0.0):.2f} ± {d.get('reference_std', 0.0):.2f} | "
            f"{d.get('current_mean', 0.0):.2f} ± {d.get('current_std', 0.0):.2f} | "
            f"{d.get('mean_shift', 0.0):+.2f} |"
        )
    table_content = "\n".join(table_rows)

    md = f"""## 🚨 {status_text}
- **Status:** {action_note}
- **Evaluation Timestamp:** `{report['timestamp']}`
- **Drifted Features:** `{report['drifted_features']} / {report['total_features']}` ({report['drift_share_pct']}%)
- **Test Method:** Kolmogorov-Smirnov (KS) two-sample test (threshold p < {report['p_value_threshold']})

### Feature Breakdown

| Feature | Status | p-value | KS Statistic | Reference Mean | Current Mean | Shift |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{table_content}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    return md


def send_drift_alert(report: dict, webhook_url: str = None) -> bool:
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
