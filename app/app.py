"""Streamlit serving application for the Mobile Phone Addiction MLOps Pipeline.

Features:
- Live model inference with risk level and addiction score visualization.
- Production payload and prediction logging for data drift tracking.
- Interactive model metadata and lineage inspector.
- Real-time data drift monitoring dashboard.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path before importing local packages
PROJECT_ROOT = Path(__file__).resolve().parent.parent if (Path(__file__).resolve().parent.parent / "models").exists() else Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import joblib
import pandas as pd
import streamlit as st

from src.monitor_drift import (
    FEATURE_COLUMNS,
    INFERENCE_LOG_PATH,
    REFERENCE_DATA_PATH,
    analyze_drift,
    load_inference_logs,
    log_inference,
)

# Project paths
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
VALIDATION_PATH = PROJECT_ROOT / "reports" / "validation_report.json"
PLOTS_DIR = PROJECT_ROOT / "reports" / "plots"

st.set_page_config(
    page_title="Mobile Addiction MLOps Production Portal",
    page_icon="📱",
    layout="wide",
)


@st.cache_resource
def load_champion_model():
    """Load the champion model and fitted preprocessor pipeline."""
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


@st.cache_data
def load_reports():
    """Load evaluation metrics and validation reports."""
    metrics, validation = None, None
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    if VALIDATION_PATH.exists():
        with open(VALIDATION_PATH, "r", encoding="utf-8") as f:
            validation = json.load(f)
    return metrics, validation


def get_risk_info(score: float):
    """Categorize addiction score into risk levels."""
    if score >= 7.5:
        return "HIGH ADDICTION RISK", "red", "🔴"
    elif score >= 4.5:
        return "MODERATE RISK", "orange", "🟡"
    return "LOW RISK / HEALTHY", "green", "🟢"


# -----------------------------------------------------------------------------
# Sidebar: MLOps Model Lineage & System Info
# -----------------------------------------------------------------------------
model, preprocessor = load_champion_model()
metrics, validation = load_reports()

with st.sidebar:
    st.title("📱 MLOps Platform")
    st.caption("Automated Champion Model Serving")

    st.subheader("🏆 Active Champion")
    st.info("**Model:** RandomForestRegressor")

    if metrics:
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Test R²", f"{metrics.get('r2_score', 0.0):.3f}")
            st.metric("RMSE", f"{metrics.get('rmse', 0.0):.3f}")
        with col_b:
            st.metric("MAE", f"{metrics.get('mae', 0.0):.3f}")
            st.metric("MSE", f"{metrics.get('mse', 0.0):.3f}")

        with st.expander("Selection Rationale"):
            st.write(
                "Selected via DVC pipeline with 100 decision trees, evaluated on "
                "1,194 unseen test records, achieving 94.05% explained variance (R²)."
            )
    else:
        st.warning("Model metrics not found. Run `dvc repro` to build.")

    st.divider()
    st.subheader("📡 Live Monitoring")
    log_df = load_inference_logs(INFERENCE_LOG_PATH)
    st.metric("Logged Inferences", len(log_df))

    st.caption("Powered by DVC, MLflow, Scikit-Learn & Streamlit")

# -----------------------------------------------------------------------------
# Main Application Tabs
# -----------------------------------------------------------------------------
tab_predict, tab_metrics, tab_monitoring = st.tabs([
    "📱 Live Inference",
    "📊 Model Lineage & Metrics",
    "🛡️ Data Drift & Monitoring",
])

# ------------------ TAB 1: PREDICTION ------------------
with tab_predict:
    st.header("Mobile Phone Addiction Level Prediction")
    st.write(
        "Provide user digital habits and lifestyle metrics below to receive real-time "
        "predictions from the production champion model:"
    )

    if model is None or preprocessor is None:
        st.error(
            f"Champion model not found at `{MODEL_PATH}`. "
            "Please run `dvc repro` to train and generate model artifacts."
        )
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Demographics & Screen Time")
        age = st.slider("Age", min_value=10, max_value=80, value=22, step=1)
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
        daily_usage = st.slider("Daily Usage (Hours)", min_value=0.0, max_value=14.0, value=5.5, step=0.5)
        weekend_usage = st.slider("Weekend Usage (Hours)", min_value=0.0, max_value=16.0, value=7.0, step=0.5)
        screen_before_bed = st.slider("Screen Time Before Bed (Hours)", min_value=0.0, max_value=3.0, value=1.0, step=0.1)

    with col2:
        st.subheader("📲 Apps & Activities")
        checks_per_day = st.slider("Phone Checks Per Day", min_value=10, max_value=150, value=75, step=5)
        apps_used = st.slider("Apps Used Daily", min_value=1, max_value=25, value=12, step=1)
        purpose = st.selectbox("Primary Purpose", options=["Social Media", "Browsing", "Education", "Gaming", "Other"])
        time_social = st.slider("Time on Social Media (Hours)", min_value=0.0, max_value=6.0, value=2.5, step=0.25)
        time_gaming = st.slider("Time on Gaming (Hours)", min_value=0.0, max_value=5.0, value=1.0, step=0.25)
        time_edu = st.slider("Time on Education (Hours)", min_value=0.0, max_value=4.0, value=1.0, step=0.25)

    with col3:
        st.subheader("🧘 Well-being & Lifestyle")
        sleep_hours = st.slider("Sleep Hours Per Night", min_value=3.0, max_value=11.0, value=6.5, step=0.5)
        exercise_hours = st.slider("Exercise Hours Per Day", min_value=0.0, max_value=4.0, value=1.0, step=0.25)
        intellectual_perf = st.slider("Intellectual Performance (50 - 100)", min_value=50, max_value=100, value=75, step=1)
        social_interactions = st.slider("Social Interactions (1 - 10)", min_value=1, max_value=10, value=5, step=1)
        family_comm = st.slider("Family Communication (1 - 10)", min_value=1, max_value=10, value=5, step=1)
        anxiety = st.slider("Anxiety Level (1 - 10)", min_value=1, max_value=10, value=5, step=1)
        depression = st.slider("Depression Level (1 - 10)", min_value=1, max_value=10, value=5, step=1)
        self_esteem = st.slider("Self Esteem (1 - 10)", min_value=1, max_value=10, value=6, step=1)

    input_data = pd.DataFrame([{
        "Age": age,
        "Daily_Usage_Hours": daily_usage,
        "Sleep_Hours": sleep_hours,
        "Interllectual_Performance": intellectual_perf,
        "Social_Interactions": social_interactions,
        "Exercise_Hours": exercise_hours,
        "Anxiety_Level": anxiety,
        "Depression_Level": depression,
        "Self_Esteem": self_esteem,
        "Screen_Time_Before_Bed": screen_before_bed,
        "Phone_Checks_Per_Day": checks_per_day,
        "Apps_Used_Daily": apps_used,
        "Time_on_Social_Media": time_social,
        "Time_on_Gaming": time_gaming,
        "Time_on_Education": time_edu,
        "Family_Communication": family_comm,
        "Weekend_Usage_Hours": weekend_usage,
        "Gender": gender,
        "Phone_Usage_Purpose": purpose,
    }])

    if st.button("Predict Addiction Level", type="primary"):
        transformed_sample = preprocessor.transform(input_data)
        raw_pred = float(model.predict(transformed_sample)[0])
        prediction = max(1.0, min(10.0, round(raw_pred, 2)))

        risk_label, risk_color, risk_icon = get_risk_info(prediction)

        # Log inference to monitoring database
        log_inference(input_data, prediction)

        st.markdown(f"### Predicted Addiction Level: :{risk_color}[**{prediction:.1f} / 10.0**] ({risk_icon} {risk_label})")
        st.progress(prediction / 10.0)

        res_col_a, res_col_b = st.columns([1, 1])

        with res_col_a:
            st.subheader("💡 Digital Wellness Recommendations")
            tips = []
            if screen_before_bed >= 1.0:
                tips.append("🌙 **Cut Nighttime Screen Time:** Discontinue phone usage at least 45 minutes before bedtime.")
            if checks_per_day > 70:
                tips.append("🔕 **Notification Batching:** Scheduled notification check times to reduce impulse pickups.")
            if daily_usage > 6.0:
                tips.append("⏳ **App Limit Timers:** Daily usage exceeds 6 hours. Set timers on social media and gaming apps.")
            if exercise_hours < 1.0:
                tips.append("🏃 **Physical Exercise:** Daily exercise elevates mood and reduces screen dependency.")
            if not tips:
                tips.append("🌟 **Great Balance:** Current digital habits appear healthy. Keep maintaining this routine!")

            for tip in tips:
                st.info(tip)

        with res_col_b:
            st.subheader("📊 Screen Time Breakdown")
            breakdown_df = pd.DataFrame({
                "Category": ["Social Media", "Gaming", "Education", "Other Usage"],
                "Hours": [time_social, time_gaming, time_edu, max(0.0, daily_usage - (time_social + time_gaming + time_edu))]
            }).set_index("Category")
            st.bar_chart(breakdown_df)

# ------------------ TAB 2: MODEL LINEAGE & METRICS ------------------
with tab_metrics:
    st.header("Model Lineage & Multi-Criteria Benchmark")
    st.write("Champion model performance metrics and validation artifacts generated by the DVC pipeline:")

    if metrics:
        st.subheader("📋 Evaluation Metrics")
        metrics_df = pd.DataFrame([metrics]).rename(columns={
            "r2_score": "R² Score",
            "rmse": "RMSE",
            "mae": "MAE",
            "mse": "MSE",
        })
        st.dataframe(metrics_df, use_container_width=True)

    if validation:
        st.subheader("✅ Data Validation Summary")
        st.json(validation)

    st.subheader("📈 Diagnostic Plots")
    p_col1, p_col2 = st.columns(2)
    feat_img = PLOTS_DIR / "feature_importance.png"
    actual_pred_img = PLOTS_DIR / "actual_vs_predicted.png"
    residual_img = PLOTS_DIR / "residuals_distribution.png"

    with p_col1:
        if feat_img.exists():
            st.image(str(feat_img), caption="Feature Importance Ranking")
        if residual_img.exists():
            st.image(str(residual_img), caption="Residual Error Distribution")

    with p_col2:
        if actual_pred_img.exists():
            st.image(str(actual_pred_img), caption="Actual vs Predicted Values")

# ------------------ TAB 3: DATA DRIFT MONITORING ------------------
with tab_monitoring:
    st.header("Production Data Drift & Quality Monitoring")
    st.write(
        "Monitoring production inputs against the reference training dataset using the "
        "**Kolmogorov-Smirnov (KS) two-sample test**."
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        run_check = st.button("Run Drift Check", type="primary")

    log_df = load_inference_logs(INFERENCE_LOG_PATH)
    if not log_df.empty:
        st.write(f"**Total production records logged:** `{len(log_df)}`")

        if run_check or len(log_df) >= 5:
            ref_path = REFERENCE_DATA_PATH if REFERENCE_DATA_PATH.exists() else PROJECT_ROOT / "data" / "processed" / "train.csv"
            if ref_path.exists():
                ref_df = pd.read_csv(ref_path).drop_duplicates()
                report = analyze_drift(ref_df, log_df)

                status_color = "red" if report["dataset_drift_detected"] else "green"
                status_msg = "⚠️ DRIFT DETECTED" if report["dataset_drift_detected"] else "✅ HEALTHY DISTRIBUTION"

                st.markdown(f"#### Status: :{status_color}[{status_msg}]")
                st.write(
                    f"Drifted features: **{report['drifted_features']} / {report['total_features']}** "
                    f"({report['drift_share_pct']}%)"
                )

                drift_rows = []
                for feat, d in report["features"].items():
                    drift_rows.append({
                        "Feature": feat,
                        "Status": "DRIFT ⚠️" if d["drift_detected"] else "HEALTHY ✅",
                        "p-value": round(d["p_value"], 4),
                        "KS Statistic": round(d.get("ks_statistic", d.get("statistic", 0.0)), 4),
                        "Reference Mean": round(d.get("reference_mean", 0.0), 2),
                        "Production Mean": round(d.get("current_mean", 0.0), 2),
                        "Mean Shift": round(d.get("mean_shift", 0.0), 2),
                    })
                st.dataframe(pd.DataFrame(drift_rows), use_container_width=True)

        with st.expander("View Recent Production Logs (Last 10 Requests)"):
            st.dataframe(log_df.tail(10), use_container_width=True)
    else:
        st.info("No production inferences logged yet. Head over to the 'Live Inference' tab to submit predictions.")
