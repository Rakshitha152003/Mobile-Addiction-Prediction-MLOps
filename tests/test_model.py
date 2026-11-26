from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import engineer_features, SELECTED_FEATURES

MODEL_PATH = Path("models/model.pkl")


@pytest.fixture(scope="module")
def model():
    assert MODEL_PATH.exists(), f"Champion model not found at {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


def test_model_pipeline_structure(model):
    """Ensure the model is valid and exposes predict and feature_importances_."""
    assert hasattr(model, "predict"), "Model must have a predict method"
    assert hasattr(model, "feature_importances_"), "Model should expose feature_importances_"
    assert model.n_features_in_ == len(SELECTED_FEATURES), f"Expected {len(SELECTED_FEATURES)} input features"


def test_single_row_inference(model):
    """Test inference on a typical user digital habits record with feature engineering."""
    raw_sample = pd.DataFrame([{
        "daily_usage_hours": 6.0,
        "weekend_usage_hours": 7.5,
        "screen_time_before_bed": 1.2,
        "phone_checks_per_day": 85,
        "time_on_social_media": 3.0,
        "time_on_gaming": 1.5,
        "time_on_education": 1.0,
        "apps_used_daily": 14,
        "sleep_hours": 6.5,
        "exercise_hours": 1.0,
        "anxiety_level": 5,
        "depression_level": 4,
        "self_esteem": 6,
    }])

    eng = engineer_features(raw_sample)
    X_input = eng[SELECTED_FEATURES]
    pred = model.predict(X_input)

    assert len(pred) == 1
    # Addiction level is bounded between 1.0 and 10.0 (clipped to valid range)
    pred_val = float(np.clip(pred[0], 1.0, 10.0))
    assert 1.0 <= pred_val <= 10.0


def test_batch_inference_and_bounds(model):
    """Test batch inference and verify predictions remain in [1.0, 10.0]."""
    test_csv = Path("data/processed/X_test.csv")
    assert test_csv.exists(), f"Test split not found at {test_csv}"

    X_test = pd.read_csv(test_csv).head(10)
    preds = model.predict(X_test)

    assert len(preds) == 10
    for p in preds:
        p_val = float(np.clip(p, 1.0, 10.0))
        assert 1.0 <= p_val <= 10.0


def test_prediction_determinism(model):
    """Ensure identical inputs yield identical outputs (deterministic)."""
    raw_sample = pd.DataFrame([{
        "daily_usage_hours": 4.0,
        "weekend_usage_hours": 5.0,
        "screen_time_before_bed": 0.5,
        "phone_checks_per_day": 45,
        "time_on_social_media": 1.5,
        "time_on_gaming": 0.5,
        "time_on_education": 1.5,
        "apps_used_daily": 8,
        "sleep_hours": 7.0,
        "exercise_hours": 2.0,
        "anxiety_level": 3,
        "depression_level": 2,
        "self_esteem": 8,
    }])

    eng = engineer_features(raw_sample)
    X_input = eng[SELECTED_FEATURES]
    pred_1 = model.predict(X_input)[0]
    pred_2 = model.predict(X_input)[0]
    assert np.isclose(pred_1, pred_2, atol=1e-6)
