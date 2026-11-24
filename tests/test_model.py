from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.monitor_drift import FEATURE_COLUMNS

MODEL_PATH = Path("models/model.joblib")
PREPROCESSOR_PATH = Path("models/preprocessor.joblib")


@pytest.fixture(scope="module")
def model():
    assert MODEL_PATH.exists(), f"Champion model not found at {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def preprocessor():
    assert PREPROCESSOR_PATH.exists(), f"Preprocessor artifact not found at {PREPROCESSOR_PATH}"
    return joblib.load(PREPROCESSOR_PATH)


def test_model_pipeline_structure(model, preprocessor):
    """Ensure the model and preprocessor structures are valid and have expected interfaces."""
    assert hasattr(model, "predict"), "Model must have a predict method"
    assert hasattr(preprocessor, "transform"), "Preprocessor must have a transform method"
    assert hasattr(model, "feature_importances_"), "Model should expose feature_importances_"


def test_single_row_inference(model, preprocessor):
    """Test inference on a typical user digital habits record."""
    sample = pd.DataFrame([{
        "Age": 22,
        "Daily_Usage_Hours": 5.5,
        "Sleep_Hours": 6.5,
        "Interllectual_Performance": 75,
        "Social_Interactions": 5,
        "Exercise_Hours": 1.0,
        "Anxiety_Level": 5,
        "Depression_Level": 5,
        "Self_Esteem": 6,
        "Screen_Time_Before_Bed": 1.0,
        "Phone_Checks_Per_Day": 75,
        "Apps_Used_Daily": 12,
        "Time_on_Social_Media": 2.5,
        "Time_on_Gaming": 1.0,
        "Time_on_Education": 1.0,
        "Family_Communication": 5,
        "Weekend_Usage_Hours": 7.0,
        "Gender": "Male",
        "Phone_Usage_Purpose": "Social Media",
    }])

    transformed = preprocessor.transform(sample)
    pred = model.predict(transformed)

    assert len(pred) == 1
    # Addiction level is bounded between 1.0 and 10.0
    assert 1.0 <= pred[0] <= 10.0


def test_batch_inference_and_bounds(model, preprocessor):
    """Test batch inference and verify predictions remain in [1.0, 10.0]."""
    test_csv = Path("data/processed/test.csv")
    assert test_csv.exists(), f"Test split not found at {test_csv}"

    test_df = pd.read_csv(test_csv).drop(columns=["Addiction_Level"]).head(10)
    transformed = preprocessor.transform(test_df)
    preds = model.predict(transformed)

    assert len(preds) == 10
    for p in preds:
        assert 1.0 <= p <= 10.0


def test_prediction_determinism(model, preprocessor):
    """Ensure identical inputs yield identical outputs (deterministic)."""
    sample = pd.DataFrame([{
        "Age": 30,
        "Daily_Usage_Hours": 4.0,
        "Sleep_Hours": 7.0,
        "Interllectual_Performance": 80,
        "Social_Interactions": 6,
        "Exercise_Hours": 2.0,
        "Anxiety_Level": 3,
        "Depression_Level": 2,
        "Self_Esteem": 8,
        "Screen_Time_Before_Bed": 0.5,
        "Phone_Checks_Per_Day": 45,
        "Apps_Used_Daily": 8,
        "Time_on_Social_Media": 1.5,
        "Time_on_Gaming": 0.5,
        "Time_on_Education": 1.5,
        "Family_Communication": 7,
        "Weekend_Usage_Hours": 5.0,
        "Gender": "Female",
        "Phone_Usage_Purpose": "Education",
    }])

    trans = preprocessor.transform(sample)
    pred_1 = model.predict(trans)[0]
    pred_2 = model.predict(trans)[0]
    assert np.isclose(pred_1, pred_2, atol=1e-6)
