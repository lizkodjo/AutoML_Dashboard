from __future__ import annotations
import json
import os

from config import TestingConfig


def test_upload_endpoint(client, sample_csv):
    """Uploading a CSV returns success with correct row/column counts."""
    with open(sample_csv, "rb") as f:
        response = client.post(
            "/upload",
            data={"file": (f, "test_data.csv")},
            content_type="multipart/form-data",
        )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["success"] is True
    assert data["rows"] == 10
    assert data["columns"] == 5


def test_analyse_endpoint(uploaded_client):
    """AutoML returns a problem type and model results."""
    response = uploaded_client.post("/analyse")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["success"] is True
    assert "problem_type" in data
    assert "model_results" in data


def test_generate_chart_endpoint(uploaded_client):
    """Ad-hoc chart generation returns a Plotly figure."""
    config = {"type": "histogram", "x_column": "age", "title": "Age Distribution"}
    response = uploaded_client.post(
        "/generate_chart",
        data=json.dumps(config),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["success"] is True
    assert "figure" in data


def test_predict_endpoint(uploaded_client):
    """After AutoML, /predict returns a prediction."""
    analyse = uploaded_client.post("/analyse")
    assert analyse.status_code == 200

    assert any(
        f.endswith(".pkl") for f in os.listdir(TestingConfig.MODEL_FOLDER)
    ), f"No .pkl in {TestingConfig.MODEL_FOLDER}"

    predict = uploaded_client.post(
        "/predict",
        json={
            "rows": [{"age": 30, "salary": 60000, "department": "IT", "experience": 5}]
        },
    )
    assert predict.status_code == 200


def test_clear_session(client):
    """Clearing a session succeeds even with no data."""
    response = client.post("/clear_session")
    assert response.status_code == 200
