from __future__ import annotations

import io
import json
import os

from config import TestingConfig

# CSV


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


# Excel
def test_excel_upload(client, sample_xlsx_bytes):
    """Uploading a single-sheet .xlsx returns success with correct counts."""
    response = client.post(
        "/upload",
        data={"file": (io.BytesIO(sample_xlsx_bytes), "test_data.xlsx")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200, response.data.decode("utf-8")
    data = json.loads(response.data)
    assert data["success"] is True
    assert data["rows"] == 10
    assert data["columns"] == 5
    # Single-sheet Excel: sheet_info is present with one sheet named "Sheet1"
    assert data["sheet_info"]["sheet_count"] == 1
    assert data["sheet_info"]["active_sheet"] == "Sheet1"


def test_excel_multi_sheet_switch(client, multi_sheet_xlsx_bytes):
    """Upload a multi-sheet .xlsx and switch between sheets."""
    resp = client.post(
        "/upload",
        data={"file": (io.BytesIO(multi_sheet_xlsx_bytes), "multi.xlsx")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200, resp.data.decode("utf-8")
    data = json.loads(resp.data)
    assert data["sheet_info"]["active_sheet"] == "Alpha"
    assert data["sheet_info"]["sheet_count"] == 2
    assert data["sheet_info"]["sheet_names"] == ["Alpha", "Beta"]

    # Switch to Beta
    resp = client.post("/switch_sheet", json={"sheet_name": "Beta"})
    assert resp.status_code == 200, resp.data.decode("utf-8")
    data = json.loads(resp.data)
    assert data["sheet_info"]["active_sheet"] == "Beta"
    assert data["columns"] == 3  # a, b, target
    assert data["rows"] == 8

    # Verify downstream (analyse) still works on the new sheet
    resp = client.post("/analyse")
    assert resp.status_code == 200
