from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from dashboard import create_app


@pytest.fixture
def app():
    """Testing-config Flask app."""
    return create_app("testing")


@pytest.fixture
def client(app):
    """Flask test client bound to the testing app."""
    return app.test_client()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Sample tabular dataset used across tests."""
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            "salary": [
                50000,
                60000,
                70000,
                80000,
                90000,
                100000,
                110000,
                120000,
                130000,
                140000,
            ],
            "department": [
                "IT",
                "HR",
                "IT",
                "Finance",
                "HR",
                "IT",
                "Finance",
                "HR",
                "IT",
                "Finance",
            ],
            "experience": [2, 5, 8, 10, 12, 15, 18, 20, 22, 25],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def sample_csv(tmp_path: Path, sample_df: pd.DataFrame) -> Path:
    """Path to the sample dataset written as CSV."""
    path = tmp_path / "test_data.csv"
    sample_df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_xlsx_bytes(sample_df: pd.DataFrame) -> bytes:
    """Sample dataset serialised as a single-sheet .xlsx, as bytes."""
    buf = io.BytesIO()
    sample_df.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


@pytest.fixture
def multi_sheet_xlsx_bytes() -> bytes:
    """Two-sheet Excel workbook, as bytes. Sheet 'Alpha' is first."""
    df_a = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6, 7, 8],
            "y": [10, 20, 30, 40, 50, 60, 70, 80],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    df_b = pd.DataFrame(
        {
            "a": ["p", "q", "r", "s", "t", "u", "v", "w"],
            "b": [100, 200, 300, 400, 500, 600, 700, 800],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_a.to_excel(writer, index=False, sheet_name="Alpha")
        df_b.to_excel(writer, index=False, sheet_name="Beta")
    return buf.getvalue()


@pytest.fixture
def uploaded_client(client, sample_csv: Path):
    """Test client with the sample CSV already uploaded."""
    with open(sample_csv, "rb") as f:
        resp = client.post(
            "/upload",
            data={"file": (f, "test_data.csv")},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 200, resp.data.decode("utf-8")
    return client
