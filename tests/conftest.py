from __future__ import annotations
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
    """Sample dataset used across tests."""
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
