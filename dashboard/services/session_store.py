from __future__ import annotations
import json
import os
import uuid

from flask import session
import pandas as pd

from config import BaseConfig


# Session id
def get_session_id() -> str:
    """Return the current session_id, creating one if needed"""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]


# Paths
def data_path_for(session_id: str) -> str:
    """Filesystem path for a session's uploaded CSV"""
    return os.path.join(BaseConfig.DATA_FOLDER, f"{session_id}.csv")


def model_path_for(session_id: str) -> str:
    """Filesystem path for a session's trained model bundle"""
    return os.path.join(BaseConfig.MODEL_FOLDER, f"{session_id}.pkl")


def excel_meta_path_for(session_id: str) -> str:
    """Filesystem path for a session's Excel-metadata sidecar."""
    return os.path.join(BaseConfig.DATA_FOLDER, f"{session_id}_excel.json")


# DataFrame persistence
def load_df() -> pd.DataFrame | None:
    """Load the current session's DataFrame or None if none is uplaoded."""
    sid = session.get("session_id")
    if not sid:
        return None
    path = data_path_for(sid)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def save_df(df: pd.DataFrame) -> str:
    """Persist a DataFrame under the current session id, return the path."""
    sid = get_session_id()
    path = data_path_for(sid)
    df.to_csv(path, index=False)
    return path


# Excel metadata


def save_excel_meta(
    session_id: str,
    original_filename: str,
    sheet_names: list[str],
    active_sheet: str,
    excel_path: str,
) -> str:
    """Persist Excel metadata to a sidecar JSON file for this session."""
    meta_path = excel_meta_path_for(session_id)
    payload = {
        "original_filename": original_filename,
        "sheet_names": sheet_names,
        "active_sheet": active_sheet,
        "excel_path": os.path.abspath(excel_path),
    }
    with open(meta_path, "w") as f:
        json.dump(payload, f)
    return meta_path


def load_excel_meta() -> dict | None:
    """Load Excel metadata for the current session or None."""
    sid = session.get("session_id")
    if not sid:
        return None

    meta_path = excel_meta_path_for(sid)
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r") as f:
        return json.load(f)


# Cleanup
def clear_session_files(session_id: str) -> None:
    """Remove all on-disk artifacts belonging to a session."""
    for path in (
        data_path_for(session_id),
        excel_meta_path_for(session_id),
        model_path_for(session_id),
    ):
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
