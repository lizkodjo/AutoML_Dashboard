from __future__ import annotations
import os
import uuid

from flask import session
import pandas as pd

from config import BaseConfig


def get_session_id() -> str:
    """Return the current session_id, creating one if needed"""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]


def data_path_for(session_id: str) -> str:
    """Filesystem path for a session's uploaded CSV"""
    return os.path.join(BaseConfig.DATA_FOLDER, f"{session_id}.pkl")


def model_path_for(session_id: str) -> str:
    """Filesystem path for a session's trained model bundle"""
    return os.path.join(BaseConfig.MODEL_FOLDER, f"{session_id}.pkl")


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
    """Persist a DataFrame under the current session id; return the path."""
    sid = get_session_id()
    path = data_path_for(sid)
    df.to_csv(path, index=False)
    return path
