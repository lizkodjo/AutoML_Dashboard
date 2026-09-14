from __future__ import annotations
from typing import Any

from flask import jsonify
import numpy as np
import pandas as pd


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas objects to JSON-safe natives."""

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(item) for item in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else val
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if obj is pd.NaT:
        return None
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    return obj


def jsonify_safe(payload: Any, status: int = 200):
    """jsonify() a payload after sanitising it for numpy/pandas types."""
    return jsonify(to_jsonable(payload)), status
