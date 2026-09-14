from __future__ import annotations
import os

from flask import Blueprint, request, send_file, session
import joblib
import numpy as np
import pandas as pd

from dashboard.services.json_utils import jsonify_safe
from dashboard.services.session_store import load_df, model_path_for

bp = Blueprint("ml", __name__)


@bp.route("/predict", methods=["POST"])
def predict():
    """Make predictions using the session's trained model."""
    try:
        sid = session.get("session_id")
        if not sid:
            return jsonify_safe({"error": "No session. Please upload data first."}, 400)

        model_path = model_path_for(sid)
        if not os.path.exists(model_path):
            return jsonify_safe(
                {"error": "No trained model. Please run AutoML first."}, 400
            )

        bundle = joblib.load(model_path)
        model = bundle["model"]
        scaler = bundle["scaler"]
        label_encoders = bundle["label_encoders"]
        feature_cols = bundle["feature_columns"]
        problem_type = bundle["problem_type"]
        target_col = bundle["target_column"]

        payload = request.get_json(silent=True) or {}
        rows = payload.get("rows", [])
        if not rows:
            return jsonify_safe({"error": "No rows provided"}, 400)

        input_df = pd.DataFrame(rows)

        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = np.nan
        input_df = input_df[feature_cols].copy()

        for col, le in label_encoders.items():
            if col == "target" or col not in input_df.columns:
                continue

            known = {str(c) for c in le.classes_}
            fallback = str(le.classes_[0])

            def _encode(v, known=known, fallback=fallback):
                if v is None:
                    return fallback
                try:
                    if pd.isna(v):
                        return fallback
                except (TypeError, ValueError):
                    pass
                s = str(v)
                return s if s in known else fallback

            input_df[col] = le.transform(input_df[col].apply(_encode))

        for col in input_df.select_dtypes(include=["object", "string"]).columns:
            input_df[col] = 0

        input_df = input_df.apply(pd.to_numeric, errors="coerce").fillna(0)

        X_scaled = scaler.transform(input_df)
        predictions = model.predict(X_scaled)

        if problem_type == "classification" and "target" in label_encoders:
            decoded = label_encoders["target"].inverse_transform(
                predictions.astype(int)
            )
            result_predictions = [str(v) for v in decoded]
        else:
            result_predictions = [float(v) for v in predictions]

        probabilities = None
        if problem_type == "classification" and hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_scaled).tolist()

        return jsonify_safe(
            {
                "success": True,
                "problem_type": problem_type,
                "target_column": target_col,
                "predictions": result_predictions,
                "probabilities": probabilities,
            }
        )
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        return jsonify_safe({"error": str(e)}, 500)


@bp.route("/model_info", methods=["GET"])
def model_info():
    """Return metadata about the current session's trained model."""
    try:
        sid = session.get("session_id")
        if not sid:
            return jsonify_safe({"error": "No session"}, 400)

        path = model_path_for(sid)
        if not os.path.exists(path):
            return jsonify_safe({"error": "No trained model"}, 400)

        bundle = joblib.load(path)

        sample_row = None
        df = load_df()
        if df is not None and len(df) > 0:
            sample_row = df.iloc[0].where(pd.notnull(df.iloc[0]), None).to_dict()
            sample_row = {k: (None if pd.isna(v) else v) for k, v in sample_row.items()}

        return jsonify_safe(
            {
                "success": True,
                "feature_columns": bundle["feature_columns"],
                "target_column": bundle["target_column"],
                "problem_type": bundle["problem_type"],
                "best_model_name": bundle.get("best_model_name", "unknown"),
                "best_score": bundle.get("best_score", 0.0),
                "sample_row": sample_row,
            }
        )
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        return jsonify_safe({"error": str(e)})


@bp.route("/download_model", methods=["GET"])
def download_model():
    """Download the trained .pkl bundle."""
    try:
        sid = session.get("session_id")
        if not sid:
            return jsonify_safe({"error": "No session"}, 400)

        path = model_path_for(sid)
        if not os.path.exists(path):
            return jsonify_safe({"error": "No trained model"}, 400)

        return send_file(
            path,
            as_attachment=True,
            download_name=f"automl_model_{sid[:8]}.pkl",
            mimetype="application/octet-stream",
        )
    except Exception as e:  # noqa: BLE001
        return jsonify_safe({"error": str(e)})
