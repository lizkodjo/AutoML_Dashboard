from __future__ import annotations

from flask import Blueprint, request, session
import numpy as np

from dashboard.ml.auto_ml import AutoML
from dashboard.ml.visualiser import Visualiser
from dashboard.services.json_utils import jsonify_safe
from dashboard.services.session_store import load_df, model_path_for

bp = Blueprint("analysis", __name__)


@bp.route("/analyse", methods=["POST"])
def analyse_data():
    """Train models, save the best bundle, and return the performance report."""
    try:
        df = load_df()
        if df is None:
            return jsonify_safe(
                {"error": "No data loaded. Please upload a file first."}, 400
            )

        automl = AutoML(df)
        problem_type, target_column = automl.identify_problem_type()
        if not problem_type or not target_column:
            return jsonify_safe(
                {"error": "Could not identify a suitable target column"}, 400
            )

        automl.prepare_data()
        results = automl.train_models()
        performance_report = automl.get_performance_report()

        sid = session.get("session_id")
        if sid:
            automl.save_model(model_path_for(sid))

        return jsonify_safe(
            {
                "success": True,
                "problem_type": problem_type,
                "target_column": target_column,
                "model_results": results,
                "performance_report": performance_report,
            }
        )
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        return jsonify_safe({"error": str(e)}, 500)


@bp.route("/generate_chart", methods=["POST"])
def generate_custom_chart():
    """Generate a single Plotly figure from an ad-hoc config."""
    try:
        df = load_df()
        if df is None:
            return jsonify_safe({"error": "No data loaded"}, 400)

        figure = Visualiser(df).generate_chart(request.json)
        return jsonify_safe({"success": True, "figure": figure})
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        return jsonify_safe({"error": str(e)}, 500)


@bp.route("/suggest_features", methods=["POST"])
def suggest_features():
    """Heuristically suggest feature-engineering ideas."""
    try:
        df = load_df()
        if df is None:
            return jsonify_safe({"error": "No data loaded"}, 400)

        suggestions: list[dict] = []
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                suggestions.append(
                    {
                        "type": "date",
                        "column": str(col),
                        "suggestion": f"Extract year, month, day, weekday from {col}",
                    }
                )
        for col in df.columns:
            if df[col].dtype == "object" and df[col].astype(str).str.len().mean() > 50:
                suggestions.append(
                    {
                        "type": "text",
                        "column": str(col),
                        "suggestion": (
                            f"Extract length, word count, or keywords from {col}"
                        ),
                    }
                )
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].nunique() > 20:
                suggestions.append(
                    {
                        "type": "binning",
                        "column": str(col),
                        "suggestion": f"Bin {col} into categories",
                    }
                )

        return jsonify_safe({"success": True, "suggestions": suggestions[:10]})
    except Exception as e:  # noqa: BLE001
        return jsonify_safe({"error": str(e)}, 500)
