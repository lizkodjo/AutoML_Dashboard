from __future__ import annotations
import os

from flask import Blueprint, request, session
import numpy as np
import pandas as pd

from dashboard.ml.auto_ml import AutoML
from dashboard.ml.profiler import DataProfiler
from dashboard.ml.visualiser import Visualiser
from dashboard.services.json_utils import jsonify_safe
from dashboard.services.session_store import (
    get_session_id,
    load_df,
    load_excel_meta,
    model_path_for,
    save_df,
    save_excel_meta,
)

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


@bp.route("/switch_sheet", methods=["POST"])
def switch_sheet():
    """Switch to a different sheet in the currently uploaded Excel file."""
    try:
        meta = load_excel_meta()
        if meta is None:
            return jsonify_safe({"error": "No Excel file in current session"}, 400)

        payload = request.get_json(silent=True) or {}
        new_sheet = payload.get("sheet_name")
        if not new_sheet:
            return jsonify_safe({"error": "sheet_name is required."}, 400)

        if new_sheet not in meta["sheet_names"]:
            return jsonify_safe(
                {
                    "error": (
                        f"Sheet '{new_sheet}' not found.  Available: '{meta["sheet_names"]}'"
                    )
                },
                400,
            )

        excel_path = meta["excel_path"]
        if not os.path.exists(excel_path):
            return jsonify_safe({"error": "Original Excel file not found."}, 400)

        df = pd.read_excel(excel_path, sheet_name=new_sheet)
        if len(df) == 0:
            return jsonify_safe({"error": f"Sheet '{new_sheet}' has no data rows"}, 400)

        save_df(df)
        save_excel_meta(
            get_session_id(),
            meta["original_filename"],
            meta["sheet_names"],
            new_sheet,
            excel_path,
        )

        profile = DataProfiler(df).generate_profile()
        charts = Visualiser(df).generate_auto_charts(num_charts=7)

        return jsonify_safe(
            {
                "success": True,
                "filename": meta["original_filename"],
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "profile": profile,
                "charts": charts,
                "preview": df.head(10).to_html(classes="table table-striped", border=0),
                "sheet_info": {
                    "sheet_names": meta["sheet_names"],
                    "active_sheet": new_sheet,
                    "sheet_count": len(meta["sheet_names"]),
                },
            }
        )
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        return jsonify_safe({"error": str(e)})
