from __future__ import annotations

from flask import Blueprint, request

from dashboard.services.json_utils import jsonify_safe
from dashboard.services.session_store import load_df

bp = Blueprint("export", __name__)


@bp.route("/export_results", methods=["POST"])
def export_results():
    """Export the session's DataFrame as CSV or JSON."""
    try:
        df = load_df()
        if df is None:
            return jsonify_safe({"error": "No data to export"}, 400)

        export_type = (request.json or {}).get("type", "json")
        if export_type == "csv":
            return df.to_csv(index=False)
        return df.to_json(orient="records")
    except Exception as e:  # noqa: BLE001
        return jsonify_safe({"error": str(e)}, 500)
