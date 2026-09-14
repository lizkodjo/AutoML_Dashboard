from __future__ import annotations
import os

from flask import Blueprint, request, session
import pandas as pd

from config import BaseConfig
from dashboard.ml.profiler import DataProfiler
from dashboard.ml.visualiser import Visualiser
from dashboard.services.json_utils import jsonify_safe
from dashboard.services.session_store import get_session_id, save_df

bp = Blueprint("upload", __name__)


def _allowed_file(filename: str) -> bool:
    """Return True if the filename has an allowed extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in BaseConfig.ALLOWED_EXTENSIONS
    )


@bp.route("/upload", methods=["POST"])
def upload_file():
    """Accept a CSV/XLSX upload, profile it, and generate auto-charts."""
    if "file" not in request.files:
        return jsonify_safe({"error": "No file uploaded"}, 400)

    file = request.files["file"]
    if file.filename == "":
        return jsonify_safe({"error": "No file selected"}, 400)

    if not _allowed_file(file.filename):
        return jsonify_safe({"error": "File type not allowed"}, 400)

    try:
        session.clear()
        session_id = get_session_id()

        ext = file.filename.rsplit(".", 1)[1].lower()
        raw_filename = f"{session_id}_raw.{ext}"
        raw_filepath = os.path.join(BaseConfig.UPLOAD_FOLDER, raw_filename)
        file.save(raw_filepath)

        df = pd.read_csv(raw_filepath) if ext == "csv" else pd.read_excel(raw_filepath)

        save_df(df)
        session["filename"] = file.filename

        profile = DataProfiler(df).generate_profile()
        charts = Visualiser(df).generate_auto_charts(num_charts=7)

        return jsonify_safe(
            {
                "success": True,
                "session_id": session_id,
                "filename": file.filename,
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "profile": profile,
                "charts": charts,
                "preview": df.head(10).to_html(classes="table table-striped", border=0),
            }
        )
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        return jsonify_safe({"error": str(e)}, 500)
