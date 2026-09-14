from __future__ import annotations
import os

from flask import Blueprint, jsonify, render_template, session

from dashboard.services.session_store import data_path_for

bp = Blueprint("main", __name__)


@bp.route("/")
def index():
    """Render the dashboard"""
    return render_template("index.html")


@bp.route("/clear_session", methods=["POST"])
def clear_session():
    """Delete the session's uploaded data and clear the session cookie"""
    sid = session.get("session_id")
    if sid:
        path = data_path_for(sid)
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    session.clear()
    return jsonify({"success": True})
