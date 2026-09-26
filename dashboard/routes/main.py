from __future__ import annotations

from flask import Blueprint, jsonify, render_template, session

from dashboard.services.session_store import clear_session_files

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
        clear_session_files(sid)
    session.clear()
    return jsonify({"success": True})
