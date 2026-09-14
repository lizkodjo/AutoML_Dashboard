from __future__ import annotations

from flask import Flask


def register_blueprints(app: Flask) -> None:
    """Register all route blueprints on the app."""
    from dashboard.routes import analysis, export, main, ml, upload

    app.register_blueprint(main.bp)
    app.register_blueprint(upload.bp)
    app.register_blueprint(analysis.bp)
    app.register_blueprint(ml.bp)
    app.register_blueprint(export.bp)
