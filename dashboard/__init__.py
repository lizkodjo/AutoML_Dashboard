from __future__ import annotations
import os

from flask import Flask

from config import BaseConfig, DevelopmentConfig, ProductionConfig, TestingConfig

_CONFIGS: dict[str, type[BaseConfig]] = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}


def create_app(config_name: str = "development") -> Flask:
    """Create and configure a Flask app instance."""
    app = Flask(__name__)
    cfg = _CONFIGS.get(config_name, DevelopmentConfig)

    app.config.from_object(cfg)
    app.secret_key = cfg.resolve_secret_key()

    # Ensure upload dirs exist
    for folder in (cfg.UPLOAD_FOLDER, cfg.DATA_FOLDER, cfg.MODEL_FOLDER):
        os.makedirs(folder, exist_ok=True)

    from dashboard.routes import register_blueprints

    register_blueprints(app)
    return app
