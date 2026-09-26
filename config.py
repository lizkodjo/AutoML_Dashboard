from __future__ import annotations
import os
import warnings

# Anchor all paths to the project root
_BASE_DIR: str = os.path.abspath(os.path.dirname(__file__))


class BaseConfig:
    """Shared configuration"""

    SECRET_KEY: str | None = os.getenv("SECRET_KEY")
    UPLOAD_FOLDER: str = os.path.join(_BASE_DIR, "uploads")
    DATA_FOLDER: str = os.path.join(UPLOAD_FOLDER, "data")
    MODEL_FOLDER: str = os.path.join(UPLOAD_FOLDER, "models")
    ALLOWED_EXTENSIONS: frozenset[str] = frozenset({"csv", "xlsx"})
    MAX_CONTENT_LENGTH: int = 50 * 1024 * 1024

    @classmethod
    def resolve_secret_key(cls) -> str:
        """Return SECRET_KEY, falling back to dev default with a warning"""
        if cls.SECRET_KEY:
            return cls.SECRET_KEY
        warnings.warn(
            "SECRET_KEY not set - using insecure dev default. Set SECRET_KEY in your .env before deploying.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "dev-secret-change-me"


class DevelopmentConfig(BaseConfig):
    DEBUG = True


class TestingConfig(BaseConfig):
    TESTING = True
    SECRET_KEY = "test-secret"


class ProductionConfig(BaseConfig):
    DEBUG = False
