"""Runtime settings for the application."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import os

from src.core.constants import ARTIFACTS_DIR, DATA_DIR, LOGS_DIR, MODELS_DIR, APP_NAME, APP_VERSION, API_PREFIX
from src.utils.paths import ensure_directory, project_root

PROJECT_ROOT = project_root()


def _load_env_files() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    for env_path in (PROJECT_ROOT / ".env", PROJECT_ROOT / "backend" / ".env"):
        if env_path.exists():
            load_dotenv(env_path, override=False)


_load_env_files()


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _database_url_env() -> str | None:
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url

    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    name = os.getenv("DB_NAME")

    if not all([host, user, password, name]):
        return None

    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{name}"


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", APP_NAME)
    app_version: str = os.getenv("APP_VERSION", APP_VERSION)
    api_prefix: str = os.getenv("API_PREFIX", API_PREFIX)
    environment: str = os.getenv("APP_ENV", "development")
    debug: bool = _bool_env("APP_DEBUG", True)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    database_url: str | None = field(default_factory=_database_url_env)
    database_echo: bool = _bool_env("ECHO", False)
    database_pool_size: int = _int_env("POOL_SIZE", 10)
    database_max_overflow: int = _int_env("MAX_OVERFLOW", 5)
    database_pool_recycle: int = _int_env("POOL_RECYCLE", 1800)

    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / DATA_DIR)
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / MODELS_DIR)
    artifacts_dir: Path = field(default_factory=lambda: PROJECT_ROOT / ARTIFACTS_DIR)
    logs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / LOGS_DIR)

    mt5_terminal_path: str = os.getenv("MT5_TERMINAL_PATH", "")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def ensure_runtime_directories(config: Settings | None = None) -> None:
    config = config or get_settings()
    for path in (
        config.data_dir,
        config.models_dir,
        config.artifacts_dir,
        config.logs_dir,
    ):
        ensure_directory(path)
