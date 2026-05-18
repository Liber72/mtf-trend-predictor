"""Logging bootstrap for the application."""

from __future__ import annotations

import logging
from logging.config import dictConfig

from src.core.settings import get_settings


def configure_logging(level: str | None = None) -> None:
    config = get_settings()
    log_level = level or config.log_level

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                }
            },
            "root": {
                "level": log_level,
                "handlers": ["console"],
            },
        }
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
