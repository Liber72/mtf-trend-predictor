from __future__ import annotations

import asyncio
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _reset_app_modules() -> None:
    for module_name in (
        "src.main",
        "src.apps.api.router",
        "src.apps.api.endpoints.system",
        "src.core.dependencies",
        "src.infrastructure.db.session",
        "src.core.settings",
    ):
        sys.modules.pop(module_name, None)


class AppStartupTests(unittest.TestCase):
    def test_settings_can_load_database_config_from_repo_root_dotenv(self) -> None:
        env_path = BACKEND_ROOT.parent / ".env"
        original_contents = env_path.read_text(encoding="utf-8") if env_path.exists() else None

        try:
            env_path.write_text(
                "DB_HOST=127.0.0.1\n"
                "DB_PORT=5432\n"
                "DB_USER=postgres\n"
                "DB_PASSWORD=postgres\n"
                "DB_NAME=autotrader\n",
                encoding="utf-8",
            )

            with patch.dict("os.environ", {}, clear=False):
                for key in (
                    "DATABASE_URL",
                    "DB_HOST",
                    "DB_PORT",
                    "DB_USER",
                    "DB_PASSWORD",
                    "DB_NAME",
                ):
                    os.environ.pop(key, None)

                _reset_app_modules()

                from src.core.settings import get_settings

                settings = get_settings()

            self.assertEqual(
                settings.database_url,
                "postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/autotrader",
            )
        finally:
            if original_contents is None:
                env_path.unlink(missing_ok=True)
            else:
                env_path.write_text(original_contents, encoding="utf-8")

    def test_lifespan_does_not_crash_without_database_settings(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "DATABASE_URL": "",
                "DB_HOST": "",
                "DB_PORT": "",
                "DB_USER": "",
                "DB_PASSWORD": "",
                "DB_NAME": "",
            },
            clear=False,
        ):
            _reset_app_modules()

            from src.main import app, lifespan

            async def run_lifespan_once() -> None:
                async with lifespan(app):
                    return None

            asyncio.run(run_lifespan_once())

    def test_healthcheck_reports_not_configured_without_database_settings(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "DATABASE_URL": "",
                "DB_HOST": "",
                "DB_PORT": "",
                "DB_USER": "",
                "DB_PASSWORD": "",
                "DB_NAME": "",
            },
            clear=False,
        ):
            _reset_app_modules()

            from src.apps.api.endpoints.system import healthcheck
            from src.core.settings import get_settings

            response = asyncio.run(healthcheck(settings=get_settings()))

        self.assertEqual(response.database, "not_configured")


if __name__ == "__main__":
    unittest.main()
