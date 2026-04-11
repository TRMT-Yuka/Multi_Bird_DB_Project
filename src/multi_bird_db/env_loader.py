from __future__ import annotations

import os
from pathlib import Path

from .config import get_project_paths


def _parse_env_line(line: str) -> tuple[str, str] | None:
    """Parse one .env line with optional export prefix. / export 付きも許容して 1 行の .env を読む。"""

    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def load_project_env(env_path: Path | None = None) -> None:
    """Load repo-local .env without overriding existing environment variables. / 既存環境変数を優先しつつ repo 直下の .env を読む。"""

    if env_path is None:
        env_path = get_project_paths().root / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)
