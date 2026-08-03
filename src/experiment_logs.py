from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def safe_path_part(value: object, max_len: int = 72) -> str:
    text = str(value).strip().replace("\\", "-").replace("/", "-")
    text = re.sub(r"[^A-Za-z0-9._=-]+", "-", text).strip("-._")
    return (text or "run")[:max_len]


def resolve_logs_root(logs_root: str | Path = "exper_logs") -> Path:
    root = Path(logs_root)
    if not root.is_absolute():
        root = repo_root() / root
    root.mkdir(parents=True, exist_ok=True)
    return root


def create_run_dir(kind: str, parts: list[object] | None = None, logs_root: str | Path = "exper_logs") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    middle = [safe_path_part(kind)]
    if parts:
        middle.extend(safe_path_part(part) for part in parts if part not in (None, ""))
    base_name = f"running_{'_'.join(middle)}_{timestamp}"
    root = resolve_logs_root(logs_root)
    run_dir = root / base_name
    suffix = 2
    while run_dir.exists():
        run_dir = root / f"{base_name}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class RunLogger:
    def __init__(self, run_dir: str | Path, prefix: str):
        self.path = Path(run_dir) / "runtime.log"
        self.prefix = prefix
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        console_line = f"[{self.prefix}] {message}"
        file_line = f"[{now}] {message}"
        print(console_line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(file_line + "\n")
