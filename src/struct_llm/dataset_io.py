from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any


def load_jsonl_objects(path: str | Path, label: str, *, missing_ok: bool = False) -> tuple[dict[str, Any], ...]:
    data_path = Path(path)
    if missing_ok and not data_path.exists():
        return ()
    records: list[dict[str, Any]] = []
    with data_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid {label} JSONL at line {line_number}: {error}") from error
            if not isinstance(raw_record, dict):
                raise ValueError(f"Invalid {label} JSONL at line {line_number}: expected object")
            records.append(raw_record)
    return tuple(records)


def append_jsonl_object(
    path: str | Path,
    record: dict[str, Any],
    *,
    sort_keys: bool = False,
) -> None:
    data_path = Path(path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("a", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, sort_keys=sort_keys)
        file.write("\n")


def file_sha256(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return ""
    digest = sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()
