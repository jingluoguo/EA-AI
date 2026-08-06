from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


UNRECOGNIZED_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "unrecognized_examples.jsonl"
UNRECOGNIZED_EXAMPLE_SCHEMA = "struct_llm.unrecognized_example.v1"


@dataclass(frozen=True)
class UnrecognizedExample:
    text: str
    confidence: float = 0.0
    reason: str = "low_confidence"
    status: str = "pending"
    source: str = "runtime"


def build_unrecognized_record(
    text: str,
    *,
    confidence: float = 0.0,
    reason: str = "low_confidence",
    status: str = "pending",
    source: str = "runtime",
) -> dict[str, Any]:
    return {
        "schema": UNRECOGNIZED_EXAMPLE_SCHEMA,
        "text": text.strip(),
        "confidence": max(0.0, min(1.0, float(confidence))),
        "reason": reason.strip() or "low_confidence",
        "status": status.strip() or "pending",
        "source": source.strip() or "runtime",
    }


def unrecognized_example_from_dict(
    record: dict[str, Any],
    *,
    line_number: int | None = None,
) -> UnrecognizedExample:
    prefix = f"Unrecognized example at line {line_number}" if line_number is not None else "Unrecognized example"
    schema = str(record.get("schema") or UNRECOGNIZED_EXAMPLE_SCHEMA).strip()
    if schema != UNRECOGNIZED_EXAMPLE_SCHEMA:
        raise ValueError(f"{prefix} has unsupported schema: {schema}")
    text = str(record.get("text") or "").strip()
    if not text:
        raise ValueError(f"{prefix} requires text.")
    confidence = float(record.get("confidence") or 0.0)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{prefix} confidence must be between 0 and 1.")
    return UnrecognizedExample(
        text=text,
        confidence=confidence,
        reason=str(record.get("reason") or "low_confidence").strip() or "low_confidence",
        status=str(record.get("status") or "pending").strip() or "pending",
        source=str(record.get("source") or "runtime").strip() or "runtime",
    )


def append_unrecognized_example(
    path: str | Path,
    example: UnrecognizedExample,
) -> UnrecognizedExample:
    data_path = Path(path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("a", encoding="utf-8") as file:
        json.dump(
            build_unrecognized_record(
                example.text,
                confidence=example.confidence,
                reason=example.reason,
                status=example.status,
                source=example.source,
            ),
            file,
            ensure_ascii=False,
        )
        file.write("\n")
    return example


def load_unrecognized_jsonl(path: str | Path) -> tuple[UnrecognizedExample, ...]:
    examples: list[UnrecognizedExample] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid unrecognized JSONL at line {line_number}: {error}") from error
            if not isinstance(raw_record, dict):
                raise ValueError(f"Invalid unrecognized JSONL at line {line_number}: expected object")
            examples.append(unrecognized_example_from_dict(raw_record, line_number=line_number))
    return tuple(examples)
