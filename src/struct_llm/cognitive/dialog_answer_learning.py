from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..structure import Query, Structure
from .query_learning import (
    query_from_dict,
    query_signature,
)


DIALOG_ANSWER_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "dialog_answer_examples.jsonl"
DIALOG_ANSWER_MODEL_PATH = Path(__file__).resolve().parents[3] / "data" / "dialog_answer_model.json"
DIALOG_ANSWER_MODEL_SCHEMA = "struct_llm.dialog_answer_model.v1"
VERIFIED_ANSWER_SOURCES = frozenset(
    {
        "curated",
        "human_verified",
        "knowledge",
        "self_model",
        "teacher",
        "training",
    }
)


@dataclass(frozen=True)
class DialogActAnswerTrainingExample:
    question: str
    query: Query
    answer: str
    source: str = "training"
    split: str = "train"


@dataclass(frozen=True)
class CompiledDialogActAnswerPattern:
    query: Query
    answer: str
    support: int = 1


@dataclass(frozen=True)
class CompiledDialogActAnswerModel:
    schema: str
    source_sha256: str
    example_count: int
    patterns: tuple[CompiledDialogActAnswerPattern, ...]


@dataclass(frozen=True)
class DialogActAnswerEvaluationResult:
    total: int
    matched: int

    @property
    def accuracy(self) -> float:
        return self.matched / self.total if self.total else 0.0


@dataclass(frozen=True)
class LearnedDialogActAnswerer:
    examples: tuple[DialogActAnswerTrainingExample, ...] = ()
    patterns: tuple[CompiledDialogActAnswerPattern, ...] = ()

    def __post_init__(self) -> None:
        if self.examples and not self.patterns:
            model = compile_dialog_answer_examples(self.examples)
            object.__setattr__(self, "patterns", model.patterns)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> LearnedDialogActAnswerer:
        return cls((), patterns=compile_dialog_answer_examples(load_dialog_answer_jsonl(path)).patterns)

    @classmethod
    def from_examples(cls, examples: tuple[DialogActAnswerTrainingExample, ...]) -> LearnedDialogActAnswerer:
        return cls((), patterns=compile_dialog_answer_examples(examples).patterns)

    @classmethod
    def from_model(cls, path: str | Path) -> LearnedDialogActAnswerer:
        return cls((), patterns=load_dialog_answer_model(path).patterns)

    def __call__(self, structure: Structure) -> str | None:
        query = structure.query
        if query is None or query.intent != "dialog_act":
            return None
        for pattern in self.patterns:
            if query_signature(pattern.query) == query_signature(query):
                return pattern.answer
        return None


def default_learned_dialog_answerer() -> LearnedDialogActAnswerer:
    if DIALOG_ANSWER_MODEL_PATH.exists():
        return LearnedDialogActAnswerer.from_model(DIALOG_ANSWER_MODEL_PATH)
    if DIALOG_ANSWER_DATA_PATH.exists():
        return LearnedDialogActAnswerer.from_jsonl(DIALOG_ANSWER_DATA_PATH)
    return LearnedDialogActAnswerer()


def compile_dialog_answer_model_from_jsonl(path: str | Path) -> CompiledDialogActAnswerModel:
    data_path = Path(path)
    return compile_dialog_answer_examples(load_dialog_answer_jsonl(data_path), source_sha256=file_sha256(data_path))


def compile_dialog_answer_examples(
    examples: tuple[DialogActAnswerTrainingExample, ...],
    source_sha256: str = "",
) -> CompiledDialogActAnswerModel:
    grouped: dict[tuple[Any, ...], CompiledDialogActAnswerPattern] = {}
    verified_examples = tuple(example for example in examples if example.source in VERIFIED_ANSWER_SOURCES)
    for example in verified_examples:
        key = query_signature(example.query)
        previous = grouped.get(key)
        if previous is None:
            grouped[key] = CompiledDialogActAnswerPattern(query=example.query, answer=example.answer, support=1)
            continue
        grouped[key] = CompiledDialogActAnswerPattern(
            query=previous.query,
            answer=previous.answer,
            support=previous.support + 1,
        )
    return CompiledDialogActAnswerModel(
        schema=DIALOG_ANSWER_MODEL_SCHEMA,
        source_sha256=source_sha256,
        example_count=len(verified_examples),
        patterns=tuple(sorted(grouped.values(), key=lambda pattern: (-pattern.support, pattern.answer))),
    )


def load_dialog_answer_model(path: str | Path) -> CompiledDialogActAnswerModel:
    with Path(path).open("r", encoding="utf-8") as file:
        raw_model = json.load(file)
    if not isinstance(raw_model, dict):
        raise ValueError("Dialog answer model must be a JSON object.")
    return dialog_answer_model_from_dict(raw_model)


def save_dialog_answer_model(model: CompiledDialogActAnswerModel, path: str | Path) -> None:
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_name(f"{model_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(dialog_answer_model_to_dict(model), file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(model_path)


def dialog_answer_model_from_dict(record: dict[str, Any]) -> CompiledDialogActAnswerModel:
    schema = str(record.get("schema") or "").strip()
    if schema != DIALOG_ANSWER_MODEL_SCHEMA:
        raise ValueError(f"Unsupported dialog answer model schema: {schema}")
    raw_patterns = record.get("patterns")
    if not isinstance(raw_patterns, list):
        raise ValueError("Dialog answer model patterns must be a list.")
    return CompiledDialogActAnswerModel(
        schema=schema,
        source_sha256=str(record.get("source_sha256") or ""),
        example_count=int(record.get("example_count") or 0),
        patterns=tuple(dialog_answer_pattern_from_dict(value) for value in raw_patterns),
    )


def dialog_answer_model_to_dict(model: CompiledDialogActAnswerModel) -> dict[str, Any]:
    return {
        "schema": model.schema,
        "source_sha256": model.source_sha256,
        "example_count": model.example_count,
        "pattern_count": len(model.patterns),
        "patterns": [dialog_answer_pattern_to_dict(pattern) for pattern in model.patterns],
    }


def dialog_answer_pattern_from_dict(record: Any) -> CompiledDialogActAnswerPattern:
    if not isinstance(record, dict):
        raise ValueError("Dialog answer model pattern entries must be objects.")
    raw_query = record.get("query")
    if not isinstance(raw_query, dict):
        raise ValueError("Dialog answer model pattern query must be an object.")
    answer = str(record.get("answer") or "").strip()
    if not answer:
        raise ValueError("Dialog answer model pattern requires answer.")
    return CompiledDialogActAnswerPattern(
        query=query_from_dict(raw_query, "Dialog answer model pattern"),
        answer=answer,
        support=int(record.get("support") or 1),
    )


def dialog_answer_pattern_to_dict(pattern: CompiledDialogActAnswerPattern) -> dict[str, Any]:
    return {
        "query": {
            "intent": pattern.query.intent,
            "target": pattern.query.target,
            "qualifiers": list(pattern.query.qualifiers),
            "subqueries": [],
        },
        "answer": pattern.answer,
        "support": pattern.support,
    }


def load_dialog_answer_jsonl(path: str | Path) -> tuple[DialogActAnswerTrainingExample, ...]:
    examples: list[DialogActAnswerTrainingExample] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid dialog answer JSONL at line {line_number}: {error}") from error
            if not isinstance(raw_record, dict):
                raise ValueError(f"Invalid dialog answer JSONL at line {line_number}: expected object")
            examples.append(dialog_answer_example_from_dict(raw_record, line_number=line_number))
    return tuple(examples)


def append_dialog_answer_record(path: str | Path, record: dict[str, Any]) -> DialogActAnswerTrainingExample:
    example = dialog_answer_example_from_dict(record)
    data_path = Path(path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("a", encoding="utf-8") as file:
        json.dump(dialog_answer_example_to_record(example), file, ensure_ascii=False)
        file.write("\n")
    return example


def save_manual_dialog_answer_feedback(
    question: str,
    query: Query,
    answer: str,
    data_path: str | Path = DIALOG_ANSWER_DATA_PATH,
    model_path: str | Path = DIALOG_ANSWER_MODEL_PATH,
    *,
    source: str = "candidate",
) -> tuple[DialogActAnswerTrainingExample, CompiledDialogActAnswerModel]:
    append_dialog_answer_record(
        data_path,
        {
            "question": question,
            "query": {
                "intent": query.intent,
                "target": query.target,
                "qualifiers": list(query.qualifiers),
                "subqueries": [],
            },
            "answer": answer,
            "source": source,
            "split": "train",
        },
    )
    model = compile_dialog_answer_model_from_jsonl(data_path)
    save_dialog_answer_model(model, model_path)
    return load_dialog_answer_jsonl(data_path)[-1], model


def build_dialog_answer_record(
    question: str,
    intent: str,
    target: str,
    answer: str,
    *,
    qualifiers: tuple[str, ...] = (),
    source: str = "human_feedback",
    split: str = "train",
) -> dict[str, Any]:
    return {
        "question": question.strip(),
        "query": {
            "intent": intent.strip(),
            "target": target.strip(),
            "qualifiers": [value.strip() for value in qualifiers if value.strip()],
        },
        "answer": answer.strip(),
        "source": source.strip() or "human_feedback",
        "split": split.strip() or "train",
    }


def dialog_answer_example_to_record(example: DialogActAnswerTrainingExample) -> dict[str, Any]:
    return {
        "question": example.question,
        "query": {
            "intent": example.query.intent,
            "target": example.query.target,
            "qualifiers": list(example.query.qualifiers),
            "subqueries": [],
        },
        "answer": example.answer,
        "source": example.source,
        "split": example.split,
    }


def dialog_answer_example_from_dict(record: dict[str, Any], *, line_number: int | None = None) -> DialogActAnswerTrainingExample:
    prefix = f"Dialog answer example at line {line_number}" if line_number is not None else "Dialog answer example"
    question = str(record.get("question") or record.get("text") or "").strip()
    if not question:
        raise ValueError(f"{prefix} requires a question or text field.")
    raw_query = record.get("query")
    if not isinstance(raw_query, dict):
        raise ValueError(f"{prefix} query must be an object.")
    query = query_from_dict(raw_query, prefix)
    answer = str(record.get("answer") or record.get("response") or "").strip()
    if not answer:
        raise ValueError(f"{prefix} requires an answer.")
    return DialogActAnswerTrainingExample(
        question=question,
        query=query,
        answer=answer,
        source=str(record.get("source") or "training").strip() or "training",
        split=str(record.get("split") or "train").strip() or "train",
    )


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()
