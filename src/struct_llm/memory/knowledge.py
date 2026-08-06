from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..comprehension.query import query_from_dict, query_signature, query_to_dict
from ..structure import Query, Structure


DATA_DIR = Path(__file__).resolve().parents[3] / "data"
MEMORY_KNOWLEDGE_DATA_PATH = DATA_DIR / "memory_knowledge_examples.jsonl"
MEMORY_KNOWLEDGE_MODEL_PATH = DATA_DIR / "memory_knowledge_model.json"
MEMORY_KNOWLEDGE_ENTRY_SCHEMA = "struct_llm.memory_knowledge_entry.v1"
MEMORY_KNOWLEDGE_MODEL_SCHEMA = "struct_llm.memory_knowledge_model.v1"
VERIFIED_KNOWLEDGE_SOURCES = frozenset(
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
class MemoryKnowledgeEntry:
    question: str
    query: Query
    answer: str
    source: str = "training"
    split: str = "train"


@dataclass(frozen=True)
class CompiledMemoryKnowledgePattern:
    query: Query
    answer: str
    support: int = 1


@dataclass(frozen=True)
class CompiledMemoryKnowledgeModel:
    schema: str
    source_sha256: str
    example_count: int
    patterns: tuple[CompiledMemoryKnowledgePattern, ...]


@dataclass(frozen=True)
class KnowledgeWriteResult:
    data_path: Path
    model_path: Path
    example_count: int
    pattern_count: int


@dataclass(frozen=True)
class LearnedMemoryKnowledgeAnswerer:
    examples: tuple[MemoryKnowledgeEntry, ...] = ()
    patterns: tuple[CompiledMemoryKnowledgePattern, ...] = ()

    def __post_init__(self) -> None:
        if self.examples and not self.patterns:
            model = compile_memory_knowledge_examples(self.examples)
            object.__setattr__(self, "patterns", model.patterns)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> LearnedMemoryKnowledgeAnswerer:
        return cls((), patterns=compile_memory_knowledge_examples(load_memory_knowledge_jsonl(path)).patterns)

    @classmethod
    def from_examples(cls, examples: tuple[MemoryKnowledgeEntry, ...]) -> LearnedMemoryKnowledgeAnswerer:
        return cls((), patterns=compile_memory_knowledge_examples(examples).patterns)

    @classmethod
    def from_model(cls, path: str | Path) -> LearnedMemoryKnowledgeAnswerer:
        return cls((), patterns=load_memory_knowledge_model(path).patterns)

    def __call__(self, structure: Structure) -> str | None:
        query = structure.query
        if query is None:
            return None
        for pattern in self.patterns:
            if query_signature(pattern.query) == query_signature(query):
                return pattern.answer
        return None


def default_learned_memory_knowledge_answerer(path: str | Path = MEMORY_KNOWLEDGE_MODEL_PATH) -> LearnedMemoryKnowledgeAnswerer:
    model_path = Path(path)
    if model_path.exists():
        return LearnedMemoryKnowledgeAnswerer.from_model(model_path)
    data_path = MEMORY_KNOWLEDGE_DATA_PATH
    if data_path.exists():
        return LearnedMemoryKnowledgeAnswerer.from_jsonl(data_path)
    return LearnedMemoryKnowledgeAnswerer()


def build_memory_knowledge_record(
    question: str,
    query: Query,
    answer: str,
    *,
    source: str = "human_feedback",
    split: str = "train",
) -> dict[str, Any]:
    return {
        "schema": MEMORY_KNOWLEDGE_ENTRY_SCHEMA,
        "question": question.strip(),
        "query": query_to_dict(query),
        "answer": answer.strip(),
        "source": source.strip() or "human_feedback",
        "split": split.strip() or "train",
    }


def memory_knowledge_entry_to_record(entry: MemoryKnowledgeEntry) -> dict[str, Any]:
    return {
        "schema": MEMORY_KNOWLEDGE_ENTRY_SCHEMA,
        "question": entry.question,
        "query": query_to_dict(entry.query),
        "answer": entry.answer,
        "source": entry.source,
        "split": entry.split,
    }


def memory_knowledge_entry_from_dict(record: dict[str, Any], *, line_number: int | None = None) -> MemoryKnowledgeEntry:
    prefix = f"Memory knowledge entry at line {line_number}" if line_number is not None else "Memory knowledge entry"
    schema = str(record.get("schema") or MEMORY_KNOWLEDGE_ENTRY_SCHEMA).strip()
    if schema != MEMORY_KNOWLEDGE_ENTRY_SCHEMA:
        raise ValueError(f"{prefix} has unsupported schema: {schema}")
    question = str(record.get("question") or record.get("text") or "").strip()
    if not question:
        raise ValueError(f"{prefix} requires a question or text field.")
    raw_query = record.get("query")
    if not isinstance(raw_query, dict):
        raise ValueError(f"{prefix} query must be an object.")
    answer = str(record.get("answer") or record.get("response") or "").strip()
    if not answer:
        raise ValueError(f"{prefix} requires an answer.")
    return MemoryKnowledgeEntry(
        question=question,
        query=query_from_dict(raw_query, prefix),
        answer=answer,
        source=str(record.get("source") or "training").strip() or "training",
        split=str(record.get("split") or "train").strip() or "train",
    )


def load_memory_knowledge_jsonl(path: str | Path) -> tuple[MemoryKnowledgeEntry, ...]:
    data_path = Path(path)
    if not data_path.exists():
        return ()
    entries: list[MemoryKnowledgeEntry] = []
    with data_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid memory knowledge JSONL at line {line_number}: {error}") from error
            if not isinstance(raw_record, dict):
                raise ValueError(f"Invalid memory knowledge JSONL at line {line_number}: expected object")
            entries.append(memory_knowledge_entry_from_dict(raw_record, line_number=line_number))
    return tuple(entries)


def append_memory_knowledge_record(path: str | Path, record: dict[str, Any]) -> MemoryKnowledgeEntry:
    entry = memory_knowledge_entry_from_dict(record)
    data_path = Path(path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("a", encoding="utf-8") as file:
        json.dump(memory_knowledge_entry_to_record(entry), file, ensure_ascii=False)
        file.write("\n")
    return entry


def append_memory_knowledge_entry(path: str | Path, entry: MemoryKnowledgeEntry) -> MemoryKnowledgeEntry:
    return append_memory_knowledge_record(path, memory_knowledge_entry_to_record(entry))


def compile_memory_knowledge_model_from_jsonl(path: str | Path = MEMORY_KNOWLEDGE_DATA_PATH) -> CompiledMemoryKnowledgeModel:
    data_path = Path(path)
    return compile_memory_knowledge_examples(load_memory_knowledge_jsonl(data_path), source_sha256=file_sha256(data_path))


def compile_memory_knowledge_examples(
    examples: tuple[MemoryKnowledgeEntry, ...],
    source_sha256: str = "",
) -> CompiledMemoryKnowledgeModel:
    grouped: dict[tuple[Any, ...], CompiledMemoryKnowledgePattern] = {}
    verified_examples = tuple(example for example in examples if example.source in VERIFIED_KNOWLEDGE_SOURCES)
    for example in verified_examples:
        key = query_signature(example.query)
        previous = grouped.get(key)
        if previous is None:
            grouped[key] = CompiledMemoryKnowledgePattern(query=example.query, answer=example.answer, support=1)
            continue
        grouped[key] = CompiledMemoryKnowledgePattern(
            query=previous.query,
            answer=previous.answer,
            support=previous.support + 1,
        )
    return CompiledMemoryKnowledgeModel(
        schema=MEMORY_KNOWLEDGE_MODEL_SCHEMA,
        source_sha256=source_sha256,
        example_count=len(verified_examples),
        patterns=tuple(sorted(grouped.values(), key=lambda pattern: (-pattern.support, pattern.answer))),
    )


def load_memory_knowledge_model(path: str | Path) -> CompiledMemoryKnowledgeModel:
    with Path(path).open("r", encoding="utf-8") as file:
        raw_model = json.load(file)
    if not isinstance(raw_model, dict):
        raise ValueError("Memory knowledge model must be a JSON object.")
    return memory_knowledge_model_from_dict(raw_model)


def save_memory_knowledge_model(model: CompiledMemoryKnowledgeModel, path: str | Path) -> None:
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_name(f"{model_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(memory_knowledge_model_to_dict(model), file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(model_path)


def memory_knowledge_model_from_dict(record: dict[str, Any]) -> CompiledMemoryKnowledgeModel:
    schema = str(record.get("schema") or "").strip()
    if schema != MEMORY_KNOWLEDGE_MODEL_SCHEMA:
        raise ValueError(f"Unsupported memory knowledge model schema: {schema}")
    raw_patterns = record.get("patterns")
    if not isinstance(raw_patterns, list):
        raise ValueError("Memory knowledge model patterns must be a list.")
    return CompiledMemoryKnowledgeModel(
        schema=schema,
        source_sha256=str(record.get("source_sha256") or ""),
        example_count=int(record.get("example_count") or 0),
        patterns=tuple(memory_knowledge_pattern_from_dict(value) for value in raw_patterns),
    )


def memory_knowledge_model_to_dict(model: CompiledMemoryKnowledgeModel) -> dict[str, Any]:
    return {
        "schema": model.schema,
        "source_sha256": model.source_sha256,
        "example_count": model.example_count,
        "pattern_count": len(model.patterns),
        "patterns": [memory_knowledge_pattern_to_dict(pattern) for pattern in model.patterns],
    }


def memory_knowledge_pattern_from_dict(record: Any) -> CompiledMemoryKnowledgePattern:
    if not isinstance(record, dict):
        raise ValueError("Memory knowledge model pattern entries must be objects.")
    raw_query = record.get("query")
    if not isinstance(raw_query, dict):
        raise ValueError("Memory knowledge model pattern query must be an object.")
    answer = str(record.get("answer") or "").strip()
    if not answer:
        raise ValueError("Memory knowledge model pattern requires answer.")
    return CompiledMemoryKnowledgePattern(
        query=query_from_dict(raw_query, "Memory knowledge model pattern"),
        answer=answer,
        support=int(record.get("support") or 1),
    )


def memory_knowledge_pattern_to_dict(pattern: CompiledMemoryKnowledgePattern) -> dict[str, Any]:
    return {
        "query": query_to_dict(pattern.query),
        "answer": pattern.answer,
        "support": pattern.support,
    }


def save_memory_knowledge_feedback(
    question: str,
    query: Query,
    answer: str,
    data_path: str | Path = MEMORY_KNOWLEDGE_DATA_PATH,
    model_path: str | Path = MEMORY_KNOWLEDGE_MODEL_PATH,
    *,
    source: str = "human_feedback",
) -> tuple[MemoryKnowledgeEntry, CompiledMemoryKnowledgeModel]:
    append_memory_knowledge_record(
        data_path,
        {
            "question": question,
            "query": query_to_dict(query),
            "answer": answer,
            "source": source,
            "split": "train",
        },
    )
    model = compile_memory_knowledge_model_from_jsonl(data_path)
    save_memory_knowledge_model(model, model_path)
    return load_memory_knowledge_jsonl(data_path)[-1], model


def file_sha256(path: Path) -> str:
    import hashlib

    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()
