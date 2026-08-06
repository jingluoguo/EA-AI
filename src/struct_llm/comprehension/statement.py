from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from ..structure import Entity, Frame, Role
from ..capabilities import StatementParseResult
from ..perception.normalizer import (
    normalize_container_slot,
    normalize_containment_expression,
    normalize_container_surface,
    normalize_slot_value,
    normalize_take_out_expression,
)


STATEMENT_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "statement_examples.jsonl"
STATEMENT_MODEL_PATH = Path(__file__).resolve().parents[3] / "data" / "statement_model.json"
STATEMENT_MODEL_SCHEMA = "struct_llm.statement_model.v1"


@dataclass(frozen=True)
class EntitySlot:
    role: str
    name: str


@dataclass(frozen=True)
class FrameTemplate:
    frame_type: str
    roles: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class StatementTrainingExample:
    sentence: str
    sentence_template: str
    entities: tuple[EntitySlot, ...]
    frames: tuple[FrameTemplate, ...]
    source: str = "training"
    split: str = "train"


@dataclass(frozen=True)
class CompiledStatementPattern:
    sentence_template: str
    entities: tuple[EntitySlot, ...]
    frames: tuple[FrameTemplate, ...]
    feature_units: tuple[str, ...]
    support: int = 1


@dataclass(frozen=True)
class CompiledStatementModel:
    schema: str
    source_sha256: str
    example_count: int
    patterns: tuple[CompiledStatementPattern, ...]


@dataclass(frozen=True)
class StatementEvaluationResult:
    total: int
    matched: int

    @property
    def accuracy(self) -> float:
        return self.matched / self.total if self.total else 0.0


@dataclass(frozen=True)
class LearnedStatementParser:
    examples: tuple[StatementTrainingExample, ...] = ()
    min_score: float = 0.58
    patterns: tuple[CompiledStatementPattern, ...] = ()

    def __post_init__(self) -> None:
        if self.examples and not self.patterns:
            model = compile_statement_examples(self.examples)
            object.__setattr__(self, "patterns", model.patterns)

    @classmethod
    def from_jsonl(cls, path: str | Path, min_score: float = 0.58) -> LearnedStatementParser:
        return cls.from_examples(load_statement_jsonl(path), min_score=min_score)

    @classmethod
    def from_examples(
        cls,
        examples: tuple[StatementTrainingExample, ...],
        min_score: float = 0.58,
    ) -> LearnedStatementParser:
        return cls((), min_score=min_score, patterns=compile_statement_examples(examples).patterns)

    @classmethod
    def from_model(cls, path: str | Path, min_score: float = 0.58) -> LearnedStatementParser:
        return cls((), min_score=min_score, patterns=load_statement_model(path).patterns)

    def __call__(self, sentence: str) -> StatementParseResult | None:
        if not self.patterns:
            return None
        normalized = normalize_statement_text(sentence)
        candidates: list[tuple[int, float, int, CompiledStatementPattern, dict[str, str]]] = []
        for pattern in self.patterns:
            slots = extract_slots(pattern.sentence_template, normalized)
            if slots is None:
                continue
            candidates.append(
                (
                    len(remove_slots(pattern.sentence_template)),
                    max(statement_pattern_score(pattern, normalized), 0.95),
                    pattern.support,
                    pattern,
                    slots,
                )
            )
        for _, score, _, pattern, slots in sorted(candidates, key=lambda item: item[:3], reverse=True):
            if score < self.min_score:
                return None
            return instantiate_statement(pattern, slots)
        return None


def default_learned_statement_parser() -> LearnedStatementParser:
    if STATEMENT_MODEL_PATH.exists():
        return LearnedStatementParser.from_model(STATEMENT_MODEL_PATH)
    return LearnedStatementParser.from_jsonl(STATEMENT_DATA_PATH)


def compile_statement_model_from_jsonl(path: str | Path) -> CompiledStatementModel:
    data_path = Path(path)
    return compile_statement_examples(load_statement_jsonl(data_path), source_sha256=file_sha256(data_path))


def compile_statement_examples(
    examples: tuple[StatementTrainingExample, ...],
    source_sha256: str = "",
) -> CompiledStatementModel:
    grouped: dict[tuple[Any, ...], CompiledStatementPattern] = {}
    for example in examples:
        feature_units = tuple(sorted(character_bigrams(remove_slots(example.sentence_template))))
        key = (
            example.sentence_template,
            tuple((entity.role, entity.name) for entity in example.entities),
            tuple(frame_signature(frame) for frame in example.frames),
        )
        previous = grouped.get(key)
        if previous is None:
            grouped[key] = CompiledStatementPattern(
                sentence_template=example.sentence_template,
                entities=example.entities,
                frames=example.frames,
                feature_units=feature_units,
                support=1,
            )
            continue
        grouped[key] = CompiledStatementPattern(
            sentence_template=previous.sentence_template,
            entities=previous.entities,
            frames=previous.frames,
            feature_units=previous.feature_units,
            support=previous.support + 1,
        )
    return CompiledStatementModel(
        schema=STATEMENT_MODEL_SCHEMA,
        source_sha256=source_sha256,
        example_count=len(examples),
        patterns=tuple(sorted(grouped.values(), key=lambda pattern: (-pattern.support, pattern.sentence_template))),
    )


def load_statement_model(path: str | Path) -> CompiledStatementModel:
    with Path(path).open("r", encoding="utf-8") as file:
        raw_model = json.load(file)
    if not isinstance(raw_model, dict):
        raise ValueError("Statement model must be a JSON object.")
    return statement_model_from_dict(raw_model)


def save_statement_model(model: CompiledStatementModel, path: str | Path) -> None:
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_name(f"{model_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(statement_model_to_dict(model), file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(model_path)


def statement_model_from_dict(record: dict[str, Any]) -> CompiledStatementModel:
    schema = str(record.get("schema") or "").strip()
    if schema != STATEMENT_MODEL_SCHEMA:
        raise ValueError(f"Unsupported statement model schema: {schema}")
    raw_patterns = record.get("patterns")
    if not isinstance(raw_patterns, list):
        raise ValueError("Statement model patterns must be a list.")
    return CompiledStatementModel(
        schema=schema,
        source_sha256=str(record.get("source_sha256") or ""),
        example_count=int(record.get("example_count") or 0),
        patterns=tuple(statement_pattern_from_dict(value) for value in raw_patterns),
    )


def statement_model_to_dict(model: CompiledStatementModel) -> dict[str, Any]:
    return {
        "schema": model.schema,
        "source_sha256": model.source_sha256,
        "example_count": model.example_count,
        "pattern_count": len(model.patterns),
        "patterns": [statement_pattern_to_dict(pattern) for pattern in model.patterns],
    }


def statement_pattern_from_dict(record: Any) -> CompiledStatementPattern:
    if not isinstance(record, dict):
        raise ValueError("Statement model pattern entries must be objects.")
    template = normalize_statement_text(str(record.get("sentence_template") or ""))
    if not template:
        raise ValueError("Statement model pattern requires sentence_template.")
    raw_entities = record.get("entities", ())
    if not isinstance(raw_entities, list):
        raise ValueError("Statement model pattern entities must be a list.")
    raw_frames = record.get("frames", ())
    if not isinstance(raw_frames, list):
        raise ValueError("Statement model pattern frames must be a list.")
    raw_units = record.get("feature_units", ())
    if not isinstance(raw_units, list):
        raise ValueError("Statement model pattern feature_units must be a list.")
    return CompiledStatementPattern(
        sentence_template=template,
        entities=tuple(entity_slot_from_dict(value, "Statement model pattern") for value in raw_entities),
        frames=tuple(frame_template_from_dict(value, "Statement model pattern") for value in raw_frames),
        feature_units=tuple(str(value) for value in raw_units if str(value)),
        support=int(record.get("support") or 1),
    )


def statement_pattern_to_dict(pattern: CompiledStatementPattern) -> dict[str, Any]:
    return {
        "sentence_template": pattern.sentence_template,
        "entities": [entity_slot_to_dict(entity) for entity in pattern.entities],
        "frames": [frame_template_to_dict(frame) for frame in pattern.frames],
        "feature_units": list(pattern.feature_units),
        "support": pattern.support,
    }


def evaluate_statement_parser(
    parser: LearnedStatementParser,
    examples: tuple[StatementTrainingExample, ...],
) -> StatementEvaluationResult:
    matched = 0
    for example in examples:
        parsed = parser(example.sentence)
        expected = instantiate_statement(example, slots_from_example(example))
        if parsed is not None and linearize_statement_result(parsed) == linearize_statement_result(expected):
            matched += 1
    return StatementEvaluationResult(total=len(examples), matched=matched)


def load_statement_jsonl(path: str | Path) -> tuple[StatementTrainingExample, ...]:
    examples: list[StatementTrainingExample] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid statement JSONL at line {line_number}: {error}") from error
            if not isinstance(raw_record, dict):
                raise ValueError(f"Invalid statement JSONL at line {line_number}: expected object")
            examples.append(statement_example_from_dict(raw_record, line_number=line_number))
    return tuple(examples)


def append_statement_record(path: str | Path, record: dict[str, Any]) -> StatementTrainingExample:
    example = statement_example_from_dict(record)
    data_path = Path(path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("a", encoding="utf-8") as file:
        json.dump(statement_example_to_record(example), file, ensure_ascii=False)
        file.write("\n")
    return example


def build_statement_record(
    sentence: str,
    sentence_template: str,
    *,
    entities: tuple[EntitySlot, ...],
    frames: tuple[FrameTemplate, ...],
    source: str = "human_feedback",
    split: str = "train",
) -> dict[str, Any]:
    return {
        "sentence": sentence.strip(),
        "sentence_template": sentence_template.strip(),
        "entities": [entity_slot_to_dict(entity) for entity in entities],
        "frames": [frame_template_to_dict(frame) for frame in frames],
        "source": source.strip() or "human_feedback",
        "split": split.strip() or "train",
    }


def statement_example_to_record(example: StatementTrainingExample) -> dict[str, Any]:
    return {
        "sentence": example.sentence,
        "sentence_template": example.sentence_template,
        "entities": [entity_slot_to_dict(entity) for entity in example.entities],
        "frames": [frame_template_to_dict(frame) for frame in example.frames],
        "source": example.source,
        "split": example.split,
    }


def statement_example_from_dict(record: dict[str, Any], *, line_number: int | None = None) -> StatementTrainingExample:
    prefix = f"Statement example at line {line_number}" if line_number is not None else "Statement example"
    sentence = normalize_statement_text(str(record.get("sentence") or record.get("text") or ""))
    template = normalize_statement_text(str(record.get("sentence_template") or ""))
    if not sentence or not template:
        raise ValueError(f"{prefix} requires sentence and sentence_template.")

    raw_entities = record.get("entities", ())
    if not isinstance(raw_entities, list):
        raise ValueError(f"{prefix} entities must be a list.")
    entities = tuple(entity_slot_from_dict(value, prefix) for value in raw_entities)

    raw_frames = record.get("frames", ())
    if not isinstance(raw_frames, list):
        raise ValueError(f"{prefix} frames must be a list.")
    frames = tuple(frame_template_from_dict(value, prefix) for value in raw_frames)
    if not frames:
        raise ValueError(f"{prefix} requires at least one frame.")

    return StatementTrainingExample(
        sentence=sentence,
        sentence_template=template,
        entities=entities,
        frames=frames,
        source=str(record.get("source") or "training").strip() or "training",
        split=str(record.get("split") or "train").strip() or "train",
    )


def entity_slot_from_dict(record: Any, prefix: str) -> EntitySlot:
    if not isinstance(record, dict):
        raise ValueError(f"{prefix} entity entries must be objects.")
    role = str(record.get("role") or "").strip()
    name = str(record.get("name") or "").strip()
    if not role or not name:
        raise ValueError(f"{prefix} entity entries require role and name.")
    return EntitySlot(role, name)


def frame_template_from_dict(record: Any, prefix: str) -> FrameTemplate:
    if not isinstance(record, dict):
        raise ValueError(f"{prefix} frame entries must be objects.")
    frame_type = str(record.get("frame_type") or "").strip()
    raw_roles = record.get("roles", {})
    if not frame_type or not isinstance(raw_roles, dict):
        raise ValueError(f"{prefix} frames require frame_type and roles object.")
    return FrameTemplate(
        frame_type=frame_type,
        roles=tuple((str(key), str(value)) for key, value in raw_roles.items()),
    )


def entity_slot_to_dict(entity: EntitySlot) -> dict[str, str]:
    return {"role": entity.role, "name": entity.name}


def frame_template_to_dict(frame: FrameTemplate) -> dict[str, Any]:
    return {"frame_type": frame.frame_type, "roles": dict(frame.roles)}


def frame_signature(frame: FrameTemplate) -> tuple[Any, ...]:
    return (frame.frame_type, frame.roles)


def instantiate_statement(
    example: StatementTrainingExample | CompiledStatementPattern,
    slots: dict[str, str],
) -> StatementParseResult:
    normalized_slots = {
        slot: normalize_entity_value(entity.role, slots.get(slot, ""))
        for entity in example.entities
        for slot in (entity.name,)
    }
    entities = [Entity(entity.role, instantiate_value(entity.name, normalized_slots)) for entity in example.entities]
    frames = [
        Frame(
            "pending",
            frame.frame_type,
            0,
            tuple(Role("pending", name, instantiate_value(value, normalized_slots)) for name, value in frame.roles),
        )
        for frame in example.frames
    ]
    return entities, frames


def normalize_entity_value(role: str, value: str) -> str:
    if role == "container":
        return normalize_container_slot(value)
    return normalize_slot_value(value)


def extract_slots(template: str, sentence: str) -> dict[str, str] | None:
    parts = split_template(template)
    slots: dict[str, str] = {}
    position = 0
    for index, part in enumerate(parts):
        if is_slot(part):
            next_literal = next((value for value in parts[index + 1 :] if not is_slot(value) and value), "")
            if next_literal:
                next_position = sentence.find(next_literal, position)
                if next_position < position:
                    return None
                value = sentence[position:next_position]
                position = next_position
            else:
                value = sentence[position:]
                position = len(sentence)
            if not value:
                return None
            slots[part] = normalize_slot_value(value)
            continue
        if not part:
            continue
        found = sentence.find(part, position)
        if found < position:
            return None
        position = found + len(part)
    return slots


def split_template(template: str) -> tuple[str, ...]:
    parts: list[str] = []
    index = 0
    while index < len(template):
        if template[index] != "$":
            next_slot = template.find("$", index)
            if next_slot < 0:
                parts.append(template[index:])
                break
            parts.append(template[index:next_slot])
            index = next_slot
            continue
        hash_index = template.find("#", index + 1)
        if hash_index >= 0:
            end = hash_index + 1
            while end < len(template) and template[end].isdigit():
                end += 1
        else:
            end = index + 1
            while end < len(template) and (template[end].isalnum() or template[end] == "_"):
                end += 1
        parts.append(template[index:end])
        index = end
    return tuple(part for part in parts if part != "")


def is_slot(value: str) -> bool:
    return value.startswith("$")


def instantiate_value(value: str, slots: dict[str, str]) -> str:
    resolved = value
    for slot, slot_value in sorted(slots.items(), key=lambda item: len(item[0]), reverse=True):
        resolved = resolved.replace(slot, slot_value)
    return resolved


def slots_from_example(example: StatementTrainingExample) -> dict[str, str]:
    slots: dict[str, str] = {}
    for entity in example.entities:
        slots[entity.name] = entity.name.replace("$", "")
    extracted = extract_slots(example.sentence_template, example.sentence)
    if extracted is not None:
        slots.update(extracted)
    return slots


def normalize_statement_text(sentence: str) -> str:
    normalized = normalize_slot_value(sentence).strip().rstrip("。！？!?，,")
    normalized = normalize_containment_expression(normalize_take_out_expression(normalized))
    normalized = normalize_container_surface(normalized)
    return normalized


def statement_score(template: str, sentence: str) -> float:
    abstract = remove_slots(template)
    if not abstract or not sentence:
        return 0.0
    if abstract == sentence:
        return 1.0
    units = character_bigrams(abstract)
    sentence_units = character_bigrams(sentence)
    if not units or not sentence_units:
        return 0.0
    return len(units & sentence_units) / len(units)


def statement_pattern_score(pattern: CompiledStatementPattern, sentence: str) -> float:
    abstract = remove_slots(pattern.sentence_template)
    if abstract == sentence:
        return 1.0
    if not abstract or not sentence:
        return 0.0
    sentence_units = character_bigrams(sentence)
    feature_units = set(pattern.feature_units)
    if not sentence_units or not feature_units:
        return 0.0
    return len(feature_units & sentence_units) / len(feature_units)


def remove_slots(template: str) -> str:
    return "".join(part for part in split_template(template) if not is_slot(part))


def character_bigrams(text: str) -> set[str]:
    if len(text) <= 1:
        return {text} if text else set()
    return {text[index : index + 2] for index in range(len(text) - 1)}


def linearize_statement_result(result: StatementParseResult) -> tuple[str, ...]:
    entities, frames = result
    lines = [entity.linearize() for entity in entities]
    for frame in frames:
        lines.append(f"FRAME {frame.frame_type}")
        lines.extend(f"ROLE {role.name}={role.value}" for role in frame.roles)
    return tuple(lines)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
