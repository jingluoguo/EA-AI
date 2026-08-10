from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..dataset_io import append_jsonl_object, file_sha256, load_jsonl_objects
from ..structure import Entity, Frame, Role
from ..capabilities import StatementParseResult
from ..perception.normalizer import (
    normalize_container_slot,
    normalize_statement,
    normalize_slot_value,
)


STATEMENT_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "statement_examples.jsonl"
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


def statement_pattern_from_dict(record: Any) -> CompiledStatementPattern:
    if not isinstance(record, dict):
        raise ValueError("Statement model pattern entries must be objects.")
    template = normalize_statement_template(str(record.get("sentence_template") or ""))
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
    parser,
    examples: tuple[StatementTrainingExample, ...],
) -> StatementEvaluationResult:
    matched = 0
    for example in examples:
        parsed = parser(example.sentence)
        if not example.frames:
            if parsed is None:
                matched += 1
            continue
        expected = instantiate_statement(example, slots_from_example(example))
        if parsed is not None and linearize_statement_result(parsed) == linearize_statement_result(expected):
            matched += 1
    return StatementEvaluationResult(total=len(examples), matched=matched)


def load_statement_jsonl(path: str | Path) -> tuple[StatementTrainingExample, ...]:
    return tuple(
        statement_example_from_dict(raw_record, line_number=line_number)
        for line_number, raw_record in enumerate(load_jsonl_objects(path, "statement"), start=1)
    )


def append_statement_record(path: str | Path, record: dict[str, Any]) -> StatementTrainingExample:
    example = statement_example_from_dict(record)
    append_jsonl_object(path, statement_example_to_record(example))
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
    template = normalize_statement_template(str(record.get("sentence_template") or ""))
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
            previous_value = slots.get(part)
            if previous_value:
                found_previous = sentence.find(previous_value, position)
                if found_previous < position:
                    return None
                position = found_previous + len(previous_value)
                continue
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
    return normalize_statement(sentence)


def normalize_statement_template(template: str) -> str:
    """Normalize a template with placeholder context preserved."""
    parts = split_template(template)
    protected = protect_template_slots(parts)
    normalized = normalize_statement(protected)
    return restore_template_slots(normalized, parts).strip().rstrip("。！？!?，,")


def protect_template_slots(parts: tuple[str, ...]) -> str:
    slot_markers = template_slot_markers(parts)
    slot_index = 0
    protected_parts: list[str] = []
    for part in parts:
        if is_slot(part):
            protected_parts.append(slot_markers[slot_index])
            slot_index += 1
        else:
            protected_parts.append(part)
    return "".join(protected_parts)


def restore_template_slots(normalized: str, parts: tuple[str, ...]) -> str:
    slot_markers = template_slot_markers(parts)
    slot_index = 0
    restored = normalized
    for part in parts:
        if not is_slot(part):
            continue
        restored = restored.replace(slot_markers[slot_index], part, 1)
        slot_index += 1
    return restored


def template_slot_markers(parts: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(chr(0xE000 + index) for index, part in enumerate(parts) if is_slot(part))


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
    lines = [entity.linearize() for entity in sorted(entities, key=lambda entity: (entity.role, entity.name))]
    for frame in frames:
        lines.append(f"FRAME {frame.frame_type}")
        lines.extend(f"ROLE {role.name}={role.value}" for role in sorted(frame.roles, key=lambda role: role.name))
    return tuple(lines)
