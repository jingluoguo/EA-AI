from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..dataset_io import append_jsonl_object, load_jsonl_objects
from ..structure import Intention


@dataclass(frozen=True)
class IntentDatasetRecord:
    observation: str
    intention: Intention
    context: tuple[str, ...] = ()
    world_state: tuple[str, ...] = ()
    belief_state: tuple[str, ...] = ()
    answer: str = ""
    source: str = "human_feedback"
    split: str = "train"

    def to_json_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "observation": self.observation,
            "intention": {
                "subject": self.intention.subject,
                "goal": self.intention.goal,
                "confidence": self.intention.confidence,
                "source": self.intention.source,
            },
            "source": self.source,
            "split": self.split,
        }
        if self.intention.belief:
            record["intention"]["belief"] = self.intention.belief
        if self.intention.strategy:
            record["intention"]["strategy"] = self.intention.strategy
        if self.intention.evidence:
            record["intention"]["evidence"] = self.intention.evidence
        if self.context:
            record["context"] = list(self.context)
        if self.world_state:
            record["world_state"] = list(self.world_state)
        if self.belief_state:
            record["belief_state"] = list(self.belief_state)
        if self.answer:
            record["answer"] = self.answer
        return record


def build_intent_record(
    observation: str,
    subject: str,
    goal: str,
    *,
    belief: str = "",
    strategy: str = "",
    evidence: str = "",
    confidence: float = 1.0,
    context: Iterable[str] = (),
    world_state: Iterable[str] = (),
    belief_state: Iterable[str] = (),
    answer: str = "",
    source: str = "human_feedback",
    split: str = "train",
) -> IntentDatasetRecord:
    return validate_intent_record(
        IntentDatasetRecord(
            observation=observation.strip(),
            intention=Intention(
                subject=subject.strip(),
                goal=goal.strip(),
                belief=belief.strip(),
                strategy=strategy.strip(),
                evidence=(evidence or observation).strip(),
                confidence=float(confidence),
                source=source.strip() or "human_feedback",
            ),
            context=clean_string_tuple(context),
            world_state=clean_string_tuple(world_state),
            belief_state=clean_string_tuple(belief_state),
            answer=answer.strip(),
            source=source.strip() or "human_feedback",
            split=split.strip() or "train",
        )
    )


def load_intent_jsonl(path: str | Path) -> tuple[IntentDatasetRecord, ...]:
    return tuple(
        intent_record_from_dict(raw_record, line_number=line_number)
        for line_number, raw_record in enumerate(load_jsonl_objects(path, "intent"), start=1)
    )


def append_intent_record(path: str | Path, record: IntentDatasetRecord) -> None:
    validated = validate_intent_record(record)
    append_jsonl_object(path, validated.to_json_record(), sort_keys=True)


def intent_record_from_dict(record: dict[str, Any], *, line_number: int | None = None) -> IntentDatasetRecord:
    prefix = f"Intent example at line {line_number}" if line_number is not None else "Intent example"
    observation = str(record.get("observation") or record.get("text") or "").strip()
    if not observation:
        raise ValueError(f"{prefix} requires an observation or text field.")

    raw_intention = record.get("intention", record)
    if not isinstance(raw_intention, dict):
        raise ValueError(f"{prefix} intention field must be an object.")

    subject = str(raw_intention.get("subject") or "").strip()
    goal = str(raw_intention.get("goal") or "").strip()
    if not subject or not goal:
        raise ValueError(f"{prefix} requires intention.subject and intention.goal.")

    confidence = parse_confidence(raw_intention.get("confidence", 1.0), prefix)
    source = str(raw_intention.get("source") or record.get("source") or "jsonl").strip() or "jsonl"

    return validate_intent_record(
        IntentDatasetRecord(
            observation=observation,
            intention=Intention(
                subject=subject,
                goal=goal,
                belief=str(raw_intention.get("belief") or "").strip(),
                strategy=str(raw_intention.get("strategy") or "").strip(),
                evidence=str(raw_intention.get("evidence") or observation).strip(),
                confidence=confidence,
                source=source,
            ),
            context=field_to_tuple(record.get("context"), "context", prefix),
            world_state=field_to_tuple(record.get("world_state"), "world_state", prefix),
            belief_state=field_to_tuple(record.get("belief_state"), "belief_state", prefix),
            answer=str(record.get("answer") or "").strip(),
            source=str(record.get("source") or source).strip() or source,
            split=str(record.get("split") or "train").strip() or "train",
        )
    )


def validate_intent_record(record: IntentDatasetRecord) -> IntentDatasetRecord:
    if not record.observation:
        raise ValueError("Intent example requires an observation.")
    if not record.intention.subject:
        raise ValueError("Intent example requires intention.subject.")
    if not record.intention.goal:
        raise ValueError("Intent example requires intention.goal.")
    if not 0.0 <= record.intention.confidence <= 1.0:
        raise ValueError("Intent example confidence must be between 0 and 1.")
    if not record.source:
        raise ValueError("Intent example requires a source.")
    if not record.split:
        raise ValueError("Intent example requires a split.")
    return record


def parse_confidence(value: Any, prefix: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{prefix} confidence must be numeric.") from error
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{prefix} confidence must be between 0 and 1.")
    return parsed


def field_to_tuple(value: Any, field_name: str, prefix: str) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if not isinstance(value, list):
        raise ValueError(f"{prefix} {field_name} must be a string or list of strings.")
    return clean_string_tuple(value)


def clean_string_tuple(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(str(value).strip() for value in values if str(value).strip())
