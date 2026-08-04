from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..structure import Intention, Structure
from .normalization import normalize_question


@dataclass(frozen=True)
class IntentTrainingExample:
    observation: str
    intention: Intention


@dataclass(frozen=True)
class InMemoryIntentAnalyzer:
    examples: tuple[IntentTrainingExample, ...] = ()
    min_score: float = 0.6

    def learn(self, observation: str, intention: Intention) -> InMemoryIntentAnalyzer:
        return InMemoryIntentAnalyzer(
            examples=(*self.examples, IntentTrainingExample(observation, intention)),
            min_score=self.min_score,
        )

    @classmethod
    def from_jsonl(cls, path: str | Path, min_score: float = 0.6) -> InMemoryIntentAnalyzer:
        return cls(from_jsonl(path), min_score=min_score)

    @classmethod
    def from_records(cls, records: list[dict[str, Any]], min_score: float = 0.6) -> InMemoryIntentAnalyzer:
        return cls(tuple(example_from_record(record) for record in records), min_score=min_score)

    def __call__(self, text: str, structure: Structure) -> tuple[Intention, ...]:
        if not self.examples:
            return ()
        normalized_text = normalize_observation(text)
        scored = [
            (observation_score(normalize_observation(example.observation), normalized_text), example.intention)
            for example in self.examples
        ]
        matches = [
            intention_with_source(intention, score)
            for score, intention in sorted(scored, key=lambda item: item[0], reverse=True)
            if score >= self.min_score
        ]
        return tuple(matches[:3])


def intention_with_source(intention: Intention, score: float) -> Intention:
    confidence = min(1.0, max(intention.confidence, score))
    return Intention(
        subject=intention.subject,
        goal=intention.goal,
        belief=intention.belief,
        strategy=intention.strategy,
        evidence=intention.evidence,
        confidence=confidence,
        source=intention.source,
    )


def from_jsonl(path: str | Path) -> tuple[IntentTrainingExample, ...]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid intent JSONL at line {line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Invalid intent JSONL at line {line_number}: expected object")
            records.append(record)
    return tuple(example_from_record(record) for record in records)


def example_from_record(record: dict[str, Any]) -> IntentTrainingExample:
    observation = str(record.get("observation") or record.get("text") or "").strip()
    if not observation:
        raise ValueError("Intent example requires an observation or text field.")

    raw_intention = record.get("intention", record)
    if not isinstance(raw_intention, dict):
        raise ValueError("Intent example intention field must be an object.")

    subject = str(raw_intention.get("subject") or "").strip()
    goal = str(raw_intention.get("goal") or "").strip()
    if not subject or not goal:
        raise ValueError("Intent example requires intention.subject and intention.goal.")

    confidence = raw_intention.get("confidence", 1.0)
    try:
        parsed_confidence = float(confidence)
    except (TypeError, ValueError) as error:
        raise ValueError("Intent example confidence must be numeric.") from error

    return IntentTrainingExample(
        observation=observation,
        intention=Intention(
            subject=subject,
            goal=goal,
            belief=str(raw_intention.get("belief") or "").strip(),
            strategy=str(raw_intention.get("strategy") or "").strip(),
            evidence=str(raw_intention.get("evidence") or observation).strip(),
            confidence=parsed_confidence,
            source=str(raw_intention.get("source") or record.get("source") or "jsonl").strip(),
        ),
    )


def observation_score(example: str, text: str) -> float:
    if not example or not text:
        return 0.0
    if example in text:
        return 1.0
    example_units = character_bigrams(example)
    text_units = character_bigrams(text)
    if not example_units or not text_units:
        return 0.0
    overlap = len(example_units & text_units)
    return overlap / len(example_units)


def character_bigrams(text: str) -> set[str]:
    if len(text) <= 1:
        return {text} if text else set()
    return {text[index : index + 2] for index in range(len(text) - 1)}


def normalize_observation(text: str) -> str:
    return normalize_question(text).replace("。", "").replace("，", "").replace(",", "").strip()
