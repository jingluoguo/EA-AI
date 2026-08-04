from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..structure import Intention, Structure
from .intent_dataset import intent_record_from_dict, load_intent_jsonl
from .normalization import normalize_question


@dataclass(frozen=True)
class IntentTrainingExample:
    observation: str
    intention: Intention


@dataclass(frozen=True)
class IntentEvaluationResult:
    total: int
    matched: int

    @property
    def accuracy(self) -> float:
        return self.matched / self.total if self.total else 0.0


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
    return tuple(IntentTrainingExample(record.observation, record.intention) for record in load_intent_jsonl(path))


def example_from_record(record: dict[str, Any]) -> IntentTrainingExample:
    dataset_record = intent_record_from_dict(record)
    return IntentTrainingExample(dataset_record.observation, dataset_record.intention)


def evaluate_intent_analyzer(
    analyzer: InMemoryIntentAnalyzer,
    examples: tuple[IntentTrainingExample, ...],
) -> IntentEvaluationResult:
    matched = 0
    empty_structure = Structure(entities=(), rules=())
    for example in examples:
        predictions = analyzer(example.observation, empty_structure)
        if any(intent_matches(prediction, example.intention) for prediction in predictions):
            matched += 1
    return IntentEvaluationResult(total=len(examples), matched=matched)


def intent_matches(predicted: Intention, expected: Intention) -> bool:
    return predicted.subject == expected.subject and predicted.goal == expected.goal


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
