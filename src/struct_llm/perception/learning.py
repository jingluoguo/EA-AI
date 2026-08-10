from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..dataset_io import load_jsonl_objects
from ..structure import Entity


PERCEPTION_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "perception_examples.jsonl"
PERCEPTION_RECORD_SCHEMA = "struct_llm.perception_example.v1"
PERCEPTION_TASKS = frozenset(
    {
        "sentence_segmentation",
        "candidate_segmentation",
        "normalization",
        "reference_resolution",
    }
)


@dataclass(frozen=True)
class SentenceSegment:
    text: str
    is_question: bool = False


@dataclass(frozen=True)
class PerceptionTrainingExample:
    task: str
    text: str
    segments: tuple[SentenceSegment, ...] = ()
    candidates: tuple[str, ...] = ()
    normalized: str = ""
    normalization_mode: str = "question"
    entities: tuple[Entity, ...] = ()
    reference: str = ""
    resolved: str = ""
    source: str = "training"
    split: str = "train"


@dataclass(frozen=True)
class PerceptionEvaluationResult:
    total: int
    matched: int
    by_task: tuple[tuple[str, int, int], ...]

    @property
    def accuracy(self) -> float:
        return self.matched / self.total if self.total else 0.0


class PerceptionModel(Protocol):
    def split_sentences(self, text: str) -> tuple[tuple[str, bool], ...]:
        ...

    def split_query_candidate(self, text: str) -> tuple[str, ...]:
        ...

    def normalize(self, text: str, mode: str = "question") -> str:
        ...

    def resolve_references(self, text: str, entities: tuple[Entity, ...]) -> str:
        ...


def load_perception_jsonl(path: str | Path = PERCEPTION_DATA_PATH) -> tuple[PerceptionTrainingExample, ...]:
    return tuple(
        perception_example_from_dict(record, line_number=line_number)
        for line_number, record in enumerate(load_jsonl_objects(path, "perception"), start=1)
    )


def perception_example_from_dict(
    record: dict[str, Any],
    *,
    line_number: int | None = None,
) -> PerceptionTrainingExample:
    prefix = f"Perception example at line {line_number}" if line_number is not None else "Perception example"
    schema = str(record.get("schema") or PERCEPTION_RECORD_SCHEMA).strip()
    if schema != PERCEPTION_RECORD_SCHEMA:
        raise ValueError(f"{prefix} has unsupported schema: {schema}")
    task = str(record.get("task") or "").strip()
    if task not in PERCEPTION_TASKS:
        raise ValueError(f"{prefix} task must be one of {sorted(PERCEPTION_TASKS)}.")
    text = str(record.get("text") or "").strip()
    if not text:
        raise ValueError(f"{prefix} requires text.")

    segments: tuple[SentenceSegment, ...] = ()
    candidates: tuple[str, ...] = ()
    normalized = ""
    normalization_mode = str(record.get("normalization_mode") or "question").strip() or "question"
    entities: tuple[Entity, ...] = ()
    reference = str(record.get("reference") or "").strip()
    resolved = str(record.get("resolved") or "").strip()

    if task == "sentence_segmentation":
        raw_segments = record.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValueError(f"{prefix} sentence_segmentation requires non-empty segments.")
        segments = tuple(sentence_segment_from_dict(value, prefix) for value in raw_segments)
    elif task == "candidate_segmentation":
        raw_candidates = record.get("candidates")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError(f"{prefix} candidate_segmentation requires non-empty candidates.")
        candidates = tuple(str(value).strip() for value in raw_candidates if str(value).strip())
        if not candidates:
            raise ValueError(f"{prefix} candidate_segmentation requires non-empty candidates.")
    elif task == "normalization":
        if "normalized" not in record:
            raise ValueError(f"{prefix} normalization requires normalized.")
        normalized = str(record.get("normalized") or "").strip()
        if normalization_mode not in {"question", "statement"}:
            raise ValueError(f"{prefix} normalization_mode must be question or statement.")
    elif task == "reference_resolution":
        raw_entities = record.get("entities")
        if not isinstance(raw_entities, list):
            raise ValueError(f"{prefix} reference_resolution requires entities list.")
        entities = tuple(entity_from_dict(value, prefix) for value in raw_entities)
        if reference and reference not in text:
            raise ValueError(f"{prefix} reference must be present in text when supplied.")
        if not resolved:
            resolved = text

    return PerceptionTrainingExample(
        task=task,
        text=text,
        segments=segments,
        candidates=candidates,
        normalized=normalized,
        normalization_mode=normalization_mode,
        entities=entities,
        reference=reference,
        resolved=resolved,
        source=str(record.get("source") or "training").strip() or "training",
        split=str(record.get("split") or "train").strip() or "train",
    )


def sentence_segment_from_dict(value: Any, prefix: str) -> SentenceSegment:
    if not isinstance(value, dict):
        raise ValueError(f"{prefix} segment entries must be objects.")
    text = str(value.get("text") or "").strip()
    if not text:
        raise ValueError(f"{prefix} segment entries require text.")
    return SentenceSegment(text, bool(value.get("is_question", False)))


def entity_from_dict(value: Any, prefix: str) -> Entity:
    if not isinstance(value, dict):
        raise ValueError(f"{prefix} entity entries must be objects.")
    role = str(value.get("role") or "").strip()
    name = str(value.get("name") or "").strip()
    if not role or not name:
        raise ValueError(f"{prefix} entity entries require role and name.")
    return Entity(role, name)


def evaluate_perception_model(
    model: PerceptionModel,
    examples: tuple[PerceptionTrainingExample, ...],
) -> PerceptionEvaluationResult:
    matched = 0
    task_counts: dict[str, list[int]] = {}
    for example in examples:
        task_total, task_matched = task_counts.setdefault(example.task, [0, 0])
        task_total += 1
        success = perception_prediction_matches(model, example)
        if success:
            matched += 1
            task_matched += 1
        task_counts[example.task] = [task_total, task_matched]
    return PerceptionEvaluationResult(
        total=len(examples),
        matched=matched,
        by_task=tuple(
            (task, counts[0], counts[1])
            for task, counts in sorted(task_counts.items())
        ),
    )


def perception_prediction_matches(model: PerceptionModel, example: PerceptionTrainingExample) -> bool:
    if example.task == "sentence_segmentation":
        expected = tuple((segment.text, segment.is_question) for segment in example.segments)
        return model.split_sentences(example.text) == expected
    if example.task == "candidate_segmentation":
        return model.split_query_candidate(example.text) == example.candidates
    if example.task == "normalization":
        return model.normalize(example.text, example.normalization_mode) == example.normalized
    if example.task == "reference_resolution":
        return model.resolve_references(example.text, example.entities) == example.resolved
    return False
