from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..structure import Query
from .capabilities import QueryParser
from .query_learning import (
    CompiledQueryPattern,
    EntityExample,
    append_query_record,
    build_query_record,
    compile_query_model_from_jsonl,
    entity_example_to_dict,
    query_to_dict,
    save_query_model,
    suggest_query_pattern,
)
from .statement_learning import (
    EntitySlot,
    FrameTemplate,
    append_statement_record,
    build_statement_record,
    compile_statement_model_from_jsonl,
    save_statement_model,
)


@dataclass(frozen=True)
class LearningPaths:
    query_data: Path = Path("data/query_examples.jsonl")
    query_model: Path = Path("data/query_model.json")
    statement_data: Path = Path("data/statement_examples.jsonl")
    statement_model: Path = Path("data/statement_model.json")


@dataclass(frozen=True)
class QuerySuggestion:
    text: str
    score: float
    pattern: CompiledQueryPattern

    @property
    def query(self) -> Query:
        return self.pattern.query


@dataclass(frozen=True)
class LearningWriteResult:
    kind: str
    data_path: Path
    model_path: Path | None = None
    example_count: int = 0
    pattern_count: int = 0


def suggest_query_feedback(
    text: str,
    query_parsers: tuple[QueryParser, ...],
    min_score: float = 0.12,
) -> QuerySuggestion | None:
    suggested = suggest_query_pattern(text, (), query_parsers, min_score=min_score)
    if suggested is None:
        return None
    score, pattern = suggested
    return QuerySuggestion(text=text, score=score, pattern=pattern)


def accept_query_suggestion(
    suggestion: QuerySuggestion,
    paths: LearningPaths,
    source: str = "human_feedback",
) -> LearningWriteResult:
    append_query_record(
        paths.query_data,
        {
            "question": suggestion.text.strip(),
            "entities": [entity_example_to_dict(entity) for entity in suggestion.pattern.entities],
            "query": query_to_dict(suggestion.pattern.query),
            "source": source,
            "split": "train",
        },
    )
    model = compile_query_model_from_jsonl(paths.query_data)
    save_query_model(model, paths.query_model)
    return LearningWriteResult(
        kind="query",
        data_path=paths.query_data,
        model_path=paths.query_model,
        example_count=model.example_count,
        pattern_count=len(model.patterns),
    )


def save_manual_query_feedback(
    text: str,
    intent: str,
    target: str,
    paths: LearningPaths,
    *,
    entities: tuple[EntityExample, ...] = (),
    qualifiers: tuple[str, ...] = (),
    source: str = "human_feedback",
) -> LearningWriteResult:
    append_query_record(
        paths.query_data,
        build_query_record(
            text,
            intent,
            target,
            entities=entities,
            qualifiers=qualifiers,
            source=source,
        ),
    )
    model = compile_query_model_from_jsonl(paths.query_data)
    save_query_model(model, paths.query_model)
    return LearningWriteResult(
        kind="query",
        data_path=paths.query_data,
        model_path=paths.query_model,
        example_count=model.example_count,
        pattern_count=len(model.patterns),
    )


def save_manual_statement_feedback(
    text: str,
    sentence_template: str,
    paths: LearningPaths,
    *,
    entities: tuple[EntitySlot, ...],
    frames: tuple[FrameTemplate, ...],
    source: str = "human_feedback",
) -> LearningWriteResult:
    append_statement_record(
        paths.statement_data,
        build_statement_record(
            text,
            sentence_template,
            entities=entities,
            frames=frames,
            source=source,
        ),
    )
    model = compile_statement_model_from_jsonl(paths.statement_data)
    save_statement_model(model, paths.statement_model)
    return LearningWriteResult(
        kind="statement",
        data_path=paths.statement_data,
        model_path=paths.statement_model,
        example_count=model.example_count,
        pattern_count=len(model.patterns),
    )
