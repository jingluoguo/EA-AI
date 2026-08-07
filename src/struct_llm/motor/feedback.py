from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..structure import Query
from ..capabilities import QueryParser
from ..comprehension.episode import (
    EPISODE_DATA_PATH,
    ActionResult,
    DialogueTurn,
    FeedbackDiagnosis,
    append_episode_record,
    build_episode_record,
    compile_episode_model_from_jsonl,
)
from .learning_queue import (
    UnrecognizedExample,
    append_unrecognized_example,
)
from ..comprehension.query import (
    CompiledQueryPattern,
    EntityExample,
    append_query_record,
    build_query_record,
    entity_example_to_dict,
    query_to_dict,
    suggest_query_pattern,
)
from ..comprehension.statement import (
    EntitySlot,
    FrameTemplate,
    append_statement_record,
    build_statement_record,
)
from ..memory.long_term import (
    MEMORY_CHAT_DATA_PATH,
    MEMORY_DIRECT_DATA_PATH,
    MEMORY_MODEL_PATH,
    MemoryEntry,
    MemoryWriteResult,
    append_memory_entry,
    compile_memory_model_from_jsonl,
    default_memory_states,
    extract_chat_memory_entries,
    save_memory_model,
)
from ..memory.knowledge import (
    MEMORY_KNOWLEDGE_DATA_PATH,
    MEMORY_KNOWLEDGE_MODEL_PATH,
    CompiledMemoryKnowledgeModel,
    KnowledgeWriteResult,
    MemoryKnowledgeEntry,
    append_memory_knowledge_entry,
    compile_memory_knowledge_model_from_jsonl,
    save_memory_knowledge_model,
)
from ..metacognition.confidence import (
    CONFIRM_CONFIDENCE_THRESHOLD,
    DIRECT_CONFIDENCE_THRESHOLD,
    confidence_band,
)
from ..neural.query_classifier import (
    QUERY_NEURAL_META_PATH,
    QUERY_NEURAL_WEIGHTS_PATH,
    train_query_neural_model,
)
from ..neural.statement_classifier import (
    STATEMENT_NEURAL_META_PATH,
    STATEMENT_NEURAL_WEIGHTS_PATH,
    train_statement_neural_model,
)


@dataclass(frozen=True)
class LearningPaths:
    query_data: Path = Path("data/query_examples.jsonl")
    query_neural_weights: Path = QUERY_NEURAL_WEIGHTS_PATH
    query_neural_meta: Path = QUERY_NEURAL_META_PATH
    statement_data: Path = Path("data/statement_examples.jsonl")
    statement_neural_weights: Path = STATEMENT_NEURAL_WEIGHTS_PATH
    statement_neural_meta: Path = STATEMENT_NEURAL_META_PATH
    dialog_answer_data: Path = Path("data/dialog_answer_examples.jsonl")
    dialog_answer_model: Path = Path("data/dialog_answer_model.json")
    unrecognized_data: Path = Path("data/unrecognized_examples.jsonl")
    memory_direct_data: Path = MEMORY_DIRECT_DATA_PATH
    memory_chat_data: Path = MEMORY_CHAT_DATA_PATH
    memory_model: Path = MEMORY_MODEL_PATH
    memory_knowledge_data: Path = MEMORY_KNOWLEDGE_DATA_PATH
    memory_knowledge_model: Path = MEMORY_KNOWLEDGE_MODEL_PATH
    episode_data: Path = EPISODE_DATA_PATH


@dataclass(frozen=True)
class QuerySuggestion:
    text: str
    score: float
    pattern: CompiledQueryPattern

    @property
    def query(self) -> Query:
        return self.pattern.query


@dataclass(frozen=True)
class QueryUncertaintyAssessment:
    text: str
    score: float
    band: str
    suggestion: QuerySuggestion | None = None


@dataclass(frozen=True)
class LearningWriteResult:
    kind: str
    data_path: Path
    model_path: Path | None = None
    example_count: int = 0
    pattern_count: int = 0


def save_chat_memory_feedback(
    text: str,
    structure,
    paths: LearningPaths,
    *,
    confidence: float = 0.85,
) -> MemoryWriteResult:
    entries = extract_chat_memory_entries(text, structure, confidence=confidence)
    for entry in entries:
        append_memory_entry(paths.memory_chat_data, entry)
    model = compile_memory_model_from_jsonl(paths.memory_direct_data, paths.memory_chat_data)
    save_memory_model(model, paths.memory_model)
    return MemoryWriteResult(
        data_path=paths.memory_chat_data,
        model_path=paths.memory_model,
        entry_count=len(entries),
        state_count=len(model.states),
    )


def save_direct_memory_structure_feedback(
    text: str,
    structure,
    paths: LearningPaths,
    *,
    confidence: float = 1.0,
    source: str = "human_feedback",
) -> MemoryWriteResult:
    entries = extract_chat_memory_entries(text, structure, confidence=confidence)
    for entry in entries:
        append_memory_entry(
            paths.memory_direct_data,
            MemoryEntry(
                state=entry.state,
                text=text,
                channel="direct",
                source=source,
                confidence=confidence,
            ),
        )
    model = compile_memory_model_from_jsonl(paths.memory_direct_data, paths.memory_chat_data)
    save_memory_model(model, paths.memory_model)
    return MemoryWriteResult(
        data_path=paths.memory_direct_data,
        model_path=paths.memory_model,
        entry_count=len(entries),
        state_count=len(model.states),
    )


def save_direct_memory_feedback(
    state,
    paths: LearningPaths,
    *,
    text: str = "",
    source: str = "human_feedback",
    channel: str = "direct",
    confidence: float = 1.0,
) -> MemoryWriteResult:
    append_memory_entry(
        paths.memory_direct_data,
        MemoryEntry(
            state=state,
            text=text,
            channel=channel,
            source=source,
            confidence=confidence,
        ),
    )
    model = compile_memory_model_from_jsonl(paths.memory_direct_data, paths.memory_chat_data)
    save_memory_model(model, paths.memory_model)
    return MemoryWriteResult(
        data_path=paths.memory_direct_data,
        model_path=paths.memory_model,
        entry_count=1,
        state_count=len(model.states),
    )


def suggest_query_feedback(
    text: str,
    query_parsers: tuple[QueryParser, ...],
    min_score: float = CONFIRM_CONFIDENCE_THRESHOLD,
) -> QuerySuggestion | None:
    suggested = suggest_query_pattern(text, (), query_parsers, min_score=min_score)
    if suggested is None:
        return None
    score, pattern = suggested
    return QuerySuggestion(text=text, score=score, pattern=pattern)


def assess_query_uncertainty(
    text: str,
    query_parsers: tuple[QueryParser, ...],
) -> QueryUncertaintyAssessment:
    suggested = suggest_query_pattern(text, (), query_parsers, min_score=0.0)
    if suggested is None:
        return QueryUncertaintyAssessment(text=text, score=0.0, band="unknown")
    score, pattern = suggested
    band = confidence_band(score)
    suggestion = QuerySuggestion(text=text, score=score, pattern=pattern) if band != "unknown" else None
    return QueryUncertaintyAssessment(text=text, score=score, band=band, suggestion=suggestion)


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
    bundle = train_query_neural_model(paths.query_data, paths.query_neural_weights, paths.query_neural_meta)
    return LearningWriteResult(
        kind="query",
        data_path=paths.query_data,
        model_path=paths.query_neural_weights,
        example_count=bundle.result.example_count,
        pattern_count=bundle.result.label_count,
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
    bundle = train_query_neural_model(paths.query_data, paths.query_neural_weights, paths.query_neural_meta)
    return LearningWriteResult(
        kind="query",
        data_path=paths.query_data,
        model_path=paths.query_neural_weights,
        example_count=bundle.result.example_count,
        pattern_count=bundle.result.label_count,
    )


def save_new_dialog_capability_feedback(
    question: str,
    capability_name: str,
    paths: LearningPaths,
    *,
    source: str = "human_feedback",
) -> LearningWriteResult:
    return save_manual_query_feedback(
        question,
        "dialog_act",
        capability_name,
        paths,
        source=source,
    )


def save_unrecognized_feedback(
    text: str,
    paths: LearningPaths,
    *,
    confidence: float = 0.0,
    reason: str = "low_confidence",
) -> UnrecognizedExample:
    return append_unrecognized_example(
        paths.unrecognized_data,
        UnrecognizedExample(
            text=text,
            confidence=confidence,
            reason=reason,
        ),
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
    bundle = train_statement_neural_model(
        paths.statement_data,
        paths.statement_neural_weights,
        paths.statement_neural_meta,
    )
    return LearningWriteResult(
        kind="statement",
        data_path=paths.statement_data,
        model_path=paths.statement_neural_weights,
        example_count=bundle.result.example_count,
        pattern_count=bundle.result.label_count,
    )


def save_manual_episode_feedback(
    text: str,
    response_policy: str,
    paths: LearningPaths,
    *,
    pragmatic_acts,
    episode_id: str = "",
    dialogue_turn: int = 0,
    speaker: str = "user",
    scene: str = "",
    previous_turns: tuple[DialogueTurn, ...] = (),
    known_world_state=(),
    belief_state=(),
    relationship_state=(),
    focus=(),
    expected_query: Query | None = None,
    expected_frames=(),
    expected_state_delta=(),
    expected_answer: str = "",
    action_result: ActionResult | None = None,
    feedback_diagnosis: FeedbackDiagnosis | None = None,
    source: str = "human_feedback",
) -> LearningWriteResult:
    append_episode_record(
        paths.episode_data,
        build_episode_record(
            text,
            response_policy,
            pragmatic_acts=tuple(pragmatic_acts),
            episode_id=episode_id,
            dialogue_turn=dialogue_turn,
            speaker=speaker,
            scene=scene,
            previous_turns=previous_turns,
            known_world_state=tuple(known_world_state),
            belief_state=tuple(belief_state),
            relationship_state=tuple(relationship_state),
            focus=focus,
            expected_query=expected_query,
            expected_frames=tuple(expected_frames),
            expected_state_delta=tuple(expected_state_delta),
            expected_answer=expected_answer,
            action_result=action_result,
            feedback_diagnosis=feedback_diagnosis,
            source=source,
        ),
    )
    model = compile_episode_model_from_jsonl(paths.episode_data)
    return LearningWriteResult(
        kind="episode",
        data_path=paths.episode_data,
        model_path=None,
        example_count=model.example_count,
        pattern_count=len(model.patterns),
    )


def save_memory_knowledge_feedback(
    question: str,
    query: Query,
    answer: str,
    paths: LearningPaths,
    *,
    source: str = "human_feedback",
) -> KnowledgeWriteResult:
    append_memory_knowledge_entry(
        paths.memory_knowledge_data,
        MemoryKnowledgeEntry(
            question=question,
            query=query,
            answer=answer,
            source=source,
        ),
    )
    model = compile_memory_knowledge_model_from_jsonl(paths.memory_knowledge_data)
    save_memory_knowledge_model(model, paths.memory_knowledge_model)
    return KnowledgeWriteResult(
        data_path=paths.memory_knowledge_data,
        model_path=paths.memory_knowledge_model,
        example_count=model.example_count,
        pattern_count=len(model.patterns),
    )
