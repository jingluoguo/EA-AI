from __future__ import annotations

"""Local neural boundary example.

This module is a small adapter you can import with:

    uv run struct-ask --neural-provider my_neural:make_model "你是谁？"

It does not host a real neural model yet. Instead, it wraps the existing
structural capabilities so the runtime path is executable end-to-end and can
later be swapped with a real model call without changing the CLI contract.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from struct_llm.capabilities import CognitiveCapabilities
from struct_llm.comprehension.intent import InMemoryIntentAnalyzer
from struct_llm.comprehension.intent import evaluate_intent_analyzer, load_intent_jsonl
from struct_llm.comprehension.query import (
    QUERY_DATA_PATH,
    LearnedQueryParser,
    evaluate_query_parser,
    load_query_jsonl,
    query_from_dict,
)
from struct_llm.comprehension.statement import (
    STATEMENT_DATA_PATH,
    LearnedStatementParser,
    evaluate_statement_parser,
    load_statement_jsonl,
)
from struct_llm.kernel import default_capabilities
from struct_llm.motor.dialogue import (
    DIALOG_ANSWER_DATA_PATH,
    LearnedDialogActAnswerer,
    compile_dialog_answer_model_from_jsonl,
)
from struct_llm.structure import Entity, Event, Frame, Intention, Query, Relation, Role, State, Structure


TRAINING_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass(frozen=True)
class LocalNeuralBoundaryModel:
    """Template model that bridges the neural interface to existing capabilities."""

    _capabilities: CognitiveCapabilities = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Train once from the repo's current JSONL data so this provider follows the
        # same examples the rest of the project already uses.
        object.__setattr__(self, "_capabilities", build_trained_capabilities())

    def predict(self, task: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if task == "parse_query":
            return self._parse_query(payload)
        if task == "parse_statement":
            return self._parse_statement(payload)
        if task == "analyze_intent":
            return self._analyze_intent(payload)
        if task == "answer":
            return self._answer(payload)
        return None

    def _parse_query(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        sentence = str(payload.get("sentence") or "").strip()
        entities = tuple(entity_from_dict(value) for value in payload.get("entities", ()) if isinstance(value, dict))
        query = self._capabilities.parse_query(sentence, entities)
        if query is None:
            return None
        return {"confidence": 0.99, "query": query_to_dict(query)}

    def _parse_statement(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        sentence = str(payload.get("sentence") or "").strip()
        parsed = self._capabilities.parse_statement(sentence)
        if parsed is None:
            return None
        entities, frames = parsed
        return {
            "confidence": 0.99,
            "entities": [entity_to_dict(entity) for entity in entities],
            "frames": [frame_to_dict(frame) for frame in frames],
        }

    def _analyze_intent(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        text = str(payload.get("text") or "").strip()
        structure = structure_from_dict(payload.get("structure"))
        if structure is None:
            return None
        intentions = self._capabilities.analyze_intentions(text, structure)
        if not intentions:
            return None
        return {
            "confidence": max(intention.confidence for intention in intentions),
            "intentions": [intention_to_dict(intention) for intention in intentions],
        }

    def _answer(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        structure = structure_from_dict(payload.get("structure"))
        if structure is None:
            return None
        answer = self._capabilities.answer(structure)
        if answer is None:
            return None
        return {"confidence": 0.99, "answer": answer}


def make_model() -> LocalNeuralBoundaryModel:
    # CLI entry point expected by `--neural-provider module:function`.
    return LocalNeuralBoundaryModel()


def train() -> None:
    # Rebuild the provider from the current repo datasets and report what the
    # model learned from them. This keeps the "training" path explicit instead
    # of hiding it behind import side effects.
    model = make_model()
    summary = train_summary(model)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_trained_capabilities() -> CognitiveCapabilities:
    # Start from the structural kernel, then replace the learnable pieces with
    # versions trained directly from the current data files.
    capabilities = default_capabilities(use_environment=False, use_memory=False)
    if STATEMENT_DATA_PATH.exists():
        capabilities = capabilities.replace_statement_parsers(
            LearnedStatementParser.from_jsonl(STATEMENT_DATA_PATH)
        )
    if QUERY_DATA_PATH.exists():
        capabilities = capabilities.replace_query_parsers(
            LearnedQueryParser.from_jsonl(QUERY_DATA_PATH)
        )
    intent_data_path = TRAINING_DATA_DIR / "intent_examples.jsonl"
    if intent_data_path.exists():
        capabilities = capabilities.with_intent_analyzers(
            InMemoryIntentAnalyzer.from_jsonl(intent_data_path)
        )
    if DIALOG_ANSWER_DATA_PATH.exists():
        capabilities = capabilities.with_answerers(
            LearnedDialogActAnswerer.from_jsonl(DIALOG_ANSWER_DATA_PATH)
        )
    return capabilities


def train_summary(model: LocalNeuralBoundaryModel) -> dict[str, Any]:
    capabilities = model._capabilities
    summary: dict[str, Any] = {}

    if QUERY_DATA_PATH.exists():
        query_examples = load_query_jsonl(QUERY_DATA_PATH)
        query_parser = capabilities.query_parsers[0]
        query_result = evaluate_query_parser(query_parser, query_examples)
        summary["query"] = {
            "examples": query_result.total,
            "matched": query_result.matched,
            "accuracy": round(query_result.accuracy, 4),
        }

    if STATEMENT_DATA_PATH.exists():
        statement_examples = load_statement_jsonl(STATEMENT_DATA_PATH)
        statement_parser = capabilities.statement_parsers[0]
        statement_result = evaluate_statement_parser(statement_parser, statement_examples)
        summary["statement"] = {
            "examples": statement_result.total,
            "matched": statement_result.matched,
            "accuracy": round(statement_result.accuracy, 4),
        }

    intent_data_path = TRAINING_DATA_DIR / "intent_examples.jsonl"
    if intent_data_path.exists():
        intent_examples = load_intent_jsonl(intent_data_path)
        intent_analyzer = capabilities.intent_analyzers[0] if capabilities.intent_analyzers else InMemoryIntentAnalyzer()
        intent_result = evaluate_intent_analyzer(intent_analyzer, intent_examples)
        summary["intent"] = {
            "examples": intent_result.total,
            "matched": intent_result.matched,
            "accuracy": round(intent_result.accuracy, 4),
        }

    if DIALOG_ANSWER_DATA_PATH.exists():
        dialog_model = compile_dialog_answer_model_from_jsonl(DIALOG_ANSWER_DATA_PATH)
        summary["dialog_answer"] = {
            "examples": dialog_model.example_count,
            "patterns": len(dialog_model.patterns),
        }

    return summary


def entity_from_dict(record: dict[str, Any]) -> Entity:
    return Entity(role=str(record.get("role") or "").strip(), name=str(record.get("name") or "").strip())


def entity_to_dict(entity: Entity) -> dict[str, str]:
    return {"role": entity.role, "name": entity.name}


def relation_from_dict(record: dict[str, Any]) -> Relation:
    return Relation(
        name=str(record.get("name") or "").strip(),
        left=str(record.get("left") or "").strip(),
        right=str(record.get("right") or "").strip(),
    )


def relation_to_dict(relation: Relation) -> dict[str, str]:
    return {"name": relation.name, "left": relation.left, "right": relation.right}


def state_from_dict(record: dict[str, Any]) -> State:
    return State(
        name=str(record.get("name") or "").strip(),
        left=str(record.get("left") or "").strip(),
        right=str(record.get("right") or "").strip(),
        source=str(record.get("source") or "").strip() or None,
    )


def state_to_dict(state: State) -> dict[str, str]:
    record = {"name": state.name, "left": state.left, "right": state.right}
    if state.source is not None:
        record["source"] = state.source
    return record


def intention_from_dict(record: dict[str, Any]) -> Intention:
    return Intention(
        subject=str(record.get("subject") or "").strip(),
        goal=str(record.get("goal") or "").strip(),
        belief=str(record.get("belief") or "").strip(),
        strategy=str(record.get("strategy") or "").strip(),
        evidence=str(record.get("evidence") or "").strip(),
        confidence=float(record.get("confidence") or 1.0),
        source=str(record.get("source") or "neural").strip() or "neural",
    )


def intention_to_dict(intention: Intention) -> dict[str, Any]:
    return {
        "subject": intention.subject,
        "goal": intention.goal,
        "belief": intention.belief,
        "strategy": intention.strategy,
        "evidence": intention.evidence,
        "confidence": intention.confidence,
        "source": intention.source,
    }


def frame_from_dict(record: dict[str, Any]) -> Frame:
    frame_id = str(record.get("frame_id") or "pending").strip() or "pending"
    raw_roles = record.get("roles") or {}
    roles = tuple(Role(frame_id, str(name), str(value)) for name, value in raw_roles.items())
    return Frame(
        frame_id=frame_id,
        frame_type=str(record.get("frame_type") or "").strip(),
        time=int(record.get("time") or 0),
        roles=roles,
    )


def frame_to_dict(frame: Frame) -> dict[str, Any]:
    return {
        "frame_id": frame.frame_id,
        "frame_type": frame.frame_type,
        "time": frame.time,
        "roles": {role.name: role.value for role in frame.roles},
    }


def event_from_dict(record: dict[str, Any]) -> Event:
    return Event(
        name=str(record.get("name") or "").strip(),
        actor=str(record.get("actor") or "").strip(),
        target=str(record.get("target") or "").strip(),
        qualifiers=tuple(str(value).strip() for value in record.get("qualifiers", ()) if str(value).strip()),
    )


def event_to_dict(event: Event) -> dict[str, Any]:
    return {"name": event.name, "actor": event.actor, "target": event.target, "qualifiers": list(event.qualifiers)}


def query_from_payload(record: dict[str, Any]) -> Query:
    return query_from_dict(
        {
            "intent": str(record.get("intent") or "").strip(),
            "target": str(record.get("target") or "").strip(),
            "qualifiers": list(record.get("qualifiers") or ()),
            "subqueries": list(record.get("subqueries") or ()),
        },
        "Neural provider",
    )


def query_to_dict(query: Query) -> dict[str, Any]:
    return {
        "intent": query.intent,
        "target": query.target,
        "qualifiers": list(query.qualifiers),
        "subqueries": [query_to_dict(subquery) for subquery in query.subqueries],
    }


def structure_from_dict(record: Any) -> Structure | None:
    if not isinstance(record, dict):
        return None
    try:
        return Structure(
            entities=tuple(entity_from_dict(value) for value in record.get("entities", ()) if isinstance(value, dict)),
            relations=tuple(relation_from_dict(value) for value in record.get("relations", ()) if isinstance(value, dict)),
            events=tuple(event_from_dict(value) for value in record.get("events", ()) if isinstance(value, dict)),
            rules=tuple(str(value) for value in record.get("rules", ()) if str(value)),
            query=query_from_payload(record["query"]) if isinstance(record.get("query"), dict) else None,
            frames=tuple(frame_from_dict(value) for value in record.get("frames", ()) if isinstance(value, dict)),
            states=tuple(state_from_dict(value) for value in record.get("states", ()) if isinstance(value, dict)),
            intentions=tuple(intention_from_dict(value) for value in record.get("intentions", ()) if isinstance(value, dict)),
        )
    except (TypeError, ValueError):
        return None
