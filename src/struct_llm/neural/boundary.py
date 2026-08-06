from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import os
from typing import Any, Callable, Mapping, Optional, Protocol

from ..capabilities import CognitiveCapabilities, StatementParseResult
from ..comprehension.query import query_from_dict
from ..perception.normalizer import normalize_slot_value
from ..structure import Entity, Frame, Intention, Query, Relation, Role, State, Structure


NeuralPayload = Mapping[str, Any]
NeuralResult = Optional[Mapping[str, Any]]
NeuralResponder = Callable[[NeuralPayload], NeuralResult]
NEURAL_PROVIDER_ENV = "EA_AI_NEURAL_PROVIDER"


class NeuralBoundaryModel(Protocol):
    """Small boundary that keeps neural inference outside the cognitive kernel."""

    def predict(self, task: str, payload: NeuralPayload) -> NeuralResult:
        ...


@dataclass(frozen=True)
class InMemoryNeuralBoundaryModel:
    """Test and prototyping model that supplies structured neural task responses."""

    responders: Mapping[str, NeuralResponder]

    def predict(self, task: str, payload: NeuralPayload) -> NeuralResult:
        responder = self.responders.get(task)
        if responder is None:
            return None
        return responder(payload)


@dataclass(frozen=True)
class NeuralQueryParser:
    model: NeuralBoundaryModel
    min_confidence: float = 0.75

    def __call__(self, sentence: str, entities: tuple[Entity, ...]) -> Query | None:
        result = self.model.predict(
            "parse_query",
            {
                "sentence": sentence,
                "entities": [entity_to_dict(entity) for entity in entities],
            },
        )
        if not confident_enough(result, self.min_confidence):
            return None
        raw_query = result.get("query") if result is not None else None
        if not isinstance(raw_query, dict):
            return None
        try:
            return query_from_dict(raw_query, "Neural query")
        except ValueError:
            return None


@dataclass(frozen=True)
class NeuralStatementParser:
    model: NeuralBoundaryModel
    min_confidence: float = 0.75

    def __call__(self, sentence: str) -> StatementParseResult | None:
        result = self.model.predict("parse_statement", {"sentence": sentence})
        if not confident_enough(result, self.min_confidence):
            return None
        if result is None:
            return None
        entities = entities_from_result(result.get("entities"))
        frames = frames_from_result(result.get("frames"))
        if entities is None or frames is None or not frames:
            return None
        return list(entities), list(frames)


@dataclass(frozen=True)
class NeuralIntentAnalyzer:
    model: NeuralBoundaryModel
    min_confidence: float = 0.55

    def __call__(self, text: str, structure: Structure) -> tuple[Intention, ...]:
        result = self.model.predict(
            "analyze_intent",
            {
                "text": text,
                "structure": structure_to_dict(structure),
            },
        )
        if not confident_enough(result, self.min_confidence):
            return ()
        raw_intentions = result.get("intentions") if result is not None else None
        if not isinstance(raw_intentions, list):
            return ()
        intentions = tuple(
            intention
            for value in raw_intentions
            for intention in (intention_from_dict(value),)
            if intention is not None and intention.confidence >= self.min_confidence
        )
        return intentions[:3]


@dataclass(frozen=True)
class NeuralAnswerer:
    model: NeuralBoundaryModel
    min_confidence: float = 0.65

    def __call__(self, structure: Structure) -> str | None:
        result = self.model.predict("answer", {"structure": structure_to_dict(structure)})
        if not confident_enough(result, self.min_confidence):
            return None
        answer = str(result.get("answer") or "").strip() if result is not None else ""
        return answer or None


def with_neural_boundary(
    capabilities: CognitiveCapabilities,
    model: NeuralBoundaryModel,
    *,
    input_first: bool = True,
    answer_priority: str = "fallback",
    query_min_confidence: float = 0.75,
    statement_min_confidence: float = 0.75,
    intent_min_confidence: float = 0.55,
    answer_min_confidence: float = 0.65,
) -> CognitiveCapabilities:
    query_parser = NeuralQueryParser(model, min_confidence=query_min_confidence)
    statement_parser = NeuralStatementParser(model, min_confidence=statement_min_confidence)
    intent_analyzer = NeuralIntentAnalyzer(model, min_confidence=intent_min_confidence)
    answerer = NeuralAnswerer(model, min_confidence=answer_min_confidence)

    statement_parsers = (
        (statement_parser, *capabilities.statement_parsers)
        if input_first
        else (*capabilities.statement_parsers, statement_parser)
    )
    query_parsers = (
        (query_parser, *capabilities.query_parsers)
        if input_first
        else (*capabilities.query_parsers, query_parser)
    )
    if answer_priority not in {"fallback", "first"}:
        raise ValueError("answer_priority must be 'fallback' or 'first'.")
    answerers = (
        (answerer, *capabilities.answerers)
        if answer_priority == "first"
        else (*capabilities.answerers, answerer)
    )
    intent_analyzers = (
        (intent_analyzer, *capabilities.intent_analyzers)
        if input_first
        else (*capabilities.intent_analyzers, intent_analyzer)
    )
    return CognitiveCapabilities(
        statement_parsers=statement_parsers,
        state_projectors=capabilities.state_projectors,
        state_reducers=capabilities.state_reducers,
        query_parsers=query_parsers,
        rule_inferers=capabilities.rule_inferers,
        answerers=answerers,
        intent_analyzers=intent_analyzers,
    )


def load_neural_boundary_model(spec: str) -> NeuralBoundaryModel:
    module_name, separator, factory_name = spec.partition(":")
    if not module_name or not separator or not factory_name:
        raise ValueError("Neural provider must look like 'module:function'.")
    module = import_module(module_name)
    factory = getattr(module, factory_name)
    model = factory()
    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise ValueError("Neural provider factory must return an object with predict(task, payload).")
    return model


def configured_neural_boundary_model() -> NeuralBoundaryModel | None:
    provider = os.getenv(NEURAL_PROVIDER_ENV, "").strip()
    if not provider:
        return None
    return load_neural_boundary_model(provider)


def confident_enough(result: NeuralResult, min_confidence: float) -> bool:
    if result is None:
        return False
    confidence = result.get("confidence", 1.0)
    try:
        return float(confidence) >= min_confidence
    except (TypeError, ValueError):
        return False


def entities_from_result(raw_entities: Any) -> tuple[Entity, ...] | None:
    if not isinstance(raw_entities, list):
        return None
    entities: list[Entity] = []
    for record in raw_entities:
        if not isinstance(record, dict):
            return None
        role = str(record.get("role") or "").strip()
        name = normalize_slot_value(str(record.get("name") or ""))
        if not role or not name:
            return None
        entities.append(Entity(role, name))
    return tuple(entities)


def frames_from_result(raw_frames: Any) -> tuple[Frame, ...] | None:
    if not isinstance(raw_frames, list):
        return None
    frames: list[Frame] = []
    for record in raw_frames:
        frame = frame_from_dict(record)
        if frame is None:
            return None
        frames.append(frame)
    return tuple(frames)


def frame_from_dict(record: Any) -> Frame | None:
    if not isinstance(record, dict):
        return None
    frame_type = str(record.get("frame_type") or "").strip()
    raw_roles = record.get("roles")
    if not frame_type or not isinstance(raw_roles, dict):
        return None
    frame_id = "pending"
    roles: list[Role] = []
    for name, value in raw_roles.items():
        role_name = str(name).strip()
        role_value = normalize_slot_value(str(value or ""))
        if not role_name or not role_value:
            return None
        roles.append(Role(frame_id, role_name, role_value))
    return Frame(frame_id, frame_type, 0, tuple(roles))


def intention_from_dict(record: Any) -> Intention | None:
    if not isinstance(record, dict):
        return None
    subject = str(record.get("subject") or "").strip()
    goal = str(record.get("goal") or "").strip()
    if not subject or not goal:
        return None
    try:
        confidence = float(record.get("confidence", 1.0))
    except (TypeError, ValueError):
        return None
    if confidence < 0 or confidence > 1:
        return None
    return Intention(
        subject=subject,
        goal=goal,
        belief=str(record.get("belief") or "").strip(),
        strategy=str(record.get("strategy") or "").strip(),
        evidence=str(record.get("evidence") or "").strip(),
        confidence=confidence,
        source=str(record.get("source") or "neural").strip() or "neural",
    )


def structure_to_dict(structure: Structure) -> dict[str, Any]:
    return {
        "entities": [entity_to_dict(entity) for entity in structure.entities],
        "relations": [relation_to_dict(relation) for relation in structure.relations],
        "events": [event_to_dict(event) for event in structure.events],
        "rules": list(structure.rules),
        "query": query_to_payload(structure.query) if structure.query is not None else None,
        "frames": [frame_to_dict(frame) for frame in structure.frames],
        "states": [state_to_dict(state) for state in structure.states],
        "intentions": [intention_to_dict(intention) for intention in structure.intentions],
        "linearized": structure.linearize(),
    }


def entity_to_dict(entity: Entity) -> dict[str, str]:
    return {"role": entity.role, "name": entity.name}


def relation_to_dict(relation: Relation) -> dict[str, str]:
    return {"name": relation.name, "left": relation.left, "right": relation.right}


def state_to_dict(state: State) -> dict[str, str]:
    record = {"name": state.name, "left": state.left, "right": state.right}
    if state.source is not None:
        record["source"] = state.source
    return record


def event_to_dict(event) -> dict[str, Any]:
    return {
        "name": event.name,
        "actor": event.actor,
        "target": event.target,
        "qualifiers": list(event.qualifiers),
    }


def frame_to_dict(frame: Frame) -> dict[str, Any]:
    return {
        "frame_id": frame.frame_id,
        "frame_type": frame.frame_type,
        "time": frame.time,
        "roles": {role.name: role.value for role in frame.roles},
    }


def query_to_payload(query: Query) -> dict[str, Any]:
    return {
        "intent": query.intent,
        "target": query.target,
        "qualifiers": list(query.qualifiers),
        "subqueries": [query_to_payload(subquery) for subquery in query.subqueries],
    }


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
