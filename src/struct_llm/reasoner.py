from __future__ import annotations

from dataclasses import dataclass

from .capabilities import StructuralCapabilities
from .errors import ParseError
from .frame_parser import DEFAULT_STATEMENT_PARSERS, dedupe_entities, with_time
from .inference import DEFAULT_ANSWERERS, DEFAULT_RULE_INFERERS, answer_from_structure
from .query_parser import DEFAULT_QUERY_PARSERS, parse_query_candidates
from .state_engine import (
    DEFAULT_STATE_PROJECTORS,
    DEFAULT_STATE_REDUCERS,
    materialize_events,
    materialize_relations,
)
from .structure import Entity, Frame, State, Structure
from .text_processing import split_sentences


@dataclass(frozen=True)
class Prediction:
    structure: Structure
    answer: str


def default_capabilities() -> StructuralCapabilities:
    return StructuralCapabilities(
        statement_parsers=DEFAULT_STATEMENT_PARSERS,
        state_projectors=DEFAULT_STATE_PROJECTORS,
        state_reducers=DEFAULT_STATE_REDUCERS,
        query_parsers=DEFAULT_QUERY_PARSERS,
        rule_inferers=DEFAULT_RULE_INFERERS,
        answerers=DEFAULT_ANSWERERS,
    )


def parse_text(text: str, capabilities: StructuralCapabilities | None = None) -> Structure:
    active_capabilities = capabilities or default_capabilities()
    entities: list[Entity] = []
    frames: list[Frame] = []
    states: list[State] = []
    query_candidates: list[str] = []
    next_time = 1

    for sentence, is_question in split_sentences(text):
        if is_question:
            query_candidates.append(sentence)
            continue

        extracted = active_capabilities.parse_statement(sentence)
        if extracted is None:
            query_candidates.append(sentence)
            continue

        new_entities, new_frames = extracted
        entities.extend(new_entities)
        for frame in new_frames:
            timed_frame = with_time(frame, next_time)
            next_time += 1
            frames.append(timed_frame)
            for state in active_capabilities.states_from_frame(timed_frame):
                active_capabilities.apply_state(states, state)

    if not entities and not states and not frames:
        raise ParseError(f"Cannot extract structure from text: {text}")

    deduped_entities = dedupe_entities(entities)
    query = parse_query_candidates(query_candidates, deduped_entities, active_capabilities.query_parsers)
    structure = Structure(
        entities=deduped_entities,
        relations=materialize_relations(states),
        events=materialize_events(frames),
        rules=(),
        query=query,
        frames=tuple(frames),
        states=tuple(states),
    )
    return Structure(
        entities=structure.entities,
        relations=structure.relations,
        events=structure.events,
        rules=active_capabilities.infer_rules(structure),
        query=structure.query,
        frames=structure.frames,
        states=structure.states,
    )


def predict(text: str, capabilities: StructuralCapabilities | None = None) -> Prediction:
    active_capabilities = capabilities or default_capabilities()
    structure = parse_text(text, active_capabilities)
    return Prediction(
        structure=structure,
        answer=answer_from_structure(structure, active_capabilities.answerers),
    )
