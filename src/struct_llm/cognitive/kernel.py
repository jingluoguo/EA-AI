from __future__ import annotations

from .capabilities import CognitiveCapabilities
from ..errors import ParseError
from .frame_parser import dedupe_entities, with_time
from .inference import expand_conditionals
from .query_parser import parse_query_candidates
from ..reference_resolution import resolve_references
from .state_engine import materialize_events, materialize_relations
from ..structure import Entity, Frame, State, Structure
from .text_processing import split_sentences


def parse_text_with_capabilities(text: str, capabilities: CognitiveCapabilities) -> Structure:
    entities: list[Entity] = []
    frames: list[Frame] = []
    states: list[State] = []
    query_candidates: list[str] = []
    next_time = 1

    for sentence, is_question in split_sentences(text):
        sentence = resolve_references(sentence, dedupe_entities(entities))
        if is_question:
            query_candidates.append(sentence)
            continue

        extracted = capabilities.parse_statement(sentence)
        if extracted is None:
            query_candidates.append(sentence)
            continue

        new_entities, new_frames = extracted
        entities.extend(new_entities)
        for frame in new_frames:
            timed_frame = with_time(frame, next_time)
            next_time += 1
            frames.append(timed_frame)
            for state in capabilities.states_from_frame(timed_frame):
                capabilities.apply_state(states, state)

    if not entities and not states and not frames:
        raise ParseError(f"Cannot extract structure from text: {text}")

    deduped_entities = dedupe_entities(entities)
    query = parse_query_candidates(query_candidates, deduped_entities, capabilities.query_parsers)
    structure = Structure(
        entities=deduped_entities,
        relations=materialize_relations(states),
        events=materialize_events(frames),
        rules=(),
        query=query,
        frames=tuple(frames),
        states=tuple(states),
    )
    structure = expand_conditionals(structure, capabilities)
    return Structure(
        entities=structure.entities,
        relations=structure.relations,
        events=structure.events,
        rules=capabilities.infer_rules(structure),
        query=structure.query,
        frames=structure.frames,
        states=structure.states,
    )
