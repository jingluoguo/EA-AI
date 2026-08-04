from __future__ import annotations

from .capabilities import CognitiveCapabilities
from ..errors import ParseError
from .inference import expand_conditionals
from .query_learning import query_candidate_is_learned_unit, resolve_query_candidates
from ..reference_resolution import resolve_references
from .state_engine import materialize_events, materialize_relations
from .structure_helpers import dedupe_entities, with_time
from ..structure import Entity, Frame, State, Structure
from .text_processing import is_query_like_fragment, split_query_candidate, split_sentences


def parse_text_with_capabilities(text: str, capabilities: CognitiveCapabilities) -> Structure:
    entities: list[Entity] = []
    frames: list[Frame] = []
    states: list[State] = []
    query_candidates: list[str] = []
    next_time = 1

    def add_extracted(extracted) -> None:
        nonlocal next_time
        new_entities, new_frames = extracted
        entities.extend(new_entities)
        for frame in new_frames:
            timed_frame = with_time(frame, next_time)
            next_time += 1
            frames.append(timed_frame)
            for state in capabilities.states_from_frame(timed_frame):
                capabilities.apply_state(states, state)

    for sentence, is_question in split_sentences(text):
        raw_sentence = sentence
        resolved_sentence = resolve_references(raw_sentence, dedupe_entities(entities))
        extracted = (
            None
            if is_question or is_query_like_fragment(resolved_sentence)
            else capabilities.parse_statement(resolved_sentence)
        )
        if extracted is not None:
            add_extracted(extracted)
            continue

        if is_question and query_candidate_is_learned_unit(
            resolved_sentence,
            dedupe_entities(entities),
            capabilities.query_parsers,
        ):
            query_candidates.append(resolved_sentence)
            continue

        for fragment in split_query_candidate(raw_sentence):
            resolved_fragment = resolve_references(fragment, dedupe_entities(entities))
            fragment_extracted = (
                None
                if is_query_like_fragment(resolved_fragment)
                else capabilities.parse_statement(resolved_fragment)
            )
            if fragment_extracted is not None:
                add_extracted(fragment_extracted)
            else:
                query_candidates.append(resolved_fragment)

    deduped_entities = dedupe_entities(entities)
    query = resolve_query_candidates(query_candidates, deduped_entities, capabilities.query_parsers)
    if not entities and not states and not frames and query is None:
        raise ParseError(f"Cannot extract structure from text: {text}")
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
    intentions = capabilities.analyze_intentions(text, structure)
    structure = Structure(
        entities=structure.entities,
        relations=structure.relations,
        events=structure.events,
        rules=(),
        query=structure.query,
        frames=structure.frames,
        states=structure.states,
        intentions=intentions,
    )
    return Structure(
        entities=structure.entities,
        relations=structure.relations,
        events=structure.events,
        rules=capabilities.infer_rules(structure),
        query=structure.query,
        frames=structure.frames,
        states=structure.states,
        intentions=structure.intentions,
    )
