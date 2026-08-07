from __future__ import annotations

from dataclasses import dataclass

from .capabilities import CognitiveCapabilities
from .comprehension.query import query_candidate_is_learned_unit, resolve_query_candidate, resolve_query_candidates
from .comprehension.structure_helpers import dedupe_entities, with_time
from .errors import ParseError
from .motor.dialogue import default_learned_dialog_answerer
from .neural import NeuralBoundaryModel, configured_neural_boundary_model, with_neural_boundary
from .neural.query_classifier import default_neural_query_parser
from .neural.statement_classifier import default_neural_statement_parser
from .memory.long_term import default_memory_states, memory_entities_from_states
from .perception.lexer import split_query_candidate, split_sentences
from .perception.reference import resolve_references
from .reasoning.pipeline import (
    DEFAULT_ANSWERERS,
    DEFAULT_RULE_INFERERS,
    answer_from_structure,
)
from .structure import Entity, Frame, Query, State, Structure
from .world.causal import expand_conditionals
from .world.state import (
    DEFAULT_STATE_PROJECTORS,
    DEFAULT_STATE_REDUCERS,
    materialize_events,
    materialize_relations,
)


@dataclass(frozen=True)
class Prediction:
    structure: Structure
    answer: str


def default_capabilities(
    neural_model: NeuralBoundaryModel | None = None,
    *,
    neural_answer_priority: str = "first",
    use_environment: bool = True,
    use_memory: bool = True,
) -> CognitiveCapabilities:
    capabilities = CognitiveCapabilities(
        statement_parsers=(default_neural_statement_parser(),),
        state_projectors=DEFAULT_STATE_PROJECTORS,
        state_reducers=DEFAULT_STATE_REDUCERS,
        query_parsers=(default_neural_query_parser(),),
        rule_inferers=DEFAULT_RULE_INFERERS,
        answerers=(*DEFAULT_ANSWERERS, default_learned_dialog_answerer()),
    )
    if use_memory:
        memory_states = default_memory_states()
        if memory_states:
            capabilities = capabilities.with_memory_states(*memory_states)
    if neural_model is None and use_environment:
        neural_model = configured_neural_boundary_model()
    if neural_model is None:
        return capabilities
    return with_neural_boundary(
        capabilities,
        neural_model,
        answer_priority=neural_answer_priority,
    )


def parse_text_with_capabilities(text: str, capabilities: CognitiveCapabilities) -> Structure:
    entities: list[Entity] = list(memory_entities_from_states(capabilities.memory_states))
    frames: list[Frame] = []
    states: list[State] = list(capabilities.memory_states)
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

        if not is_question:
            sentence_extracted = capabilities.parse_statement(resolved_sentence)
            if sentence_extracted is not None:
                sentence_query = resolve_query_candidate(resolved_sentence, dedupe_entities(entities), capabilities.query_parsers)
                if sentence_query is not None and profile_statement_should_yield_to_query(
                    sentence_extracted,
                    sentence_query,
                ) and query_candidate_is_learned_unit(
                    resolved_sentence,
                    dedupe_entities(entities),
                    capabilities.query_parsers,
                ):
                    query_candidates.append(resolved_sentence)
                    continue
                add_extracted(sentence_extracted)
                continue

        if query_candidate_is_learned_unit(
            resolved_sentence,
            dedupe_entities(entities),
            capabilities.query_parsers,
        ):
            query_candidates.append(resolved_sentence)
            continue

        for fragment in split_query_candidate(raw_sentence):
            resolved_fragment = resolve_references(fragment, dedupe_entities(entities))
            if is_question:
                if resolve_query_candidate(resolved_fragment, dedupe_entities(entities), capabilities.query_parsers) is not None:
                    query_candidates.append(resolved_fragment)
                    continue

                fragment_extracted = capabilities.parse_statement(resolved_fragment)
                if fragment_extracted is not None:
                    add_extracted(fragment_extracted)
                else:
                    query_candidates.append(resolved_fragment)
            else:
                fragment_extracted = capabilities.parse_statement(resolved_fragment)
                if fragment_extracted is not None:
                    add_extracted(fragment_extracted)
                    continue

                if resolve_query_candidate(resolved_fragment, dedupe_entities(entities), capabilities.query_parsers) is not None:
                    query_candidates.append(resolved_fragment)
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


def profile_statement_should_yield_to_query(
    extracted: tuple[list[Entity], list[Frame]],
    query: Query,
) -> bool:
    if query.intent != "profile":
        return False
    statement_attributes = profile_statement_attributes(extracted)
    if not statement_attributes:
        return False
    query_attribute = profile_query_attribute(query)
    if query_attribute and query_attribute not in statement_attributes:
        return True
    return profile_statement_value_is_under_specified(extracted)


def profile_statement_attributes(extracted: tuple[list[Entity], list[Frame]]) -> set[str]:
    _, frames = extracted
    attributes: set[str] = set()
    for frame in frames:
        if frame.frame_type == "profile_name":
            attributes.add("name")
        elif frame.frame_type == "profile_like":
            attributes.add("likes")
        elif frame.frame_type == "profile_dislike":
            attributes.add("dislikes")
    return attributes


def profile_query_attribute(query: Query) -> str | None:
    for qualifier in query.qualifiers:
        if qualifier.startswith("attribute="):
            return qualifier.split("=", 1)[1]
    return None


def profile_statement_value_is_under_specified(extracted: tuple[list[Entity], list[Frame]]) -> bool:
    entities, _ = extracted
    profile_values = [entity.name for entity in entities if entity.role == "profile_value"]
    return bool(profile_values) and all(len(value.strip()) <= 1 for value in profile_values)


def parse_text(text: str, capabilities: CognitiveCapabilities | None = None) -> Structure:
    active_capabilities = capabilities or default_capabilities()
    return parse_text_with_capabilities(text, active_capabilities)


def predict(text: str, capabilities: CognitiveCapabilities | None = None) -> Prediction:
    active_capabilities = capabilities or default_capabilities()
    structure = parse_text_with_capabilities(text, active_capabilities)
    answer = answer_from_structure(structure, active_capabilities.answerers)
    return Prediction(
        structure=structure,
        answer=answer,
    )
