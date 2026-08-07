from __future__ import annotations

from dataclasses import dataclass, replace

from .capabilities import CognitiveCapabilities
from .comprehension.query import query_candidate_is_learned_unit, resolve_query_candidate, resolve_query_candidates
from .comprehension.structure_helpers import dedupe_entities, with_time
from .errors import ParseError
from .memory.long_term import memory_entities_from_states
from .perception.lexer import split_query_candidate, split_sentences
from .perception.reference import resolve_references
from .structure import Entity, Frame, Query, State, Structure
from .world.causal import expand_conditionals
from .world.state import materialize_events, materialize_relations


@dataclass
class ParseContext:
    entities: list[Entity]
    frames: list[Frame]
    states: list[State]
    query_candidates: list[str]
    next_time: int = 1

    def known_entities(self) -> tuple[Entity, ...]:
        return dedupe_entities(self.entities)


def initial_parse_context(capabilities: CognitiveCapabilities) -> ParseContext:
    return ParseContext(
        entities=list(memory_entities_from_states(capabilities.memory_states)),
        frames=[],
        states=list(capabilities.memory_states),
        query_candidates=[],
    )


def ingest_sentence(
    sentence: str,
    is_question: bool,
    context: ParseContext,
    capabilities: CognitiveCapabilities,
) -> None:
    resolved_sentence = resolve_references(sentence, context.known_entities())

    if not is_question and ingest_statement_sentence(resolved_sentence, context, capabilities):
        return

    if query_candidate_is_learned_unit(
        resolved_sentence,
        context.known_entities(),
        capabilities.query_parsers,
    ):
        context.query_candidates.append(resolved_sentence)
        return

    for fragment in split_query_candidate(sentence):
        ingest_query_fragment(fragment, is_question, context, capabilities)


def ingest_statement_sentence(
    sentence: str,
    context: ParseContext,
    capabilities: CognitiveCapabilities,
) -> bool:
    extracted = capabilities.parse_statement(sentence)
    if extracted is None:
        return False

    sentence_query = resolve_query_candidate(sentence, context.known_entities(), capabilities.query_parsers)
    if sentence_query is not None and profile_statement_should_yield_to_query(
        extracted,
        sentence_query,
    ) and query_candidate_is_learned_unit(
        sentence,
        context.known_entities(),
        capabilities.query_parsers,
    ):
        context.query_candidates.append(sentence)
        return True

    add_extracted_structure(extracted, context, capabilities)
    return True


def ingest_query_fragment(
    fragment: str,
    is_question: bool,
    context: ParseContext,
    capabilities: CognitiveCapabilities,
) -> None:
    resolved_fragment = resolve_references(fragment, context.known_entities())
    if is_question:
        if resolve_query_candidate(resolved_fragment, context.known_entities(), capabilities.query_parsers) is not None:
            context.query_candidates.append(resolved_fragment)
            return

        extracted = capabilities.parse_statement(resolved_fragment)
        if extracted is not None:
            add_extracted_structure(extracted, context, capabilities)
        else:
            context.query_candidates.append(resolved_fragment)
        return

    extracted = capabilities.parse_statement(resolved_fragment)
    if extracted is not None:
        add_extracted_structure(extracted, context, capabilities)
        return

    if resolve_query_candidate(resolved_fragment, context.known_entities(), capabilities.query_parsers) is not None:
        context.query_candidates.append(resolved_fragment)
    else:
        context.query_candidates.append(resolved_fragment)


def add_extracted_structure(
    extracted: tuple[list[Entity], list[Frame]],
    context: ParseContext,
    capabilities: CognitiveCapabilities,
) -> None:
    new_entities, new_frames = extracted
    context.entities.extend(new_entities)
    for frame in new_frames:
        timed_frame = with_time(frame, context.next_time)
        context.next_time += 1
        context.frames.append(timed_frame)
        for state in capabilities.states_from_frame(timed_frame):
            capabilities.apply_state(context.states, state)


def finalize_parse_context(
    text: str,
    context: ParseContext,
    capabilities: CognitiveCapabilities,
) -> Structure:
    deduped_entities = context.known_entities()
    query = resolve_query_candidates(context.query_candidates, deduped_entities, capabilities.query_parsers)
    if not context.entities and not context.states and not context.frames and query is None:
        raise ParseError(f"Cannot extract structure from text: {text}")
    structure = structure_from_context(context, deduped_entities, query)
    structure = expand_conditionals(structure, capabilities)
    structure = structure_with_intentions(text, structure, capabilities)
    return structure_with_rules(structure, capabilities)


def structure_from_context(
    context: ParseContext,
    entities: tuple[Entity, ...],
    query: Query | None,
) -> Structure:
    return Structure(
        entities=entities,
        relations=materialize_relations(context.states),
        events=materialize_events(context.frames),
        rules=(),
        query=query,
        frames=tuple(context.frames),
        states=tuple(context.states),
    )


def structure_with_intentions(
    text: str,
    structure: Structure,
    capabilities: CognitiveCapabilities,
) -> Structure:
    return replace(structure, intentions=capabilities.analyze_intentions(text, structure))


def structure_with_rules(
    structure: Structure,
    capabilities: CognitiveCapabilities,
) -> Structure:
    return replace(structure, rules=capabilities.infer_rules(structure))


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
