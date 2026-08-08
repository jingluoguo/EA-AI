from __future__ import annotations

from dataclasses import dataclass, replace

from .capabilities import CognitiveCapabilities
from .comprehension.query import query_candidate_is_learned_unit, resolve_query_candidate, resolve_query_candidates
from .comprehension.structure_helpers import dedupe_entities, with_time
from .errors import ParseError
from .memory.long_term import memory_entities_from_states
from .perception.lexer import split_query_candidate, split_sentences
from .perception.reference import resolve_references, unresolved_reference_mention
from .structure import Entity, Frame, PragmaticAct, Query, Role, ScopedFrame, ScopedState, State, Structure
from .world.causal import expand_conditionals
from .world.state import materialize_events, materialize_relations

INTEGRATED_CLAUSE_FRAME_TYPES = frozenset({"if_then", "because"})
SCOPED_PROPOSITION_ROLES: dict[str, tuple[tuple[str, str, str], ...]] = {
    "say": (("proposition", "claim", "speaker"),),
    "believe": (("proposition", "belief", "person"),),
    "because": (("cause", "cause", ""), ("effect", "effect", "")),
}


@dataclass
class ParseContext:
    entities: list[Entity]
    discourse_entities: tuple[Entity, ...]
    unresolved_references: list[Entity]
    frames: list[Frame]
    states: list[State]
    scoped_frames: list[ScopedFrame]
    scoped_states: list[ScopedState]
    query_candidates: list[str]
    next_time: int = 1
    current_frame_start_time: int = 1

    def known_entities(self) -> tuple[Entity, ...]:
        return dedupe_entities(self.entities)


def initial_parse_context(capabilities: CognitiveCapabilities) -> ParseContext:
    memory_entities = list(memory_entities_from_states(capabilities.memory_states))
    memory_entities.extend(memory_entities_from_frames(capabilities.memory_frames))
    discourse_entities = (Entity("self", "我"),)
    memory_frames = tuple(capabilities.memory_frames)
    next_time = next_time_for_memory_frames(memory_frames)
    return ParseContext(
        entities=memory_entities,
        discourse_entities=tuple(
            entity for entity in discourse_entities if all(entity.name != known.name for known in memory_entities)
        ),
        unresolved_references=[],
        frames=list(memory_frames),
        states=list(capabilities.memory_states),
        scoped_frames=[],
        scoped_states=[],
        query_candidates=[],
        next_time=next_time,
        current_frame_start_time=next_time,
    )


def ingest_sentence(
    sentence: str,
    is_question: bool,
    context: ParseContext,
    capabilities: CognitiveCapabilities,
) -> None:
    resolved_sentence = resolve_references(sentence, context.known_entities())
    mention = unresolved_reference_mention(sentence, resolved_sentence)
    try:
        fragments = split_query_candidate(sentence)

        if not is_question and ingest_mixed_statement_fragments(sentence, context, capabilities):
            return

        if not is_question and ingest_statement_sentence(
            resolved_sentence,
            context,
            capabilities,
        ):
            return

        if not is_question and query_candidate_is_learned_unit(
            resolved_sentence,
            context.known_entities(),
            capabilities.query_parsers,
        ):
            context.query_candidates.append(resolved_sentence)
            return

        if is_question and query_candidate_is_learned_unit(
            resolved_sentence,
            context.known_entities(),
            capabilities.query_parsers,
        ):
            context.query_candidates.append(resolved_sentence)
            return

        for fragment in fragments:
            ingest_query_fragment(fragment, is_question, context, capabilities)
    finally:
        if mention is not None:
            context.unresolved_references.append(mention)


def ingest_mixed_statement_fragments(
    sentence: str,
    context: ParseContext,
    capabilities: CognitiveCapabilities,
) -> bool:
    fragments = split_query_candidate(sentence)
    if len(fragments) <= 1:
        return False

    parsed_fragments: list[tuple[str, tuple[list[Entity], list[Frame]] | None]] = []
    has_statement = False
    known_entities = context.known_entities()
    resolved_sentence = resolve_references(sentence, known_entities)
    full_statement = capabilities.parse_statement(resolved_sentence)
    for fragment in fragments:
        resolved_fragment = resolve_references(fragment, known_entities)
        extracted = capabilities.parse_statement(resolved_fragment)
        fragment_query = resolve_query_candidate(resolved_fragment, known_entities, capabilities.query_parsers)
        if extracted is not None and fragment_query is not None and statement_should_yield_to_query(
            extracted,
            fragment_query,
            resolved_fragment,
        ):
            extracted = None
        parsed_fragments.append((fragment, extracted))
        if extracted is not None:
            has_statement = True

    if full_statement_should_cover_fragments(full_statement, tuple(extracted for _, extracted in parsed_fragments)):
        add_extracted_structure(full_statement, context, capabilities)
        return True

    if not has_statement:
        return False

    for fragment, extracted in parsed_fragments:
        if extracted is not None:
            add_extracted_structure(extracted, context, capabilities)
            continue
        ingest_query_fragment(fragment, False, context, capabilities)
    return True


def full_statement_should_cover_fragments(
    full_statement: tuple[list[Entity], list[Frame]] | None,
    fragment_statements: tuple[tuple[list[Entity], list[Frame]] | None, ...],
) -> bool:
    if full_statement is None:
        return False
    # Integrated clause frames must stay whole; fragment parses are only a fallback.
    if any(frame.frame_type in INTEGRATED_CLAUSE_FRAME_TYPES for frame in full_statement[1]):
        return True
    fragment_signatures = {
        frame_signature(frame)
        for extracted in fragment_statements
        if extracted is not None
        for frame in extracted[1]
    }
    return any(frame_signature(frame) not in fragment_signatures for frame in full_statement[1])


def frame_signature(frame: Frame) -> tuple[str, tuple[tuple[str, str], ...]]:
    return (
        frame.frame_type,
        tuple(sorted((role.name, role.value) for role in frame.roles)),
    )


def ingest_statement_sentence(
    sentence: str,
    context: ParseContext,
    capabilities: CognitiveCapabilities,
) -> bool:
    extracted = capabilities.parse_statement(sentence)
    if extracted is None:
        return False

    add_extracted_structure(extracted, context, capabilities)
    return True


def statement_should_yield_to_query(
    extracted: tuple[list[Entity], list[Frame]],
    query: Query,
    sentence: str = "",
) -> bool:
    if profile_statement_should_yield_to_query(extracted, query):
        return True
    if query.intent not in {"dialog_act", "profile"}:
        return True
    if query.intent == "dialog_act" and query.target in {
        "thanks",
        "farewell",
        "clarification",
        "apology",
        "emotion",
        "affection",
        "empathy",
        "refusal",
    }:
        return True
    return query.intent == "dialog_act" and profile_statement_value_is_under_specified(extracted)


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
    fragment_query = resolve_query_candidate(resolved_fragment, context.known_entities(), capabilities.query_parsers)
    if extracted is not None and fragment_query is not None and statement_should_yield_to_query(
        extracted,
        fragment_query,
        resolved_fragment,
    ):
        context.query_candidates.append(resolved_fragment)
        return
    if extracted is not None:
        add_extracted_structure(extracted, context, capabilities)
        return

    if fragment_query is not None:
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
        add_scoped_proposition_structure(timed_frame, context, capabilities)


def next_time_for_memory_frames(frames: tuple[Frame, ...]) -> int:
    if not frames:
        return 1
    return max(frame.time for frame in frames) + 1


def memory_entities_from_frames(frames: tuple[Frame, ...]) -> tuple[Entity, ...]:
    entities: list[Entity] = []
    for frame in frames:
        for role in frame.roles:
            entity_role = entity_role_for_frame_role(frame, role.name)
            if entity_role is None or not role.value:
                continue
            entities.append(Entity(entity_role, role.value))
    return dedupe_entities(tuple(entities))


def entity_role_for_frame_role(frame: Frame, role_name: str) -> str | None:
    if role_name in {"actor", "speaker", "person", "giver", "receiver"}:
        return "person"
    if role_name in {"theme", "item", "object"}:
        return "item"
    if role_name == "subject":
        return "person"
    if role_name == "value":
        return "profile_value"
    if role_name == "goal":
        if frame.frame_type in {"put_in", "give", "handle"}:
            return "container"
        if frame.frame_type in {"move", "be_in", "if_then", "because"}:
            return "place"
        return "place"
    if role_name == "source":
        if frame.frame_type == "take_out":
            return "container"
        return "place"
    if role_name == "recipient":
        return "person"
    if role_name == "proposition":
        return "thing"
    return None


def add_scoped_proposition_structure(
    source_frame: Frame,
    context: ParseContext,
    capabilities: CognitiveCapabilities,
) -> None:
    for role_name, kind, owner_role in SCOPED_PROPOSITION_ROLES.get(source_frame.frame_type, ()):
        proposition = source_frame.role(role_name)
        if not proposition:
            continue
        parsed = capabilities.parse_statement(proposition)
        if parsed is None:
            continue
        proposition_entities, proposition_frames = parsed
        context.entities.extend(proposition_entities)
        owner = source_frame.role(owner_role) if owner_role else ""
        for index, frame in enumerate(proposition_frames, start=1):
            scoped_frame_id = f"{source_frame.frame_id}:{role_name}{index}"
            scoped_frame = retime_scoped_frame(frame, scoped_frame_id, source_frame.time)
            context.scoped_frames.append(
                ScopedFrame(
                    scope=source_frame.frame_id,
                    kind=kind,
                    owner=owner or "",
                    proposition=proposition,
                    frame=scoped_frame,
                )
            )
            for state in capabilities.states_from_frame(scoped_frame):
                context.scoped_states.append(
                    ScopedState(
                        scope=source_frame.frame_id,
                        kind=kind,
                        owner=owner or "",
                        proposition=proposition,
                        state=state,
                    )
                )


def retime_scoped_frame(frame: Frame, frame_id: str, time: int) -> Frame:
    return Frame(
        frame_id=frame_id,
        frame_type=frame.frame_type,
        time=time,
        roles=tuple(Role(frame_id, role.name, role.value) for role in frame.roles),
    )


def finalize_parse_context(
    text: str,
    context: ParseContext,
    capabilities: CognitiveCapabilities,
) -> Structure:
    deduped_entities = context.known_entities()
    query_error: ParseError | None = None
    try:
        query = resolve_query_candidates(context.query_candidates, deduped_entities, capabilities.query_parsers)
    except ParseError as error:
        query_error = error
        query = None
    structure = structure_from_context(context, output_entities(context, query), query)
    structure = expand_conditionals(structure, capabilities)
    structure = structure_with_intentions(text, structure, capabilities)
    structure = structure_with_pragmatics(text, structure, capabilities)
    if (
        not context.entities
        and not context.states
        and not context.frames
        and query is None
        and not structure.pragmatic_acts
    ):
        if query_error is not None:
            raise query_error
        raise ParseError(f"Cannot extract structure from text: {text}")
    return structure_with_rules(structure, capabilities)


def output_entities(context: ParseContext, query: Query | None) -> tuple[Entity, ...]:
    entities = [
        *(entity for entity in context.entities if entity.role != "query_intent"),
        *context.unresolved_references,
    ]
    if query is not None:
        entities.extend(query_referenced_discourse_entities(query, context.discourse_entities))
    return dedupe_entities(entities)


def query_referenced_discourse_entities(query: Query, discourse_entities: tuple[Entity, ...]) -> tuple[Entity, ...]:
    values = query_values(query)
    return tuple(entity for entity in discourse_entities if entity.name in values)


def query_values(query: Query) -> tuple[str, ...]:
    values = [query.target, *query.qualifiers]
    for subquery in query.subqueries:
        values.extend(query_values(subquery))
    return tuple(values)


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
        scoped_frames=tuple(context.scoped_frames),
        scoped_states=tuple(context.scoped_states),
        current_frame_start_time=context.current_frame_start_time,
    )


def structure_with_intentions(
    text: str,
    structure: Structure,
    capabilities: CognitiveCapabilities,
) -> Structure:
    return replace(structure, intentions=capabilities.analyze_intentions(text, structure))


def structure_with_pragmatics(
    text: str,
    structure: Structure,
    capabilities: CognitiveCapabilities,
) -> Structure:
    return replace(
        structure,
        pragmatic_acts=filter_resolved_pragmatic_acts(
            capabilities.analyze_pragmatics(text, structure),
            structure,
        ),
    )


def filter_resolved_pragmatic_acts(
    acts: tuple[PragmaticAct, ...],
    structure: Structure,
) -> tuple[PragmaticAct, ...]:
    if not acts:
        return ()
    return tuple(act for act in acts if not pragmatic_act_is_resolved_by_structure(act, structure))


def pragmatic_act_is_resolved_by_structure(act: PragmaticAct, structure: Structure) -> bool:
    if structure.query is not None and act.act in {
        "ambiguous_reference",
        "clarification_request",
        "incomplete_utterance",
        "underspecified_action_request",
        "underspecified_reference_query",
    }:
        return True
    if any(frame.time >= structure.current_frame_start_time for frame in structure.frames) and act.act in {
        "ambiguous_reference",
        "clarification_request",
        "incomplete_utterance",
        "underspecified_action_request",
        "underspecified_reference_query",
    }:
        return True
    return False


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
