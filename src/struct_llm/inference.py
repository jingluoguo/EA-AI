from __future__ import annotations

from .capabilities import Answerer, RuleInferer, StructuralCapabilities
from .errors import ParseError
from .event_schema import frame_matches_qualifiers
from .state_engine import apply_state, states_from_frame
from .structure import Frame, Query, Role, State, Structure


def infer_rules(
    structure: Structure,
    inferers: tuple[RuleInferer, ...] | None = None,
) -> tuple[str, ...]:
    if structure.query is None:
        return ()

    rules: list[str] = []
    for inferer in inferers or DEFAULT_RULE_INFERERS:
        rule = inferer(structure)
        if rule is not None:
            rules.append(rule)
    return tuple(rules)


def expand_conditionals(structure: Structure, capabilities: StructuralCapabilities) -> Structure:
    frames = list(structure.frames)
    states = list(structure.states)
    known_signatures = {frame_signature(frame) for frame in frames}
    next_time = max((frame.time for frame in frames), default=0) + 1

    for _ in range(20):
        changed = False
        for rule in [frame for frame in frames if frame.frame_type == "if_then"]:
            antecedent = required_frame_role(rule, "antecedent")
            consequent = required_frame_role(rule, "consequent")
            if not statement_satisfied(antecedent, frames, states, capabilities):
                continue
            parsed = capabilities.parse_statement(consequent)
            if parsed is None:
                continue
            _, consequence_frames = parsed
            for consequence in consequence_frames:
                signature = frame_signature(consequence)
                if signature in known_signatures:
                    continue
                timed = retime_frame(consequence, next_time)
                next_time += 1
                frames.append(timed)
                known_signatures.add(signature)
                for state in capabilities.states_from_frame(timed):
                    capabilities.apply_state(states, state)
                changed = True
        if not changed:
            break

    return structure_with_frames_states(structure, frames, states)


def statement_satisfied(
    sentence: str,
    frames: list[Frame],
    states: list[State],
    capabilities: StructuralCapabilities,
) -> bool:
    parsed = capabilities.parse_statement(sentence)
    if parsed is None:
        return False
    _, expected_frames = parsed
    current_frame_signatures = {frame_signature(frame) for frame in frames}
    current_state_signatures = {state_signature(state) for state in states}
    for frame in expected_frames:
        projected_states = capabilities.states_from_frame(frame)
        if frame.frame_type in {"be_in", "not_in"} and projected_states:
            if all(state_signature(state) in current_state_signatures for state in projected_states):
                continue
        if frame_signature(frame) not in current_frame_signatures:
            return False
    return True


def frame_signature(frame: Frame) -> tuple[str, tuple[tuple[str, str], ...]]:
    return (
        frame.frame_type,
        tuple(sorted((role.name, role.value) for role in frame.roles)),
    )


def state_signature(state: State) -> tuple[str, str, str]:
    return state.name, state.left, state.right


def retime_frame(frame: Frame, time: int) -> Frame:
    frame_id = f"f{time}"
    return Frame(
        frame_id=frame_id,
        frame_type=frame.frame_type,
        time=time,
        roles=tuple(Role(frame_id, role.name, role.value) for role in frame.roles),
    )


def infer_event_actor_matches(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "actor_for_event" and has_actor_for_event_query(structure, query):
        return "event_actor_matches"
    return None


def infer_latest_event_actor_matches(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "latest_actor_for_event" and has_latest_actor_for_event_query(structure, query):
        return "latest_event_actor_matches"
    return None


def infer_earliest_event_actor_matches(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "earliest_actor_for_event" and has_earliest_actor_for_event_query(structure, query):
        return "earliest_event_actor_matches"
    return None


def infer_compound_query(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "compound" and query.subqueries:
        return "compound_query"
    return None


def infer_actor_handles_item(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "actor_for_item" and has_frame_with_role(
        structure, "handle", "theme", query.target
    ):
        return "actor_handles_item"
    return None


def infer_latest_actor_for_item(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "latest_actor_for_item" and latest_actor_for_item(structure, query.target):
        return "latest_actor_handles_item"
    return None


def infer_initial_location(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "initial_location" and first_location(structure, query.target) is not None:
        return "initial_location_found"
    return None


def infer_location_before_actor_action(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "location_before_actor_action":
        return None
    actor = query_qualifier(query, "actor")
    if location_before_actor_action(structure, query.target, actor) is not None:
        return "location_before_actor_action_found"
    return None


def infer_location_before_event(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "location_before_event":
        return None
    if temporal_event_location(structure, query, include_anchor=False) is not None:
        return "location_before_event_found"
    return "location_before_event_unknown"


def infer_location_after_event(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "location_after_event":
        return None
    if temporal_event_location(structure, query, include_anchor=True) is not None:
        return "location_after_event_found"
    return "location_after_event_unknown"


def infer_contents_before_event(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "contents_before_event":
        return None
    if temporal_event_contents(structure, query, include_anchor=False) is not None:
        return "contents_before_event_found"
    return "contents_before_event_unknown"


def infer_contents_after_event(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "contents_after_event":
        return None
    if temporal_event_contents(structure, query, include_anchor=True) is not None:
        return "contents_after_event_found"
    return "contents_after_event_unknown"


def infer_events_after_event(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "events_after_event":
        return None
    if events_after_query(structure, query):
        return "events_after_event"
    return None


def infer_why(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "why" and explanation_for_target(structure, query.target):
        return "causal_explanation"
    return None


def infer_claim_source(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "claim_source" and claim_speakers(structure, query.target):
        return "claim_has_source"
    return None


def infer_belief_location(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "belief_location":
        return None
    person = query_qualifier(query, "person")
    if belief_location(structure, person, query.target) is not None:
        return "belief_location_found"
    return None


def infer_belief_source(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "belief_source" and belief_sources(structure, query.target):
        return "belief_has_source"
    return None


def infer_contradictions_found(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "contradictions" and contradictions(structure):
        return "contradictions_found"
    return None


def infer_no_contradictions(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "contradictions" and not contradictions(structure):
        return "no_contradictions"
    return None


def infer_counterfactual_location(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "counterfactual_location":
        return None
    if counterfactual_location(structure, query) is not None:
        return "counterfactual_location_found"
    return "counterfactual_location_unknown"


def infer_polar_existence(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "polar_existence":
        return None
    value = object_exists_value(structure, query.target)
    if value == "存在":
        return "polar_existence_true"
    if value == "不存在":
        return "polar_existence_false"
    if object_is_known(structure, query.target):
        return "polar_existence_true"
    return "polar_existence_unknown"


def infer_polar_location(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "polar_location":
        return None
    status = polar_location_status(structure, query)
    if status is None:
        return "polar_location_unknown"
    return "polar_location_true" if status else "polar_location_false"


def infer_polar_contents(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "polar_contents":
        return None
    status = polar_contents_status(structure, query)
    if status is None:
        return "polar_contents_unknown"
    return "polar_contents_true" if status else "polar_contents_false"


def infer_same_location(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "same_location":
        return None
    status = same_location_status(structure, query)
    if status is None:
        return "same_location_unknown"
    return "same_location_true" if status else "same_location_false"


def infer_object_not_exists(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent in {"existence", "location"} and object_exists_value(structure, query.target) == "不存在":
        return "object_not_exists"
    return None


def infer_object_exists(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "existence":
        return None
    if object_exists_value(structure, query.target) == "存在" or object_is_known(structure, query.target):
        return "object_exists"
    return None


def infer_existence_unknown(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "existence" and not object_is_known(structure, query.target):
        return "existence_unknown"
    return None


def infer_holder_contains_things(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "contents" and contents_in_holder(structure, query.target):
        return "holder_contains_things"
    return None


def infer_contents_unknown(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "contents" and not contents_in_holder(structure, query.target):
        return "contents_unknown"
    return None


def infer_contents_except(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "contents_except":
        return None
    query_qualifier(query, "exclude")
    return "holder_contains_except"


def infer_count_known_contents(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "count":
        return "count_known_contents"
    return None


def infer_compare_count(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "compare_count":
        query_qualifier(query, "left")
        query_qualifier(query, "right")
        return "compare_count_known_contents"
    return None


def infer_places_visited(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "places_visited" and places_visited(structure, query.target):
        return "places_visited"
    return None


def infer_actions_by_actors(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "actions_by_actors":
        return None
    actors = query_qualifier(query, "actors").split("|")
    if any(action_descriptions_for_actor(structure, actor) for actor in actors):
        return "actor_actions"
    return None


def infer_inventories(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "inventories" and inventory_by_owner(structure):
        return "owner_inventories"
    return None


def infer_object_at_place(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "location" and has_state_left(structure, "at", query.target):
        return "object_at_place"
    return None


def infer_container_moves_contents(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "location":
        return None
    place, containers = location_path(structure, query.target)
    if place is not None and containers:
        return "container_moves_contents"
    return None


def infer_object_in_container(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "location":
        return None
    place, containers = location_path(structure, query.target)
    if place is None and containers:
        return "object_in_container"
    return None


def infer_unknown_location(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "location":
        return None
    if object_exists_value(structure, query.target) == "不存在":
        return None
    if has_state_left(structure, "at", query.target):
        return None
    place, containers = location_path(structure, query.target)
    if place is not None or containers:
        return None
    return "location_unknown"


def infer_transfer_changes_owner(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "owner" and has_state_left(structure, "owner", query.target):
        return "transfer_changes_owner"
    return None


def infer_paint_changes_color(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "color" and has_state_left(structure, "color", query.target):
        return "paint_changes_color"
    return None


def infer_object_access_state(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "object_state":
        return None
    state_name = optional_query_qualifier(query, "state") or "access"
    if has_state_left(structure, state_name, query.target):
        return "object_access_state"
    return None


def answer_from_structure(
    structure: Structure,
    answerers: tuple[Answerer, ...] | None = None,
) -> str:
    for answerer in answerers or DEFAULT_ANSWERERS:
        answer = answerer(structure)
        if answer is not None:
            return answer

    raise ParseError(f"No rule can answer structure: {structure.linearize()}")


def answer_compound_query(structure: Structure) -> str | None:
    if "compound_query" not in set(structure.rules):
        return None
    query = require_query(structure)
    answers = [answer_subquery(structure, subquery).rstrip("。！？!?") for subquery in query.subqueries]
    return "；".join(answers) + "。"


def answer_subquery(structure: Structure, query: Query) -> str:
    substructure = structure_with_query(structure, query)
    substructure = Structure(
        entities=substructure.entities,
        relations=substructure.relations,
        events=substructure.events,
        rules=infer_rules(substructure),
        query=substructure.query,
        frames=substructure.frames,
        states=substructure.states,
    )
    return answer_from_structure(substructure)


def answer_event_actor(structure: Structure) -> str | None:
    if "event_actor_matches" not in set(structure.rules):
        return None
    query = require_query(structure)
    actor = actor_for_event_query(structure, query)
    item = query_qualifier(query, "item")
    if query.target == "put_in":
        holder = query_qualifier(query, "holder")
        return f"{actor}把{item}放进{holder}。"
    if query.target == "take_out":
        source = query_qualifier(query, "source")
        return f"{actor}把{item}从{source}取出。"
    return f"{actor}执行了{query.target}。"


def answer_latest_event_actor(structure: Structure) -> str | None:
    if "latest_event_actor_matches" not in set(structure.rules):
        return None
    query = require_query(structure)
    actor = latest_actor_for_event_query(structure, query)
    item = query_qualifier(query, "item")
    if query.target == "put_in":
        holder = query_qualifier(query, "holder")
        return f"最后是{actor}把{item}放进{holder}。"
    if query.target == "take_out":
        source = query_qualifier(query, "source")
        return f"最后是{actor}把{item}从{source}取出。"
    return f"最后是{actor}执行了{query.target}。"


def answer_earliest_event_actor(structure: Structure) -> str | None:
    if "earliest_event_actor_matches" not in set(structure.rules):
        return None
    query = require_query(structure)
    actor = earliest_actor_for_event_query(structure, query)
    item = query_qualifier(query, "item")
    if query.target == "put_in":
        holder = query_qualifier(query, "holder")
        return f"最先是{actor}把{item}放进{holder}。"
    if query.target == "take_out":
        source = query_qualifier(query, "source")
        return f"最先是{actor}把{item}从{source}取出。"
    return f"最先是{actor}执行了{query.target}。"


def answer_actor_handles_item(structure: Structure) -> str | None:
    if "actor_handles_item" not in set(structure.rules):
        return None
    query = require_query(structure)
    frame = latest_frame_with_role(structure, "handle", "theme", query.target)
    actor = required_frame_role(frame, "actor")
    return f"{actor}拿的{query.target}。"


def answer_latest_actor_for_item(structure: Structure) -> str | None:
    if "latest_actor_handles_item" not in set(structure.rules):
        return None
    query = require_query(structure)
    actor = latest_actor_for_item(structure, query.target)
    if actor is None:
        raise ParseError(f"Expected latest actor for {query.target}.")
    return f"最后是{actor}处理过{query.target}。"


def answer_initial_location(structure: Structure) -> str | None:
    if "initial_location_found" not in set(structure.rules):
        return None
    query = require_query(structure)
    location = first_location(structure, query.target)
    if location is None:
        raise ParseError(f"Expected initial location for {query.target}.")
    place, containers = location
    return f"{query.target}最开始{location_phrase(place, containers)}。"


def answer_location_before_actor_action(structure: Structure) -> str | None:
    if "location_before_actor_action_found" not in set(structure.rules):
        return None
    query = require_query(structure)
    actor = query_qualifier(query, "actor")
    location = location_before_actor_action(structure, query.target, actor)
    if location is None:
        raise ParseError(f"Expected location before {actor} action for {query.target}.")
    place, containers = location
    return f"{actor}操作之前，{query.target}{location_phrase(place, containers)}。"


def answer_location_before_event(structure: Structure) -> str | None:
    if "location_before_event_found" not in set(structure.rules):
        return None
    query = require_query(structure)
    location = temporal_event_location(structure, query, include_anchor=False)
    if location is None:
        raise ParseError(f"Expected location before event for {query.target}.")
    place, containers = location
    anchor = query_qualifier(query, "anchor")
    return f"在{anchor}之前，{query.target}{location_phrase(place, containers)}。"


def answer_location_before_event_unknown(structure: Structure) -> str | None:
    if "location_before_event_unknown" not in set(structure.rules):
        return None
    query = require_query(structure)
    anchor = query_qualifier(query, "anchor")
    return f"不知道{query.target}在{anchor}之前在哪里。"


def answer_location_after_event(structure: Structure) -> str | None:
    if "location_after_event_found" not in set(structure.rules):
        return None
    query = require_query(structure)
    location = temporal_event_location(structure, query, include_anchor=True)
    if location is None:
        raise ParseError(f"Expected location after event for {query.target}.")
    place, containers = location
    anchor = query_qualifier(query, "anchor")
    return f"在{anchor}之后，{query.target}{location_phrase(place, containers)}。"


def answer_location_after_event_unknown(structure: Structure) -> str | None:
    if "location_after_event_unknown" not in set(structure.rules):
        return None
    query = require_query(structure)
    anchor = query_qualifier(query, "anchor")
    return f"不知道{query.target}在{anchor}之后在哪里。"


def answer_contents_before_event(structure: Structure) -> str | None:
    if "contents_before_event_found" not in set(structure.rules):
        return None
    query = require_query(structure)
    contents = temporal_event_contents(structure, query, include_anchor=False)
    if contents is None:
        raise ParseError(f"Expected contents before event for {query.target}.")
    anchor = query_qualifier(query, "anchor")
    return f"在{anchor}之前，{query.target}里至少有{join_names(contents)}。"


def answer_contents_before_event_unknown(structure: Structure) -> str | None:
    if "contents_before_event_unknown" not in set(structure.rules):
        return None
    query = require_query(structure)
    anchor = query_qualifier(query, "anchor")
    return f"不知道{query.target}在{anchor}之前有什么。"


def answer_contents_after_event(structure: Structure) -> str | None:
    if "contents_after_event_found" not in set(structure.rules):
        return None
    query = require_query(structure)
    contents = temporal_event_contents(structure, query, include_anchor=True)
    if contents is None:
        raise ParseError(f"Expected contents after event for {query.target}.")
    anchor = query_qualifier(query, "anchor")
    return f"在{anchor}之后，{query.target}里至少有{join_names(contents)}。"


def answer_contents_after_event_unknown(structure: Structure) -> str | None:
    if "contents_after_event_unknown" not in set(structure.rules):
        return None
    query = require_query(structure)
    anchor = query_qualifier(query, "anchor")
    return f"不知道{query.target}在{anchor}之后有什么。"


def answer_events_after_event(structure: Structure) -> str | None:
    if "events_after_event" not in set(structure.rules):
        return None
    query = require_query(structure)
    descriptions = events_after_query(structure, query)
    return f"之后发生了：{'；'.join(descriptions)}。"


def answer_why(structure: Structure) -> str | None:
    if "causal_explanation" not in set(structure.rules):
        return None
    query = require_query(structure)
    explanation = explanation_for_target(structure, query.target)
    if explanation is None:
        raise ParseError(f"Expected explanation for {query.target}.")
    return explanation


def answer_claim_source(structure: Structure) -> str | None:
    if "claim_has_source" not in set(structure.rules):
        return None
    query = require_query(structure)
    speakers = claim_speakers(structure, query.target)
    return f"{join_names(speakers)}说的。"


def answer_belief_location(structure: Structure) -> str | None:
    if "belief_location_found" not in set(structure.rules):
        return None
    query = require_query(structure)
    person = query_qualifier(query, "person")
    location = belief_location(structure, person, query.target)
    if location is None:
        raise ParseError(f"Expected belief location for {person}:{query.target}.")
    place, containers = location
    return f"{person}认为{query.target}{location_phrase(place, containers)}。"


def answer_belief_source(structure: Structure) -> str | None:
    if "belief_has_source" not in set(structure.rules):
        return None
    query = require_query(structure)
    believers = belief_sources(structure, query.target)
    return f"{join_names(believers)}这么认为。"


def answer_contradictions_found(structure: Structure) -> str | None:
    if "contradictions_found" not in set(structure.rules):
        return None
    found = contradictions(structure)
    return f"存在矛盾：{'；'.join(found)}。"


def answer_no_contradictions(structure: Structure) -> str | None:
    if "no_contradictions" not in set(structure.rules):
        return None
    return "没有发现矛盾。"


def answer_counterfactual_location(structure: Structure) -> str | None:
    if "counterfactual_location_found" not in set(structure.rules):
        return None
    query = require_query(structure)
    location = counterfactual_location(structure, query)
    if location is None:
        raise ParseError(f"Expected counterfactual location for {query.target}.")
    place, containers = location
    return f"如果没有这个事件，{query.target}会{location_phrase(place, containers)}。"


def answer_counterfactual_location_unknown(structure: Structure) -> str | None:
    if "counterfactual_location_unknown" not in set(structure.rules):
        return None
    query = require_query(structure)
    return f"如果没有这个事件，不知道{query.target}会在哪里。"


def answer_polar_existence_true(structure: Structure) -> str | None:
    if "polar_existence_true" not in set(structure.rules):
        return None
    query = require_query(structure)
    return f"是，{query.target}存在。"


def answer_polar_existence_false(structure: Structure) -> str | None:
    if "polar_existence_false" not in set(structure.rules):
        return None
    query = require_query(structure)
    return f"不是，{query.target}不存在。"


def answer_polar_existence_unknown(structure: Structure) -> str | None:
    if "polar_existence_unknown" not in set(structure.rules):
        return None
    query = require_query(structure)
    return f"不知道{query.target}是否存在。"


def answer_polar_location_true(structure: Structure) -> str | None:
    if "polar_location_true" not in set(structure.rules):
        return None
    query = require_query(structure)
    expected = query_qualifier(query, "expected")
    kind = query_qualifier(query, "kind")
    if kind == "at":
        return f"是，{query.target}在{expected}。"
    return f"是，{query.target}在{expected}里。"


def answer_polar_location_false(structure: Structure) -> str | None:
    if "polar_location_false" not in set(structure.rules):
        return None
    query = require_query(structure)
    expected = query_qualifier(query, "expected")
    kind = query_qualifier(query, "kind")
    actual = location_path(structure, query.target)
    if kind == "at":
        place, containers = actual
        if place is None and not containers:
            return f"不是，不知道{query.target}在哪里。"
        return f"不是，{query.target}{location_phrase(place, containers)}。"
    place, containers = actual
    if place is None and not containers:
        return f"不是，不知道{query.target}在哪里。"
    return f"不是，{query.target}{location_phrase(place, containers)}。"


def answer_polar_location_unknown(structure: Structure) -> str | None:
    if "polar_location_unknown" not in set(structure.rules):
        return None
    query = require_query(structure)
    expected = query_qualifier(query, "expected")
    kind = query_qualifier(query, "kind")
    if kind == "at":
        return f"不知道{query.target}是不是在{expected}。"
    return f"不知道{query.target}是不是在{expected}里。"


def answer_polar_contents_true(structure: Structure) -> str | None:
    if "polar_contents_true" not in set(structure.rules):
        return None
    query = require_query(structure)
    item = query_qualifier(query, "item")
    return f"是，{query.target}里有{item}。"


def answer_polar_contents_false(structure: Structure) -> str | None:
    if "polar_contents_false" not in set(structure.rules):
        return None
    query = require_query(structure)
    item = query_qualifier(query, "item")
    return f"不是，{query.target}里没有{item}。"


def answer_polar_contents_unknown(structure: Structure) -> str | None:
    if "polar_contents_unknown" not in set(structure.rules):
        return None
    query = require_query(structure)
    item = query_qualifier(query, "item")
    return f"不知道{query.target}里有没有{item}。"


def answer_same_location_true(structure: Structure) -> str | None:
    if "same_location_true" not in set(structure.rules):
        return None
    query = require_query(structure)
    left = query_qualifier(query, "left")
    right = query_qualifier(query, "right")
    return f"是，{left}和{right}在同一个地方。"


def answer_same_location_false(structure: Structure) -> str | None:
    if "same_location_false" not in set(structure.rules):
        return None
    query = require_query(structure)
    left = query_qualifier(query, "left")
    right = query_qualifier(query, "right")
    left_phrase = describe_object_location(structure, left)
    right_phrase = describe_object_location(structure, right)
    return f"不是，{left}{left_phrase}，{right}{right_phrase}。"


def answer_same_location_unknown(structure: Structure) -> str | None:
    if "same_location_unknown" not in set(structure.rules):
        return None
    query = require_query(structure)
    left = query_qualifier(query, "left")
    right = query_qualifier(query, "right")
    return f"不知道{left}和{right}是不是在同一个地方。"


def answer_object_not_exists(structure: Structure) -> str | None:
    if "object_not_exists" not in set(structure.rules):
        return None
    query = require_query(structure)
    return f"{query.target}不存在。"


def answer_object_exists(structure: Structure) -> str | None:
    if "object_exists" not in set(structure.rules):
        return None
    query = require_query(structure)
    return f"{query.target}存在。"


def answer_existence_unknown(structure: Structure) -> str | None:
    if "existence_unknown" not in set(structure.rules):
        return None
    query = require_query(structure)
    return f"不知道{query.target}是否存在。"


def answer_holder_contains_things(structure: Structure) -> str | None:
    if "holder_contains_things" not in set(structure.rules):
        return None
    query = require_query(structure)
    contents = contents_in_holder(structure, query.target)
    return f"{query.target}里至少有{join_names(contents)}。"


def answer_contents_unknown(structure: Structure) -> str | None:
    if "contents_unknown" not in set(structure.rules):
        return None
    query = require_query(structure)
    return f"不知道{query.target}里有什么。"


def answer_contents_except(structure: Structure) -> str | None:
    if "holder_contains_except" not in set(structure.rules):
        return None
    query = require_query(structure)
    excluded = query_qualifier(query, "exclude")
    contents = tuple(content for content in contents_in_holder(structure, query.target) if content != excluded)
    if not contents:
        return f"{query.target}里除了{excluded}没有已知物品。"
    return f"{query.target}里除了{excluded}还有{join_names(contents)}。"


def answer_count_known_contents(structure: Structure) -> str | None:
    if "count_known_contents" not in set(structure.rules):
        return None
    query = require_query(structure)
    count = len(contents_in_holder(structure, query.target))
    if count == 0:
        return f"{query.target}里没有已知物品。"
    return f"{query.target}里至少有{count}个已知物品。"


def answer_compare_count(structure: Structure) -> str | None:
    if "compare_count_known_contents" not in set(structure.rules):
        return None
    query = require_query(structure)
    left = query_qualifier(query, "left")
    right = query_qualifier(query, "right")
    left_count = len(contents_in_holder(structure, left))
    right_count = len(contents_in_holder(structure, right))
    if left_count > right_count:
        return f"{left}里的已知物品更多，至少有{left_count}个；{right}里至少有{right_count}个。"
    if right_count > left_count:
        return f"{right}里的已知物品更多，至少有{right_count}个；{left}里至少有{left_count}个。"
    if left_count == 0:
        return f"{left}和{right}里都没有已知物品。"
    return f"{left}和{right}里的已知物品一样多，都是{left_count}个。"


def answer_places_visited(structure: Structure) -> str | None:
    if "places_visited" not in set(structure.rules):
        return None
    query = require_query(structure)
    places = places_visited(structure, query.target)
    return f"{query.target}经过了{join_names(places)}。"


def answer_actions_by_actors(structure: Structure) -> str | None:
    if "actor_actions" not in set(structure.rules):
        return None
    query = require_query(structure)
    actors = query_qualifier(query, "actors").split("|")
    parts = []
    for actor in actors:
        descriptions = action_descriptions_for_actor(structure, actor)
        if descriptions:
            parts.append(f"{actor}{'，'.join(descriptions)}")
        else:
            parts.append(f"{actor}没有已知动作")
    return "；".join(parts) + "。"


def answer_inventories(structure: Structure) -> str | None:
    if "owner_inventories" not in set(structure.rules):
        return None
    inventories = inventory_by_owner(structure)
    parts = [f"{owner}手里有{join_names(items)}" for owner, items in inventories.items()]
    return "；".join(parts) + "。"


def answer_object_at_place(structure: Structure) -> str | None:
    if "object_at_place" not in set(structure.rules):
        return None
    query = require_query(structure)
    state = state_for_left(structure, "at", query.target)
    return f"{query.target}在{state.right}。"


def answer_container_moves_contents(structure: Structure) -> str | None:
    if "container_moves_contents" not in set(structure.rules):
        return None
    query = require_query(structure)
    place, containers = location_path(structure, query.target)
    if place is None:
        raise ParseError(f"Expected place for container location of {query.target}.")
    return f"{query.target}在{place}的{container_chain_text(containers)}。"


def answer_object_in_container(structure: Structure) -> str | None:
    if "object_in_container" not in set(structure.rules):
        return None
    query = require_query(structure)
    _, containers = location_path(structure, query.target)
    return f"{query.target}在{container_chain_text(containers)}。"


def answer_unknown_location(structure: Structure) -> str | None:
    if "location_unknown" not in set(structure.rules):
        return None
    query = require_query(structure)
    return f"不知道{query.target}在哪里。"


def answer_transfer_changes_owner(structure: Structure) -> str | None:
    if "transfer_changes_owner" not in set(structure.rules):
        return None
    query = structure.query
    owner = state_for_left(structure, "owner", query.target) if query else only_state(structure, "owner")
    item = owner.left
    receiver = owner.right
    return f"{receiver}拥有{item}。"


def answer_paint_changes_color(structure: Structure) -> str | None:
    if "paint_changes_color" not in set(structure.rules):
        return None
    query = structure.query
    color_relation = state_for_left(structure, "color", query.target) if query else only_state(structure, "color")
    item = color_relation.left
    color = color_relation.right
    return f"{item}是{color}。"


def answer_object_access_state(structure: Structure) -> str | None:
    if "object_access_state" not in set(structure.rules):
        return None
    query = require_query(structure)
    state_name = optional_query_qualifier(query, "state") or "access"
    state = state_for_left(structure, state_name, query.target)
    return f"{state.left}是{state.right}状态。"


def explanation_for_target(structure: Structure, target: str) -> str | None:
    target = target.strip().rstrip("。！？!?")

    because_matches = [
        required_frame_role(frame, "cause")
        for frame in structure.frames
        if frame.frame_type == "because" and matches_clause_target(required_frame_role(frame, "effect"), target)
    ]
    if because_matches:
        return f"因为{because_matches[-1]}。"

    if "在" in target and "里" in target:
        location = explain_location_target(structure, target)
        if location is not None:
            return location

    if "拥有" in target:
        owner = explain_owner_target(structure, target)
        if owner is not None:
            return owner

    if "颜色" in target or "是" in target:
        color = explain_color_target(structure, target)
        if color is not None:
            return color

    return None


def explain_location_target(structure: Structure, target: str) -> str | None:
    object_name = target.split("在", 1)[0]
    if not object_name:
        return None
    place, containers = location_path(structure, object_name)
    if place is None and not containers:
        return None
    if containers:
        if place is not None:
            return f"因为{object_name}在{container_chain_text(containers)}，而且{containers[-1]}在{place}。"
        return f"因为{object_name}在{container_chain_text(containers)}。"
    if place is not None:
        at_frames = [frame for frame in structure.frames if frame.frame_type == "move" and frame.role("theme") == object_name]
        if at_frames:
            return f"因为{object_name}被带到{place}。"
        return f"因为{object_name}在{place}。"
    return None


def explain_owner_target(structure: Structure, target: str) -> str | None:
    object_name = target.split("拥有", 1)[0]
    owner = state_for_left(structure, "owner", object_name) if any(
        state.name == "owner" and state.left == object_name for state in structure.states
    ) else None
    if owner is None:
        return None
    return f"因为{owner.right}拥有{owner.left}。"


def explain_color_target(structure: Structure, target: str) -> str | None:
    object_name = target.split("是", 1)[0].replace("颜色", "")
    color = state_for_left(structure, "color", object_name) if any(
        state.name == "color" and state.left == object_name for state in structure.states
    ) else None
    if color is None:
        return None
    return f"因为{color.left}是{color.right}。"


def claim_speakers(structure: Structure, proposition: str) -> tuple[str, ...]:
    normalized = proposition.strip().rstrip("。！？!?")
    speakers = [
        required_frame_role(frame, "speaker")
        for frame in structure.frames
        if frame.frame_type == "say" and matches_clause_target(required_frame_role(frame, "proposition"), normalized)
    ]
    return tuple(dict.fromkeys(speakers))


def belief_sources(structure: Structure, proposition: str) -> tuple[str, ...]:
    normalized = proposition.strip().rstrip("。！？!?")
    believers = [
        required_frame_role(frame, "person")
        for frame in structure.frames
        if frame.frame_type == "believe" and matches_clause_target(required_frame_role(frame, "proposition"), normalized)
    ]
    return tuple(dict.fromkeys(believers))


def belief_location(structure: Structure, person: str, target: str) -> tuple[str | None, tuple[str, ...]] | None:
    states = belief_states(structure, person)
    snapshot = structure_with_states(structure, states)
    place, containers = location_path(snapshot, target)
    if place is None and not containers:
        return None
    return place, containers


def belief_states(structure: Structure, person: str) -> list[State]:
    states: list[State] = []
    for frame in sorted(structure.frames, key=lambda candidate: candidate.time):
        if frame.frame_type != "believe" or frame.role("person") != person:
            continue
        for state in states_from_proposition(structure, required_frame_role(frame, "proposition")):
            apply_state(states, state)
    return states


def states_from_proposition(structure: Structure, proposition: str) -> tuple[State, ...]:
    from .frame_parser import parse_effect_clause

    parsed = parse_effect_clause(proposition)
    if parsed is None:
        return fallback_states_from_proposition(structure, proposition)

    states: list[State] = []
    _, frames = parsed
    for frame in frames:
        for state in states_from_frame(frame):
            apply_state(states, state)
    return tuple(states)


def fallback_states_from_proposition(structure: Structure, proposition: str) -> tuple[State, ...]:
    normalized = proposition.strip().rstrip("。！？!?")
    if normalized.endswith("不存在"):
        return (State("exists", normalized.removesuffix("不存在"), "不存在"),)
    if normalized.endswith("存在"):
        return (State("exists", normalized.removesuffix("存在"), "存在"),)

    if "在" not in proposition:
        return ()
    left, right = proposition.split("在", 1)
    left = left.strip()
    right = right.strip().rstrip("。！？!?")
    if not left or not right:
        return ()
    place_names = {entity.name for entity in structure.entities if entity.role == "place"}
    state_name = "at" if right in place_names else "in"
    return (State(state_name, left, right),)


def contradictions(structure: Structure) -> tuple[str, ...]:
    found: list[str] = []
    for frame in sorted(structure.frames, key=lambda candidate: candidate.time):
        source = source_for_view_frame(frame)
        if source is None:
            continue
        actor, verb = source
        proposition = required_frame_role(frame, "proposition")
        for state in states_from_proposition(structure, proposition):
            fact = conflicting_fact_phrase(structure, state)
            if fact is not None:
                found.append(f"{actor}{verb}{proposition}，但事实是{fact}")
                break
    return tuple(dict.fromkeys(found))


def source_for_view_frame(frame: Frame) -> tuple[str, str] | None:
    if frame.frame_type == "say":
        return required_frame_role(frame, "speaker"), "说"
    if frame.frame_type == "believe":
        return required_frame_role(frame, "person"), "认为"
    return None


def conflicting_fact_phrase(structure: Structure, claimed: State) -> str | None:
    if claimed.name == "exists":
        actual_value = object_exists_value(structure, claimed.left)
        if actual_value is not None:
            return None if actual_value == claimed.right else fact_phrase(structure, State("exists", claimed.left, actual_value))
        if claimed.right == "不存在" and object_is_known(structure, claimed.left):
            return fact_phrase(structure, State("exists", claimed.left, "存在"))
        return None

    if claimed.name == "not_in":
        actual = state_for_left_or_none(structure, "in", claimed.left)
        if actual is not None and (not claimed.right or actual.right == claimed.right):
            return fact_phrase(structure, State("in", claimed.left, actual.right))
        return None

    actual = state_for_left_or_none(structure, claimed.name, claimed.left)
    if actual is not None:
        return None if actual.right == claimed.right else fact_phrase(structure, actual)

    if claimed.name in {"in", "at"}:
        place, containers = location_path(structure, claimed.left)
        if claimed.name == "in" and claimed.right in containers:
            return None
        if claimed.name == "at" and place == claimed.right:
            return None
        if place is not None or containers:
            return f"{claimed.left}{location_phrase(place, containers)}"
    return None


def fact_phrase(structure: Structure, state: State) -> str:
    if state.name in {"in", "at"}:
        place, containers = location_path(structure, state.left)
        if place is not None or containers:
            return f"{state.left}{location_phrase(place, containers)}"
    if state.name == "owner":
        return f"{state.right}拥有{state.left}"
    if state.name == "color":
        return f"{state.left}是{state.right}"
    if state.name == "exists":
        return f"{state.left}{state.right}"
    return f"{state.name}({state.left},{state.right})"


def counterfactual_location(structure: Structure, query: Query) -> tuple[str | None, tuple[str, ...]] | None:
    states: list[State] = []
    for frame in sorted(structure.frames, key=lambda candidate: candidate.time):
        if frame_matches_counterfactual_exclusion(frame, query):
            continue
        for state in states_from_frame(frame):
            apply_state(states, state)
    snapshot = structure_with_states(structure, states)
    place, containers = location_path(snapshot, query.target)
    if place is None and not containers:
        return None
    return place, containers


def frame_matches_counterfactual_exclusion(frame: Frame, query: Query) -> bool:
    event = query_qualifier(query, "without_event")
    if frame.frame_type != event:
        return False
    return frame_matches_qualifiers(frame, query.qualifiers, ignored_keys=("without_event",))


def matches_clause_target(clause: str, target: str) -> bool:
    normalized_clause = clause.strip().rstrip("。！？!?")
    normalized_target = target.strip().rstrip("。！？!?")
    return normalized_clause == normalized_target or normalized_target in normalized_clause or normalized_clause in normalized_target


def structure_with_frames_states(structure: Structure, frames: list[Frame], states: list[State]) -> Structure:
    return Structure(
        entities=structure.entities,
        rules=structure.rules,
        relations=tuple(state.to_relation() for state in states),
        events=tuple(frame.to_event() for frame in frames),
        query=structure.query,
        frames=tuple(frames),
        states=tuple(states),
    )


def contents_in_holder(structure: Structure, holder: str) -> tuple[str, ...]:
    contents: list[str] = []
    frontier = direct_contents(structure, holder)

    while frontier:
        current = frontier.pop(0)
        if current in contents:
            continue
        contents.append(current)
        frontier.extend(item for item in direct_contents(structure, current) if item not in contents)

    return tuple(contents)


def direct_contents(structure: Structure, holder: str) -> list[str]:
    contents = [state.left for state in structure.states if state.name == "at" and state.right == holder]
    contents.extend(state.left for state in structure.states if state.name == "in" and state.right == holder)
    return [content for content in contents if object_exists_value(structure, content) != "不存在"]


def places_visited(structure: Structure, target: str) -> tuple[str, ...]:
    states: list[State] = []
    places: list[str] = []
    for frame in sorted(structure.frames, key=lambda candidate: candidate.time):
        for state in states_from_frame(frame):
            apply_state(states, state)
        snapshot = structure_with_states(structure, states)
        place, _ = location_path(snapshot, target)
        if place is not None and place not in places:
            places.append(place)
    return tuple(places)


def inventory_by_owner(structure: Structure) -> dict[str, tuple[str, ...]]:
    inventories: dict[str, list[str]] = {}
    for state in structure.states:
        if state.name == "owner":
            inventories.setdefault(state.right, []).append(state.left)
    return {owner: tuple(items) for owner, items in inventories.items()}


def action_descriptions_for_actor(structure: Structure, actor: str) -> tuple[str, ...]:
    descriptions: list[str] = []
    for frame in structure.frames:
        if frame.frame_type == "handle" or frame.role("actor") != actor:
            continue
        descriptions.append(describe_frame_action(frame))
    return tuple(descriptions)


def describe_frame_action(frame: Frame) -> str:
    if frame.frame_type == "put_in":
        return f"把{required_frame_role(frame, 'theme')}放进{required_frame_role(frame, 'goal')}"
    if frame.frame_type == "take_out":
        return f"把{required_frame_role(frame, 'theme')}从{required_frame_role(frame, 'source')}取出"
    if frame.frame_type == "move":
        return f"把{required_frame_role(frame, 'theme')}带到{required_frame_role(frame, 'goal')}"
    if frame.frame_type == "give":
        return f"把{required_frame_role(frame, 'theme')}交给{required_frame_role(frame, 'recipient')}"
    if frame.frame_type == "paint":
        return f"把{required_frame_role(frame, 'theme')}涂成{required_frame_role(frame, 'result')}"
    if frame.frame_type == "open":
        return f"打开{required_frame_role(frame, 'theme')}"
    if frame.frame_type == "close":
        return f"关闭{required_frame_role(frame, 'theme')}"
    return f"执行{frame.frame_type}"


def describe_historical_frame(frame: Frame) -> str:
    actor = frame.role("actor")
    if actor:
        return f"{actor}{describe_frame_action(frame)}"
    if frame.frame_type == "move":
        return f"{required_frame_role(frame, 'theme')}被带到{required_frame_role(frame, 'goal')}"
    if frame.frame_type == "take_out":
        return f"{required_frame_role(frame, 'theme')}从{required_frame_role(frame, 'source')}被取出"
    if frame.frame_type == "be_in":
        return f"{required_frame_role(frame, 'theme')}在{required_frame_role(frame, 'goal')}里"
    if frame.frame_type == "not_in":
        return f"{required_frame_role(frame, 'theme')}不在{required_frame_role(frame, 'source')}里"
    return describe_frame_action(frame)


def location_path(structure: Structure, target: str) -> tuple[str | None, tuple[str, ...]]:
    if object_exists_value(structure, target) == "不存在":
        return None, ()

    containers: list[str] = []
    current = target
    visited = {target}

    while True:
        at_states = [state for state in structure.states if state.name == "at" and state.left == current]
        if at_states:
            return at_states[-1].right, tuple(containers)

        in_states = [state for state in structure.states if state.name == "in" and state.left == current]
        if not in_states:
            return None, tuple(containers)

        container = in_states[-1].right
        if container in visited:
            raise ParseError(f"Containment cycle detected at {container}.")
        containers.append(container)
        visited.add(container)
        current = container


def container_chain_text(containers: tuple[str, ...]) -> str:
    if not containers:
        raise ParseError("Expected at least one container.")
    ordered = tuple(reversed(containers))
    text = ordered[0]
    for container in ordered[1:]:
        text = f"{text}里的{container}"
    return f"{text}里"


def location_phrase(place: str | None, containers: tuple[str, ...]) -> str:
    if place is not None and containers:
        return f"在{place}的{container_chain_text(containers)}"
    if place is not None:
        return f"在{place}"
    if containers:
        return f"在{container_chain_text(containers)}"
    raise ParseError("Expected a place or container path.")


def first_location(structure: Structure, target: str) -> tuple[str | None, tuple[str, ...]] | None:
    states: list[State] = []
    for frame in sorted(structure.frames, key=lambda candidate: candidate.time):
        for state in states_from_frame(frame):
            apply_state(states, state)
        snapshot = structure_with_states(structure, states)
        place, containers = location_path(snapshot, target)
        if place is not None or containers:
            return place, containers
    return None


def location_before_actor_action(
    structure: Structure,
    target: str,
    actor: str,
) -> tuple[str | None, tuple[str, ...]] | None:
    action = first_actor_action_frame(structure, target, actor)
    if action is None:
        return None
    states = states_before_time(structure, action.time)
    snapshot = structure_with_states(structure, states)
    place, containers = location_path(snapshot, target)
    if place is None and not containers:
        return None
    return place, containers


def temporal_event_location(
    structure: Structure,
    query: Query,
    include_anchor: bool,
) -> tuple[str | None, tuple[str, ...]] | None:
    anchor = temporal_anchor_frame(structure, query)
    if anchor is None:
        return None
    states = states_before_time(structure, anchor.time + 1) if include_anchor else states_before_time(structure, anchor.time)
    snapshot = structure_with_states(structure, states)
    place, containers = location_path(snapshot, query.target)
    if place is None and not containers:
        return None
    return place, containers


def temporal_event_contents(
    structure: Structure,
    query: Query,
    include_anchor: bool,
) -> tuple[str, ...] | None:
    anchor = temporal_anchor_frame(structure, query)
    if anchor is None:
        return None
    states = states_before_time(structure, anchor.time + 1) if include_anchor else states_before_time(structure, anchor.time)
    snapshot = structure_with_states(structure, states)
    contents = contents_in_holder(snapshot, query.target)
    if not contents:
        return None
    return contents


def temporal_anchor_frame(structure: Structure, query: Query) -> Frame | None:
    event_type = query_qualifier(query, "event")
    matches = [
        frame
        for frame in structure.frames
        if frame.frame_type == event_type
        and frame_matches_qualifiers(frame, query.qualifiers, ignored_keys=("anchor", "event"))
    ]
    return matches[0] if matches else None


def first_actor_action_frame(structure: Structure, target: str, actor: str) -> Frame | None:
    frames = [
        frame
        for frame in structure.frames
        if frame.frame_type != "handle" and frame.role("actor") == actor and frame.role("theme") == target
    ]
    if not frames:
        frames = [
            frame
            for frame in structure.frames
            if frame.role("actor") == actor and frame.role("theme") == target
        ]
    return frames[0] if frames else None


def states_before_time(structure: Structure, time: int) -> list[State]:
    states: list[State] = []
    for frame in sorted(structure.frames, key=lambda candidate: candidate.time):
        if frame.time >= time:
            break
        for state in states_from_frame(frame):
            apply_state(states, state)
    return states


def structure_with_states(structure: Structure, states: list[State]) -> Structure:
    return Structure(
        entities=structure.entities,
        rules=structure.rules,
        relations=tuple(state.to_relation() for state in states),
        events=structure.events,
        query=structure.query,
        frames=structure.frames,
        states=tuple(states),
    )


def structure_with_query(structure: Structure, query: Query) -> Structure:
    return Structure(
        entities=structure.entities,
        rules=(),
        relations=structure.relations,
        events=structure.events,
        query=query,
        frames=structure.frames,
        states=structure.states,
    )


def latest_actor_for_item(structure: Structure, target: str) -> str | None:
    matches = [
        required_frame_role(frame, "actor")
        for frame in structure.frames
        if frame.role("theme") == target and frame.role("actor") is not None
    ]
    return matches[-1] if matches else None


def earliest_actor_for_item(structure: Structure, target: str) -> str | None:
    matches = [
        required_frame_role(frame, "actor")
        for frame in structure.frames
        if frame.role("theme") == target and frame.role("actor") is not None
    ]
    return matches[0] if matches else None


def events_after_query(structure: Structure, query: Query) -> tuple[str, ...]:
    anchor = first_matching_event_frame(structure, query)
    if anchor is None:
        return ()
    frames = [
        frame
        for frame in structure.frames
        if frame.time > anchor.time and frame.frame_type != "handle"
    ]
    return tuple(describe_historical_frame(frame) for frame in frames)


def first_matching_event_frame(structure: Structure, query: Query) -> Frame | None:
    matches = [
        frame
        for frame in structure.frames
        if frame.frame_type == query.target and frame_matches_query(frame, query)
    ]
    return matches[0] if matches else None


def has_actor_for_event_query(structure: Structure, query: Query) -> bool:
    try:
        actor_for_event_query(structure, query)
    except ParseError:
        return False
    return True


def actor_for_event_query(structure: Structure, query: Query) -> str:
    matches = [
        required_frame_role(frame, "actor")
        for frame in structure.frames
        if frame.frame_type == query.target
        and frame_matches_query(frame, query)
    ]
    if len(matches) != 1:
        raise ParseError(f"Expected exactly one actor for event query, got {len(matches)}.")
    return matches[0]


def has_latest_actor_for_event_query(structure: Structure, query: Query) -> bool:
    try:
        latest_actor_for_event_query(structure, query)
    except ParseError:
        return False
    return True


def has_earliest_actor_for_event_query(structure: Structure, query: Query) -> bool:
    try:
        earliest_actor_for_event_query(structure, query)
    except ParseError:
        return False
    return True


def latest_actor_for_event_query(structure: Structure, query: Query) -> str:
    matches = [
        required_frame_role(frame, "actor")
        for frame in structure.frames
        if frame.frame_type == query.target
        and frame_matches_query(frame, query)
    ]
    if not matches:
        raise ParseError(f"Expected at least one {query.target} frame for latest actor query.")
    return matches[-1]


def earliest_actor_for_event_query(structure: Structure, query: Query) -> str:
    matches = [
        required_frame_role(frame, "actor")
        for frame in structure.frames
        if frame.frame_type == query.target
        and frame_matches_query(frame, query)
    ]
    if not matches:
        raise ParseError(f"Expected at least one {query.target} frame for earliest actor query.")
    return matches[0]


def frame_matches_query(frame: Frame, query: Query) -> bool:
    return frame_matches_qualifiers(frame, query.qualifiers)


def query_qualifier(query: Query, key: str) -> str:
    prefix = f"{key}="
    matches = [qualifier.removeprefix(prefix) for qualifier in query.qualifiers if qualifier.startswith(prefix)]
    if len(matches) != 1:
        raise ParseError(f"Expected exactly one query qualifier {key}, got {len(matches)}.")
    return matches[0]


def optional_query_qualifier(query: Query, key: str) -> str | None:
    prefix = f"{key}="
    matches = [qualifier.removeprefix(prefix) for qualifier in query.qualifiers if qualifier.startswith(prefix)]
    if not matches:
        return None
    if len(matches) != 1:
        raise ParseError(f"Expected at most one query qualifier {key}, got {len(matches)}.")
    return matches[0]


def split_qualifier(qualifier: str) -> tuple[str, str]:
    if "=" not in qualifier:
        raise ParseError(f"Expected query qualifier key=value, got {qualifier}.")
    key, value = qualifier.split("=", 1)
    return key, value


def only_state(structure: Structure, name: str) -> State:
    matches = [state for state in structure.states if state.name == name]
    if len(matches) != 1:
        raise ParseError(f"Expected exactly one {name} state, got {len(matches)}.")
    return matches[0]


def state_for_left(structure: Structure, name: str, left: str) -> State:
    matches = [state for state in structure.states if state.name == name and state.left == left]
    if len(matches) != 1:
        raise ParseError(f"Expected exactly one {name} state for {left}, got {len(matches)}.")
    return matches[0]


def state_for_left_or_none(structure: Structure, name: str, left: str) -> State | None:
    matches = [state for state in structure.states if state.name == name and state.left == left]
    return matches[-1] if matches else None


def has_state_left(structure: Structure, name: str, left: str) -> bool:
    return any(state.name == name and state.left == left for state in structure.states)


def object_exists_value(structure: Structure, target: str) -> str | None:
    states = [state for state in structure.states if state.name == "exists" and state.left == target]
    return states[-1].right if states else None


def object_is_known(structure: Structure, target: str) -> bool:
    if any(entity.name == target for entity in structure.entities):
        return True
    if any(state.left == target or state.right == target for state in structure.states):
        return True
    if any(role.value == target for frame in structure.frames for role in frame.roles):
        return True
    return False


def polar_location_status(structure: Structure, query: Query) -> bool | None:
    expected = query_qualifier(query, "expected")
    kind = query_qualifier(query, "kind")
    if object_exists_value(structure, query.target) == "不存在":
        return False
    place, containers = location_path(structure, query.target)
    if kind == "at":
        if place is None and not containers:
            return None
        if place is None:
            return False
        return place == expected
    if place is None and not containers:
        return None
    return expected in containers


def polar_contents_status(structure: Structure, query: Query) -> bool | None:
    item = query_qualifier(query, "item")
    if object_exists_value(structure, item) == "不存在":
        return False
    contents = contents_in_holder(structure, query.target)
    if item in contents:
        return True
    if any(
        state.name == "not_in" and state.left == item and state.right == query.target
        for state in structure.states
    ):
        return False
    if contents:
        return False
    return None


def same_location_status(structure: Structure, query: Query) -> bool | None:
    left = query_qualifier(query, "left")
    right = query_qualifier(query, "right")
    left_key = location_key(structure, left)
    right_key = location_key(structure, right)
    if left_key is None or right_key is None:
        return None
    return left_key == right_key


def location_key(structure: Structure, target: str) -> tuple[str, str] | None:
    if object_exists_value(structure, target) == "不存在":
        return None
    place, containers = location_path(structure, target)
    if place is not None:
        return ("place", place)
    if containers:
        return ("container", containers[-1])
    return None


def describe_object_location(structure: Structure, target: str) -> str:
    if object_exists_value(structure, target) == "不存在":
        return "不存在"
    place, containers = location_path(structure, target)
    if place is None and not containers:
        return "在哪里未知"
    return location_phrase(place, containers)


def has_frame_with_role(structure: Structure, frame_type: str, role_name: str, value: str) -> bool:
    return any(frame.frame_type == frame_type and frame.role(role_name) == value for frame in structure.frames)


def latest_frame_with_role(structure: Structure, frame_type: str, role_name: str, value: str) -> Frame:
    matches = [frame for frame in structure.frames if frame.frame_type == frame_type and frame.role(role_name) == value]
    if not matches:
        raise ParseError(f"Expected at least one {frame_type} frame with {role_name}={value}.")
    return matches[-1]


def required_frame_role(frame: Frame, role_name: str) -> str:
    value = frame.role(role_name)
    if value is None:
        raise ParseError(f"Expected role {role_name} in frame {frame.frame_id}.")
    return value


def require_query(structure: Structure) -> Query:
    if structure.query is None:
        raise ParseError("Expected query in structure.")
    return structure.query


def join_names(names: tuple[str, ...]) -> str:
    if len(names) == 1:
        return names[0]
    return "和".join(names)


DEFAULT_RULE_INFERERS: tuple[RuleInferer, ...] = (
    infer_earliest_event_actor_matches,
    infer_latest_event_actor_matches,
    infer_compound_query,
    infer_event_actor_matches,
    infer_actor_handles_item,
    infer_latest_actor_for_item,
    infer_initial_location,
    infer_location_before_actor_action,
    infer_location_before_event,
    infer_location_after_event,
    infer_contents_before_event,
    infer_contents_after_event,
    infer_events_after_event,
    infer_why,
    infer_claim_source,
    infer_belief_location,
    infer_belief_source,
    infer_contradictions_found,
    infer_no_contradictions,
    infer_counterfactual_location,
    infer_polar_existence,
    infer_polar_location,
    infer_polar_contents,
    infer_same_location,
    infer_object_not_exists,
    infer_object_exists,
    infer_existence_unknown,
    infer_holder_contains_things,
    infer_contents_unknown,
    infer_contents_except,
    infer_count_known_contents,
    infer_compare_count,
    infer_places_visited,
    infer_actions_by_actors,
    infer_inventories,
    infer_object_at_place,
    infer_container_moves_contents,
    infer_object_in_container,
    infer_unknown_location,
    infer_transfer_changes_owner,
    infer_paint_changes_color,
    infer_object_access_state,
)
DEFAULT_ANSWERERS: tuple[Answerer, ...] = (
    answer_earliest_event_actor,
    answer_latest_event_actor,
    answer_compound_query,
    answer_event_actor,
    answer_actor_handles_item,
    answer_latest_actor_for_item,
    answer_initial_location,
    answer_location_before_actor_action,
    answer_location_before_event,
    answer_location_before_event_unknown,
    answer_location_after_event,
    answer_location_after_event_unknown,
    answer_contents_before_event,
    answer_contents_before_event_unknown,
    answer_contents_after_event,
    answer_contents_after_event_unknown,
    answer_events_after_event,
    answer_why,
    answer_claim_source,
    answer_belief_location,
    answer_belief_source,
    answer_contradictions_found,
    answer_no_contradictions,
    answer_counterfactual_location,
    answer_counterfactual_location_unknown,
    answer_polar_existence_true,
    answer_polar_existence_false,
    answer_polar_existence_unknown,
    answer_polar_location_true,
    answer_polar_location_false,
    answer_polar_location_unknown,
    answer_polar_contents_true,
    answer_polar_contents_false,
    answer_polar_contents_unknown,
    answer_same_location_true,
    answer_same_location_false,
    answer_same_location_unknown,
    answer_object_not_exists,
    answer_object_exists,
    answer_existence_unknown,
    answer_holder_contains_things,
    answer_contents_unknown,
    answer_contents_except,
    answer_count_known_contents,
    answer_compare_count,
    answer_places_visited,
    answer_actions_by_actors,
    answer_inventories,
    answer_object_at_place,
    answer_container_moves_contents,
    answer_object_in_container,
    answer_unknown_location,
    answer_transfer_changes_owner,
    answer_paint_changes_color,
    answer_object_access_state,
)
