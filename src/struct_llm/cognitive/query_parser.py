from __future__ import annotations

import re
from typing import Optional

from .capabilities import QueryParser
from ..errors import ParseError
from .normalization import (
    QUESTION_FILLERS,
    is_question_noise,
    normalize_container_slot,
    normalize_entity_slot,
    normalize_question,
    normalize_slot_value,
    normalize_take_out_expression,
)
from .frame_parser import normalize_clause_text, parse_effect_clause
from ..structure import Entity, Query
from .text_processing import split_query_candidate


BELIEF_VERBS = ("认为", "相信", "觉得", "以为")
LIGHT_DIALOG_ACT_TARGETS = {"greeting", "thanks", "farewell"}
COUNTERFACTUAL_TAKE_OUT_RE = (
    re.compile(r"(?P<actor>.+?)没有把(?P<theme>.+?)从(?P<source>.+?)取出$"),
    re.compile(r"(?P<actor>.+?)没有从(?P<source>.+?)取出(?P<theme>.+?)$"),
)
COUNTERFACTUAL_EVENT_RE = (
    (
        "put_in",
        re.compile(r"(?P<actor>.+?)没有把(?P<theme>.+?)放进(?P<goal>.+?)$"),
    ),
    (
        "move",
        re.compile(r"(?P<actor>.+?)没有把(?P<theme>.+?)带到(?P<goal>.+?)$"),
    ),
)


def parse_query_candidates(
    candidates: list[str],
    entities: tuple[Entity, ...],
    parsers: tuple[QueryParser, ...] | None = None,
) -> Optional[Query]:
    if not candidates:
        return None

    errors: list[ParseError] = []
    parsed_queries: list[Query] = []
    for candidate in candidates:
        if is_question_noise(candidate):
            continue
        if should_parse_candidate_as_unit(candidate):
            try:
                parsed_queries.append(parse_query(candidate, entities, parsers))
                continue
            except ParseError as error:
                errors.append(error)

        fragments = split_query_candidate(candidate)
        for fragment in fragments:
            try:
                parsed_queries.append(parse_query(fragment, entities, parsers))
            except ParseError as error:
                errors.append(error)

    if any(not is_light_dialog_act_query(query) for query in parsed_queries):
        parsed_queries = [query for query in parsed_queries if not is_light_dialog_act_query(query)]

    if len(parsed_queries) > 1:
        return Query("compound", "multi", subqueries=tuple(parsed_queries))
    if len(parsed_queries) == 1:
        return parsed_queries[0]

    combined = "，".join(candidates)
    try:
        return parse_query(combined, entities, parsers)
    except ParseError:
        if errors:
            raise errors[-1]
        raise ParseError(f"Cannot parse question: {combined}")


def parse_query(
    sentence: str,
    entities: tuple[Entity, ...],
    parsers: tuple[QueryParser, ...] | None = None,
) -> Query:
    raw_sentence = sentence.strip()
    polar = parse_polar_query(raw_sentence, entities)
    if polar is not None:
        return polar

    normalized = normalize_question(sentence)
    for parser in parsers or DEFAULT_QUERY_PARSERS:
        if parser is parse_polar_query:
            continue
        query = parser(normalized, entities)
        if query is not None:
            return query

    raise ParseError(f"Cannot parse question: {sentence}")


def is_light_dialog_act_query(query: Query) -> bool:
    return query.intent == "dialog_act" and query.target in LIGHT_DIALOG_ACT_TARGETS


def parse_dialog_act_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    normalized = normalize_question(sentence).strip()
    if normalized in {"你好", "您好", "嗨", "哈喽", "在吗", "在不在", "在么", "有人吗"}:
        return Query("dialog_act", "greeting")
    if "谢谢" in normalized or "感谢" in normalized:
        return Query("dialog_act", "thanks")
    if normalized in {"再见", "拜拜"}:
        return Query("dialog_act", "farewell")
    if normalized in {"你是谁", "你叫什么", "你叫什么名字"}:
        return Query("dialog_act", "identity")
    if any(
        phrase in normalized
        for phrase in ("你能做什么", "你会什么", "你可以做什么", "你能帮我什么", "你可以帮我什么")
    ):
        return Query("dialog_act", "capabilities")
    if any(word in normalized for word in ("总结", "概括", "回顾")) or (
        "刚才" in normalized and "说了什么" in normalized
    ):
        return Query("dialog_act", "summary")
    return None


def parse_profile_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    normalized = normalize_question(sentence).strip()
    if any(phrase in normalized for phrase in ("我叫什么", "我是谁", "我的名字", "我的姓名")):
        return Query("profile", "我", ("attribute=name",))
    if any(phrase in normalized for phrase in ("我喜欢什么", "我的爱好", "我爱好什么")):
        return Query("profile", "我", ("attribute=likes",))
    if any(phrase in normalized for phrase in ("我讨厌什么", "我不喜欢什么")):
        return Query("profile", "我", ("attribute=dislikes",))
    return None


def should_parse_candidate_as_unit(candidate: str) -> bool:
    normalized = candidate.strip()
    if normalized.startswith("如果") and "没有" in normalized:
        return True
    if "之前" in normalized:
        return True
    if "之后" in normalized:
        return True
    return False


def parse_event_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    put_in = parse_put_in_event_question_semantic(sentence, entities)
    if put_in is not None:
        return put_in

    take_out = parse_take_out_event_question_semantic(sentence, entities)
    if take_out is not None:
        return take_out

    return None


def parse_earliest_event_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    put_in = parse_earliest_put_in_event_question_semantic(sentence, entities)
    if put_in is not None:
        return put_in

    take_out = parse_earliest_take_out_event_question_semantic(sentence, entities)
    if take_out is not None:
        return take_out

    return None


def parse_latest_event_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    put_in = parse_latest_put_in_event_question_semantic(sentence, entities)
    if put_in is not None:
        return put_in

    take_out = parse_latest_take_out_event_question_semantic(sentence, entities)
    if take_out is not None:
        return take_out

    return None


def parse_why_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    if sentence.startswith(("为什么", "为啥", "为何", "怎么会")):
        target = sentence
        for word in ("为什么", "为啥", "为何", "怎么会"):
            target = target.replace(word, "")
        target = normalize_slot_value(target)
        return Query("why", target)
    return None


def parse_claim_source_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    if "谁" in sentence and "说" in sentence:
        target = extract_query_target(sentence, ("谁", "说", "说的", "说吗", "说过"), entities)
        return Query("claim_source", target)
    return None


def parse_belief_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    verb = first_present(sentence, BELIEF_VERBS)
    if verb is None:
        return None

    if "谁" in sentence and sentence.index("谁") < sentence.index(verb):
        proposition = normalize_slot_value(sentence.split(verb, 1)[1])
        return Query("belief_source", proposition)

    if not any(word in sentence for word in ("哪里", "哪儿", "什么地方")):
        return None
    person = entity_in_text(sentence.split(verb, 1)[0], entities, ("person", "giver", "receiver"))
    if person is None:
        return None
    rest = sentence.split(verb, 1)[1]
    target = extract_query_target(rest, ("哪里", "哪儿", "什么地方", "在"), entities)
    return Query("belief_location", target, (f"person={person}",))


def parse_contradiction_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    if any(word in sentence for word in ("矛盾", "冲突", "不一致")):
        return Query("contradictions", "world")
    return None


def parse_polar_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    normalized = normalize_question(sentence).strip()
    if not is_polar_question(sentence, normalized):
        return None

    existence = parse_polar_existence_query_semantic(normalized, entities)
    if existence is not None:
        return existence

    same_location = parse_polar_same_location_query_semantic(normalized, entities)
    if same_location is not None:
        return same_location

    location = parse_polar_location_query_semantic(normalized, entities)
    if location is not None:
        return location

    contents = parse_polar_contents_query_semantic(normalized, entities)
    if contents is not None:
        return contents

    return None


def is_polar_question(sentence: str, normalized: str) -> bool:
    if sentence.endswith("吗") or normalized.endswith("吗"):
        return True
    return any(marker in normalized for marker in ("是不是", "有没有"))


def parse_polar_existence_query_semantic(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    normalized = normalize_question(sentence).strip()
    body = normalized.removesuffix("吗")
    for marker in ("是否存在", "还在", "存在"):
        if marker not in body:
            continue
        target_text = body.split(marker, 1)[0]
        if not target_text:
            return None
        return Query("polar_existence", normalize_entity_slot(target_text, entities))
    return None


def parse_polar_location_query_semantic(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    normalized = normalize_question(sentence).strip()
    body = normalized.removesuffix("吗")
    if any(word in body for word in ("哪里", "哪儿", "什么地方", "同一个", "同一")):
        return None
    marker = find_last_marker(body, "在")
    if marker is None:
        return None
    target_text = body[:marker]
    place_text = body[marker + len("在") :]
    if not target_text or not place_text:
        return None
    target = normalize_entity_slot(target_text, entities)
    place_role = next((entity.role for entity in entities if entity.name == normalize_entity_slot(place_text, entities)), None)
    expected_kind = "at" if place_role == "place" else "in"
    place = normalize_entity_slot(place_text, entities) if expected_kind == "at" else normalize_container_slot(place_text)
    return Query("polar_location", target, (f"expected={place}", f"kind={expected_kind}"))


def parse_polar_contents_query_semantic(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    normalized = normalize_question(sentence).strip()
    body = normalized.removesuffix("吗")
    if any(word in body for word in ("什么", "几个", "多少", "数量")):
        return None
    if "有没有" in body:
        holder_text, item_text = body.split("有没有", 1)
    else:
        marker = find_last_marker(body, "有")
        if marker is None:
            return None
        holder_text = body[:marker]
        item_text = body[marker + len("有") :]
    if not holder_text or not item_text:
        return None
    holder = normalize_entity_slot(normalize_container_slot(holder_text), entities)
    item = normalize_entity_slot(item_text, entities)
    return Query("polar_contents", holder, (f"item={item}",))


def parse_polar_same_location_query_semantic(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    normalized = normalize_question(sentence).strip()
    body = normalized.removesuffix("吗")
    if "和" not in body or ("在同一个" not in body and "在同一" not in body):
        return None
    left_text, rest = body.split("和", 1)
    right_text = rest.split("在", 1)[0]
    if not left_text or not right_text:
        return None
    left = normalize_entity_slot(left_text, entities)
    right = normalize_entity_slot(right_text, entities)
    return Query("same_location", f"{left}和{right}", (f"left={left}", f"right={right}"))


def parse_counterfactual_location_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    if "没有" not in sentence or not any(
        word in sentence for word in ("哪里", "哪儿", "什么地方")
    ):
        return None
    split = split_counterfactual(sentence, entities)
    if split is None:
        return None
    condition, question = split
    event, qualifiers = parse_counterfactual_event(condition, entities)
    if event is None:
        return None
    target = extract_query_target(question, ("会", "哪里", "哪儿", "什么地方", "在"), entities)
    return Query("counterfactual_location", target, (f"without_event={event}", *qualifiers))


def parse_initial_location_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    if ("最开始" in sentence or "一开始" in sentence or "起初" in sentence) and (
        "哪里" in sentence or "哪儿" in sentence
    ):
        target = extract_query_target(sentence, ("最开始", "一开始", "起初", "哪里", "哪儿", "在"), entities)
        return Query("initial_location", target)
    return None


def parse_location_before_actor_action_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    if "之前" not in sentence or ("哪里" not in sentence and "哪儿" not in sentence):
        return None
    actor = entity_in_text(sentence, entities, ("person", "giver", "receiver"))
    target = entity_in_text(sentence, entities, ("item", "container", "thing"))
    if actor is None or target is None:
        return None
    return Query("location_before_actor_action", target, (f"actor={actor}",))


def parse_temporal_location_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    if not any(word in sentence for word in ("哪里", "哪儿", "什么地方")):
        return None

    for marker, intent in (("之前", "location_before_event"), ("之后", "location_after_event")):
        if marker not in sentence:
            continue
        anchor_text, question_text = sentence.split(marker, 1)
        anchor_text = strip_temporal_anchor_prefix(anchor_text)
        question_text = question_text.strip(" ，,")
        if not anchor_text or not question_text:
            continue
        anchor_frame = parse_anchor_frame(anchor_text)
        if anchor_frame is None:
            continue
        target = extract_query_target(question_text, ("哪里", "哪儿", "什么地方", "在"), entities)
        return Query(
            intent,
            target,
            temporal_event_qualifiers(anchor_text, anchor_frame),
        )
    return None


def parse_temporal_contents_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    if "什么" not in sentence or "有" not in sentence:
        return None

    for marker, intent in (("之前", "contents_before_event"), ("之后", "contents_after_event")):
        if marker not in sentence:
            continue
        anchor_text, question_text = sentence.split(marker, 1)
        anchor_text = strip_temporal_anchor_prefix(anchor_text)
        question_text = question_text.strip(" ，,")
        if not anchor_text or not question_text:
            continue
        anchor_frame = parse_anchor_frame(anchor_text)
        if anchor_frame is None:
            continue
        holder = entity_before(question_text, "里", entities) or entity_in_text(
            question_text, entities, ("place", "container", "person", "giver", "receiver")
        )
        if holder is None:
            continue
        return Query(
            intent,
            holder,
            temporal_event_qualifiers(anchor_text, anchor_frame),
        )
    return None


def parse_events_after_event_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    if "之后" not in sentence or "发生" not in sentence or "什么" not in sentence:
        return None
    normalized = normalize_question(sentence)
    put_in = re.search(r"(?P<item>[^，,。？！?]+?)(?:被)?放进(?P<holder>.+?)之后", normalized)
    if put_in:
        return Query(
            "events_after_event",
            "put_in",
            (
                f"item={normalize_entity_slot(put_in.group('item'), entities)}",
                f"holder={normalize_entity_slot(normalize_container_slot(put_in.group('holder')), entities)}",
            ),
        )
    take_out_patterns = (
        re.compile(r"(?P<item>[^，,。？！?]+?)从(?P<source>.+?)取出之后"),
        re.compile(r"(?P<item>[^，,。？！?]+?)被.+?从(?P<source>.+?)取出之后"),
    )
    for take_out in take_out_patterns:
        match = take_out.search(normalized)
        if not match:
            continue
        return Query(
            "events_after_event",
            "take_out",
            (
                f"item={normalize_entity_slot(match.group('item'), entities)}",
                f"source={normalize_entity_slot(normalize_container_slot(match.group('source')), entities)}",
            ),
        )
    return None


def parse_location_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    if "哪里" in sentence or "哪儿" in sentence or "什么地方" in sentence:
        target = extract_query_target(sentence, ("哪里", "哪儿", "什么地方", "在"), entities)
        return Query("location", target)
    return None


def parse_owner_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    if "谁" in sentence and "拥有" in sentence:
        target = extract_query_target(sentence, ("谁", "拥有"), entities)
        return Query("owner", target)
    return None


def parse_color_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    if "颜色" in sentence and "什么" in sentence:
        target = extract_query_target(sentence, ("什么", "颜色"), entities)
        return Query("color", target)
    return None


def parse_object_state_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    if "状态" in sentence and "什么" in sentence:
        target = extract_query_target(sentence, ("什么", "状态"), entities)
        return Query("object_state", target, ("state=access",))
    if "打开还是关闭" in sentence or "关闭还是打开" in sentence:
        target = extract_query_target(sentence, ("打开还是关闭", "关闭还是打开"), entities)
        return Query("object_state", target, ("state=access",))
    return None


def parse_existence_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    if "存在" in sentence:
        target = extract_query_target(sentence, ("存在", "是否", "还", "有", "没", "没有"), entities)
        return Query("existence", target)
    if sentence.endswith("在") and "哪里" not in sentence and "哪儿" not in sentence:
        target = extract_query_target(sentence, ("在", "还"), entities)
        return Query("existence", target)
    if sentence.startswith("有") and "什么" not in sentence:
        target = extract_query_target(sentence, ("有", "还"), entities)
        return Query("existence", target)
    return None


def parse_actor_for_item_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    if "谁" in sentence and ("最后" in sentence or "最近" in sentence) and any(
        word in sentence for word in ("拿", "处理", "接触")
    ):
        target = extract_query_target(sentence, ("最后", "最近", "谁", "拿", "处理", "接触", "过"), entities)
        return Query("latest_actor_for_item", target)

    if "谁" in sentence and "拿" in sentence:
        target = extract_query_target(sentence, ("谁", "拿"), entities)
        return Query("actor_for_item", target)
    return None


def parse_places_visited_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    if "经过" in sentence and ("哪些地方" in sentence or "什么地方" in sentence or "哪里" in sentence):
        target = extract_query_target(sentence, ("经过", "哪些地方", "什么地方", "哪里", "哪儿"), entities)
        return Query("places_visited", target)
    return None


def parse_actions_by_actors_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    if "什么" not in sentence or not any(word in sentence for word in ("做", "干")):
        return None
    actors = entities_in_text(sentence, entities, ("person", "giver", "receiver"))
    if not actors:
        return None
    return Query("actions_by_actors", "和".join(actors), (f"actors={'|'.join(actors)}",))


def parse_inventories_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    if "手里" in sentence and "什么" in sentence:
        return Query("inventories", "person")
    return None


def parse_contents_except_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    if "除了" not in sentence or "什么" not in sentence or "有" not in sentence:
        return None
    holder = entity_before(sentence, "里", entities) or entity_in_text(sentence, entities, ("place", "container"))
    excluded = entity_after(sentence, "除了", entities)
    if holder is None or excluded is None:
        return None
    return Query("contents_except", holder, (f"exclude={excluded}",))


def parse_count_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    if not any(word in sentence for word in ("几个", "多少", "数量")):
        return None
    if not any(word in sentence for word in ("有", "数量", "几个", "多少")):
        return None
    holder = entity_before(sentence, "里", entities) or entity_in_text(
        sentence,
        entities,
        ("place", "container", "person", "giver", "receiver"),
    )
    if holder is None:
        return None
    return Query("count", holder)


def parse_count_comparison_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    if not any(word in sentence for word in ("更多", "多一些", "一样多", "哪个多", "哪里多")):
        return None
    holders = entities_in_text(sentence, entities, ("place", "container"))
    if len(holders) < 2:
        return None
    left, right = holders[:2]
    return Query("compare_count", f"{left}和{right}", (f"left={left}", f"right={right}"))


def parse_contents_query(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    if "什么" in sentence and "有" in sentence:
        target = extract_query_target(sentence, ("什么", "有", "里", "至少"), entities)
        return Query("contents", target)
    return None


def strip_temporal_anchor_prefix(anchor: str) -> str:
    normalized = normalize_slot_value(anchor).strip(" ，,")
    for prefix in ("在", "当", "等到", "等"):
        if normalized.startswith(prefix) and len(normalized) > len(prefix):
            normalized = normalized[len(prefix) :]
    return normalize_clause_text(normalized)


def parse_anchor_frame(anchor: str):
    parsed = parse_effect_clause(anchor)
    if parsed is None:
        return None
    _, frames = parsed
    return next((frame for frame in frames if frame.frame_type != "handle"), None)


def temporal_event_qualifiers(anchor: str, frame) -> tuple[str, ...]:
    qualifiers = [f"anchor={normalize_clause_text(anchor)}", f"event={frame.frame_type}"]
    for role_name in ("actor", "theme", "goal", "source", "recipient", "result"):
        value = frame.role(role_name)
        if value is not None:
            qualifiers.append(f"{role_name}={value}")
    return tuple(qualifiers)


def parse_put_in_event_question_semantic(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    normalized = normalize_question(sentence).strip()
    if "放进" not in normalized or "谁" not in normalized:
        return None
    left, right = normalized.split("放进", 1)
    item = extract_event_item(left, entities, ("把", "是", "被"))
    if item is None:
        return None
    holder = normalize_entity_slot(normalize_container_slot(right), entities)
    return Query("actor_for_event", "put_in", (f"item={item}", f"holder={holder}"))


def parse_take_out_event_question_semantic(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    normalized = normalize_take_out_expression(normalize_question(sentence)).strip()
    if "取出" not in normalized or "谁" not in normalized:
        return None
    left, right = normalized.split("取出", 1)
    if "从" not in left:
        return None
    before_source, source_text = left.rsplit("从", 1)
    item = extract_take_out_event_item(before_source, right, entities)
    source = normalize_entity_slot(normalize_container_slot(source_text), entities)
    if item is None:
        return None
    return Query("actor_for_event", "take_out", (f"item={item}", f"source={source}"))


def parse_earliest_put_in_event_question_semantic(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    normalized = normalize_question(sentence).strip()
    if "放进" not in normalized or "谁" not in normalized or not any(prefix in normalized for prefix in ("最先", "先")):
        return None
    query = parse_put_in_event_question_semantic(normalized, entities)
    if query is None:
        return None
    return Query("earliest_actor_for_event", query.target, query.qualifiers)


def parse_earliest_take_out_event_question_semantic(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    normalized = normalize_take_out_expression(normalize_question(sentence)).strip()
    if "取出" not in normalized or "谁" not in normalized or not any(prefix in normalized for prefix in ("最先", "先")):
        return None
    query = parse_take_out_event_question_semantic(normalized, entities)
    if query is None:
        return None
    return Query("earliest_actor_for_event", query.target, query.qualifiers)


def parse_latest_put_in_event_question_semantic(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    normalized = normalize_question(sentence).strip()
    if "放进" not in normalized or "谁" not in normalized or not any(prefix in normalized for prefix in ("最后", "最近")):
        return None
    query = parse_put_in_event_question_semantic(normalized, entities)
    if query is None:
        return None
    return Query("latest_actor_for_event", query.target, query.qualifiers)


def parse_latest_take_out_event_question_semantic(sentence: str, entities: tuple[Entity, ...]) -> Optional[Query]:
    normalized = normalize_take_out_expression(normalize_question(sentence)).strip()
    if "取出" not in normalized or "谁" not in normalized or not any(prefix in normalized for prefix in ("最后", "最近")):
        return None
    query = parse_take_out_event_question_semantic(normalized, entities)
    if query is None:
        return None
    return Query("latest_actor_for_event", query.target, query.qualifiers)


def extract_event_item(text: str, entities: tuple[Entity, ...], markers: tuple[str, ...]) -> str | None:
    candidate = text
    for marker in markers:
        if marker not in candidate:
            continue
        before, after = candidate.split(marker, 1)
        if marker in {"把", "是"}:
            preferred = entity_in_text(after, entities, ("item", "container", "thing")) or entity_in_text(
                before, entities, ("item", "container", "thing")
            )
        elif marker == "被":
            preferred = entity_in_text(before, entities, ("item", "container", "thing")) or entity_in_text(
                after, entities, ("item", "container", "thing")
            )
        else:
            preferred = entity_in_text(before, entities, ("item", "container", "thing")) or entity_in_text(
                after, entities, ("item", "container", "thing")
            )
        if preferred is not None:
            return preferred
    matches = [entity.name for entity in entities if entity.role in ("item", "container", "thing") and entity.name in candidate]
    if matches:
        return max(matches, key=len)
    normalized = normalize_entity_slot(candidate, entities)
    return normalized if normalized else None


def extract_take_out_event_item(before_source: str, after_action: str, entities: tuple[Entity, ...]) -> str | None:
    right_item = entity_in_text(after_action, entities, ("item", "container", "thing"))
    if right_item is not None:
        return right_item

    if "把" in before_source:
        _, after = before_source.rsplit("把", 1)
        item = entity_in_text(after, entities, ("item", "container", "thing"))
        if item is not None:
            return item

    if "被" in before_source:
        before, after = before_source.split("被", 1)
        item = entity_in_text(before, entities, ("item", "container", "thing")) or entity_in_text(
            after, entities, ("item", "container", "thing")
        )
        if item is not None:
            return item

    if "是" in before_source:
        before, after = before_source.split("是", 1)
        item = entity_in_text(before, entities, ("item", "container", "thing"))
        if item is not None:
            return item
        if "谁" not in after:
            item = entity_in_text(after, entities, ("item", "container", "thing"))
            if item is not None:
                return item

    return entity_in_text(before_source, entities, ("item", "container", "thing"))


DEFAULT_QUERY_PARSERS: tuple[QueryParser, ...] = (
    parse_dialog_act_query,
    parse_profile_query,
    parse_earliest_event_query,
    parse_latest_event_query,
    parse_event_query,
    parse_why_query,
    parse_claim_source_query,
    parse_belief_query,
    parse_contradiction_query,
    parse_polar_query,
    parse_counterfactual_location_query,
    parse_initial_location_query,
    parse_temporal_location_query,
    parse_temporal_contents_query,
    parse_location_before_actor_action_query,
    parse_events_after_event_query,
    parse_count_comparison_query,
    parse_location_query,
    parse_owner_query,
    parse_color_query,
    parse_object_state_query,
    parse_existence_query,
    parse_actor_for_item_query,
    parse_places_visited_query,
    parse_actions_by_actors_query,
    parse_inventories_query,
    parse_contents_except_query,
    parse_count_query,
    parse_contents_query,
)


def extract_query_target(
    sentence: str,
    intent_words: tuple[str, ...],
    entities: tuple[Entity, ...],
) -> str:
    target = sentence
    words = sorted(set((*intent_words, *QUESTION_FILLERS)), key=len, reverse=True)
    for word in words:
        target = target.replace(word, "")
    if not target:
        raise ParseError(f"Cannot extract query target from question: {sentence}")
    normalized = normalize_entity_slot(target, entities)
    if normalized in {"前者", "后者"}:
        raise ParseError(f"Cannot resolve query target from question: {sentence}")
    return normalized


def entity_in_text(sentence: str, entities: tuple[Entity, ...], roles: tuple[str, ...]) -> str | None:
    matches = [entity.name for entity in entities if entity.role in roles and entity.name in sentence]
    if not matches:
        return None
    return max(matches, key=len)


def entities_in_text(sentence: str, entities: tuple[Entity, ...], roles: tuple[str, ...]) -> tuple[str, ...]:
    matches = []
    for entity in entities:
        if entity.role in roles and entity.name in sentence and entity.name not in matches:
            matches.append(entity.name)
    return tuple(matches)


def entity_before(sentence: str, marker: str, entities: tuple[Entity, ...]) -> str | None:
    if marker not in sentence:
        return None
    before = sentence.split(marker, 1)[0]
    matches = [entity.name for entity in entities if entity.name in before]
    if not matches:
        return None
    return max(matches, key=len)


def entity_after(sentence: str, marker: str, entities: tuple[Entity, ...]) -> str | None:
    if marker not in sentence:
        return None
    after = sentence.split(marker, 1)[1]
    matches = [entity.name for entity in entities if entity.name in after]
    if not matches:
        return None
    return max(matches, key=len)


def first_present(sentence: str, words: tuple[str, ...]) -> str | None:
    matches = [word for word in words if word in sentence]
    if not matches:
        return None
    return min(matches, key=sentence.index)


def find_last_marker(sentence: str, marker: str) -> int | None:
    index = sentence.rfind(marker)
    return index if index >= 0 else None


def split_counterfactual(sentence: str, entities: tuple[Entity, ...]) -> tuple[str, str] | None:
    body = sentence.removeprefix("如果")
    for separator in ("，", ","):
        if separator in body:
            condition, question = body.split(separator, 1)
            return normalize_slot_value(condition), question
    for marker in ("会在哪里", "会在哪儿", "会在什么地方", "在哪里", "在哪儿", "在什么地方"):
        if marker in body:
            before, _ = body.split(marker, 1)
            query_targets = [
                entity.name
                for entity in entities
                if entity.name in before and entity.role in ("item", "container", "thing")
            ]
            if query_targets:
                target = max(query_targets, key=lambda value: before.rfind(value))
                question_start = before.rfind(target)
            else:
                question_start = before.rfind("它")
            if question_start < 0:
                return None
            return normalize_slot_value(before[:question_start]), before[question_start:] + marker
    return None


def parse_counterfactual_event(condition: str, entities: tuple[Entity, ...]) -> tuple[str | None, tuple[str, ...]]:
    normalized = normalize_take_out_expression(condition)
    for pattern in COUNTERFACTUAL_TAKE_OUT_RE:
        match = pattern.match(normalized)
        if match:
            data = match.groupdict()
            return (
                "take_out",
                (
                    f"actor={normalize_entity_slot(data['actor'], entities)}",
                    f"theme={normalize_entity_slot(data['theme'], entities)}",
                    f"source={normalize_entity_slot(normalize_container_slot(data['source']), entities)}",
                ),
            )
    for event, pattern in COUNTERFACTUAL_EVENT_RE:
        match = pattern.match(normalized)
        if not match:
            continue
        data = match.groupdict()
        right_role = "holder" if event == "put_in" else "goal"
        right_value = normalize_container_slot(data["goal"]) if event == "put_in" else normalize_slot_value(data["goal"])
        return (
            event,
            (
                f"actor={normalize_entity_slot(data['actor'], entities)}",
                f"theme={normalize_entity_slot(data['theme'], entities)}",
                f"{right_role}={normalize_entity_slot(right_value, entities)}",
            ),
        )
    return None, ()
