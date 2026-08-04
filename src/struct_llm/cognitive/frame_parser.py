from __future__ import annotations

import re

from .capabilities import StatementParser, StatementParseResult
from .normalization import (
    normalize_container_slot,
    normalize_containment_expression,
    normalize_slot_value,
    normalize_take_out_expression,
)
from ..structure import Entity, Frame, Role


CONTAINMENT_VERBS = ("放到", "放入", "放进")
TAKE_OUT_VERBS = ("取出来", "拿出来", "取出", "拿出", "取走", "拿走")
CONTAINER_SUFFIXES = ("里面", "里边", "里头", "内部", "里", "内", "中")


def first_marker(text: str, markers: tuple[str, ...]) -> tuple[int, str] | None:
    matches = [(text.find(marker), marker) for marker in markers if marker in text]
    if not matches:
        return None
    return min(matches, key=lambda item: item[0])


def split_once(text: str, marker: str) -> tuple[str, str] | None:
    if marker not in text:
        return None
    before, after = text.split(marker, 1)
    return before, after


def strip_any_prefix(text: str, prefixes: tuple[str, ...]) -> str:
    normalized = text
    changed = True
    while changed:
        changed = False
        for prefix in sorted(prefixes, key=len, reverse=True):
            if normalized.startswith(prefix) and len(normalized) > len(prefix):
                normalized = normalized[len(prefix) :]
                changed = True
    return normalized


def parse_surface_statement(sentence: str) -> StatementParseResult | None:
    parsed = parse_surface_control_statement(sentence)
    if parsed is not None:
        return parsed
    parsed = parse_surface_event_statement(sentence)
    if parsed is not None:
        return parsed
    return None


def parse_surface_control_statement(sentence: str) -> StatementParseResult | None:
    normalized = normalize_slot_value(sentence).strip().rstrip("。！？!?")

    parsed = split_if_then(normalized)
    if parsed is not None:
        antecedent, consequent = parsed
        return (
            [],
            [frame_from_roles("if_then", antecedent=normalize_clause_text(antecedent), consequent=normalize_clause_text(consequent))],
        )

    for prefix in ("因为", "由于"):
        if not normalized.startswith(prefix):
            continue
        body = normalized[len(prefix) :]
        for separator in ("所以", "因此", "就"):
            if separator not in body:
                continue
            cause, effect = body.split(separator, 1)
            effect = effect.lstrip("，,")
            effect_result = parse_effect_clause(normalize_clause_text(effect))
            frames = [frame_from_roles("because", cause=normalize_clause_text(cause), effect=normalize_clause_text(effect))]
            if effect_result is not None:
                effect_entities, effect_frames = effect_result
                return effect_entities, frames + effect_frames
            return [], frames

    for prefix in ("据",):
        if "说" not in normalized or not normalized.startswith(prefix):
            continue
        body = normalized[len(prefix) :]
        speaker, proposition = body.split("说", 1)
        proposition = proposition.lstrip("：:，,")
        return (
            [Entity("person", normalize_slot_value(speaker))],
            [frame_from_roles("say", speaker=normalize_slot_value(speaker), proposition=normalize_clause_text(proposition))],
        )

    if "说" in normalized and not normalized.startswith("据"):
        speaker, proposition = normalized.split("说", 1)
        proposition = proposition.lstrip("：:，,")
        if speaker and proposition:
            return (
                [Entity("person", normalize_slot_value(speaker))],
                [frame_from_roles("say", speaker=normalize_slot_value(speaker), proposition=normalize_clause_text(proposition))],
            )

    for verb in ("认为", "相信", "觉得", "以为"):
        if verb not in normalized:
            continue
        person, proposition = normalized.split(verb, 1)
        if person and proposition:
            return (
                [Entity("person", normalize_slot_value(person))],
                [frame_from_roles("believe", person=normalize_slot_value(person), proposition=normalize_clause_text(proposition))],
            )

    profile = parse_profile_statement(normalized)
    if profile is not None:
        return profile

    return None


def parse_surface_event_statement(sentence: str) -> StatementParseResult | None:
    normalized = normalize_slot_value(sentence).strip().rstrip("。！？!?")

    correction = parse_surface_location_correction(normalized)
    if correction is not None:
        return correction

    negated = parse_surface_location_negation(normalized)
    if negated is not None:
        return negated

    located = parse_surface_location_statement(normalized)
    if located is not None:
        return located

    if any(marker in normalized for marker in CONTAINMENT_VERBS):
        normalized = normalize_containment_expression(normalized)
        marker = first_marker(normalized, ("放进",))
        if marker is not None and "把" in normalized[: marker[0]]:
            before, after = normalized[: marker[0]], normalized[marker[0] + len(marker[1]) :]
            subject_part, item_part = before.split("把", 1) if "把" in before else ("", "")
            item = normalize_slot_value(item_part)
            container = normalize_container_slot(after)
            if subject_part and item and container:
                return (
                    [
                        Entity("person", normalize_slot_value(subject_part)),
                        Entity("item", item),
                        Entity("container", container),
                    ],
                    [frame_from_roles("put_in", actor=normalize_slot_value(subject_part), theme=item, goal=container), handle_frame(normalize_slot_value(subject_part), item)],
                )

    take_out_markers = ("取出",)
    normalized_take_out = normalize_take_out_expression(normalized)
    if any(marker in normalized_take_out for marker in take_out_markers):
        marker = first_marker(normalized_take_out, take_out_markers)
        if marker is not None:
            before, after = normalized_take_out[: marker[0]], normalized_take_out[marker[0] + len(marker[1]) :]
            if "把" in before and "从" in before:
                actor_part, rest = before.split("把", 1)
                item_part, source_part = rest.split("从", 1)
                return (
                    [
                        Entity("person", normalize_slot_value(actor_part)),
                        Entity("item", normalize_slot_value(item_part)),
                        Entity("container", normalize_container_slot(source_part)),
                    ],
                    [frame_from_roles("take_out", actor=normalize_slot_value(actor_part), theme=normalize_slot_value(item_part), source=normalize_container_slot(source_part)), handle_frame(normalize_slot_value(actor_part), normalize_slot_value(item_part))],
                )
            if "被" in before and "从" in before and before.index("被") < before.index("从"):
                item_part, rest = before.split("被", 1)
                actor_part, source_part = rest.split("从", 1)
                return (
                    [
                        Entity("person", normalize_slot_value(actor_part)),
                        Entity("item", normalize_slot_value(item_part)),
                        Entity("container", normalize_container_slot(source_part)),
                    ],
                    [frame_from_roles("take_out", actor=normalize_slot_value(actor_part), theme=normalize_slot_value(item_part), source=normalize_container_slot(source_part)), handle_frame(normalize_slot_value(actor_part), normalize_slot_value(item_part))],
                )
            if "从" in before and "被" in before and before.index("从") < before.index("被"):
                item_part, rest = before.split("从", 1)
                source_part, _ = rest.split("被", 1)
                return (
                    [
                        Entity("item", normalize_slot_value(item_part)),
                        Entity("container", normalize_container_slot(source_part)),
                    ],
                    [frame_from_roles("take_out", theme=normalize_slot_value(item_part), source=normalize_container_slot(source_part))],
                )
            if "从" in before:
                actor_part, source_part = before.split("从", 1)
                item = normalize_slot_value(after)
                return (
                    [
                        Entity("person", normalize_slot_value(actor_part)),
                        Entity("item", item),
                        Entity("container", normalize_container_slot(source_part)),
                    ],
                    [frame_from_roles("take_out", actor=normalize_slot_value(actor_part), theme=item, source=normalize_container_slot(source_part)), handle_frame(normalize_slot_value(actor_part), item)],
                )

    move_marker = first_marker(normalized, ("带到",))
    if move_marker is not None:
        before, after = normalized[: move_marker[0]], normalized[move_marker[0] + len(move_marker[1]) :]
        if "把" in before:
            actor_part, thing_part = before.split("把", 1)
            return (
                [
                    Entity("person", normalize_slot_value(actor_part)),
                    Entity(moved_role(thing_part), normalize_slot_value(thing_part)),
                    Entity("place", normalize_slot_value(after)),
                ],
                [frame_from_roles("move", actor=normalize_slot_value(actor_part), theme=normalize_slot_value(thing_part), goal=normalize_slot_value(after)), handle_frame(normalize_slot_value(actor_part), normalize_slot_value(thing_part))],
            )
        if "被" in before:
            thing_part, actor_part = before.split("被", 1)
            return (
                [
                    Entity(moved_role(thing_part), normalize_slot_value(thing_part)),
                    Entity("place", normalize_slot_value(after)),
                ],
                [frame_from_roles("move", actor=normalize_slot_value(actor_part), theme=normalize_slot_value(thing_part), goal=normalize_slot_value(after))],
            )

    if "交给" in normalized:
        giver_part, receiver_part = normalized.split("交给", 1)
        if "把" in giver_part:
            actor_part, item_part = giver_part.split("把", 1)
            return (
                [
                    Entity("giver", normalize_slot_value(actor_part)),
                    Entity("receiver", normalize_slot_value(receiver_part)),
                    Entity("item", normalize_slot_value(item_part)),
                ],
                [frame_from_roles("give", actor=normalize_slot_value(actor_part), theme=normalize_slot_value(item_part), recipient=normalize_slot_value(receiver_part)), handle_frame(normalize_slot_value(actor_part), normalize_slot_value(item_part))],
            )

    if "涂成" in normalized:
        person_part, rest = normalized.split("涂成", 1)
        if "把" in person_part:
            actor_part, item_part = person_part.split("把", 1)
            return (
                [
                    Entity("person", normalize_slot_value(actor_part)),
                    Entity("item", normalize_slot_value(item_part)),
                    Entity("color", normalize_slot_value(rest)),
                ],
                [frame_from_roles("paint", actor=normalize_slot_value(actor_part), theme=normalize_slot_value(item_part), result=normalize_slot_value(rest)), handle_frame(normalize_slot_value(actor_part), normalize_slot_value(item_part))],
            )

    for action, frame_type in (("打开", "open"), ("关闭", "close"), ("关上", "close"), ("合上", "close")):
        if action not in normalized:
            continue
        before, after = normalized.split(action, 1)
        if "被" in before:
            thing_part, actor_part = before.split("被", 1)
            thing = normalize_slot_value(thing_part)
            result = "打开" if frame_type == "open" else "关闭"
            return (
                [Entity("person", normalize_slot_value(actor_part)), Entity(moved_role(thing), thing)],
                [frame_from_roles(frame_type, actor=normalize_slot_value(actor_part), theme=thing, result=result), handle_frame(normalize_slot_value(actor_part), thing)],
            )
        if "把" in before:
            actor_part, thing_part = before.split("把", 1)
            thing = normalize_slot_value(thing_part)
            result = "打开" if frame_type == "open" else "关闭"
            return (
                [Entity("person", normalize_slot_value(actor_part)), Entity(moved_role(thing), thing)],
                [frame_from_roles(frame_type, actor=normalize_slot_value(actor_part), theme=thing, result=result), handle_frame(normalize_slot_value(actor_part), thing)],
            )
        if before and after:
            thing = normalize_slot_value(after)
            result = "打开" if frame_type == "open" else "关闭"
            return (
                [Entity("person", normalize_slot_value(before)), Entity(moved_role(thing), thing)],
                [frame_from_roles(frame_type, actor=normalize_slot_value(before), theme=thing, result=result), handle_frame(normalize_slot_value(before), thing)],
            )
        if before and after == "":
            thing = normalize_slot_value(before)
            result = "打开" if frame_type == "open" else "关闭"
            return (
                [Entity(moved_role(thing), thing)],
                [frame_from_roles(frame_type, theme=thing, result=result)],
            )

    for action, frame_type, result in (
        ("制造出来", "create", "存在"),
        ("制造", "create", "存在"),
        ("创建", "create", "存在"),
        ("生成", "create", "存在"),
        ("销毁", "destroy", "不存在"),
        ("删除", "destroy", "不存在"),
        ("消灭", "destroy", "不存在"),
    ):
        if action not in normalized:
            continue
        before, after = normalized.split(action, 1)
        if "被" in before:
            thing_part, actor_part = before.split("被", 1)
            thing = normalize_slot_value(thing_part)
            return (
                [Entity("person", normalize_slot_value(actor_part)), Entity(moved_role(thing), thing)],
                [frame_from_roles(frame_type, actor=normalize_slot_value(actor_part), theme=thing, result=result), handle_frame(normalize_slot_value(actor_part), thing)],
            )
        if "把" in before:
            actor_part, thing_part = before.split("把", 1)
            thing = normalize_slot_value(thing_part or after)
            return (
                [Entity("person", normalize_slot_value(actor_part)), Entity(moved_role(thing), thing)],
                [frame_from_roles(frame_type, actor=normalize_slot_value(actor_part), theme=thing, result=result), handle_frame(normalize_slot_value(actor_part), thing)],
            )
        if before and after:
            actor_part = before
            thing = normalize_slot_value(after)
            return (
                [Entity("person", normalize_slot_value(actor_part)), Entity(moved_role(thing), thing)],
                [frame_from_roles(frame_type, actor=normalize_slot_value(actor_part), theme=thing, result=result), handle_frame(normalize_slot_value(actor_part), thing)],
            )

    return None


def parse_surface_location_correction(sentence: str) -> StatementParseResult | None:
    split = split_location_negation(sentence)
    if split is None:
        return None
    item, rest = split
    corrected = split_location_correction(rest)
    if corrected is None:
        return None
    old_container, new_container = corrected
    return (
        [
            Entity("item", item),
            Entity("container", old_container),
            Entity("container", new_container),
        ],
        [
            frame_from_roles("not_in", theme=item, source=old_container),
            frame_from_roles("be_in", theme=item, goal=new_container),
        ],
    )


def parse_surface_location_negation(sentence: str) -> StatementParseResult | None:
    for marker in ("没有", "不包含", "没"):
        if marker not in sentence:
            continue
        holder_text, item_text = sentence.split(marker, 1)
        if has_container_suffix(holder_text) and item_text:
            container = normalize_container_slot(holder_text)
            item = normalize_slot_value(item_text)
            return (
                [Entity("item", item), Entity("container", container)],
                [frame_from_roles("not_in", theme=item, source=container)],
            )

    split = split_location_negation(sentence)
    if split is None:
        return None
    item, container_text = split
    if not item or not container_text:
        return None
    container = normalize_container_slot(container_text)
    return (
        [Entity("item", item), Entity("container", container)],
        [frame_from_roles("not_in", theme=item, source=container)],
    )


def parse_surface_location_statement(sentence: str) -> StatementParseResult | None:
    if "在" not in sentence:
        return None
    item_text, location_text = sentence.split("在", 1)
    item = normalize_slot_value(item_text)
    location = normalize_slot_value(location_text)
    if not item or not location:
        return None

    if "的" in location:
        place_text, container_text = location.split("的", 1)
        place = normalize_slot_value(place_text)
        container = normalize_container_slot(container_text)
        if place and container:
            return (
                [
                    Entity("item", item),
                    Entity("place", place),
                    Entity("container", container),
                ],
                [
                    frame_from_roles("move", theme=container, goal=place),
                    frame_from_roles("be_in", theme=item, goal=container),
                ],
            )

    if has_container_suffix(location):
        container = normalize_container_slot(location)
        return (
            [Entity("item", item), Entity("container", container)],
            [frame_from_roles("be_in", theme=item, goal=container)],
        )

    return (
        [Entity(moved_role(item), item), Entity("place", location)],
        [frame_from_roles("move", theme=item, goal=location)],
    )


def split_location_negation(sentence: str) -> tuple[str, str] | None:
    for marker in ("没有在", "不是在", "不在", "没在"):
        if marker not in sentence:
            continue
        item_text, container_text = sentence.split(marker, 1)
        item = normalize_slot_value(item_text)
        if item:
            return item, container_text
    return None


def split_location_correction(rest: str) -> tuple[str, str] | None:
    normalized = rest.strip(" ，,")
    for connector in ("而是在", "，是在", ",是在", "，在", ",在", "而在", "是在"):
        if connector not in normalized:
            continue
        old_text, new_text = normalized.split(connector, 1)
        old_container = normalize_container_slot(old_text.strip(" ，,"))
        new_container = normalize_container_slot(new_text.strip(" ，,"))
        if old_container and new_container:
            return old_container, new_container
    return None


def has_container_suffix(text: str) -> bool:
    normalized = normalize_slot_value(text)
    return any(normalized.endswith(suffix) and len(normalized) > len(suffix) for suffix in CONTAINER_SUFFIXES)


PUT_IN_RE = re.compile(r"^(?P<person>.+?)把(?P<item>.+?)放进(?P<container>.+?)$")
BECAUSE_RE = re.compile(r"^(?:因为|由于)(?P<cause>.+?)(?:，|,)?(?:所以|因此|就)(?P<effect>.+?)$")
REPORT_RE = re.compile(r"^据(?P<speaker>.+?)说[：:，,]?(?P<proposition>.+?)$")
SAY_RE = re.compile(r"^(?P<speaker>.+?)说[：:，,]?(?P<proposition>.+?)$")
BELIEVE_RE = re.compile(r"^(?P<person>.+?)(?:认为|相信|觉得|以为)(?P<proposition>.+?)$")
PROFILE_NAME_RE = re.compile(r"^(?P<subject>我|本人|自己)(?:的(?:名字|姓名)(?:叫|是)?|叫|是)(?P<value>.+?)$")
PROFILE_PREFERENCE_RE = re.compile(r"^(?P<subject>我|本人|自己)(?P<verb>不喜欢|讨厌|喜欢|爱)(?P<value>.+?)$")
ACTIVE_TAKE_OUT_RE = re.compile(r"^(?P<person>.+?)把(?P<item>.+?)从(?P<container>.+?)取出$")
FRONTED_TAKE_OUT_RE = re.compile(r"^(?P<person>.+?)从(?P<container>.+?)取出(?P<item>.+?)$")
PASSIVE_TAKE_OUT_RE = re.compile(r"^(?P<item>.+?)被(?P<person>.+?)从(?P<container>.+?)取出$")
PASSIVE_SOURCE_TAKE_OUT_RE = re.compile(r"^(?P<item>.+?)从(?P<container>.+?)被取出$")
NEGATED_IN_RE = re.compile(r"^(?P<item>.+?)(?:没有|没|不)在(?P<container>.+?)$")
NEGATED_CONTENT_RE = re.compile(
    r"^(?P<container>.+?(?:里面|里边|里头|内部|里|内|中))(?:没有|不包含|没)(?P<item>.+?)$"
)
CORRECTED_IN_RE = re.compile(
    r"^(?P<item>.+?)(?:没有|不是|没|不)在(?P<old>.+?)(?:，)?(?:而是|是在|而在|在|是)(?P<new>.+?)$"
)
LOCATED_IN_RE = re.compile(
    r"^(?P<item>.+?)在(?P<place>.+?)的(?P<container>.+?(?:里面|里边|里头|内部|里|内|中))$"
)
SIMPLE_IN_RE = re.compile(r"^(?P<item>.+?)在(?P<container>.+?(?:里面|里边|里头|内部|里|内|中))$")
PASSIVE_MOVE_RE = re.compile(r"^(?P<thing>.+?)被带到(?P<place>.+?)$")
ACTIVE_MOVE_RE = re.compile(r"^(?P<person>.+?)把(?P<thing>.+?)带到(?P<place>.+?)$")
GIVE_RE = re.compile(r"^(?P<giver>.+?)把(?P<item>.+?)交给(?P<receiver>.+?)$")
PAINT_RE = re.compile(r"^(?P<person>.+?)把(?P<item>.+?)涂成(?P<color>.+?)$")
ACTIVE_OPEN_RE = re.compile(r"^(?P<person>.+?)(?:把(?P<thing1>.+?)打开|打开(?P<thing2>.+?))$")
PASSIVE_OPEN_RE = re.compile(r"^(?P<thing>.+?)被(?P<person>.+?)?打开$")
ACTIVE_CLOSE_RE = re.compile(r"^(?P<person>.+?)(?:把(?P<thing1>.+?)(?:关闭|关上|合上)|(?:关闭|关上|合上)(?P<thing2>.+?))$")
PASSIVE_CLOSE_RE = re.compile(r"^(?P<thing>.+?)被(?P<person>.+?)?(?:关闭|关上|合上)$")
ACTIVE_CREATE_RE = re.compile(r"^(?P<person>.+?)(?:把(?P<thing1>.+?)制造出来|制造(?P<thing2>.+?)|创建(?P<thing3>.+?)|生成(?P<thing4>.+?))$")
PASSIVE_CREATE_RE = re.compile(r"^(?P<thing>.+?)被(?P<person>.+?)?(?:制造出来|创建|生成)$")
ACTIVE_DESTROY_RE = re.compile(r"^(?P<person>.+?)(?:把(?P<thing1>.+?)销毁|销毁(?P<thing2>.+?)|删除(?P<thing3>.+?)|消灭(?P<thing4>.+?))$")
PASSIVE_DESTROY_RE = re.compile(r"^(?P<thing>.+?)被(?P<person>.+?)?(?:销毁|删除|消灭)$")


def parse_statement(
    sentence: str,
    parsers: tuple[StatementParser, ...] | None = None,
) -> StatementParseResult | None:
    for parser in parsers or DEFAULT_STATEMENT_PARSERS:
        parsed = parser(sentence)
        if parsed is not None:
            return parsed
    return None


def parse_if_then_statement(sentence: str) -> StatementParseResult | None:
    split = split_if_then(sentence)
    if split is None:
        return None
    antecedent, consequent = split
    return (
        [],
        [frame_from_roles("if_then", antecedent=normalize_clause_text(antecedent), consequent=normalize_clause_text(consequent))],
    )


def parse_because_statement(sentence: str) -> StatementParseResult | None:
    match = BECAUSE_RE.match(sentence)
    if not match:
        return None
    data = match.groupdict()
    cause_text = normalize_clause_text(data["cause"])
    effect_text = normalize_clause_text(data["effect"])
    effect = parse_effect_clause(effect_text)
    frames = [frame_from_roles("because", cause=cause_text, effect=effect_text)]
    if effect is not None:
        effect_entities, effect_frames = effect
        return (effect_entities, frames + effect_frames)
    return ([], frames)


def parse_report_statement(sentence: str) -> StatementParseResult | None:
    match = REPORT_RE.match(sentence)
    if not match:
        return None
    data = match.groupdict()
    speaker = normalize_slot_value(data["speaker"])
    proposition = normalize_clause_text(data["proposition"])
    return (
        [Entity("person", speaker)],
        [frame_from_roles("say", speaker=speaker, proposition=proposition)],
    )


def parse_say_statement(sentence: str) -> StatementParseResult | None:
    match = SAY_RE.match(sentence)
    if not match or sentence.startswith("据"):
        return None
    data = match.groupdict()
    speaker = normalize_slot_value(data["speaker"])
    proposition = normalize_clause_text(data["proposition"])
    return (
        [Entity("person", speaker)],
        [frame_from_roles("say", speaker=speaker, proposition=proposition)],
    )


def parse_believe_statement(sentence: str) -> StatementParseResult | None:
    match = BELIEVE_RE.match(sentence)
    if not match:
        return None
    data = match.groupdict()
    person = normalize_slot_value(data["person"])
    proposition = normalize_clause_text(data["proposition"])
    return (
        [Entity("person", person)],
        [frame_from_roles("believe", person=person, proposition=proposition)],
    )


def parse_profile_statement(sentence: str) -> StatementParseResult | None:
    normalized = normalize_slot_value(sentence)
    name = PROFILE_NAME_RE.match(normalized)
    if name:
        data = name.groupdict()
        subject = normalize_slot_value(data["subject"])
        value = normalize_slot_value(data["value"])
        return (
            [
                Entity("self", subject),
                Entity("profile_value", value),
            ],
            [frame_from_roles("profile_name", subject=subject, value=value)],
        )

    preference = PROFILE_PREFERENCE_RE.match(normalized)
    if not preference:
        return None
    data = preference.groupdict()
    subject = normalize_slot_value(data["subject"])
    value = normalize_slot_value(data["value"])
    frame_type = "profile_dislike" if data["verb"] in {"不喜欢", "讨厌"} else "profile_like"
    return (
        [
            Entity("self", subject),
            Entity("profile_value", value),
        ],
        [frame_from_roles(frame_type, subject=subject, value=value)],
    )


def parse_put_in_statement(sentence: str) -> StatementParseResult | None:
    sentence = normalize_containment_expression(sentence)
    put_in = PUT_IN_RE.match(sentence)
    if not put_in:
        return None
    data = put_in.groupdict()
    item = normalize_slot_value(data["item"])
    container = normalize_container_slot(data["container"])
    frame = frame_from_roles(
        "put_in",
        actor=data["person"],
        theme=item,
        goal=container,
    )
    return (
        [
            Entity("person", data["person"]),
            Entity("item", item),
            Entity("container", container),
        ],
        [frame, handle_frame(data["person"], item)],
    )


def parse_take_out_statement(sentence: str) -> StatementParseResult | None:
    normalized = normalize_take_out_expression(sentence)
    for pattern in (
        ACTIVE_TAKE_OUT_RE,
        FRONTED_TAKE_OUT_RE,
        PASSIVE_TAKE_OUT_RE,
        PASSIVE_SOURCE_TAKE_OUT_RE,
    ):
        take_out = pattern.match(normalized)
        if take_out:
            data = take_out.groupdict()
            actor = data.get("person")
            item = normalize_slot_value(data["item"])
            source = normalize_container_slot(data["container"])
            frames = [frame_from_roles("take_out", theme=item, source=source)]
            if actor:
                frames[0] = frame_from_roles("take_out", actor=actor, theme=item, source=source)
                frames.append(handle_frame(actor, item))
            return (
                [
                    *( [Entity("person", actor)] if actor else [] ),
                    Entity("item", item),
                    Entity("container", source),
                ],
                frames,
            )
    return None


def parse_negated_in_statement(sentence: str) -> StatementParseResult | None:
    normalized = normalize_slot_value(sentence)

    corrected = CORRECTED_IN_RE.match(normalized)
    if corrected:
        data = corrected.groupdict()
        item = normalize_slot_value(data["item"])
        old_container = normalize_container_slot(data["old"])
        new_container = normalize_container_slot(data["new"])
        return (
            [
                Entity("item", item),
                Entity("container", old_container),
                Entity("container", new_container),
            ],
            [
                frame_from_roles("not_in", theme=item, source=old_container),
                frame_from_roles("be_in", theme=item, goal=new_container),
            ],
        )

    negated_content = NEGATED_CONTENT_RE.match(normalized)
    if negated_content:
        data = negated_content.groupdict()
        item = normalize_slot_value(data["item"])
        container = normalize_container_slot(data["container"])
        return (
            [
                Entity("item", item),
                Entity("container", container),
            ],
            [frame_from_roles("not_in", theme=item, source=container)],
        )

    negated = NEGATED_IN_RE.match(normalized)
    if not negated:
        return None
    data = negated.groupdict()
    item = normalize_slot_value(data["item"])
    container = normalize_container_slot(data["container"])
    return (
        [
            Entity("item", item),
            Entity("container", container),
        ],
        [frame_from_roles("not_in", theme=item, source=container)],
    )


def parse_located_in_statement(sentence: str) -> StatementParseResult | None:
    contained = LOCATED_IN_RE.match(normalize_slot_value(sentence))
    if not contained:
        return None
    data = contained.groupdict()
    item = normalize_slot_value(data["item"])
    place = normalize_slot_value(data["place"])
    container = normalize_container_slot(data["container"])
    return (
        [
            Entity("item", item),
            Entity("place", place),
            Entity("container", container),
        ],
        [
            frame_from_roles("move", theme=container, goal=place),
            frame_from_roles("be_in", theme=item, goal=container),
        ],
    )


def parse_simple_in_statement(sentence: str) -> StatementParseResult | None:
    contained = SIMPLE_IN_RE.match(normalize_slot_value(sentence))
    if not contained:
        return None
    data = contained.groupdict()
    item = normalize_slot_value(data["item"])
    container = normalize_container_slot(data["container"])
    return (
        [
            Entity("item", item),
            Entity("container", container),
        ],
        [frame_from_roles("be_in", theme=item, goal=container)],
    )


def parse_passive_move_statement(sentence: str) -> StatementParseResult | None:
    move = PASSIVE_MOVE_RE.match(sentence)
    if not move:
        return None
    data = move.groupdict()
    thing = normalize_slot_value(data["thing"])
    place = normalize_slot_value(data["place"])
    return (
        [
            Entity(moved_role(thing), thing),
            Entity("place", place),
        ],
        [frame_from_roles("move", theme=thing, goal=place)],
    )


def parse_active_move_statement(sentence: str) -> StatementParseResult | None:
    active_move = ACTIVE_MOVE_RE.match(sentence)
    if not active_move:
        return None
    data = active_move.groupdict()
    thing = normalize_slot_value(data["thing"])
    place = normalize_slot_value(data["place"])
    frame = frame_from_roles("move", actor=data["person"], theme=thing, goal=place)
    return (
        [
            Entity("person", data["person"]),
            Entity(moved_role(thing), thing),
            Entity("place", place),
        ],
        [frame, handle_frame(data["person"], thing)],
    )


def parse_give_statement(sentence: str) -> StatementParseResult | None:
    give = GIVE_RE.match(sentence)
    if not give:
        return None
    data = give.groupdict()
    frame = frame_from_roles(
        "give",
        actor=data["giver"],
        theme=data["item"],
        recipient=data["receiver"],
    )
    return (
        [
            Entity("giver", data["giver"]),
            Entity("receiver", data["receiver"]),
            Entity("item", data["item"]),
        ],
        [frame, handle_frame(data["giver"], data["item"])],
    )


def parse_paint_statement(sentence: str) -> StatementParseResult | None:
    paint = PAINT_RE.match(sentence)
    if not paint:
        return None
    data = paint.groupdict()
    frame = frame_from_roles(
        "paint",
        actor=data["person"],
        theme=data["item"],
        result=data["color"],
    )
    return (
        [
            Entity("person", data["person"]),
            Entity("item", data["item"]),
            Entity("color", data["color"]),
        ],
        [frame, handle_frame(data["person"], data["item"])],
    )


def parse_open_close_statement(sentence: str) -> StatementParseResult | None:
    for frame_type, result, patterns in (
        ("open", "打开", (ACTIVE_OPEN_RE, PASSIVE_OPEN_RE)),
        ("close", "关闭", (ACTIVE_CLOSE_RE, PASSIVE_CLOSE_RE)),
    ):
        for pattern in patterns:
            match = pattern.match(sentence)
            if not match:
                continue
            data = match.groupdict()
            actor = normalize_slot_value(data.get("person") or "")
            thing = normalize_slot_value(data.get("thing") or data.get("thing1") or data.get("thing2") or "")
            roles = {"theme": thing, "result": result}
            entities = [Entity(moved_role(thing), thing)]
            frames = []
            if actor:
                roles["actor"] = actor
                entities.insert(0, Entity("person", actor))
                frames.append(handle_frame(actor, thing))
            frames.insert(0, frame_from_roles(frame_type, **roles))
            return (entities, frames)
    return None


def parse_create_destroy_statement(sentence: str) -> StatementParseResult | None:
    for frame_type, result, patterns in (
        ("create", "存在", (PASSIVE_CREATE_RE, ACTIVE_CREATE_RE)),
        ("destroy", "不存在", (PASSIVE_DESTROY_RE, ACTIVE_DESTROY_RE)),
    ):
        for pattern in patterns:
            match = pattern.match(sentence)
            if not match:
                continue
            data = match.groupdict()
            actor = normalize_slot_value(data.get("person") or "")
            thing = normalize_slot_value(
                data.get("thing") or data.get("thing1") or data.get("thing2") or data.get("thing3") or data.get("thing4") or ""
            )
            roles = {"theme": thing, "result": result}
            entities = [Entity(moved_role(thing), thing)]
            frames = []
            if actor:
                roles["actor"] = actor
                entities.insert(0, Entity("person", actor))
                frames.append(handle_frame(actor, thing))
            frames.insert(0, frame_from_roles(frame_type, **roles))
            return (entities, frames)
    return None


def parse_effect_clause(sentence: str) -> StatementParseResult | None:
    parsers = (
        parse_surface_statement,
    )
    for parser in parsers:
        parsed = parser(sentence)
        if parsed is not None:
            return parsed
    return None


def frame_from_roles(frame_type: str, **roles: str) -> Frame:
    frame_id = "pending"
    return Frame(
        frame_id=frame_id,
        frame_type=frame_type,
        time=0,
        roles=tuple(Role(frame_id, name, value) for name, value in roles.items()),
    )


def with_time(frame: Frame, time: int) -> Frame:
    frame_id = f"f{time}"
    return Frame(
        frame_id=frame_id,
        frame_type=frame.frame_type,
        time=time,
        roles=tuple(Role(frame_id, role.name, role.value) for role in frame.roles),
    )


def handle_frame(actor: str, theme: str) -> Frame:
    return frame_from_roles("handle", actor=actor, theme=theme)


def moved_role(name: str) -> str:
    if name.endswith(("盒子", "背包", "抽屉", "托盘")):
        return "container"
    return "thing"


def dedupe_entities(entities: list[Entity]) -> tuple[Entity, ...]:
    by_name: dict[str, Entity] = {}
    for entity in entities:
        by_name.setdefault(entity.name, entity)
    return tuple(by_name.values())


DEFAULT_STATEMENT_PARSERS: tuple[StatementParser, ...] = (
    parse_surface_statement,
)


def normalize_clause_text(text: str) -> str:
    normalized = normalize_slot_value(text)
    normalized = normalized.strip().rstrip("。！？!?，,")
    for prefix in ("就会", "就", "会", "则", "于是", "然后"):
        if normalized.startswith(prefix) and len(normalized) > len(prefix):
            normalized = normalized[len(prefix) :]
    normalized = normalized.replace("就把", "把").replace("会把", "把").replace("就被", "被").replace("会被", "被")
    normalized = normalize_containment_expression(normalize_take_out_expression(normalized))
    return normalized


def split_if_then(sentence: str) -> tuple[str, str] | None:
    for prefix in ("如果", "只要", "假如", "要是"):
        if sentence.startswith(prefix):
            body = sentence[len(prefix) :]
            break
    else:
        return None

    if "，" in body:
        antecedent, consequent = body.split("，", 1)
        return antecedent, strip_then_connector(consequent)
    if "," in body:
        antecedent, consequent = body.split(",", 1)
        return antecedent, strip_then_connector(consequent)

    for connector in ("那么", "就会", "就", "则"):
        if connector in body:
            antecedent, consequent = body.split(connector, 1)
            return antecedent, strip_then_connector(consequent)

    return None


def strip_then_connector(sentence: str) -> str:
    normalized = sentence.strip()
    for connector in ("那么", "就会", "就", "则"):
        if normalized.startswith(connector):
            return normalized[len(connector) :]
    return normalized
