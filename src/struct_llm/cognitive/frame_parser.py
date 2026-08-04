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


PUT_IN_RE = re.compile(r"^(?P<person>.+?)把(?P<item>.+?)放进(?P<container>.+?)$")
BECAUSE_RE = re.compile(r"^(?:因为|由于)(?P<cause>.+?)(?:，|,)?(?:所以|因此|就)(?P<effect>.+?)$")
REPORT_RE = re.compile(r"^据(?P<speaker>.+?)说[：:，,]?(?P<proposition>.+?)$")
SAY_RE = re.compile(r"^(?P<speaker>.+?)说[：:，,]?(?P<proposition>.+?)$")
BELIEVE_RE = re.compile(r"^(?P<person>.+?)(?:认为|相信|觉得|以为)(?P<proposition>.+?)$")
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
        parse_put_in_statement,
        parse_take_out_statement,
        parse_negated_in_statement,
        parse_located_in_statement,
        parse_simple_in_statement,
        parse_passive_move_statement,
        parse_active_move_statement,
        parse_give_statement,
        parse_paint_statement,
        parse_open_close_statement,
        parse_create_destroy_statement,
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
    parse_if_then_statement,
    parse_because_statement,
    parse_report_statement,
    parse_say_statement,
    parse_believe_statement,
    parse_put_in_statement,
    parse_take_out_statement,
    parse_negated_in_statement,
    parse_located_in_statement,
    parse_simple_in_statement,
    parse_passive_move_statement,
    parse_active_move_statement,
    parse_give_statement,
    parse_paint_statement,
    parse_open_close_statement,
    parse_create_destroy_statement,
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
