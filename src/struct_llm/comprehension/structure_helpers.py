from __future__ import annotations

from ..structure import Entity, Frame, Role


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


def dedupe_entities(entities: list[Entity]) -> tuple[Entity, ...]:
    by_name: dict[str, Entity] = {}
    for entity in entities:
        previous = by_name.get(entity.name)
        if previous is None or entity_role_priority(entity.role) > entity_role_priority(previous.role):
            by_name[entity.name] = entity
    return tuple(by_name.values())


def entity_role_priority(role: str) -> int:
    if role in {"query_intent", "topic"}:
        return 0
    if role in {"unresolved_reference", "profile_value"}:
        return 1
    return 2
