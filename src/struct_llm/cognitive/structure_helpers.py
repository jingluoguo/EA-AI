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
        by_name.setdefault(entity.name, entity)
    return tuple(by_name.values())
