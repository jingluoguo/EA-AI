from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Entity:
    role: str
    name: str

    def linearize(self) -> str:
        return f"ENTITY {self.role}={self.name}"


@dataclass(frozen=True)
class Relation:
    name: str
    left: str
    right: str

    def linearize(self) -> str:
        return f"REL {self.name}({self.left},{self.right})"


@dataclass(frozen=True)
class State:
    name: str
    left: str
    right: str
    source: str | None = None

    def linearize(self) -> str:
        return f"STATE {self.name}({self.left},{self.right})"

    def to_relation(self) -> Relation:
        return Relation(self.name, self.left, self.right)


@dataclass(frozen=True)
class Event:
    name: str
    actor: str
    target: str
    qualifiers: tuple[str, ...] = ()

    def linearize(self) -> str:
        if self.qualifiers:
            return f"EVENT {self.name}({self.actor},{self.target}) WITH {','.join(self.qualifiers)}"
        return f"EVENT {self.name}({self.actor},{self.target})"


@dataclass(frozen=True)
class Role:
    frame_id: str
    name: str
    value: str

    def linearize(self) -> str:
        return f"ROLE {self.frame_id} {self.name}={self.value}"


@dataclass(frozen=True)
class Frame:
    frame_id: str
    frame_type: str
    time: int
    roles: tuple[Role, ...]

    def role(self, name: str) -> str | None:
        matches = [role.value for role in self.roles if role.name == name]
        if not matches:
            return None
        if len(matches) != 1:
            raise ValueError(f"Expected at most one role {name} in {self.frame_id}.")
        return matches[0]

    def linearize(self) -> str:
        lines = [f"FRAME {self.frame_id} type={self.frame_type} time={self.time}"]
        lines.extend(role.linearize() for role in self.roles)
        return "\n".join(lines)

    def to_event(self) -> Event:
        if self.frame_type == "put_in":
            actor = self.role("actor") or ""
            theme = self.role("theme") or ""
            goal = self.role("goal")
            qualifiers = (f"holder={goal}",) if goal else ()
            return Event("put_in", actor, theme, qualifiers)
        if self.frame_type == "move":
            theme = self.role("theme") or ""
            goal = self.role("goal") or ""
            actor = self.role("actor")
            qualifiers = (f"by={actor}",) if actor else ()
            return Event("move", theme, goal, qualifiers)
        if self.frame_type == "give":
            actor = self.role("actor") or ""
            recipient = self.role("recipient") or ""
            return Event("give", actor, recipient)
        if self.frame_type == "paint":
            theme = self.role("theme") or ""
            color = self.role("result") or ""
            return Event("paint", theme, color)
        if self.frame_type == "handle":
            actor = self.role("actor") or ""
            theme = self.role("theme") or ""
            return Event("handle", actor, theme)
        return Event(self.frame_type, self.role("actor") or "", self.role("theme") or "")


@dataclass(frozen=True)
class Query:
    intent: str
    target: str
    qualifiers: tuple[str, ...] = ()

    def linearize(self) -> str:
        if self.qualifiers:
            return f"QUERY {self.intent}({self.target},{','.join(self.qualifiers)})"
        return f"QUERY {self.intent}({self.target})"


@dataclass(frozen=True)
class Structure:
    """Explicit intermediate state between natural language and answers."""

    entities: tuple[Entity, ...]
    rules: tuple[str, ...]
    relations: tuple[Relation, ...] = ()
    events: tuple[Event, ...] = ()
    query: Query | None = None
    frames: tuple[Frame, ...] = ()
    states: tuple[State, ...] = ()

    def linearize(self) -> str:
        lines = [entity.linearize() for entity in self.entities]
        states = self.states or tuple(State(relation.name, relation.left, relation.right) for relation in self.relations)
        frames = self.frames
        events = self.events or tuple(frame.to_event() for frame in frames)
        relations = self.relations or tuple(state.to_relation() for state in states)
        lines.extend(relation.linearize() for relation in relations)
        lines.extend(event.linearize() for event in events)
        lines.extend(frame.linearize() for frame in frames)
        lines.extend(f"RULE {rule}" for rule in self.rules)
        if self.query is not None:
            lines.append(self.query.linearize())
        return "\n".join(lines)


def linearize_target(structure: Structure, answer: str) -> str:
    return f"<STRUCT>\n{structure.linearize()}\n<ANSWER>\n{answer}"


def normalize_lines(text: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in text.splitlines() if line.strip())


def lines_match(expected: Iterable[str], actual: Iterable[str]) -> bool:
    return tuple(expected) == tuple(actual)
