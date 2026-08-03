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
class Event:
    name: str
    actor: str
    target: str

    def linearize(self) -> str:
        return f"EVENT {self.name}({self.actor},{self.target})"


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
    relations: tuple[Relation, ...]
    events: tuple[Event, ...]
    rules: tuple[str, ...]
    query: Query | None = None

    def linearize(self) -> str:
        lines = [entity.linearize() for entity in self.entities]
        lines.extend(relation.linearize() for relation in self.relations)
        lines.extend(event.linearize() for event in self.events)
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
