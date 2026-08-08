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
class ScopedFrame:
    scope: str
    kind: str
    owner: str
    proposition: str
    frame: Frame

    def linearize(self) -> str:
        roles = ",".join(f"{role.name}={role.value}" for role in self.frame.roles)
        return (
            f"SCOPED_FRAME {self.scope} kind={self.kind} owner={self.owner} "
            f"proposition={self.proposition} type={self.frame.frame_type} roles={roles}"
        )


@dataclass(frozen=True)
class ScopedState:
    scope: str
    kind: str
    owner: str
    proposition: str
    state: State

    def linearize(self) -> str:
        return (
            f"SCOPED_STATE {self.scope} kind={self.kind} owner={self.owner} "
            f"proposition={self.proposition} STATE {self.state.name}({self.state.left},{self.state.right})"
        )


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
class Intention:
    subject: str
    goal: str
    belief: str = ""
    strategy: str = ""
    evidence: str = ""
    confidence: float = 1.0
    source: str = "learned"

    def linearize(self) -> str:
        parts = [f"subject={self.subject}", f"goal={self.goal}"]
        if self.belief:
            parts.append(f"belief={self.belief}")
        if self.strategy:
            parts.append(f"strategy={self.strategy}")
        if self.evidence:
            parts.append(f"evidence={self.evidence}")
        parts.append(f"confidence={self.confidence:.2f}")
        parts.append(f"source={self.source}")
        return f"INTENT {','.join(parts)}"


@dataclass(frozen=True)
class PragmaticAct:
    act: str
    target: str = ""
    qualifiers: tuple[str, ...] = ()
    confidence: float = 1.0
    source: str = "learned"

    def linearize(self) -> str:
        target = self.target or "_"
        if self.qualifiers:
            head = f"PRAGMATIC_ACT {self.act}({target},{','.join(self.qualifiers)})"
        else:
            head = f"PRAGMATIC_ACT {self.act}({target})"
        return f"{head} confidence={self.confidence:.2f} source={self.source}"


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
        if self.frame_type == "take_out":
            actor = self.role("actor") or ""
            theme = self.role("theme") or ""
            source = self.role("source")
            qualifiers = (f"source={source}",) if source else ()
            return Event("take_out", actor, theme, qualifiers)
        if self.frame_type == "if_then":
            antecedent = self.role("antecedent") or ""
            consequent = self.role("consequent") or ""
            return Event("if_then", antecedent, consequent)
        if self.frame_type == "because":
            cause = self.role("cause") or ""
            effect = self.role("effect") or ""
            return Event("because", cause, effect)
        if self.frame_type == "say":
            speaker = self.role("speaker") or ""
            proposition = self.role("proposition") or ""
            return Event("say", speaker, proposition)
        if self.frame_type == "believe":
            person = self.role("person") or ""
            proposition = self.role("proposition") or ""
            return Event("believe", person, proposition)
        if self.frame_type == "be_in":
            theme = self.role("theme") or ""
            goal = self.role("goal") or ""
            return Event("be_in", theme, goal)
        if self.frame_type == "not_in":
            theme = self.role("theme") or ""
            source = self.role("source") or ""
            return Event("not_in", theme, source)
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
        if self.frame_type in {"open", "close"}:
            actor = self.role("actor") or ""
            theme = self.role("theme") or ""
            result = self.role("result")
            qualifiers = (f"result={result}",) if result else ()
            return Event(self.frame_type, actor, theme, qualifiers)
        if self.frame_type in {"create", "destroy"}:
            actor = self.role("actor") or ""
            theme = self.role("theme") or ""
            result = self.role("result") or ""
            qualifiers = (f"result={result}",) if result else ()
            return Event(self.frame_type, actor, theme, qualifiers)
        if self.frame_type in {"exist", "not_exist"}:
            theme = self.role("theme") or ""
            result = self.role("result") or ""
            qualifiers = (f"result={result}",) if result else ()
            return Event(self.frame_type, "", theme, qualifiers)
        if self.frame_type in {"profile_name", "profile_like", "profile_dislike"}:
            subject = self.role("subject") or ""
            value = self.role("value") or ""
            return Event(self.frame_type, subject, value)
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
    subqueries: tuple[Query, ...] = ()

    def linearize(self) -> str:
        lines = []
        if self.qualifiers:
            lines.append(f"QUERY {self.intent}({self.target},{','.join(self.qualifiers)})")
        else:
            lines.append(f"QUERY {self.intent}({self.target})")
        lines.extend(subquery.linearize().replace("QUERY ", "SUBQUERY ", 1) for subquery in self.subqueries)
        return "\n".join(lines)


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
    scoped_frames: tuple[ScopedFrame, ...] = ()
    scoped_states: tuple[ScopedState, ...] = ()
    intentions: tuple[Intention, ...] = ()
    pragmatic_acts: tuple[PragmaticAct, ...] = ()
    current_frame_start_time: int = 1

    def linearize(self) -> str:
        lines = [entity.linearize() for entity in self.entities]
        states = self.states or tuple(State(relation.name, relation.left, relation.right) for relation in self.relations)
        frames = self.frames
        events = self.events or tuple(frame.to_event() for frame in frames)
        relations = self.relations or tuple(state.to_relation() for state in states)
        lines.extend(relation.linearize() for relation in relations)
        lines.extend(event.linearize() for event in events)
        lines.extend(intention.linearize() for intention in self.intentions)
        lines.extend(act.linearize() for act in self.pragmatic_acts)
        lines.extend(frame.linearize() for frame in frames)
        lines.extend(frame.linearize() for frame in self.scoped_frames)
        lines.extend(state.linearize() for state in self.scoped_states)
        lines.extend(f"RULE {rule}" for rule in self.rules)
        query = self.query
        if query is not None:
            lines.append(query.linearize())
        return "\n".join(lines)


def linearize_target(structure: Structure, answer: str) -> str:
    return f"<STRUCT>\n{structure.linearize()}\n<ANSWER>\n{answer}"


def normalize_lines(text: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in text.splitlines() if line.strip())


def lines_match(expected: Iterable[str], actual: Iterable[str]) -> bool:
    return tuple(expected) == tuple(actual)
