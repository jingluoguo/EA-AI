from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ..structure import Entity, Frame, Intention, Query, State, Structure


StatementParseResult = tuple[list[Entity], list[Frame]]
StatementParser = Callable[[str], Optional[StatementParseResult]]
StateProjector = Callable[[Frame], tuple[State, ...]]
StateReducer = Callable[[list[State], State], bool]
QueryParser = Callable[[str, tuple[Entity, ...]], Optional[Query]]
RuleInferer = Callable[[Structure], Optional[str]]
Answerer = Callable[[Structure], Optional[str]]
IntentAnalyzer = Callable[[str, Structure], tuple[Intention, ...]]


@dataclass(frozen=True)
class CognitiveCapabilities:
    statement_parsers: tuple[StatementParser, ...]
    state_projectors: tuple[StateProjector, ...]
    state_reducers: tuple[StateReducer, ...]
    query_parsers: tuple[QueryParser, ...]
    rule_inferers: tuple[RuleInferer, ...]
    answerers: tuple[Answerer, ...]
    intent_analyzers: tuple[IntentAnalyzer, ...] = ()

    def parse_statement(self, sentence: str) -> StatementParseResult | None:
        for parser in self.statement_parsers:
            parsed = parser(sentence)
            if parsed is not None:
                return parsed
        return None

    def states_from_frame(self, frame: Frame) -> tuple[State, ...]:
        states: list[State] = []
        for projector in self.state_projectors:
            states.extend(projector(frame))
        return tuple(states)

    def apply_state(self, states: list[State], state: State) -> None:
        for reducer in self.state_reducers:
            if reducer(states, state):
                return
        states.append(state)

    def parse_query(self, sentence: str, entities: tuple[Entity, ...]) -> Query | None:
        for parser in self.query_parsers:
            query = parser(sentence, entities)
            if query is not None:
                return query
        return None

    def infer_rules(self, structure: Structure) -> tuple[str, ...]:
        rules: list[str] = []
        for inferer in self.rule_inferers:
            rule = inferer(structure)
            if rule is not None:
                rules.append(rule)
        return tuple(rules)

    def analyze_intentions(self, text: str, structure: Structure) -> tuple[Intention, ...]:
        intentions: list[Intention] = []
        for analyzer in self.intent_analyzers:
            intentions.extend(analyzer(text, structure))
        return tuple(intentions)

    def answer(self, structure: Structure) -> str | None:
        for answerer in self.answerers:
            answer = answerer(structure)
            if answer is not None:
                return answer
        return None

    def with_statement_parsers(self, *parsers: StatementParser) -> CognitiveCapabilities:
        return CognitiveCapabilities(
            statement_parsers=(*self.statement_parsers, *parsers),
            state_projectors=self.state_projectors,
            state_reducers=self.state_reducers,
            query_parsers=self.query_parsers,
            rule_inferers=self.rule_inferers,
            answerers=self.answerers,
            intent_analyzers=self.intent_analyzers,
        )

    def with_state_projectors(self, *projectors: StateProjector) -> CognitiveCapabilities:
        return CognitiveCapabilities(
            statement_parsers=self.statement_parsers,
            state_projectors=(*self.state_projectors, *projectors),
            state_reducers=self.state_reducers,
            query_parsers=self.query_parsers,
            rule_inferers=self.rule_inferers,
            answerers=self.answerers,
            intent_analyzers=self.intent_analyzers,
        )

    def with_state_reducers(self, *reducers: StateReducer) -> CognitiveCapabilities:
        return CognitiveCapabilities(
            statement_parsers=self.statement_parsers,
            state_projectors=self.state_projectors,
            state_reducers=(*self.state_reducers, *reducers),
            query_parsers=self.query_parsers,
            rule_inferers=self.rule_inferers,
            answerers=self.answerers,
            intent_analyzers=self.intent_analyzers,
        )

    def with_query_parsers(self, *parsers: QueryParser) -> CognitiveCapabilities:
        return CognitiveCapabilities(
            statement_parsers=self.statement_parsers,
            state_projectors=self.state_projectors,
            state_reducers=self.state_reducers,
            query_parsers=(*self.query_parsers, *parsers),
            rule_inferers=self.rule_inferers,
            answerers=self.answerers,
            intent_analyzers=self.intent_analyzers,
        )

    def with_rule_inferers(self, *inferers: RuleInferer) -> CognitiveCapabilities:
        return CognitiveCapabilities(
            statement_parsers=self.statement_parsers,
            state_projectors=self.state_projectors,
            state_reducers=self.state_reducers,
            query_parsers=self.query_parsers,
            rule_inferers=(*self.rule_inferers, *inferers),
            answerers=self.answerers,
            intent_analyzers=self.intent_analyzers,
        )

    def with_answerers(self, *answerers: Answerer) -> CognitiveCapabilities:
        return CognitiveCapabilities(
            statement_parsers=self.statement_parsers,
            state_projectors=self.state_projectors,
            state_reducers=self.state_reducers,
            query_parsers=self.query_parsers,
            rule_inferers=self.rule_inferers,
            answerers=(*self.answerers, *answerers),
            intent_analyzers=self.intent_analyzers,
        )

    def with_intent_analyzers(self, *analyzers: IntentAnalyzer) -> CognitiveCapabilities:
        return CognitiveCapabilities(
            statement_parsers=self.statement_parsers,
            state_projectors=self.state_projectors,
            state_reducers=self.state_reducers,
            query_parsers=self.query_parsers,
            rule_inferers=self.rule_inferers,
            answerers=self.answerers,
            intent_analyzers=(*self.intent_analyzers, *analyzers),
        )
