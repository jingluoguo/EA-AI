from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable, Optional, TypeVar

from .structure import Entity, Frame, Intention, PragmaticAct, Query, State, Structure


StatementParseResult = tuple[list[Entity], list[Frame]]
StatementParser = Callable[[str], Optional[StatementParseResult]]
StateProjector = Callable[[Frame], tuple[State, ...]]
StateReducer = Callable[[list[State], State], bool]
QueryParser = Callable[[str, tuple[Entity, ...]], Optional[Query]]
RuleInferer = Callable[[Structure], Optional[str]]
Answerer = Callable[[Structure], Optional[str]]
IntentAnalyzer = Callable[[str, Structure], tuple[Intention, ...]]
PragmaticAnalyzer = Callable[[str, Structure], tuple[PragmaticAct, ...]]
SentenceSegmenter = Callable[[str], tuple[tuple[str, bool], ...]]
CandidateSegmenter = Callable[[str], tuple[str, ...]]
TextNormalizer = Callable[[str, str], str]
ReferenceResolver = Callable[[str, tuple[Entity, ...]], str]
T = TypeVar("T")


@dataclass(frozen=True)
class CognitiveCapabilities:
    statement_parsers: tuple[StatementParser, ...]
    state_projectors: tuple[StateProjector, ...]
    state_reducers: tuple[StateReducer, ...]
    query_parsers: tuple[QueryParser, ...]
    rule_inferers: tuple[RuleInferer, ...]
    answerers: tuple[Answerer, ...]
    intent_analyzers: tuple[IntentAnalyzer, ...] = ()
    pragmatic_analyzers: tuple[PragmaticAnalyzer, ...] = ()
    sentence_segmenters: tuple[SentenceSegmenter, ...] = ()
    candidate_segmenters: tuple[CandidateSegmenter, ...] = ()
    text_normalizers: tuple[TextNormalizer, ...] = ()
    reference_resolvers: tuple[ReferenceResolver, ...] = ()
    memory_states: tuple[State, ...] = ()
    memory_frames: tuple[Frame, ...] = ()

    def parse_statement(self, sentence: str) -> StatementParseResult | None:
        return self._first_non_none(parser(sentence) for parser in self.statement_parsers)

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
        return self._first_non_none(parser(sentence, entities) for parser in self.query_parsers)

    def segment_sentences(self, text: str) -> tuple[tuple[str, bool], ...]:
        if not self.sentence_segmenters:
            raise RuntimeError("No neural sentence segmenter is registered.")
        return self.sentence_segmenters[0](text)

    def segment_candidates(self, text: str) -> tuple[str, ...]:
        if not self.candidate_segmenters:
            raise RuntimeError("No neural candidate segmenter is registered.")
        return self.candidate_segmenters[0](text)

    def normalize_text(self, text: str, mode: str = "question") -> str:
        if not self.text_normalizers:
            raise RuntimeError("No neural text normalizer is registered.")
        return self.text_normalizers[0](text, mode)

    def resolve_references(self, text: str, entities: tuple[Entity, ...]) -> str:
        if not self.reference_resolvers:
            raise RuntimeError("No neural reference resolver is registered.")
        return self.reference_resolvers[0](text, entities)

    def infer_rules(self, structure: Structure) -> tuple[str, ...]:
        return tuple(
            rule
            for inferer in self.rule_inferers
            if (rule := inferer(structure)) is not None
        )

    def analyze_intentions(self, text: str, structure: Structure) -> tuple[Intention, ...]:
        intentions: list[Intention] = []
        for analyzer in self.intent_analyzers:
            intentions.extend(analyzer(text, structure))
        return tuple(intentions)

    def analyze_pragmatics(self, text: str, structure: Structure) -> tuple[PragmaticAct, ...]:
        acts: list[PragmaticAct] = []
        seen: set[tuple[str, str, tuple[str, ...]]] = set()
        for analyzer in self.pragmatic_analyzers:
            for act in analyzer(text, structure):
                signature = (act.act, act.target, act.qualifiers)
                if signature in seen:
                    continue
                seen.add(signature)
                acts.append(act)
        return tuple(acts)

    def answer(self, structure: Structure) -> str | None:
        return self._first_non_none(answerer(structure) for answerer in self.answerers)

    def evolve(self, **changes) -> CognitiveCapabilities:
        return replace(self, **changes)

    def _first_non_none(self, values: Iterable[T | None]) -> T | None:
        for value in values:
            if value is not None:
                return value
        return None

    def _with_tuple(self, field: str, *values):
        current = getattr(self, field)
        return self.evolve(**{field: (*current, *values)})

    def replace_statement_parsers(self, *parsers: StatementParser) -> CognitiveCapabilities:
        return self.evolve(statement_parsers=parsers)

    def with_state_projectors(self, *projectors: StateProjector) -> CognitiveCapabilities:
        return self._with_tuple("state_projectors", *projectors)

    def with_state_reducers(self, *reducers: StateReducer) -> CognitiveCapabilities:
        return self._with_tuple("state_reducers", *reducers)

    def replace_query_parsers(self, *parsers: QueryParser) -> CognitiveCapabilities:
        return self.evolve(query_parsers=parsers)

    def with_rule_inferers(self, *inferers: RuleInferer) -> CognitiveCapabilities:
        return self._with_tuple("rule_inferers", *inferers)

    def with_answerers(self, *answerers: Answerer) -> CognitiveCapabilities:
        return self._with_tuple("answerers", *answerers)

    def with_intent_analyzers(self, *analyzers: IntentAnalyzer) -> CognitiveCapabilities:
        return self._with_tuple("intent_analyzers", *analyzers)

    def with_pragmatic_analyzers(self, *analyzers: PragmaticAnalyzer) -> CognitiveCapabilities:
        return self._with_tuple("pragmatic_analyzers", *analyzers)

    def with_sentence_segmenters(self, *segmenters: SentenceSegmenter) -> CognitiveCapabilities:
        return self._with_tuple("sentence_segmenters", *segmenters)

    def with_candidate_segmenters(self, *segmenters: CandidateSegmenter) -> CognitiveCapabilities:
        return self._with_tuple("candidate_segmenters", *segmenters)

    def with_text_normalizers(self, *normalizers: TextNormalizer) -> CognitiveCapabilities:
        return self._with_tuple("text_normalizers", *normalizers)

    def with_reference_resolvers(self, *resolvers: ReferenceResolver) -> CognitiveCapabilities:
        return self._with_tuple("reference_resolvers", *resolvers)

    def with_memory_states(self, *states: State) -> CognitiveCapabilities:
        return self._with_tuple("memory_states", *states)

    def with_memory_frames(self, *frames: Frame) -> CognitiveCapabilities:
        return self._with_tuple("memory_frames", *frames)
