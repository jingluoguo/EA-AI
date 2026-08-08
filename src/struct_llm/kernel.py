from __future__ import annotations

from dataclasses import dataclass

from .capabilities import CognitiveCapabilities
from .comprehension.episode import EPISODE_DATA_PATH
from .kernel_flow import finalize_parse_context, ingest_sentence, initial_parse_context
from .motor.dialogue import default_learned_dialog_answerer
from .neural import NeuralBoundaryModel, configured_neural_boundary_model, with_neural_boundary
from .neural.intent_classifier import default_neural_intent_analyzer
from .neural.pragmatic_classifier import default_neural_pragmatic_analyzer
from .neural.query_classifier import default_neural_query_parser
from .neural.statement_classifier import default_neural_statement_parser
from .memory.long_term import default_memory_states
from .perception.lexer import split_sentences
from .reasoning.pipeline import (
    DEFAULT_ANSWERERS,
    DEFAULT_RULE_INFERERS,
    answer_from_structure,
)
from .structure import Structure
from .world.state import (
    DEFAULT_STATE_PROJECTORS,
    DEFAULT_STATE_REDUCERS,
)


@dataclass(frozen=True)
class Prediction:
    structure: Structure
    answer: str


def default_capabilities(
    neural_model: NeuralBoundaryModel | None = None,
    *,
    neural_answer_priority: str = "first",
    use_environment: bool = True,
    use_memory: bool = True,
) -> CognitiveCapabilities:
    capabilities = CognitiveCapabilities(
        statement_parsers=(default_neural_statement_parser(),),
        state_projectors=DEFAULT_STATE_PROJECTORS,
        state_reducers=DEFAULT_STATE_REDUCERS,
        query_parsers=(default_neural_query_parser(),),
        rule_inferers=DEFAULT_RULE_INFERERS,
        answerers=(default_learned_dialog_answerer(), *DEFAULT_ANSWERERS),
        intent_analyzers=(default_neural_intent_analyzer(),),
    )
    if use_memory:
        memory_states = default_memory_states()
        if memory_states:
            capabilities = capabilities.with_memory_states(*memory_states)
    if EPISODE_DATA_PATH.exists():
        capabilities = capabilities.with_pragmatic_analyzers(
            default_neural_pragmatic_analyzer(EPISODE_DATA_PATH)
        )
    if neural_model is None and use_environment:
        neural_model = configured_neural_boundary_model()
    if neural_model is None:
        return capabilities
    return with_neural_boundary(
        capabilities,
        neural_model,
        answer_priority=neural_answer_priority,
    )


def parse_text_with_capabilities(text: str, capabilities: CognitiveCapabilities) -> Structure:
    context = initial_parse_context(capabilities)

    for sentence, is_question in split_sentences(text):
        ingest_sentence(sentence, is_question, context, capabilities)

    return finalize_parse_context(text, context, capabilities)


def parse_text(text: str, capabilities: CognitiveCapabilities | None = None) -> Structure:
    active_capabilities = capabilities or default_capabilities()
    return parse_text_with_capabilities(text, active_capabilities)


def predict(text: str, capabilities: CognitiveCapabilities | None = None) -> Prediction:
    active_capabilities = capabilities or default_capabilities()
    structure = parse_text_with_capabilities(text, active_capabilities)
    answer = answer_from_structure(structure, active_capabilities.answerers)
    return Prediction(
        structure=structure,
        answer=answer,
    )
