from __future__ import annotations

from dataclasses import dataclass

from .cognitive import CognitiveCapabilities
from .cognitive.dialog_answer_learning import default_learned_dialog_answerer
from .cognitive.inference import DEFAULT_ANSWERERS, DEFAULT_RULE_INFERERS
from .cognitive.kernel import parse_text_with_capabilities
from .errors import ParseError
from .modules import ModuleContext, default_module_registry
from .cognitive.query_learning import default_learned_query_parser
from .cognitive.statement_learning import default_learned_statement_parser
from .cognitive.state_engine import (
    DEFAULT_STATE_PROJECTORS,
    DEFAULT_STATE_REDUCERS,
)
from .structure import Structure


@dataclass(frozen=True)
class Prediction:
    structure: Structure
    answer: str


def default_capabilities() -> CognitiveCapabilities:
    return CognitiveCapabilities(
        statement_parsers=(default_learned_statement_parser(),),
        state_projectors=DEFAULT_STATE_PROJECTORS,
        state_reducers=DEFAULT_STATE_REDUCERS,
        query_parsers=(default_learned_query_parser(),),
        rule_inferers=DEFAULT_RULE_INFERERS,
        answerers=(*DEFAULT_ANSWERERS, default_learned_dialog_answerer()),
    )


def parse_text(text: str, capabilities: CognitiveCapabilities | None = None) -> Structure:
    active_capabilities = capabilities or default_capabilities()
    return parse_text_with_capabilities(text, active_capabilities)


def predict(text: str, capabilities: CognitiveCapabilities | None = None) -> Prediction:
    active_capabilities = capabilities or default_capabilities()
    result = default_module_registry(active_capabilities).run(ModuleContext(text=text))
    structure = result.context.structure
    answer = result.context.answer
    if structure is None or answer is None:
        raise ParseError(f"Module registry did not produce a prediction for text: {text}")
    return Prediction(
        structure=structure,
        answer=answer,
    )
