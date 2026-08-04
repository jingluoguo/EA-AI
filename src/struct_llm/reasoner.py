from __future__ import annotations

from dataclasses import dataclass

from .cognitive import CognitiveCapabilities
from .cognitive.frame_parser import DEFAULT_STATEMENT_PARSERS
from .cognitive.inference import DEFAULT_ANSWERERS, DEFAULT_RULE_INFERERS
from .cognitive.kernel import parse_text_with_capabilities
from .errors import ParseError
from .modules import ModuleContext, default_module_registry
from .cognitive.query_parser import DEFAULT_QUERY_PARSERS
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
        statement_parsers=DEFAULT_STATEMENT_PARSERS,
        state_projectors=DEFAULT_STATE_PROJECTORS,
        state_reducers=DEFAULT_STATE_REDUCERS,
        query_parsers=DEFAULT_QUERY_PARSERS,
        rule_inferers=DEFAULT_RULE_INFERERS,
        answerers=DEFAULT_ANSWERERS,
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
