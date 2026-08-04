from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..cognitive import CognitiveCapabilities
from ..cognitive.inference import answer_from_structure
from ..cognitive.kernel import parse_text_with_capabilities
from .base import ModuleContext, ModuleResult


@dataclass(frozen=True)
class CognitiveKernelModule:
    capabilities: CognitiveCapabilities

    name: ClassVar[str] = "cognitive_kernel"

    def run(self, context: ModuleContext) -> ModuleResult:
        structure = parse_text_with_capabilities(context.text, self.capabilities)
        answer = answer_from_structure(structure, self.capabilities.answerers)
        return ModuleResult(ModuleContext(text=context.text, structure=structure, answer=answer))
