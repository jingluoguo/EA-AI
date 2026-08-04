from __future__ import annotations

from dataclasses import dataclass

from ..cognitive import CognitiveCapabilities
from .alignment import NoOpAlignmentModule
from .base import ModuleContext, ModuleResult, StructuralModule
from .cognitive import CognitiveKernelModule
from .embodiment import NoOpEmbodimentModule
from .emotion import NoOpEmotionModule
from .generation import NoOpGenerationModule
from .knowledge import NoOpKnowledgeModule
from .learning import NoOpLearningModule
from .memory import NoOpMemoryModule
from .planning import NoOpPlanningModule
from .self_model import NoOpSelfModelModule


@dataclass(frozen=True)
class ModuleRegistry:
    modules: tuple[StructuralModule, ...]

    def run(self, context: ModuleContext) -> ModuleResult:
        active_context = context
        notes: list[str] = []
        for module in self.modules:
            result = module.run(active_context)
            active_context = result.context
            notes.extend(result.notes)
        return ModuleResult(active_context, tuple(notes))

    def module_names(self) -> tuple[str, ...]:
        return tuple(module.name for module in self.modules)

    def with_modules(self, *modules: StructuralModule) -> ModuleRegistry:
        return ModuleRegistry((*self.modules, *modules))


def default_module_registry(capabilities: CognitiveCapabilities) -> ModuleRegistry:
    return ModuleRegistry(
        (
            NoOpAlignmentModule(),
            NoOpMemoryModule(),
            NoOpKnowledgeModule(),
            CognitiveKernelModule(capabilities),
            NoOpGenerationModule(),
            NoOpPlanningModule(),
            NoOpEmbodimentModule(),
            NoOpEmotionModule(),
            NoOpSelfModelModule(),
            NoOpLearningModule(),
        )
    )
