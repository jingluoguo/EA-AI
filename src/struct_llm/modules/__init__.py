"""Pluggable outer modules for the structural reasoning runtime."""

from .base import ModuleContext, ModuleResult, NoOpModule, StructuralModule
from .cognitive import CognitiveKernelModule
from .registry import ModuleRegistry, default_module_registry

__all__ = [
    "CognitiveKernelModule",
    "ModuleContext",
    "ModuleRegistry",
    "ModuleResult",
    "NoOpModule",
    "StructuralModule",
    "default_module_registry",
]
