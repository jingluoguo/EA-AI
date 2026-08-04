from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..structure import Structure


@dataclass(frozen=True)
class ModuleContext:
    text: str
    structure: Structure | None = None
    answer: str | None = None


@dataclass(frozen=True)
class ModuleResult:
    context: ModuleContext
    notes: tuple[str, ...] = ()


class StructuralModule(Protocol):
    name: str

    def run(self, context: ModuleContext) -> ModuleResult:
        ...


class NoOpModule:
    name = "noop"

    def run(self, context: ModuleContext) -> ModuleResult:
        return ModuleResult(context)
