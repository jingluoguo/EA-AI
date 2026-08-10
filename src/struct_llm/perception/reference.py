from __future__ import annotations

import unicodedata

from ..comprehension.surface_lexicon import surface_forms
from ..structure import Entity


def resolve_references(sentence: str, entities: tuple[Entity, ...]) -> str:
    from ..neural.perception_classifier import default_perception_model

    return default_perception_model().resolve_references(sentence, entities)


def unresolved_reference_pronoun(sentence: str) -> str | None:
    """Return a neural reference mention that remains for discourse resolution."""
    from ..neural.perception_classifier import default_perception_model

    mentions = default_perception_model().reference_mentions(sentence)
    return mentions[0] if mentions else None


def unresolved_reference_mention(sentence: str, resolved_sentence: str) -> Entity | None:
    """Expose an unresolved reference as a perception-layer entity."""
    pronoun = unresolved_reference_pronoun(sentence)
    if pronoun is None or pronoun not in resolved_sentence:
        return None
    return Entity("unresolved_reference", pronoun)


def strip_ellipsis_particles(sentence: str) -> str:
    stripped = strip_terminal_punctuation(sentence)
    particles = surface_forms("terminal_discourse_particle")
    while any(stripped.endswith(particle) for particle in particles):
        particle = next(particle for particle in particles if stripped.endswith(particle))
        stripped = stripped[: -len(particle)].strip()
        stripped = strip_terminal_punctuation(stripped)
    return stripped


def strip_terminal_punctuation(sentence: str) -> str:
    stripped = sentence.strip()
    while stripped and unicodedata.category(stripped[-1]).startswith("P"):
        stripped = stripped[:-1].strip()
    return stripped
