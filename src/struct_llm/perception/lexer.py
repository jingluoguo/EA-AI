from __future__ import annotations


def split_sentences(text: str) -> tuple[tuple[str, bool], ...]:
    """Segment text using the persisted neural perception model."""
    from ..neural.perception_classifier import default_perception_model

    return default_perception_model().split_sentences(text)


def split_query_candidate(candidate: str) -> tuple[str, ...]:
    """Preserve candidate clauses using the persisted neural perception model."""
    from ..neural.perception_classifier import default_perception_model

    return default_perception_model().split_query_candidate(candidate)
