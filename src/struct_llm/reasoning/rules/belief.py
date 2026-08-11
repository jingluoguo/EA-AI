from __future__ import annotations

from ...structure import Structure
from ..selectors import *

__all__ = (
    "infer_why",
    "infer_claim_source",
    "infer_belief_location",
    "infer_belief_source",
    "infer_contradictions_found",
    "infer_no_contradictions",
)

def infer_why(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "why" and explanation_for_target(structure, query.target):
        return "causal_explanation"
    if query is not None and query.intent == "why" and condition_state_for_target(structure, query.target) is not None:
        return "condition_reason_needs_context"
    return None
def infer_claim_source(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "claim_source" and claim_speakers(structure, query.target):
        return "claim_has_source"
    return None
def infer_belief_location(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "belief_location":
        return None
    person = query_qualifier(query, "person")
    if belief_location(structure, person, query.target) is not None:
        return "belief_location_found"
    return None
def infer_belief_source(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "belief_source" and belief_sources(structure, query.target):
        return "belief_has_source"
    return None
def infer_contradictions_found(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "contradictions" and contradictions(structure):
        return "contradictions_found"
    return None
def infer_no_contradictions(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "contradictions" and not contradictions(structure):
        return "no_contradictions"
    return None
