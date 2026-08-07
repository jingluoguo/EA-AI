from __future__ import annotations

from ...errors import ParseError
from ...structure import Structure
from ..selectors import *

__all__ = (
    "answer_why",
    "answer_claim_source",
    "answer_belief_location",
    "answer_belief_source",
    "answer_contradictions_found",
    "answer_no_contradictions",
)

def answer_why(structure: Structure) -> str | None:
    if "causal_explanation" not in set(structure.rules):
        return None
    query = require_query(structure)
    explanation = explanation_for_target(structure, query.target)
    if explanation is None:
        raise ParseError(f"Expected explanation for {query.target}.")
    return explanation
def answer_claim_source(structure: Structure) -> str | None:
    if "claim_has_source" not in set(structure.rules):
        return None
    query = require_query(structure)
    speakers = claim_speakers(structure, query.target)
    return f"{join_names(speakers)}说的。"
def answer_belief_location(structure: Structure) -> str | None:
    if "belief_location_found" not in set(structure.rules):
        return None
    query = require_query(structure)
    person = query_qualifier(query, "person")
    location = belief_location(structure, person, query.target)
    if location is None:
        raise ParseError(f"Expected belief location for {person}:{query.target}.")
    place, containers = location
    return f"{person}认为{query.target}{location_phrase(place, containers)}。"
def answer_belief_source(structure: Structure) -> str | None:
    if "belief_has_source" not in set(structure.rules):
        return None
    query = require_query(structure)
    believers = belief_sources(structure, query.target)
    return f"{join_names(believers)}这么认为。"
def answer_contradictions_found(structure: Structure) -> str | None:
    if "contradictions_found" not in set(structure.rules):
        return None
    found = contradictions(structure)
    return f"存在矛盾：{'；'.join(found)}。"
def answer_no_contradictions(structure: Structure) -> str | None:
    if "no_contradictions" not in set(structure.rules):
        return None
    return "没有发现矛盾。"
