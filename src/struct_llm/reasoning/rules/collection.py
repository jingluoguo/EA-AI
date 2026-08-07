from __future__ import annotations

from ...structure import Structure
from ..selectors import *

__all__ = (
    "infer_contents_before_event",
    "infer_contents_after_event",
    "infer_polar_contents",
    "infer_holder_contains_things",
    "infer_contents_unknown",
    "infer_contents_except",
    "infer_count_known_contents",
    "infer_compare_count",
    "infer_inventories",
)

def infer_contents_before_event(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "contents_before_event":
        return None
    if temporal_event_contents(structure, query, include_anchor=False) is not None:
        return "contents_before_event_found"
    return "contents_before_event_unknown"
def infer_contents_after_event(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "contents_after_event":
        return None
    if temporal_event_contents(structure, query, include_anchor=True) is not None:
        return "contents_after_event_found"
    return "contents_after_event_unknown"
def infer_polar_contents(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "polar_contents":
        return None
    status = polar_contents_status(structure, query)
    if status is None:
        return "polar_contents_unknown"
    return "polar_contents_true" if status else "polar_contents_false"
def infer_holder_contains_things(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "contents" and contents_in_holder(structure, query.target):
        return "holder_contains_things"
    return None
def infer_contents_unknown(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "contents" and not contents_in_holder(structure, query.target):
        return "contents_unknown"
    return None
def infer_contents_except(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "contents_except":
        return None
    query_qualifier(query, "exclude")
    return "holder_contains_except"
def infer_count_known_contents(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "count":
        return "count_known_contents"
    return None
def infer_compare_count(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "compare_count":
        query_qualifier(query, "left")
        query_qualifier(query, "right")
        return "compare_count_known_contents"
    return None
def infer_inventories(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "inventories" and inventory_by_owner(structure):
        return "owner_inventories"
    return None
