from __future__ import annotations

from ...structure import Structure
from ..selectors import *

__all__ = (
    "infer_polar_existence",
    "infer_object_not_exists",
    "infer_object_exists",
    "infer_existence_unknown",
    "infer_transfer_changes_owner",
    "infer_paint_changes_color",
    "infer_object_access_state",
)

def infer_polar_existence(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "polar_existence":
        return None
    value = object_exists_value(structure, query.target)
    if value == "存在":
        return "polar_existence_true"
    if value == "不存在":
        return "polar_existence_false"
    if object_is_known(structure, query.target):
        return "polar_existence_true"
    return "polar_existence_unknown"
def infer_object_not_exists(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent in {"existence", "location"} and object_exists_value(structure, query.target) == "不存在":
        return "object_not_exists"
    return None
def infer_object_exists(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "existence":
        return None
    if object_exists_value(structure, query.target) == "存在" or object_is_known(structure, query.target):
        return "object_exists"
    return None
def infer_existence_unknown(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "existence" and not object_is_known(structure, query.target):
        return "existence_unknown"
    return None
def infer_transfer_changes_owner(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "owner" and has_state_left(structure, "owner", query.target):
        return "transfer_changes_owner"
    return None
def infer_paint_changes_color(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "color" and has_state_left(structure, "color", query.target):
        return "paint_changes_color"
    return None
def infer_object_access_state(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "object_state":
        return None
    state_name = optional_query_qualifier(query, "state") or "access"
    if has_state_left(structure, state_name, query.target):
        return "object_access_state"
    return None
