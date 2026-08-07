from __future__ import annotations

from ...structure import Structure
from ..selectors import *

__all__ = (
    "infer_compound_query",
    "infer_dialog_act",
    "infer_profile_lookup",
)

def infer_compound_query(structure: Structure) -> str | None:
    query = structure.query
    if query is not None and query.intent == "compound" and query.subqueries:
        return "compound_query"
    return None
def infer_dialog_act(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "dialog_act":
        return None
    if query.target == "summary":
        return "conversation_summary" if summary_descriptions(structure) else "conversation_summary_empty"
    return f"dialog_{query.target}"
def infer_profile_lookup(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "profile":
        return None
    attribute = query_qualifier(query, "attribute")
    return f"profile_{attribute}_found" if profile_values(structure, query.target, attribute) else f"profile_{attribute}_unknown"
