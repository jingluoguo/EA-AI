from __future__ import annotations

from ...structure import Structure
from ...memory.working import last_user_utterance
from ..selectors import *

__all__ = (
    "infer_compound_query",
    "infer_dialog_act",
    "infer_pragmatic_response_policy",
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


def infer_pragmatic_response_policy(structure: Structure) -> str | None:
    for act in structure.pragmatic_acts:
        if act.act == "recall_previous_turn":
            return "pragmatic_recall_previous_turn_found" if last_user_utterance(structure.states) else "pragmatic_recall_previous_turn_unknown"
    for act in structure.pragmatic_acts:
        for qualifier in act.qualifiers:
            if qualifier.startswith("response_policy="):
                policy = qualifier.split("=", 1)[1].strip()
                if policy:
                    return f"pragmatic_response_{policy}"
    return None


def infer_profile_lookup(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "profile":
        return None
    attribute = query_qualifier(query, "attribute")
    return f"profile_{attribute}_found" if profile_values(structure, query.target, attribute) else f"profile_{attribute}_unknown"
