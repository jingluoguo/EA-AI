from __future__ import annotations

from ...errors import ParseError
from ...structure import Structure
from ..selectors import *

__all__ = (
    "answer_contents_before_event",
    "answer_contents_before_event_unknown",
    "answer_contents_after_event",
    "answer_contents_after_event_unknown",
    "answer_polar_contents_true",
    "answer_polar_contents_false",
    "answer_polar_contents_unknown",
    "answer_holder_contains_things",
    "answer_contents_unknown",
    "answer_contents_except",
    "answer_count_known_contents",
    "answer_compare_count",
    "answer_inventories",
)

def answer_contents_before_event(structure: Structure) -> str | None:
    if "contents_before_event_found" not in set(structure.rules):
        return None
    query = require_query(structure)
    contents = temporal_event_contents(structure, query, include_anchor=False)
    if contents is None:
        raise ParseError(f"Expected contents before event for {query.target}.")
    anchor = query_qualifier(query, "anchor")
    return f"在{anchor}之前，{query.target}里至少有{join_names(contents)}。"
def answer_contents_before_event_unknown(structure: Structure) -> str | None:
    if "contents_before_event_unknown" not in set(structure.rules):
        return None
    query = require_query(structure)
    anchor = query_qualifier(query, "anchor")
    return f"不知道{query.target}在{anchor}之前有什么。"
def answer_contents_after_event(structure: Structure) -> str | None:
    if "contents_after_event_found" not in set(structure.rules):
        return None
    query = require_query(structure)
    contents = temporal_event_contents(structure, query, include_anchor=True)
    if contents is None:
        raise ParseError(f"Expected contents after event for {query.target}.")
    anchor = query_qualifier(query, "anchor")
    return f"在{anchor}之后，{query.target}里至少有{join_names(contents)}。"
def answer_contents_after_event_unknown(structure: Structure) -> str | None:
    if "contents_after_event_unknown" not in set(structure.rules):
        return None
    query = require_query(structure)
    anchor = query_qualifier(query, "anchor")
    return f"不知道{query.target}在{anchor}之后有什么。"
def answer_polar_contents_true(structure: Structure) -> str | None:
    if "polar_contents_true" not in set(structure.rules):
        return None
    query = require_query(structure)
    item = query_qualifier(query, "item")
    return f"是，{query.target}里有{item}。"
def answer_polar_contents_false(structure: Structure) -> str | None:
    if "polar_contents_false" not in set(structure.rules):
        return None
    query = require_query(structure)
    item = query_qualifier(query, "item")
    return f"不是，{query.target}里没有{item}。"
def answer_polar_contents_unknown(structure: Structure) -> str | None:
    if "polar_contents_unknown" not in set(structure.rules):
        return None
    query = require_query(structure)
    item = query_qualifier(query, "item")
    return f"不知道{query.target}里有没有{item}。"
def answer_holder_contains_things(structure: Structure) -> str | None:
    if "holder_contains_things" not in set(structure.rules):
        return None
    query = require_query(structure)
    contents = contents_in_holder(structure, query.target)
    return f"{query.target}里至少有{join_names(contents)}。"
def answer_contents_unknown(structure: Structure) -> str | None:
    if "contents_unknown" not in set(structure.rules):
        return None
    query = require_query(structure)
    return f"不知道{query.target}里有什么。"
def answer_contents_except(structure: Structure) -> str | None:
    if "holder_contains_except" not in set(structure.rules):
        return None
    query = require_query(structure)
    excluded = query_qualifier(query, "exclude")
    contents = tuple(content for content in contents_in_holder(structure, query.target) if content != excluded)
    if not contents:
        return f"{query.target}里除了{excluded}没有已知物品。"
    return f"{query.target}里除了{excluded}还有{join_names(contents)}。"
def answer_count_known_contents(structure: Structure) -> str | None:
    if "count_known_contents" not in set(structure.rules):
        return None
    query = require_query(structure)
    count = len(contents_in_holder(structure, query.target))
    if count == 0:
        return f"{query.target}里没有已知物品。"
    return f"{query.target}里至少有{count}个已知物品。"
def answer_compare_count(structure: Structure) -> str | None:
    if "compare_count_known_contents" not in set(structure.rules):
        return None
    query = require_query(structure)
    left = query_qualifier(query, "left")
    right = query_qualifier(query, "right")
    left_count = len(contents_in_holder(structure, left))
    right_count = len(contents_in_holder(structure, right))
    if left_count > right_count:
        return f"{left}里的已知物品更多，至少有{left_count}个；{right}里至少有{right_count}个。"
    if right_count > left_count:
        return f"{right}里的已知物品更多，至少有{right_count}个；{left}里至少有{left_count}个。"
    if left_count == 0:
        return f"{left}和{right}里都没有已知物品。"
    return f"{left}和{right}里的已知物品一样多，都是{left_count}个。"
def answer_inventories(structure: Structure) -> str | None:
    if "owner_inventories" not in set(structure.rules):
        return None
    inventories = inventory_by_owner(structure)
    parts = [f"{owner}手里有{join_names(items)}" for owner, items in inventories.items()]
    return "；".join(parts) + "。"
