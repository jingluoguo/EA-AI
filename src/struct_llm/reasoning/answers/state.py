from __future__ import annotations

from ...errors import ParseError
from ...structure import Structure
from ..selectors import *

__all__ = (
    "answer_polar_existence_true",
    "answer_polar_existence_false",
    "answer_polar_existence_unknown",
    "answer_object_not_exists",
    "answer_object_exists",
    "answer_existence_unknown",
    "answer_object_attribute",
    "answer_transfer_changes_owner",
    "answer_paint_changes_color",
    "answer_object_access_state",
)

def answer_polar_existence_true(structure: Structure) -> str | None:
    if "polar_existence_true" not in set(structure.rules):
        return None
    query = require_query(structure)
    return f"是，{query.target}存在。"
def answer_polar_existence_false(structure: Structure) -> str | None:
    if "polar_existence_false" not in set(structure.rules):
        return None
    query = require_query(structure)
    return f"不是，{query.target}不存在。"
def answer_polar_existence_unknown(structure: Structure) -> str | None:
    if "polar_existence_unknown" not in set(structure.rules):
        return None
    query = require_query(structure)
    return f"不知道{query.target}是否存在。"
def answer_object_not_exists(structure: Structure) -> str | None:
    if "object_not_exists" not in set(structure.rules):
        return None
    query = require_query(structure)
    return f"{query.target}不存在。"
def answer_object_exists(structure: Structure) -> str | None:
    if "object_exists" not in set(structure.rules):
        return None
    query = require_query(structure)
    return f"{query.target}存在。"
def answer_existence_unknown(structure: Structure) -> str | None:
    if "existence_unknown" not in set(structure.rules):
        return None
    query = require_query(structure)
    return f"不知道{query.target}是否存在。"


def answer_object_attribute(structure: Structure) -> str | None:
    rules = set(structure.rules)
    if not any(rule.startswith("object_attribute_") for rule in rules):
        return None
    query = require_query(structure)
    attribute = optional_query_qualifier(query, "attribute")
    attribute_label = object_attribute_label(attribute)
    if attribute and f"object_attribute_{attribute}_found" in rules:
        state = state_for_left(structure, attribute, query.target)
        return f"{query.target}的{attribute_label}是{state.right}。"
    return f"我还不知道{query.target}的{attribute_label}。你可以告诉我它是什么材质，或者描述一下外观。"


def answer_transfer_changes_owner(structure: Structure) -> str | None:
    if "transfer_changes_owner" not in set(structure.rules):
        return None
    query = structure.query
    owner = state_for_left(structure, "owner", query.target) if query else only_state(structure, "owner")
    item = owner.left
    receiver = owner.right
    return f"{receiver}拥有{item}。"
def answer_paint_changes_color(structure: Structure) -> str | None:
    if "paint_changes_color" not in set(structure.rules):
        return None
    query = structure.query
    color_relation = state_for_left(structure, "color", query.target) if query else only_state(structure, "color")
    item = color_relation.left
    color = color_relation.right
    return f"{item}是{color}。"
def answer_object_access_state(structure: Structure) -> str | None:
    if "object_access_state" not in set(structure.rules):
        return None
    query = require_query(structure)
    state_name = optional_query_qualifier(query, "state") or "access"
    state = state_for_left(structure, state_name, query.target)
    return f"{state.left}是{state.right}状态。"


def object_attribute_label(attribute: str | None) -> str:
    labels = {
        "material": "材质",
        "color": "颜色",
        "condition": "状态",
    }
    return labels.get(attribute or "", attribute or "属性")
