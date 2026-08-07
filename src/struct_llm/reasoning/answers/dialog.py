from __future__ import annotations

from ...errors import ParseError
from ...structure import Structure
from ..selectors import *

__all__ = (
    "answer_dialog_act",
    "answer_profile_lookup",
    "answer_profile_statement_acknowledgement",
)

def answer_dialog_act(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "dialog_act":
        return None
    rules = set(structure.rules)
    if "dialog_greeting" in rules:
        return "你好，我在。"
    if "dialog_thanks" in rules:
        return "不客气。"
    if "dialog_farewell" in rules:
        return "再见。"
    if "dialog_identity" in rules:
        return "我是结构智能原型，会把对话里的事实、状态、信念和问题先整理成结构再回答。"
    if "dialog_capabilities" in rules:
        return "我可以整理聊天里的事实、状态变化、信念、条件和追问，再回答位置、归属、历史事件、矛盾和摘要。"
    if "conversation_summary" in rules:
        return f"已知：{'；'.join(summary_descriptions(structure))}。"
    if "conversation_summary_empty" in rules:
        return "我还没有可总结的内容。"
    return None
def answer_profile_lookup(structure: Structure) -> str | None:
    query = structure.query
    if query is None or query.intent != "profile":
        return None
    attribute = query_qualifier(query, "attribute")
    values = profile_values(structure, query.target, attribute)
    subject = "你" if query.target == "我" else query.target
    if not values:
        if attribute == "name":
            return f"我还不知道{subject}叫什么。"
        if attribute == "likes":
            return f"我还不知道{subject}喜欢什么。"
        if attribute == "dislikes":
            return f"我还不知道{subject}不喜欢什么。"
        return "我还不知道这项信息。"
    if attribute == "name":
        return f"{subject}叫{values[-1]}。"
    if attribute == "likes":
        return f"{subject}喜欢{join_names(values)}。"
    if attribute == "dislikes":
        return f"{subject}不喜欢{join_names(values)}。"
    return None


def answer_profile_statement_acknowledgement(structure: Structure) -> str | None:
    if structure.query is not None:
        return None
    name_values = profile_values(structure, "我", "name")
    if name_values:
        return f"我知道了，你叫{name_values[-1]}。"
    like_values = profile_values(structure, "我", "likes")
    if like_values:
        return f"我知道了，你喜欢{join_names(like_values)}。"
    dislike_values = profile_values(structure, "我", "dislikes")
    if dislike_values:
        return f"我知道了，你不喜欢{join_names(dislike_values)}。"
    return None
