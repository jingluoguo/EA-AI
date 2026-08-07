from __future__ import annotations

from ...errors import ParseError
from ...memory.working import last_user_utterance
from ...structure import Structure
from ..selectors import *

__all__ = (
    "answer_dialog_act",
    "answer_pragmatic_response_policy",
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


def answer_pragmatic_response_policy(structure: Structure) -> str | None:
    rules = set(structure.rules)
    if "pragmatic_recall_previous_turn_found" in rules:
        return f"你刚刚说的是：{last_user_utterance(structure.states)}"
    if "pragmatic_recall_previous_turn_unknown" in rules:
        return "我这里还没有上一条用户输入。"
    if "pragmatic_response_ask_clarification" in rules:
        return "这句话还缺少可计算的对象或上下文，你想让我具体处理什么？"
    if "pragmatic_response_wait_for_completion" in rules:
        return "我先等你把话说完整。"
    if "pragmatic_response_confirm" in rules:
        return "我理解你是在确认我是否跟上了。"
    if "pragmatic_response_repair" in rules:
        return "收到，我会按你纠正后的结构来更新。"
    if "pragmatic_response_acknowledge" in rules:
        return "我知道了。"
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
    latest_frame = latest_profile_frame(structure)
    if latest_frame is not None:
        value = latest_frame.role("value")
        if value and latest_frame.frame_type == "profile_name":
            return f"我知道了，你叫{value}。"
        if value and latest_frame.frame_type == "profile_like":
            return f"我知道了，你喜欢{value}。"
        if value and latest_frame.frame_type == "profile_dislike":
            return f"我知道了，你不喜欢{value}。"
    return None


def latest_profile_frame(structure: Structure):
    frames = [
        frame
        for frame in structure.frames
        if frame.frame_type in {"profile_name", "profile_like", "profile_dislike"}
    ]
    if not frames:
        return None
    return max(frames, key=lambda frame: frame.time)
