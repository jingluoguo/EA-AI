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
    "answer_structural_update_acknowledgement",
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
    if "dialog_meal_suggestion" in rules:
        preference = optional_query_qualifier(query, "preference")
        request = optional_query_qualifier(query, "request")
        if request == "alternative" and preference == "light":
            return "也可以换成虾仁蒸蛋、冬瓜丸子汤、鸡丝凉面，或者一份青菜瘦肉粥。"
        if request == "alternative" and preference == "rich":
            return "那可以再看看卤肉饭、咖喱鸡饭、牛肉粉，或者加蛋加菜的盖饭。"
        if request == "alternative":
            return "可以，再给你几个方向：面条、盖饭、汤粉、沙拉碗，或者热汤配主食。"
        if preference == "light":
            return "清淡点的话，可以考虑小米粥配鸡蛋、番茄鸡蛋面、青菜豆腐汤，或者蒸鱼配米饭。"
        if preference == "rich":
            return "想吃得满足一点，可以考虑牛肉饭、鸡腿饭、热汤面，或者一份有主食和蛋白质的简餐。"
    if "conversation_summary" in rules:
        return f"已知：{'；'.join(summary_descriptions(structure))}。"
    if "conversation_summary_empty" in rules:
        return "我还没有可总结的内容。"
    return None


def answer_pragmatic_response_policy(structure: Structure) -> str | None:
    rules = set(structure.rules)
    for act in structure.pragmatic_acts:
        if act.act == "ambiguous_reference":
            candidates = pragmatic_candidates(act.qualifiers)
            if candidates:
                return f"你说的是{join_alternatives(candidates)}？"
        if act.act == "underspecified_reference_query" and act.target:
            return f"你想问{act.target}的哪方面？比如位置、状态、归属或相关信息。"
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


def pragmatic_candidates(qualifiers: tuple[str, ...]) -> tuple[str, ...]:
    for qualifier in qualifiers:
        if qualifier.startswith("candidates="):
            return tuple(value for value in qualifier.split("=", 1)[1].split("|") if value)
    return ()


def join_alternatives(candidates: tuple[str, ...]) -> str:
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 2:
        return f"{candidates[0]}还是{candidates[1]}"
    return f"{'、'.join(candidates[:-1])}还是{candidates[-1]}"


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
    latest = latest_profile_frame(structure)
    if latest is not None and latest.time >= structure.current_frame_start_time:
        return "我知道了。"
    return None


def answer_structural_update_acknowledgement(structure: Structure) -> str | None:
    if structure.query is not None:
        return None
    if "structural_update_acknowledgement" not in set(structure.rules):
        return None
    return "我知道了。"


def profile_statement_subject(subject: str | None) -> str:
    if subject in {"", None, "我", "self"}:
        return "你"
    return subject or ''


def latest_profile_frame(structure: Structure):
    frames = [
        frame
        for frame in structure.frames
        if frame.frame_type in {"profile_name", "profile_like", "profile_dislike"}
    ]
    if not frames:
        return None
    return max(frames, key=lambda frame: frame.time)
