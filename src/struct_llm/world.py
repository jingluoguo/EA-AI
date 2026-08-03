from __future__ import annotations

from dataclasses import dataclass

from .structure import Entity, Event, Relation, Structure


PEOPLE = ("小明", "小红", "工程师", "研究员")
ITEMS = ("钥匙", "芯片", "药瓶", "笔记本")
CONTAINERS = ("盒子", "背包", "抽屉", "托盘")
PLACES = ("厨房", "实验室", "办公室", "仓库")
OWNERS = ("小李", "医生", "助理", "老师")
COLORS = ("红色", "蓝色", "绿色", "黄色")


@dataclass(frozen=True)
class Example:
    task_type: str
    text: str
    structure: Structure
    answer: str
    split: str

    def to_record(self) -> dict[str, str]:
        return {
            "task_type": self.task_type,
            "split": self.split,
            "text": self.text,
            "structure": self.structure.linearize(),
            "answer": self.answer,
        }


def containment_example(
    person: str,
    item: str,
    container: str,
    place: str,
    split: str = "train",
) -> Example:
    text = f"{person}把{item}放进{container}。{container}被带到{place}。{item}在哪里？"
    structure = Structure(
        entities=(
            Entity("person", person),
            Entity("item", item),
            Entity("container", container),
            Entity("place", place),
        ),
        relations=(Relation("in", item, container),),
        events=(Event("move", container, place),),
        rules=("container_moves_contents",),
    )
    answer = f"{item}在{place}的{container}里。"
    return Example("containment_move", text, structure, answer, split)


def ownership_example(
    giver: str,
    receiver: str,
    item: str,
    split: str = "train",
) -> Example:
    text = f"{giver}把{item}交给{receiver}。现在谁拥有{item}？"
    structure = Structure(
        entities=(
            Entity("giver", giver),
            Entity("receiver", receiver),
            Entity("item", item),
        ),
        relations=(Relation("owns_before", giver, item),),
        events=(Event("give", giver, receiver),),
        rules=("transfer_changes_owner",),
    )
    answer = f"{receiver}拥有{item}。"
    return Example("ownership_transfer", text, structure, answer, split)


def color_example(
    person: str,
    item: str,
    color: str,
    split: str = "train",
) -> Example:
    text = f"{person}把{item}涂成{color}。现在{item}是什么颜色？"
    structure = Structure(
        entities=(
            Entity("person", person),
            Entity("item", item),
            Entity("color", color),
        ),
        relations=(),
        events=(Event("paint", item, color),),
        rules=("paint_changes_color",),
    )
    answer = f"{item}是{color}。"
    return Example("color_change", text, structure, answer, split)
