from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .world import (
    COLORS,
    CONTAINERS,
    ITEMS,
    OWNERS,
    PEOPLE,
    PLACES,
    Example,
    color_example,
    containment_example,
    ownership_example,
)


def generate_examples() -> list[Example]:
    examples: list[Example] = []

    for person in PEOPLE:
        for item in ITEMS:
            for container in CONTAINERS:
                for place in PLACES:
                    split = "test" if item == "芯片" and place == "实验室" else "train"
                    examples.append(containment_example(person, item, container, place, split))

    for giver in PEOPLE:
        for receiver in OWNERS:
            for item in ITEMS:
                split = "test" if receiver == "医生" and item == "药瓶" else "train"
                examples.append(ownership_example(giver, receiver, item, split))

    for person in PEOPLE:
        for item in ITEMS:
            for color in COLORS:
                split = "test" if item == "笔记本" and color == "绿色" else "train"
                examples.append(color_example(person, item, color, split))

    return examples


def write_jsonl(examples: Iterable[Example], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example.to_record(), ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
