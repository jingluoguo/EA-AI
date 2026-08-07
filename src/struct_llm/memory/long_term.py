from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..dataset_io import append_jsonl_object, load_jsonl_objects
from ..structure import Entity, State, Structure
from ..world.state import apply_state


DATA_DIR = Path(__file__).resolve().parents[3] / "data"
MEMORY_DIRECT_DATA_PATH = DATA_DIR / "memory_direct_examples.jsonl"
MEMORY_CHAT_DATA_PATH = DATA_DIR / "memory_chat_examples.jsonl"
MEMORY_MODEL_PATH = DATA_DIR / "memory_model.json"
MEMORY_ENTRY_SCHEMA = "struct_llm.memory_entry.v1"
MEMORY_MODEL_SCHEMA = "struct_llm.memory_model.v1"
CHAT_MEMORY_STATE_NAMES = frozenset(
    {
        "name",
        "likes",
        "dislikes",
        "at",
        "in",
        "owner",
        "color",
        "access",
        "exists",
    }
)


@dataclass(frozen=True)
class MemoryEntry:
    state: State
    text: str = ""
    channel: str = "direct"
    source: str = "human_seed"
    confidence: float = 1.0
    status: str = "active"


@dataclass(frozen=True)
class CompiledMemoryModel:
    schema: str
    source_sha256: str
    example_count: int
    states: tuple[State, ...]


@dataclass(frozen=True)
class MemoryWriteResult:
    data_path: Path
    model_path: Path
    entry_count: int
    state_count: int


def build_memory_record(
    state: State,
    *,
    text: str = "",
    channel: str = "direct",
    source: str = "human_seed",
    confidence: float = 1.0,
    status: str = "active",
) -> dict[str, Any]:
    return memory_entry_to_record(
        MemoryEntry(
            state=State(state.name, state.left, state.right),
            text=text.strip(),
            channel=channel.strip() or "direct",
            source=source.strip() or "human_seed",
            confidence=max(0.0, min(1.0, float(confidence))),
            status=status.strip() or "active",
        )
    )


def memory_entry_to_record(entry: MemoryEntry) -> dict[str, Any]:
    return {
        "schema": MEMORY_ENTRY_SCHEMA,
        "state": state_to_dict(entry.state),
        "text": entry.text,
        "channel": entry.channel,
        "source": entry.source,
        "confidence": max(0.0, min(1.0, float(entry.confidence))),
        "status": entry.status,
    }


def memory_entry_from_dict(record: dict[str, Any], *, line_number: int | None = None) -> MemoryEntry:
    prefix = f"Memory entry at line {line_number}" if line_number is not None else "Memory entry"
    schema = str(record.get("schema") or MEMORY_ENTRY_SCHEMA).strip()
    if schema != MEMORY_ENTRY_SCHEMA:
        raise ValueError(f"{prefix} has unsupported schema: {schema}")
    raw_state = record.get("state")
    if not isinstance(raw_state, dict):
        raise ValueError(f"{prefix} requires state.")
    confidence = float(record.get("confidence") or 0.0)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{prefix} confidence must be between 0 and 1.")
    status = str(record.get("status") or "active").strip() or "active"
    return MemoryEntry(
        state=state_from_dict(raw_state),
        text=str(record.get("text") or "").strip(),
        channel=str(record.get("channel") or "direct").strip() or "direct",
        source=str(record.get("source") or "human_seed").strip() or "human_seed",
        confidence=confidence,
        status=status,
    )


def load_memory_jsonl(path: str | Path) -> tuple[MemoryEntry, ...]:
    data_path = Path(path)
    if not data_path.exists():
        return ()
    return tuple(
        memory_entry_from_dict(raw_record, line_number=line_number)
        for line_number, raw_record in enumerate(load_jsonl_objects(data_path, "memory"), start=1)
    )


def append_memory_record(path: str | Path, record: dict[str, Any]) -> MemoryEntry:
    entry = memory_entry_from_dict(record)
    append_jsonl_object(path, memory_entry_to_record(entry))
    return entry


def append_memory_entry(path: str | Path, entry: MemoryEntry) -> MemoryEntry:
    return append_memory_record(path, memory_entry_to_record(entry))


def compile_memory_model_from_jsonl(
    direct_path: str | Path = MEMORY_DIRECT_DATA_PATH,
    chat_path: str | Path = MEMORY_CHAT_DATA_PATH,
) -> CompiledMemoryModel:
    direct = load_memory_jsonl(direct_path)
    chat = load_memory_jsonl(chat_path)
    return compile_memory_entries(
        (*direct, *chat),
        source_sha256=combined_file_sha256((Path(direct_path), Path(chat_path))),
    )


def compile_memory_entries(
    entries: tuple[MemoryEntry, ...],
    source_sha256: str = "",
) -> CompiledMemoryModel:
    states: list[State] = []
    active_entries = tuple(entry for entry in entries if entry.status == "active" and entry.confidence >= 0.5)
    for entry in active_entries:
        # Reuse the world-state reducer, so direct memory and chat memory obey the
        # same overwrite/correction rules as ordinary conversation state.
        apply_state(states, State(entry.state.name, entry.state.left, entry.state.right, "memory"))
    return CompiledMemoryModel(
        schema=MEMORY_MODEL_SCHEMA,
        source_sha256=source_sha256,
        example_count=len(active_entries),
        states=tuple(states),
    )


def load_memory_model(path: str | Path) -> CompiledMemoryModel:
    with Path(path).open("r", encoding="utf-8") as file:
        raw_model = json.load(file)
    if not isinstance(raw_model, dict):
        raise ValueError("Memory model must be a JSON object.")
    return memory_model_from_dict(raw_model)


def save_memory_model(model: CompiledMemoryModel, path: str | Path) -> None:
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_name(f"{model_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(memory_model_to_dict(model), file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(model_path)


def memory_model_from_dict(record: dict[str, Any]) -> CompiledMemoryModel:
    schema = str(record.get("schema") or "").strip()
    if schema != MEMORY_MODEL_SCHEMA:
        raise ValueError(f"Unsupported memory model schema: {schema}")
    raw_states = record.get("states")
    if not isinstance(raw_states, list):
        raise ValueError("Memory model states must be a list.")
    return CompiledMemoryModel(
        schema=schema,
        source_sha256=str(record.get("source_sha256") or ""),
        example_count=int(record.get("example_count") or 0),
        states=tuple(state_from_dict(value, source="memory") for value in raw_states),
    )


def memory_model_to_dict(model: CompiledMemoryModel) -> dict[str, Any]:
    return {
        "schema": model.schema,
        "source_sha256": model.source_sha256,
        "example_count": model.example_count,
        "state_count": len(model.states),
        "states": [state_to_dict(state) for state in model.states],
    }


def default_memory_states(path: str | Path = MEMORY_MODEL_PATH) -> tuple[State, ...]:
    model_path = Path(path)
    if not model_path.exists():
        return ()
    return _cached_default_memory_states(str(model_path), combined_file_sha256((model_path,)))


@lru_cache(maxsize=8)
def _cached_default_memory_states(path: str, source_sha: str) -> tuple[State, ...]:
    model_path = Path(path)
    if not model_path.exists():
        return ()
    return load_memory_model(model_path).states


def extract_chat_memory_entries(
    text: str,
    structure: Structure,
    *,
    confidence: float = 0.85,
) -> tuple[MemoryEntry, ...]:
    entries: list[MemoryEntry] = []
    seen: set[tuple[str, str, str]] = set()
    for state in structure.states:
        if state.source == "memory" or state.name not in CHAT_MEMORY_STATE_NAMES:
            continue
        key = (state.name, state.left, state.right)
        if key in seen:
            continue
        seen.add(key)
        entries.append(
            MemoryEntry(
                state=State(state.name, state.left, state.right),
                text=text.strip(),
                channel="chat",
                source="chat_sediment",
                confidence=confidence,
            )
        )
    return tuple(entries)


def memory_entities_from_states(states: tuple[State, ...]) -> tuple[Entity, ...]:
    entities: list[Entity] = []
    for state in states:
        entities.extend(entities_for_state(state))
    return tuple(dict.fromkeys(entities))


def entities_for_state(state: State) -> tuple[Entity, ...]:
    if state.name == "at":
        return (Entity("thing", state.left), Entity("place", state.right))
    if state.name == "in":
        return (Entity("item", state.left), Entity("container", state.right))
    if state.name == "owner":
        return (Entity("item", state.left), Entity("person", state.right))
    if state.name in {"name", "likes", "dislikes"}:
        subject_role = "self" if state.left == "我" else "person"
        return (Entity(subject_role, state.left), Entity("profile_value", state.right))
    if state.name == "color":
        return (Entity("item", state.left), Entity("color", state.right))
    if state.name == "access":
        return (Entity("container", state.left),)
    return (Entity("thing", state.left),)


def state_to_dict(state: State) -> dict[str, str]:
    return {"name": state.name, "left": state.left, "right": state.right}


def state_from_dict(record: dict[str, Any], *, source: str | None = None) -> State:
    name = str(record.get("name") or "").strip()
    left = str(record.get("left") or "").strip()
    right = str(record.get("right") or "").strip()
    if not name or not left:
        raise ValueError("Memory state requires name and left.")
    return State(name, left, right, source)


def combined_file_sha256(paths: tuple[Path, ...]) -> str:
    digest = sha256()
    for path in paths:
        digest.update(str(path).encode("utf-8"))
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()
