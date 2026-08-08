from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..dataset_io import load_jsonl_objects


SURFACE_LEXICON_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "surface_lexicon_examples.jsonl"
SURFACE_LEXICON_RECORD_SCHEMA = "struct_llm.surface_lexicon_example.v1"


@dataclass(frozen=True)
class SurfaceLexiconEntry:
    category: str
    forms: tuple[str, ...]
    canonical: str = ""
    source: str = "linguistic_lexicon"


def load_surface_lexicon_jsonl(path: str | Path) -> tuple[SurfaceLexiconEntry, ...]:
    return tuple(
        surface_lexicon_entry_from_dict(record, line_number=line_number)
        for line_number, record in enumerate(load_jsonl_objects(path, "surface lexicon"), start=1)
    )


@lru_cache(maxsize=None)
def surface_forms(category: str, path: str | Path = SURFACE_LEXICON_DATA_PATH) -> tuple[str, ...]:
    forms: list[str] = []
    for entry in load_surface_lexicon_jsonl(path):
        if entry.category == category:
            forms.extend(entry.forms)
    return tuple(dict.fromkeys(forms))


@lru_cache(maxsize=None)
def surface_replacements(category: str, path: str | Path = SURFACE_LEXICON_DATA_PATH) -> tuple[tuple[str, str], ...]:
    replacements: list[tuple[str, str]] = []
    for entry in load_surface_lexicon_jsonl(path):
        if entry.category != category or not entry.canonical:
            continue
        replacements.extend((form, entry.canonical) for form in entry.forms)
    return tuple(dict.fromkeys(replacements))


def surface_lexicon_entry_from_dict(record: dict, *, line_number: int | None = None) -> SurfaceLexiconEntry:
    prefix = f"Surface lexicon entry at line {line_number}" if line_number is not None else "Surface lexicon entry"
    schema = str(record.get("schema") or SURFACE_LEXICON_RECORD_SCHEMA).strip()
    if schema != SURFACE_LEXICON_RECORD_SCHEMA:
        raise ValueError(f"{prefix} has unsupported schema: {schema}")
    category = str(record.get("category") or "").strip()
    raw_forms = record.get("forms")
    if not category or not isinstance(raw_forms, list):
        raise ValueError(f"{prefix} requires category and forms list.")
    forms = tuple(str(form).strip() for form in raw_forms if str(form).strip())
    if not forms:
        raise ValueError(f"{prefix} requires at least one non-empty form.")
    return SurfaceLexiconEntry(
        category=category,
        forms=forms,
        canonical=str(record.get("canonical") or "").strip(),
        source=str(record.get("source") or "linguistic_lexicon").strip() or "linguistic_lexicon",
    )
