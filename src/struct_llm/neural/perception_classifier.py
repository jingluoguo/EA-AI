from __future__ import annotations

import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor, nn

from ..dataset_io import file_sha256
from ..perception.learning import (
    PERCEPTION_DATA_PATH,
    PerceptionTrainingExample,
    load_perception_jsonl,
)
from ..structure import Entity
from .common import build_character_vocabulary, encode_characters, set_seed


PERCEPTION_NEURAL_SCHEMA = "struct_llm.perception_neural_model.v1"
PERCEPTION_NEURAL_WEIGHTS_PATH = Path(__file__).resolve().parents[3] / "data" / "perception_neural_model.pt"
PERCEPTION_NEURAL_META_PATH = Path(__file__).resolve().parents[3] / "data" / "perception_neural_model.json"
PERCEPTION_EMBED_DIM = 48
PERCEPTION_HIDDEN_DIM = 64
PERCEPTION_EPOCHS = 96
PERCEPTION_LEARNING_RATE = 0.02
PERCEPTION_SEED = 20260809


@dataclass(frozen=True)
class PerceptionTrainingResult:
    example_count: int
    train_accuracy: float
    test_accuracy: float
    source_sha256: str


class _CharTagger(nn.Module):
    def __init__(self, vocab_size: int, label_count: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, PERCEPTION_EMBED_DIM, padding_idx=0)
        self.encoder = nn.GRU(
            PERCEPTION_EMBED_DIM,
            PERCEPTION_HIDDEN_DIM,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(PERCEPTION_HIDDEN_DIM * 2, label_count)

    def forward(self, token_ids: Tensor, lengths: Tensor) -> Tensor:
        embedded = self.embedding(token_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.encoder(packed)
        padded, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True)
        return self.head(padded)


class _PairClassifier(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, PERCEPTION_EMBED_DIM, padding_idx=0)
        self.encoder = nn.GRU(
            PERCEPTION_EMBED_DIM,
            PERCEPTION_HIDDEN_DIM,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(PERCEPTION_HIDDEN_DIM * 2, 2)

    def forward(self, token_ids: Tensor, lengths: Tensor) -> Tensor:
        embedded = self.embedding(token_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.encoder(packed)
        pooled = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        return self.head(pooled)


def _pad_sequences(sequences: list[list[int]]) -> tuple[Tensor, Tensor]:
    lengths = torch.tensor([max(1, len(value)) for value in sequences], dtype=torch.long)
    width = int(lengths.max().item())
    result = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        result[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return result, lengths


def _fit_tagger(
    model: _CharTagger,
    sequences: list[list[int]],
    labels: list[list[int]],
    label_count: int,
) -> None:
    token_ids, lengths = _pad_sequences(sequences)
    target = torch.full_like(token_ids, -100)
    for row, values in enumerate(labels):
        target[row, : len(values)] = torch.tensor(values, dtype=torch.long)
    counts = torch.bincount(target[target >= 0], minlength=label_count).float()
    weights = torch.ones(label_count)
    present = counts > 0
    weights[present] = counts[present].sum() / (counts[present] * present.sum())
    weights = weights.clamp(max=8.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=PERCEPTION_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    model.train()
    for _ in range(PERCEPTION_EPOCHS):
        optimizer.zero_grad()
        logits = model(token_ids, lengths)
        loss = criterion(logits.reshape(-1, label_count), target.reshape(-1))
        loss.backward()
        optimizer.step()


def _fit_pair_classifier(
    model: _PairClassifier,
    sequences: list[list[int]],
    labels: list[int],
) -> None:
    token_ids, lengths = _pad_sequences(sequences)
    target = torch.tensor(labels, dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=PERCEPTION_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(PERCEPTION_EPOCHS):
        optimizer.zero_grad()
        loss = criterion(model(token_ids, lengths), target)
        loss.backward()
        optimizer.step()


def _sequence_labels(source: str, target: str) -> list[str]:
    labels = ["KEEP"] * len(source)
    matcher = SequenceMatcher(a=source, b=target, autojunk=False)
    for tag, a_start, a_end, b_start, b_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete":
            for index in range(a_start, a_end):
                labels[index] = "DROP"
            continue
        if tag == "replace":
            if a_start < a_end:
                labels[a_start] = f"REPLACE::{target[b_start:b_end]}"
                for index in range(a_start + 1, a_end):
                    labels[index] = "DROP"
            continue
        if tag == "insert":
            anchor = min(a_start, len(source) - 1)
            labels[anchor] = f"INSERT::{target[b_start:b_end]}::{labels[anchor]}"
    return labels


def _boundary_labels(
    text: str,
    *,
    task: str,
    expected: tuple[str, ...] | tuple[tuple[str, bool], ...],
) -> list[str]:
    labels = ["KEEP"] * len(text)
    if task == "sentence_segmentation":
        boundaries = list(expected)
        cursor = 0
        for segment, is_question in boundaries:
            position = text.find(segment, cursor)
            if position < 0:
                raise ValueError(f"Cannot align sentence segment {segment!r} in {text!r}." )
            cursor = position + len(segment)
            if cursor < len(text) and text[cursor] in "。？！?!，,":
                labels[cursor] = "BREAK_QUESTION" if is_question else "BREAK_STATEMENT"
                cursor += 1
        return labels
    candidates = list(expected)
    cursor = 0
    for index, candidate in enumerate(candidates):
        position = text.find(candidate, cursor)
        if position < 0:
            raise ValueError(f"Cannot align candidate {candidate!r} in {text!r}.")
        cursor = position + len(candidate)
        if index + 1 < len(candidates):
            while cursor < len(text) and text[cursor] in "，,；;":
                labels[cursor] = "BREAK"
                cursor += 1
    for index in range(len(text)):
        if text[index] in "。！？!?" and (index == len(text) - 1 or all(char in "。！？!?" for char in text[index + 1:])):
            labels[index] = "DROP"
    return labels


def _reference_labels(text: str, reference: str) -> list[str]:
    labels = ["OTHER"] * len(text)
    if not reference:
        return labels
    start = text.find(reference)
    if start < 0:
        return labels
    labels[start] = "REF_START"
    for index in range(start + 1, start + len(reference)):
        labels[index] = "REF_CONT"
    return labels


def _sample_records(
    records: list[tuple[list[int], list[str]]],
    *,
    task: str,
) -> list[tuple[list[int], list[str]]]:
    if task == "reference":
        positive = [record for record in records if any(label != "OTHER" for label in record[1])]
        negative = [record for record in records if all(label == "OTHER" for label in record[1])]
        return positive + negative[:320] + negative[-320:]
    if task == "candidate":
        split = [record for record in records if "BREAK" in record[1]]
        whole = [record for record in records if "BREAK" not in record[1]]
        return split + whole[:800]
    if task.startswith("normalization_"):
        edited = [record for record in records if any(label != "KEEP" for label in record[1])]
        identity = [record for record in records if all(label == "KEEP" for label in record[1])]
        return edited + identity[:300] + identity[-400:]
    return records


def _serialize_pair(
    sentence: str,
    reference: str,
    entity: str,
    role: str,
    distance: int = 0,
    total: int = 1,
) -> str:
    return f"句:{sentence}|指:{reference}|角色:{role}|候选:{entity}|倒序:{distance}|总数:{total}"


def _best_reference_target(example: PerceptionTrainingExample) -> str | None:
    if example.resolved == example.text:
        return None
    prefix, suffix = example.text.split(example.reference, 1)
    if not example.resolved.startswith(prefix) or not example.resolved.endswith(suffix):
        return None
    end = len(example.resolved) - len(suffix) if suffix else len(example.resolved)
    replacement = example.resolved[len(prefix):end]
    return replacement or None


@dataclass
class PerceptionNeuralModel:
    vocab: dict[str, int]
    taggers: dict[str, _CharTagger]
    labels: dict[str, tuple[str, ...]]
    pair_vocab: dict[str, int]
    reference_ranker: _PairClassifier
    source_sha256: str
    metadata: dict[str, Any]

    def _predict_tags(self, task: str, text: str) -> tuple[str, ...]:
        tagger = self.taggers[task]
        token_ids, lengths = _pad_sequences([encode_characters(text, self.vocab)])
        tagger.eval()
        with torch.no_grad():
            prediction = tagger(token_ids, lengths)[0, : len(text)].argmax(dim=-1).tolist()
        return tuple(self.labels[task][index] for index in prediction)

    def split_sentences(self, text: str) -> tuple[tuple[str, bool], ...]:
        stripped = text.strip()
        tags = self._predict_tags("sentence", stripped)
        parts: list[tuple[str, bool]] = []
        start = 0
        for index, tag in enumerate(tags):
            if tag not in {"BREAK_STATEMENT", "BREAK_QUESTION"}:
                continue
            if stripped[index] not in "。？！?!，,":
                continue
            segment = stripped[start:index].strip(" ，,")
            if segment:
                parts.append((segment, tag == "BREAK_QUESTION"))
            start = index + 1
        tail = stripped[start:].strip(" ，,")
        if tail:
            parts.append((tail, False))
        return tuple(parts)

    def split_query_candidate(self, text: str) -> tuple[str, ...]:
        stripped = text.strip()
        tags = self._predict_tags("candidate", stripped)
        parts: list[str] = []
        start = 0
        for index, tag in enumerate(tags):
            if tag != "BREAK":
                continue
            if stripped[index] not in "，,；;":
                continue
            candidate = stripped[start:index].strip("，,；;")
            if candidate:
                parts.append(candidate)
            start = index + 1
        tail = stripped[start:].strip("，,；;")
        if tail:
            parts.append(tail)
        return tuple(parts) if parts else (stripped,)

    def normalize(self, text: str, mode: str = "question") -> str:
        task = f"normalization_{mode}"
        tags = self._predict_tags(task, text)
        output: list[str] = []
        for char, tag in zip(text, tags):
            if tag == "KEEP":
                output.append(char)
            elif tag.startswith("REPLACE::"):
                output.append(tag.split("::", 1)[1])
            elif tag.startswith("INSERT::"):
                _, inserted, keep = tag.split("::", 2)
                output.append(inserted)
                if keep == "KEEP":
                    output.append(char)
                elif keep.startswith("REPLACE::"):
                    output.append(keep.split("::", 1)[1])
        return "".join(output).strip("。！？?!，,；;").strip()

    def reference_mentions(self, text: str) -> tuple[str, ...]:
        tags = self._predict_tags("reference", text)
        mentions: list[str] = []
        start: int | None = None
        for index, tag in enumerate((*tags, "OTHER")):
            if tag == "REF_START":
                if start is not None:
                    mentions.append(text[start:index])
                start = index
            elif tag != "REF_CONT" and start is not None:
                mentions.append(text[start:index])
                start = None
        return tuple(mention for mention in mentions if mention.strip())

    def resolve_references(self, text: str, entities: tuple[Entity, ...]) -> str:
        mentions = self.reference_mentions(text)
        if not mentions or not entities:
            return text
        resolved = text
        for mention in sorted(mentions, key=lambda value: text.rfind(value), reverse=True):
            candidates = [(entity.name, entity.role) for entity in entities]
            candidates.append(("", "NONE"))
            pairs = [
                _serialize_pair(text, mention, name, role, len(candidates) - index, len(candidates))
                for index, (name, role) in enumerate(candidates)
            ]
            token_ids, lengths = _pad_sequences([encode_characters(pair, self.pair_vocab) for pair in pairs])
            self.reference_ranker.eval()
            with torch.no_grad():
                scores = torch.softmax(self.reference_ranker(token_ids, lengths), dim=-1)[:, 1]
            selected = candidates[int(scores.argmax().item())][0]
            if not selected:
                continue
            start = resolved.find(mention)
            if start < 0:
                continue
            end = start + len(mention)
            resolved = f"{resolved[:start]}{selected}{resolved[end:]}"
        return resolved

    def predict(self, task: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if task == "split_sentences":
            return {"segments": self.split_sentences(str(payload.get("text") or "")), "confidence": 1.0}
        if task == "split_query_candidate":
            return {"candidates": self.split_query_candidate(str(payload.get("text") or "")), "confidence": 1.0}
        if task == "normalize":
            return {"text": self.normalize(str(payload.get("text") or ""), str(payload.get("mode") or "question")), "confidence": 1.0}
        if task == "resolve_references":
            raw_entities = payload.get("entities") or []
            entities = tuple(Entity(str(item["role"]), str(item["name"])) for item in raw_entities)
            return {"text": self.resolve_references(str(payload.get("text") or ""), entities), "confidence": 1.0}
        return None

    def save(self, weights_path: Path = PERCEPTION_NEURAL_WEIGHTS_PATH, meta_path: Path = PERCEPTION_NEURAL_META_PATH) -> None:
        torch.save(
            {
                "vocab": self.vocab,
                "labels": self.labels,
                "taggers": {task: model.state_dict() for task, model in self.taggers.items()},
                "pair_vocab": self.pair_vocab,
                "reference_ranker": self.reference_ranker.state_dict(),
            },
            weights_path,
        )
        meta_path.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
def _train_model(examples: tuple[PerceptionTrainingExample, ...], source_sha256: str) -> PerceptionNeuralModel:
    set_seed(PERCEPTION_SEED)
    texts = [example.text for example in examples]
    texts.extend(
        _serialize_pair(example.text, example.reference, entity.name, entity.role, len(example.entities) - index, len(example.entities) + 1)
        for example in examples
        if example.task == "reference_resolution"
        for index, entity in enumerate(example.entities)
    )
    vocab = build_character_vocabulary(texts, sort_characters=True)
    task_examples: dict[str, list[tuple[list[int], list[str]]]] = {"sentence": [], "candidate": [], "normalization_question": [], "normalization_statement": [], "reference": []}
    for example in examples:
        if example.task == "sentence_segmentation":
            task_examples["sentence"].append((list(encode_characters(example.text, vocab)), _boundary_labels(example.text, task="sentence_segmentation", expected=tuple((segment.text, segment.is_question) for segment in example.segments))))
        elif example.task == "candidate_segmentation":
            task_examples["candidate"].append((list(encode_characters(example.text, vocab)), _boundary_labels(example.text, task="candidate_segmentation", expected=example.candidates)))
        elif example.task == "normalization":
            task = f"normalization_{example.normalization_mode}"
            task_examples[task].append((list(encode_characters(example.text, vocab)), _sequence_labels(example.text, example.normalized)))
        elif example.task == "reference_resolution":
            task_examples["reference"].append((list(encode_characters(example.text, vocab)), _reference_labels(example.text, example.reference)))
    taggers: dict[str, _CharTagger] = {}
    label_map: dict[str, tuple[str, ...]] = {}
    for task, records in task_examples.items():
        records = _sample_records(records, task=task)
        labels = tuple(sorted({label for _, values in records for label in values} | ({"KEEP"} if task in {"sentence", "candidate"} else set())))
        label_map[task] = labels
        label_index = {label: index for index, label in enumerate(labels)}
        model = _CharTagger(len(vocab), len(labels))
        _fit_tagger(model, [sequence for sequence, _ in records], [[label_index[label] for label in values] for _, values in records], len(labels))
        taggers[task] = model
    pair_records: list[tuple[str, int]] = []
    for example in examples:
        if example.task != "reference_resolution":
            continue
        if not example.reference:
            continue
        target = _best_reference_target(example)
        total = len(example.entities) + 1
        for index, entity in enumerate(example.entities):
            is_positive = int(target is not None and entity.name == target)
            pair_records.append((_serialize_pair(example.text, example.reference, entity.name, entity.role, total - index, total), is_positive))
        pair_records.append((_serialize_pair(example.text, example.reference, "", "NONE", 1, total), int(target is None)))
    pair_vocab = build_character_vocabulary((text for text, _ in pair_records), sort_characters=True)
    ranker = _PairClassifier(len(pair_vocab))
    _fit_pair_classifier(ranker, [list(encode_characters(text, pair_vocab)) for text, _ in pair_records], [label for _, label in pair_records])
    metadata = {
        "schema": PERCEPTION_NEURAL_SCHEMA,
        "source_sha256": source_sha256,
        "example_count": len(examples),
        "labels": {task: list(values) for task, values in label_map.items()},
    }
    return PerceptionNeuralModel(vocab, taggers, label_map, pair_vocab, ranker, source_sha256, metadata)


def _load_model(weights_path: Path, meta_path: Path) -> PerceptionNeuralModel | None:
    if not weights_path.exists() or not meta_path.exists():
        return None
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != PERCEPTION_NEURAL_SCHEMA:
        return None
    payload = torch.load(weights_path, map_location="cpu", weights_only=False)
    vocab = {str(key): int(value) for key, value in payload["vocab"].items()}
    labels = {str(task): tuple(str(value) for value in values) for task, values in payload["labels"].items()}
    taggers: dict[str, _CharTagger] = {}
    for task, values in labels.items():
        model = _CharTagger(len(vocab), len(values))
        model.load_state_dict(payload["taggers"][task])
        taggers[task] = model
    pair_vocab = {str(key): int(value) for key, value in payload["pair_vocab"].items()}
    ranker = _PairClassifier(len(pair_vocab))
    ranker.load_state_dict(payload["reference_ranker"])
    return PerceptionNeuralModel(vocab, taggers, labels, pair_vocab, ranker, str(metadata["source_sha256"]), metadata)


@lru_cache(maxsize=2)
def default_perception_model() -> PerceptionNeuralModel:
    source_sha256 = file_sha256(PERCEPTION_DATA_PATH)
    model = _load_model(PERCEPTION_NEURAL_WEIGHTS_PATH, PERCEPTION_NEURAL_META_PATH)
    if model is not None and model.source_sha256 == source_sha256:
        return model
    examples = load_perception_jsonl(PERCEPTION_DATA_PATH)
    model = _train_model(examples, source_sha256)
    model.save()
    return model


def train_perception_model(
    examples: tuple[PerceptionTrainingExample, ...] | None = None,
) -> PerceptionTrainingResult:
    source_sha256 = file_sha256(PERCEPTION_DATA_PATH)
    examples = examples or load_perception_jsonl(PERCEPTION_DATA_PATH)
    model = _train_model(examples, source_sha256)
    model.save()
    train = tuple(example for example in examples if example.split == "train")
    test = tuple(example for example in examples if example.split == "test")
    from ..perception.learning import evaluate_perception_model
    train_result = evaluate_perception_model(model, train)
    test_result = evaluate_perception_model(model, test)
    default_perception_model.cache_clear()
    return PerceptionTrainingResult(len(examples), train_result.accuracy, test_result.accuracy, source_sha256)


__all__ = (
    "PERCEPTION_NEURAL_META_PATH",
    "PERCEPTION_NEURAL_SCHEMA",
    "PERCEPTION_NEURAL_WEIGHTS_PATH",
    "PerceptionNeuralModel",
    "PerceptionTrainingResult",
    "default_perception_model",
    "train_perception_model",
)
