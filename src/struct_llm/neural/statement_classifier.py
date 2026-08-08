from __future__ import annotations

import json
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from .common import (
    balanced_class_weights,
    build_character_vocabulary,
    encode_characters as encode_text,
    file_sha256,
    masked_max,
    masked_mean,
    sequence_mask,
    set_seed,
)
from ..comprehension.statement import (
    STATEMENT_DATA_PATH,
    EntitySlot,
    FrameTemplate,
    StatementTrainingExample,
    evaluate_statement_parser,
    extract_slots,
    frame_template_to_dict,
    load_statement_jsonl,
    normalize_entity_value,
    normalize_statement_text,
)
from ..structure import Entity, Frame, Role


STATEMENT_NEURAL_SCHEMA = "struct_llm.statement_neural_model.v1"
STATEMENT_NEURAL_WEIGHTS_PATH = Path(__file__).resolve().parents[3] / "data" / "statement_neural_model.pt"
STATEMENT_NEURAL_META_PATH = Path(__file__).resolve().parents[3] / "data" / "statement_neural_model.json"
STATEMENT_NEURAL_PAD = "<pad>"
STATEMENT_NEURAL_UNK = "<unk>"
STATEMENT_NEURAL_MIN_CONFIDENCE = 0.60
STATEMENT_NEURAL_EMBED_DIM = 128
STATEMENT_NEURAL_HIDDEN_DIM = 128
STATEMENT_NEURAL_DROPOUT = 0.15
STATEMENT_NEURAL_BATCH_SIZE = 32
STATEMENT_NEURAL_EPOCHS = 120
STATEMENT_NEURAL_LR = 2e-3
STATEMENT_NEURAL_TAG_LOSS_WEIGHT = 1.00
STATEMENT_NEURAL_SEED = 20260806


@dataclass(frozen=True)
class NeuralStatementPattern:
    entities: tuple[EntitySlot, ...]
    frames: tuple[FrameTemplate, ...]
    support: int = 1


@dataclass(frozen=True)
class NeuralStatementTrainingExample:
    text: str
    label_key: str
    tag_ids: tuple[int, ...]


@dataclass(frozen=True)
class StatementNeuralModelState:
    schema: str
    source_sha256: str
    vocab: dict[str, int]
    tag_labels: tuple[str, ...]
    label_keys: tuple[str, ...]
    patterns: tuple[NeuralStatementPattern, ...]
    embed_dim: int
    hidden_dim: int
    dropout: float
    min_confidence: float
    example_count: int


@dataclass(frozen=True)
class StatementNeuralTrainingResult:
    example_count: int
    label_count: int
    tag_count: int
    train_accuracy: float
    train_loss: float
    source_sha256: str


@dataclass(frozen=True)
class StatementNeuralTrainingBundle:
    parser: LoadedNeuralStatementParser
    result: StatementNeuralTrainingResult


class NeuralStatementDataset(Dataset):
    def __init__(
        self,
        examples: tuple[NeuralStatementTrainingExample, ...],
        vocab: dict[str, int],
        label_index: dict[str, int],
    ) -> None:
        self._examples = examples
        self._vocab = vocab
        self._label_index = label_index

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> tuple[list[int], int, list[int]]:
        example = self._examples[index]
        return encode_text(example.text, self._vocab), self._label_index[example.label_key], list(example.tag_ids)


class NeuralStatementClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        label_count: int,
        tag_count: int,
        *,
        embed_dim: int = STATEMENT_NEURAL_EMBED_DIM,
        hidden_dim: int = STATEMENT_NEURAL_HIDDEN_DIM,
        dropout: float = STATEMENT_NEURAL_DROPOUT,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, label_count),
        )
        self.tagger = nn.Linear(hidden_dim * 2, tag_count)

    def forward(self, token_ids: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        embedded = self.embedding(token_ids)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.encoder(packed)
        outputs, _ = pad_packed_sequence(packed_output, batch_first=True)
        mask = sequence_mask(lengths, outputs.size(1), outputs.device)
        forward_last = hidden[-2]
        backward_last = hidden[-1]
        mean_pool = masked_mean(outputs, mask)
        max_pool = masked_max(outputs, mask)
        features = torch.cat((forward_last, backward_last, mean_pool, max_pool), dim=1)
        return self.classifier(self.dropout(features)), self.tagger(outputs)


@dataclass(frozen=True)
class LoadedNeuralStatementParser:
    model: NeuralStatementClassifier
    vocab: dict[str, int]
    tag_labels: tuple[str, ...]
    label_keys: tuple[str, ...]
    patterns: tuple[NeuralStatementPattern, ...]
    min_confidence: float = STATEMENT_NEURAL_MIN_CONFIDENCE

    def __call__(self, sentence: str) -> tuple[list[Entity], list[Frame]] | None:
        normalized = normalize_statement_text(sentence)
        if not normalized:
            return None
        self.model.eval()
        with torch.no_grad():
            token_ids = torch.tensor([encode_text(normalized, self.vocab)], dtype=torch.long)
            lengths = torch.tensor([token_ids.shape[1]], dtype=torch.long)
            class_logits, tag_logits = self.model(token_ids, lengths)
            probabilities = torch.softmax(class_logits, dim=-1)[0]
            label_index, confidence = select_statement_label(probabilities, self.patterns)
            if confidence < self.min_confidence:
                return None
            predicted_tags = torch.argmax(tag_logits, dim=-1)[0].tolist()[: int(lengths[0].item())]

        pattern = self.patterns[label_index]
        if not pattern.frames:
            return None
        entities = decode_entities(normalized, predicted_tags, self.tag_labels)
        entities = expand_short_entity_boundaries(normalized, entities)
        entities = recover_missing_actor_entities(normalized, entities, pattern.entities)
        entities = filter_expected_entities(entities, pattern.entities)
        if not expected_entities_present(entities, pattern.entities):
            return None
        frames = instantiate_pattern_frames(pattern, entities)
        if frames is None:
            return None
        if not integrated_frame_roles_have_text_evidence(normalized, frames):
            return None
        if not statement_has_required_frame_text_evidence(normalized, frames):
            return None
        return entities, frames


def default_neural_statement_parser(
    statement_data_path: str | Path = STATEMENT_DATA_PATH,
    weights_path: str | Path = STATEMENT_NEURAL_WEIGHTS_PATH,
    meta_path: str | Path = STATEMENT_NEURAL_META_PATH,
) -> LoadedNeuralStatementParser:
    data_path = Path(statement_data_path)
    weights = Path(weights_path)
    metadata = Path(meta_path)
    source_sha = file_sha256(data_path)
    return _default_neural_statement_parser_from_signature(str(data_path), str(weights), str(metadata), source_sha)


@lru_cache(maxsize=8)
def _default_neural_statement_parser_from_signature(
    data_path: str,
    weights_path: str,
    meta_path: str,
    source_sha: str,
) -> LoadedNeuralStatementParser:
    data = Path(data_path)
    weights = Path(weights_path)
    metadata = Path(meta_path)
    if weights.exists() and metadata.exists():
        state = load_statement_neural_metadata(metadata)
        if state.source_sha256 == source_sha:
            return load_statement_neural_parser(weights, metadata)
    return train_statement_neural_model(data, weights, metadata).parser


def train_statement_neural_model(
    statement_data_path: str | Path = STATEMENT_DATA_PATH,
    weights_path: str | Path = STATEMENT_NEURAL_WEIGHTS_PATH,
    meta_path: str | Path = STATEMENT_NEURAL_META_PATH,
) -> StatementNeuralTrainingBundle:
    data_path = Path(statement_data_path)
    weights = Path(weights_path)
    metadata_path = Path(meta_path)
    examples = load_statement_jsonl(data_path)
    source_sha = file_sha256(data_path)
    patterns = representative_patterns_by_label(examples)
    label_keys = tuple(sorted(patterns))
    pattern_list = tuple(patterns[key] for key in label_keys)
    roles = sorted({entity.role for example in examples for entity in example.entities})
    tag_labels = ("O",) + tuple(
        label
        for role in roles
        for label in (f"B:{role}", f"I:{role}")
    )
    tag_index = {label: index for index, label in enumerate(tag_labels)}
    neural_examples = tuple(
        build_training_example(example, tag_index, label_key(example))
        for example in examples
    )
    vocab = build_character_vocabulary((example.text for example in neural_examples), sort_characters=True)
    label_index = {key: index for index, key in enumerate(label_keys)}
    dataset = NeuralStatementDataset(neural_examples, vocab, label_index)
    set_seed(STATEMENT_NEURAL_SEED)
    model = NeuralStatementClassifier(len(vocab), len(label_keys), len(tag_labels))
    result = fit_statement_classifier(model, dataset, source_sha256=source_sha)
    state = StatementNeuralModelState(
        schema=STATEMENT_NEURAL_SCHEMA,
        source_sha256=source_sha,
        vocab=vocab,
        tag_labels=tag_labels,
        label_keys=label_keys,
        patterns=pattern_list,
        embed_dim=STATEMENT_NEURAL_EMBED_DIM,
        hidden_dim=STATEMENT_NEURAL_HIDDEN_DIM,
        dropout=STATEMENT_NEURAL_DROPOUT,
        min_confidence=STATEMENT_NEURAL_MIN_CONFIDENCE,
        example_count=len(examples),
    )
    save_statement_neural_model(model, state, weights, metadata_path)
    parser = LoadedNeuralStatementParser(
        model=model,
        vocab=vocab,
        tag_labels=tag_labels,
        label_keys=label_keys,
        patterns=pattern_list,
    )
    return StatementNeuralTrainingBundle(parser=parser, result=result)


def fit_statement_classifier(
    model: NeuralStatementClassifier,
    dataset: NeuralStatementDataset,
    *,
    source_sha256: str = "",
) -> StatementNeuralTrainingResult:
    if not len(dataset):
        raise ValueError("Statement training dataset is empty.")
    model.train()
    label_count = model.classifier[-1].out_features
    tag_count = model.tagger.out_features
    class_counts = torch.zeros(label_count, dtype=torch.float32)
    for _, label, _ in dataset:
        class_counts[label] += 1
    class_weights = balanced_class_weights(class_counts)
    class_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    tag_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=STATEMENT_NEURAL_LR)
    loader = DataLoader(dataset, batch_size=STATEMENT_NEURAL_BATCH_SIZE, shuffle=True, collate_fn=collate_statement_batch)
    total_loss = 0.0
    for _ in range(STATEMENT_NEURAL_EPOCHS):
        for token_ids, lengths, labels, tag_labels in loader:
            optimizer.zero_grad()
            class_logits, tag_logits = model(token_ids, lengths)
            class_loss = class_loss_fn(class_logits, labels)
            tag_loss = tag_loss_fn(tag_logits.reshape(-1, tag_count), tag_labels.reshape(-1))
            loss = class_loss + STATEMENT_NEURAL_TAG_LOSS_WEIGHT * tag_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(labels.shape[0])
    accuracy = evaluate_statement_classifier(model, dataset)
    return StatementNeuralTrainingResult(
        example_count=len(dataset),
        label_count=label_count,
        tag_count=tag_count,
        train_accuracy=accuracy,
        train_loss=total_loss / len(dataset) / STATEMENT_NEURAL_EPOCHS,
        source_sha256=source_sha256,
    )


def evaluate_statement_classifier(model: NeuralStatementClassifier, dataset: NeuralStatementDataset) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=STATEMENT_NEURAL_BATCH_SIZE, shuffle=False, collate_fn=collate_statement_batch)
    matched = 0
    total = 0
    with torch.no_grad():
        for token_ids, lengths, labels, _ in loader:
            class_logits, _ = model(token_ids, lengths)
            predictions = torch.argmax(class_logits, dim=-1)
            matched += int((predictions == labels).sum().item())
            total += int(labels.shape[0])
    return matched / total if total else 0.0


def statement_neural_summary(
    statement_data_path: str | Path = STATEMENT_DATA_PATH,
    weights_path: str | Path = STATEMENT_NEURAL_WEIGHTS_PATH,
    meta_path: str | Path = STATEMENT_NEURAL_META_PATH,
) -> dict[str, Any]:
    parser = default_neural_statement_parser(statement_data_path, weights_path, meta_path)
    examples = load_statement_jsonl(statement_data_path)
    evaluation = evaluate_statement_parser(parser, examples)
    state = load_statement_neural_metadata(meta_path)
    return {
        "examples": len(examples),
        "labels": len(state.label_keys),
        "accuracy": round(evaluation.accuracy, 4),
        "classifier_accuracy": round(
            evaluate_statement_classifier(
                parser.model,
                NeuralStatementDataset(
                    tuple(
                        build_training_example(example, {label: index for index, label in enumerate(state.tag_labels)}, label_key(example))
                        for example in examples
                    ),
                    state.vocab,
                    {key: index for index, key in enumerate(state.label_keys)},
                ),
            ),
            4,
        ),
        "min_confidence": state.min_confidence,
    }


def load_statement_neural_parser(
    weights_path: str | Path = STATEMENT_NEURAL_WEIGHTS_PATH,
    meta_path: str | Path = STATEMENT_NEURAL_META_PATH,
) -> LoadedNeuralStatementParser:
    state = load_statement_neural_metadata(meta_path)
    model = NeuralStatementClassifier(
        len(state.vocab),
        len(state.label_keys),
        len(state.tag_labels),
        embed_dim=state.embed_dim,
        hidden_dim=state.hidden_dim,
        dropout=state.dropout,
    )
    model.load_state_dict(torch.load(Path(weights_path), map_location="cpu"))
    model.eval()
    return LoadedNeuralStatementParser(
        model=model,
        vocab=state.vocab,
        tag_labels=state.tag_labels,
        label_keys=state.label_keys,
        patterns=state.patterns,
        min_confidence=state.min_confidence,
    )


def load_statement_neural_metadata(path: str | Path) -> StatementNeuralModelState:
    with Path(path).open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict) or raw.get("schema") != STATEMENT_NEURAL_SCHEMA:
        raise ValueError("Unsupported neural statement model schema.")
    raw_patterns = raw.get("patterns", [])
    if not isinstance(raw_patterns, list):
        raise ValueError("Neural statement patterns must be a list.")
    return StatementNeuralModelState(
        schema=STATEMENT_NEURAL_SCHEMA,
        source_sha256=str(raw.get("source_sha256") or ""),
        vocab={str(key): int(value) for key, value in dict(raw.get("vocab") or {}).items()},
        tag_labels=tuple(str(value) for value in raw.get("tag_labels", [])),
        label_keys=tuple(str(value) for value in raw.get("label_keys", [])),
        patterns=tuple(statement_pattern_from_dict(value) for value in raw_patterns),
        embed_dim=int(raw.get("embed_dim") or STATEMENT_NEURAL_EMBED_DIM),
        hidden_dim=int(raw.get("hidden_dim") or STATEMENT_NEURAL_HIDDEN_DIM),
        dropout=float(raw.get("dropout") or STATEMENT_NEURAL_DROPOUT),
        min_confidence=float(raw.get("min_confidence") or STATEMENT_NEURAL_MIN_CONFIDENCE),
        example_count=int(raw.get("example_count") or 0),
    )


def save_statement_neural_model(
    model: NeuralStatementClassifier,
    state: StatementNeuralModelState,
    weights_path: str | Path,
    meta_path: str | Path,
) -> None:
    weights = Path(weights_path)
    metadata = Path(meta_path)
    weights.parent.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights)
    with metadata.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "schema": state.schema,
                "source_sha256": state.source_sha256,
                "vocab": state.vocab,
                "tag_labels": list(state.tag_labels),
                "label_keys": list(state.label_keys),
                "patterns": [statement_pattern_to_dict(pattern) for pattern in state.patterns],
                "embed_dim": state.embed_dim,
                "hidden_dim": state.hidden_dim,
                "dropout": state.dropout,
                "min_confidence": state.min_confidence,
                "example_count": state.example_count,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")


def representative_patterns_by_label(
    examples: tuple[StatementTrainingExample, ...],
) -> dict[str, NeuralStatementPattern]:
    grouped: dict[str, NeuralStatementPattern] = {}
    for example in examples:
        key = label_key(example)
        previous = grouped.get(key)
        if previous is None:
            grouped[key] = NeuralStatementPattern(example.entities, example.frames, support=1)
        else:
            grouped[key] = NeuralStatementPattern(previous.entities, previous.frames, support=previous.support + 1)
    return grouped


def label_key(example: StatementTrainingExample) -> str:
    return json.dumps(
        {
            "entities": [{"role": entity.role, "name": entity.name} for entity in example.entities],
            "frames": [frame_template_to_dict(frame) for frame in example.frames],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def statement_pattern_to_dict(pattern: NeuralStatementPattern) -> dict[str, Any]:
    return {
        "entities": [{"role": entity.role, "name": entity.name} for entity in pattern.entities],
        "frames": [frame_template_to_dict(frame) for frame in pattern.frames],
        "support": pattern.support,
    }


def statement_pattern_from_dict(record: Any) -> NeuralStatementPattern:
    if not isinstance(record, dict):
        raise ValueError("Neural statement pattern must be an object.")
    raw_entities = record.get("entities", [])
    raw_frames = record.get("frames", [])
    if not isinstance(raw_entities, list) or not isinstance(raw_frames, list):
        raise ValueError("Neural statement pattern entities and frames must be lists.")
    entities = tuple(
        EntitySlot(str(value.get("role") or ""), str(value.get("name") or ""))
        for value in raw_entities
        if isinstance(value, dict)
    )
    frames = tuple(
        FrameTemplate(
            str(value.get("frame_type") or ""),
            tuple((str(key), str(item)) for key, item in dict(value.get("roles") or {}).items()),
        )
        for value in raw_frames
        if isinstance(value, dict)
    )
    return NeuralStatementPattern(entities, frames, int(record.get("support") or 1))


def select_statement_label(
    probabilities: Tensor,
    patterns: tuple[NeuralStatementPattern, ...],
) -> tuple[int, float]:
    confidence, label_index = torch.max(probabilities, dim=-1)
    best_index = int(label_index.item())
    best_confidence = float(confidence.item())
    grouped: dict[tuple[Any, ...], tuple[float, int]] = {}
    for index, pattern in enumerate(patterns):
        signature = statement_semantic_signature(pattern)
        total, representative = grouped.get(signature, (0.0, index))
        score = float(probabilities[index].item())
        if score > float(probabilities[representative].item()):
            representative = index
        grouped[signature] = (total + score, representative)
    if not grouped:
        return best_index, best_confidence
    aggregate, representative = max(grouped.values(), key=lambda item: item[0])
    if aggregate > best_confidence:
        return representative, aggregate
    return best_index, best_confidence


def statement_semantic_signature(pattern: NeuralStatementPattern) -> tuple[Any, ...]:
    entity_counts: dict[str, int] = {}
    for entity in pattern.entities:
        entity_counts[entity.role] = entity_counts.get(entity.role, 0) + 1
    return (
        tuple(sorted(entity_counts.items())),
        tuple((frame.frame_type, tuple(frame.roles)) for frame in pattern.frames),
    )


def build_training_example(
    example: StatementTrainingExample,
    tag_index: dict[str, int],
    semantic_label: str,
) -> NeuralStatementTrainingExample:
    text = normalize_statement_text(example.sentence)
    tags = [tag_index["O"]] * len(text)
    slots = extract_slots(example.sentence_template, text) or {}
    cursor = 0
    for entity in example.entities:
        value = normalize_entity_value(entity.role, slots.get(entity.name, ""))
        if not value:
            continue
        start = text.find(value, cursor)
        if start < 0:
            start = text.find(value)
        if start < 0:
            continue
        end = start + len(value)
        begin_label = tag_index.get(f"B:{entity.role}")
        inside_label = tag_index.get(f"I:{entity.role}")
        if begin_label is None or inside_label is None:
            continue
        tags[start] = begin_label
        for index in range(start + 1, min(end, len(tags))):
            tags[index] = inside_label
        cursor = end
    return NeuralStatementTrainingExample(text=text, label_key=semantic_label, tag_ids=tuple(tags))


def decode_entities(text: str, tag_ids: list[int], tag_labels: tuple[str, ...]) -> list[Entity]:
    entities: list[Entity] = []
    current_role = ""
    current_start = 0

    def flush(end: int) -> None:
        nonlocal current_role, current_start
        if current_role and current_start < end:
            value = normalize_entity_value(current_role, text[current_start:end])
            if value:
                entities.append(Entity(current_role, value))
        current_role = ""
        current_start = end

    for index, tag_id in enumerate(tag_ids):
        label = tag_labels[tag_id] if 0 <= tag_id < len(tag_labels) else "O"
        if label.startswith("B:"):
            flush(index)
            current_role = label[2:]
            current_start = index
        elif label.startswith("I:"):
            role = label[2:]
            if current_role == role:
                continue
            flush(index)
            current_role = role
            current_start = index
        else:
            flush(index)
    flush(len(tag_ids))
    return entities


def expand_short_entity_boundaries(text: str, entities: list[Entity]) -> list[Entity]:
    return [expand_short_entity_boundary(text, entity) for entity in entities]


def expand_short_entity_boundary(text: str, entity: Entity) -> Entity:
    if entity.role not in EXPANDABLE_SHORT_ENTITY_ROLES or len(entity.name) != 1:
        return entity
    start = text.find(entity.name)
    if start < 0:
        return entity
    end = start + len(entity.name)
    while end < len(text) and text[end] not in ENTITY_BOUNDARY_STOP_CHARS:
        end += 1
    expanded = text[start:end].strip()
    if len(expanded) <= len(entity.name):
        return entity
    return Entity(entity.role, expanded)


EXPANDABLE_SHORT_ENTITY_ROLES = frozenset(
    {"person", "giver", "receiver", "item", "thing", "container", "place", "profile_value", "color"}
)
ENTITY_BOUNDARY_STOP_CHARS = frozenset(" ，,。！？!?；;、的是了吗呢吧和与及把被从到给在里中上下注放交搬带取拿")


def recover_missing_actor_entities(
    text: str,
    entities: list[Entity],
    expected: tuple[EntitySlot, ...],
) -> list[Entity]:
    if any(entity.role == "person" for entity in entities):
        return entities
    if not any(entity.role == "person" for entity in expected):
        return entities
    actor = leading_actor_before_action(text)
    if not actor:
        return entities
    return [Entity("person", actor), *entities]


def leading_actor_before_action(text: str) -> str:
    stripped = strip_clause_prefix(text)
    for marker in ("把", "从", "给", "打开", "关闭", "关上", "合上"):
        index = stripped.find(marker)
        if index <= 0:
            continue
        candidate = stripped[:index].strip()
        if candidate and len(candidate) <= 6:
            return candidate
    return ""


def strip_clause_prefix(text: str) -> str:
    stripped = text.strip()
    changed = True
    while changed:
        changed = False
        for prefix in ("因为", "由于", "既然", "如果", "假如", "假使", "后来", "随后", "起初", "先", "刚才"):
            if stripped.startswith(prefix) and len(stripped) > len(prefix):
                stripped = stripped[len(prefix) :].strip()
                changed = True
    return stripped


def filter_expected_entities(entities: list[Entity], expected: tuple[EntitySlot, ...]) -> list[Entity]:
    counts: dict[str, int] = {}
    for entity in expected:
        counts[entity.role] = counts.get(entity.role, 0) + 1
    used: dict[str, int] = {}
    filtered: list[Entity] = []
    for entity in entities:
        if entity.role not in counts:
            continue
        if used.get(entity.role, 0) >= counts[entity.role]:
            continue
        filtered.append(entity)
        used[entity.role] = used.get(entity.role, 0) + 1
    return filtered


def expected_entities_present(entities: list[Entity], expected: tuple[EntitySlot, ...]) -> bool:
    actual_counts: dict[str, int] = {}
    expected_counts: dict[str, int] = {}
    for entity in entities:
        actual_counts[entity.role] = actual_counts.get(entity.role, 0) + 1
    for entity in expected:
        expected_counts[entity.role] = expected_counts.get(entity.role, 0) + 1
    return all(actual_counts.get(role, 0) >= count for role, count in expected_counts.items())


def instantiate_pattern_frames(
    pattern: NeuralStatementPattern,
    entities: list[Entity],
) -> list[Frame] | None:
    role_entities: dict[str, list[Entity]] = {}
    for entity in entities:
        role_entities.setdefault(entity.role, []).append(entity)
    frames: list[Frame] = []
    for template in pattern.frames:
        roles: list[Role] = []
        for role_name, value in template.roles:
            resolved = resolve_pattern_value(value, role_entities)
            if resolved is None:
                return None
            roles.append(Role("pending", role_name, resolved))
        frames.append(Frame("pending", template.frame_type, 0, tuple(roles)))
    return frames


def resolve_pattern_value(value: str, role_entities: dict[str, list[Entity]]) -> str | None:
    resolved = value
    for role, index_text in placeholder_parts(value):
        index = int(index_text) - 1
        candidates = role_entities.get(role, [])
        if index < 0 or index >= len(candidates):
            return None
        resolved = resolved.replace(f"${role}#{index + 1}", candidates[index].name)
    return resolved.replace("$", "") if "$" in resolved else resolved


def placeholder_parts(value: str) -> tuple[tuple[str, str], ...]:
    parts: list[tuple[str, str]] = []
    index = 0
    while index < len(value):
        start = value.find("$", index)
        if start < 0:
            break
        hash_index = value.find("#", start + 1)
        if hash_index < 0:
            break
        end = hash_index + 1
        while end < len(value) and value[end].isdigit():
            end += 1
        parts.append((value[start + 1 : hash_index], value[hash_index + 1 : end]))
        index = end
    return tuple(parts)


def integrated_frame_roles_have_text_evidence(text: str, frames: list[Frame]) -> bool:
    checked_roles = {
        "if_then": ("antecedent", "consequent"),
        "because": ("cause", "effect"),
    }
    for frame in frames:
        for role_name in checked_roles.get(frame.frame_type, ()):
            value = frame.role(role_name)
            if value and normalize_statement_text(value) not in text:
                return False
    return True


def statement_has_required_frame_text_evidence(text: str, frames: list[Frame]) -> bool:
    if not frames:
        return False
    # Semantic evidence comes from the trained label and role tagger. Keep
    # this final check surface-neutral: only empty placeholder values are
    # rejected here, while lexical variation is learned from the dataset.
    return not any(
        role.value.strip() in {"……", "..."}
        for frame in frames
        for role in frame.roles
        if role.value is not None
    )


def collate_statement_batch(batch: list[tuple[list[int], int, list[int]]]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    lengths = torch.tensor([len(item[0]) for item in batch], dtype=torch.long)
    max_length = int(lengths.max().item())
    token_ids = torch.zeros((len(batch), max_length), dtype=torch.long)
    tag_ids = torch.full((len(batch), max_length), -100, dtype=torch.long)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    for row, (tokens, _, tags) in enumerate(batch):
        token_ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        tag_ids[row, : len(tags)] = torch.tensor(tags, dtype=torch.long)
    return token_ids, lengths, labels, tag_ids
