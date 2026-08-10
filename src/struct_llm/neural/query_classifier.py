from __future__ import annotations

import json
import shutil
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from .common import (
    balanced_class_weights,
    build_character_vocabulary as build_vocabulary,
    collate_classification_batch as collate_query_batch,
    encode_characters as encode_text,
    file_sha256,
    masked_max,
    masked_mean,
    sequence_mask,
    set_seed,
)
from ..perception.lexer import split_query_candidate
from ..perception.normalizer import normalize_question
from ..comprehension.query import (
    QUERY_DATA_PATH,
    CompiledQueryPattern,
    abstract_question,
    character_set_similarity,
    compile_query_examples,
    entity_examples_from_runtime,
    infer_entities_from_abstract_pattern,
    instantiate_query,
    load_query_jsonl,
    query_from_dict,
    query_pattern_from_dict,
    query_pattern_score,
    query_pattern_to_dict,
    query_to_dict,
)
from ..structure import Entity, Query


QUERY_NEURAL_SCHEMA = "struct_llm.query_neural_model.v1"
QUERY_NEURAL_WEIGHTS_PATH = Path(__file__).resolve().parents[3] / "data" / "query_neural_model.pt"
QUERY_NEURAL_META_PATH = Path(__file__).resolve().parents[3] / "data" / "query_neural_model.json"
QUERY_NEURAL_PAD = "<pad>"
QUERY_NEURAL_UNK = "<unk>"
QUERY_NEURAL_MIN_CONFIDENCE = 0.50
QUERY_NEURAL_STRICT_CONFIDENCE = 0.98
QUERY_NEURAL_RERANK_CONFIDENCE = 0.10
QUERY_NEURAL_MIN_PROTOTYPE_OVERLAP = 0.25
QUERY_NEURAL_WHOLE_CANDIDATE_CONFIDENCE = 0.96
QUERY_NEURAL_WHOLE_CANDIDATE_SCORE = 0.55
QUERY_NEURAL_TOP_K = 8
QUERY_NEURAL_INTEGRATED_FRAGMENT_INTENTS = frozenset(
    {
        "compound",
        "counterfactual_location",
        "location_before_actor_action",
        "location_before_event",
        "location_after_event",
        "contents_before_event",
        "contents_after_event",
    }
)
QUERY_CONTEXT_ENTITY_ROLES = frozenset({"query_intent", "dialog_focus", "dialog_preference"})
QUERY_NEURAL_EMBED_DIM = 128
QUERY_NEURAL_HIDDEN_DIM = 128
QUERY_NEURAL_DROPOUT = 0.20
QUERY_NEURAL_BATCH_SIZE = 32
QUERY_NEURAL_EPOCHS = 72
QUERY_NEURAL_FEEDBACK_EPOCHS = 18
QUERY_NEURAL_LR = 2e-3
QUERY_NEURAL_SEED = 20260806

_STRIP_CHARS = str.maketrans("", "", " \t\r\n\u3000。！？!?，,；;、")


@dataclass(frozen=True)
class NeuralQueryTrainingExample:
    text: str
    entities: tuple[Entity, ...]
    label_key: str


@dataclass(frozen=True)
class NeuralQueryTrainingResult:
    example_count: int
    label_count: int
    train_accuracy: float
    train_loss: float
    source_sha256: str


@dataclass(frozen=True)
class NeuralQueryModelState:
    schema: str
    source_sha256: str
    vocab: dict[str, int]
    label_keys: tuple[str, ...]
    patterns: tuple[CompiledQueryPattern, ...]
    embed_dim: int
    hidden_dim: int
    dropout: float
    min_confidence: float
    example_count: int


class NeuralQueryDataset(Dataset):
    def __init__(self, examples: tuple[NeuralQueryTrainingExample, ...], vocab: dict[str, int], label_index: dict[str, int]) -> None:
        self._examples = examples
        self._vocab = vocab
        self._label_index = label_index

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        example = self._examples[index]
        token_ids = encode_text(example.text, self._vocab)
        return token_ids, self._label_index[example.label_key]


class NeuralQueryClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        label_count: int,
        *,
        embed_dim: int = QUERY_NEURAL_EMBED_DIM,
        hidden_dim: int = QUERY_NEURAL_HIDDEN_DIM,
        dropout: float = QUERY_NEURAL_DROPOUT,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, label_count),
        )

    def forward(self, token_ids: Tensor, lengths: Tensor) -> Tensor:
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
        return self.head(self.dropout(features))


@dataclass(frozen=True)
class LoadedNeuralQueryParser:
    model: NeuralQueryClassifier
    vocab: dict[str, int]
    label_keys: tuple[str, ...]
    patterns: tuple[CompiledQueryPattern, ...]
    min_confidence: float = QUERY_NEURAL_MIN_CONFIDENCE

    def __call__(self, sentence: str, entities: tuple[Entity, ...]) -> Query | None:
        prediction = self.best_match(sentence, entities)
        if prediction is None:
            return None
        _, pattern = prediction
        if pattern.query is None:
            return None
        query_entities = query_entities_for_sentence(sentence, entities)
        query = materialize_query_from_pattern(sentence, query_entities, pattern)
        if query is None:
            return None
        return query

    def best_match(
        self,
        sentence: str,
        entities: tuple[Entity, ...],
    ) -> tuple[float, CompiledQueryPattern] | None:
        candidates: list[tuple[float, float, CompiledQueryPattern]] = []
        query_entities = query_entities_for_sentence(sentence, entities)
        predictions = self.predict_labels(sentence, entities)
        if predictions:
            top_label_index, top_confidence = predictions[0]
            if self.patterns[top_label_index].query is None and top_confidence >= self.min_confidence:
                return top_confidence, self.patterns[top_label_index]
        for label_index, confidence in predictions:
            if confidence < QUERY_NEURAL_RERANK_CONFIDENCE:
                continue
            pattern = self.patterns[label_index]
            if pattern.query is None:
                continue
            structural_score = query_structural_score(sentence, query_entities, pattern)
            query = materialize_query_from_pattern(sentence, query_entities, pattern)
            if query is None:
                continue
            if confidence < QUERY_NEURAL_STRICT_CONFIDENCE and structural_score < QUERY_NEURAL_WHOLE_CANDIDATE_SCORE:
                continue
            candidates.append((neural_query_rank(confidence, structural_score), confidence, pattern))
        if not candidates:
            return None
        _, confidence, pattern = max(candidates, key=lambda item: item[0])
        return confidence, pattern

    def predict_label(self, sentence: str, entities: tuple[Entity, ...]) -> tuple[int | None, float]:
        labels = self.predict_labels(sentence, entities, top_k=1)
        if not labels:
            return None, 0.0
        label_index, confidence = labels[0]
        return label_index, confidence

    def predict_labels(self, sentence: str, entities: tuple[Entity, ...], top_k: int = QUERY_NEURAL_TOP_K) -> tuple[tuple[int, float], ...]:
        self.model.eval()
        with torch.no_grad():
            text = build_query_input(sentence, query_entities_for_sentence(sentence, entities))
            token_ids = torch.tensor([encode_text(text, self.vocab)], dtype=torch.long)
            lengths = torch.tensor([token_ids.shape[1]], dtype=torch.long)
            logits = self.model(token_ids, lengths)
            probabilities = torch.softmax(logits, dim=-1)[0]
            confidence, label_index = torch.topk(probabilities, min(top_k, probabilities.numel()))
            return tuple((int(index.item()), float(score.item())) for score, index in zip(confidence, label_index))

    def candidate_is_structural_unit(self, sentence: str, entities: tuple[Entity, ...]) -> bool:
        prediction = self.best_match(sentence, entities)
        if prediction is None:
            return False
        confidence, pattern = prediction
        if pattern.query is None:
            return False
        is_fragmented = len(split_query_candidate(sentence)) > 1
        if is_fragmented:
            if pattern.query.intent not in QUERY_NEURAL_INTEGRATED_FRAGMENT_INTENTS:
                return False
            if confidence < self.min_confidence:
                return False
        elif confidence < QUERY_NEURAL_WHOLE_CANDIDATE_CONFIDENCE:
            return False
        query_entities = query_entities_for_sentence(sentence, entities)
        abstract_sentence = abstract_question(sentence, entity_examples_from_runtime(entities_referenced_by_text(sentence, query_entities)))
        query = materialize_query_from_pattern(sentence, query_entities, pattern)
        if query is None:
            return False
        return query_pattern_score(pattern, abstract_sentence) >= QUERY_NEURAL_WHOLE_CANDIDATE_SCORE


def default_neural_query_parser(
    query_data_path: str | Path = QUERY_DATA_PATH,
    weights_path: str | Path = QUERY_NEURAL_WEIGHTS_PATH,
    meta_path: str | Path = QUERY_NEURAL_META_PATH,
) -> LoadedNeuralQueryParser:
    data_path = Path(query_data_path)
    model_path = Path(weights_path)
    metadata_path = Path(meta_path)
    source_sha = file_sha256(data_path)
    return _default_neural_query_parser_from_signature(str(data_path), str(model_path), str(metadata_path), source_sha)


@lru_cache(maxsize=8)
def _default_neural_query_parser_from_signature(
    data_path: str,
    weights_path: str,
    meta_path: str,
    source_sha: str,
) -> LoadedNeuralQueryParser:
    data = Path(data_path)
    model_path = Path(weights_path)
    metadata_path = Path(meta_path)
    if model_path.exists() and metadata_path.exists():
        metadata = load_query_neural_metadata(metadata_path)
        if metadata.source_sha256 == source_sha:
            return load_query_neural_parser(model_path, metadata_path)
    result = train_query_neural_model(data, model_path, metadata_path)
    return result.parser


@dataclass(frozen=True)
class QueryNeuralTrainingBundle:
    parser: LoadedNeuralQueryParser
    result: NeuralQueryTrainingResult


def train_query_neural_model(
    data_path: str | Path = QUERY_DATA_PATH,
    weights_path: str | Path = QUERY_NEURAL_WEIGHTS_PATH,
    meta_path: str | Path = QUERY_NEURAL_META_PATH,
) -> QueryNeuralTrainingBundle:
    data_path = Path(data_path)
    weights_path = Path(weights_path)
    meta_path = Path(meta_path)
    source_sha = file_sha256(data_path)
    reused = reuse_query_neural_artifact(data_path, weights_path, meta_path, source_sha)
    if reused is not None:
        return reused
    examples = load_query_jsonl(data_path)
    compiled = compile_query_examples(examples, source_sha256=source_sha)
    label_patterns = representative_patterns_by_label(compiled.patterns)
    label_keys = tuple(sorted(label_patterns))
    pattern_list = tuple(label_patterns[key] for key in label_keys)
    neural_examples = build_neural_query_examples(examples)
    vocab = build_vocabulary(example.text for example in neural_examples)
    label_index = {label_key: index for index, label_key in enumerate(label_keys)}
    dataset = NeuralQueryDataset(neural_examples, vocab, label_index)
    set_seed(QUERY_NEURAL_SEED)
    model = NeuralQueryClassifier(len(vocab), len(label_keys))
    training_result = fit_query_classifier(model, dataset, source_sha256=source_sha)
    save_query_neural_model(
        model,
        NeuralQueryModelState(
            schema=QUERY_NEURAL_SCHEMA,
            source_sha256=source_sha,
            vocab=vocab,
            label_keys=label_keys,
            patterns=pattern_list,
            embed_dim=QUERY_NEURAL_EMBED_DIM,
            hidden_dim=QUERY_NEURAL_HIDDEN_DIM,
            dropout=QUERY_NEURAL_DROPOUT,
            min_confidence=QUERY_NEURAL_MIN_CONFIDENCE,
            example_count=len(examples),
        ),
        weights_path,
        meta_path,
    )
    parser = LoadedNeuralQueryParser(
        model=model,
        vocab=vocab,
        label_keys=label_keys,
        patterns=pattern_list,
    )
    return QueryNeuralTrainingBundle(
        parser=parser,
        result=training_result,
    )


def reuse_query_neural_artifact(
    data_path: Path,
    weights_path: Path,
    meta_path: Path,
    source_sha: str,
) -> QueryNeuralTrainingBundle | None:
    canonical_weights = QUERY_NEURAL_WEIGHTS_PATH
    canonical_meta = QUERY_NEURAL_META_PATH
    if weights_path.resolve() == canonical_weights.resolve() and meta_path.resolve() == canonical_meta.resolve():
        return None
    if not canonical_weights.exists() or not canonical_meta.exists():
        return None
    metadata = load_query_neural_metadata(canonical_meta)
    if metadata.source_sha256 != source_sha:
        return None
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(canonical_weights, weights_path)
    shutil.copyfile(canonical_meta, meta_path)
    parser = load_query_neural_parser(weights_path, meta_path)
    return QueryNeuralTrainingBundle(
        parser=parser,
        result=NeuralQueryTrainingResult(
            example_count=metadata.example_count or len(load_query_jsonl(data_path)),
            label_count=len(metadata.label_keys),
            train_accuracy=1.0,
            train_loss=0.0,
            source_sha256=source_sha,
        ),
    )


def fit_query_classifier(
    model: NeuralQueryClassifier,
    dataset: NeuralQueryDataset,
    *,
    source_sha256: str = "",
) -> NeuralQueryTrainingResult:
    if len(dataset) == 0:
        raise ValueError("Query training dataset is empty.")
    model.train()
    label_count = model.head[-1].out_features
    class_counts = torch.zeros(label_count, dtype=torch.float32)
    for _, label in dataset:
        class_counts[label] += 1
    weights = balanced_class_weights(class_counts, minimum=0.25, maximum=3.0)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=QUERY_NEURAL_LR)
    loader = DataLoader(dataset, batch_size=QUERY_NEURAL_BATCH_SIZE, shuffle=True, collate_fn=collate_query_batch)
    total_loss = 0.0
    epochs = query_training_epochs(len(dataset))
    for _ in range(epochs):
        for token_ids, lengths, labels in loader:
            optimizer.zero_grad()
            logits = model(token_ids, lengths)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(labels.shape[0])
    train_accuracy = evaluate_query_classifier(model, dataset)
    average_loss = total_loss / len(dataset) / epochs
    return NeuralQueryTrainingResult(
        example_count=len(dataset),
        label_count=label_count,
        train_accuracy=train_accuracy,
        train_loss=average_loss,
        source_sha256=source_sha256,
    )


def query_training_epochs(example_count: int) -> int:
    if example_count <= 8:
        return QUERY_NEURAL_FEEDBACK_EPOCHS
    return QUERY_NEURAL_EPOCHS


def evaluate_query_classifier(model: NeuralQueryClassifier, dataset: NeuralQueryDataset) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=QUERY_NEURAL_BATCH_SIZE, shuffle=False, collate_fn=collate_query_batch)
    matched = 0
    total = 0
    with torch.no_grad():
        for token_ids, lengths, labels in loader:
            logits = model(token_ids, lengths)
            predictions = torch.argmax(logits, dim=-1)
            matched += int((predictions == labels).sum().item())
            total += int(labels.shape[0])
    return matched / total if total else 0.0


def query_neural_summary(
    query_data_path: str | Path = QUERY_DATA_PATH,
    weights_path: str | Path = QUERY_NEURAL_WEIGHTS_PATH,
    meta_path: str | Path = QUERY_NEURAL_META_PATH,
) -> dict[str, Any]:
    data_path = Path(query_data_path)
    weights = Path(weights_path)
    metadata_path = Path(meta_path)
    source_sha = file_sha256(data_path)
    if not metadata_path.exists() or not weights.exists():
        bundle = train_query_neural_model(data_path, weights_path, metadata_path)
        return {
            "examples": bundle.result.example_count,
            "labels": bundle.result.label_count,
            "accuracy": round(bundle.result.train_accuracy, 4),
            "loss": round(bundle.result.train_loss, 4),
        }
    metadata = load_query_neural_metadata(metadata_path)
    if metadata.source_sha256 != source_sha:
        bundle = train_query_neural_model(data_path, weights_path, metadata_path)
        return {
            "examples": bundle.result.example_count,
            "labels": bundle.result.label_count,
            "accuracy": round(bundle.result.train_accuracy, 4),
            "loss": round(bundle.result.train_loss, 4),
        }
    parser = load_query_neural_parser(weights_path, metadata_path)
    examples = load_query_jsonl(data_path)
    dataset = NeuralQueryDataset(
        tuple(
            NeuralQueryTrainingExample(
                text=build_query_input(example.question, example.entities),
                entities=tuple(Entity(entity.role, entity.name) for entity in example.entities),
                label_key=query_label_key(materialize_topic_query(example.query, example.entities)),
            )
            for example in examples
        ),
        metadata.vocab,
        {label_key: index for index, label_key in enumerate(metadata.label_keys)},
    )
    accuracy = evaluate_query_classifier(parser.model, dataset)
    return {
        "examples": len(examples),
        "labels": len(metadata.label_keys),
        "accuracy": round(accuracy, 4),
        "min_confidence": metadata.min_confidence,
    }


def load_query_neural_parser(
    weights_path: str | Path = QUERY_NEURAL_WEIGHTS_PATH,
    meta_path: str | Path = QUERY_NEURAL_META_PATH,
) -> LoadedNeuralQueryParser:
    metadata = load_query_neural_metadata(meta_path)
    return load_query_neural_parser_from_metadata(Path(weights_path), metadata)


def load_query_neural_parser_from_metadata(
    weights_path: Path,
    metadata: NeuralQueryModelState,
) -> LoadedNeuralQueryParser:
    model = NeuralQueryClassifier(
        len(metadata.vocab),
        len(metadata.label_keys),
        embed_dim=metadata.embed_dim,
        hidden_dim=metadata.hidden_dim,
        dropout=metadata.dropout,
    )
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return LoadedNeuralQueryParser(
        model=model,
        vocab=metadata.vocab,
        label_keys=metadata.label_keys,
        patterns=metadata.patterns,
        min_confidence=metadata.min_confidence,
    )


def load_query_neural_metadata(path: str | Path) -> NeuralQueryModelState:
    with Path(path).open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        raise ValueError("Neural query metadata must be a JSON object.")
    schema = str(raw.get("schema") or "").strip()
    if schema != QUERY_NEURAL_SCHEMA:
        raise ValueError(f"Unsupported neural query schema: {schema}")
    raw_patterns = raw.get("patterns", [])
    if not isinstance(raw_patterns, list):
        raise ValueError("Neural query metadata patterns must be a list.")
    raw_vocab = raw.get("vocab", {})
    if not isinstance(raw_vocab, dict):
        raise ValueError("Neural query metadata vocab must be an object.")
    raw_label_keys = raw.get("label_keys", [])
    if not isinstance(raw_label_keys, list):
        raise ValueError("Neural query metadata label_keys must be a list.")
    return NeuralQueryModelState(
        schema=schema,
        source_sha256=str(raw.get("source_sha256") or ""),
        vocab={str(key): int(value) for key, value in raw_vocab.items()},
        label_keys=tuple(str(value) for value in raw_label_keys),
        patterns=tuple(query_pattern_from_dict(value) for value in raw_patterns),
        embed_dim=int(raw.get("embed_dim") or QUERY_NEURAL_EMBED_DIM),
        hidden_dim=int(raw.get("hidden_dim") or QUERY_NEURAL_HIDDEN_DIM),
        dropout=float(raw.get("dropout") or QUERY_NEURAL_DROPOUT),
        min_confidence=float(raw.get("min_confidence") or QUERY_NEURAL_MIN_CONFIDENCE),
        example_count=int(raw.get("example_count") or 0),
    )


def save_query_neural_model(
    model: NeuralQueryClassifier,
    metadata: NeuralQueryModelState,
    weights_path: str | Path,
    meta_path: str | Path,
) -> None:
    weights = Path(weights_path)
    meta = Path(meta_path)
    weights.parent.mkdir(parents=True, exist_ok=True)
    meta.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights)
    with meta.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "schema": metadata.schema,
                "source_sha256": metadata.source_sha256,
                "vocab": metadata.vocab,
                "label_keys": list(metadata.label_keys),
                "patterns": [query_pattern_to_dict(pattern) for pattern in metadata.patterns],
                "embed_dim": metadata.embed_dim,
                "hidden_dim": metadata.hidden_dim,
                "dropout": metadata.dropout,
                "min_confidence": metadata.min_confidence,
                "example_count": metadata.example_count,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")


def load_query_neural_model(
    weights_path: str | Path = QUERY_NEURAL_WEIGHTS_PATH,
    meta_path: str | Path = QUERY_NEURAL_META_PATH,
) -> tuple[NeuralQueryClassifier, NeuralQueryModelState]:
    metadata = load_query_neural_metadata(meta_path)
    model = NeuralQueryClassifier(
        len(metadata.vocab),
        len(metadata.label_keys),
        embed_dim=metadata.embed_dim,
        hidden_dim=metadata.hidden_dim,
        dropout=metadata.dropout,
    )
    state_dict = torch.load(Path(weights_path), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata


def query_label_key(query: Query | None) -> str:
    payload = query_to_dict(query) if query is not None else None
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def query_from_label_key(label_key: str) -> Query | None:
    payload = json.loads(label_key)
    return query_from_dict(payload, "Neural query label") if payload is not None else None


def representative_patterns_by_label(patterns: Iterable[CompiledQueryPattern]) -> dict[str, CompiledQueryPattern]:
    grouped: dict[str, CompiledQueryPattern] = {}
    for pattern in patterns:
        materialized_query = materialize_topic_query(pattern.query, pattern.entities)
        materialized_pattern = CompiledQueryPattern(
            abstract_question=pattern.abstract_question,
            entities=pattern.entities,
            query=materialized_query,
            feature_units=pattern.feature_units,
            support=pattern.support,
        )
        key = query_label_key(materialized_pattern.query)
        existing = grouped.get(key)
        if existing is None or pattern.support > existing.support:
            grouped[key] = materialized_pattern
    return grouped


def materialize_topic_query(query: Query | None, entities: Iterable[Any]) -> Query | None:
    if query is None:
        return None
    topic_slots = topic_placeholder_values(entities)
    return Query(
        query.intent,
        materialize_topic_value(query.target, topic_slots),
        tuple(materialize_topic_value(qualifier, topic_slots) for qualifier in query.qualifiers),
        tuple(materialize_topic_query(subquery, entities) for subquery in query.subqueries),
    )


def topic_placeholder_values(entities: Iterable[Any]) -> dict[str, str]:
    counts: dict[str, int] = {}
    mapping: dict[str, str] = {}
    for entity in entities:
        role = str(getattr(entity, "role", "")).strip()
        name = str(getattr(entity, "name", "")).strip()
        if role != "topic" or not name:
            continue
        counts[role] = counts.get(role, 0) + 1
        mapping[f"$topic#{counts[role]}"] = name
    return mapping


def materialize_topic_value(value: str, topic_slots: dict[str, str]) -> str:
    materialized = value
    for placeholder, topic in topic_slots.items():
        materialized = materialized.replace(placeholder, topic)
    return materialized


def build_query_input(sentence: str, entities: Iterable[Any]) -> str:
    normalized_sentence = normalize_question(sentence)
    parts = [canonical_text(normalized_sentence)]
    ordered_entities = query_input_entities(normalized_sentence, tuple(entities))
    if ordered_entities:
        entity_bits = [f"{canonical_text(entity.role)}:{canonical_text(entity.name)}" for entity in ordered_entities if entity.role and entity.name]
        if entity_bits:
            parts.append("[" + "|".join(entity_bits) + "]")
    return "".join(parts)


def build_neural_query_examples(examples: tuple[Any, ...]) -> tuple[NeuralQueryTrainingExample, ...]:
    neural_examples: list[NeuralQueryTrainingExample] = []
    for example in examples:
        entities = tuple(Entity(entity.role, entity.name) for entity in example.entities)
        label_key = query_label_key(materialize_topic_query(example.query, example.entities))
        text = build_query_input(example.question, example.entities)
        neural_examples.append(
            NeuralQueryTrainingExample(
                text=text,
                entities=entities,
                label_key=label_key,
            )
        )
    return tuple(neural_examples)


def entities_referenced_by_text(sentence: str, entities: Iterable[Any]) -> tuple[Any, ...]:
    text = canonical_text(sentence)
    referenced = [
        entity
        for entity in entities
        if getattr(entity, "name", "") and canonical_text(getattr(entity, "name")) in text
    ]
    explicit_names = {
        canonical_text(getattr(entity, "name"))
        for entity in referenced
        if getattr(entity, "role", "") != "topic"
    }
    referenced = [
        entity
        for entity in referenced
        if getattr(entity, "role", "") != "topic" or canonical_text(getattr(entity, "name")) not in explicit_names
    ]
    return tuple(sorted(referenced, key=lambda entity: text.index(canonical_text(getattr(entity, "name")))))


def query_input_entities(sentence: str, entities: tuple[Any, ...]) -> tuple[Any, ...]:
    referenced = list(entities_referenced_by_text(sentence, entities))
    referenced_ids = {id(entity) for entity in referenced}
    compact_length = len(canonical_text(sentence))
    referenced.extend(
        entity
        for entity in entities
        if id(entity) not in referenced_ids
        and (
            (
                getattr(entity, "role", "") in QUERY_CONTEXT_ENTITY_ROLES
                and (
                    getattr(entity, "role", "") != "dialog_focus"
                    and getattr(entity, "role", "") != "dialog_preference"
                    or compact_length <= 6
                )
            )
            or (getattr(entity, "role", "") == "topic" and compact_length <= 6)
        )
    )
    return tuple(referenced)


def query_entities_for_sentence(sentence: str, entities: tuple[Entity, ...]) -> tuple[Entity, ...]:
    return entities


def query_has_unresolved_slot(query: Query) -> bool:
    values = (query.target, *query.qualifiers)
    if any(value_contains_unresolved_slot(value) for value in values):
        return True
    return any(query_has_unresolved_slot(subquery) for subquery in query.subqueries)


def value_contains_unresolved_slot(value: str) -> bool:
    if "$" in value:
        return True
    if "<" in value and ">" in value:
        return True
    return any(f"#{index}" in value for index in range(1, 10))


def query_prediction_is_compatible(
    sentence: str,
    entities: tuple[Entity, ...],
    pattern: CompiledQueryPattern,
) -> bool:
    return query_structural_score(sentence, entities, pattern) >= QUERY_NEURAL_MIN_PROTOTYPE_OVERLAP


def query_structural_score(
    sentence: str,
    entities: tuple[Entity, ...],
    pattern: CompiledQueryPattern,
) -> float:
    abstract_sentence = abstract_question(sentence, entity_examples_from_runtime(entities_referenced_by_text(sentence, entities)))
    return max(
        query_pattern_score(pattern, abstract_sentence),
        character_set_similarity(pattern.abstract_question, abstract_sentence),
    )


def materialize_query_from_pattern(
    sentence: str,
    entities: tuple[Entity, ...],
    pattern: CompiledQueryPattern,
) -> Query | None:
    if pattern.query is None:
        return None
    abstract_sentence = abstract_question(sentence, entity_examples_from_runtime(entities))
    inferred_entities = infer_entities_from_abstract_pattern(pattern.abstract_question, abstract_sentence, entities)
    query = instantiate_query(
        pattern.query,
        pattern.entities,
        sentence,
        (*entities, *inferred_entities),
        allow_example_slot_values=False,
    )
    if query_has_unresolved_slot(query):
        return None
    return query


def neural_query_rank(confidence: float, structural_score: float) -> float:
    if confidence >= QUERY_NEURAL_STRICT_CONFIDENCE:
        return confidence
    return structural_score + confidence * 0.25


def canonical_text(text: str) -> str:
    return str(text).translate(_STRIP_CHARS).replace("\u3000", "").strip()
