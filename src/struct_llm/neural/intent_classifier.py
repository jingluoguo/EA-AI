from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from .common import (
    balanced_class_weights,
    build_character_vocabulary,
    collate_classification_batch,
    encode_characters,
    file_sha256,
    masked_max,
    masked_mean,
    sequence_mask,
    set_seed,
)
from ..comprehension.intent import evaluate_intent_analyzer, from_jsonl, normalize_observation
from ..comprehension.intent_dataset import load_intent_jsonl
from ..structure import Intention, Structure


INTENT_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "intent_examples.jsonl"
INTENT_NEURAL_SCHEMA = "struct_llm.intent_neural_model.v1"
INTENT_NEURAL_WEIGHTS_PATH = Path(__file__).resolve().parents[3] / "data" / "intent_neural_model.pt"
INTENT_NEURAL_META_PATH = Path(__file__).resolve().parents[3] / "data" / "intent_neural_model.json"
INTENT_NEURAL_MIN_CONFIDENCE = 0.45
INTENT_NEURAL_EMBED_DIM = 96
INTENT_NEURAL_HIDDEN_DIM = 96
INTENT_NEURAL_DROPOUT = 0.15
INTENT_NEURAL_BATCH_SIZE = 32
INTENT_NEURAL_EPOCHS = 90
INTENT_NEURAL_LR = 2e-3
INTENT_NEURAL_SEED = 20260808


@dataclass(frozen=True)
class NeuralIntentTrainingExample:
    text: str
    label_key: str


@dataclass(frozen=True)
class IntentNeuralModelState:
    schema: str
    source_sha256: str
    vocab: dict[str, int]
    label_keys: tuple[str, ...]
    intentions: tuple[Intention, ...]
    embed_dim: int
    hidden_dim: int
    dropout: float
    min_confidence: float
    example_count: int


@dataclass(frozen=True)
class IntentNeuralTrainingResult:
    example_count: int
    label_count: int
    train_accuracy: float
    train_loss: float
    source_sha256: str


@dataclass(frozen=True)
class IntentNeuralTrainingBundle:
    analyzer: LoadedNeuralIntentAnalyzer
    result: IntentNeuralTrainingResult


class NeuralIntentDataset(Dataset):
    def __init__(
        self,
        examples: tuple[NeuralIntentTrainingExample, ...],
        vocab: dict[str, int],
        label_index: dict[str, int],
    ) -> None:
        self._examples = examples
        self._vocab = vocab
        self._label_index = label_index

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        example = self._examples[index]
        return encode_characters(example.text, self._vocab), self._label_index[example.label_key]


class NeuralIntentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        label_count: int,
        *,
        embed_dim: int = INTENT_NEURAL_EMBED_DIM,
        hidden_dim: int = INTENT_NEURAL_HIDDEN_DIM,
        dropout: float = INTENT_NEURAL_DROPOUT,
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
class LoadedNeuralIntentAnalyzer:
    model: NeuralIntentClassifier
    vocab: dict[str, int]
    label_keys: tuple[str, ...]
    intentions: tuple[Intention, ...]
    min_confidence: float = INTENT_NEURAL_MIN_CONFIDENCE

    def __call__(self, text: str, structure: Structure) -> tuple[Intention, ...]:
        if structure.query is not None or structure.frames or structure.states:
            return ()
        normalized = build_intent_input(text)
        if not normalized:
            return ()
        self.model.eval()
        with torch.no_grad():
            token_ids = torch.tensor([encode_characters(normalized, self.vocab)], dtype=torch.long)
            lengths = torch.tensor([token_ids.shape[1]], dtype=torch.long)
            logits = self.model(token_ids, lengths)
            probabilities = torch.softmax(logits, dim=-1)[0]
            confidence, label_index = torch.max(probabilities, dim=-1)
        score = float(confidence.item())
        if score < self.min_confidence:
            return ()
        intention = self.intentions[int(label_index.item())]
        return (
            Intention(
                subject=intention.subject,
                goal=intention.goal,
                belief=intention.belief,
                strategy=intention.strategy,
                evidence=intention.evidence,
                confidence=min(1.0, max(intention.confidence, score)),
                source=intention.source,
            ),
        )


def default_neural_intent_analyzer(
    intent_data_path: str | Path = INTENT_DATA_PATH,
    weights_path: str | Path = INTENT_NEURAL_WEIGHTS_PATH,
    meta_path: str | Path = INTENT_NEURAL_META_PATH,
) -> LoadedNeuralIntentAnalyzer:
    data_path = Path(intent_data_path)
    weights = Path(weights_path)
    metadata = Path(meta_path)
    source_sha = file_sha256(data_path)
    return _default_neural_intent_analyzer_from_signature(str(data_path), str(weights), str(metadata), source_sha)


@lru_cache(maxsize=8)
def _default_neural_intent_analyzer_from_signature(
    data_path: str,
    weights_path: str,
    meta_path: str,
    source_sha: str,
) -> LoadedNeuralIntentAnalyzer:
    weights = Path(weights_path)
    metadata = Path(meta_path)
    if weights.exists() and metadata.exists():
        state = load_intent_neural_metadata(metadata)
        if state.source_sha256 == source_sha:
            return load_intent_neural_analyzer(weights, metadata)
    return train_intent_neural_model(data_path, weights, metadata).analyzer


def train_intent_neural_model(
    intent_data_path: str | Path = INTENT_DATA_PATH,
    weights_path: str | Path = INTENT_NEURAL_WEIGHTS_PATH,
    meta_path: str | Path = INTENT_NEURAL_META_PATH,
) -> IntentNeuralTrainingBundle:
    data_path = Path(intent_data_path)
    examples = load_intent_jsonl(data_path)
    source_sha = file_sha256(data_path)
    label_intentions = representative_intentions_by_label(tuple(record.intention for record in examples))
    label_keys = tuple(sorted(label_intentions))
    intentions = tuple(label_intentions[key] for key in label_keys)
    neural_examples = tuple(
        NeuralIntentTrainingExample(build_intent_input(record.observation), intent_label_key(record.intention))
        for record in examples
    )
    vocab = build_character_vocabulary((example.text for example in neural_examples), sort_characters=True)
    label_index = {label: index for index, label in enumerate(label_keys)}
    dataset = NeuralIntentDataset(neural_examples, vocab, label_index)
    set_seed(INTENT_NEURAL_SEED)
    model = NeuralIntentClassifier(len(vocab), len(label_keys))
    result = fit_intent_classifier(model, dataset, source_sha256=source_sha)
    state = IntentNeuralModelState(
        schema=INTENT_NEURAL_SCHEMA,
        source_sha256=source_sha,
        vocab=vocab,
        label_keys=label_keys,
        intentions=intentions,
        embed_dim=INTENT_NEURAL_EMBED_DIM,
        hidden_dim=INTENT_NEURAL_HIDDEN_DIM,
        dropout=INTENT_NEURAL_DROPOUT,
        min_confidence=INTENT_NEURAL_MIN_CONFIDENCE,
        example_count=len(examples),
    )
    save_intent_neural_model(model, state, weights_path, meta_path)
    return IntentNeuralTrainingBundle(
        analyzer=LoadedNeuralIntentAnalyzer(model, vocab, label_keys, intentions),
        result=result,
    )


def fit_intent_classifier(
    model: NeuralIntentClassifier,
    dataset: NeuralIntentDataset,
    *,
    source_sha256: str = "",
) -> IntentNeuralTrainingResult:
    if len(dataset) == 0:
        raise ValueError("Intent training dataset is empty.")
    model.train()
    label_count = model.head[-1].out_features
    class_counts = torch.zeros(label_count, dtype=torch.float32)
    for _, label in dataset:
        class_counts[label] += 1
    loss_fn = nn.CrossEntropyLoss(weight=balanced_class_weights(class_counts, minimum=0.25, maximum=3.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=INTENT_NEURAL_LR)
    loader = DataLoader(dataset, batch_size=INTENT_NEURAL_BATCH_SIZE, shuffle=True, collate_fn=collate_classification_batch)
    total_loss = 0.0
    for _ in range(INTENT_NEURAL_EPOCHS):
        for token_ids, lengths, labels in loader:
            optimizer.zero_grad()
            logits = model(token_ids, lengths)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(labels.shape[0])
    accuracy = evaluate_intent_classifier(model, dataset)
    return IntentNeuralTrainingResult(
        example_count=len(dataset),
        label_count=label_count,
        train_accuracy=accuracy,
        train_loss=total_loss / len(dataset) / INTENT_NEURAL_EPOCHS,
        source_sha256=source_sha256,
    )


def evaluate_intent_classifier(model: NeuralIntentClassifier, dataset: NeuralIntentDataset) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=INTENT_NEURAL_BATCH_SIZE, shuffle=False, collate_fn=collate_classification_batch)
    matched = 0
    total = 0
    with torch.no_grad():
        for token_ids, lengths, labels in loader:
            predictions = torch.argmax(model(token_ids, lengths), dim=-1)
            matched += int((predictions == labels).sum().item())
            total += int(labels.shape[0])
    return matched / total if total else 0.0


def intent_neural_summary(
    intent_data_path: str | Path = INTENT_DATA_PATH,
    weights_path: str | Path = INTENT_NEURAL_WEIGHTS_PATH,
    meta_path: str | Path = INTENT_NEURAL_META_PATH,
) -> dict[str, Any]:
    analyzer = default_neural_intent_analyzer(intent_data_path, weights_path, meta_path)
    records = load_intent_jsonl(intent_data_path)
    examples = from_jsonl(intent_data_path)
    result = evaluate_intent_analyzer(analyzer, examples)
    state = load_intent_neural_metadata(meta_path)
    return {
        "examples": len(records),
        "labels": len(state.label_keys),
        "accuracy": round(result.accuracy, 4),
        "classifier_accuracy": round(
            evaluate_intent_classifier(
                analyzer.model,
                NeuralIntentDataset(
                    tuple(
                        NeuralIntentTrainingExample(
                            build_intent_input(record.observation),
                            intent_label_key(record.intention),
                        )
                        for record in records
                    ),
                    state.vocab,
                    {label: index for index, label in enumerate(state.label_keys)},
                ),
            ),
            4,
        ),
        "min_confidence": state.min_confidence,
    }


def load_intent_neural_analyzer(
    weights_path: str | Path = INTENT_NEURAL_WEIGHTS_PATH,
    meta_path: str | Path = INTENT_NEURAL_META_PATH,
) -> LoadedNeuralIntentAnalyzer:
    state = load_intent_neural_metadata(meta_path)
    model = NeuralIntentClassifier(
        len(state.vocab),
        len(state.label_keys),
        embed_dim=state.embed_dim,
        hidden_dim=state.hidden_dim,
        dropout=state.dropout,
    )
    model.load_state_dict(torch.load(Path(weights_path), map_location="cpu"))
    model.eval()
    return LoadedNeuralIntentAnalyzer(
        model=model,
        vocab=state.vocab,
        label_keys=state.label_keys,
        intentions=state.intentions,
        min_confidence=state.min_confidence,
    )


def load_intent_neural_metadata(path: str | Path) -> IntentNeuralModelState:
    with Path(path).open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict) or raw.get("schema") != INTENT_NEURAL_SCHEMA:
        raise ValueError("Unsupported neural intent model schema.")
    return IntentNeuralModelState(
        schema=INTENT_NEURAL_SCHEMA,
        source_sha256=str(raw.get("source_sha256") or ""),
        vocab={str(key): int(value) for key, value in dict(raw.get("vocab") or {}).items()},
        label_keys=tuple(str(value) for value in raw.get("label_keys", [])),
        intentions=tuple(intention_from_dict(value) for value in raw.get("intentions", [])),
        embed_dim=int(raw.get("embed_dim") or INTENT_NEURAL_EMBED_DIM),
        hidden_dim=int(raw.get("hidden_dim") or INTENT_NEURAL_HIDDEN_DIM),
        dropout=float(raw.get("dropout") or INTENT_NEURAL_DROPOUT),
        min_confidence=float(raw.get("min_confidence") or INTENT_NEURAL_MIN_CONFIDENCE),
        example_count=int(raw.get("example_count") or 0),
    )


def save_intent_neural_model(
    model: NeuralIntentClassifier,
    state: IntentNeuralModelState,
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
                "label_keys": list(state.label_keys),
                "intentions": [intention_to_dict(intention) for intention in state.intentions],
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


def representative_intentions_by_label(intentions: tuple[Intention, ...]) -> dict[str, Intention]:
    grouped: dict[str, Intention] = {}
    for intention in intentions:
        grouped.setdefault(intent_label_key(intention), intention)
    return grouped


def build_intent_input(text: str) -> str:
    return normalize_observation(text)


def intent_label_key(intention: Intention) -> str:
    return json.dumps(intent_label_payload(intention), ensure_ascii=False, sort_keys=True)


def intention_to_dict(intention: Intention) -> dict[str, Any]:
    return {
        "subject": intention.subject,
        "goal": intention.goal,
        "belief": intention.belief,
        "strategy": intention.strategy,
        "evidence": intention.evidence,
        "confidence": intention.confidence,
        "source": intention.source,
    }


def intent_label_payload(intention: Intention) -> dict[str, Any]:
    payload = {
        "subject": intention.subject,
        "goal": intention.goal,
    }
    if intention.belief:
        payload["belief"] = intention.belief
    if intention.strategy:
        payload["strategy"] = intention.strategy
    return payload


def intention_from_dict(record: Any) -> Intention:
    if not isinstance(record, dict):
        raise ValueError("Neural intent entries must be objects.")
    subject = str(record.get("subject") or "").strip()
    goal = str(record.get("goal") or "").strip()
    if not subject or not goal:
        raise ValueError("Neural intent entries require subject and goal.")
    return Intention(
        subject=subject,
        goal=goal,
        belief=str(record.get("belief") or "").strip(),
        strategy=str(record.get("strategy") or "").strip(),
        evidence=str(record.get("evidence") or "").strip(),
        confidence=float(record.get("confidence", 1.0)),
        source=str(record.get("source") or "neural").strip() or "neural",
    )
