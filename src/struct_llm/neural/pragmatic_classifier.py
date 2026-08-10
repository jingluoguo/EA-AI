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
from ..comprehension.episode import (
    EPISODE_DATA_PATH,
    ambiguous_reference_act,
    evaluate_pragmatic_analyzer,
    load_episode_jsonl,
    normalize_episode_text,
    pragmatic_act_from_dict,
    pragmatic_act_matches,
    pragmatic_act_to_dict,
)
from ..structure import PragmaticAct, Structure


PRAGMATIC_NEURAL_SCHEMA = "struct_llm.pragmatic_neural_model.v1"
PRAGMATIC_NEURAL_WEIGHTS_PATH = Path(__file__).resolve().parents[3] / "data" / "pragmatic_neural_model.pt"
PRAGMATIC_NEURAL_META_PATH = Path(__file__).resolve().parents[3] / "data" / "pragmatic_neural_model.json"
PRAGMATIC_NEURAL_MIN_CONFIDENCE = 0.42
PRAGMATIC_NEURAL_EMBED_DIM = 96
PRAGMATIC_NEURAL_HIDDEN_DIM = 96
PRAGMATIC_NEURAL_DROPOUT = 0.10
PRAGMATIC_NEURAL_BATCH_SIZE = 16
PRAGMATIC_NEURAL_EPOCHS = 140
PRAGMATIC_NEURAL_LR = 2e-3
PRAGMATIC_NEURAL_SEED = 20260808


@dataclass(frozen=True)
class NeuralPragmaticTrainingExample:
    text: str
    label_key: str


@dataclass(frozen=True)
class PragmaticNeuralModelState:
    schema: str
    source_sha256: str
    vocab: dict[str, int]
    label_keys: tuple[str, ...]
    acts_by_label: tuple[tuple[PragmaticAct, ...], ...]
    embed_dim: int
    hidden_dim: int
    dropout: float
    min_confidence: float
    example_count: int


@dataclass(frozen=True)
class PragmaticNeuralTrainingResult:
    example_count: int
    label_count: int
    train_accuracy: float
    train_loss: float
    source_sha256: str


@dataclass(frozen=True)
class PragmaticNeuralTrainingBundle:
    analyzer: LoadedNeuralPragmaticAnalyzer
    result: PragmaticNeuralTrainingResult


class NeuralPragmaticDataset(Dataset):
    def __init__(
        self,
        examples: tuple[NeuralPragmaticTrainingExample, ...],
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


class NeuralPragmaticClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        label_count: int,
        *,
        embed_dim: int = PRAGMATIC_NEURAL_EMBED_DIM,
        hidden_dim: int = PRAGMATIC_NEURAL_HIDDEN_DIM,
        dropout: float = PRAGMATIC_NEURAL_DROPOUT,
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
class LoadedNeuralPragmaticAnalyzer:
    model: NeuralPragmaticClassifier
    vocab: dict[str, int]
    label_keys: tuple[str, ...]
    acts_by_label: tuple[tuple[PragmaticAct, ...], ...]
    min_confidence: float = PRAGMATIC_NEURAL_MIN_CONFIDENCE

    def __call__(self, text: str, structure: Structure) -> tuple[PragmaticAct, ...]:
        acts: list[PragmaticAct] = []
        seen: set[tuple[str, str, tuple[str, ...]]] = set()
        structural_targets: set[tuple[str, str]] = set()
        structural_reference = ambiguous_reference_act(structure)
        if structural_reference is not None:
            signature = pragmatic_runtime_signature(structural_reference)
            seen.add(signature)
            structural_targets.add((structural_reference.act, structural_reference.target))
            acts.append(structural_reference)
        normalized = build_pragmatic_input(text)
        if not normalized:
            return tuple(acts)
        self.model.eval()
        with torch.no_grad():
            token_ids = torch.tensor([encode_characters(normalized, self.vocab)], dtype=torch.long)
            lengths = torch.tensor([token_ids.shape[1]], dtype=torch.long)
            logits = self.model(token_ids, lengths)
            probabilities = torch.softmax(logits, dim=-1)[0]
            confidence, label_index = torch.max(probabilities, dim=-1)
        score = float(confidence.item())
        if score < self.min_confidence:
            return tuple(acts)
        for act in self.acts_by_label[int(label_index.item())]:
            lifted = PragmaticAct(
                act=act.act,
                target=act.target,
                qualifiers=act.qualifiers,
                confidence=min(1.0, max(act.confidence, score)),
                source=act.source,
            )
            signature = pragmatic_runtime_signature(lifted)
            if signature in seen or (lifted.act, lifted.target) in structural_targets:
                continue
            seen.add(signature)
            acts.append(lifted)
        return tuple(acts[:3])


def default_neural_pragmatic_analyzer(
    episode_data_path: str | Path = EPISODE_DATA_PATH,
    weights_path: str | Path = PRAGMATIC_NEURAL_WEIGHTS_PATH,
    meta_path: str | Path = PRAGMATIC_NEURAL_META_PATH,
) -> LoadedNeuralPragmaticAnalyzer:
    data_path = Path(episode_data_path)
    weights = Path(weights_path)
    metadata = Path(meta_path)
    source_sha = file_sha256(data_path)
    return _default_neural_pragmatic_analyzer_from_signature(str(data_path), str(weights), str(metadata), source_sha)


@lru_cache(maxsize=8)
def _default_neural_pragmatic_analyzer_from_signature(
    data_path: str,
    weights_path: str,
    meta_path: str,
    source_sha: str,
) -> LoadedNeuralPragmaticAnalyzer:
    weights = Path(weights_path)
    metadata = Path(meta_path)
    if weights.exists() and metadata.exists():
        state = load_pragmatic_neural_metadata(metadata)
        if state.source_sha256 == source_sha:
            return load_pragmatic_neural_analyzer(weights, metadata)
    return train_pragmatic_neural_model(data_path, weights, metadata).analyzer


def train_pragmatic_neural_model(
    episode_data_path: str | Path = EPISODE_DATA_PATH,
    weights_path: str | Path = PRAGMATIC_NEURAL_WEIGHTS_PATH,
    meta_path: str | Path = PRAGMATIC_NEURAL_META_PATH,
) -> PragmaticNeuralTrainingBundle:
    data_path = Path(episode_data_path)
    examples = load_episode_jsonl(data_path)
    source_sha = file_sha256(data_path)
    labels = representative_acts_by_label(tuple(example.expected_pragmatic_acts for example in examples))
    label_keys = tuple(sorted(labels))
    acts_by_label = tuple(labels[key] for key in label_keys)
    neural_examples = tuple(
        NeuralPragmaticTrainingExample(
            build_pragmatic_input(example.text),
            pragmatic_label_key(example.expected_pragmatic_acts),
        )
        for example in examples
    )
    vocab = build_character_vocabulary((example.text for example in neural_examples), sort_characters=True)
    label_index = {label: index for index, label in enumerate(label_keys)}
    dataset = NeuralPragmaticDataset(neural_examples, vocab, label_index)
    set_seed(PRAGMATIC_NEURAL_SEED)
    model = NeuralPragmaticClassifier(len(vocab), len(label_keys))
    result = fit_pragmatic_classifier(model, dataset, source_sha256=source_sha)
    state = PragmaticNeuralModelState(
        schema=PRAGMATIC_NEURAL_SCHEMA,
        source_sha256=source_sha,
        vocab=vocab,
        label_keys=label_keys,
        acts_by_label=acts_by_label,
        embed_dim=PRAGMATIC_NEURAL_EMBED_DIM,
        hidden_dim=PRAGMATIC_NEURAL_HIDDEN_DIM,
        dropout=PRAGMATIC_NEURAL_DROPOUT,
        min_confidence=PRAGMATIC_NEURAL_MIN_CONFIDENCE,
        example_count=len(examples),
    )
    save_pragmatic_neural_model(model, state, weights_path, meta_path)
    return PragmaticNeuralTrainingBundle(
        analyzer=LoadedNeuralPragmaticAnalyzer(model, vocab, label_keys, acts_by_label),
        result=result,
    )


def fit_pragmatic_classifier(
    model: NeuralPragmaticClassifier,
    dataset: NeuralPragmaticDataset,
    *,
    source_sha256: str = "",
) -> PragmaticNeuralTrainingResult:
    if len(dataset) == 0:
        raise ValueError("Pragmatic training dataset is empty.")
    model.train()
    label_count = model.head[-1].out_features
    class_counts = torch.zeros(label_count, dtype=torch.float32)
    for _, label in dataset:
        class_counts[label] += 1
    loss_fn = nn.CrossEntropyLoss(weight=balanced_class_weights(class_counts, minimum=0.25, maximum=4.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=PRAGMATIC_NEURAL_LR)
    loader = DataLoader(dataset, batch_size=PRAGMATIC_NEURAL_BATCH_SIZE, shuffle=True, collate_fn=collate_classification_batch)
    total_loss = 0.0
    for _ in range(PRAGMATIC_NEURAL_EPOCHS):
        for token_ids, lengths, labels in loader:
            optimizer.zero_grad()
            logits = model(token_ids, lengths)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(labels.shape[0])
    accuracy = evaluate_pragmatic_classifier(model, dataset)
    return PragmaticNeuralTrainingResult(
        example_count=len(dataset),
        label_count=label_count,
        train_accuracy=accuracy,
        train_loss=total_loss / len(dataset) / PRAGMATIC_NEURAL_EPOCHS,
        source_sha256=source_sha256,
    )


def evaluate_pragmatic_classifier(model: NeuralPragmaticClassifier, dataset: NeuralPragmaticDataset) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=PRAGMATIC_NEURAL_BATCH_SIZE, shuffle=False, collate_fn=collate_classification_batch)
    matched = 0
    total = 0
    with torch.no_grad():
        for token_ids, lengths, labels in loader:
            predictions = torch.argmax(model(token_ids, lengths), dim=-1)
            matched += int((predictions == labels).sum().item())
            total += int(labels.shape[0])
    return matched / total if total else 0.0


def pragmatic_neural_summary(
    episode_data_path: str | Path = EPISODE_DATA_PATH,
    weights_path: str | Path = PRAGMATIC_NEURAL_WEIGHTS_PATH,
    meta_path: str | Path = PRAGMATIC_NEURAL_META_PATH,
) -> dict[str, Any]:
    analyzer = default_neural_pragmatic_analyzer(episode_data_path, weights_path, meta_path)
    examples = load_episode_jsonl(episode_data_path)
    evaluation = evaluate_pragmatic_analyzer(analyzer, examples)
    state = load_pragmatic_neural_metadata(meta_path)
    return {
        "examples": len(examples),
        "labels": len(state.label_keys),
        "accuracy": round(evaluation.accuracy, 4),
        "classifier_accuracy": round(
            evaluate_pragmatic_classifier(
                analyzer.model,
                NeuralPragmaticDataset(
                    tuple(
                        NeuralPragmaticTrainingExample(
                            build_pragmatic_input(example.text),
                            pragmatic_label_key(example.expected_pragmatic_acts),
                        )
                        for example in examples
                    ),
                    state.vocab,
                    {label: index for index, label in enumerate(state.label_keys)},
                ),
            ),
            4,
        ),
        "min_confidence": state.min_confidence,
    }


def load_pragmatic_neural_analyzer(
    weights_path: str | Path = PRAGMATIC_NEURAL_WEIGHTS_PATH,
    meta_path: str | Path = PRAGMATIC_NEURAL_META_PATH,
) -> LoadedNeuralPragmaticAnalyzer:
    state = load_pragmatic_neural_metadata(meta_path)
    model = NeuralPragmaticClassifier(
        len(state.vocab),
        len(state.label_keys),
        embed_dim=state.embed_dim,
        hidden_dim=state.hidden_dim,
        dropout=state.dropout,
    )
    model.load_state_dict(torch.load(Path(weights_path), map_location="cpu"))
    model.eval()
    return LoadedNeuralPragmaticAnalyzer(
        model=model,
        vocab=state.vocab,
        label_keys=state.label_keys,
        acts_by_label=state.acts_by_label,
        min_confidence=state.min_confidence,
    )


def load_pragmatic_neural_metadata(path: str | Path) -> PragmaticNeuralModelState:
    with Path(path).open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict) or raw.get("schema") != PRAGMATIC_NEURAL_SCHEMA:
        raise ValueError("Unsupported neural pragmatic model schema.")
    raw_acts = raw.get("acts_by_label", [])
    if not isinstance(raw_acts, list):
        raise ValueError("Neural pragmatic acts_by_label must be a list.")
    return PragmaticNeuralModelState(
        schema=PRAGMATIC_NEURAL_SCHEMA,
        source_sha256=str(raw.get("source_sha256") or ""),
        vocab={str(key): int(value) for key, value in dict(raw.get("vocab") or {}).items()},
        label_keys=tuple(str(value) for value in raw.get("label_keys", [])),
        acts_by_label=tuple(
            tuple(pragmatic_act_from_dict(act, "Neural pragmatic label") for act in acts)
            for acts in raw_acts
            if isinstance(acts, list)
        ),
        embed_dim=int(raw.get("embed_dim") or PRAGMATIC_NEURAL_EMBED_DIM),
        hidden_dim=int(raw.get("hidden_dim") or PRAGMATIC_NEURAL_HIDDEN_DIM),
        dropout=float(raw.get("dropout") or PRAGMATIC_NEURAL_DROPOUT),
        min_confidence=float(raw.get("min_confidence") or PRAGMATIC_NEURAL_MIN_CONFIDENCE),
        example_count=int(raw.get("example_count") or 0),
    )


def save_pragmatic_neural_model(
    model: NeuralPragmaticClassifier,
    state: PragmaticNeuralModelState,
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
                "acts_by_label": [
                    [pragmatic_act_to_dict(act) for act in acts]
                    for acts in state.acts_by_label
                ],
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


def representative_acts_by_label(act_groups: tuple[tuple[PragmaticAct, ...], ...]) -> dict[str, tuple[PragmaticAct, ...]]:
    grouped: dict[str, tuple[PragmaticAct, ...]] = {}
    for acts in act_groups:
        grouped.setdefault(pragmatic_label_key(acts), acts)
    return grouped


def build_pragmatic_input(text: str) -> str:
    return normalize_episode_text(text)


def pragmatic_label_key(acts: tuple[PragmaticAct, ...]) -> str:
    return json.dumps([pragmatic_label_payload(act) for act in acts], ensure_ascii=False, sort_keys=True)


def pragmatic_label_payload(act: PragmaticAct) -> dict[str, Any]:
    return {
        "act": act.act,
        "target": act.target,
        "qualifiers": list(act.qualifiers),
    }


def pragmatic_runtime_signature(act: PragmaticAct) -> tuple[str, str, tuple[str, ...]]:
    return act.act, act.target, act.qualifiers
