from __future__ import annotations

import random
from typing import Iterable

import torch
from torch import Tensor

from ..dataset_io import file_sha256

NEURAL_PAD = "<pad>"
NEURAL_UNK = "<unk>"


def build_character_vocabulary(
    texts: Iterable[str],
    *,
    pad_token: str = NEURAL_PAD,
    unknown_token: str = NEURAL_UNK,
    sort_characters: bool = False,
) -> dict[str, int]:
    vocab = {pad_token: 0, unknown_token: 1}
    characters = (char for text in texts for char in text)
    if sort_characters:
        characters = iter(sorted(set(characters)))
    for character in characters:
        if character not in vocab:
            vocab[character] = len(vocab)
    return vocab


def encode_characters(text: str, vocab: dict[str, int], *, unknown_token: str = NEURAL_UNK) -> list[int]:
    if not text:
        return [vocab[unknown_token]]
    return [vocab.get(character, vocab[unknown_token]) for character in text]


def collate_classification_batch(batch: list[tuple[list[int], int]]) -> tuple[Tensor, Tensor, Tensor]:
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    max_length = int(lengths.max().item()) if lengths.numel() else 0
    token_ids = torch.zeros((len(sequences), max_length), dtype=torch.long)
    for row_index, sequence in enumerate(sequences):
        if sequence:
            token_ids[row_index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return token_ids, lengths, torch.tensor(labels, dtype=torch.long)


def sequence_mask(lengths: Tensor, width: int, device: torch.device) -> Tensor:
    positions = torch.arange(width, device=device).unsqueeze(0)
    return positions < lengths.to(device).unsqueeze(1)


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.unsqueeze(-1).to(values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def masked_max(values: Tensor, mask: Tensor) -> Tensor:
    masked = values.masked_fill(~mask.unsqueeze(-1), torch.finfo(values.dtype).min)
    return masked.max(dim=1).values


def balanced_class_weights(class_counts: Tensor, *, minimum: float | None = None, maximum: float | None = None) -> Tensor:
    nonzero = class_counts > 0
    weights = torch.ones_like(class_counts)
    if nonzero.any():
        weights[nonzero] = class_counts[nonzero].sum() / (class_counts[nonzero] * nonzero.sum())
    if minimum is not None or maximum is not None:
        min_value = minimum if minimum is not None else float("-inf")
        max_value = maximum if maximum is not None else float("inf")
        weights = weights.clamp(min=min_value, max=max_value)
    return weights


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

