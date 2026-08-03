from __future__ import annotations

from dataclasses import dataclass


PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"


@dataclass
class CharVocab:
    token_to_id: dict[str, int]

    @classmethod
    def build(cls, texts: list[str]) -> "CharVocab":
        tokens = [PAD, BOS, EOS, UNK]
        seen = set(tokens)
        for text in texts:
            for char in text:
                if char not in seen:
                    seen.add(char)
                    tokens.append(char)
        return cls({token: index for index, token in enumerate(tokens)})

    @property
    def id_to_token(self) -> dict[int, str]:
        return {index: token for token, index in self.token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS]

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = [self.token_to_id.get(char, self.token_to_id[UNK]) for char in text]
        if add_special:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        id_to_token = self.id_to_token
        chars = []
        for index in ids:
            token = id_to_token.get(index, UNK)
            if token in {PAD, BOS, EOS}:
                continue
            chars.append(token)
        return "".join(chars)
