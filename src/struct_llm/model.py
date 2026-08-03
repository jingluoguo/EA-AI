from __future__ import annotations


def require_torch():
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "PyTorch is required for the neural model. Install it with: "
            "uv sync --extra neural"
        ) from error
    return torch, nn


def build_tiny_transformer(vocab_size: int, d_model: int = 128):
    torch, nn = require_torch()

    class TinyTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.position = nn.Embedding(512, d_model)
            self.transformer = nn.Transformer(
                d_model=d_model,
                nhead=4,
                num_encoder_layers=2,
                num_decoder_layers=2,
                dim_feedforward=256,
                batch_first=True,
            )
            self.output = nn.Linear(d_model, vocab_size)

        def forward(self, source, target):
            source = self._embed(source)
            target = self._embed(target)
            target_mask = nn.Transformer.generate_square_subsequent_mask(target.size(1))
            target_mask = target_mask.to(target.device)
            decoded = self.transformer(source, target, tgt_mask=target_mask)
            return self.output(decoded)

        def _embed(self, token_ids):
            positions = torch.arange(token_ids.size(1), device=token_ids.device)
            positions = positions.unsqueeze(0).expand_as(token_ids)
            return self.embedding(token_ids) + self.position(positions)

    return TinyTransformer()


def generate_text(model, vocab, text: str, max_new_tokens: int = 256) -> str:
    torch, _ = require_torch()

    model.eval()
    source = torch.tensor(vocab.encode(text), dtype=torch.long).unsqueeze(0)
    target_ids = [vocab.bos_id]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            target = torch.tensor(target_ids, dtype=torch.long).unsqueeze(0)
            logits = model(source, target)
            next_id = int(logits[0, -1].argmax().item())
            if next_id == vocab.eos_id:
                break
            target_ids.append(next_id)

    return vocab.decode(target_ids[1:])
