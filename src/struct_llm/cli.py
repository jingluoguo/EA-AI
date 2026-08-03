from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import generate_examples, write_jsonl
from .reasoner import predict


QUESTIONS = (
    "小明把钥匙放进盒子。盒子被带到厨房。钥匙在哪里？",
    "研究员把芯片放进托盘。托盘被带到实验室。芯片在哪里？",
    "小红把药瓶交给医生。现在谁拥有药瓶？",
    "工程师把笔记本涂成绿色。现在笔记本是什么颜色？",
)


def run_symbolic_demo() -> None:
    for question in QUESTIONS:
        print_prediction(question)


def ask_symbolic() -> None:
    parser = argparse.ArgumentParser(description="Ask the explicit structural reasoner.")
    parser.add_argument("text", nargs="?", help="Question text. Omit it to enter interactive mode.")
    args = parser.parse_args()

    if args.text:
        print_prediction(args.text)
        return

    print("输入一句话，按回车推理；输入 exit 退出。")
    while True:
        text = input("> ").strip()
        if text.lower() in {"exit", "quit", "q"}:
            break
        if text:
            print_prediction(text)


def ask_neural() -> None:
    parser = argparse.ArgumentParser(description="Ask the trained tiny Transformer.")
    parser.add_argument("text", nargs="?", help="Question text. Omit it to enter interactive mode.")
    parser.add_argument("--checkpoint", default="data/tiny_model.pt")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    predictor = load_neural_predictor(Path(args.checkpoint), args.max_new_tokens)

    if args.text:
        print_neural_prediction(predictor(args.text))
        return

    print("输入一句话，按回车让 tiny 模型生成；输入 exit 退出。")
    while True:
        text = input("> ").strip()
        if text.lower() in {"exit", "quit", "q"}:
            break
        if text:
            print_neural_prediction(predictor(text))


def print_prediction(question: str) -> None:
    prediction = predict(question)
    print("=" * 60)
    print(question)
    print()
    print(prediction.structure.linearize())
    print()
    print(prediction.answer)


def print_neural_prediction(output: str) -> None:
    answer = extract_answer(output)
    if answer:
        print(answer)
        print()
        print("[raw]")
    print(output)


def extract_answer(output: str) -> str:
    if "<ANSWER>" not in output:
        return ""
    answer = output.rsplit("<ANSWER>", 1)[-1].strip()
    return answer.splitlines()[0].strip() if answer else ""


def make_dataset() -> None:
    examples = generate_examples()
    write_jsonl((example for example in examples if example.split == "train"), Path("data/train.jsonl"))
    write_jsonl((example for example in examples if example.split == "test"), Path("data/test.jsonl"))

    train_count = sum(1 for example in examples if example.split == "train")
    test_count = sum(1 for example in examples if example.split == "test")
    print(f"Wrote {train_count} train examples and {test_count} test examples.")


def train_tiny_model() -> None:
    from .model import build_tiny_transformer, require_torch
    from .structure import linearize_target
    from .vocab import CharVocab

    torch, nn = require_torch()

    examples = [example for example in generate_examples() if example.split == "train"]
    sources = [example.text for example in examples]
    targets = [linearize_target(example.structure, example.answer) for example in examples]
    vocab = CharVocab.build(sources + targets)
    model = build_tiny_transformer(len(vocab.token_to_id))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    encoded_sources = [torch.tensor(vocab.encode(text), dtype=torch.long) for text in sources]
    encoded_targets = [torch.tensor(vocab.encode(text), dtype=torch.long) for text in targets]

    for epoch in range(10):
        total_loss = 0.0
        for source, target in zip(encoded_sources, encoded_targets):
            source = source.unsqueeze(0)
            decoder_input = target[:-1].unsqueeze(0)
            expected = target[1:].unsqueeze(0)

            logits = model(source, decoder_input)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), expected.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        print(f"epoch={epoch + 1} loss={total_loss / len(examples):.4f}")

    Path("data").mkdir(exist_ok=True)
    torch.save(
        {
            "config": {"d_model": 128},
            "model": model.state_dict(),
            "vocab": vocab.token_to_id,
        },
        "data/tiny_model.pt",
    )
    print("Saved data/tiny_model.pt")


def load_neural_predictor(checkpoint_path: Path, max_new_tokens: int):
    from .model import build_tiny_transformer, generate_text, require_torch
    from .vocab import CharVocab

    torch, _ = require_torch()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Cannot find {checkpoint_path}. Run `make train` first to create the tiny model."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    vocab = CharVocab(checkpoint["vocab"])
    config = checkpoint.get("config", {})
    model = build_tiny_transformer(
        vocab_size=len(vocab.token_to_id),
        d_model=int(config.get("d_model", 128)),
    )
    model.load_state_dict(checkpoint["model"])

    def _predict(text: str) -> str:
        return generate_text(model, vocab, text, max_new_tokens=max_new_tokens)

    return _predict
