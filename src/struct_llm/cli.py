from __future__ import annotations

import argparse
from pathlib import Path

from .cognitive.intent_dataset import append_intent_record, build_intent_record
from .cognitive.intent_learning import InMemoryIntentAnalyzer, evaluate_intent_analyzer, from_jsonl
from .cognitive.query_learning import LearnedQueryParser, evaluate_query_parser, load_query_jsonl
from .cognitive.query_learning import compile_query_model_from_jsonl, save_query_model
from .cognitive.statement_learning import (
    LearnedStatementParser,
    compile_statement_model_from_jsonl,
    evaluate_statement_parser,
    load_statement_jsonl,
    save_statement_model,
)
from .dataset import generate_examples, write_jsonl
from .reasoner import default_capabilities, predict


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
    parser.add_argument("--intent-data", help="JSONL file with learned intent examples.")
    parser.add_argument("--intent-min-score", type=float, default=0.6)
    parser.add_argument("--query-data", help="JSONL file with learned query examples.")
    parser.add_argument("--query-model", help="Compiled query model artifact.")
    parser.add_argument("--query-min-score", type=float, default=0.72)
    parser.add_argument("--statement-data", help="JSONL file with learned statement examples.")
    parser.add_argument("--statement-model", help="Compiled statement model artifact.")
    parser.add_argument("--statement-min-score", type=float, default=0.58)
    args = parser.parse_args()
    capabilities = default_capabilities()
    if args.statement_model:
        capabilities = capabilities.replace_statement_parsers(
            LearnedStatementParser.from_model(Path(args.statement_model), min_score=args.statement_min_score)
        )
    elif args.statement_data:
        capabilities = capabilities.replace_statement_parsers(
            LearnedStatementParser.from_jsonl(Path(args.statement_data), min_score=args.statement_min_score)
        )
    if args.query_model:
        capabilities = capabilities.replace_query_parsers(
            LearnedQueryParser.from_model(Path(args.query_model), min_score=args.query_min_score)
        )
    elif args.query_data:
        capabilities = capabilities.replace_query_parsers(
            LearnedQueryParser.from_jsonl(Path(args.query_data), min_score=args.query_min_score)
        )
    if args.intent_data:
        analyzer = InMemoryIntentAnalyzer.from_jsonl(Path(args.intent_data), min_score=args.intent_min_score)
        capabilities = capabilities.with_intent_analyzers(analyzer)

    if args.text:
        print_prediction(args.text, capabilities)
        return

    print("输入一句话，按回车推理；输入 exit 退出。")
    while True:
        text = input("> ").strip()
        if text.lower() in {"exit", "quit", "q"}:
            break
        if text:
            print_prediction(text, capabilities)


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


def add_intent_example() -> None:
    parser = argparse.ArgumentParser(description="Append one intent-training example to a JSONL dataset.")
    parser.add_argument("observation", help="Observed behavior or utterance.")
    parser.add_argument("--subject", required=True, help="Who holds the inferred intention.")
    parser.add_argument("--goal", required=True, help="Goal inferred from the observation.")
    parser.add_argument("--belief", default="", help="Belief state inferred from the observation.")
    parser.add_argument("--strategy", default="", help="Likely strategy or action plan.")
    parser.add_argument("--evidence", default="", help="Evidence span; defaults to the observation.")
    parser.add_argument("--confidence", type=float, default=1.0)
    parser.add_argument("--context", action="append", default=(), help="Optional context line; repeatable.")
    parser.add_argument("--world-state", action="append", default=(), help="Optional world-state label; repeatable.")
    parser.add_argument("--belief-state", action="append", default=(), help="Optional belief-state label; repeatable.")
    parser.add_argument("--answer", default="", help="Optional expected answer for supervised evaluation.")
    parser.add_argument("--source", default="human_feedback")
    parser.add_argument("--split", default="train")
    parser.add_argument("--path", default="data/intent_examples.jsonl", help="Target JSONL dataset path.")
    args = parser.parse_args()

    record = build_intent_record(
        args.observation,
        args.subject,
        args.goal,
        belief=args.belief,
        strategy=args.strategy,
        evidence=args.evidence,
        confidence=args.confidence,
        context=args.context,
        world_state=args.world_state,
        belief_state=args.belief_state,
        answer=args.answer,
        source=args.source,
        split=args.split,
    )
    append_intent_record(Path(args.path), record)
    print(f"Appended intent example to {args.path}: {record.observation}")


def eval_intent_examples() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an intent analyzer against JSONL examples.")
    parser.add_argument("--train-data", default="data/intent_examples.jsonl")
    parser.add_argument("--eval-data", default="")
    parser.add_argument("--intent-min-score", type=float, default=0.6)
    args = parser.parse_args()

    analyzer = InMemoryIntentAnalyzer.from_jsonl(Path(args.train_data), min_score=args.intent_min_score)
    examples = from_jsonl(Path(args.eval_data or args.train_data))
    result = evaluate_intent_analyzer(analyzer, examples)
    print(f"intent_examples={result.total} matched={result.matched} accuracy={result.accuracy:.2f}")


def eval_query_examples() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the learned query parser against JSONL examples.")
    parser.add_argument("--query-data", default="data/query_examples.jsonl")
    parser.add_argument("--query-model", default="")
    parser.add_argument("--query-min-score", type=float, default=0.72)
    args = parser.parse_args()

    examples = load_query_jsonl(Path(args.query_data))
    query_parser = (
        LearnedQueryParser.from_model(Path(args.query_model), min_score=args.query_min_score)
        if args.query_model
        else LearnedQueryParser.from_examples(examples, min_score=args.query_min_score)
    )
    result = evaluate_query_parser(query_parser, examples)
    print(f"query_examples={result.total} matched={result.matched} accuracy={result.accuracy:.2f}")


def eval_statement_examples() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the learned statement parser against JSONL examples.")
    parser.add_argument("--statement-data", default="data/statement_examples.jsonl")
    parser.add_argument("--statement-model", default="")
    parser.add_argument("--statement-min-score", type=float, default=0.58)
    args = parser.parse_args()

    examples = load_statement_jsonl(Path(args.statement_data))
    statement_parser = (
        LearnedStatementParser.from_model(Path(args.statement_model), min_score=args.statement_min_score)
        if args.statement_model
        else LearnedStatementParser.from_examples(examples, min_score=args.statement_min_score)
    )
    result = evaluate_statement_parser(statement_parser, examples)
    print(f"statement_examples={result.total} matched={result.matched} accuracy={result.accuracy:.2f}")


def compile_query_model() -> None:
    parser = argparse.ArgumentParser(description="Compile Query JSONL examples into a runtime model artifact.")
    parser.add_argument("--query-data", default="data/query_examples.jsonl")
    parser.add_argument("--output", default="data/query_model.json")
    args = parser.parse_args()

    model = compile_query_model_from_jsonl(Path(args.query_data))
    save_query_model(model, Path(args.output))
    print(
        f"compiled_query_model examples={model.example_count} patterns={len(model.patterns)} output={args.output}"
    )


def compile_statement_model() -> None:
    parser = argparse.ArgumentParser(description="Compile statement JSONL examples into a runtime model artifact.")
    parser.add_argument("--statement-data", default="data/statement_examples.jsonl")
    parser.add_argument("--output", default="data/statement_model.json")
    args = parser.parse_args()

    model = compile_statement_model_from_jsonl(Path(args.statement_data))
    save_statement_model(model, Path(args.output))
    print(
        f"compiled_statement_model examples={model.example_count} patterns={len(model.patterns)} output={args.output}"
    )


def print_prediction(question: str, capabilities=None) -> None:
    prediction = predict(question, capabilities)
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
