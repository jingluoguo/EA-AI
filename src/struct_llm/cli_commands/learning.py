from __future__ import annotations

import argparse
from pathlib import Path

from ..comprehension.intent import evaluate_intent_analyzer, from_jsonl
from ..comprehension.intent_dataset import append_intent_record, build_intent_record
from ..motor.dialogue import compile_dialog_answer_model_from_jsonl, save_dialog_answer_model
from ..neural.intent_classifier import default_neural_intent_analyzer
from ..neural.query_classifier import train_query_neural_model
from ..neural.statement_classifier import train_statement_neural_model

def train_neural_models() -> None:
    from my_neural import train

    train()


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

    analyzer = default_neural_intent_analyzer(Path(args.train_data))
    examples = from_jsonl(Path(args.eval_data or args.train_data))
    result = evaluate_intent_analyzer(analyzer, examples)
    print(f"意图样本={result.total} 命中={result.matched} 准确率={result.accuracy:.2f}")


def eval_query_examples() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the neural query parser against JSONL examples.")
    parser.add_argument("--query-data", default="data/query_examples.jsonl")
    args = parser.parse_args()

    bundle = train_query_neural_model(Path(args.query_data))
    print(
        f"问题样本={bundle.result.example_count} 标签={bundle.result.label_count} "
        f"训练准确率={bundle.result.train_accuracy:.2f}"
    )


def eval_statement_examples() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the neural statement parser against JSONL examples.")
    parser.add_argument("--statement-data", default="data/statement_examples.jsonl")
    args = parser.parse_args()

    bundle = train_statement_neural_model(Path(args.statement_data))
    print(
        f"陈述样本={bundle.result.example_count} 标签={bundle.result.label_count} "
        f"训练准确率={bundle.result.train_accuracy:.2f}"
    )


def compile_dialog_answer_model() -> None:
    parser = argparse.ArgumentParser(description="Compile verified dialog answer examples into a runtime model artifact.")
    parser.add_argument("--dialog-answer-data", default="data/dialog_answer_examples.jsonl")
    parser.add_argument("--output", default="data/dialog_answer_model.json")
    args = parser.parse_args()

    model = compile_dialog_answer_model_from_jsonl(Path(args.dialog_answer_data))
    save_dialog_answer_model(model, Path(args.output))
    print(f"已生成回答模型：样本={model.example_count} 模式={len(model.patterns)} 输出={args.output}")
