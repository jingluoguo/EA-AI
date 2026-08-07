from __future__ import annotations

import argparse
from pathlib import Path

from ..comprehension.intent import InMemoryIntentAnalyzer
from ..kernel import default_capabilities
from ..motor.dialogue import LearnedDialogActAnswerer
from .common import (
    apply_memory_args,
    apply_memory_knowledge_args,
    apply_neural_provider_args,
    print_prediction,
    print_prediction_with_learning,
)


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
    parser.add_argument("--dialog-answer-data", help="JSONL file with learned dialog answer examples.")
    parser.add_argument("--dialog-answer-model", help="Verified dialog answer artifact.")
    parser.add_argument("--memory-direct-data", help="JSONL file with direct memory entries.")
    parser.add_argument("--memory-chat-data", help="JSONL file with chat-sedimented memory entries.")
    parser.add_argument("--memory-model", help="Long-term memory state artifact.")
    parser.add_argument("--memory-knowledge-data", help="JSONL file with long-term knowledge entries.")
    parser.add_argument("--memory-knowledge-model", help="Verified long-term knowledge artifact.")
    parser.add_argument("--remember-chat", action="store_true", help="Ask before storing stable facts from successful chat turns.")
    parser.add_argument(
        "--neural-provider",
        help="Python factory in module:function form; defaults to EA_AI_NEURAL_PROVIDER.",
    )
    parser.add_argument(
        "--neural-answer-priority",
        default="after-verified",
        help="Use neural answers only after verified answerers, or before them.",
    )
    parser.add_argument("--unrecognized-data", help="JSONL file for low-confidence inputs awaiting offline labeling.")
    parser.add_argument("--learn-on-fail", action="store_true", help="Prompt for feedback when structure extraction fails.")
    args = parser.parse_args()
    capabilities = default_capabilities(
        neural_answer_priority=args.neural_answer_priority,
        use_environment=False,
        use_memory=False,
    )
    if args.intent_data:
        analyzer = InMemoryIntentAnalyzer.from_jsonl(Path(args.intent_data), min_score=args.intent_min_score)
        capabilities = capabilities.with_intent_analyzers(analyzer)
    if args.dialog_answer_model and Path(args.dialog_answer_model).exists():
        capabilities = capabilities.with_answerers(
            LearnedDialogActAnswerer.from_model(Path(args.dialog_answer_model))
        )
    elif args.dialog_answer_data and Path(args.dialog_answer_data).exists():
        capabilities = capabilities.with_answerers(
            LearnedDialogActAnswerer.from_jsonl(Path(args.dialog_answer_data))
        )
    capabilities = apply_memory_args(capabilities, args)
    capabilities = apply_memory_knowledge_args(capabilities, args)
    capabilities = apply_neural_provider_args(capabilities, args)

    if args.text:
        print_prediction_with_learning(args.text, capabilities, args)
        return

    print("输入一句话，按回车推理；输入 exit 退出。")
    while True:
        text = input("> ").strip()
        if text.lower() in {"exit", "quit", "q"}:
            break
        if text:
            print_prediction_with_learning(text, capabilities, args)
