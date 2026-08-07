from __future__ import annotations

import argparse
from pathlib import Path

from ..dataset_io import load_jsonl_objects
from ..kernel import default_capabilities, parse_text
from ..memory.knowledge import (
    compile_memory_knowledge_model_from_jsonl,
    save_memory_knowledge_model,
)
from ..motor.feedback import (
    save_direct_memory_feedback,
    save_direct_memory_structure_feedback,
    save_memory_knowledge_feedback,
)
from ..structure import State
from .common import apply_neural_provider_args, learning_paths_from_args

def compile_memory_knowledge_model() -> None:
    parser = argparse.ArgumentParser(description="Compile verified long-term knowledge entries into a runtime artifact.")
    parser.add_argument("--memory-knowledge-data", default="data/memory_knowledge_examples.jsonl")
    parser.add_argument("--output", default="data/memory_knowledge_model.json")
    args = parser.parse_args()

    model = compile_memory_knowledge_model_from_jsonl(Path(args.memory_knowledge_data))
    save_memory_knowledge_model(model, Path(args.output))
    print(f"已生成长期知识模型：样本={model.example_count} 模式={len(model.patterns)} 输出={args.output}")


def add_memory_entry() -> None:
    parser = argparse.ArgumentParser(description="Append one memory entry to the long-term memory store.")
    parser.add_argument("text", nargs="?", help="Plain text to parse into memory states.")
    parser.add_argument("--state", nargs=3, metavar=("NAME", "LEFT", "RIGHT"), help="Direct state insert.")
    parser.add_argument("--channel", default="direct", choices=("direct", "chat"))
    parser.add_argument("--source", default="human_feedback")
    parser.add_argument("--confidence", type=float, default=1.0)
    parser.add_argument("--memory-direct-data", default="data/memory_direct_examples.jsonl")
    parser.add_argument("--memory-chat-data", default="data/memory_chat_examples.jsonl")
    parser.add_argument("--memory-model", default="data/memory_model.json")
    parser.add_argument("--neural-provider")
    args = parser.parse_args()

    paths = learning_paths_from_args(args)
    if args.state:
        name, left, right = args.state
        result = save_direct_memory_feedback(
            State(name, left, right),
            paths,
            text=args.text or "",
            source=args.source,
            channel=args.channel,
            confidence=args.confidence,
        )
        print(f"已写入记忆状态，当前记忆状态 {result.state_count} 条。")
        return

    if not args.text:
        raise SystemExit("需要提供 TEXT 或 --state。")
    capabilities = default_capabilities(use_environment=False, use_memory=False)
    capabilities = apply_neural_provider_args(capabilities, args)
    structure = parse_text(args.text, capabilities)
    result = save_direct_memory_structure_feedback(args.text, structure, paths, confidence=args.confidence)
    print(f"已写入记忆样本 {result.entry_count} 条，当前记忆状态 {result.state_count} 条。")


def add_memory_knowledge_entry() -> None:
    parser = argparse.ArgumentParser(description="Append verified long-term knowledge entries.")
    parser.add_argument("text", nargs="?", help="Question text. Omit it when using --file.")
    parser.add_argument("--answer", required=False, default="", help="Verified answer for the question.")
    parser.add_argument("--file", help="JSONL source file with question/text and answer/response fields.")
    parser.add_argument("--source", default="human_feedback")
    parser.add_argument("--memory-knowledge-data", default="data/memory_knowledge_examples.jsonl")
    parser.add_argument("--memory-knowledge-model", default="data/memory_knowledge_model.json")
    parser.add_argument("--neural-provider")
    args = parser.parse_args()

    if args.file and args.text:
        raise SystemExit("TEXT 和 --file 只能选一个。")
    if not args.file and not args.text:
        raise SystemExit("需要提供 TEXT 或 --file。")
    if args.text and not args.answer.strip():
        raise SystemExit("单条写入需要提供 --answer。")
    paths = learning_paths_from_args(args)
    capabilities = default_capabilities(use_environment=False, use_memory=False)
    capabilities = apply_neural_provider_args(capabilities, args)
    records = (
        load_memory_knowledge_source_records(Path(args.file), default_source=args.source)
        if args.file
        else ((args.text, args.answer, args.source),)
    )
    written = 0
    result = None
    for question, answer, source in records:
        structure = parse_text(question, capabilities)
        if structure.query is None:
            raise SystemExit(f"当前解析结果没有生成可入库的 Query：{question}")
        result = save_memory_knowledge_feedback(
            question,
            structure.query,
            answer,
            paths,
            source=source,
        )
        written += 1
    if result is None:
        print("没有可写入的长期知识样本。")
        return
    print(f"已写入长期知识样本 {written} 条，当前知识样本 {result.example_count} 条，知识模式 {result.pattern_count} 条。")


def load_memory_knowledge_source_records(
    path: Path,
    *,
    default_source: str,
) -> tuple[tuple[str, str, str], ...]:
    records: list[tuple[str, str, str]] = []
    for line_number, raw_record in enumerate(load_jsonl_objects(path, "knowledge source"), start=1):
        question = str(raw_record.get("question") or raw_record.get("text") or "").strip()
        answer = str(raw_record.get("answer") or raw_record.get("response") or "").strip()
        source = str(raw_record.get("source") or default_source).strip() or default_source
        if not question or not answer:
            raise ValueError(f"Knowledge source at line {line_number} requires question/text and answer/response.")
        records.append((question, answer, source))
    return tuple(records)


def compile_memory_model() -> None:
    parser = argparse.ArgumentParser(description="Compile memory JSONL entries into a runtime model artifact.")
    parser.add_argument("--memory-direct-data", default="data/memory_direct_examples.jsonl")
    parser.add_argument("--memory-chat-data", default="data/memory_chat_examples.jsonl")
    parser.add_argument("--output", default="data/memory_model.json")
    args = parser.parse_args()

    from ..memory.long_term import compile_memory_model_from_jsonl, save_memory_model

    model = compile_memory_model_from_jsonl(Path(args.memory_direct_data), Path(args.memory_chat_data))
    save_memory_model(model, Path(args.output))
    print(f"已生成记忆模型：样本={model.example_count} 状态={len(model.states)} 输出={args.output}")
