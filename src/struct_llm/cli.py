from __future__ import annotations

import sys

from .cli_commands.ask import ask_symbolic, run_symbolic_demo
from .cli_commands.learning import (
    add_intent_example,
    compile_dialog_answer_model,
    eval_intent_examples,
    eval_query_examples,
    eval_statement_examples,
    train_neural_models,
)
from .cli_commands.memory import (
    add_memory_entry,
    add_memory_knowledge_entry,
    compile_memory_knowledge_model,
    compile_memory_model,
)


def main() -> None:
    # Keep packaging quiet: expose one `struct` executable and dispatch
    # subcommands here instead of growing many project.scripts entries.
    commands = {
        "demo": run_symbolic_demo,
        "ask": ask_symbolic,
        "add-intent": add_intent_example,
        "eval-intent": eval_intent_examples,
        "eval-query": eval_query_examples,
        "eval-statement": eval_statement_examples,
        "compile-dialog-answer": compile_dialog_answer_model,
        "compile-memory": compile_memory_model,
        "compile-memory-knowledge": compile_memory_knowledge_model,
        "add-memory": add_memory_entry,
        "add-knowledge": add_memory_knowledge_entry,
        "train": train_neural_models,
    }
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("用法: struct <command> [args...]")
        print("命令: " + ", ".join(sorted(commands)))
        raise SystemExit(0)
    command = sys.argv[1]
    handler = commands.get(command)
    if handler is None:
        print(f"未知命令: {command}", file=sys.stderr)
        print("命令: " + ", ".join(sorted(commands)), file=sys.stderr)
        raise SystemExit(2)
    sys.argv = [f"{sys.argv[0]} {command}", *sys.argv[2:]]
    handler()
