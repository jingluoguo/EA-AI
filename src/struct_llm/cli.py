from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import ParseError
from .comprehension.intent_dataset import append_intent_record, build_intent_record
from .comprehension.intent import InMemoryIntentAnalyzer, evaluate_intent_analyzer, from_jsonl
from .kernel import default_capabilities, parse_text, predict
from .neural import load_neural_boundary_model, with_neural_boundary
from .motor.feedback import (
    LearningPaths,
    accept_query_suggestion,
    assess_query_uncertainty,
    save_manual_query_feedback,
    save_manual_statement_feedback,
    save_chat_memory_feedback,
    save_direct_memory_feedback,
    save_direct_memory_structure_feedback,
    save_memory_knowledge_feedback,
    save_unrecognized_feedback,
)
from .motor.dialogue import (
    LearnedDialogActAnswerer,
    compile_dialog_answer_model_from_jsonl,
    save_dialog_answer_model,
)
from .comprehension.query import EntityExample
from .comprehension.statement import EntitySlot, FrameTemplate
from .memory.long_term import (
    extract_chat_memory_entries,
    load_memory_model,
)
from .memory.knowledge import (
    compile_memory_knowledge_model_from_jsonl,
    default_learned_memory_knowledge_answerer,
    load_memory_knowledge_model,
    save_memory_knowledge_model,
)
from .structure import State
from .neural.query_classifier import train_query_neural_model
from .neural.statement_classifier import train_statement_neural_model


QUESTIONS = (
    "小明把钥匙放进盒子。盒子被带到厨房。钥匙在哪里？",
    "研究员把芯片放进托盘。托盘被带到实验室。芯片在哪里？",
    "小红把药瓶交给医生。现在谁拥有药瓶？",
    "工程师把笔记本涂成绿色。现在笔记本是什么颜色？",
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


def train_neural_models() -> None:
    from my_neural import train

    train()


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
        choices=("fallback", "first"),
        default="fallback",
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


def compile_memory_knowledge_model() -> None:
    parser = argparse.ArgumentParser(description="Compile verified long-term knowledge entries into a runtime artifact.")
    parser.add_argument("--memory-knowledge-data", default="data/memory_knowledge_examples.jsonl")
    parser.add_argument("--output", default="data/memory_knowledge_model.json")
    args = parser.parse_args()

    model = compile_memory_knowledge_model_from_jsonl(Path(args.memory_knowledge_data))
    save_memory_knowledge_model(model, Path(args.output))
    print(f"已生成长期知识模型：样本={model.example_count} 模式={len(model.patterns)} 输出={args.output}")


def print_prediction(question: str, capabilities=None) -> None:
    prediction = predict(question, capabilities)
    print("=" * 60)
    print(question)
    print()
    print(prediction.structure.linearize())
    print()
    print(prediction.answer)
    return prediction


def print_prediction_with_learning(question: str, capabilities, args) -> None:
    try:
        prediction = print_prediction(question, capabilities)
        maybe_save_chat_memory(question, prediction.structure, args)
    except ParseError:
        try:
            structure = parse_text(question, capabilities)
        except ParseError:
            structure = None
        if structure is not None and structure.query is not None:
            print_unanswered_structure(question, structure)
            return
        if not getattr(args, "learn_on_fail", False):
            raise
        learned = prompt_learning_feedback(question, args)
        if learned == "queued":
            return
        if not learned:
            print("已跳过，不写入训练集。")
            return
        refreshed = capabilities_after_training(args)
        print("我已经记下并重新整理模型了。现在重试一次：")
        try:
            print_prediction(question, refreshed)
        except ParseError:
            try:
                structure = parse_text(question, refreshed)
            except ParseError:
                structure = None
            if structure is not None and structure.query is not None:
                print_unanswered_structure(question, structure)
            else:
                print("样本已经保存，不过当前还不能完整理解。后续补充训练样本后再试。")


def prompt_learning_feedback(text: str, args) -> str | bool:
    capabilities = capabilities_after_training(args)
    assessment = assess_query_uncertainty(text, capabilities.query_parsers)
    if assessment.band == "unknown":
        save_unrecognized_feedback(
            text,
            learning_paths_from_args(args),
            confidence=assessment.score,
        )
        print("暂时无法识别。")
        return "queued"
    similar_result = prompt_similar_query_feedback(text, args, assessment)
    if similar_result is True:
        return "updated"
    save_unrecognized_feedback(
        text,
        learning_paths_from_args(args),
        confidence=assessment.score,
        reason="user_rejected_suggestion",
    )
    print("暂时无法识别。")
    return "queued"


def prompt_similar_query_feedback(text: str, args, assessment=None) -> bool | None:
    if assessment is None:
        capabilities = capabilities_after_training(args)
        assessment = assess_query_uncertainty(text, capabilities.query_parsers)
    suggestion = assessment.suggestion
    if suggestion is None:
        return None
    description = describe_query(suggestion.query.target, suggestion.query.intent, suggestion.query.qualifiers)
    choice = choose_from_menu(
        f"我有点把握，但还不想直接答。它是不是在{description}？",
        (
            ("1", "是这个意思", "把这句话按这个结构沉淀到训练集。"),
            ("2", "不是这个意思", "先记录下来，稍后整理。"),
        ),
    )
    if choice == "1":
        result = accept_query_suggestion(suggestion, learning_paths_from_args(args))
        print(f"明白了。这个判断的置信度约 {assessment.score:.2f}，问题样本现在有 {result.example_count} 条。")
        return True
    return False


def prompt_query_feedback(text: str, args) -> bool:
    question = text
    kind = choose_from_menu(
        "它想问什么？",
        (
            ("1", "你能做什么", "学习为能力介绍。"),
            ("2", "你是谁", "学习为身份介绍。"),
            ("3", "打招呼", "学习为问候。"),
            ("4", "表示感谢", "学习为感谢。"),
            ("5", "告别", "学习为告别。"),
            ("6", "总结一下", "学习为对话摘要。"),
            ("7", "自定义结构", "手动提供结构类型、目标和限定条件。"),
        ),
    )
    intent, target, entities, qualifiers = query_feedback_values(kind)
    if kind == "7":
        raw_entities = input_default("实体列表，留空表示没有", "[]")
        intent = input_required("结构类型")
        target = input_required("查询目标")
        raw_qualifiers = input_default("限定条件，多个用逗号分开", "")
        entities = tuple(
            EntityExample(str(item["role"]), str(item["name"])) for item in parse_json_list(raw_entities, "实体列表")
        )
        qualifiers = tuple(value.strip() for value in raw_qualifiers.split(",") if value.strip())
    result = save_manual_query_feedback(
        question,
        intent,
        target,
        learning_paths_from_args(args),
        entities=entities,
        qualifiers=qualifiers,
    )
    print(f"已保存问题样本并重新训练神经模型，问题样本现在有 {result.example_count} 条。")
    return True


def print_unanswered_structure(question: str, structure) -> None:
    print("=" * 60)
    print(question)
    print()
    print(structure.linearize())
    print()
    print("我已经理解你的问题，但还没有经过验证的相关回答。")
    print("我不会把刚才的输入直接当成答案。")


def describe_query(target: str, intent: str, qualifiers: tuple[str, ...]) -> str:
    if intent == "dialog_act":
        if target == "capabilities":
            return "询问我能做什么"
        if target == "identity":
            return "询问我是谁"
        if target == "greeting":
            return "打招呼"
        if target == "thanks":
            return "表示感谢"
        if target == "farewell":
            return "告别"
        if target == "summary":
            return "让总结一下"
        return f"询问{target}"
    if intent == "location":
        return f"询问{target}在哪里"
    if intent == "contents":
        return f"询问{target}里有什么"
    if qualifiers:
        return f"{intent}，目标是 {target}，条件是 {'，'.join(qualifiers)}"
    return f"{intent}，目标是 {target}"


def prompt_statement_feedback(text: str, args) -> bool:
    print("这类样本需要标出“实体槽位”和“事件结构”。")
    print('例：句子模板可以写成 "$person#1把$item#1放进$container#1"。')
    sentence = input_default("要学习的原句", text)
    sentence_template = input_required("句子模板")
    raw_entities = input_required("实体列表")
    raw_frames = input_required("事件结构列表")
    entities = tuple(EntitySlot(str(item["role"]), str(item["name"])) for item in parse_json_list(raw_entities, "实体列表"))
    frames = tuple(frame_template_from_feedback(item) for item in parse_json_list(raw_frames, "事件结构列表"))
    result = save_manual_statement_feedback(
        sentence,
        sentence_template,
        learning_paths_from_args(args),
        entities=entities,
        frames=frames,
    )
    print(f"已保存陈述样本并重新训练神经模型，陈述样本现在有 {result.example_count} 条。")
    return True


def prompt_intent_feedback(text: str, args) -> bool:
    subject = input_required("这是在描述谁的意图")
    goal = input_required("这个人想达成什么")
    belief = input_default("这个人当时相信什么", "")
    strategy = input_default("这个人可能会怎么做", "")
    intent_data = Path(getattr(args, "intent_data", None) or "data/intent_examples.jsonl")
    append_intent_record(
        intent_data,
        build_intent_record(
            text,
            subject,
            goal,
            belief=belief,
            strategy=strategy,
            source="human_feedback",
        ),
    )
    return True


def learning_paths_from_args(args) -> LearningPaths:
    return LearningPaths(
        query_data=Path(getattr(args, "query_data", None) or "data/query_examples.jsonl"),
        query_neural_weights=Path(getattr(args, "query_neural_weights", None) or "data/query_neural_model.pt"),
        query_neural_meta=Path(getattr(args, "query_neural_meta", None) or "data/query_neural_model.json"),
        statement_data=Path(getattr(args, "statement_data", None) or "data/statement_examples.jsonl"),
        statement_neural_weights=Path(getattr(args, "statement_neural_weights", None) or "data/statement_neural_model.pt"),
        statement_neural_meta=Path(getattr(args, "statement_neural_meta", None) or "data/statement_neural_model.json"),
        dialog_answer_data=Path(getattr(args, "dialog_answer_data", None) or "data/dialog_answer_examples.jsonl"),
        dialog_answer_model=Path(getattr(args, "dialog_answer_model", None) or "data/dialog_answer_model.json"),
        unrecognized_data=Path(
            getattr(args, "unrecognized_data", None) or "data/unrecognized_examples.jsonl"
        ),
        memory_direct_data=Path(getattr(args, "memory_direct_data", None) or "data/memory_direct_examples.jsonl"),
        memory_chat_data=Path(getattr(args, "memory_chat_data", None) or "data/memory_chat_examples.jsonl"),
        memory_model=Path(getattr(args, "memory_model", None) or "data/memory_model.json"),
        memory_knowledge_data=Path(
            getattr(args, "memory_knowledge_data", None) or "data/memory_knowledge_examples.jsonl"
        ),
        memory_knowledge_model=Path(
            getattr(args, "memory_knowledge_model", None) or "data/memory_knowledge_model.json"
        ),
    )


def frame_template_from_feedback(record: dict) -> FrameTemplate:
    frame_type = str(record.get("frame_type") or "").strip()
    roles = record.get("roles")
    if not frame_type or not isinstance(roles, dict):
        raise ValueError("事件结构需要包含 frame_type 和 roles。")
    return FrameTemplate(frame_type, tuple((str(key), str(value)) for key, value in roles.items()))


def parse_json_list(raw: str, label: str) -> list[dict]:
    value = json.loads(raw)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{label}需要是对象列表。")
    return value


def query_feedback_values(kind: str) -> tuple[str, str, tuple[EntityExample, ...], tuple[str, ...]]:
    if kind == "1":
        return "dialog_act", "capabilities", (), ()
    if kind == "2":
        return "dialog_act", "identity", (), ()
    if kind == "3":
        return "dialog_act", "greeting", (), ()
    if kind == "4":
        return "dialog_act", "thanks", (), ()
    if kind == "5":
        return "dialog_act", "farewell", (), ()
    if kind == "6":
        return "dialog_act", "summary", (), ()
    return "", "", (), ()


def choose_from_menu(title: str, options: tuple[tuple[str, str, str], ...]) -> str:
    print(title)
    for key, label, description in options:
        print(f"{key}. {label}：{description}")
    allowed = {key for key, _, _ in options}
    while True:
        choice = input("请选择编号> ").strip()
        if choice in allowed:
            return choice


def input_default(label: str, default: str) -> str:
    value = input(f"{label} [{default}]> ").strip()
    return value or default


def input_required(label: str) -> str:
    while True:
        value = input(f"{label}> ").strip()
        if value:
            return value


def capabilities_after_training(args):
    capabilities = default_capabilities(
        neural_answer_priority=getattr(args, "neural_answer_priority", "first"),
        use_environment=False,
        use_memory=False,
    )
    dialog_answer_model = getattr(args, "dialog_answer_model", None)
    if dialog_answer_model and Path(dialog_answer_model).exists():
        capabilities = capabilities.with_answerers(LearnedDialogActAnswerer.from_model(Path(dialog_answer_model)))
    capabilities = apply_memory_args(capabilities, args)
    capabilities = apply_memory_knowledge_args(capabilities, args)
    return apply_neural_provider_args(capabilities, args)


def apply_neural_provider_args(capabilities, args):
    provider = getattr(args, "neural_provider", None)
    if provider:
        model = load_neural_boundary_model(provider)
    else:
        from .neural import configured_neural_boundary_model

        model = configured_neural_boundary_model()
    if model is None:
        return capabilities
    return with_neural_boundary(
        capabilities,
        model,
        statement_priority="replace",
        query_priority="replace",
        answer_priority=getattr(args, "neural_answer_priority", "fallback"),
    )


def apply_memory_args(capabilities, args):
    memory_model = getattr(args, "memory_model", None) or "data/memory_model.json"
    memory_path = Path(memory_model)
    if not memory_path.exists():
        return capabilities
    return capabilities.with_memory_states(*load_memory_model(memory_path).states)


def apply_memory_knowledge_args(capabilities, args):
    knowledge_model = getattr(args, "memory_knowledge_model", None) or "data/memory_knowledge_model.json"
    knowledge_path = Path(knowledge_model)
    if knowledge_path.exists():
        return capabilities.with_answerers(default_learned_memory_knowledge_answerer(knowledge_path))
    knowledge_data = getattr(args, "memory_knowledge_data", None)
    if knowledge_data and Path(knowledge_data).exists():
        return capabilities.with_answerers(default_learned_memory_knowledge_answerer(Path(knowledge_data)))
    return capabilities


def maybe_save_chat_memory(question: str, structure, args) -> None:
    if not getattr(args, "remember_chat", False):
        return
    entries = extract_chat_memory_entries(question, structure)
    if not entries:
        return
    print("候选记忆：")
    for index, entry in enumerate(entries, start=1):
        state = entry.state
        print(f"{index}. STATE {state.name}({state.left},{state.right})")
    choice = choose_from_menu(
        "这些内容要写入长期记忆吗？",
        (
            ("1", "写入", "确认这些结构事实可信，保存到长期记忆。"),
            ("2", "跳过", "只用于当前推理，不写入长期记忆。"),
        ),
    )
    if choice != "1":
        print("已跳过，不写入长期记忆。")
        return
    paths = learning_paths_from_args(args)
    result = save_chat_memory_feedback(question, structure, paths)
    if result.entry_count:
        print(f"已沉淀 {result.entry_count} 条记忆，当前记忆状态 {result.state_count} 条。")


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
        load_memory_knowledge_source_records(Path(args.file), fallback_source=args.source)
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
    fallback_source: str,
) -> tuple[tuple[str, str, str], ...]:
    records: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid knowledge source JSONL at line {line_number}: {error}") from error
            if not isinstance(raw_record, dict):
                raise ValueError(f"Invalid knowledge source JSONL at line {line_number}: expected object")
            question = str(raw_record.get("question") or raw_record.get("text") or "").strip()
            answer = str(raw_record.get("answer") or raw_record.get("response") or "").strip()
            source = str(raw_record.get("source") or fallback_source).strip() or fallback_source
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

    from .memory.long_term import compile_memory_model_from_jsonl, save_memory_model

    model = compile_memory_model_from_jsonl(Path(args.memory_direct_data), Path(args.memory_chat_data))
    save_memory_model(model, Path(args.output))
    print(f"已生成记忆模型：样本={model.example_count} 状态={len(model.states)} 输出={args.output}")
