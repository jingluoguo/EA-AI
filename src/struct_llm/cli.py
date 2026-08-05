from __future__ import annotations

import argparse
import json
from pathlib import Path

from .errors import ParseError
from .cognitive.intent_dataset import append_intent_record, build_intent_record
from .cognitive.intent_learning import InMemoryIntentAnalyzer, evaluate_intent_analyzer, from_jsonl
from .cognitive.feedback_learning import (
    LearningPaths,
    accept_query_suggestion,
    assess_query_uncertainty,
    save_manual_query_feedback,
    save_manual_statement_feedback,
)
from .cognitive.query_learning import (
    EntityExample,
    LearnedQueryParser,
    QUERY_DIRECT_CONFIDENCE,
    evaluate_query_parser,
    load_query_jsonl,
)
from .cognitive.query_learning import compile_query_model_from_jsonl, save_query_model
from .cognitive.statement_learning import (
    EntitySlot,
    FrameTemplate,
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
    parser.add_argument("--query-min-score", type=float, default=QUERY_DIRECT_CONFIDENCE)
    parser.add_argument("--statement-data", help="JSONL file with learned statement examples.")
    parser.add_argument("--statement-model", help="Compiled statement model artifact.")
    parser.add_argument("--statement-min-score", type=float, default=0.58)
    parser.add_argument("--learn-on-fail", action="store_true", help="Prompt for feedback when structure extraction fails.")
    args = parser.parse_args()
    capabilities = default_capabilities()
    if args.statement_model and Path(args.statement_model).exists():
        capabilities = capabilities.replace_statement_parsers(
            LearnedStatementParser.from_model(Path(args.statement_model), min_score=args.statement_min_score)
        )
    elif args.statement_data and Path(args.statement_data).exists():
        capabilities = capabilities.replace_statement_parsers(
            LearnedStatementParser.from_jsonl(Path(args.statement_data), min_score=args.statement_min_score)
        )
    if args.query_model and Path(args.query_model).exists():
        capabilities = capabilities.replace_query_parsers(
            LearnedQueryParser.from_model(Path(args.query_model), min_score=args.query_min_score)
        )
    elif args.query_data and Path(args.query_data).exists():
        capabilities = capabilities.replace_query_parsers(
            LearnedQueryParser.from_jsonl(Path(args.query_data), min_score=args.query_min_score)
        )
    if args.intent_data:
        analyzer = InMemoryIntentAnalyzer.from_jsonl(Path(args.intent_data), min_score=args.intent_min_score)
        capabilities = capabilities.with_intent_analyzers(analyzer)

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
    print(f"意图样本={result.total} 命中={result.matched} 准确率={result.accuracy:.2f}")


def eval_query_examples() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the learned query parser against JSONL examples.")
    parser.add_argument("--query-data", default="data/query_examples.jsonl")
    parser.add_argument("--query-model", default="")
    parser.add_argument("--query-min-score", type=float, default=QUERY_DIRECT_CONFIDENCE)
    args = parser.parse_args()

    examples = load_query_jsonl(Path(args.query_data))
    query_parser = (
        LearnedQueryParser.from_model(Path(args.query_model), min_score=args.query_min_score)
        if args.query_model
        else LearnedQueryParser.from_examples(examples, min_score=args.query_min_score)
    )
    result = evaluate_query_parser(query_parser, examples)
    print(f"问题样本={result.total} 命中={result.matched} 准确率={result.accuracy:.2f}")


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
    print(f"陈述样本={result.total} 命中={result.matched} 准确率={result.accuracy:.2f}")


def compile_query_model() -> None:
    parser = argparse.ArgumentParser(description="Compile Query JSONL examples into a runtime model artifact.")
    parser.add_argument("--query-data", default="data/query_examples.jsonl")
    parser.add_argument("--output", default="data/query_model.json")
    args = parser.parse_args()

    model = compile_query_model_from_jsonl(Path(args.query_data))
    save_query_model(model, Path(args.output))
    print(f"已生成问题模型：样本={model.example_count} 模式={len(model.patterns)} 输出={args.output}")


def compile_statement_model() -> None:
    parser = argparse.ArgumentParser(description="Compile statement JSONL examples into a runtime model artifact.")
    parser.add_argument("--statement-data", default="data/statement_examples.jsonl")
    parser.add_argument("--output", default="data/statement_model.json")
    args = parser.parse_args()

    model = compile_statement_model_from_jsonl(Path(args.statement_data))
    save_statement_model(model, Path(args.output))
    print(f"已生成陈述模型：样本={model.example_count} 模式={len(model.patterns)} 输出={args.output}")


def print_prediction(question: str, capabilities=None) -> None:
    prediction = predict(question, capabilities)
    print("=" * 60)
    print(question)
    print()
    print(prediction.structure.linearize())
    print()
    print(prediction.answer)


def print_prediction_with_learning(question: str, capabilities, args) -> None:
    try:
        print_prediction(question, capabilities)
    except ParseError:
        if not getattr(args, "learn_on_fail", False):
            raise
        learned = prompt_learning_feedback(question, args)
        if not learned:
            print("已跳过，不写入训练集。")
            return
        refreshed = capabilities_after_recompile(args)
        print("我已经记下并重新整理模型了。现在重试一次：")
        try:
            print_prediction(question, refreshed)
        except ParseError:
            print("样本已经保存，不过当前还不能完整回答。后续补更多同类样本后会更稳。")


def prompt_learning_feedback(text: str, args) -> bool:
    similar_result = prompt_similar_query_feedback(text, args)
    if similar_result is True:
        return True
    if similar_result is None:
        print("我确实还没懂这句话。")
        print("没有找到足够相近的已学结构，我们一步步来。")
    else:
        print("好，那我不按刚才的猜测处理。")
        print("我们一步步来，你告诉我它该沉淀成什么结构。")
    choice = choose_from_menu(
        "这句话更像哪一类？",
        (
            ("1", "问题或追问", "用户在问一件事，希望我回答。"),
            ("2", "事实陈述", "用户在告诉我一件事实或事件。"),
            ("3", "行为意图", "用户在描述某人的目标、想法或动机。"),
            ("4", "先跳过", "这次不写入训练集。"),
        ),
    )
    if choice == "1":
        return prompt_query_feedback(text, args)
    if choice == "2":
        return prompt_statement_feedback(text, args)
    if choice == "3":
        return prompt_intent_feedback(text, args)
    return False


def prompt_similar_query_feedback(text: str, args) -> bool | None:
    capabilities = capabilities_after_recompile(args)
    assessment = assess_query_uncertainty(text, capabilities.query_parsers)
    suggestion = assessment.suggestion
    if suggestion is None:
        return None
    description = describe_query(suggestion.query.target, suggestion.query.intent, suggestion.query.qualifiers)
    choice = choose_from_menu(
        f"我有点把握，但还不想直接答。它是不是在{description}？",
        (
            ("1", "是这个意思", "把这句话按这个结构沉淀到训练集。"),
            ("2", "不是", "继续手动教我。"),
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
    print(f"已保存问题样本并重新编译，问题样本现在有 {result.example_count} 条。")
    return True


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
    print(f"已保存陈述样本并重新编译，陈述样本现在有 {result.example_count} 条。")
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
        query_model=Path(getattr(args, "query_model", None) or "data/query_model.json"),
        statement_data=Path(getattr(args, "statement_data", None) or "data/statement_examples.jsonl"),
        statement_model=Path(getattr(args, "statement_model", None) or "data/statement_model.json"),
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


def capabilities_after_recompile(args):
    capabilities = default_capabilities()
    query_model = getattr(args, "query_model", None)
    statement_model = getattr(args, "statement_model", None)
    if statement_model and Path(statement_model).exists():
        capabilities = capabilities.replace_statement_parsers(LearnedStatementParser.from_model(Path(statement_model)))
    if query_model and Path(query_model).exists():
        capabilities = capabilities.replace_query_parsers(LearnedQueryParser.from_model(Path(query_model)))
    return capabilities


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
