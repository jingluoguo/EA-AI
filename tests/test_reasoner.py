from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from struct_llm.errors import ParseError
from struct_llm.capabilities import CognitiveCapabilities
from struct_llm.comprehension.structure_helpers import frame_from_roles, with_time
from struct_llm.comprehension.intent_dataset import (
    append_intent_record,
    build_intent_record,
    intent_record_from_dict,
    load_intent_jsonl,
)
from struct_llm.kernel import default_capabilities, parse_text, predict as _predict
from struct_llm.neural import (
    InMemoryNeuralBoundaryModel,
    NeuralQueryParser,
    NeuralStatementParser,
    load_neural_boundary_model,
    with_neural_boundary,
)
from my_neural import make_model, train_summary
from struct_llm.motor.feedback import (
    LearningPaths,
    accept_query_suggestion,
    assess_query_uncertainty,
    confidence_band,
    save_chat_memory_feedback,
    save_direct_memory_feedback,
    save_manual_query_feedback,
    save_manual_statement_feedback,
    save_new_dialog_capability_feedback,
    save_unrecognized_feedback,
    suggest_query_feedback,
)
from struct_llm.motor.dialogue import LearnedDialogActAnswerer, save_manual_dialog_answer_feedback
from struct_llm.motor.feedback import save_memory_knowledge_feedback
from struct_llm.comprehension.intent import InMemoryIntentAnalyzer, evaluate_intent_analyzer
from struct_llm.motor.learning_queue import load_unrecognized_jsonl
from struct_llm.memory.long_term import load_memory_model
from struct_llm.memory.knowledge import (
    LearnedMemoryKnowledgeAnswerer,
    default_learned_memory_knowledge_answerer,
    load_memory_knowledge_model,
)
from struct_llm.comprehension.query import (
    LearnedQueryParser,
    evaluate_query_parser,
    load_query_jsonl,
)
from struct_llm.comprehension.statement import (
    EntitySlot,
    FrameTemplate,
    LearnedStatementParser,
    evaluate_statement_parser,
    linearize_statement_result,
    load_statement_jsonl,
    normalize_statement_text,
    statement_example_from_dict,
)
from struct_llm.world.event_schema import EVENT_SCHEMAS, frame_matches_qualifiers, states_for_frame_schema
from struct_llm.structure import Entity, Intention, Query, Structure
from struct_llm.structure import State
from struct_llm.neural.query_classifier import default_neural_query_parser, train_query_neural_model
from struct_llm.neural.statement_classifier import default_neural_statement_parser, train_statement_neural_model


def predict(text: str, capabilities: CognitiveCapabilities | None = None):
    try:
        prediction = _predict(text, capabilities)
    except Exception as error:
        print(f"{text} -> ERROR: {error}", flush=True)
        raise
    print(f"{text} -> {prediction.answer}", flush=True)
    return prediction


class ReasonerTest(unittest.TestCase):
    def test_default_capabilities_are_registered_as_cognitive_kernel(self) -> None:
        capabilities = default_capabilities()

        self.assertIsInstance(capabilities, CognitiveCapabilities)

    def test_default_query_capability_uses_neural_model(self) -> None:
        capabilities = default_capabilities()

        self.assertEqual(len(capabilities.query_parsers), 1)
        parser = capabilities.query_parsers[0]
        self.assertGreater(len(parser.patterns), 0)

    def test_default_statement_capability_uses_neural_model(self) -> None:
        capabilities = default_capabilities()

        self.assertEqual(len(capabilities.statement_parsers), 1)
        parser = capabilities.statement_parsers[0]
        self.assertGreater(len(parser.patterns), 0)

    def test_neural_query_parser_can_replace_input_boundary_without_kernel_branch(self) -> None:
        model = InMemoryNeuralBoundaryModel(
            {
                "parse_query": lambda payload: {
                    "confidence": 0.96,
                    "query": {"intent": "dialog_act", "target": "identity", "qualifiers": []},
                },
            }
        )
        capabilities = with_neural_boundary(
            default_capabilities(use_environment=False, use_memory=False),
            model,
            query_priority="replace",
        )

        prediction = predict("你是哪位？", capabilities)

        self.assertIsInstance(capabilities.query_parsers[0], NeuralQueryParser)
        self.assertEqual(len(capabilities.query_parsers), 1)
        self.assertIn("QUERY dialog_act(identity)", prediction.structure.linearize())
        self.assertEqual(
            prediction.answer,
            "我是结构智能原型，会把对话里的事实、状态、信念和问题先整理成结构再回答。",
        )

    def test_neural_statement_parser_projects_into_existing_state_reasoning(self) -> None:
        model = InMemoryNeuralBoundaryModel(
            {
                "parse_statement": lambda payload: {
                    "confidence": 0.95,
                    "entities": [
                        {"role": "person", "name": "阿明"},
                        {"role": "item", "name": "芯片"},
                        {"role": "place", "name": "库房"},
                    ],
                    "frames": [
                        {
                            "frame_type": "move",
                            "roles": {"actor": "阿明", "theme": "芯片", "goal": "库房"},
                        }
                    ],
                }
            }
        )
        capabilities = default_capabilities(neural_model=model)

        prediction = predict("阿明递送芯片到库房。芯片在哪里？", capabilities)

        structure = prediction.structure.linearize()
        self.assertIsInstance(capabilities.statement_parsers[0], NeuralStatementParser)
        self.assertIn("FRAME f1 type=move time=1", structure)
        self.assertIn("ROLE f1 actor=阿明", structure)
        self.assertIn("REL at(芯片,库房)", structure)
        self.assertEqual(prediction.answer, "芯片在库房。")

    def test_neural_answerer_handles_unverified_dialog_output_as_fallback(self) -> None:
        def parse_query(payload):
            return {
                "confidence": 0.97,
                "query": {"intent": "dialog_act", "target": "poem_translation", "qualifiers": []},
            }

        def answer(payload):
            structure = payload["structure"]
            if structure["query"]["target"] != "poem_translation":
                return None
            return {"confidence": 0.91, "answer": "可以，我会先保留意思，再把语气整理得更有诗意。"}

        model = InMemoryNeuralBoundaryModel({"parse_query": parse_query, "answer": answer})
        capabilities = default_capabilities(neural_model=model)

        prediction = predict("你能帮我把这句话改得像诗吗？", capabilities)

        self.assertIn("QUERY dialog_act(poem_translation)", prediction.structure.linearize())
        self.assertEqual(prediction.answer, "可以，我会先保留意思，再把语气整理得更有诗意。")

    def test_neural_boundary_rejects_low_confidence_structures(self) -> None:
        model = InMemoryNeuralBoundaryModel(
            {
                "parse_query": lambda payload: {
                    "confidence": 0.2,
                    "query": {"intent": "dialog_act", "target": "identity", "qualifiers": []},
                },
            }
        )
        capabilities = default_capabilities(neural_model=model)

        with self.assertRaises(ParseError):
            predict("星图回声？", capabilities)

    def test_project_root_neural_provider_can_be_loaded_from_cli_spec(self) -> None:
        model = load_neural_boundary_model("my_neural:make_model")
        capabilities = default_capabilities(neural_model=model)

        prediction = predict("你是谁？", capabilities)

        self.assertIn("QUERY dialog_act(identity)", prediction.structure.linearize())
        self.assertEqual(
            prediction.answer,
            "我是结构智能原型，会把对话里的事实、状态、信念和问题先整理成结构再回答。",
        )

    def test_neural_provider_answers_knowledge_questions_without_question_rewrite(self) -> None:
        model = make_model()
        capabilities = default_capabilities(use_environment=False, use_memory=False)
        capabilities = capabilities.with_answerers(
            default_learned_memory_knowledge_answerer("data/memory_knowledge_model.json")
        )
        capabilities = with_neural_boundary(capabilities, model)

        prediction = predict("我家的铁怎么生锈了", capabilities)

        self.assertIn("QUERY why(铁会生锈,type=why)", prediction.structure.linearize())
        self.assertEqual(prediction.answer, "铁和空气里的氧、水发生反应后，会生成疏松的氧化物，也就是锈。")

    def test_default_neural_query_parser_handles_close_variants_of_the_same_question(self) -> None:
        parser = default_neural_query_parser()

        for text in ("我家的铁咋生锈了", "我家的铁为啥生锈了", "我家的铁为什么生锈了", "铁生锈是什么原理"):
            with self.subTest(text=text):
                query = parser(text, ())
                self.assertIsNotNone(query)
                assert query is not None
                self.assertEqual(query.linearize(), "QUERY why(铁会生锈,type=why)")

    def test_neural_training_summary_uses_current_datasets(self) -> None:
        model = make_model()
        summary = train_summary(model)

        self.assertIn("query_neural", summary)
        self.assertIn("statement", summary)
        self.assertIn("statement_neural", summary)
        self.assertIn("intent", summary)
        self.assertIn("dialog_answer", summary)
        self.assertGreater(summary["query_neural"]["examples"], 0)
        self.assertGreater(summary["statement"]["examples"], 0)
        self.assertGreater(summary["statement_neural"]["examples"], 0)
        self.assertGreater(summary["intent"]["examples"], 0)

    def test_neural_statement_parser_normalizes_active_passive_and_reordered_sentences(self) -> None:
        parser = default_neural_statement_parser()

        for text in ("阿明递送芯片到库房", "阿明把芯片送到库房", "芯片被阿明送到库房"):
            with self.subTest(text=text):
                parsed = parser(text)
                self.assertIsNotNone(parsed)
                assert parsed is not None
                entities, frames = parsed
                self.assertEqual(
                    {(entity.role, entity.name) for entity in entities},
                    {("person", "阿明"), ("item", "芯片"), ("place", "库房")},
                )
                move_frames = [frame for frame in frames if frame.frame_type == "move"]
                self.assertTrue(move_frames)
                self.assertEqual(
                    {role.name: role.value for role in move_frames[0].roles},
                    {"actor": "阿明", "theme": "芯片", "goal": "库房"},
                )

    def test_statement_examples_can_evaluate_neural_parser(self) -> None:
        examples = load_statement_jsonl("data/statement_examples.jsonl")
        result = evaluate_statement_parser(default_neural_statement_parser(), examples)

        self.assertEqual(result.total, len(examples))
        self.assertGreaterEqual(result.accuracy, 0.90)

    def test_statement_examples_train_to_neural_runtime_model(self) -> None:
        examples = load_statement_jsonl("data/statement_examples.jsonl")
        with tempfile.TemporaryDirectory() as directory:
            weights_path = Path(directory) / "statement_neural_model.pt"
            meta_path = Path(directory) / "statement_neural_model.json"
            bundle = train_statement_neural_model("data/statement_examples.jsonl", weights_path, meta_path)
            parser = default_neural_statement_parser("data/statement_examples.jsonl", weights_path, meta_path)
            result = evaluate_statement_parser(parser, examples)

            self.assertTrue(weights_path.exists())
            self.assertTrue(meta_path.exists())

        self.assertEqual(bundle.result.example_count, len(examples))
        self.assertGreater(bundle.result.label_count, 0)
        self.assertGreaterEqual(result.accuracy, 0.90)

    def test_statement_feedback_appends_and_trains_neural_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "statement_examples.jsonl"
            weights_path = Path(directory) / "statement_neural_model.pt"
            meta_path = Path(directory) / "statement_neural_model.json"
            result = save_manual_statement_feedback(
                "小王打开盒子",
                "$person#1打开$container#1",
                LearningPaths(
                    statement_data=data_path,
                    statement_neural_weights=weights_path,
                    statement_neural_meta=meta_path,
                ),
                entities=(
                    EntitySlot("person", "$person#1"),
                    EntitySlot("container", "$container#1"),
                ),
                frames=(
                    FrameTemplate(
                        "open",
                        (("actor", "$person#1"), ("theme", "$container#1"), ("result", "打开")),
                    ),
                ),
            )
            parser = default_neural_statement_parser(data_path, weights_path, meta_path)
            parsed = parser("小王打开盒子")

            self.assertTrue(weights_path.exists())
            self.assertTrue(meta_path.exists())

        self.assertEqual(result.example_count, 1)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertIn("FRAME open", linearize_statement_result(parsed))

    def test_statement_normalization_collapses_container_surface_forms(self) -> None:
        self.assertEqual(
            normalize_statement_text("小王把芯片从托盘里面拿出来"),
            normalize_statement_text("小王把芯片从托盘里取出"),
        )

    def test_statement_dataset_rejects_missing_template(self) -> None:
        with self.assertRaisesRegex(ValueError, "sentence and sentence_template"):
            statement_example_from_dict(
                {
                    "sentence": "小王把芯片放进托盘",
                    "entities": [],
                    "frames": [{"frame_type": "put_in", "roles": {}}],
                }
            )

    def test_neural_query_dataset_replaces_question_parser_stack(self) -> None:
        parser = default_neural_query_parser()

        query = parser("盒子里有芯片吗", (Entity("container", "盒子"),))

        self.assertIsNotNone(query)
        assert query is not None
        self.assertEqual(query.linearize(), "QUERY polar_contents(盒子,item=芯片)")

    def test_query_examples_can_evaluate_neural_parser(self) -> None:
        examples = load_query_jsonl("data/query_examples.jsonl")
        result = evaluate_query_parser(default_neural_query_parser(), examples)

        self.assertEqual(result.total, len(examples))
        self.assertGreaterEqual(result.accuracy, 0.99)

    def test_query_examples_train_to_neural_runtime_model(self) -> None:
        examples = load_query_jsonl("data/query_examples.jsonl")
        with tempfile.TemporaryDirectory() as directory:
            weights_path = Path(directory) / "query_neural_model.pt"
            meta_path = Path(directory) / "query_neural_model.json"
            bundle = train_query_neural_model("data/query_examples.jsonl", weights_path, meta_path)
            parser = default_neural_query_parser("data/query_examples.jsonl", weights_path, meta_path)
            result = evaluate_query_parser(parser, examples)

            self.assertTrue(weights_path.exists())
            self.assertTrue(meta_path.exists())

        self.assertEqual(bundle.result.example_count, len(examples))
        self.assertGreater(bundle.result.label_count, 0)
        self.assertGreaterEqual(result.accuracy, 0.99)

    def test_query_feedback_appends_and_trains_neural_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "query_examples.jsonl"
            weights_path = Path(directory) / "query_neural_model.pt"
            meta_path = Path(directory) / "query_neural_model.json"
            result = save_manual_query_feedback(
                "你能干嘛",
                "dialog_act",
                "capabilities",
                LearningPaths(query_data=data_path, query_neural_weights=weights_path, query_neural_meta=meta_path),
            )
            parser = default_neural_query_parser(data_path, weights_path, meta_path)
            query = parser("你能干嘛", ())

            self.assertTrue(weights_path.exists())
            self.assertTrue(meta_path.exists())

        self.assertEqual(result.example_count, 1)
        self.assertIsNotNone(query)
        assert query is not None
        self.assertEqual(query.linearize(), "QUERY dialog_act(capabilities)")

    def test_query_feedback_can_suggest_similar_learned_meaning(self) -> None:
        parser = default_neural_query_parser()

        suggestion = suggest_query_feedback("你擅长啥", (parser,))

        self.assertIsNotNone(suggestion)
        assert suggestion is not None
        self.assertEqual(suggestion.query.linearize(), "QUERY dialog_act(capabilities)")

    def test_query_uncertainty_bands_direct_confirm_and_unknown(self) -> None:
        parser = default_neural_query_parser()

        direct = parser("你能做什么", ())
        confirm = suggest_query_feedback("你会做啥", (parser,))
        unknown = suggest_query_feedback("风雨雷电云山河", (parser,))
        confirm_assessment = assess_query_uncertainty("你会做啥", (parser,))
        unknown_assessment = assess_query_uncertainty("风雨雷电云山河", (parser,))

        self.assertIsNotNone(direct)
        self.assertIsNotNone(confirm)
        assert confirm is not None
        self.assertIn(confidence_band(confirm.score), {"confirm", "direct"})
        self.assertEqual(confirm.query.linearize(), "QUERY dialog_act(capabilities)")
        self.assertIsNone(unknown)
        self.assertIn(confirm_assessment.band, {"confirm", "direct"})
        self.assertIsNotNone(confirm_assessment.suggestion)
        self.assertEqual(unknown_assessment.band, "unknown")
        self.assertIsNone(unknown_assessment.suggestion)

    def test_query_feedback_accepts_suggested_structure_without_cli_coupling(self) -> None:
        base_parser = default_neural_query_parser()
        suggestion = suggest_query_feedback("你擅长啥", (base_parser,))
        self.assertIsNotNone(suggestion)
        assert suggestion is not None
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "query_examples.jsonl"
            weights_path = Path(directory) / "query_neural_model.pt"
            meta_path = Path(directory) / "query_neural_model.json"
            result = accept_query_suggestion(
                suggestion,
                LearningPaths(query_data=data_path, query_neural_weights=weights_path, query_neural_meta=meta_path),
            )
            parser = default_neural_query_parser(data_path, weights_path, meta_path)
            query = parser("你擅长啥", ())
            self.assertTrue(weights_path.exists())
            self.assertTrue(meta_path.exists())

        self.assertEqual(result.example_count, 1)
        self.assertIsNotNone(query)
        assert query is not None
        self.assertEqual(query.linearize(), "QUERY dialog_act(capabilities)")

    def test_verified_dialog_answer_feedback_compiles_new_capability(self) -> None:
        query = Query("dialog_act", "emotion_status")
        structure = Structure(entities=(), rules=(), query=query)
        answer = "我没有人类意义上的情绪，但可以和你讨论情绪。"
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "dialog_answer_examples.jsonl"
            model_path = Path(directory) / "dialog_answer_model.json"
            example, model = save_manual_dialog_answer_feedback(
                "你有情绪吗",
                query,
                answer,
                data_path,
                model_path,
                source="self_model",
            )
            answerer = LearnedDialogActAnswerer.from_model(model_path)

        self.assertEqual(example.answer, answer)
        self.assertEqual(model.example_count, 1)
        self.assertEqual(answerer(structure), answer)

    def test_candidate_dialog_answer_feedback_is_not_runtime_answer(self) -> None:
        query = Query("dialog_act", "emotion_status")
        structure = Structure(entities=(), rules=(), query=query)
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "dialog_answer_examples.jsonl"
            model_path = Path(directory) / "dialog_answer_model.json"
            example, model = save_manual_dialog_answer_feedback(
                "你有情绪吗",
                query,
                "按你现在的状态",
                data_path,
                model_path,
            )
            answerer = LearnedDialogActAnswerer.from_model(model_path)

        self.assertEqual(example.source, "candidate")
        self.assertEqual(model.example_count, 0)
        self.assertIsNone(answerer(structure))

    def test_new_dialog_capability_trains_neural_query_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = LearningPaths(
                query_data=Path(directory) / "query_examples.jsonl",
                query_neural_weights=Path(directory) / "query_neural_model.pt",
                query_neural_meta=Path(directory) / "query_neural_model.json",
                dialog_answer_data=Path(directory) / "dialog_answer_examples.jsonl",
                dialog_answer_model=Path(directory) / "dialog_answer_model.json",
            )
            result = save_new_dialog_capability_feedback(
                "你有情绪吗",
                "emotion_status",
                paths,
            )
            query = default_neural_query_parser(paths.query_data, paths.query_neural_weights, paths.query_neural_meta)(
                "你有情绪吗",
                (),
            )
            self.assertIsNotNone(query)
            assert query is not None
            self.assertTrue(paths.query_neural_weights.exists())
            self.assertTrue(paths.query_neural_meta.exists())

        self.assertEqual(result.example_count, 1)
        self.assertFalse(paths.dialog_answer_model.exists())

    def test_low_confidence_feedback_is_saved_to_unrecognized_queue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            queue_path = Path(directory) / "unrecognized_examples.jsonl"
            example = save_unrecognized_feedback(
                "你会梦见电子羊吗",
                LearningPaths(unrecognized_data=queue_path),
                confidence=0.18,
            )
            records = load_unrecognized_jsonl(queue_path)

        self.assertEqual(example.text, "你会梦见电子羊吗")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].confidence, 0.18)
        self.assertEqual(records[0].status, "pending")
        self.assertEqual(records[0].reason, "low_confidence")

    def test_event_schema_projects_registered_state_effects(self) -> None:
        examples = (
            (frame_from_roles("put_in", actor="小郭", theme="芯片", goal="托盘"), "in", "芯片", "托盘"),
            (frame_from_roles("take_out", actor="小王", theme="芯片", source="托盘"), "not_in", "芯片", "托盘"),
            (frame_from_roles("move", actor="小王", theme="托盘", goal="实验室"), "at", "托盘", "实验室"),
            (frame_from_roles("give", actor="小红", theme="药瓶", recipient="医生"), "owner", "药瓶", "医生"),
            (frame_from_roles("paint", actor="工程师", theme="笔记本", result="绿色"), "color", "笔记本", "绿色"),
            (frame_from_roles("open", actor="小王", theme="盒子", result="打开"), "access", "盒子", "打开"),
            (frame_from_roles("close", actor="小王", theme="盒子", result="关闭"), "access", "盒子", "关闭"),
            (frame_from_roles("create", actor="工程师", theme="芯片", result="存在"), "exists", "芯片", "存在"),
            (frame_from_roles("destroy", actor="工程师", theme="芯片", result="不存在"), "exists", "芯片", "不存在"),
        )

        for frame, state_name, left, right in examples:
            with self.subTest(frame_type=frame.frame_type):
                timed = with_time(frame, 1)
                states = states_for_frame_schema(timed)
                self.assertEqual(len(states), 1)
                self.assertEqual((states[0].name, states[0].left, states[0].right), (state_name, left, right))

    def test_event_schema_owns_query_role_aliases(self) -> None:
        put_in = with_time(frame_from_roles("put_in", actor="小郭", theme="芯片", goal="托盘"), 1)
        take_out = with_time(frame_from_roles("take_out", actor="小王", theme="芯片", source="托盘"), 2)

        self.assertIn("put_in", EVENT_SCHEMAS)
        self.assertTrue(frame_matches_qualifiers(put_in, ("item=芯片", "holder=托盘")))
        self.assertTrue(frame_matches_qualifiers(take_out, ("item=芯片", "source=托盘")))
        self.assertFalse(frame_matches_qualifiers(put_in, ("item=芯片", "holder=盒子")))

    def test_default_intent_analysis_does_not_guess_without_learning_signal(self) -> None:
        prediction = predict("小郭把芯片放进托盘。芯片在哪里？")

        self.assertNotIn("INTENT ", prediction.structure.linearize())

    def test_learned_intent_analyzer_adds_goal_belief_strategy_hypothesis(self) -> None:
        analyzer = InMemoryIntentAnalyzer().learn(
            "小郭把芯片放进托盘",
            Intention(
                subject="小郭",
                goal="让芯片进入托盘",
                belief="芯片还不在托盘里",
                strategy="把芯片放进托盘",
                evidence="小郭把芯片放进托盘",
                confidence=0.8,
                source="feedback",
            ),
        )
        capabilities = default_capabilities().with_intent_analyzers(analyzer)

        prediction = predict("小郭把芯片放进托盘。芯片在哪里？", capabilities)
        structure = prediction.structure.linearize()

        self.assertIn(
            "INTENT subject=小郭,goal=让芯片进入托盘,belief=芯片还不在托盘里,strategy=把芯片放进托盘,evidence=小郭把芯片放进托盘,confidence=1.00,source=feedback",
            structure,
        )
        self.assertIn("QUERY location(芯片)", structure)
        self.assertEqual(prediction.answer, "芯片在托盘里。")

    def test_intent_training_examples_can_be_loaded_from_jsonl(self) -> None:
        record = {
            "observation": "妈妈在找眼镜",
            "intention": {
                "subject": "妈妈",
                "goal": "找到眼镜",
                "belief": "妈妈不知道眼镜在哪里",
                "strategy": "在可能的位置寻找眼镜",
                "evidence": "妈妈在找眼镜",
                "confidence": 0.75,
                "source": "human_feedback",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "intent_examples.jsonl"
            path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

            analyzer = InMemoryIntentAnalyzer.from_jsonl(path)
            capabilities = default_capabilities().with_intent_analyzers(analyzer)
            prediction = predict("妈妈在找眼镜。你是谁？", capabilities)

        structure = prediction.structure.linearize()
        self.assertIn(
            "INTENT subject=妈妈,goal=找到眼镜,belief=妈妈不知道眼镜在哪里,strategy=在可能的位置寻找眼镜,evidence=妈妈在找眼镜,confidence=1.00,source=human_feedback",
            structure,
        )

    def test_intent_dataset_record_preserves_training_context(self) -> None:
        record = {
            "observation": "妈妈在找眼镜",
            "context": ["眼镜平时放在桌上"],
            "world_state": ["at(眼镜,桌上)"],
            "belief_state": ["believes(妈妈,unknown_location(眼镜))"],
            "answer": "妈妈想找到眼镜。",
            "source": "human_feedback",
            "split": "train",
            "intention": {
                "subject": "妈妈",
                "goal": "找到眼镜",
                "belief": "妈妈不知道眼镜在哪里",
                "strategy": "在可能的位置寻找眼镜",
                "confidence": 0.75,
            },
        }

        parsed = intent_record_from_dict(record)

        self.assertEqual(parsed.context, ("眼镜平时放在桌上",))
        self.assertEqual(parsed.world_state, ("at(眼镜,桌上)",))
        self.assertEqual(parsed.belief_state, ("believes(妈妈,unknown_location(眼镜))",))
        self.assertEqual(parsed.answer, "妈妈想找到眼镜。")
        self.assertEqual(parsed.intention.source, "human_feedback")

    def test_intent_feedback_can_be_appended_and_reused_as_training_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "intent_examples.jsonl"
            record = build_intent_record(
                "孩子伸手去拿杯子",
                "孩子",
                "拿到杯子",
                belief="孩子认为杯子在眼前",
                strategy="伸手抓取杯子",
                context=("杯子在桌上",),
                world_state=("at(杯子,桌上)",),
                answer="孩子想拿到杯子。",
                source="human_feedback",
            )

            append_intent_record(path, record)
            loaded_records = load_intent_jsonl(path)
            analyzer = InMemoryIntentAnalyzer.from_jsonl(path)

        self.assertEqual(len(loaded_records), 1)
        self.assertEqual(loaded_records[0].context, ("杯子在桌上",))
        self.assertEqual(loaded_records[0].answer, "孩子想拿到杯子。")
        self.assertEqual(analyzer.examples[0].intention.goal, "拿到杯子")

    def test_intent_examples_can_evaluate_an_analyzer(self) -> None:
        analyzer = InMemoryIntentAnalyzer.from_records(
            [
                {
                    "observation": "孩子伸手去拿杯子",
                    "intention": {
                        "subject": "孩子",
                        "goal": "拿到杯子",
                        "belief": "孩子认为杯子在眼前",
                        "strategy": "伸手抓取杯子",
                    },
                }
            ]
        )

        result = evaluate_intent_analyzer(analyzer, analyzer.examples)

        self.assertEqual(result.total, 1)
        self.assertEqual(result.matched, 1)
        self.assertEqual(result.accuracy, 1.0)

    def test_intent_dataset_rejects_invalid_confidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "confidence"):
            intent_record_from_dict(
                {
                    "observation": "妈妈在找眼镜",
                    "intention": {
                        "subject": "妈妈",
                        "goal": "找到眼镜",
                        "confidence": 1.5,
                    },
                }
            )

    def test_containment_move(self) -> None:
        prediction = predict("研究员把芯片放进托盘。托盘被带到实验室。芯片在哪里？")

        self.assertIn("RULE container_moves_contents", prediction.structure.linearize())
        self.assertEqual(prediction.answer, "芯片在实验室的托盘里。")

    def test_location_question_allows_different_word_order(self) -> None:
        prediction = predict("研究员把芯片放进托盘。托盘被带到实验室。芯片现在在哪里？")

        self.assertIn("QUERY location(芯片)", prediction.structure.linearize())
        self.assertEqual(prediction.answer, "芯片在实验室的托盘里。")

    def test_later_put_in_updates_current_container(self) -> None:
        prediction = predict(
            "小郭把芯片放进托盘。托盘被带到实验室。"
            "小王把芯片放进盒子。盒子被带到办公室。芯片在哪里？"
        )

        structure = prediction.structure.linearize()
        self.assertIn("REL in(芯片,盒子)", structure)
        self.assertNotIn("REL in(芯片,托盘)", structure)
        self.assertEqual(prediction.answer, "芯片在办公室的盒子里。")

    def test_later_move_updates_current_place(self) -> None:
        prediction = predict("小郭把芯片放进托盘。托盘被带到实验室。托盘被带到办公室。芯片在哪里？")

        structure = prediction.structure.linearize()
        self.assertIn("REL at(托盘,办公室)", structure)
        self.assertNotIn("REL at(托盘,实验室)", structure)
        self.assertEqual(prediction.answer, "芯片在办公室的托盘里。")

    def test_move_adverb_does_not_pollute_entity_slot(self) -> None:
        prediction = predict(
            "小郭把芯片放进托盘。小王把托盘带到了实验室。托盘又被带到办公室。"
            "你知道吗？我想知道芯片现在在哪里，可以告诉我吗？"
        )

        structure = prediction.structure.linearize()
        self.assertIn("REL at(托盘,办公室)", structure)
        self.assertNotIn("托盘又", structure)
        self.assertEqual(prediction.answer, "芯片在办公室的托盘里。")

    def test_active_move_updates_object_location(self) -> None:
        prediction = predict(
            "小郭把芯片放进托盘。小王把托盘带到了实验室。"
            "可以告诉我托盘在哪里吗，你了解嘛？你知道的话，给我说一下"
        )

        structure = prediction.structure.linearize()
        self.assertIn("REL at(托盘,实验室)", structure)
        self.assertIn("EVENT move(托盘,实验室) WITH by=小王", structure)
        self.assertIn("FRAME f3 type=move time=3", structure)
        self.assertIn("ROLE f3 actor=小王", structure)
        self.assertIn("ROLE f3 theme=托盘", structure)
        self.assertIn("ROLE f3 goal=实验室", structure)
        self.assertIn("QUERY location(托盘)", structure)
        self.assertIn("RULE object_at_place", structure)
        self.assertEqual(prediction.answer, "托盘在实验室。")

    def test_ownership_transfer(self) -> None:
        prediction = predict("小红把药瓶交给医生。现在谁拥有药瓶？")

        self.assertIn("RULE transfer_changes_owner", prediction.structure.linearize())
        self.assertEqual(prediction.answer, "医生拥有药瓶。")

    def test_later_transfer_updates_current_owner(self) -> None:
        prediction = predict("小红把药瓶交给医生。医生把药瓶交给老师。现在谁拥有药瓶？")

        structure = prediction.structure.linearize()
        self.assertIn("REL owner(药瓶,老师)", structure)
        self.assertNotIn("REL owner(药瓶,医生)", structure)
        self.assertEqual(prediction.answer, "老师拥有药瓶。")

    def test_owner_question_allows_different_word_order(self) -> None:
        prediction = predict("小红把药瓶交给医生。药瓶是谁拥有的？")

        self.assertIn("QUERY owner(药瓶)", prediction.structure.linearize())
        self.assertEqual(prediction.answer, "医生拥有药瓶。")

    def test_color_change(self) -> None:
        prediction = predict("工程师把笔记本涂成绿色。现在笔记本是什么颜色？")

        self.assertIn("RULE paint_changes_color", prediction.structure.linearize())
        self.assertEqual(prediction.answer, "笔记本是绿色。")

    def test_later_paint_updates_current_color(self) -> None:
        prediction = predict("工程师把笔记本涂成绿色。研究员把笔记本涂成黄色。现在笔记本是什么颜色？")

        structure = prediction.structure.linearize()
        self.assertIn("REL color(笔记本,黄色)", structure)
        self.assertNotIn("REL color(笔记本,绿色)", structure)
        self.assertEqual(prediction.answer, "笔记本是黄色。")

    def test_color_question_allows_different_word_order(self) -> None:
        prediction = predict("工程师把笔记本涂成绿色。笔记本颜色是什么？")

        self.assertIn("QUERY color(笔记本)", prediction.structure.linearize())
        self.assertEqual(prediction.answer, "笔记本是绿色。")

    def test_open_close_events_update_access_state(self) -> None:
        examples = (
            ("小王打开盒子。盒子是什么状态？", "EVENT open(小王,盒子) WITH result=打开", "盒子是打开状态。"),
            ("小王把盒子打开。盒子打开还是关闭？", "EVENT open(小王,盒子) WITH result=打开", "盒子是打开状态。"),
            ("盒子被小王打开。盒子是什么状态？", "EVENT open(小王,盒子) WITH result=打开", "盒子是打开状态。"),
            ("小王关闭盒子。盒子是什么状态？", "EVENT close(小王,盒子) WITH result=关闭", "盒子是关闭状态。"),
            ("小王把盒子关上。盒子打开还是关闭？", "EVENT close(小王,盒子) WITH result=关闭", "盒子是关闭状态。"),
            ("盒子被小王合上。盒子是什么状态？", "EVENT close(小王,盒子) WITH result=关闭", "盒子是关闭状态。"),
        )

        for text, event_line, answer in examples:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn(event_line, structure)
                self.assertIn("QUERY object_state(盒子,state=access)", structure)
                self.assertIn("RULE object_access_state", structure)
                self.assertEqual(prediction.answer, answer)

    def test_later_open_close_overwrites_access_state(self) -> None:
        prediction = predict("小王打开盒子。小郭把盒子关上。盒子现在是什么状态？")

        structure = prediction.structure.linearize()
        self.assertIn("REL access(盒子,关闭)", structure)
        self.assertNotIn("REL access(盒子,打开)", structure)
        self.assertEqual(prediction.answer, "盒子是关闭状态。")

    def test_create_destroy_events_update_existence_state(self) -> None:
        examples = (
            ("工程师制造芯片。芯片是否存在？", "EVENT create(工程师,芯片) WITH result=存在", "芯片存在。"),
            ("工程师把芯片制造出来。芯片是否存在？", "EVENT create(工程师,芯片) WITH result=存在", "芯片存在。"),
            ("芯片被工程师制造出来。芯片是否存在？", "EVENT create(工程师,芯片) WITH result=存在", "芯片存在。"),
            ("工程师销毁芯片。芯片是否存在？", "EVENT destroy(工程师,芯片) WITH result=不存在", "芯片不存在。"),
            ("工程师把芯片销毁。芯片是否存在？", "EVENT destroy(工程师,芯片) WITH result=不存在", "芯片不存在。"),
            ("芯片被工程师销毁。芯片是否存在？", "EVENT destroy(工程师,芯片) WITH result=不存在", "芯片不存在。"),
        )

        for text, event_line, answer in examples:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn(event_line, structure)
                self.assertIn("QUERY existence(芯片)", structure)
                self.assertEqual(prediction.answer, answer)

    def test_polar_existence_location_and_contents_queries(self) -> None:
        examples = (
            (
                "小郭把芯片放进托盘。芯片存在吗？",
                "QUERY polar_existence(芯片)",
                "是，芯片存在。",
            ),
            (
                "小郭把芯片放进托盘。芯片在托盘里吗？",
                "QUERY polar_location(芯片,expected=托盘,kind=in)",
                "是，芯片在托盘里。",
            ),
            (
                "小郭把芯片放进托盘。托盘被带到实验室。芯片在实验室吗？",
                "QUERY polar_location(芯片,expected=实验室,kind=at)",
                "是，芯片在实验室。",
            ),
            (
                "小郭把芯片放进托盘。托盘被带到实验室。实验室里有芯片吗？",
                "QUERY polar_contents(实验室,item=芯片)",
                "是，实验室里有芯片。",
            ),
        )

        for text, query_line, answer in examples:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn(query_line, structure)
                self.assertEqual(prediction.answer, answer)

    def test_polar_query_surface_markers_normalize_to_existing_query_types(self) -> None:
        examples = (
            (
                "小郭把芯片放进托盘。芯片是不是在托盘里面？",
                "QUERY polar_location(芯片,expected=托盘,kind=in)",
                "是，芯片在托盘里。",
            ),
            (
                "小郭把芯片放进托盘。托盘被带到实验室。实验室里面有没有芯片？",
                "QUERY polar_contents(实验室,item=芯片)",
                "是，实验室里有芯片。",
            ),
            (
                "工程师制造芯片。芯片是不是存在？",
                "QUERY polar_existence(芯片)",
                "是，芯片存在。",
            ),
            (
                "小郭把芯片放进托盘。小王把药瓶放进托盘。芯片和药瓶是不是在同一个位置？",
                "QUERY same_location(芯片和药瓶,left=芯片,right=药瓶)",
                "是，芯片和药瓶在同一个地方。",
            ),
        )

        for text, query_line, answer in examples:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn(query_line, structure)
                self.assertEqual(prediction.answer, answer)

    def test_polar_queries_return_negative_and_unknown_answers(self) -> None:
        examples = (
            ("工程师销毁芯片。芯片存在吗？", "QUERY polar_existence(芯片)", "不是，芯片不存在。"),
            ("小郭把芯片放进托盘。芯片在盒子里吗？", "QUERY polar_location(芯片,expected=盒子,kind=in)", "不是，芯片在托盘里。"),
            ("小王打开盒子。盒子里有芯片吗？", "QUERY polar_contents(盒子,item=芯片)", "不知道盒子里有没有芯片。"),
        )

        for text, query_line, answer in examples:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn(query_line, structure)
                self.assertEqual(prediction.answer, answer)

    def test_same_location_query_uses_shared_place_or_container_key(self) -> None:
        examples = (
            (
                "小郭把芯片放进托盘。小王把药瓶放进托盘。芯片和药瓶在同一个地方吗？",
                "QUERY same_location(芯片和药瓶,left=芯片,right=药瓶)",
                "是，芯片和药瓶在同一个地方。",
            ),
            (
                "小郭把芯片放进托盘。托盘被带到实验室。小王把药瓶放进盒子。盒子被带到实验室。芯片和药瓶在同一个地方吗？",
                "QUERY same_location(芯片和药瓶,left=芯片,right=药瓶)",
                "是，芯片和药瓶在同一个地方。",
            ),
            (
                "小郭把芯片放进托盘。托盘被带到实验室。小王把药瓶放进盒子。盒子被带到办公室。芯片和药瓶在同一个地方吗？",
                "QUERY same_location(芯片和药瓶,left=芯片,right=药瓶)",
                "不是，芯片在实验室的托盘里，药瓶在办公室的盒子里。",
            ),
        )

        for text, query_line, answer in examples:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn(query_line, structure)
                self.assertEqual(prediction.answer, answer)

    def test_same_location_query_returns_unknown_when_one_side_is_unknown(self) -> None:
        prediction = predict("小王打开盒子。芯片和药瓶在同一个地方吗？")

        structure = prediction.structure.linearize()
        self.assertIn("QUERY same_location(芯片和药瓶,left=芯片,right=药瓶)", structure)
        self.assertEqual(prediction.answer, "不知道芯片和药瓶是不是在同一个地方。")

    def test_destroy_clears_current_location_and_attributes(self) -> None:
        prediction = predict("小郭把芯片放进托盘。工程师把芯片涂成绿色。工程师销毁芯片。芯片在哪里？")

        structure = prediction.structure.linearize()
        self.assertIn("REL exists(芯片,不存在)", structure)
        self.assertNotIn("REL in(芯片,托盘)", structure)
        self.assertNotIn("REL color(芯片,绿色)", structure)
        self.assertIn("RULE object_not_exists", structure)
        self.assertEqual(prediction.answer, "芯片不存在。")

    def test_destroyed_item_is_removed_from_contents_closure(self) -> None:
        prediction = predict("小郭把芯片放进托盘。托盘被带到实验室。工程师销毁芯片。实验室里有什么？")

        structure = prediction.structure.linearize()
        self.assertIn("REL exists(芯片,不存在)", structure)
        self.assertNotIn("REL in(芯片,托盘)", structure)
        self.assertEqual(prediction.answer, "实验室里至少有托盘。")

    def test_later_state_can_restore_destroyed_object_for_ordered_correction(self) -> None:
        prediction = predict("工程师销毁芯片。小郭把芯片放进托盘。芯片是否存在？")

        structure = prediction.structure.linearize()
        self.assertIn("REL in(芯片,托盘)", structure)
        self.assertNotIn("REL exists(芯片,不存在)", structure)
        self.assertEqual(prediction.answer, "芯片存在。")

    def test_existence_claims_can_conflict_with_fact(self) -> None:
        destroyed = predict("工程师销毁芯片。小王说芯片存在。有没有矛盾？")
        existing = predict("工程师制造芯片。小王说芯片不存在。有没有矛盾？")

        self.assertEqual(destroyed.answer, "存在矛盾：小王说芯片存在，但事实是芯片不存在。")
        self.assertEqual(existing.answer, "存在矛盾：小王说芯片不存在，但事实是芯片存在。")

    def test_handler_question_reuses_extracted_structure(self) -> None:
        questions = (
            "谁拿的芯片？",
            "芯片是谁拿的？",
            "现在芯片是谁拿了？",
        )

        for question in questions:
            with self.subTest(question=question):
                prediction = predict(f"小郭把芯片放进托盘。托盘被带到实验室。{question}")

                structure = prediction.structure.linearize()
                self.assertIn("EVENT handle(小郭,芯片)", structure)
                self.assertIn("QUERY actor_for_item(芯片)", structure)
                self.assertIn("RULE actor_handles_item", structure)
                self.assertEqual(prediction.answer, "小郭拿的芯片。")

    def test_handler_question_uses_latest_handler(self) -> None:
        prediction = predict("小郭把芯片放进托盘。小王把芯片放进盒子。谁拿的芯片？")

        structure = prediction.structure.linearize()
        self.assertIn("EVENT handle(小郭,芯片)", structure)
        self.assertIn("EVENT handle(小王,芯片)", structure)
        self.assertEqual(prediction.answer, "小王拿的芯片。")

    def test_event_actor_question_reuses_event_structure(self) -> None:
        questions = (
            "谁把芯片放进托盘？",
            "芯片是谁放进托盘的？",
            "芯片被谁放进托盘的？",
            "芯片被谁放进托盘的了？",
            "谁把芯片放到托盘里面的？",
            "芯片是谁放到托盘里边的？",
            "芯片被谁放入托盘里的？",
            "现在芯片被谁放进托盘的了？",
            "我想知道芯片被谁放进托盘了的？",
            "我想问芯片被谁放进托盘了的？",
            "请问我想知道芯片被谁放进托盘了的？",
            "你可以告诉我芯片被谁放进托盘了的，我想知道下？",
            "我想知道下你可以告诉我芯片被谁放进托盘了的吗？",
            "你知道吗？，你知道的话，可以告诉我芯片被谁放进托盘了的，我想知道下",
        )

        for question in questions:
            with self.subTest(question=question):
                prediction = predict(f"小郭把芯片放进托盘。托盘被带到实验室。{question}")

                structure = prediction.structure.linearize()
                self.assertIn("EVENT put_in(小郭,芯片)", structure)
                self.assertIn("REL in(芯片,托盘)", structure)
                self.assertIn("QUERY actor_for_event(put_in,item=芯片,holder=托盘)", structure)
                self.assertIn("RULE event_actor_matches", structure)
                self.assertEqual(prediction.answer, "小郭把芯片放进托盘。")

    def test_latest_event_actor_question_reuses_event_history(self) -> None:
        questions = (
            "最后谁把芯片放进托盘？",
            "谁最后把芯片放进托盘？",
            "芯片最后是谁放进托盘的？",
            "芯片被最后谁放进托盘的？",
            "最近谁把芯片从托盘里取出？",
            "芯片最近是谁从托盘里取出的？",
        )

        for question in questions:
            with self.subTest(question=question):
                prediction = predict(
                    "小郭把芯片放进托盘。小王把芯片放进托盘。小李把芯片从托盘里取出。"
                    f"{question}"
                )
                structure = prediction.structure.linearize()
                if "取出" in question:
                    self.assertIn("QUERY latest_actor_for_event(take_out,item=芯片,source=托盘)", structure)
                    self.assertIn("RULE latest_event_actor_matches", structure)
                    self.assertEqual(prediction.answer, "最后是小李把芯片从托盘取出。")
                else:
                    self.assertIn("QUERY latest_actor_for_event(put_in,item=芯片,holder=托盘)", structure)
                    self.assertIn("RULE latest_event_actor_matches", structure)
                    self.assertEqual(prediction.answer, "最后是小王把芯片放进托盘。")

    def test_earliest_event_actor_question_reuses_event_history(self) -> None:
        questions = (
            "最先谁把芯片放进托盘？",
            "谁最先把芯片放进托盘？",
            "芯片最先是谁放进托盘的？",
            "芯片被最先谁放进托盘的？",
            "最先谁把芯片从托盘里取出？",
            "芯片最先是谁从托盘里取出的？",
        )

        for question in questions:
            with self.subTest(question=question):
                prediction = predict(
                    "小郭把芯片放进托盘。小王把芯片放进托盘。小李把芯片从托盘里取出。"
                    f"{question}"
                )
                structure = prediction.structure.linearize()
                if "取出" in question:
                    self.assertIn("QUERY earliest_actor_for_event(take_out,item=芯片,source=托盘)", structure)
                    self.assertIn("RULE earliest_event_actor_matches", structure)
                    self.assertEqual(prediction.answer, "最先是小李把芯片从托盘取出。")
                else:
                    self.assertIn("QUERY earliest_actor_for_event(put_in,item=芯片,holder=托盘)", structure)
                    self.assertIn("RULE earliest_event_actor_matches", structure)
                    self.assertEqual(prediction.answer, "最先是小郭把芯片放进托盘。")

    def test_earliest_and_latest_event_queries_differentiate_order(self) -> None:
        prediction = predict("小郭把芯片放进托盘。小王把芯片放进托盘。最先谁把芯片放进托盘？最后谁把芯片放进托盘？")

        structure = prediction.structure.linearize()
        self.assertIn("QUERY compound(multi)", structure)
        self.assertIn("SUBQUERY earliest_actor_for_event(put_in,item=芯片,holder=托盘)", structure)
        self.assertIn("SUBQUERY latest_actor_for_event(put_in,item=芯片,holder=托盘)", structure)
        self.assertEqual(prediction.answer, "最先是小郭把芯片放进托盘；最后是小王把芯片放进托盘。")

    def test_event_actor_question_can_ask_historical_event_after_state_changes(self) -> None:
        prediction = predict("小郭把芯片放进托盘。小王把芯片放进盒子。谁把芯片放进托盘？")

        structure = prediction.structure.linearize()
        self.assertIn("REL in(芯片,盒子)", structure)
        self.assertIn("EVENT put_in(小郭,芯片) WITH holder=托盘", structure)
        self.assertIn("EVENT put_in(小王,芯片) WITH holder=盒子", structure)
        self.assertIn("FRAME f1 type=put_in time=1", structure)
        self.assertIn("ROLE f1 actor=小郭", structure)
        self.assertIn("ROLE f1 theme=芯片", structure)
        self.assertIn("ROLE f1 goal=托盘", structure)
        self.assertEqual(prediction.answer, "小郭把芯片放进托盘。")

    def test_put_in_statement_and_query_normalize_container_surface_forms(self) -> None:
        prediction = predict("小郭把芯片放到托盘里面。小王把芯片放入盒子里。谁把芯片放到盒子里面的？")

        structure = prediction.structure.linearize()
        self.assertIn("REL in(芯片,盒子)", structure)
        self.assertIn("EVENT put_in(小郭,芯片) WITH holder=托盘", structure)
        self.assertIn("EVENT put_in(小王,芯片) WITH holder=盒子", structure)
        self.assertIn("QUERY actor_for_event(put_in,item=芯片,holder=盒子)", structure)
        self.assertEqual(prediction.answer, "小王把芯片放进盒子。")

    def test_event_actor_query_uses_role_slots_for_chatty_surface_forms(self) -> None:
        examples = (
            (
                "小郭把芯片放进托盘。现在到底是谁把芯片放进了托盘？",
                "QUERY actor_for_event(put_in,item=芯片,holder=托盘)",
                "小郭把芯片放进托盘。",
            ),
            (
                "小郭把芯片放进托盘。芯片到底被谁放进了托盘里面？",
                "QUERY actor_for_event(put_in,item=芯片,holder=托盘)",
                "小郭把芯片放进托盘。",
            ),
            (
                "小郭把芯片放进托盘。你知道吗？你知道的话，可以告诉我芯片到底被谁放进托盘里面？",
                "QUERY actor_for_event(put_in,item=芯片,holder=托盘)",
                "小郭把芯片放进托盘。",
            ),
            (
                "小郭把芯片放进托盘。小王从托盘里取出芯片。谁从托盘里面拿出来芯片？",
                "QUERY actor_for_event(take_out,item=芯片,source=托盘)",
                "小王把芯片从托盘取出。",
            ),
        )

        for text, query_line, answer in examples:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn(query_line, structure)
                self.assertIn("RULE event_actor_matches", structure)
                self.assertEqual(prediction.answer, answer)

    def test_take_out_removes_current_container_state(self) -> None:
        prediction = predict("小郭把芯片放进托盘。小王把芯片从托盘里取出来。芯片在哪里？")

        structure = prediction.structure.linearize()
        self.assertNotIn("REL in(芯片,托盘)", structure)
        self.assertIn("EVENT take_out(小王,芯片) WITH source=托盘", structure)
        self.assertIn("FRAME f3 type=take_out time=3", structure)
        self.assertIn("ROLE f3 actor=小王", structure)
        self.assertIn("ROLE f3 theme=芯片", structure)
        self.assertIn("ROLE f3 source=托盘", structure)
        self.assertIn("RULE location_unknown", structure)
        self.assertEqual(prediction.answer, "不知道芯片在哪里。")

    def test_take_out_statement_allows_surface_variants(self) -> None:
        examples = (
            "小王把芯片从托盘里取出",
            "小王把芯片从托盘里面拿出来",
            "小王从托盘里取出芯片",
            "芯片被小王从托盘里拿出",
            "芯片从托盘里被取出",
        )

        for statement in examples:
            with self.subTest(statement=statement):
                prediction = predict(f"小郭把芯片放进托盘。{statement}。托盘里有什么？")
                structure = prediction.structure.linearize()
                self.assertNotIn("REL in(芯片,托盘)", structure)
                self.assertIn("QUERY contents(托盘)", structure)
                self.assertEqual(prediction.answer, "不知道托盘里有什么。")

    def test_take_out_actor_query_reuses_event_structure(self) -> None:
        questions = (
            "谁把芯片从托盘里取出来的？",
            "芯片是谁从托盘里面拿出来的？",
            "芯片被谁从托盘里取出的？",
        )

        for question in questions:
            with self.subTest(question=question):
                prediction = predict(f"小郭把芯片放进托盘。小王把芯片从托盘里取出。{question}")
                structure = prediction.structure.linearize()
                self.assertIn("QUERY actor_for_event(take_out,item=芯片,source=托盘)", structure)
                self.assertIn("RULE event_actor_matches", structure)
                self.assertEqual(prediction.answer, "小王把芯片从托盘取出。")

    def test_negation_and_correction_update_current_state(self) -> None:
        negated = predict("小郭把芯片放进托盘。其实芯片不在托盘里。芯片在哪里？")
        negated_structure = negated.structure.linearize()
        self.assertNotIn("REL in(芯片,托盘)", negated_structure)
        self.assertEqual(negated.answer, "不知道芯片在哪里。")

        corrected = predict("小郭把芯片放进托盘。其实芯片不在托盘里。其实芯片在盒子里。芯片在哪里？")
        corrected_structure = corrected.structure.linearize()
        self.assertNotIn("REL in(芯片,托盘)", corrected_structure)
        self.assertIn("REL in(芯片,盒子)", corrected_structure)
        self.assertEqual(corrected.answer, "芯片在盒子里。")

    def test_holder_negation_removes_matching_content(self) -> None:
        prediction = predict("小郭把芯片放进托盘。托盘里没有芯片。托盘里有什么？")

        structure = prediction.structure.linearize()
        self.assertNotIn("REL in(芯片,托盘)", structure)
        self.assertIn("EVENT not_in(芯片,托盘)", structure)
        self.assertEqual(prediction.answer, "不知道托盘里有什么。")

    def test_inline_correction_replaces_old_container_state(self) -> None:
        questions = (
            "小郭把芯片放进托盘。芯片不在托盘里而在盒子里。芯片在哪里？",
            "小郭把芯片放进托盘。芯片不是在托盘里，是在盒子里。芯片在哪里？",
        )

        for text in questions:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertNotIn("REL in(芯片,托盘)", structure)
                self.assertIn("REL in(芯片,盒子)", structure)
                self.assertIn("EVENT not_in(芯片,托盘)", structure)
                self.assertIn("EVENT be_in(芯片,盒子)", structure)
                self.assertEqual(prediction.answer, "芯片在盒子里。")

    def test_initial_location_query_uses_frame_history(self) -> None:
        prediction = predict("小郭把芯片放进托盘。小王把芯片放进盒子。芯片最开始在哪里？")

        structure = prediction.structure.linearize()
        self.assertIn("QUERY initial_location(芯片)", structure)
        self.assertIn("RULE initial_location_found", structure)
        self.assertEqual(prediction.answer, "芯片最开始在托盘里。")

    def test_initial_location_can_include_place_when_known_at_that_time(self) -> None:
        prediction = predict("托盘被带到实验室。小郭把芯片放进托盘。小王把芯片放进盒子。芯片一开始在哪里？")

        self.assertEqual(prediction.answer, "芯片最开始在实验室的托盘里。")

    def test_latest_actor_query_uses_latest_historical_handler(self) -> None:
        prediction = predict("小郭把芯片放进托盘。小王把芯片放进盒子。最后谁处理过芯片？")

        structure = prediction.structure.linearize()
        self.assertIn("QUERY latest_actor_for_item(芯片)", structure)
        self.assertIn("RULE latest_actor_handles_item", structure)
        self.assertEqual(prediction.answer, "最后是小王处理过芯片。")

    def test_location_before_actor_action_replays_state_before_frame(self) -> None:
        prediction = predict("小郭把芯片放进托盘。小王把芯片放进盒子。小王操作之前，芯片在哪里？")

        structure = prediction.structure.linearize()
        self.assertIn("QUERY location_before_actor_action(芯片,actor=小王)", structure)
        self.assertIn("RULE location_before_actor_action_found", structure)
        self.assertEqual(prediction.answer, "小王操作之前，芯片在托盘里。")

    def test_location_before_and_after_event_query_uses_temporal_frame_replay(self) -> None:
        examples = (
            (
                "小郭把芯片放进托盘。小王把托盘带到实验室。小王把托盘带到实验室之前，芯片在哪里？",
                "QUERY location_before_event(芯片,anchor=小王把托盘带到实验室,event=move,actor=小王,theme=托盘,goal=实验室)",
                "在小王把托盘带到实验室之前，芯片在托盘里。",
            ),
            (
                "小郭把芯片放进托盘。小王把托盘带到实验室。在小王把托盘带到实验室之前，芯片在哪里？",
                "QUERY location_before_event(芯片,anchor=小王把托盘带到实验室,event=move,actor=小王,theme=托盘,goal=实验室)",
                "在小王把托盘带到实验室之前，芯片在托盘里。",
            ),
            (
                "小郭把芯片放进托盘。小王把托盘带到实验室。小王把托盘带到实验室之后，芯片在哪里？",
                "QUERY location_after_event(芯片,anchor=小王把托盘带到实验室,event=move,actor=小王,theme=托盘,goal=实验室)",
                "在小王把托盘带到实验室之后，芯片在实验室的托盘里。",
            ),
            (
                "小郭把芯片放进托盘。小王把托盘带到实验室。小王把托盘带到实验室之后，托盘在哪里？",
                "QUERY location_after_event(托盘,anchor=小王把托盘带到实验室,event=move,actor=小王,theme=托盘,goal=实验室)",
                "在小王把托盘带到实验室之后，托盘在实验室。",
            ),
        )

        for text, query_line, answer in examples:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn(query_line, structure)
                self.assertIn("RULE location_before_event_found" if "before" in query_line else "RULE location_after_event_found", structure)
                self.assertEqual(prediction.answer, answer)

    def test_location_after_event_can_track_take_out_replay(self) -> None:
        prediction = predict("小郭把芯片放进托盘。小王把芯片从托盘里取出之后，芯片在哪里？")

        structure = prediction.structure.linearize()
        self.assertIn(
            "QUERY location_after_event(芯片,anchor=小王把芯片从托盘里取出,event=take_out,actor=小王,theme=芯片,source=托盘)",
            structure,
        )
        self.assertIn("RULE location_after_event_unknown", structure)
        self.assertEqual(prediction.answer, "不知道芯片在小王把芯片从托盘里取出之后在哪里。")

    def test_temporal_contents_query_replays_container_state_before_and_after_event(self) -> None:
        examples = (
            (
                "小郭把芯片放进托盘。小王把托盘带到实验室。小王把托盘带到实验室之前，托盘里有什么？",
                "QUERY contents_before_event(托盘,anchor=小王把托盘带到实验室,event=move,actor=小王,theme=托盘,goal=实验室)",
                "在小王把托盘带到实验室之前，托盘里至少有芯片。",
            ),
            (
                "小郭把芯片放进托盘。小王把托盘带到实验室。小王把托盘带到实验室之后，托盘里有什么？",
                "QUERY contents_after_event(托盘,anchor=小王把托盘带到实验室,event=move,actor=小王,theme=托盘,goal=实验室)",
                "在小王把托盘带到实验室之后，托盘里至少有芯片。",
            ),
        )

        for text, query_line, answer in examples:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn(query_line, structure)
                self.assertIn(
                    "RULE contents_before_event_found" if "before" in query_line else "RULE contents_after_event_found",
                    structure,
                )
                self.assertEqual(prediction.answer, answer)

    def test_temporal_contents_query_can_become_unknown_after_removal(self) -> None:
        prediction = predict("小郭把芯片放进托盘。小王把芯片从托盘里取出。小王把芯片从托盘里取出之后，托盘里有什么？")

        structure = prediction.structure.linearize()
        self.assertIn(
            "QUERY contents_after_event(托盘,anchor=小王把芯片从托盘里取出,event=take_out,actor=小王,theme=芯片,source=托盘)",
            structure,
        )
        self.assertIn("RULE contents_after_event_unknown", structure)
        self.assertEqual(prediction.answer, "不知道托盘在小王把芯片从托盘里取出之后有什么。")

    def test_historical_location_still_works_after_take_out(self) -> None:
        prediction = predict("小郭把芯片放进托盘。小王把芯片从托盘里取出。芯片最开始在哪里？")

        structure = prediction.structure.linearize()
        self.assertNotIn("REL in(芯片,托盘)", structure)
        self.assertIn("RULE initial_location_found", structure)
        self.assertEqual(prediction.answer, "芯片最开始在托盘里。")

    def test_events_after_put_in_query_uses_event_anchor(self) -> None:
        prediction = predict("小郭把芯片放进盒子。盒子被带到仓库。小王把芯片从盒子里取出。芯片被放进盒子之后发生了什么？")

        structure = prediction.structure.linearize()
        self.assertIn("QUERY events_after_event(put_in,item=芯片,holder=盒子)", structure)
        self.assertIn("RULE events_after_event", structure)
        self.assertEqual(prediction.answer, "之后发生了：盒子被带到仓库；小王把芯片从盒子取出。")

    def test_events_after_take_out_query_uses_event_anchor(self) -> None:
        prediction = predict("小郭把芯片放进盒子。小王把芯片从盒子里取出。小王把芯片放进托盘。芯片从盒子里取出之后发生了什么？")

        structure = prediction.structure.linearize()
        self.assertIn("QUERY events_after_event(take_out,item=芯片,source=盒子)", structure)
        self.assertIn("RULE events_after_event", structure)
        self.assertEqual(prediction.answer, "之后发生了：小王把芯片放进托盘。")

    def test_place_contents_question_uses_world_state_closure(self) -> None:
        questions = (
            "现在实验室里至少有什么？",
            "实验室里有什么？",
        )

        for question in questions:
            with self.subTest(question=question):
                prediction = predict(f"小郭把芯片放进托盘。托盘被带到实验室。{question}")

                structure = prediction.structure.linearize()
                self.assertIn("REL in(芯片,托盘)", structure)
                self.assertIn("EVENT move(托盘,实验室)", structure)
                self.assertIn("QUERY contents(实验室)", structure)
                self.assertIn("RULE holder_contains_things", structure)
                self.assertEqual(prediction.answer, "实验室里至少有托盘和芯片。")

    def test_container_contents_question_uses_same_closure(self) -> None:
        prediction = predict("小郭把芯片放进托盘。托盘被带到实验室。托盘里有什么？")

        structure = prediction.structure.linearize()
        self.assertIn("QUERY contents(托盘)", structure)
        self.assertIn("RULE holder_contains_things", structure)
        self.assertEqual(prediction.answer, "托盘里至少有芯片。")

    def test_contents_question_uses_current_state(self) -> None:
        prediction = predict(
            "小郭把芯片放进托盘。托盘被带到实验室。"
            "小王把芯片放进盒子。盒子被带到办公室。办公室里有什么？"
        )

        structure = prediction.structure.linearize()
        self.assertIn("REL in(芯片,盒子)", structure)
        self.assertIn("REL at(盒子,办公室)", structure)
        self.assertEqual(prediction.answer, "办公室里至少有盒子和芯片。")

    def test_contents_query_target_uses_entity_boundary_correction(self) -> None:
        prediction = predict(
            "小郭把芯片放进托盘。小王把芯片放进盒子。盒子被带到仓库。"
            "你可以告诉我仓库里有什么吗？"
        )

        structure = prediction.structure.linearize()
        self.assertIn("QUERY contents(仓库)", structure)
        self.assertEqual(prediction.answer, "仓库里至少有盒子和芯片。")

    def test_complex_historical_event_query_after_current_state_changes(self) -> None:
        prediction = predict(
            "小郭把芯片放进托盘。小王把芯片放进盒子。盒子被带到仓库。"
            "你知道吗？，你知道的话，可以告诉我芯片被谁放进托盘了的，我想知道下"
        )

        structure = prediction.structure.linearize()
        self.assertIn("REL in(芯片,盒子)", structure)
        self.assertIn("FRAME f1 type=put_in time=1", structure)
        self.assertIn("QUERY actor_for_event(put_in,item=芯片,holder=托盘)", structure)
        self.assertEqual(prediction.answer, "小郭把芯片放进托盘。")

    def test_filler_question_fragment_does_not_block_real_query(self) -> None:
        prediction = predict("小郭把芯片放进托盘。托盘被带到实验室。你知道吗？芯片在哪里？")

        structure = prediction.structure.linearize()
        self.assertIn("QUERY location(芯片)", structure)
        self.assertNotIn("QUERY location(你知道)", structure)
        self.assertEqual(prediction.answer, "芯片在实验室的托盘里。")

    def test_complex_contents_query_after_repeated_moves(self) -> None:
        prediction = predict(
            "小郭把芯片放进托盘。小王把托盘带到了实验室。托盘被带到办公室。"
            "可以告诉我办公室里有什么吗？"
        )

        structure = prediction.structure.linearize()
        self.assertIn("REL at(托盘,办公室)", structure)
        self.assertIn("QUERY contents(办公室)", structure)
        self.assertEqual(prediction.answer, "办公室里至少有托盘和芯片。")

    def test_nested_container_location_uses_recursive_closure(self) -> None:
        prediction = predict(
            "小郭把芯片放进小盒子。小王把小盒子放进大盒子。大盒子被带到实验室。"
            "芯片在哪里？"
        )

        structure = prediction.structure.linearize()
        self.assertIn("REL in(芯片,小盒子)", structure)
        self.assertIn("REL in(小盒子,大盒子)", structure)
        self.assertIn("REL at(大盒子,实验室)", structure)
        self.assertIn("RULE container_moves_contents", structure)
        self.assertEqual(prediction.answer, "芯片在实验室的大盒子里的小盒子里。")

    def test_nested_contents_query_uses_recursive_closure(self) -> None:
        prediction = predict(
            "小郭把芯片放进小盒子。小王把小盒子放进大盒子。大盒子被带到实验室。"
            "实验室里有什么？"
        )

        self.assertEqual(prediction.answer, "实验室里至少有大盒子和小盒子和芯片。")

    def test_contents_except_query_filters_named_item(self) -> None:
        prediction = predict("小郭把芯片放进托盘。托盘被带到实验室。实验室里除了托盘还有什么？")

        structure = prediction.structure.linearize()
        self.assertIn("QUERY contents_except(实验室,exclude=托盘)", structure)
        self.assertIn("RULE holder_contains_except", structure)
        self.assertEqual(prediction.answer, "实验室里除了托盘还有芯片。")

    def test_count_query_uses_contents_closure_for_places_and_containers(self) -> None:
        examples = (
            ("小郭把芯片放进托盘。托盘被带到实验室。实验室里有几个东西？", "QUERY count(实验室)", "实验室里至少有2个已知物品。"),
            ("小郭把芯片放进托盘。托盘里有几个物品？", "QUERY count(托盘)", "托盘里至少有1个已知物品。"),
            ("小郭把芯片放进托盘。托盘被带到实验室。实验室里数量是多少？", "QUERY count(实验室)", "实验室里至少有2个已知物品。"),
        )

        for text, query_line, answer in examples:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn(query_line, structure)
                self.assertIn("RULE count_known_contents", structure)
                self.assertEqual(prediction.answer, answer)

    def test_count_query_uses_nested_closure(self) -> None:
        prediction = predict(
            "小郭把芯片放进小盒子。小王把小盒子放进大盒子。大盒子被带到实验室。"
            "实验室里有几个物品？"
        )

        structure = prediction.structure.linearize()
        self.assertIn("QUERY count(实验室)", structure)
        self.assertEqual(prediction.answer, "实验室里至少有3个已知物品。")

    def test_count_query_filters_destroyed_objects(self) -> None:
        prediction = predict("小郭把芯片放进托盘。托盘被带到实验室。工程师销毁芯片。实验室里有几个东西？")

        structure = prediction.structure.linearize()
        self.assertIn("REL exists(芯片,不存在)", structure)
        self.assertIn("QUERY count(实验室)", structure)
        self.assertEqual(prediction.answer, "实验室里至少有1个已知物品。")

    def test_count_query_reports_no_known_items_for_empty_holder(self) -> None:
        prediction = predict("小王打开盒子。盒子里有几个东西？")

        structure = prediction.structure.linearize()
        self.assertIn("QUERY count(盒子)", structure)
        self.assertEqual(prediction.answer, "盒子里没有已知物品。")

    def test_compare_count_query_uses_current_contents_closure(self) -> None:
        examples = (
            (
                "小郭把芯片放进托盘。托盘被带到实验室。盒子被带到办公室。实验室和办公室哪个地方东西更多？",
                "QUERY compare_count(实验室和办公室,left=实验室,right=办公室)",
                "实验室里的已知物品更多，至少有2个；办公室里至少有1个。",
            ),
            (
                "托盘被带到实验室。小郭把芯片放进盒子。盒子被带到办公室。实验室和办公室哪里东西更多？",
                "QUERY compare_count(实验室和办公室,left=实验室,right=办公室)",
                "办公室里的已知物品更多，至少有2个；实验室里至少有1个。",
            ),
            (
                "托盘被带到实验室。盒子被带到办公室。实验室和办公室里的已知物品一样多吗？",
                "QUERY compare_count(实验室和办公室,left=实验室,right=办公室)",
                "实验室和办公室里的已知物品一样多，都是1个。",
            ),
        )

        for text, query_line, answer in examples:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn(query_line, structure)
                self.assertIn("RULE compare_count_known_contents", structure)
                self.assertEqual(prediction.answer, answer)

    def test_compare_count_query_filters_destroyed_nested_objects(self) -> None:
        prediction = predict(
            "小郭把芯片放进托盘。托盘被带到实验室。小王把药瓶放进盒子。盒子被带到办公室。"
            "工程师销毁药瓶。实验室和办公室哪个地方东西更多？"
        )

        structure = prediction.structure.linearize()
        self.assertIn("REL exists(药瓶,不存在)", structure)
        self.assertIn("QUERY compare_count(实验室和办公室,left=实验室,right=办公室)", structure)
        self.assertEqual(prediction.answer, "实验室里的已知物品更多，至少有2个；办公室里至少有1个。")

    def test_places_visited_query_replays_location_history(self) -> None:
        prediction = predict(
            "小郭把芯片放进托盘。托盘被带到实验室。托盘被带到办公室。芯片经过了哪些地方？"
        )

        structure = prediction.structure.linearize()
        self.assertIn("QUERY places_visited(芯片)", structure)
        self.assertIn("RULE places_visited", structure)
        self.assertEqual(prediction.answer, "芯片经过了实验室和办公室。")

    def test_actions_by_actors_query_uses_event_frames(self) -> None:
        prediction = predict(
            "小郭把芯片放进托盘。小王把托盘带到了实验室。小王把芯片从托盘里取出。"
            "小郭和小王分别做了什么？"
        )

        structure = prediction.structure.linearize()
        self.assertIn("QUERY actions_by_actors(小郭和小王,actors=小郭|小王)", structure)
        self.assertIn("RULE actor_actions", structure)
        self.assertEqual(
            prediction.answer,
            "小郭把芯片放进托盘；小王把托盘带到实验室，把芯片从托盘取出。",
        )

    def test_single_actor_actions_query_reuses_event_frames(self) -> None:
        prediction = predict("小郭把芯片放进托盘。小王把芯片放进盒子。小郭做了什么？")

        structure = prediction.structure.linearize()
        self.assertIn("QUERY actions_by_actors(小郭,actors=小郭)", structure)
        self.assertIn("RULE actor_actions", structure)
        self.assertEqual(prediction.answer, "小郭把芯片放进托盘。")

    def test_inventory_query_lists_current_owned_items(self) -> None:
        prediction = predict("小红把药瓶交给医生。小郭把芯片交给医生。现在每个人手里有什么？")

        structure = prediction.structure.linearize()
        self.assertIn("QUERY inventories(person)", structure)
        self.assertIn("RULE owner_inventories", structure)
        self.assertEqual(prediction.answer, "医生手里有药瓶和芯片。")

    def test_typed_demonstrative_query_resolves_to_known_entity(self) -> None:
        prediction = predict("小郭把芯片放进托盘。托盘被带到实验室。这个芯片在哪里？")

        structure = prediction.structure.linearize()
        self.assertIn("QUERY location(芯片)", structure)
        self.assertEqual(prediction.answer, "芯片在实验室的托盘里。")

    def test_pronoun_query_resolves_to_latest_non_place_entity(self) -> None:
        prediction = predict("小郭把芯片放进托盘。小王把芯片放进盒子。它在哪里？")

        structure = prediction.structure.linearize()
        self.assertIn("QUERY location(盒子)", structure)
        self.assertEqual(prediction.answer, "不知道盒子在哪里。")

    def test_typed_demonstrative_statement_can_update_state(self) -> None:
        prediction = predict("小郭把芯片放进托盘。小王把这个芯片从托盘里取出来。这个芯片在哪里？")

        structure = prediction.structure.linearize()
        self.assertIn("EVENT take_out(小王,芯片) WITH source=托盘", structure)
        self.assertIn("QUERY location(芯片)", structure)
        self.assertEqual(prediction.answer, "不知道芯片在哪里。")

    def test_place_pronoun_resolves_for_contents_query(self) -> None:
        prediction = predict("小郭把芯片放进托盘。托盘被带到实验室。这里有什么？")

        structure = prediction.structure.linearize()
        self.assertIn("QUERY contents(实验室)", structure)
        self.assertEqual(prediction.answer, "实验室里至少有托盘和芯片。")

    def test_relative_pronouns_resolve_to_previous_two_salient_entities(self) -> None:
        prediction = predict(
            "小郭把芯片放进托盘。托盘被带到实验室。小王把药瓶放进盒子。盒子被带到办公室。前者在哪里，后者在哪里？"
        )

        structure = prediction.structure.linearize()
        self.assertIn("QUERY compound(multi)", structure)
        self.assertIn("SUBQUERY location(药瓶)", structure)
        self.assertIn("SUBQUERY location(盒子)", structure)
        self.assertEqual(prediction.answer, "药瓶在办公室的盒子里；盒子在办公室。")

    def test_relative_pronouns_fail_closed_when_context_is_too_short(self) -> None:
        with self.assertRaises(ParseError):
            predict("前者在哪里？")

    def test_dialog_act_queries_can_answer_without_domain_state(self) -> None:
        examples = (
            ("你好", "QUERY dialog_act(greeting)", "你好，我在。"),
            ("谢谢你", "QUERY dialog_act(thanks)", "不客气。"),
            ("你是谁？", "QUERY dialog_act(identity)", "我是结构智能原型，会把对话里的事实、状态、信念和问题先整理成结构再回答。"),
            (
                "你能做什么？",
                "QUERY dialog_act(capabilities)",
                "我可以整理聊天里的事实、状态变化、信念、条件和追问，再回答位置、归属、历史事件、矛盾和摘要。",
            ),
            (
                "你能干嘛",
                "QUERY dialog_act(capabilities)",
                "我可以整理聊天里的事实、状态变化、信念、条件和追问，再回答位置、归属、历史事件、矛盾和摘要。",
            ),
            (
                "你好，你能干嘛",
                "QUERY dialog_act(capabilities)",
                "我可以整理聊天里的事实、状态变化、信念、条件和追问，再回答位置、归属、历史事件、矛盾和摘要。",
            ),
        )

        for text, query_line, answer in examples:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn(query_line, structure)
                self.assertIn("RULE dialog_", structure)
                self.assertEqual(prediction.answer, answer)

    def test_chat_summary_reuses_frame_history(self) -> None:
        prediction = predict("小郭把芯片放进托盘。小王把托盘带到实验室。总结一下")

        structure = prediction.structure.linearize()
        self.assertIn("QUERY dialog_act(summary)", structure)
        self.assertIn("RULE conversation_summary", structure)
        self.assertEqual(prediction.answer, "已知：小郭把芯片放进托盘；小王把托盘带到实验室。")

    def test_user_profile_statements_are_queryable_chat_state(self) -> None:
        prediction = predict("我叫小王。我喜欢咖啡。我叫什么，我喜欢什么？")

        structure = prediction.structure.linearize()
        self.assertIn("REL name(我,小王)", structure)
        self.assertIn("REL likes(我,咖啡)", structure)
        self.assertIn("QUERY compound(multi)", structure)
        self.assertIn("SUBQUERY profile(我,attribute=name)", structure)
        self.assertIn("SUBQUERY profile(我,attribute=likes)", structure)
        self.assertEqual(prediction.answer, "你叫小王；你喜欢咖啡。")

    def test_profile_name_overwrites_and_preferences_can_be_corrected(self) -> None:
        prediction = predict("我叫小王。其实我叫小李。我喜欢咖啡。后来我不喜欢咖啡。我叫什么，我喜欢什么，我不喜欢什么？")

        structure = prediction.structure.linearize()
        self.assertIn("REL name(我,小李)", structure)
        self.assertNotIn("REL name(我,小王)", structure)
        self.assertNotIn("REL likes(我,咖啡)", structure)
        self.assertIn("REL dislikes(我,咖啡)", structure)
        self.assertEqual(prediction.answer, "你叫小李；我还不知道你喜欢什么；你不喜欢咖啡。")

    def test_direct_memory_entries_can_be_written_and_reloaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = LearningPaths(
                memory_direct_data=Path(directory) / "memory_direct_examples.jsonl",
                memory_chat_data=Path(directory) / "memory_chat_examples.jsonl",
                memory_model=Path(directory) / "memory_model.json",
            )
            save_direct_memory_feedback(State("name", "我", "小王"), paths)

            loaded = load_memory_model(paths.memory_model)
            capabilities = default_capabilities(use_environment=False, use_memory=False).with_memory_states(*loaded.states)
            prediction = predict("我叫什么", capabilities)

        self.assertEqual(prediction.answer, "你叫小王。")

    def test_chat_memory_entries_can_be_sedimented_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = LearningPaths(
                memory_direct_data=Path(directory) / "memory_direct_examples.jsonl",
                memory_chat_data=Path(directory) / "memory_chat_examples.jsonl",
                memory_model=Path(directory) / "memory_model.json",
            )
            seed_prediction = predict("我叫小李。我叫什么？", default_capabilities(use_environment=False, use_memory=False))
            result = save_chat_memory_feedback("我叫小李。我叫什么？", seed_prediction.structure, paths)

            loaded = load_memory_model(result.model_path)
            capabilities = default_capabilities(use_environment=False, use_memory=False).with_memory_states(*loaded.states)
            prediction = predict("我叫什么", capabilities)

        self.assertGreaterEqual(result.entry_count, 1)
        self.assertEqual(prediction.answer, "你叫小李。")

    def test_long_term_knowledge_entries_can_be_written_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = LearningPaths(
                memory_knowledge_data=Path(directory) / "memory_knowledge_examples.jsonl",
                memory_knowledge_model=Path(directory) / "memory_knowledge_model.json",
            )
            result = save_memory_knowledge_feedback(
                "为什么天是蓝的？",
                Query("why", "天是蓝的", ("type=why",)),
                "因为阳光进入大气后会被空气分子散射，短波长的蓝光更容易散开，所以天空看起来偏蓝。",
                paths,
                source="curated",
            )

            loaded = load_memory_knowledge_model(result.model_path)
            capabilities = default_capabilities(use_environment=False, use_memory=False).with_answerers(
                LearnedMemoryKnowledgeAnswerer.from_model(result.model_path)
            )
            prediction = predict("为什么天是蓝的？", capabilities)

        self.assertEqual(result.example_count, 1)
        self.assertEqual(len(loaded.patterns), 1)
        self.assertEqual(prediction.answer, "因为阳光进入大气后会被空气分子散射，短波长的蓝光更容易散开，所以天空看起来偏蓝。")

    def test_mixed_chat_fragments_preserve_task_core(self) -> None:
        prediction = predict("小郭把芯片放进托盘。你好，我想知道芯片在哪里？")

        structure = prediction.structure.linearize()
        self.assertIn("QUERY location(芯片)", structure)
        self.assertNotIn("SUBQUERY dialog_act(greeting)", structure)
        self.assertEqual(prediction.answer, "芯片在托盘里。")

    def test_mixed_question_fragment_can_record_profile_statement(self) -> None:
        prediction = predict("我叫小王，你能做什么？我叫什么？")

        structure = prediction.structure.linearize()
        self.assertIn("FRAME f1 type=profile_name time=1", structure)
        self.assertIn("QUERY compound(multi)", structure)
        self.assertIn("SUBQUERY dialog_act(capabilities)", structure)
        self.assertIn("SUBQUERY profile(我,attribute=name)", structure)
        self.assertEqual(
            prediction.answer,
            "我可以整理聊天里的事实、状态变化、信念、条件和追问，再回答位置、归属、历史事件、矛盾和摘要；你叫小王。",
        )

    def test_learned_statement_uses_role_boundaries_instead_of_exact_sentence_text(self) -> None:
        examples = (
            ("小红把药瓶交给了医生。药瓶是谁拥有的？", "REL owner(药瓶,医生)", "了医生", "医生拥有药瓶。"),
            ("小王把盒子关上了。盒子是什么状态？", "REL access(盒子,关闭)", "盒子了", "盒子是关闭状态。"),
            ("工程师把芯片销毁了。芯片是否存在？", "REL exists(芯片,不存在)", "芯片了", "芯片不存在。"),
        )

        for text, expected_line, polluted_slot, answer in examples:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn(expected_line, structure)
                self.assertNotIn(polluted_slot, structure)
                self.assertEqual(prediction.answer, answer)

    def test_if_then_rule_applies_when_antecedent_holds(self) -> None:
        prediction = predict(
            "如果小郭把芯片放进托盘，小王就把托盘带到实验室。小郭把芯片放进托盘。芯片在哪里？"
        )

        structure = prediction.structure.linearize()
        self.assertIn("FRAME f1 type=if_then time=1", structure)
        self.assertIn("RULE container_moves_contents", structure)
        self.assertEqual(prediction.answer, "芯片在实验室的托盘里。")

    def test_if_then_rule_does_not_fire_without_antecedent(self) -> None:
        prediction = predict("如果小郭把芯片放进托盘，小王就把托盘带到实验室。芯片在哪里？")

        structure = prediction.structure.linearize()
        self.assertIn("FRAME f1 type=if_then time=1", structure)
        self.assertNotIn("REL at(托盘,实验室)", structure)
        self.assertEqual(prediction.answer, "不知道芯片在哪里。")

    def test_because_statement_records_cause_and_effect(self) -> None:
        prediction = predict("因为小王把托盘带到实验室，所以芯片在实验室的托盘里。芯片在哪里？")

        structure = prediction.structure.linearize()
        self.assertIn("FRAME f1 type=because time=1", structure)
        self.assertIn("REL at(托盘,实验室)", structure)
        self.assertEqual(prediction.answer, "芯片在实验室的托盘里。")

    def test_why_query_uses_causal_frame(self) -> None:
        prediction = predict(
            "因为小王把托盘带到实验室，所以芯片在实验室的托盘里。为什么芯片在实验室的托盘里？"
        )

        structure = prediction.structure.linearize()
        self.assertIn("QUERY why(芯片在实验室的托盘里)", structure)
        self.assertIn("RULE causal_explanation", structure)
        self.assertEqual(prediction.answer, "因为小王把托盘带到实验室。")

    def test_why_query_can_explain_from_state_chain(self) -> None:
        prediction = predict("小郭把芯片放进托盘。托盘被带到实验室。为什么芯片在实验室的托盘里？")

        structure = prediction.structure.linearize()
        self.assertIn("QUERY why(芯片在实验室的托盘里)", structure)
        self.assertIn("RULE causal_explanation", structure)
        self.assertEqual(prediction.answer, "因为芯片在托盘里，而且托盘在实验室。")

    def test_claim_source_is_separate_from_world_fact(self) -> None:
        prediction = predict("小王说芯片在托盘里。芯片在哪里？")

        structure = prediction.structure.linearize()
        self.assertIn("FRAME f1 type=say time=1", structure)
        self.assertIn("QUERY location(芯片)", structure)
        self.assertEqual(prediction.answer, "不知道芯片在哪里。")

    def test_claim_source_query_finds_speaker(self) -> None:
        prediction = predict("小王说芯片在托盘里。谁说芯片在托盘里？")

        structure = prediction.structure.linearize()
        self.assertIn("QUERY claim_source(芯片在托盘里)", structure)
        self.assertIn("RULE claim_has_source", structure)
        self.assertEqual(prediction.answer, "小王说的。")

    def test_claim_source_query_uses_reference_resolution(self) -> None:
        prediction = predict("小王说这个芯片在托盘里。谁说这个芯片在托盘里？")

        structure = prediction.structure.linearize()
        self.assertIn("QUERY claim_source(芯片在托盘里)", structure)
        self.assertEqual(prediction.answer, "小王说的。")

    def test_belief_does_not_update_factual_world_state(self) -> None:
        prediction = predict("小王以为芯片在盒子里。芯片在哪里？")

        structure = prediction.structure.linearize()
        self.assertIn("FRAME f1 type=believe time=1", structure)
        self.assertIn("EVENT believe(小王,芯片在盒子里)", structure)
        self.assertNotIn("REL in(芯片,盒子)", structure)
        self.assertEqual(prediction.answer, "不知道芯片在哪里。")

    def test_belief_location_query_uses_personal_world_view(self) -> None:
        questions = (
            "小王认为芯片在哪里？",
            "小王相信芯片在什么地方？",
            "你知道的话，可以告诉我小王以为芯片在哪里吗？",
        )

        for question in questions:
            with self.subTest(question=question):
                prediction = predict(f"小郭把芯片放进托盘。小王以为芯片在盒子里。{question}")
                structure = prediction.structure.linearize()
                self.assertIn("REL in(芯片,托盘)", structure)
                self.assertIn("QUERY belief_location(芯片,person=小王)", structure)
                self.assertIn("RULE belief_location_found", structure)
                self.assertEqual(prediction.answer, "小王认为芯片在盒子里。")

    def test_belief_source_query_finds_all_believers(self) -> None:
        prediction = predict("小王认为芯片在盒子里。小郭相信芯片在盒子里。谁认为芯片在盒子里？")

        structure = prediction.structure.linearize()
        self.assertIn("QUERY belief_source(芯片在盒子里)", structure)
        self.assertIn("RULE belief_has_source", structure)
        self.assertEqual(prediction.answer, "小王和小郭这么认为。")

    def test_belief_updates_only_that_person_view_in_order(self) -> None:
        prediction = predict("小王以为芯片在托盘里。小王后来认为芯片在盒子里。小王认为芯片在哪里？")

        structure = prediction.structure.linearize()
        self.assertIn("EVENT believe(小王,芯片在托盘里)", structure)
        self.assertIn("EVENT believe(小王,芯片在盒子里)", structure)
        self.assertEqual(prediction.answer, "小王认为芯片在盒子里。")

    def test_contradiction_query_detects_belief_against_fact(self) -> None:
        prediction = predict("小郭把芯片放进托盘。小王以为芯片在盒子里。有没有矛盾？")

        structure = prediction.structure.linearize()
        self.assertIn("QUERY contradictions(world)", structure)
        self.assertIn("RULE contradictions_found", structure)
        self.assertEqual(prediction.answer, "存在矛盾：小王认为芯片在盒子里，但事实是芯片在托盘里。")

    def test_contradiction_query_detects_claim_against_fact(self) -> None:
        prediction = predict("小郭把芯片放进托盘。小王说芯片在盒子里。哪里有冲突？")

        structure = prediction.structure.linearize()
        self.assertIn("FRAME f3 type=say time=3", structure)
        self.assertIn("RULE contradictions_found", structure)
        self.assertEqual(prediction.answer, "存在矛盾：小王说芯片在盒子里，但事实是芯片在托盘里。")

    def test_contradiction_query_allows_matching_claim(self) -> None:
        prediction = predict("小郭把芯片放进托盘。小王说芯片在托盘里。有没有矛盾？")

        structure = prediction.structure.linearize()
        self.assertIn("RULE no_contradictions", structure)
        self.assertEqual(prediction.answer, "没有发现矛盾。")

    def test_ordered_state_correction_is_not_treated_as_contradiction(self) -> None:
        prediction = predict("小郭把芯片放进托盘。芯片不在托盘里而在盒子里。有没有矛盾？")

        structure = prediction.structure.linearize()
        self.assertIn("REL in(芯片,盒子)", structure)
        self.assertIn("RULE no_contradictions", structure)
        self.assertEqual(prediction.answer, "没有发现矛盾。")

    def test_counterfactual_move_replays_state_without_event(self) -> None:
        prediction = predict(
            "小郭把芯片放进托盘。小王把托盘带到实验室。"
            "如果小王没有把托盘带到实验室，芯片会在哪里？"
        )

        structure = prediction.structure.linearize()
        self.assertIn("REL at(托盘,实验室)", structure)
        self.assertIn(
            "QUERY counterfactual_location(芯片,without_event=move,actor=小王,theme=托盘,goal=实验室)",
            structure,
        )
        self.assertIn("RULE counterfactual_location_found", structure)
        self.assertEqual(prediction.answer, "如果没有这个事件，芯片会在托盘里。")

    def test_counterfactual_location_allows_omitted_if(self) -> None:
        prediction = predict("小郭把芯片放进托盘。小王没有把托盘带到实验室，芯片会在哪里？")

        structure = prediction.structure.linearize()
        self.assertIn("QUERY counterfactual_location(芯片,without_event=move,actor=小王,theme=托盘,goal=实验室)", structure)
        self.assertIn("RULE counterfactual_location_found", structure)
        self.assertEqual(prediction.answer, "如果没有这个事件，芯片会在托盘里。")

    def test_counterfactual_take_out_restores_previous_container_state(self) -> None:
        prediction = predict(
            "小郭把芯片放进托盘。小王把芯片从托盘里取出。"
            "如果小王没有把芯片从托盘里取出，芯片会在哪里？"
        )

        structure = prediction.structure.linearize()
        self.assertNotIn("REL in(芯片,托盘)", structure)
        self.assertIn(
            "QUERY counterfactual_location(芯片,without_event=take_out,actor=小王,theme=芯片,source=托盘)",
            structure,
        )
        self.assertEqual(prediction.answer, "如果没有这个事件，芯片会在托盘里。")

    def test_counterfactual_put_in_can_make_location_unknown(self) -> None:
        examples = (
            "小郭把芯片放进托盘。如果小郭没有把芯片放进托盘，芯片会在哪里？",
            "小红把药瓶放进盒子。如果小红没有把药瓶放进盒子药瓶会在哪里？",
        )

        for text in examples:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn("RULE counterfactual_location_unknown", structure)
                self.assertIn("QUERY counterfactual_location", structure)

    def test_fact_belief_and_counterfactual_views_do_not_collapse(self) -> None:
        fact = predict("小郭把芯片放进托盘。小王以为芯片在盒子里。芯片实际在哪里？")
        belief = predict("小郭把芯片放进托盘。小王以为芯片在盒子里。小王认为芯片在哪里？")
        counterfactual = predict(
            "小郭把芯片放进托盘。小王把芯片从托盘里取出。"
            "如果小王没有把芯片从托盘里取出，芯片会在哪里？"
        )

        self.assertEqual(fact.answer, "芯片在托盘里。")
        self.assertEqual(belief.answer, "小王认为芯片在盒子里。")
        self.assertEqual(counterfactual.answer, "如果没有这个事件，芯片会在托盘里。")

    def test_complex_world_views_share_event_schema_without_state_leakage(self) -> None:
        base = (
            "小郭把芯片放进托盘。小王把托盘带到实验室。"
            "小李说芯片在盒子里。小张认为芯片在托盘里。"
        )
        fact = predict(base + "芯片在哪里？")
        belief = predict(base + "小张认为芯片在哪里？")
        conflict = predict(base + "有没有矛盾？")
        counterfactual = predict(base + "如果小王没有把托盘带到实验室，芯片会在哪里？")

        self.assertEqual(fact.answer, "芯片在实验室的托盘里。")
        self.assertEqual(belief.answer, "小张认为芯片在托盘里。")
        self.assertEqual(conflict.answer, "存在矛盾：小李说芯片在盒子里，但事实是芯片在实验室的托盘里。")
        self.assertEqual(counterfactual.answer, "如果没有这个事件，芯片会在托盘里。")

    def test_compound_query_keeps_multiple_subqueries_in_source_order(self) -> None:
        prediction = predict(
            "小郭把芯片放进托盘。小王把托盘带到实验室。"
            "托盘在哪里，托盘里有什么，谁把芯片放进托盘？"
        )

        structure = prediction.structure.linearize()
        self.assertIn("QUERY compound(multi)", structure)
        location_index = structure.index("SUBQUERY location(托盘)")
        contents_index = structure.index("SUBQUERY contents(托盘)")
        actor_index = structure.index("SUBQUERY actor_for_event(put_in,item=芯片,holder=托盘)")
        self.assertLess(location_index, contents_index)
        self.assertLess(contents_index, actor_index)
        self.assertIn("RULE compound_query", structure)
        self.assertEqual(prediction.answer, "托盘在实验室；托盘里至少有芯片；小郭把芯片放进托盘。")

    def test_compound_query_ignores_filler_fragments_and_keeps_real_tasks(self) -> None:
        prediction = predict(
            "小郭把芯片放进托盘。小王把托盘带到实验室。"
            "你知道吗？，你知道的话，可以告诉我托盘在哪里，托盘里有什么，我想知道下"
        )

        structure = prediction.structure.linearize()
        self.assertIn("QUERY compound(multi)", structure)
        self.assertIn("SUBQUERY location(托盘)", structure)
        self.assertIn("SUBQUERY contents(托盘)", structure)
        self.assertNotIn("SUBQUERY location(你知道)", structure)
        self.assertEqual(prediction.answer, "托盘在实验室；托盘里至少有芯片。")

    def test_compound_query_can_mix_fact_belief_and_counterfactual_views(self) -> None:
        prediction = predict(
            "小郭把芯片放进托盘。小王把托盘带到实验室。小张认为芯片在盒子里。"
            "芯片在哪里，小张认为芯片在哪里，如果小王没有把托盘带到实验室，芯片会在哪里？"
        )

        structure = prediction.structure.linearize()
        self.assertIn("QUERY compound(multi)", structure)
        self.assertIn("SUBQUERY location(芯片)", structure)
        self.assertIn("SUBQUERY belief_location(芯片,person=小张)", structure)
        self.assertIn("SUBQUERY counterfactual_location(芯片,without_event=move,actor=小王,theme=托盘,goal=实验室)", structure)
        self.assertEqual(
            prediction.answer,
            "芯片在实验室的托盘里；小张认为芯片在盒子里；如果没有这个事件，芯片会在托盘里。",
        )


if __name__ == "__main__":
    unittest.main()
