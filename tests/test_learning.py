from __future__ import annotations

import unittest

from tests.support import *


class LearningTest(unittest.TestCase):
    def test_default_capabilities_register_single_neural_query_and_statement_parsers(self) -> None:
        capabilities = default_capabilities()

        self.assertIsInstance(capabilities, CognitiveCapabilities)
        self.assertEqual(len(capabilities.query_parsers), 1)
        query_parser = capabilities.query_parsers[0]
        assert isinstance(query_parser, LoadedNeuralQueryParser)
        self.assertGreater(len(query_parser.patterns), 0)
        self.assertEqual(len(capabilities.statement_parsers), 1)
        statement_parser = capabilities.statement_parsers[0]
        assert isinstance(statement_parser, LoadedNeuralStatementParser)
        self.assertGreater(len(statement_parser.patterns), 0)

    def test_default_capabilities_reuse_cached_loaded_components(self) -> None:
        first = default_capabilities()
        second = default_capabilities()

        self.assertIs(first.query_parsers[0], second.query_parsers[0])
        self.assertIs(first.statement_parsers[0], second.statement_parsers[0])
        self.assertIs(first.answerers[-1], second.answerers[-1])
        self.assertIs(default_learned_dialog_answerer(), default_learned_dialog_answerer())

    def test_cli_neural_provider_default_path_imports_package_neural_module(self) -> None:
        import os
        from types import SimpleNamespace
        from unittest.mock import patch

        from struct_llm.cli_commands.common import apply_neural_provider_args

        capabilities = default_capabilities(use_environment=False, use_memory=False)
        args = SimpleNamespace(neural_provider=None, neural_answer_priority="after_verified")

        with patch.dict(os.environ, {"EA_AI_NEURAL_PROVIDER": ""}):
            configured = apply_neural_provider_args(capabilities, args)

        self.assertIs(configured, capabilities)

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

    def test_neural_answerer_runs_after_verified_dialog_answers_by_default(self) -> None:
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

    def test_neural_provider_answers_knowledge_questions_via_learned_memory(self) -> None:
        model = make_model()
        capabilities = default_capabilities(use_environment=False, use_memory=False)
        capabilities = capabilities.with_answerers(
            default_learned_memory_knowledge_answerer("data/memory_knowledge_model.json")
        )
        capabilities = with_neural_boundary(capabilities, model)

        cases = (
            ("我家的铁怎么生锈了", "QUERY why(铁会生锈,type=why)",
             "铁和空气里的氧、水发生反应后，会生成疏松的氧化物，也就是锈。"),
            ("苹果会掉到地上的原因是什么", "QUERY why(苹果会掉到地上,type=why)",
             "地球对物体有引力，把苹果拉向地面，这就是重力作用。"),
        )
        for text, query_line, answer in cases:
            with self.subTest(text=text):
                prediction = predict(text, capabilities)
                self.assertIn(query_line, prediction.structure.linearize())
                self.assertEqual(prediction.answer, answer)

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

    def test_statement_examples_evaluate_and_train_neural_parser(self) -> None:
        examples = load_statement_jsonl("data/statement_examples.jsonl")
        # default parser meets accuracy on current dataset
        default_result = evaluate_statement_parser(default_neural_statement_parser(), examples)
        self.assertEqual(default_result.total, len(examples))
        self.assertGreaterEqual(default_result.accuracy, 0.90)

        # freshly trained runtime model from same dataset meets accuracy and writes artifacts
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

    def test_query_examples_evaluate_and_train_neural_parser(self) -> None:
        examples = load_query_jsonl("data/query_examples.jsonl")
        # default parser meets accuracy on current dataset
        default_result = evaluate_query_parser(default_neural_query_parser(), examples)
        self.assertEqual(default_result.total, len(examples))
        self.assertGreaterEqual(default_result.accuracy, 0.99)

        # freshly trained runtime model from same dataset meets accuracy and writes artifacts
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

    def test_event_schema_projects_state_effects_and_owns_query_role_aliases(self) -> None:
        # registered frame schemas project the expected state effects
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

        # registered schemas also own query role aliases used by event_actor queries
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


if __name__ == "__main__":
    unittest.main()
