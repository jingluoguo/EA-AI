from __future__ import annotations

import unittest
from collections import Counter

from struct_llm.perception.lexer import split_query_candidate
from struct_llm.perception.normalizer import bare_topic_followup, normalize_question
from struct_llm.perception.reference import strip_ellipsis_particles
from struct_llm.comprehension.surface_lexicon import load_surface_lexicon_jsonl, surface_forms, surface_replacements
from struct_llm.world.event_schema import event_schemas, load_event_schema_jsonl
from tests.support import *


class StructuralCoverageTest(unittest.TestCase):
    def test_terminal_discourse_particles_are_loaded_from_surface_lexicon_data(self) -> None:
        entries = load_surface_lexicon_jsonl("data/surface_lexicon_examples.jsonl")

        categories = {entry.category for entry in entries}
        self.assertGreaterEqual(len(entries), 18)
        self.assertIn("terminal_discourse_particle", categories)
        self.assertIn("containment_verb", categories)
        self.assertIn("object_pronoun", categories)
        self.assertEqual(surface_forms("terminal_discourse_particle"), ("呢", "吗", "吧", "呀", "啊"))
        self.assertIn(("放入", "放进"), surface_replacements("containment_verb"))
        self.assertIn(("取走", "取出"), surface_replacements("take_out_verb"))
        self.assertIn("吃", surface_forms("meal_topic_marker"))
        self.assertIn("建议", surface_forms("advice_request_marker"))
        self.assertIn("不知道", surface_forms("choice_uncertainty_marker"))
        self.assertIn("清淡", surface_forms("meal_light_preference_marker"))
        self.assertIn("别太油", surface_forms("meal_light_preference_marker"))
        self.assertIn("满足", surface_forms("meal_rich_preference_marker"))
        self.assertIn("吃饱", surface_forms("meal_rich_preference_marker"))
        self.assertIn("还有别的", surface_forms("alternative_request_marker"))
        self.assertIn("换一个", surface_forms("alternative_request_marker"))
        self.assertIn("叫什么", surface_forms("profile_name_query_marker"))
        self.assertIn("喜欢", surface_forms("profile_like_query_marker"))
        self.assertIn("不喜欢", surface_forms("profile_dislike_query_marker"))
        self.assertIn("能干嘛", surface_forms("capabilities_query_marker"))
        self.assertIn("擅长什么", surface_forms("capabilities_query_marker"))
        self.assertIn("meal_suggestion", surface_forms("integrated_dialog_act_target"))
        self.assertIn("经过", surface_forms("route_history_verb_marker"))
        self.assertIn("移动过", surface_forms("route_history_verb_marker"))
        self.assertIn("哪些地点", surface_forms("place_collection_query_marker"))
        self.assertIn("哪里", surface_forms("place_collection_query_marker"))
        self.assertIn("认为", surface_forms("belief_attitude_marker"))
        self.assertIn("以为", surface_forms("belief_attitude_marker"))
        self.assertIn("在哪里", surface_forms("location_query_marker"))
        self.assertEqual(strip_ellipsis_particles("那个呢？"), "那个")
        self.assertEqual(normalize_question("我想知道物品放入托盘里面了吗？"), "东西放进托盘里面")
        self.assertEqual(bare_topic_followup("实验室呢"), "实验室")

    def test_event_state_schemas_are_loaded_from_data(self) -> None:
        schemas = load_event_schema_jsonl("data/event_schema_examples.jsonl")
        by_type = event_schemas()

        self.assertGreaterEqual(len(schemas), 14)
        self.assertIn("put_in", by_type)
        self.assertIn("profile_dislike", by_type)
        self.assertEqual(by_type["put_in"].effects[0].name, "in")
        self.assertEqual(by_type["put_in"].role_for_qualifier("holder"), "goal")

    def test_query_test_split_covers_core_structural_families(self) -> None:
        examples = load_query_jsonl("data/query_examples.jsonl")
        train_intents = query_intents_for_split(examples, "train")
        test_intents = query_intents_for_split(examples, "test")

        for intent in (
            "location",
            "belief_location",
            "claim_source",
            "belief_source",
            "compound",
            "dialog_act",
            "counterfactual_location",
            "location_before_event",
            "location_after_event",
            "polar_location",
            "owner",
            "color",
            "object_state",
            "contents",
            "existence",
            "polar_existence",
            "polar_contents",
            "count",
            "compare_count",
            "contents_except",
            "same_location",
            "initial_location",
            "actor_for_item",
            "latest_actor_for_item",
            "earliest_actor_for_event",
            "latest_actor_for_event",
            "actor_for_event",
            "places_visited",
            "actions_by_actors",
            "inventories",
            "location_before_actor_action",
            "contents_before_event",
            "contents_after_event",
            "events_after_event",
            "contradictions",
            "claim_source",
            "belief_source",
        ):
            with self.subTest(intent=intent):
                self.assertGreater(train_intents[intent], 0)
                self.assertGreater(test_intents[intent], 0)

        dialog_targets_test = query_dialog_targets_for_split(examples, "test")
        self.assertTrue({"greeting", "thanks", "farewell"}.issubset(dialog_targets_test))
        for source in structural_pattern_sources(examples):
            with self.subTest(source=source):
                self.assertGreater(source_count_for_split(examples, source, "train"), 0)
                self.assertGreater(source_count_for_split(examples, source, "test"), 0)

    def test_statement_test_split_covers_frame_and_rejection_families(self) -> None:
        examples = load_statement_jsonl("data/statement_examples.jsonl")
        train_frames = statement_frames_for_split(examples, "train")
        test_frames = statement_frames_for_split(examples, "test")

        for frame_type in (
            "put_in",
            "take_out",
            "move",
            "give",
            "paint",
            "open",
            "close",
            "create",
            "destroy",
            "say",
            "believe",
            "if_then",
            "because",
            "be_in",
            "not_in",
            "profile_name",
            "profile_like",
            "profile_dislike",
        ):
            with self.subTest(frame_type=frame_type):
                self.assertGreater(train_frames[frame_type], 0)
                self.assertGreater(test_frames[frame_type], 0)

        self.assertGreater(empty_statement_count(examples, "train"), 0)
        self.assertGreater(empty_statement_count(examples, "test"), 0)
        self.assertGreater(source_count_for_split(examples, "structural_pattern_ownership_state_eval", "train"), 0)
        self.assertGreater(source_count_for_split(examples, "structural_pattern_ownership_state_eval", "test"), 0)
        self.assertGreater(source_count_for_split(examples, "structural_pattern_attribute_state_eval", "train"), 0)
        self.assertGreater(source_count_for_split(examples, "structural_pattern_attribute_state_eval", "test"), 0)
        for source in structural_pattern_sources(examples):
            with self.subTest(source=source):
                self.assertGreater(source_count_for_split(examples, source, "train"), 0)
                self.assertGreater(source_count_for_split(examples, source, "test"), 0)

    def test_structural_pattern_examples_are_consumed_by_learning_paths(self) -> None:
        query_examples = load_query_jsonl("data/query_examples.jsonl")
        statement_examples = load_statement_jsonl("data/statement_examples.jsonl")

        query_parser = default_neural_query_parser()
        statement_parser = default_neural_statement_parser()

        for split in ("train", "test"):
            with self.subTest(dataset="query", split=split):
                examples = structural_pattern_examples_for_split(query_examples, split)
                self.assertGreater(len(examples), 0)
                result = evaluate_query_parser(query_parser, examples)
                self.assertEqual(result.matched, result.total)

            with self.subTest(dataset="statement", split=split):
                examples = structural_pattern_examples_for_split(statement_examples, split)
                self.assertGreater(len(examples), 0)
                result = evaluate_statement_parser(statement_parser, examples)
                self.assertEqual(result.matched, result.total)

    def test_episode_test_split_covers_pragmatic_context_families(self) -> None:
        examples = load_episode_jsonl("data/episode_examples.jsonl")
        analyzer = InMemoryPragmaticAnalyzer(examples)
        result = evaluate_pragmatic_analyzer(analyzer, examples)
        train_acts = pragmatic_acts_for_split(examples, "train")
        test_acts = pragmatic_acts_for_split(examples, "test")

        self.assertEqual(result.matched, result.total)
        for act in (
            "incomplete_utterance",
            "ambiguous_reference",
            "underspecified_action_request",
            "confirm_understanding",
            "repair_previous_understanding",
            "action_result_report",
            "recall_previous_turn",
            "clarification_request",
            "confirmation_check",
            "continuation_request",
            "repair_previous_understanding",
        ):
            with self.subTest(act=act):
                self.assertGreater(train_acts[act] + test_acts[act], 0)

        self.assertGreater(test_acts["ambiguous_reference"], 0)
        self.assertGreater(test_acts["confirmation_check"], 0)
        self.assertGreater(test_acts["clarification_request"], 0)
        self.assertGreater(test_acts["continuation_request"], 0)
        self.assertGreater(test_acts["recall_previous_turn"], 0)
        self.assertGreater(train_acts["repair_previous_understanding"], 0)
        self.assertGreater(test_acts["repair_previous_understanding"], 0)
        self.assertGreater(train_acts["action_result_report"], 0)
        self.assertGreater(test_acts["action_result_report"], 0)
        pronoun_reference_examples = [
            example
            for example in examples
            if example.source == "structural_pattern_pronoun_reference_clarification"
        ]
        self.assertEqual({example.split for example in pronoun_reference_examples}, {"train", "test"})
        self.assertTrue(all(example.known_world_state for example in pronoun_reference_examples))
        self.assertTrue(all(example.expected_entities == (Entity("unresolved_reference", "它"),) for example in pronoun_reference_examples))
        self.assertTrue(all("candidates=" in " ".join(act.qualifiers) for example in pronoun_reference_examples for act in example.expected_pragmatic_acts))
        self.assertGreater(episode_expected_frame_count(examples, "train", "repair_previous_understanding"), 0)
        self.assertGreater(episode_expected_frame_count(examples, "test", "repair_previous_understanding"), 0)
        self.assertGreater(episode_action_result_count(examples, "train"), 0)
        self.assertGreater(episode_action_result_count(examples, "test"), 0)
        self.assertGreater(episode_action_result_status_count(examples, "train", "success"), 0)
        self.assertGreater(episode_action_result_status_count(examples, "test", "success"), 0)
        self.assertGreater(episode_action_result_status_count(examples, "train", "failure"), 0)
        self.assertGreater(episode_action_result_status_count(examples, "test", "failure"), 0)
        self.assertGreater(episode_pragmatic_qualifier_count(examples, "train", "status=failure"), 0)
        self.assertGreater(episode_pragmatic_qualifier_count(examples, "test", "status=failure"), 0)

        train_examples = tuple(example for example in examples if example.split == "train")
        test_examples = tuple(example for example in examples if example.split == "test")
        heldout_result = evaluate_pragmatic_analyzer(InMemoryPragmaticAnalyzer(train_examples), test_examples)
        self.assertEqual(heldout_result.matched, heldout_result.total)

    def test_intent_test_split_covers_contextual_intention_families(self) -> None:
        records = load_intent_jsonl("data/intent_examples.jsonl")
        structural_records = [
            record
            for record in records
            if record.source == "structural_pattern_intent_context"
        ]
        self.assertGreaterEqual(len(structural_records), 32)
        self.assertTrue(all(record.context for record in structural_records))
        self.assertTrue(all(record.belief_state for record in structural_records))
        self.assertTrue(all(record.intention.belief for record in structural_records))
        self.assertTrue(all(record.intention.strategy for record in structural_records))

        train_predicates = intent_belief_predicates_for_split(records, "train")
        test_predicates = intent_belief_predicates_for_split(records, "test")
        for predicate in (
            "dialog_opening",
            "clarification_request",
            "confirmation_check",
            "continuation_request",
            "belief_location_query",
            "dialog_then_task",
            "recall_previous_turn",
            "source_contrast_query",
            "counterfactual_location_query",
            "incomplete_intention",
            "repair_previous_understanding",
            "action_result_report",
            "resolved_reference_query",
            "resolved_focus_ellipsis",
        ):
            with self.subTest(predicate=predicate):
                self.assertGreater(train_predicates[predicate], 0)
                self.assertGreater(test_predicates[predicate], 0)

        self.assertGreater(intent_world_state_count(structural_records, "train"), 0)
        self.assertGreater(intent_world_state_count(structural_records, "test"), 0)
        self.assertGreater(intent_belief_contains_count(structural_records, "train", "status=success"), 0)
        self.assertGreater(intent_belief_contains_count(structural_records, "test", "status=success"), 0)
        self.assertGreater(intent_belief_contains_count(structural_records, "train", "status=failure"), 0)
        self.assertGreater(intent_belief_contains_count(structural_records, "test", "status=failure"), 0)

        analyzer = InMemoryIntentAnalyzer.from_jsonl("data/intent_examples.jsonl")
        structural_examples = tuple(
            example
            for example in analyzer.examples
            if example.intention.source == "structural_pattern_intent_context"
        )
        result = evaluate_intent_analyzer(InMemoryIntentAnalyzer(structural_examples), structural_examples)
        self.assertEqual(result.matched, result.total)

        train_examples = tuple(
            example
            for example in structural_examples
            if any(record.observation == example.observation and record.split == "train" for record in structural_records)
        )
        test_examples = tuple(
            example
            for example in structural_examples
            if any(record.observation == example.observation and record.split == "test" for record in structural_records)
        )
        heldout_result = evaluate_intent_analyzer(InMemoryIntentAnalyzer(train_examples), test_examples)
        self.assertEqual(heldout_result.matched, heldout_result.total)

    def test_fact_belief_and_source_queries_remain_structurally_distinct(self) -> None:
        fact = predict("小郭把芯片放进托盘。小王以为芯片在盒子里。不是问小王怎么想，芯片实际在哪里？")
        fact_structure = fact.structure.linearize()
        fact_lines = set(fact_structure.splitlines())
        self.assertIn("QUERY location(芯片)", fact_lines)
        self.assertNotIn("QUERY compound(multi)", fact_lines)
        self.assertNotIn("QUERY belief_location", fact_structure)
        self.assertNotIn("SUBQUERY same_location", fact_structure)

        belief = predict("小郭把芯片放进托盘。小王以为芯片在盒子里。不是问事实，问小王认为芯片在哪里？")
        belief_structure = belief.structure.linearize()
        self.assertIn("QUERY belief_location(芯片,person=小王)", belief_structure)
        self.assertIn("RULE belief_location_found", belief_structure)

        source = predict("小王说芯片在盒子里。小李认为芯片在盒子里。谁相信芯片在盒子里，谁说芯片在盒子里？")
        source_structure = source.structure.linearize()
        self.assertIn("QUERY compound(multi)", source_structure)
        self.assertIn("SUBQUERY belief_source(芯片在盒子里)", source_structure)
        self.assertIn("SUBQUERY claim_source(芯片在盒子里)", source_structure)

    def test_collection_query_constructions_preserve_comparison_and_exclusion(self) -> None:
        comparison = predict(
            "小郭把芯片放进托盘。托盘被带到实验室。"
            "小王把药瓶放进盒子。盒子被带到办公室。"
            "实验室和办公室是否一样多"
        )
        comparison_structure = comparison.structure.linearize()
        self.assertIn("QUERY compare_count(实验室和办公室,left=实验室,right=办公室)", comparison_structure)
        self.assertNotIn("QUERY contents(实验室和办公室)", comparison_structure)

        exclusion = predict(
            "小王把芯片放进盒子。盒子被带到办公室。"
            "办公室里面除盒子之外还有什么"
        )
        exclusion_structure = exclusion.structure.linearize()
        self.assertIn("QUERY contents_except(办公室,exclude=盒子)", exclusion_structure)
        self.assertNotIn("QUERY contents(办公室)", exclusion_structure)

    def test_contextual_ellipsis_uses_focus_query_intent(self) -> None:
        parser = default_neural_query_parser()

        location = parser(
            "盒子呢",
            (Entity("container", "盒子"), Entity("query_intent", "location")),
        )
        self.assertIsNotNone(location)
        assert location is not None
        self.assertEqual(location.linearize(), "QUERY location(盒子)")

        contents = parser(
            "盒子呢",
            (Entity("container", "盒子"), Entity("query_intent", "contents")),
        )
        self.assertIsNotNone(contents)
        assert contents is not None
        self.assertEqual(contents.linearize(), "QUERY contents(盒子)")

        topic_location = parser(
            "药瓶呢",
            (Entity("topic", "药瓶"), Entity("query_intent", "location")),
        )
        self.assertIsNotNone(topic_location)
        assert topic_location is not None
        self.assertEqual(topic_location.linearize(), "QUERY location(药瓶)")

        topic_contents = parser(
            "盒子呢",
            (Entity("topic", "盒子"), Entity("query_intent", "contents")),
        )
        self.assertIsNotNone(topic_contents)
        assert topic_contents is not None
        self.assertEqual(topic_contents.linearize(), "QUERY contents(盒子)")

        capabilities = default_capabilities(use_environment=False, use_memory=False)
        first = predict("小郭把芯片放进托盘。小王把药瓶放进盒子。芯片在哪里？", capabilities)
        working = capabilities_with_working_turn(
            capabilities,
            "小郭把芯片放进托盘。小王把药瓶放进盒子。芯片在哪里？",
            first.structure.states,
            first.structure.query,
        )
        follow_up = predict("盒子呢？", working)
        follow_up_structure = follow_up.structure.linearize()
        self.assertIn("REL focus_query_intent(user,location)", follow_up_structure)
        self.assertIn("QUERY location(盒子)", follow_up_structure)
        self.assertNotIn("QUERY object_state(盒子,state=access)", follow_up_structure)

    def test_focus_topic_ellipsis_dataset_covers_object_and_container_carryover(self) -> None:
        examples = [
            example
            for example in load_query_jsonl("data/query_examples.jsonl")
            if example.source == "structural_pattern_focus_topic_ellipsis_eval"
        ]
        self.assertGreaterEqual(len(examples), 4)

        splits = {example.split for example in examples}
        intents = {example.query.intent for example in examples if example.query is not None}
        entity_roles = {
            tuple(entity.role for entity in example.entities)
            for example in examples
        }

        self.assertEqual(splits, {"train", "test"})
        self.assertEqual(intents, {"location", "contents"})
        self.assertEqual(entity_roles, {("topic", "query_intent")})

    def test_statement_sequence_keeps_history_while_current_state_overwrites(self) -> None:
        parser = default_neural_statement_parser()
        parsed = parser("小李先表示药瓶在柜子里，随后相信药瓶在盒子里")

        self.assertIsNotNone(parsed)
        assert parsed is not None
        lines = linearize_statement_result(parsed)
        self.assertIn("FRAME say", lines)
        self.assertIn("ROLE speaker=小李", lines)
        self.assertIn("ROLE proposition=药瓶在柜子里", lines)
        self.assertIn("FRAME believe", lines)
        self.assertIn("ROLE person=小李", lines)
        self.assertIn("ROLE proposition=药瓶在盒子里", lines)

        prediction = predict("药瓶一开始在柜子里，之后在盒子里。药瓶在哪里？")
        structure = prediction.structure.linearize()
        self.assertIn("FRAME f1 type=be_in time=1", structure)
        self.assertIn("FRAME f2 type=be_in time=2", structure)
        self.assertIn("REL in(药瓶,盒子)", structure)
        self.assertNotIn("REL in(药瓶,柜子)", structure)
        self.assertEqual(prediction.answer, "药瓶在盒子里。")

    def test_discourse_relative_references_resolve_before_query_learning(self) -> None:
        prediction = predict("小红把药瓶放进盒子。盒子被带到办公室。前者在哪里，后者在哪里？")
        structure = prediction.structure.linearize()

        self.assertIn("QUERY compound(multi)", structure)
        self.assertIn("SUBQUERY location(药瓶)", structure)
        self.assertIn("SUBQUERY location(盒子)", structure)
        self.assertNotIn("QUERY location(前者)", structure)
        self.assertNotIn("QUERY location(后者)", structure)
        self.assertEqual(prediction.answer, "药瓶在办公室的盒子里；盒子在办公室。")

        with self.assertRaises(ParseError):
            predict("前者在哪里？")

    def test_existence_and_polar_existence_stay_structurally_distinct(self) -> None:
        existence = predict("工程师制造芯片。芯片是否存在？")
        existence_structure = existence.structure.linearize()
        self.assertIn("FRAME f1 type=create", existence_structure)
        self.assertIn("ROLE f1 result=存在", existence_structure)
        self.assertIn("REL exists(芯片,存在)", existence_structure)
        self.assertIn("QUERY existence(芯片)", existence_structure)
        self.assertNotIn("QUERY polar_existence(芯片)", existence_structure)
        self.assertEqual(existence.answer, "芯片存在。")

        polar = predict("工程师销毁芯片。芯片存在吗？")
        polar_structure = polar.structure.linearize()
        self.assertIn("FRAME f1 type=destroy", polar_structure)
        self.assertIn("ROLE f1 result=不存在", polar_structure)
        self.assertIn("REL exists(芯片,不存在)", polar_structure)
        self.assertIn("QUERY polar_existence(芯片)", polar_structure)
        self.assertNotIn("QUERY existence(芯片)", polar_structure)
        self.assertEqual(polar.answer, "不是，芯片不存在。")

    def test_ownership_transfer_keeps_history_while_current_owner_overwrites(self) -> None:
        prediction = predict("小红把药瓶交给医生。医生把药瓶交给老师。现在谁拥有药瓶？")
        structure = prediction.structure.linearize()

        self.assertIn("FRAME f1 type=give", structure)
        self.assertIn("ROLE f1 actor=小红", structure)
        self.assertIn("ROLE f1 recipient=医生", structure)
        self.assertIn("FRAME f3 type=give", structure)
        self.assertIn("ROLE f3 actor=医生", structure)
        self.assertIn("ROLE f3 recipient=老师", structure)
        self.assertIn("REL owner(药瓶,老师)", structure)
        self.assertNotIn("REL owner(药瓶,医生)", structure)
        self.assertIn("QUERY owner(药瓶)", structure)
        self.assertEqual(prediction.answer, "老师拥有药瓶。")

    def test_attribute_state_keeps_history_while_current_value_overwrites(self) -> None:
        color_prediction = predict("工程师把笔记本涂成绿色。研究员把笔记本涂成黄色。现在笔记本是什么颜色？")
        color_structure = color_prediction.structure.linearize()
        self.assertIn("FRAME f1 type=paint", color_structure)
        self.assertIn("ROLE f1 result=绿色", color_structure)
        self.assertIn("FRAME f3 type=paint", color_structure)
        self.assertIn("ROLE f3 result=黄色", color_structure)
        self.assertIn("REL color(笔记本,黄色)", color_structure)
        self.assertNotIn("REL color(笔记本,绿色)", color_structure)
        self.assertIn("QUERY color(笔记本)", color_structure)
        self.assertEqual(color_prediction.answer, "笔记本是黄色。")

        access_prediction = predict("小王打开盒子。小郭把盒子关上。盒子现在是什么状态？")
        access_structure = access_prediction.structure.linearize()
        self.assertIn("FRAME f1 type=open", access_structure)
        self.assertIn("ROLE f1 result=打开", access_structure)
        self.assertIn("FRAME f3 type=close", access_structure)
        self.assertIn("ROLE f3 result=关闭", access_structure)
        self.assertIn("REL access(盒子,关闭)", access_structure)
        self.assertNotIn("REL access(盒子,打开)", access_structure)
        self.assertIn("QUERY object_state(盒子,state=access)", access_structure)
        self.assertEqual(access_prediction.answer, "盒子是关闭状态。")

    def test_negation_correction_keeps_negated_frame_and_sets_corrected_state(self) -> None:
        parser = default_neural_statement_parser()
        parsed = parser("文件不是在抽屉里，是在文件夹里")

        self.assertIsNotNone(parsed)
        assert parsed is not None
        lines = linearize_statement_result(parsed)
        self.assertIn("FRAME not_in", lines)
        self.assertIn("ROLE source=抽屉", lines)
        self.assertIn("FRAME be_in", lines)
        self.assertIn("ROLE goal=文件夹", lines)

        prediction = predict("小红把文件放进抽屉。文件不是在抽屉里，是在文件夹里。文件在哪里？")
        structure = prediction.structure.linearize()
        self.assertIn("FRAME f3 type=not_in time=3", structure)
        self.assertIn("ROLE f3 source=抽屉", structure)
        self.assertIn("FRAME f4 type=be_in time=4", structure)
        self.assertIn("ROLE f4 goal=文件夹", structure)
        self.assertIn("REL in(文件,文件夹)", structure)
        self.assertNotIn("REL in(文件,抽屉)", structure)
        self.assertEqual(prediction.answer, "文件在文件夹里。")

    def test_belief_revision_keeps_belief_history_and_uses_latest_belief(self) -> None:
        parser = default_neural_statement_parser()
        parsed = parser("小李起初相信药瓶在柜子里，随后认为药瓶在盒子里")

        self.assertIsNotNone(parsed)
        assert parsed is not None
        lines = linearize_statement_result(parsed)
        self.assertEqual(lines.count("FRAME believe"), 2)
        self.assertNotIn("FRAME say", lines)
        self.assertIn("ROLE proposition=药瓶在柜子里", lines)
        self.assertIn("ROLE proposition=药瓶在盒子里", lines)

        prediction = predict("小王先以为芯片在托盘里，后来认为芯片在盒子里。小王认为芯片在哪里？")
        structure = prediction.structure.linearize()
        self.assertIn("FRAME f1 type=believe time=1", structure)
        self.assertIn("ROLE f1 proposition=芯片在托盘里", structure)
        self.assertIn("FRAME f2 type=believe time=2", structure)
        self.assertIn("ROLE f2 proposition=芯片在盒子里", structure)
        self.assertNotIn("FRAME f1 type=say", structure)
        self.assertIn("QUERY belief_location(芯片,person=小王)", structure)
        self.assertIn("RULE belief_location_found", structure)
        self.assertEqual(prediction.answer, "小王认为芯片在盒子里。")

    def test_temporal_parallel_clauses_are_recovered_by_structural_statement_choice(self) -> None:
        self.assertEqual(
            split_query_candidate("小王先以为芯片在托盘里，后来认为芯片在盒子里"),
            ("小王先以为芯片在托盘里", "后来认为芯片在盒子里"),
        )
        self.assertEqual(
            split_query_candidate("小李起初相信药瓶在柜子里，随后认为药瓶在盒子里"),
            ("小李起初相信药瓶在柜子里", "随后认为药瓶在盒子里"),
        )

        prediction = predict("小王先以为芯片在托盘里，后来认为芯片在盒子里。小王认为芯片在哪里？")
        structure = prediction.structure.linearize()
        self.assertIn("FRAME f1 type=believe time=1", structure)
        self.assertIn("FRAME f2 type=believe time=2", structure)

    def test_conditional_slots_do_not_reuse_old_entities_and_consequent_state_expands(self) -> None:
        parser = default_neural_statement_parser()

        consequent = parser("芯片就在托盘里")
        self.assertIsNotNone(consequent)
        assert consequent is not None
        consequent_lines = linearize_statement_result(consequent)
        self.assertIn("ENTITY item=芯片", consequent_lines)
        self.assertNotIn("ENTITY item=芯片就", consequent_lines)
        self.assertIn("ROLE theme=芯片", consequent_lines)
        self.assertIn("ROLE goal=托盘", consequent_lines)

        conditional = parser("假如研究员把芯片放进托盘，芯片就在托盘里")
        self.assertIsNotNone(conditional)
        assert conditional is not None
        conditional_lines = linearize_statement_result(conditional)
        self.assertIn("ENTITY person=研究员", conditional_lines)
        self.assertIn("ROLE antecedent=研究员把芯片放进托盘", conditional_lines)
        self.assertIn("ROLE consequent=芯片就在托盘里", conditional_lines)
        self.assertNotIn("ROLE antecedent=小王把芯片放进托盘", conditional_lines)

        prediction = predict("如果小红把药瓶放进柜子，药瓶就在柜子里。小红把药瓶放进柜子。药瓶在哪里？")
        structure = prediction.structure.linearize()
        self.assertIn("FRAME f1 type=if_then time=1", structure)
        self.assertIn("ROLE f1 antecedent=小红把药瓶放进柜子", structure)
        self.assertIn("ROLE f1 consequent=药瓶就在柜子里", structure)
        self.assertIn("FRAME f4 type=be_in time=4", structure)
        self.assertIn("ROLE f4 theme=药瓶", structure)
        self.assertIn("ROLE f4 goal=柜子", structure)
        self.assertIn("REL in(药瓶,柜子)", structure)
        self.assertEqual(prediction.answer, "药瓶在柜子里。")

        no_antecedent = predict("如果小郭把芯片放进托盘，小王就把托盘带到实验室。芯片在哪里？")
        no_antecedent_structure = no_antecedent.structure.linearize()
        self.assertIn("FRAME f1 type=if_then time=1", no_antecedent_structure)
        self.assertNotIn("REL at(托盘,实验室)", no_antecedent_structure)
        self.assertEqual(no_antecedent.answer, "不知道芯片在哪里。")

    def test_incomplete_statement_boundary_does_not_write_profile_state(self) -> None:
        parser = default_neural_statement_parser()

        self.assertIsNone(parser("我想说"))
        self.assertIsNone(parser("我想说一下"))


def query_intents_for_split(examples, split: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for example in examples:
        if example.split != split or example.query is None:
            continue
        add_query_intents(counter, example.query)
    return counter


def add_query_intents(counter: Counter[str], query: Query) -> None:
    counter[query.intent] += 1
    for subquery in query.subqueries:
        add_query_intents(counter, subquery)


def statement_frames_for_split(examples, split: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for example in examples:
        if example.split != split:
            continue
        for frame in example.frames:
            counter[frame.frame_type] += 1
    return counter


def empty_statement_count(examples, split: str) -> int:
    return sum(1 for example in examples if example.split == split and not example.frames)


def pragmatic_acts_for_split(examples, split: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for example in examples:
        if example.split != split:
            continue
        for act in example.expected_pragmatic_acts:
            counter[act.act] += 1
    return counter


def episode_expected_frame_count(examples, split: str, act: str) -> int:
    return sum(
        1
        for example in examples
        if example.split == split
        and example.expected_frames
        and any(pragmatic_act.act == act for pragmatic_act in example.expected_pragmatic_acts)
    )


def episode_action_result_count(examples, split: str) -> int:
    return sum(1 for example in examples if example.split == split and example.action_result is not None)


def episode_action_result_status_count(examples, split: str, status: str) -> int:
    return sum(
        1
        for example in examples
        if example.split == split
        and example.action_result is not None
        and example.action_result.status == status
    )


def episode_pragmatic_qualifier_count(examples, split: str, qualifier: str) -> int:
    return sum(
        1
        for example in examples
        if example.split == split
        and any(qualifier in pragmatic_act.qualifiers for pragmatic_act in example.expected_pragmatic_acts)
    )


def query_dialog_targets_for_split(examples, split: str) -> set[str]:
    targets: set[str] = set()
    for example in examples:
        if example.split != split or example.query is None:
            continue
        if example.query.intent == "dialog_act":
            targets.add(example.query.target)
    return targets


def structural_pattern_examples_for_split(examples, split: str):
    return tuple(
        example
        for example in examples
        if example.split == split and example.source.startswith("structural_pattern_")
    )


def structural_pattern_sources(examples) -> set[str]:
    return {example.source for example in examples if example.source.startswith("structural_pattern_")}


def source_count_for_split(examples, source: str, split: str) -> int:
    return sum(1 for example in examples if example.source == source and example.split == split)


def intent_belief_predicates_for_split(records, split: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in records:
        if record.split != split or record.source != "structural_pattern_intent_context":
            continue
        for belief in record.belief_state:
            counter[belief.split("(", 1)[0]] += 1
    return counter


def intent_world_state_count(records, split: str) -> int:
    return sum(1 for record in records if record.split == split and record.world_state)


def intent_belief_contains_count(records, split: str, fragment: str) -> int:
    return sum(
        1
        for record in records
        if record.split == split and any(fragment in belief for belief in record.belief_state)
    )
