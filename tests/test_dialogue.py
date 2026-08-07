from __future__ import annotations

import unittest

from tests.support import *


class DialogueTest(unittest.TestCase):
    def test_dialog_act_queries_can_answer_without_domain_state(self) -> None:
        examples = (
            ("你好", "QUERY dialog_act(greeting)", "你好，我在。"),
            ("谢谢你", "QUERY dialog_act(thanks)", "不客气。"),
            ("再见", "QUERY dialog_act(farewell)", "再见。"),
            ("明天见", "QUERY dialog_act(farewell)", "再见。"),
            ("保重", "QUERY dialog_act(farewell)", "再见。"),
            ("你是谁？", "QUERY dialog_act(identity)", "我是结构智能原型，会把对话里的事实、状态、信念和问题先整理成结构再回答。"),
            (
                "你是人吗",
                "QUERY dialog_act(personhood)",
                "我不是真人，也没有人的身体和真实体验；我是 AI，会尽力用清楚、诚实的方式和你交流。",
            ),
            (
                "你是真人吗",
                "QUERY dialog_act(personhood)",
                "我不是真人，也没有人的身体和真实体验；我是 AI，会尽力用清楚、诚实的方式和你交流。",
            ),
            (
                "你会是人吗",
                "QUERY dialog_act(personhood)",
                "我不是真人，也没有人的身体和真实体验；我是 AI，会尽力用清楚、诚实的方式和你交流。",
            ),
            (
                "我没听懂",
                "QUERY dialog_act(clarification)",
                "抱歉没说清楚，我换个方式再讲一遍，你看这样是否更明白？",
            ),
            (
                "我没听明白",
                "QUERY dialog_act(clarification)",
                "抱歉没说清楚，我换个方式再讲一遍，你看这样是否更明白？",
            ),
            (
                "你能解释一下吗",
                "QUERY dialog_act(clarification)",
                "抱歉没说清楚，我换个方式再讲一遍，你看这样是否更明白？",
            ),
            (
                "你会想我吗",
                "QUERY dialog_act(affection)",
                "我不会像人一样想念你，但我会记得当前对话里的上下文；你回来找我时，我会认真接住你的话。",
            ),
            (
                "你会喜欢我吗",
                "QUERY dialog_act(affection)",
                "我不会像人一样想念你，但我会记得当前对话里的上下文；你回来找我时，我会认真接住你的话。",
            ),
            (
                "你爱我吗",
                "QUERY dialog_act(affection)",
                "我不会像人一样想念你，但我会记得当前对话里的上下文；你回来找我时，我会认真接住你的话。",
            ),
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

    def test_profile_name_statement_without_task_query_is_acknowledged(self) -> None:
        examples = (
            "我是小郭",
            "我是小郭，你知道吗",
            "我叫小郭，你知道吗",
            "我叫小郭。你知道吗？",
        )

        for text in examples:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn("REL name(我,小郭)", structure)
                self.assertIn("FRAME f1 type=profile_name time=1", structure)
                self.assertNotIn("QUERY dialog_act(greeting)", structure)
                self.assertEqual(prediction.answer, "我知道了，你叫小郭。")

    def test_profile_statement_with_identity_query_keeps_both_structures(self) -> None:
        prediction = predict("我叫小郭，你叫什么")

        structure = prediction.structure.linearize()
        self.assertIn("REL name(我,小郭)", structure)
        self.assertIn("FRAME f1 type=profile_name time=1", structure)
        self.assertIn("QUERY dialog_act(identity)", structure)
        self.assertEqual(prediction.answer, "我是结构智能原型，会把对话里的事实、状态、信念和问题先整理成结构再回答。")

    def test_profile_name_overwrites_and_preferences_can_be_corrected(self) -> None:
        prediction = predict("我叫小王。其实我叫小李。我喜欢咖啡。后来我不喜欢咖啡。我叫什么，我喜欢什么，我不喜欢什么？")

        structure = prediction.structure.linearize()
        self.assertIn("REL name(我,小李)", structure)
        self.assertNotIn("REL name(我,小王)", structure)
        self.assertNotIn("REL likes(我,咖啡)", structure)
        self.assertIn("REL dislikes(我,咖啡)", structure)
        self.assertEqual(prediction.answer, "你叫小李；我还不知道你喜欢什么；你不喜欢咖啡。")

    def test_direct_and_chat_memory_entries_can_be_written_and_reloaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = LearningPaths(
                memory_direct_data=Path(directory) / "memory_direct_examples.jsonl",
                memory_chat_data=Path(directory) / "memory_chat_examples.jsonl",
                memory_model=Path(directory) / "memory_model.json",
            )

            # direct memory feedback writes explicit state and reloads via memory model
            save_direct_memory_feedback(State("name", "我", "小王"), paths)
            loaded = load_memory_model(paths.memory_model)
            capabilities = default_capabilities(use_environment=False, use_memory=False).with_memory_states(*loaded.states)
            direct_prediction = predict("我叫什么", capabilities)
            self.assertEqual(direct_prediction.answer, "你叫小王。")

            # chat memory feedback sediments structure-derived states and reloads them
            seed_prediction = predict(
                "我叫小李。我叫什么？",
                default_capabilities(use_environment=False, use_memory=False),
            )
            result = save_chat_memory_feedback("我叫小李。我叫什么？", seed_prediction.structure, paths)
            loaded = load_memory_model(result.model_path)
            capabilities = default_capabilities(use_environment=False, use_memory=False).with_memory_states(*loaded.states)
            chat_prediction = predict("我叫什么", capabilities)
            self.assertGreaterEqual(result.entry_count, 1)
            self.assertEqual(chat_prediction.answer, "你叫小李。")

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

    def test_mixed_chat_fragments_preserve_task_core_and_record_profile_statement(self) -> None:
        # greeting fragment is dropped while real location query survives
        prediction = predict("小郭把芯片放进托盘。你好，我想知道芯片在哪里？")
        structure = prediction.structure.linearize()
        self.assertIn("QUERY location(芯片)", structure)
        self.assertNotIn("SUBQUERY dialog_act(greeting)", structure)
        self.assertEqual(prediction.answer, "芯片在托盘里。")

        for text in (
            "小郭把芯片放进托盘。谢谢，芯片在哪里？",
            "小郭把芯片放进托盘。再见，芯片在哪里？",
            "小郭把芯片放进托盘。我没听明白，芯片在哪里？",
        ):
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn("QUERY location(芯片)", structure)
                self.assertNotIn("SUBQUERY dialog_act", structure)
                self.assertEqual(prediction.answer, "芯片在托盘里。")

        # profile statement followed by dialog_act + profile query composes cleanly
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

    def test_causal_frames_cover_if_then_because_and_why_queries(self) -> None:
        cases = (
            # if_then fires when antecedent holds
            ("如果小郭把芯片放进托盘，小王就把托盘带到实验室。小郭把芯片放进托盘。芯片在哪里？",
             "FRAME f1 type=if_then time=1", "RULE container_moves_contents",
             "芯片在实验室的托盘里。", True),
            # if_then does not fire without antecedent
            ("如果小郭把芯片放进托盘，小王就把托盘带到实验室。芯片在哪里？",
             "FRAME f1 type=if_then time=1", None,
             "不知道芯片在哪里。", False),
            # because records cause and effect
            ("因为小王把托盘带到实验室，所以芯片在实验室的托盘里。芯片在哪里？",
             "FRAME f1 type=because time=1", None,
             "芯片在实验室的托盘里。", True),
            # why query uses causal frame
            ("因为小王把托盘带到实验室，所以芯片在实验室的托盘里。为什么芯片在实验室的托盘里？",
             "QUERY why(芯片在实验室的托盘里)", "RULE causal_explanation",
             "因为小王把托盘带到实验室。", True),
            # why query can explain from state chain
            ("小郭把芯片放进托盘。托盘被带到实验室。为什么芯片在实验室的托盘里？",
             "QUERY why(芯片在实验室的托盘里)", "RULE causal_explanation",
             "因为芯片在托盘里，而且托盘在实验室。", True),
        )
        for case in cases:
            text, expected, rule, answer, fires = case
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn(expected, structure)
                if rule:
                    self.assertIn(rule, structure)
                if not fires:
                    self.assertNotIn("REL at(托盘,实验室)", structure)
                self.assertEqual(prediction.answer, answer)

    def test_claim_source_is_separate_from_world_fact_and_can_be_queried(self) -> None:
        # claim does not update factual world state
        prediction = predict("小王说芯片在托盘里。芯片在哪里？")
        structure = prediction.structure.linearize()
        self.assertIn("FRAME f1 type=say time=1", structure)
        self.assertIn("QUERY location(芯片)", structure)
        self.assertEqual(prediction.answer, "不知道芯片在哪里。")

        # claim_source query finds speaker (incl. demonstrative reference)
        for text, answer in (
            ("小王说芯片在托盘里。谁说芯片在托盘里？", "小王说的。"),
            ("小王说这个芯片在托盘里。谁说这个芯片在托盘里？", "小王说的。"),
        ):
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn("QUERY claim_source(芯片在托盘里)", structure)
                self.assertIn("RULE claim_has_source", structure)
                self.assertEqual(prediction.answer, answer)

    def test_belief_views_are_isolated_sourceable_and_order_aware(self) -> None:
        # belief does not update factual world state
        prediction = predict("小王以为芯片在盒子里。芯片在哪里？")
        structure = prediction.structure.linearize()
        self.assertIn("FRAME f1 type=believe time=1", structure)
        self.assertIn("EVENT believe(小王,芯片在盒子里)", structure)
        self.assertNotIn("REL in(芯片,盒子)", structure)
        self.assertEqual(prediction.answer, "不知道芯片在哪里。")

        # belief_location query uses personal world view (surface variants)
        for question in (
            "小王认为芯片在哪里？",
            "小王相信芯片在什么地方？",
            "你知道的话，可以告诉我小王以为芯片在哪里吗？",
        ):
            with self.subTest(question=question):
                prediction = predict(f"小郭把芯片放进托盘。小王以为芯片在盒子里。{question}")
                structure = prediction.structure.linearize()
                self.assertIn("REL in(芯片,托盘)", structure)
                self.assertIn("QUERY belief_location(芯片,person=小王)", structure)
                self.assertIn("RULE belief_location_found", structure)
                self.assertEqual(prediction.answer, "小王认为芯片在盒子里。")

        # belief_source query finds all believers
        prediction = predict("小王认为芯片在盒子里。小郭相信芯片在盒子里。谁认为芯片在盒子里？")
        structure = prediction.structure.linearize()
        self.assertIn("QUERY belief_source(芯片在盒子里)", structure)
        self.assertIn("RULE belief_has_source", structure)
        self.assertEqual(prediction.answer, "小王和小郭这么认为。")

        # belief updates only that person view, in order
        prediction = predict("小王以为芯片在托盘里。小王后来认为芯片在盒子里。小王认为芯片在哪里？")
        structure = prediction.structure.linearize()
        self.assertIn("EVENT believe(小王,芯片在托盘里)", structure)
        self.assertIn("EVENT believe(小王,芯片在盒子里)", structure)
        self.assertEqual(prediction.answer, "小王认为芯片在盒子里。")

    def test_contradiction_query_covers_found_missing_and_no_contradiction_cases(self) -> None:
        cases = (
            ("小郭把芯片放进托盘。小王以为芯片在盒子里。有没有矛盾？",
             "RULE contradictions_found",
             "存在矛盾：小王认为芯片在盒子里，但事实是芯片在托盘里。"),
            ("小郭把芯片放进托盘。小王说芯片在盒子里。哪里有冲突？",
             "RULE contradictions_found",
             "存在矛盾：小王说芯片在盒子里，但事实是芯片在托盘里。"),
            ("小郭把芯片放进托盘。小王说芯片在托盘里。有没有矛盾？",
             "RULE no_contradictions",
             "没有发现矛盾。"),
            ("小郭把芯片放进托盘。芯片不在托盘里而在盒子里。有没有矛盾？",
             "RULE no_contradictions",
             "没有发现矛盾。"),
        )
        for text, rule, answer in cases:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn("QUERY contradictions(world)", structure)
                self.assertIn(rule, structure)
                self.assertEqual(prediction.answer, answer)

    def test_counterfactual_queries_replay_state_and_keep_world_views_separate(self) -> None:
        # counterfactual_location replays state without the named event
        cases = (
            {
                "name": "move_with_if",
                "text": "小郭把芯片放进托盘。小王把托盘带到实验室。如果小王没有把托盘带到实验室，芯片会在哪里？",
                "query": "QUERY counterfactual_location(芯片,without_event=move,actor=小王,theme=托盘,goal=实验室)",
                "rule": "RULE counterfactual_location_found",
                "answer": "如果没有这个事件，芯片会在托盘里。",
            },
            {
                "name": "move_omitted_if",
                "text": "小郭把芯片放进托盘。小王没有把托盘带到实验室，芯片会在哪里？",
                "query": "QUERY counterfactual_location(芯片,without_event=move,actor=小王,theme=托盘,goal=实验室)",
                "rule": "RULE counterfactual_location_found",
                "answer": "如果没有这个事件，芯片会在托盘里。",
            },
            {
                "name": "take_out_restores_container",
                "text": "小郭把芯片放进托盘。小王把芯片从托盘里取出。如果小王没有把芯片从托盘里取出，芯片会在哪里？",
                "query": "QUERY counterfactual_location(芯片,without_event=take_out,actor=小王,theme=芯片,source=托盘)",
                "rule": None,
                "answer": "如果没有这个事件，芯片会在托盘里。",
                "absent": "REL in(芯片,托盘)",
            },
            {
                "name": "put_in_makes_unknown",
                "text": "小郭把芯片放进托盘。如果小郭没有把芯片放进托盘，芯片会在哪里？",
                "query": "QUERY counterfactual_location",
                "rule": "RULE counterfactual_location_unknown",
                "answer": None,
            },
            {
                "name": "put_in_makes_unknown_alt_entity",
                "text": "小红把药瓶放进盒子。如果小红没有把药瓶放进盒子药瓶会在哪里？",
                "query": "QUERY counterfactual_location",
                "rule": "RULE counterfactual_location_unknown",
                "answer": None,
            },
        )
        for case in cases:
            with self.subTest(name=case["name"]):
                prediction = predict(case["text"])
                structure = prediction.structure.linearize()
                self.assertIn(case["query"], structure)
                if case["rule"]:
                    self.assertIn(case["rule"], structure)
                if case.get("absent"):
                    self.assertNotIn(case["absent"], structure)
                if case["answer"]:
                    self.assertEqual(prediction.answer, case["answer"])

        # fact, belief, claim, contradiction, and counterfactual stay separated by world view
        self.assertEqual(
            predict("小郭把芯片放进托盘。小王以为芯片在盒子里。芯片实际在哪里？").answer,
            "芯片在托盘里。",
        )
        self.assertEqual(
            predict("小郭把芯片放进托盘。小王以为芯片在盒子里。小王认为芯片在哪里？").answer,
            "小王认为芯片在盒子里。",
        )
        base = (
            "小郭把芯片放进托盘。小王把托盘带到实验室。"
            "小李说芯片在盒子里。小张认为芯片在托盘里。"
        )
        self.assertEqual(predict(base + "芯片在哪里？").answer, "芯片在实验室的托盘里。")
        self.assertEqual(predict(base + "小张认为芯片在哪里？").answer, "小张认为芯片在托盘里。")
        self.assertEqual(
            predict(base + "有没有矛盾？").answer,
            "存在矛盾：小李说芯片在盒子里，但事实是芯片在实验室的托盘里。",
        )
        self.assertEqual(
            predict(base + "如果小王没有把托盘带到实验室，芯片会在哪里？").answer,
            "如果没有这个事件，芯片会在托盘里。",
        )

    def test_compound_query_keeps_source_order_ignores_filler_and_mixed_views(self) -> None:
        # multiple subqueries are kept in source order
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

        # filler fragments are dropped while real subqueries survive
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

        # fact, belief, and counterfactual views compose in a single compound query
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
