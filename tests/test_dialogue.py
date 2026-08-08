from __future__ import annotations

import unittest

from tests.support import *


class DialogueTest(unittest.TestCase):
    def test_dialog_act_queries_can_answer_without_domain_state(self) -> None:
        examples = (
            ("你好", "QUERY dialog_act(greeting)", "你好呀，很高兴见到你，有什么我可以帮你的吗？"),
            ("谢谢你", "QUERY dialog_act(thanks)", "不客气，能帮到你就好，有需要随时找我。"),
            ("再见", "QUERY dialog_act(farewell)", "再见，很高兴和你聊天，欢迎随时再来找我。"),
            ("明天见", "QUERY dialog_act(farewell)", "再见，很高兴和你聊天，欢迎随时再来找我。"),
            ("保重", "QUERY dialog_act(farewell)", "再见，很高兴和你聊天，欢迎随时再来找我。"),
            ("你是谁？", "QUERY dialog_act(identity)", "我是你的 AI 助手，可以陪你聊天、回答问题、帮你处理各种任务。"),
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
                "我可以陪你聊天、回答问题、整理信息、写作、翻译、做总结，还能帮你梳理思路。",
            ),
            (
                "你能干嘛",
                "QUERY dialog_act(capabilities)",
                "我可以陪你聊天、回答问题、整理信息、写作、翻译、做总结，还能帮你梳理思路。",
            ),
            (
                "你好，你能干嘛",
                "QUERY dialog_act(capabilities)",
                "我可以陪你聊天、回答问题、整理信息、写作、翻译、做总结，还能帮你梳理思路。",
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

    def test_activity_preference_statement_is_queryable_chat_state(self) -> None:
        capabilities = default_capabilities(use_environment=False, use_memory=False)
        name_turn = predict("我叫郭士君", capabilities)
        capabilities = capabilities_with_working_turn(capabilities, "我叫郭士君", name_turn.structure.states)

        first = predict("我喜欢徒步", capabilities)
        first_structure = first.structure.linearize()
        self.assertIn("REL name(我,郭士君)", first_structure)
        self.assertIn("REL likes(我,徒步)", first_structure)
        self.assertIn("FRAME f1 type=profile_like time=1", first_structure)
        self.assertEqual(first.answer, "我知道了。")

        capabilities = capabilities_with_working_turn(capabilities, "我喜欢徒步", first.structure.states)
        second = predict("我喜欢什么", capabilities)

        structure = second.structure.linearize()
        self.assertIn("REL likes(我,徒步)", structure)
        self.assertIn("QUERY profile(我,attribute=likes)", structure)
        self.assertEqual(second.answer, "你喜欢徒步。")

    def test_profile_like_coordination_preserves_all_values(self) -> None:
        prediction = predict("我爱爬山，也爱游泳")

        structure = prediction.structure.linearize()
        self.assertIn("FRAME f1 type=profile_like time=1", structure)
        self.assertIn("FRAME f2 type=profile_like time=2", structure)
        self.assertIn("REL likes(我,爬山)", structure)
        self.assertIn("REL likes(我,游泳)", structure)
        self.assertEqual(prediction.answer, "我知道了。")

    def test_unresolved_profile_pronouns_request_contextual_referent_choice(self) -> None:
        capabilities = default_capabilities(use_environment=False, use_memory=False)
        first = predict("我爱爬山，也爱游泳", capabilities)
        capabilities = capabilities_with_working_turn(capabilities, "我爱爬山，也爱游泳", first.structure.states)

        for text in ("它是什么", "它对于身体的好处是啥"):
            with self.subTest(text=text):
                follow_up = predict(text, capabilities)
                structure = follow_up.structure.linearize()
                self.assertIn("ENTITY unresolved_reference=它", structure)
                self.assertIn("REL likes(我,爬山)", structure)
                self.assertIn("REL likes(我,游泳)", structure)
                self.assertIn(
                    "PRAGMATIC_ACT ambiguous_reference(它,missing=referent,candidates=爬山|游泳,depends_on=focus,response_policy=ask_clarification)",
                    structure,
                )
                self.assertIn("RULE pragmatic_response_ask_clarification", structure)
                self.assertEqual(follow_up.answer, "你说的是爬山还是游泳？")

    def test_self_profile_queries_use_discourse_participant_context(self) -> None:
        examples = ("我是谁", "我叫啥", "我叫什么")

        for text in examples:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn("ENTITY self=我", structure)
                self.assertIn("QUERY profile(我,attribute=name)", structure)
                self.assertIn("RULE profile_name_unknown", structure)
                self.assertEqual(prediction.answer, "我还不知道你叫什么。")

    def test_real_chinese_profile_name_statements_are_queryable(self) -> None:
        examples = ("我叫郭士君。我是谁？", "我是郭士君。我叫啥？")

        for text in examples:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn("REL name(我,郭士君)", structure)
                self.assertIn("FRAME f1 type=profile_name time=1", structure)
                self.assertIn("QUERY profile(我,attribute=name)", structure)
                self.assertEqual(prediction.answer, "你叫郭士君。")

    def test_episode_pragmatic_supervision_handles_incomplete_and_ambiguous_inputs(self) -> None:
        cases = (
            (
                "我是……",
                "PRAGMATIC_ACT incomplete_utterance(profile_name,missing=profile_value,response_policy=wait_for_completion)",
                "RULE pragmatic_response_wait_for_completion",
                "我先等你把话说完整。",
            ),
            (
                "那个呢？",
                "PRAGMATIC_ACT ambiguous_reference(那个,missing=referent,depends_on=focus,response_policy=ask_clarification)",
                "RULE pragmatic_response_ask_clarification",
                "这句话还缺少可计算的对象或上下文，你想让我具体处理什么？",
            ),
            (
                "帮我弄一下",
                "PRAGMATIC_ACT underspecified_action_request(task,missing=object,missing=operation,response_policy=ask_clarification)",
                "RULE pragmatic_response_ask_clarification",
                "这句话还缺少可计算的对象或上下文，你想让我具体处理什么？",
            ),
            (
                "你懂吧？",
                "PRAGMATIC_ACT confirm_understanding(shared_ground,not_capability_query=true,response_policy=confirm)",
                "RULE pragmatic_response_confirm",
                "我理解你是在确认我是否跟上了。",
            ),
            (
                "你",
                "PRAGMATIC_ACT incomplete_utterance(addressee,missing=predicate,response_policy=wait_for_completion)",
                "RULE pragmatic_response_wait_for_completion",
                "我先等你把话说完整。",
            ),
            (
                "我想说",
                "PRAGMATIC_ACT incomplete_utterance(user_intention,intent=say,missing=content,response_policy=wait_for_completion)",
                "RULE pragmatic_response_wait_for_completion",
                "我先等你把话说完整。",
            ),
            (
                "我想了解",
                "PRAGMATIC_ACT incomplete_utterance(user_intention,intent=learn,missing=topic,response_policy=wait_for_completion)",
                "RULE pragmatic_response_wait_for_completion",
                "我先等你把话说完整。",
            ),
            (
                "小郭呢",
                "PRAGMATIC_ACT underspecified_reference_query(小郭,missing=query_intent,response_policy=ask_clarification)",
                "RULE pragmatic_response_ask_clarification",
                "你想问小郭的哪方面？比如位置、状态、归属或相关信息。",
            ),
            (
                "爬山呢",
                "PRAGMATIC_ACT underspecified_reference_query(爬山,missing=query_intent,response_policy=ask_clarification)",
                "RULE pragmatic_response_ask_clarification",
                "你想问爬山的哪方面？比如位置、状态、归属或相关信息。",
            ),
            (
                "实验室呢",
                "PRAGMATIC_ACT underspecified_reference_query(实验室,missing=query_intent,response_policy=ask_clarification)",
                "RULE pragmatic_response_ask_clarification",
                "你想问实验室的哪方面？比如位置、状态、归属或相关信息。",
            ),
            (
                "游泳呢",
                "PRAGMATIC_ACT underspecified_reference_query(游泳,missing=query_intent,response_policy=ask_clarification)",
                "RULE pragmatic_response_ask_clarification",
                "你想问游泳的哪方面？比如位置、状态、归属或相关信息。",
            ),
        )

        for text, pragmatic_line, rule, answer in cases:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn(pragmatic_line, structure)
                self.assertIn(rule, structure)
                self.assertEqual(prediction.answer, answer)

    def test_incomplete_inputs_do_not_ack_stale_profile_memory(self) -> None:
        capabilities = default_capabilities(use_environment=False, use_memory=False)
        name_turn = predict("我叫郭士君", capabilities)
        capabilities = capabilities_with_working_turn(capabilities, "我叫郭士君", name_turn.structure.states)
        like_turn = predict("我喜欢徒步", capabilities)
        capabilities = capabilities_with_working_turn(capabilities, "我喜欢徒步", like_turn.structure.states)

        for text in ("你", "我想说", "我想了解"):
            with self.subTest(text=text):
                prediction = predict(text, capabilities)
                structure = prediction.structure.linearize()
                self.assertIn("REL name(我,郭士君)", structure)
                self.assertIn("REL likes(我,徒步)", structure)
                self.assertIn("RULE pragmatic_response_wait_for_completion", structure)
                self.assertEqual(prediction.answer, "我先等你把话说完整。")

    def test_episode_pragmatic_supervision_coexists_with_query_structure(self) -> None:
        prediction = predict("你会想我吗")

        structure = prediction.structure.linearize()
        self.assertIn("QUERY dialog_act(affection)", structure)
        self.assertIn("PRAGMATIC_ACT relationship_probe(affection,not_capability_query=true,response_policy=answer)", structure)
        self.assertEqual(
            prediction.answer,
            "我不会像人一样想念你，但我会记得当前对话里的上下文；你回来找我时，我会认真接住你的话。",
        )

    def test_working_memory_answers_previous_user_turn_query(self) -> None:
        capabilities = capabilities_with_last_user_utterance(
            default_capabilities(use_environment=False, use_memory=False),
            "你是谁",
        )

        prediction = predict("我刚刚说的啥", capabilities)

        structure = prediction.structure.linearize()
        self.assertIn("REL last_user_utterance(user,你是谁)", structure)
        self.assertIn("PRAGMATIC_ACT recall_previous_turn(user,turn=previous,response_policy=answer)", structure)
        self.assertIn("RULE pragmatic_recall_previous_turn_found", structure)
        self.assertEqual(prediction.answer, "你刚刚说的是：你是谁")

    def test_working_memory_carries_profile_state_between_turns(self) -> None:
        capabilities = default_capabilities(use_environment=False, use_memory=False)
        first = predict("我叫郭士君", capabilities)
        capabilities = capabilities_with_working_turn(capabilities, "我叫郭士君", first.structure.states)

        second = predict("我是谁", capabilities)

        structure = second.structure.linearize()
        self.assertIn("REL name(我,郭士君)", structure)
        self.assertIn("REL last_user_utterance(user,我叫郭士君)", structure)
        self.assertIn("QUERY profile(我,attribute=name)", structure)
        self.assertEqual(second.answer, "你叫郭士君。")

    def test_follow_up_principle_question_reuses_previous_query_focus(self) -> None:
        capabilities = default_capabilities(use_environment=False, use_memory=False).with_answerers(
            default_learned_memory_knowledge_answerer("data/memory_knowledge_model.json")
        )
        first = predict("我看到了铁生锈", capabilities)
        capabilities = capabilities_with_working_turn(
            capabilities,
            "我看到了铁生锈",
            first.structure.states,
            first.structure.query,
        )

        second = predict("我想了解下原理", capabilities)

        structure = second.structure.linearize()
        self.assertIn("QUERY why(铁会生锈,type=why)", structure)
        self.assertNotIn("PRAGMATIC_ACT incomplete_utterance", structure)
        self.assertEqual(second.answer, "铁和空气里的氧、水发生反应后，会生成疏松的氧化物，也就是锈。")

    def test_focus_topic_ellipsis_reuses_previous_query_intent(self) -> None:
        capabilities = default_capabilities(use_environment=False, use_memory=False)
        first = predict("芯片在哪里？", capabilities)
        capabilities = capabilities_with_working_turn(
            capabilities,
            "芯片在哪里？",
            first.structure.states,
            first.structure.query,
        )

        follow_up = predict("那个呢", capabilities)

        structure = follow_up.structure.linearize()
        self.assertIn("REL focus_topic(user,芯片)", structure)
        self.assertIn("REL focus_query_intent(user,location)", structure)
        self.assertIn("QUERY location(芯片)", structure)
        self.assertNotIn("PRAGMATIC_ACT ambiguous_reference", structure)
        self.assertEqual(follow_up.answer, "不知道芯片在哪里。")

        known = predict("小王把芯片放进盒子。盒子里有什么？", capabilities)
        capabilities = capabilities_with_working_turn(
            capabilities,
            "小王把芯片放进盒子。盒子里有什么？",
            known.structure.states,
            known.structure.query,
        )

        pronoun_follow_up = predict("它呢", capabilities)

        pronoun_structure = pronoun_follow_up.structure.linearize()
        self.assertIn("REL focus_topic(user,盒子)", pronoun_structure)
        self.assertIn("REL focus_query_intent(user,contents)", pronoun_structure)
        self.assertIn("QUERY contents(盒子)", pronoun_structure)
        self.assertEqual(pronoun_follow_up.answer, "盒子里至少有芯片。")

    def test_observation_query_can_replace_stale_profile_focus(self) -> None:
        capabilities = default_capabilities(use_environment=False, use_memory=False).with_answerers(
            default_learned_memory_knowledge_answerer("data/memory_knowledge_model.json")
        )
        name_turn = predict("我叫郭士君", capabilities)
        capabilities = capabilities_with_working_turn(capabilities, "我叫郭士君", name_turn.structure.states, name_turn.structure.query)
        like_turn = predict("我喜欢徒步", capabilities)
        capabilities = capabilities_with_working_turn(capabilities, "我喜欢徒步", like_turn.structure.states, like_turn.structure.query)
        profile_query = predict("我喜欢什么", capabilities)
        capabilities = capabilities_with_working_turn(
            capabilities,
            "我喜欢什么",
            profile_query.structure.states,
            profile_query.structure.query,
        )

        observation = predict("我看到了铁生锈", capabilities)
        capabilities = capabilities_with_working_turn(
            capabilities,
            "我看到了铁生锈",
            observation.structure.states,
            observation.structure.query,
        )
        followup = predict("我想了解下原理", capabilities)

        self.assertIn("QUERY why(铁会生锈,type=why)", observation.structure.linearize())
        self.assertEqual(observation.answer, "铁和空气里的氧、水发生反应后，会生成疏松的氧化物，也就是锈。")
        self.assertIn("QUERY why(铁会生锈,type=why)", followup.structure.linearize())
        self.assertEqual(followup.answer, "铁和空气里的氧、水发生反应后，会生成疏松的氧化物，也就是锈。")

    def test_principle_request_without_focus_waits_for_completion(self) -> None:
        prediction = predict("我想了解下原理", default_capabilities(use_environment=False, use_memory=False))

        structure = prediction.structure.linearize()
        self.assertIn("PRAGMATIC_ACT incomplete_utterance(user_intention,intent=learn,missing=topic,response_policy=wait_for_completion)", structure)
        self.assertIn("RULE pragmatic_response_wait_for_completion", structure)
        self.assertEqual(prediction.answer, "我先等你把话说完整。")

    def test_pragmatic_acts_are_deduped_across_local_neural_provider_layers(self) -> None:
        capabilities = with_neural_boundary(
            default_capabilities(use_environment=False, use_memory=False),
            make_model(),
        )

        prediction = predict("那个呢", capabilities)

        structure = prediction.structure.linearize()
        self.assertEqual(structure.count("PRAGMATIC_ACT ambiguous_reference"), 1)
        self.assertEqual(prediction.answer, "这句话还缺少可计算的对象或上下文，你想让我具体处理什么？")

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
                self.assertEqual(prediction.answer, "我知道了。")

    def test_profile_statement_with_identity_query_keeps_both_structures(self) -> None:
        prediction = predict("我叫小郭，你叫什么")

        structure = prediction.structure.linearize()
        self.assertIn("REL name(我,小郭)", structure)
        self.assertIn("FRAME f1 type=profile_name time=1", structure)
        self.assertIn("QUERY dialog_act(identity)", structure)
        self.assertEqual(prediction.answer, "我是你的 AI 助手，可以陪你聊天、回答问题、帮你处理各种任务。")

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

    def test_continuous_dialogue_keeps_event_history_and_state_updates(self) -> None:
        capabilities = default_capabilities(use_environment=False, use_memory=False)

        turns = (
            ("星图回声？", "你想问星图回声的哪方面？比如位置、状态、归属或相关信息。"),
            ("小郭把芯片放进托盘", "我知道了。"),
            ("你好", "你好呀，很高兴见到你，有什么我可以帮你的吗？"),
            ("托盘被带到实验室", "我知道了。"),
            ("谁把芯片放进托盘？", "小郭把芯片放进托盘。"),
            ("芯片在哪里？", "芯片在实验室的托盘里。"),
            ("总结一下", "已知：小郭把芯片放进托盘；托盘被带到实验室。"),
        )

        for text, answer in turns:
            with self.subTest(text=text):
                prediction = predict(text, capabilities)
                structure = prediction.structure.linearize()
                self.assertEqual(prediction.answer, answer)
                if text == "星图回声？":
                    self.assertIn("PRAGMATIC_ACT underspecified_reference_query(星图回声,missing=query_intent,response_policy=ask_clarification)", structure)
                    self.assertIn("RULE pragmatic_response_ask_clarification", structure)
                elif text == "小郭把芯片放进托盘":
                    self.assertIn("RULE structural_update_acknowledgement", structure)
                    self.assertIn("FRAME f1 type=put_in time=1", structure)
                elif text == "托盘被带到实验室":
                    self.assertIn("REL at(托盘,实验室)", structure)
                    self.assertIn("FRAME f3 type=move time=3", structure)
                elif text == "谁把芯片放进托盘？":
                    self.assertIn("QUERY actor_for_event(put_in,item=芯片,holder=托盘)", structure)
                    self.assertIn("RULE event_actor_matches", structure)
                elif text == "芯片在哪里？":
                    self.assertIn("QUERY location(芯片)", structure)
                    self.assertIn("RULE container_moves_contents", structure)
                elif text == "总结一下":
                    self.assertIn("QUERY dialog_act(summary)", structure)
                    self.assertIn("RULE conversation_summary", structure)
                capabilities = capabilities_with_working_turn(
                    capabilities,
                    text,
                    prediction.structure.states,
                    prediction.structure.query,
                    prediction.structure.frames,
                )

        # profile statement followed by dialog_act + profile query composes cleanly
        prediction = predict("我叫小王，你能做什么？我叫什么？")
        structure = prediction.structure.linearize()
        self.assertIn("FRAME f1 type=profile_name time=1", structure)
        self.assertIn("QUERY compound(multi)", structure)
        self.assertIn("SUBQUERY dialog_act(capabilities)", structure)
        self.assertIn("SUBQUERY profile(我,attribute=name)", structure)
        self.assertEqual(
            prediction.answer,
            "我可以陪你聊天、回答问题、整理信息、写作、翻译、做总结，还能帮你梳理思路；你叫小王。",
        )

    def test_continuous_dialogue_keeps_profile_acknowledgements_natural(self) -> None:
        capabilities = default_capabilities(use_environment=False, use_memory=False)
        turns = (
            ("你好", "你好呀，很高兴见到你，有什么我可以帮你的吗？"),
            ("我叫郭士君", "我知道了。"),
            ("我喜欢徒步", "我知道了。"),
            ("小郭把芯片放进托盘", "我知道了。"),
            ("托盘被带到实验室", "我知道了。"),
            ("你能做什么？", "我可以陪你聊天、回答问题、整理信息、写作、翻译、做总结，还能帮你梳理思路。"),
            ("我叫什么，我喜欢什么？", "你叫郭士君；你喜欢徒步。"),
        )

        for text, answer in turns:
            with self.subTest(text=text):
                prediction = predict(text, capabilities)
                self.assertEqual(prediction.answer, answer)
                capabilities = capabilities_with_working_turn(
                    capabilities,
                    text,
                    prediction.structure.states,
                    prediction.structure.query,
                    prediction.structure.frames,
                )

    def test_daily_task_requests_do_not_ack_stale_profile_memory(self) -> None:
        capabilities = default_capabilities(use_environment=False, use_memory=False)
        turns = (
            ("我叫张三", "我知道了。"),
            ("我叫什么", "你叫张三。"),
            ("你能帮我写邮件吗", "当然可以。你把收件人、目的和大致内容告诉我，我来帮你起草。"),
            ("帮我写封邮件", "当然可以。你把收件人、目的和大致内容告诉我，我来帮你起草。"),
            (
                "能帮我想想明天吃什么吗",
                "可以呀。你想吃清淡点还是有满足感一点？不知道的话，明天可以先考虑粥粉面、简餐或热汤这几类。",
            ),
            ("你会记住我吗", "我会记住当前对话里你告诉我的信息，并在后续聊天里尽量接上上下文。"),
        )

        for text, answer in turns:
            with self.subTest(text=text):
                prediction = predict(text, capabilities)
                structure = prediction.structure.linearize()
                self.assertEqual(prediction.answer, answer)
                if text == "你能帮我写邮件吗":
                    self.assertIn("QUERY dialog_act(task_request,task=email)", structure)
                    self.assertIn("RULE dialog_task_request", structure)
                    self.assertNotIn("RULE structural_update_acknowledgement", structure)
                    self.assertNotIn("QUERY profile(我,attribute=dislikes)", structure)
                elif text == "帮我写封邮件":
                    self.assertIn("QUERY dialog_act(task_request,task=email)", structure)
                elif text == "能帮我想想明天吃什么吗":
                    self.assertIn("QUERY dialog_act(meal_suggestion)", structure)
                elif text == "你会记住我吗":
                    self.assertIn("QUERY dialog_act(memory_capability)", structure)
                capabilities = capabilities_with_working_turn(
                    capabilities,
                    text,
                    prediction.structure.states,
                    prediction.structure.query,
                    prediction.structure.frames,
                )

    def test_meal_suggestion_uses_semantic_structure_not_profile_dislikes(self) -> None:
        capabilities = default_capabilities(use_environment=False, use_memory=False)
        name = predict("我叫张三", capabilities)
        capabilities = capabilities_with_working_turn(
            capabilities,
            "我叫张三",
            name.structure.states,
            name.structure.query,
            name.structure.frames,
        )
        seed = predict("能帮我想想明天吃什么吗", capabilities)
        capabilities = capabilities_with_working_turn(
            capabilities,
            "能帮我想想明天吃什么吗",
            seed.structure.states,
            seed.structure.query,
            seed.structure.frames,
        )

        examples = (
            "我明天不知道吃啥，你有什么建议吗",
            "明天不知道吃什么，有什么建议吗",
            "不知道吃啥，你有什么建议",
            "我不知道吃什么",
            "给我推荐明天吃什么",
            "明天吃啥比较好",
            "我不知道午饭吃什么，你帮我推荐一下",
        )

        for text in examples:
            with self.subTest(text=text):
                prediction = predict(text, capabilities)
                structure = prediction.structure.linearize()
                self.assertEqual(
                    prediction.answer,
                    "可以呀。你想吃清淡点还是有满足感一点？不知道的话，明天可以先考虑粥粉面、简餐或热汤这几类。",
                )
                self.assertIn("QUERY dialog_act(meal_suggestion)", structure)
                self.assertIn("RULE dialog_meal_suggestion", structure)
                self.assertNotIn("QUERY profile(我,attribute=dislikes)", structure)
                self.assertNotIn("RULE profile_dislikes_unknown", structure)

    def test_meal_suggestion_preference_followup_uses_dialog_focus(self) -> None:
        capabilities = default_capabilities(use_environment=False, use_memory=False)
        seed = predict("明天吃东西你有什么建议吗", capabilities)
        self.assertEqual(
            seed.answer,
            "可以呀。你想吃清淡点还是有满足感一点？不知道的话，明天可以先考虑粥粉面、简餐或热汤这几类。",
        )
        capabilities = capabilities_with_working_turn(
            capabilities,
            "明天吃东西你有什么建议吗",
            seed.structure.states,
            seed.structure.query,
            seed.structure.frames,
        )

        cases = (
            (
                "清淡点的",
                "QUERY dialog_act(meal_suggestion,preference=light)",
                "清淡点的话，可以考虑小米粥配鸡蛋、番茄鸡蛋面、青菜豆腐汤，或者蒸鱼配米饭。",
            ),
            (
                "别太油的",
                "QUERY dialog_act(meal_suggestion,preference=light)",
                "清淡点的话，可以考虑小米粥配鸡蛋、番茄鸡蛋面、青菜豆腐汤，或者蒸鱼配米饭。",
            ),
            (
                "有满足感一点",
                "QUERY dialog_act(meal_suggestion,preference=rich)",
                "想吃得满足一点，可以考虑牛肉饭、鸡腿饭、热汤面，或者一份有主食和蛋白质的简餐。",
            ),
        )

        for text, query_line, answer in cases:
            with self.subTest(text=text):
                prediction = predict(text, capabilities)
                structure = prediction.structure.linearize()
                self.assertIn("REL focus_dialog_act(user,meal_suggestion)", structure)
                self.assertIn(query_line, structure)
                self.assertIn("RULE dialog_meal_suggestion", structure)
                self.assertEqual(prediction.answer, answer)

    def test_meal_suggestion_can_request_more_options_after_preference(self) -> None:
        capabilities = default_capabilities(use_environment=False, use_memory=False)
        seed = predict("马上中午了，吃啥有好的建议吗", capabilities)
        capabilities = capabilities_with_working_turn(
            capabilities,
            "马上中午了，吃啥有好的建议吗",
            seed.structure.states,
            seed.structure.query,
            seed.structure.frames,
        )
        first = predict("清淡点吧", capabilities)
        self.assertIn("QUERY dialog_act(meal_suggestion,preference=light)", first.structure.linearize())
        capabilities = capabilities_with_working_turn(
            capabilities,
            "清淡点吧",
            first.structure.states,
            first.structure.query,
            first.structure.frames,
        )

        followups = (
            ("还有别的建议吗", "QUERY dialog_act(meal_suggestion,preference=light,request=alternative)", "也可以换成虾仁蒸蛋、冬瓜丸子汤、鸡丝凉面，或者一份青菜瘦肉粥。"),
            ("换几个", "QUERY dialog_act(meal_suggestion,preference=light,request=alternative)", "也可以换成虾仁蒸蛋、冬瓜丸子汤、鸡丝凉面，或者一份青菜瘦肉粥。"),
        )

        for text, query_line, answer in followups:
            with self.subTest(text=text):
                prediction = predict(text, capabilities)
                structure = prediction.structure.linearize()
                self.assertIn("REL focus_dialog_act(user,meal_suggestion)", structure)
                self.assertIn(query_line, structure)
                self.assertIn("RULE dialog_meal_suggestion", structure)
                self.assertEqual(prediction.answer, answer)

    def test_meal_preference_fragment_requires_meal_dialog_focus(self) -> None:
        capabilities = default_capabilities(use_environment=False, use_memory=False)

        with self.assertRaises(ParseError):
            predict("清淡点的", capabilities)

    def test_previous_turn_recall_does_not_block_next_profile_statement(self) -> None:
        capabilities = default_capabilities(use_environment=False, use_memory=False)
        turns = (
            ("我叫小李", "我知道了。"),
            ("你好", "你好呀，很高兴见到你，有什么我可以帮你的吗？"),
            ("我刚刚说什么？", "你刚刚说的是：你好"),
            ("我喜欢游泳", "我知道了。"),
            ("我叫什么，我喜欢什么？", "你叫小李；你喜欢游泳。"),
        )

        for text, answer in turns:
            with self.subTest(text=text):
                prediction = predict(text, capabilities)
                self.assertEqual(prediction.answer, answer)
                capabilities = capabilities_with_working_turn(
                    capabilities,
                    text,
                    prediction.structure.states,
                    prediction.structure.query,
                    prediction.structure.frames,
                )

    def test_owner_followup_prefers_explicit_item_over_stale_topic(self) -> None:
        capabilities = default_capabilities(use_environment=False, use_memory=False)
        turns = (
            ("小红把药瓶交给医生", "我知道了。"),
            ("现在谁拥有药瓶？", "医生拥有药瓶。"),
            ("医生把药瓶交给老师", "我知道了。"),
            ("现在谁拥有药瓶？", "老师拥有药瓶。"),
        )

        for text, answer in turns:
            with self.subTest(text=text):
                prediction = predict(text, capabilities)
                structure = prediction.structure.linearize()
                self.assertEqual(prediction.answer, answer)
                if text == "现在谁拥有药瓶？":
                    self.assertIn("QUERY owner(药瓶)", structure)
                    self.assertIn("RULE transfer_changes_owner", structure)
                capabilities = capabilities_with_working_turn(
                    capabilities,
                    text,
                    prediction.structure.states,
                    prediction.structure.query,
                    prediction.structure.frames,
                )

    def test_daily_profile_name_nickname_and_preference_phrasing(self) -> None:
        capabilities = default_capabilities(use_environment=False, use_memory=False)
        turns = (
            ("我叫陈一诺", "我知道了。"),
            ("你记得我叫什么吗？", "你叫陈一诺。"),
            ("其实你叫我一诺就行", "我知道了。"),
            ("我最近喜欢喝拿铁", "我知道了。"),
            ("也爱看纪录片", "我知道了。"),
            ("我不太喜欢熬夜", "我知道了。"),
            ("我叫什么？", "你叫一诺。"),
            ("我喜欢什么，不喜欢什么？", "你喜欢拿铁和纪录片；你不喜欢熬夜。"),
        )

        for text, answer in turns:
            with self.subTest(text=text):
                prediction = predict(text, capabilities)
                self.assertEqual(prediction.answer, answer)
                capabilities = capabilities_with_working_turn(
                    capabilities,
                    text,
                    prediction.structure.states,
                    prediction.structure.query,
                    prediction.structure.frames,
                )

    def test_daily_empathy_pause_and_recall_phrasing(self) -> None:
        capabilities = default_capabilities(use_environment=False, use_memory=False)
        turns = (
            ("我有点烦", "听你这么说我也心疼，难过的时候不用硬扛，想说说看吗？我陪着你。"),
            ("你能安慰我吗", "听你这么说我也心疼，难过的时候不用硬扛，想说说看吗？我陪着你。"),
            ("算了", "没关系的，不想做就先放一放，愿意告诉我是什么让你不想做吗？"),
            ("我刚刚说了啥？", "你刚刚说的是：算了"),
            ("等一下", "好，我等你。"),
            ("我今天有点累", "听你这么说我也心疼，难过的时候不用硬扛，想说说看吗？我陪着你。"),
            ("我想先缓一缓", "好，我等你。"),
            ("我先去忙了", "再见，很高兴和你聊天，欢迎随时再来找我。"),
        )

        for text, answer in turns:
            with self.subTest(text=text):
                prediction = predict(text, capabilities)
                self.assertEqual(prediction.answer, answer)
                capabilities = capabilities_with_working_turn(
                    capabilities,
                    text,
                    prediction.structure.states,
                    prediction.structure.query,
                    prediction.structure.frames,
                )

    def test_daily_task_phrasing_corrections_and_person_inventory(self) -> None:
        capabilities = default_capabilities(use_environment=False, use_memory=False)
        turns = (
            ("刚才小周把钥匙放到抽屉里", "我知道了。"),
            ("后来抽屉被搬到办公室", "我知道了。"),
            ("那钥匙现在在哪儿？", "钥匙在办公室的抽屉里。"),
            ("抽屉里都有啥？", "抽屉里至少有钥匙。"),
            ("其实钥匙不在抽屉里", "我知道了。"),
            ("钥匙在哪里？", "不知道钥匙在哪里。"),
        )

        for text, answer in turns:
            with self.subTest(text=text):
                prediction = predict(text, capabilities)
                self.assertEqual(prediction.answer, answer)
                capabilities = capabilities_with_working_turn(
                    capabilities,
                    text,
                    prediction.structure.states,
                    prediction.structure.query,
                    prediction.structure.frames,
                )

        ownership = default_capabilities(use_environment=False, use_memory=False)
        for text, answer in (
            ("妈妈把伞交给爸爸", "我知道了。"),
            ("爸爸又把伞交给孩子", "我知道了。"),
            ("现在伞归谁？", "孩子拥有伞。"),
            ("爸爸手里有什么？", "不知道爸爸手里有什么。"),
            ("孩子手里有什么？", "孩子手里有伞。"),
        ):
            with self.subTest(text=text):
                prediction = predict(text, ownership)
                self.assertEqual(prediction.answer, answer)
                ownership = capabilities_with_working_turn(
                    ownership,
                    text,
                    prediction.structure.states,
                    prediction.structure.query,
                    prediction.structure.frames,
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

    def test_belief_propositions_materialize_scoped_structure_without_leaking_fact_state(self) -> None:
        prediction = predict("小王认为芯片被放进盒子了。小王认为芯片在哪里？")
        structure = prediction.structure.linearize()
        self.assertIn("SCOPED_FRAME f1 kind=belief owner=小王 proposition=芯片在盒子里 type=be_in", structure)
        self.assertIn("SCOPED_STATE f1 kind=belief owner=小王 proposition=芯片在盒子里 STATE in(芯片,盒子)", structure)
        self.assertNotIn("REL in(芯片,盒子)", structure)
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
