from __future__ import annotations

import unittest

from struct_llm.errors import ParseError
from struct_llm.event_schema import EVENT_SCHEMAS, frame_matches_qualifiers, states_for_frame_schema
from struct_llm.cognitive import CognitiveCapabilities
from struct_llm.cognitive.frame_parser import frame_from_roles, with_time
from struct_llm.cognitive.normalization import normalize_entity_slot
from struct_llm.capabilities import StructuralCapabilities
from struct_llm.modules import ModuleContext, default_module_registry
from struct_llm.modules.cognitive import CognitiveKernelModule
from struct_llm.reasoner import default_capabilities, parse_text, predict as _predict
from struct_llm.structure import Entity, Query


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
        self.assertIs(StructuralCapabilities, CognitiveCapabilities)

    def test_default_module_registry_exposes_outer_system_slots(self) -> None:
        registry = default_module_registry(default_capabilities())

        self.assertEqual(
            registry.module_names(),
            (
                "alignment",
                "memory",
                "knowledge",
                "cognitive_kernel",
                "generation",
                "planning",
                "embodiment",
                "emotion",
                "self_model",
                "learning",
            ),
        )

    def test_noop_outer_modules_preserve_cognitive_kernel_result(self) -> None:
        text = "研究员把芯片放进托盘。托盘被带到实验室。芯片在哪里？"
        capabilities = default_capabilities()

        parsed = parse_text(text, capabilities)
        direct = CognitiveKernelModule(capabilities).run(ModuleContext(text=text))
        modular = default_module_registry(capabilities).run(ModuleContext(text=text))

        self.assertEqual(modular.notes, ())
        direct_structure = direct.context.structure
        modular_structure = modular.context.structure
        self.assertIsNotNone(direct_structure)
        self.assertIsNotNone(modular_structure)
        assert direct_structure is not None
        assert modular_structure is not None
        self.assertEqual(parsed.linearize(), direct_structure.linearize())
        self.assertEqual(direct_structure.linearize(), modular_structure.linearize())
        self.assertEqual(direct.context.answer, modular.context.answer)

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

    def test_query_capability_can_be_injected_without_changing_pipeline(self) -> None:
        def parse_keeper_query(sentence: str, entities: tuple[Entity, ...]) -> Query | None:
            if "保管者" not in sentence or "谁" not in sentence:
                return None
            target = sentence.replace("保管者", "").replace("谁", "").replace("是", "")
            return Query("owner", normalize_entity_slot(target, entities))

        capabilities = default_capabilities().with_query_parsers(parse_keeper_query)
        prediction = predict("小红把药瓶交给医生。药瓶保管者是谁？", capabilities)

        structure = prediction.structure.linearize()
        self.assertIn("QUERY owner(药瓶)", structure)
        self.assertIn("RULE transfer_changes_owner", structure)
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
