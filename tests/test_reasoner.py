from __future__ import annotations

import unittest

from struct_llm.reasoner import predict


class ReasonerTest(unittest.TestCase):
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

    def test_event_actor_question_can_ask_historical_event_after_state_changes(self) -> None:
        prediction = predict("小郭把芯片放进托盘。小王把芯片放进盒子。谁把芯片放进托盘？")

        structure = prediction.structure.linearize()
        self.assertIn("REL in(芯片,盒子)", structure)
        self.assertIn("EVENT put_in(小郭,芯片) WITH holder=托盘", structure)
        self.assertIn("EVENT put_in(小王,芯片) WITH holder=盒子", structure)
        self.assertEqual(prediction.answer, "小郭把芯片放进托盘。")

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


if __name__ == "__main__":
    unittest.main()
