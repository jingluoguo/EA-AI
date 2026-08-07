from __future__ import annotations

import unittest

from tests.support import *


class StateTest(unittest.TestCase):
    def test_containment_move(self) -> None:
        prediction = predict("研究员把芯片放进托盘。托盘被带到实验室。芯片在哪里？")

        self.assertIn("RULE container_moves_contents", prediction.structure.linearize())
        self.assertEqual(prediction.answer, "芯片在实验室的托盘里。")

    def test_later_event_overwrites_previous_state(self) -> None:
        cases = (
            {
                "name": "put_in_overwrite",
                "text": (
                    "小郭把芯片放进托盘。托盘被带到实验室。"
                    "小王把芯片放进盒子。盒子被带到办公室。芯片在哪里？"
                ),
                "present": "REL in(芯片,盒子)",
                "absent": "REL in(芯片,托盘)",
                "answer": "芯片在办公室的盒子里。",
            },
            {
                "name": "move_overwrite",
                "text": "小郭把芯片放进托盘。托盘被带到实验室。托盘被带到办公室。芯片在哪里？",
                "present": "REL at(托盘,办公室)",
                "absent": "REL at(托盘,实验室)",
                "answer": "芯片在办公室的托盘里。",
            },
            {
                "name": "transfer_overwrite",
                "text": "小红把药瓶交给医生。医生把药瓶交给老师。现在谁拥有药瓶？",
                "present": "REL owner(药瓶,老师)",
                "absent": "REL owner(药瓶,医生)",
                "answer": "老师拥有药瓶。",
            },
            {
                "name": "paint_overwrite",
                "text": "工程师把笔记本涂成绿色。研究员把笔记本涂成黄色。现在笔记本是什么颜色？",
                "present": "REL color(笔记本,黄色)",
                "absent": "REL color(笔记本,绿色)",
                "answer": "笔记本是黄色。",
            },
            {
                "name": "open_close_overwrite",
                "text": "小王打开盒子。小郭把盒子关上。盒子现在是什么状态？",
                "present": "REL access(盒子,关闭)",
                "absent": "REL access(盒子,打开)",
                "answer": "盒子是关闭状态。",
            },
        )
        for case in cases:
            with self.subTest(name=case["name"]):
                prediction = predict(case["text"])
                structure = prediction.structure.linearize()
                self.assertIn(case["present"], structure)
                self.assertNotIn(case["absent"], structure)
                self.assertEqual(prediction.answer, case["answer"])

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

    def test_basic_event_rules_and_word_order_variants(self) -> None:
        cases = (
            ("小红把药瓶交给医生。现在谁拥有药瓶？", "RULE transfer_changes_owner", "医生拥有药瓶。"),
            ("小红把药瓶交给医生。药瓶是谁拥有的？", "QUERY owner(药瓶)", "医生拥有药瓶。"),
            ("工程师把笔记本涂成绿色。现在笔记本是什么颜色？", "RULE paint_changes_color", "笔记本是绿色。"),
            ("工程师把笔记本涂成绿色。笔记本颜色是什么？", "QUERY color(笔记本)", "笔记本是绿色。"),
        )
        for text, expected, answer in cases:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn(expected, structure)
                self.assertEqual(prediction.answer, answer)

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

    def test_destroy_clears_state_and_can_be_restored_by_later_events(self) -> None:
        cases = (
            # destroy clears location and attributes
            ("小郭把芯片放进托盘。工程师把芯片涂成绿色。工程师销毁芯片。芯片在哪里？",
             "REL exists(芯片,不存在)", "REL in(芯片,托盘)", "REL color(芯片,绿色)",
             "RULE object_not_exists", "芯片不存在。"),
            # destroyed item is removed from contents closure
            ("小郭把芯片放进托盘。托盘被带到实验室。工程师销毁芯片。实验室里有什么？",
             "REL exists(芯片,不存在)", "REL in(芯片,托盘)", None,
             None, "实验室里至少有托盘。"),
            # later state can restore destroyed object for ordered correction
            ("工程师销毁芯片。小郭把芯片放进托盘。芯片是否存在？",
             "REL in(芯片,托盘)", "REL exists(芯片,不存在)", None,
             None, "芯片存在。"),
        )
        for text, present, absent1, absent2, rule, answer in cases:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn(present, structure)
                self.assertNotIn(absent1, structure)
                if absent2:
                    self.assertNotIn(absent2, structure)
                if rule:
                    self.assertIn(rule, structure)
                self.assertEqual(prediction.answer, answer)

    def test_existence_claims_can_conflict_with_fact(self) -> None:
        destroyed = predict("工程师销毁芯片。小王说芯片存在。有没有矛盾？")
        existing = predict("工程师制造芯片。小王说芯片不存在。有没有矛盾？")

        self.assertEqual(destroyed.answer, "存在矛盾：小王说芯片存在，但事实是芯片不存在。")
        self.assertEqual(existing.answer, "存在矛盾：小王说芯片不存在，但事实是芯片存在。")

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


if __name__ == "__main__":
    unittest.main()
