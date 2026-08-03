from __future__ import annotations

import unittest

from struct_llm.reasoner import predict


class ReasonerTest(unittest.TestCase):
    def test_containment_move(self) -> None:
        prediction = predict("研究员把芯片放进托盘。托盘被带到实验室。芯片在哪里？")

        self.assertIn("RULE container_moves_contents", prediction.structure.linearize())
        self.assertEqual(prediction.answer, "芯片在实验室的托盘里。")

    def test_ownership_transfer(self) -> None:
        prediction = predict("小红把药瓶交给医生。现在谁拥有药瓶？")

        self.assertIn("RULE transfer_changes_owner", prediction.structure.linearize())
        self.assertEqual(prediction.answer, "医生拥有药瓶。")

    def test_color_change(self) -> None:
        prediction = predict("工程师把笔记本涂成绿色。现在笔记本是什么颜色？")

        self.assertIn("RULE paint_changes_color", prediction.structure.linearize())
        self.assertEqual(prediction.answer, "笔记本是绿色。")


if __name__ == "__main__":
    unittest.main()
