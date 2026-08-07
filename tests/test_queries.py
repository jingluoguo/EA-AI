from __future__ import annotations

import unittest

from tests.support import *


class QueryTest(unittest.TestCase):
    def test_polar_queries_cover_positive_negative_unknown_and_surface_markers(self) -> None:
        cases = (
            # positive
            ("小郭把芯片放进托盘。芯片存在吗？", "QUERY polar_existence(芯片)", "是，芯片存在。"),
            ("小郭把芯片放进托盘。芯片在托盘里吗？", "QUERY polar_location(芯片,expected=托盘,kind=in)", "是，芯片在托盘里。"),
            ("小郭把芯片放进托盘。托盘被带到实验室。芯片在实验室吗？", "QUERY polar_location(芯片,expected=实验室,kind=at)", "是，芯片在实验室。"),
            ("小郭把芯片放进托盘。托盘被带到实验室。实验室里有芯片吗？", "QUERY polar_contents(实验室,item=芯片)", "是，实验室里有芯片。"),
            # surface markers normalize to existing query types
            ("小郭把芯片放进托盘。芯片是不是在托盘里面？", "QUERY polar_location(芯片,expected=托盘,kind=in)", "是，芯片在托盘里。"),
            ("小郭把芯片放进托盘。托盘被带到实验室。实验室里面有没有芯片？", "QUERY polar_contents(实验室,item=芯片)", "是，实验室里有芯片。"),
            ("工程师制造芯片。芯片是不是存在？", "QUERY polar_existence(芯片)", "是，芯片存在。"),
            ("小郭把芯片放进托盘。小王把药瓶放进托盘。芯片和药瓶是不是在同一个位置？", "QUERY same_location(芯片和药瓶,left=芯片,right=药瓶)", "是，芯片和药瓶在同一个地方。"),
            # negative and unknown
            ("工程师销毁芯片。芯片存在吗？", "QUERY polar_existence(芯片)", "不是，芯片不存在。"),
            ("小郭把芯片放进托盘。芯片在盒子里吗？", "QUERY polar_location(芯片,expected=盒子,kind=in)", "不是，芯片在托盘里。"),
            ("小王打开盒子。盒子里有芯片吗？", "QUERY polar_contents(盒子,item=芯片)", "不知道盒子里有没有芯片。"),
        )

        for text, query_line, answer in cases:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn(query_line, structure)
                self.assertEqual(prediction.answer, answer)

    def test_same_location_query_covers_shared_key_distinct_places_and_unknown_side(self) -> None:
        cases = (
            # both items share the same container
            ("小郭把芯片放进托盘。小王把药瓶放进托盘。芯片和药瓶在同一个地方吗？",
             "QUERY same_location(芯片和药瓶,left=芯片,right=药瓶)",
             "是，芯片和药瓶在同一个地方。"),
            # both items share the same place (via different containers)
            ("小郭把芯片放进托盘。托盘被带到实验室。小王把药瓶放进盒子。盒子被带到实验室。芯片和药瓶在同一个地方吗？",
             "QUERY same_location(芯片和药瓶,left=芯片,right=药瓶)",
             "是，芯片和药瓶在同一个地方。"),
            # items are at distinct places
            ("小郭把芯片放进托盘。托盘被带到实验室。小王把药瓶放进盒子。盒子被带到办公室。芯片和药瓶在同一个地方吗？",
             "QUERY same_location(芯片和药瓶,left=芯片,right=药瓶)",
             "不是，芯片在实验室的托盘里，药瓶在办公室的盒子里。"),
        )
        for text, query_line, answer in cases:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn(query_line, structure)
                self.assertEqual(prediction.answer, answer)

        # unknown side returns unknown
        prediction = predict("小王打开盒子。芯片和药瓶在同一个地方吗？")
        structure = prediction.structure.linearize()
        self.assertIn("QUERY same_location(芯片和药瓶,left=芯片,right=药瓶)", structure)
        self.assertEqual(prediction.answer, "不知道芯片和药瓶是不是在同一个地方。")

    def test_handler_question_reuses_structure_and_picks_latest_handler(self) -> None:
        # surface variants reuse extracted handle event and rule
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

        # latest handler overrides earlier one when item has multiple handlers
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

    def test_earliest_latest_and_historical_event_actor_queries(self) -> None:
        cases = (
            # earliest event actor (put_in + take_out)
            ("小郭把芯片放进托盘。小王把芯片放进托盘。小李把芯片从托盘里取出。最先谁把芯片放进托盘？",
             "QUERY earliest_actor_for_event(put_in,item=芯片,holder=托盘)",
             "最先是小郭把芯片放进托盘。"),
            ("小郭把芯片放进托盘。小王把芯片放进托盘。小李把芯片从托盘里取出。芯片最先是谁放进托盘的？",
             "QUERY earliest_actor_for_event(put_in,item=芯片,holder=托盘)",
             "最先是小郭把芯片放进托盘。"),
            ("小郭把芯片放进托盘。小王把芯片放进托盘。小李把芯片从托盘里取出。最先谁把芯片从托盘里取出？",
             "QUERY earliest_actor_for_event(take_out,item=芯片,source=托盘)",
             "最先是小李把芯片从托盘取出。"),
            # latest event actor (put_in + take_out)
            ("小郭把芯片放进托盘。小王把芯片放进托盘。小李把芯片从托盘里取出。最后谁把芯片放进托盘？",
             "QUERY latest_actor_for_event(put_in,item=芯片,holder=托盘)",
             "最后是小王把芯片放进托盘。"),
            ("小郭把芯片放进托盘。小王把芯片放进托盘。小李把芯片从托盘里取出。芯片最后是谁放进托盘的？",
             "QUERY latest_actor_for_event(put_in,item=芯片,holder=托盘)",
             "最后是小王把芯片放进托盘。"),
            ("小郭把芯片放进托盘。小王把芯片放进托盘。小李把芯片从托盘里取出。最近谁把芯片从托盘里取出？",
             "QUERY latest_actor_for_event(take_out,item=芯片,source=托盘)",
             "最后是小李把芯片从托盘取出。"),
            # earliest + latest side by side in same context (differentiate order)
            ("小郭把芯片放进托盘。小王把芯片放进托盘。最先谁把芯片放进托盘？最后谁把芯片放进托盘？",
             "QUERY compound(multi)",
             "最先是小郭把芯片放进托盘；最后是小王把芯片放进托盘。"),
        )
        for text, query_line, answer in cases:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn(query_line, structure)
                self.assertEqual(prediction.answer, answer)

    def test_historical_actor_query_and_put_in_surface_forms_normalize_to_same_structure(self) -> None:
        # historical put_in actor is queryable after later state changes overwrite current state
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

        # surface container forms (放到/里面/放入/里) normalize to same put_in structure
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

    def test_take_out_event_updates_state_and_supports_surface_variants_and_actor_query(self) -> None:
        # core state transition: take_out removes current container state
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

        # statement surface variants all normalize to the same event
        for statement in (
            "小王把芯片从托盘里取出",
            "小王把芯片从托盘里面拿出来",
            "小王从托盘里取出芯片",
            "芯片被小王从托盘里拿出",
            "芯片从托盘里被取出",
        ):
            with self.subTest(statement=statement):
                prediction = predict(f"小郭把芯片放进托盘。{statement}。托盘里有什么？")
                structure = prediction.structure.linearize()
                self.assertNotIn("REL in(芯片,托盘)", structure)
                self.assertIn("QUERY contents(托盘)", structure)
                self.assertEqual(prediction.answer, "不知道托盘里有什么。")

        # actor query reuses the event structure
        for question in (
            "谁把芯片从托盘里取出来的？",
            "芯片是谁从托盘里面拿出来的？",
            "芯片被谁从托盘里取出的？",
        ):
            with self.subTest(question=question):
                prediction = predict(f"小郭把芯片放进托盘。小王把芯片从托盘里取出。{question}")
                structure = prediction.structure.linearize()
                self.assertIn("QUERY actor_for_event(take_out,item=芯片,source=托盘)", structure)
                self.assertIn("RULE event_actor_matches", structure)
                self.assertEqual(prediction.answer, "小王把芯片从托盘取出。")

    def test_historical_replay_queries_cover_initial_location_latest_actor_and_before_action(self) -> None:
        # initial_location replays first put_in regardless of later state changes
        cases = (
            ("小郭把芯片放进托盘。小王把芯片放进盒子。芯片最开始在哪里？",
             "芯片最开始在托盘里。"),
            ("托盘被带到实验室。小郭把芯片放进托盘。小王把芯片放进盒子。芯片一开始在哪里？",
             "芯片最开始在实验室的托盘里。"),
            ("小郭把芯片放进托盘。小王把芯片从托盘里取出。芯片最开始在哪里？",
             "芯片最开始在托盘里。"),
        )
        for text, answer in cases:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn("QUERY initial_location(芯片)", structure)
                self.assertIn("RULE initial_location_found", structure)
                self.assertEqual(prediction.answer, answer)

        # latest_actor_for_item picks most recent handler from history
        prediction = predict("小郭把芯片放进托盘。小王把芯片放进盒子。最后谁处理过芯片？")
        structure = prediction.structure.linearize()
        self.assertIn("QUERY latest_actor_for_item(芯片)", structure)
        self.assertIn("RULE latest_actor_handles_item", structure)
        self.assertEqual(prediction.answer, "最后是小王处理过芯片。")

        # location_before_actor_action replays state just before an actor's latest frame
        prediction = predict("小郭把芯片放进托盘。小王把芯片放进盒子。小王操作之前，芯片在哪里？")
        structure = prediction.structure.linearize()
        self.assertIn("QUERY location_before_actor_action(芯片,actor=小王)", structure)
        self.assertIn("RULE location_before_actor_action_found", structure)
        self.assertEqual(prediction.answer, "小王操作之前，芯片在托盘里。")

    def test_temporal_before_after_queries_replay_state_around_event(self) -> None:
        cases = (
            # location before/after move
            ("小郭把芯片放进托盘。小王把托盘带到实验室。小王把托盘带到实验室之前，芯片在哪里？",
             "QUERY location_before_event(芯片,anchor=小王把托盘带到实验室,event=move,actor=小王,theme=托盘,goal=实验室)",
             "RULE location_before_event_found",
             "在小王把托盘带到实验室之前，芯片在托盘里。"),
            ("小郭把芯片放进托盘。小王把托盘带到实验室。在小王把托盘带到实验室之前，芯片在哪里？",
             "QUERY location_before_event(芯片,anchor=小王把托盘带到实验室,event=move,actor=小王,theme=托盘,goal=实验室)",
             "RULE location_before_event_found",
             "在小王把托盘带到实验室之前，芯片在托盘里。"),
            ("小郭把芯片放进托盘。小王把托盘带到实验室。小王把托盘带到实验室之后，芯片在哪里？",
             "QUERY location_after_event(芯片,anchor=小王把托盘带到实验室,event=move,actor=小王,theme=托盘,goal=实验室)",
             "RULE location_after_event_found",
             "在小王把托盘带到实验室之后，芯片在实验室的托盘里。"),
            ("小郭把芯片放进托盘。小王把托盘带到实验室。小王把托盘带到实验室之后，托盘在哪里？",
             "QUERY location_after_event(托盘,anchor=小王把托盘带到实验室,event=move,actor=小王,theme=托盘,goal=实验室)",
             "RULE location_after_event_found",
             "在小王把托盘带到实验室之后，托盘在实验室。"),
            # location after take_out (unknown)
            ("小郭把芯片放进托盘。小王把芯片从托盘里取出之后，芯片在哪里？",
             "QUERY location_after_event(芯片,anchor=小王把芯片从托盘里取出,event=take_out,actor=小王,theme=芯片,source=托盘)",
             "RULE location_after_event_unknown",
             "不知道芯片在小王把芯片从托盘里取出之后在哪里。"),
            # contents before/after move
            ("小郭把芯片放进托盘。小王把托盘带到实验室。小王把托盘带到实验室之前，托盘里有什么？",
             "QUERY contents_before_event(托盘,anchor=小王把托盘带到实验室,event=move,actor=小王,theme=托盘,goal=实验室)",
             "RULE contents_before_event_found",
             "在小王把托盘带到实验室之前，托盘里至少有芯片。"),
            ("小郭把芯片放进托盘。小王把托盘带到实验室。小王把托盘带到实验室之后，托盘里有什么？",
             "QUERY contents_after_event(托盘,anchor=小王把托盘带到实验室,event=move,actor=小王,theme=托盘,goal=实验室)",
             "RULE contents_after_event_found",
             "在小王把托盘带到实验室之后，托盘里至少有芯片。"),
            # contents after take_out (unknown)
            ("小郭把芯片放进托盘。小王把芯片从托盘里取出。小王把芯片从托盘里取出之后，托盘里有什么？",
             "QUERY contents_after_event(托盘,anchor=小王把芯片从托盘里取出,event=take_out,actor=小王,theme=芯片,source=托盘)",
             "RULE contents_after_event_unknown",
             "不知道托盘在小王把芯片从托盘里取出之后有什么。"),
        )
        for text, query_line, rule, answer in cases:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn(query_line, structure)
                self.assertIn(rule, structure)
                self.assertEqual(prediction.answer, answer)

    def test_events_after_event_query_uses_event_anchor(self) -> None:
        cases = (
            ("小郭把芯片放进盒子。盒子被带到仓库。小王把芯片从盒子里取出。芯片被放进盒子之后发生了什么？",
             "QUERY events_after_event(put_in,item=芯片,holder=盒子)",
             "之后发生了：盒子被带到仓库；小王把芯片从盒子取出。"),
            ("小郭把芯片放进盒子。小王把芯片从盒子里取出。小王把芯片放进托盘。芯片从盒子里取出之后发生了什么？",
             "QUERY events_after_event(take_out,item=芯片,source=盒子)",
             "之后发生了：小王把芯片放进托盘。"),
        )
        for text, query_line, answer in cases:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn(query_line, structure)
                self.assertIn("RULE events_after_event", structure)
                self.assertEqual(prediction.answer, answer)

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

    def test_contents_queries_use_current_state_closure(self) -> None:
        cases = (
            ("小郭把芯片放进托盘。托盘被带到实验室。托盘里有什么？", "QUERY contents(托盘)", "托盘里至少有芯片。"),
            ("小郭把芯片放进托盘。托盘被带到实验室。小王把芯片放进盒子。盒子被带到办公室。办公室里有什么？", "QUERY contents(办公室)", "办公室里至少有盒子和芯片。"),
            ("小郭把芯片放进托盘。小王把芯片放进盒子。盒子被带到仓库。你可以告诉我仓库里有什么吗？", "QUERY contents(仓库)", "仓库里至少有盒子和芯片。"),
        )
        for text, query_line, answer in cases:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn(query_line, structure)
                self.assertIn("RULE holder_contains_things", structure)
                self.assertEqual(prediction.answer, answer)

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

    def test_nested_containers_use_recursive_closure_for_location_and_contents(self) -> None:
        # location query resolves through nested containers
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

        # contents query collects every nested item recursively
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

    def test_count_query_uses_contents_closure_with_filters(self) -> None:
        cases = (
            ("小郭把芯片放进托盘。托盘被带到实验室。实验室里有几个东西？", "QUERY count(实验室)", "实验室里至少有2个已知物品。"),
            ("小郭把芯片放进托盘。托盘里有几个物品？", "QUERY count(托盘)", "托盘里至少有1个已知物品。"),
            ("小郭把芯片放进托盘。托盘被带到实验室。实验室里数量是多少？", "QUERY count(实验室)", "实验室里至少有2个已知物品。"),
            ("小郭把芯片放进小盒子。小王把小盒子放进大盒子。大盒子被带到实验室。实验室里有几个物品？", "QUERY count(实验室)", "实验室里至少有3个已知物品。"),
            ("小王打开盒子。盒子里有几个东西？", "QUERY count(盒子)", "盒子里没有已知物品。"),
        )
        for text, query_line, answer in cases:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn(query_line, structure)
                self.assertEqual(prediction.answer, answer)

    def test_count_query_filters_destroyed_objects(self) -> None:
        prediction = predict("小郭把芯片放进托盘。托盘被带到实验室。工程师销毁芯片。实验室里有几个东西？")

        structure = prediction.structure.linearize()
        self.assertIn("REL exists(芯片,不存在)", structure)
        self.assertIn("QUERY count(实验室)", structure)
        self.assertEqual(prediction.answer, "实验室里至少有1个已知物品。")

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
        cases = (
            ("小郭把芯片放进托盘。小王把托盘带到了实验室。小王把芯片从托盘里取出。小郭和小王分别做了什么？",
             "QUERY actions_by_actors(小郭和小王,actors=小郭|小王)",
             "小郭把芯片放进托盘；小王把托盘带到实验室，把芯片从托盘取出。"),
            ("小郭把芯片放进托盘。小王把芯片放进盒子。小郭做了什么？",
             "QUERY actions_by_actors(小郭,actors=小郭)",
             "小郭把芯片放进托盘。"),
        )
        for text, query_line, answer in cases:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn(query_line, structure)
                self.assertIn("RULE actor_actions", structure)
                self.assertEqual(prediction.answer, answer)

    def test_inventory_query_lists_current_owned_items(self) -> None:
        prediction = predict("小红把药瓶交给医生。小郭把芯片交给医生。现在每个人手里有什么？")

        structure = prediction.structure.linearize()
        self.assertIn("QUERY inventories(person)", structure)
        self.assertIn("RULE owner_inventories", structure)
        self.assertEqual(prediction.answer, "医生手里有药瓶和芯片。")

    def test_demonstrative_and_pronoun_resolution_covers_entities_places_and_pairs(self) -> None:
        cases = (
            # typed demonstrative resolves to known entity
            ("小郭把芯片放进托盘。托盘被带到实验室。这个芯片在哪里？",
             "QUERY location(芯片)", "芯片在实验室的托盘里。"),
            # pronoun resolves to latest non-place entity
            ("小郭把芯片放进托盘。小王把芯片放进盒子。它在哪里？",
             "QUERY location(盒子)", "不知道盒子在哪里。"),
            # typed demonstrative in statement updates state
            ("小郭把芯片放进托盘。小王把这个芯片从托盘里取出来。这个芯片在哪里？",
             "EVENT take_out(小王,芯片) WITH source=托盘", "不知道芯片在哪里。"),
            # place pronoun resolves for contents query
            ("小郭把芯片放进托盘。托盘被带到实验室。这里有什么？",
             "QUERY contents(实验室)", "实验室里至少有托盘和芯片。"),
        )
        for text, expected, answer in cases:
            with self.subTest(text=text):
                prediction = predict(text)
                structure = prediction.structure.linearize()
                self.assertIn(expected, structure)
                self.assertEqual(prediction.answer, answer)

        # relative pronouns resolve to previous two salient entities
        prediction = predict(
            "小郭把芯片放进托盘。托盘被带到实验室。小王把药瓶放进盒子。盒子被带到办公室。前者在哪里，后者在哪里？"
        )
        structure = prediction.structure.linearize()
        self.assertIn("QUERY compound(multi)", structure)
        self.assertIn("SUBQUERY location(药瓶)", structure)
        self.assertIn("SUBQUERY location(盒子)", structure)
        self.assertEqual(prediction.answer, "药瓶在办公室的盒子里；盒子在办公室。")

        # relative pronouns fail closed when context is too short
        with self.assertRaises(ParseError):
            predict("前者在哪里？")


if __name__ == "__main__":
    unittest.main()
