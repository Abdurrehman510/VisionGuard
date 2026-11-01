"""Tests for the selectable counting rule ("feet", "centre", "overlap") and the switch that counts people
first seen already inside the zone (main.py). Run: python -m unittest tests.test_counting_rules

The defaults must stay exactly what the project did before these options existed: the bottom-centre of the
box decides, and a person first seen inside is not counted.
"""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main

SQUARE = np.array([(100, 100), (200, 100), (200, 200), (100, 200)], dtype=np.float32)
# An "L": the square 0-200 with its bottom-right quarter (x 100-200, y 100-200) cut away.
L_SHAPE = np.array([(0, 0), (200, 0), (200, 100), (100, 100), (100, 200), (0, 200)], dtype=np.float32)

# Three boxes that the three rules judge differently against SQUARE:
FEET_OUTSIDE_CENTRE_INSIDE = (140, 120, 160, 230)   # feet at y=230 (below the zone), middle at y=175 (inside)
ONLY_OVERLAPS = (150, 0, 260, 105)                  # feet and middle both outside, but the box clips the zone
FULLY_INSIDE = (130, 120, 170, 180)


class DefaultsTests(unittest.TestCase):
    def test_the_shipped_defaults_keep_the_original_behaviour(self):
        self.assertEqual(main.COUNT_RULE, "feet")
        self.assertIs(main.COUNT_FIRST_SEEN_INSIDE, False)

    def test_the_three_rules_are_listed(self):
        self.assertEqual(main.COUNT_RULES, ("feet", "centre", "overlap"))


class CentrePointTests(unittest.TestCase):
    def test_centre_is_the_middle_of_the_box(self):
        self.assertEqual(main.centre_point((10, 20, 30, 60)), (20.0, 40.0))


class BoxOverlapTests(unittest.TestCase):
    def test_box_fully_inside_overlaps(self):
        self.assertTrue(main.box_overlaps_zone(FULLY_INSIDE, SQUARE))

    def test_box_fully_outside_does_not_overlap(self):
        self.assertFalse(main.box_overlaps_zone((300, 300, 400, 400), SQUARE))

    def test_box_hanging_over_an_edge_overlaps(self):
        self.assertTrue(main.box_overlaps_zone((150, 150, 250, 250), SQUARE))

    def test_box_around_the_whole_zone_overlaps(self):
        self.assertTrue(main.box_overlaps_zone((50, 50, 250, 250), SQUARE))

    def test_thin_box_cutting_straight_through_the_zone_overlaps(self):
        # No box corner is inside the zone and no zone corner is inside the box: only the edges cross.
        self.assertTrue(main.box_overlaps_zone((140, 0, 160, 300), SQUARE))

    def test_box_touching_the_boundary_counts_like_a_point_on_the_boundary(self):
        self.assertTrue(main.box_overlaps_zone((200, 100, 300, 200), SQUARE))

    def test_box_one_pixel_away_does_not_overlap(self):
        self.assertFalse(main.box_overlaps_zone((201, 100, 300, 200), SQUARE))

    def test_box_in_the_notch_of_a_concave_zone_does_not_overlap(self):
        self.assertFalse(main.box_overlaps_zone((120, 120, 180, 180), L_SHAPE))

    def test_box_reaching_into_the_arm_of_a_concave_zone_overlaps(self):
        self.assertTrue(main.box_overlaps_zone((120, 40, 180, 180), L_SHAPE))


class PersonInZoneTests(unittest.TestCase):
    def test_feet_rule_uses_the_bottom_centre(self):
        self.assertFalse(main.person_in_zone(FEET_OUTSIDE_CENTRE_INSIDE, SQUARE, "feet"))

    def test_centre_rule_uses_the_middle_of_the_box(self):
        self.assertTrue(main.person_in_zone(FEET_OUTSIDE_CENTRE_INSIDE, SQUARE, "centre"))

    def test_overlap_rule_counts_any_touch(self):
        self.assertFalse(main.person_in_zone(ONLY_OVERLAPS, SQUARE, "feet"))
        self.assertFalse(main.person_in_zone(ONLY_OVERLAPS, SQUARE, "centre"))
        self.assertTrue(main.person_in_zone(ONLY_OVERLAPS, SQUARE, "overlap"))

    def test_a_person_fully_inside_is_inside_under_every_rule(self):
        for rule in main.COUNT_RULES:
            self.assertTrue(main.person_in_zone(FULLY_INSIDE, SQUARE, rule), rule)

    def test_unknown_rule_is_refused(self):
        with self.assertRaises(ValueError):
            main.person_in_zone(FULLY_INSIDE, SQUARE, "head")

    def test_check_zone_defaults_to_the_feet_rule(self):
        people = [(1, *FEET_OUTSIDE_CENTRE_INSIDE, 0.9), (2, *FULLY_INSIDE, 0.9)]
        self.assertEqual(main.check_zone(people, SQUARE), {1: False, 2: True})

    def test_check_zone_applies_the_chosen_rule(self):
        people = [(1, *FEET_OUTSIDE_CENTRE_INSIDE, 0.9), (2, *ONLY_OVERLAPS, 0.9)]
        self.assertEqual(main.check_zone(people, SQUARE, "centre"), {1: True, 2: False})
        self.assertEqual(main.check_zone(people, SQUARE, "overlap"), {1: True, 2: True})


class FirstSeenInsideTests(unittest.TestCase):
    @staticmethod
    def entries(frames, confirm_frames=1, count_first_seen_inside=False):
        zone_states, entered = {}, []
        for inside_by_id in frames:
            entered.extend(main.find_entries(zone_states, inside_by_id, confirm_frames,
                                             count_first_seen_inside=count_first_seen_inside))
        return entered

    def test_by_default_a_person_first_seen_inside_is_not_counted(self):
        self.assertEqual(self.entries([{1: True}, {1: True}, {1: True}]), [])

    def test_with_the_switch_a_person_first_seen_inside_is_counted_once(self):
        self.assertEqual(self.entries([{1: True}, {1: True}, {1: True}], count_first_seen_inside=True), [1])

    def test_with_the_switch_a_person_first_seen_outside_is_still_counted_only_on_entering(self):
        frames = [{1: False}, {1: False}, {1: True}, {1: True}]
        self.assertEqual(self.entries(frames, count_first_seen_inside=True), [1])

    def test_the_confirmation_window_also_applies_to_a_person_first_seen_inside(self):
        frames = [{1: True}] * 3
        zone_states, per_frame = {}, []
        for inside_by_id in frames:
            per_frame.append(main.find_entries(zone_states, inside_by_id, 3, count_first_seen_inside=True))
        self.assertEqual(per_frame, [[], [], [1]])

    def test_a_flicker_at_first_sight_is_not_counted(self):
        frames = [{1: True}, {1: False}, {1: True}]
        self.assertEqual(self.entries(frames, confirm_frames=3, count_first_seen_inside=True), [])

    def test_leaving_and_coming_back_is_a_second_entry(self):
        frames = [{1: True}, {1: False}, {1: True}]
        self.assertEqual(self.entries(frames, count_first_seen_inside=True), [1, 1])

    def test_each_new_person_first_seen_inside_is_counted_separately(self):
        frames = [{1: True, 2: True}, {1: True, 2: True, 3: False}]
        self.assertEqual(self.entries(frames, count_first_seen_inside=True), [1, 2])


class UpdateMonitoringRuleTests(unittest.TestCase):
    def test_an_entry_under_the_overlap_rule_is_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "events.csv"
            zone_states = {}
            away = [(7, 300, 300, 340, 380, 0.9)]
            touching = [(7, *ONLY_OVERLAPS, 0.9)]
            main.update_monitoring(away, SQUARE, zone_states, csv_path, rule="overlap")
            _, entered = main.update_monitoring(touching, SQUARE, zone_states, csv_path, rule="overlap")
            self.assertEqual(entered, [7])
            with open(csv_path, newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([(row["tracking_id"], row["event_type"]) for row in rows], [("7", "ENTRY")])

    def test_the_same_movement_is_not_an_entry_under_the_feet_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            zone_states = {}
            main.update_monitoring([(7, 300, 300, 340, 380, 0.9)], SQUARE, zone_states, Path(tmp) / "e.csv")
            _, entered = main.update_monitoring([(7, *ONLY_OVERLAPS, 0.9)], SQUARE, zone_states, Path(tmp) / "e.csv")
            self.assertEqual(entered, [])

    def test_the_first_seen_inside_switch_is_passed_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, entered = main.update_monitoring([(4, *FULLY_INSIDE, 0.9)], SQUARE, {}, Path(tmp) / "e.csv",
                                                count_first_seen_inside=True)
            self.assertEqual(entered, [4])


class ValidateRuleTests(unittest.TestCase):
    def test_every_listed_rule_is_accepted(self):
        for rule in main.COUNT_RULES:
            main.validate_rule(rule)

    def test_an_unknown_rule_is_rejected_with_the_choices(self):
        with self.assertRaises(RuntimeError) as context:
            main.validate_rule("head")
        for rule in main.COUNT_RULES:
            self.assertIn(rule, str(context.exception))

    def test_a_rule_that_is_not_text_is_rejected(self):
        for bad in (None, 3, ["feet"]):
            with self.assertRaises(RuntimeError):
                main.validate_rule(bad)

    def test_the_first_seen_switch_must_be_true_or_false(self):
        main.validate_count_first_seen(True)
        main.validate_count_first_seen(False)
        for bad in ("yes", 1, 0, None):
            with self.assertRaises(RuntimeError) as context:
                main.validate_count_first_seen(bad)
            self.assertIn("COUNT_FIRST_SEEN_INSIDE", str(context.exception))

    def test_validate_config_checks_both_new_settings(self):
        zone = main.ZONE_POLYGON
        main.validate_config(zone, 0.4, 3, "centre", True)
        with self.assertRaises(RuntimeError):
            main.validate_config(zone, 0.4, 3, "head", False)
        with self.assertRaises(RuntimeError):
            main.validate_config(zone, 0.4, 3, "feet", "yes")

    def test_validate_config_still_works_with_the_old_three_arguments(self):
        main.validate_config(main.ZONE_POLYGON, 0.4, 3)


class DrawPeopleReferencePointTests(unittest.TestCase):
    """The dot on screen shows the point that decides inside/outside, so it must follow the rule."""
    BOX = (50, 50, 150, 150)

    def dot_pixel(self, rule, at):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        main.draw_people(frame, [(1, *self.BOX, 0.9)], {1: False}, rule=rule)
        return tuple(int(v) for v in frame[at[1], at[0]])

    def test_feet_rule_marks_the_bottom_centre(self):
        self.assertNotEqual(self.dot_pixel("feet", (100, 148)), (0, 0, 0))
        self.assertEqual(self.dot_pixel("feet", (100, 100)), (0, 0, 0))

    def test_centre_rule_marks_the_middle_of_the_box(self):
        self.assertNotEqual(self.dot_pixel("centre", (100, 100)), (0, 0, 0))

    def test_overlap_rule_draws_no_dot(self):
        self.assertEqual(self.dot_pixel("overlap", (100, 100)), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
