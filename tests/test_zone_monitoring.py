"""Unit tests for VisionGuard Phase 2 zone monitoring. Run: python -m unittest discover -s tests"""
import csv
import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main

SQUARE = np.array([(100, 100), (200, 100), (200, 200), (100, 200)], dtype=np.float32)


class BottomCentreTests(unittest.TestCase):
    """bottom_centre(): the reference point of a person is the middle of the bottom edge of the box."""
    def test_bottom_centre_is_middle_of_bottom_edge(self):
        self.assertEqual(main.bottom_centre((10, 20, 30, 60)), (20.0, 60.0))


class ScaleZoneTests(unittest.TestCase):
    """scale_zone(): 0-1 fractions become pixel coordinates for the frame size."""
    def test_fractions_are_scaled_to_frame_pixels(self):
        zone = main.scale_zone([(0.5, 0.5), (1.0, 0.5), (1.0, 1.0)], width=200, height=100)
        np.testing.assert_array_equal(zone, [[100, 50], [200, 50], [200, 100]])


class PointInZoneTests(unittest.TestCase):
    """point_in_zone(): inside and on the boundary count as in; outside does not."""
    def test_point_inside_polygon(self):
        self.assertTrue(main.point_in_zone((150, 150), SQUARE))

    def test_point_outside_polygon(self):
        self.assertFalse(main.point_in_zone((50, 150), SQUARE))

    def test_point_on_boundary_counts_as_inside(self):
        self.assertTrue(main.point_in_zone((100, 150), SQUARE))


def entries_over_frames(frames):
    """Feed one {track_id: is_inside} dict per frame through find_entries; return IDs entered, in order."""
    zone_states, entered = {}, []
    for inside_by_id in frames:
        entered.extend(main.find_entries(zone_states, inside_by_id))
    return entered


class FindEntriesTests(unittest.TestCase):
    """find_entries() with no confirmation window: only an observed outside -> inside move is an entry (Technical Rule 5)."""
    def test_person_outside_zone_records_no_entry(self):
        self.assertEqual(entries_over_frames([{1: False}, {1: False}, {1: False}]), [])

    def test_person_entering_zone_records_one_entry(self):
        self.assertEqual(entries_over_frames([{1: False}, {1: True}]), [1])

    def test_person_remaining_inside_records_no_additional_entry(self):
        self.assertEqual(entries_over_frames([{1: False}, {1: True}, {1: True}, {1: True}]), [1])

    def test_person_exiting_zone_records_no_entry(self):
        self.assertEqual(entries_over_frames([{1: False}, {1: True}, {1: False}, {1: False}]), [1])

    def test_person_reentering_zone_records_a_second_entry(self):
        frames = [{1: False}, {1: True}, {1: False}, {1: True}]
        self.assertEqual(entries_over_frames(frames), [1, 1])

    def test_multiple_people_entering_are_each_counted(self):
        frames = [{1: False, 2: False, 3: False}, {1: True, 2: False, 3: True}, {1: True, 2: True, 3: True}]
        self.assertEqual(sorted(entries_over_frames(frames)), [1, 2, 3])

    def test_person_first_seen_inside_is_not_counted(self):
        self.assertEqual(entries_over_frames([{7: True}, {7: True}]), [])

    def test_track_missing_for_some_frames_keeps_its_state(self):
        frames = [{1: False}, {1: True}, {}, {}, {1: True}]
        self.assertEqual(entries_over_frames(frames), [1])

def entries_with_confirmation(frames, confirm_frames):
    """Like entries_over_frames, but a state change must last `confirm_frames` frames to count."""
    zone_states, entered = {}, []
    for inside_by_id in frames:
        entered.extend(main.find_entries(zone_states, inside_by_id, confirm_frames))
    return entered


class ConfirmFramesTests(unittest.TestCase):
    """Filtering of boundary flicker: a change of state only counts once it has lasted N frames."""

    def test_brief_blip_inside_is_ignored(self):
        frames = [{1: False}, {1: True}, {1: True}, {1: False}, {1: False}]
        self.assertEqual(entries_with_confirmation(frames, 3), [])

    def test_entry_is_counted_once_when_inside_lasts_long_enough(self):
        frames = [{1: False}] + [{1: True}] * 6
        self.assertEqual(entries_with_confirmation(frames, 3), [1])

    def test_entry_is_registered_on_the_confirming_frame(self):
        zone_states = {}
        results = [main.find_entries(zone_states, {1: inside}, 3) for inside in (False, True, True, True)]
        self.assertEqual(results, [[], [], [], [1]])

    def test_brief_blip_outside_while_inside_does_not_cause_a_second_entry(self):
        frames = [{1: False}] + [{1: True}] * 4 + [{1: False}] * 2 + [{1: True}] * 4
        self.assertEqual(entries_with_confirmation(frames, 3), [1])

    def test_real_exit_and_reentry_are_still_counted(self):
        frames = [{1: False}] + [{1: True}] * 4 + [{1: False}] * 4 + [{1: True}] * 4
        self.assertEqual(entries_with_confirmation(frames, 3), [1, 1])

    def test_repeated_flicker_that_never_lasts_long_enough_is_never_counted(self):
        frames = [{1: False}] + [{1: True}, {1: True}, {1: False}] * 5
        self.assertEqual(entries_with_confirmation(frames, 3), [])

    def test_confirm_frames_of_one_counts_immediately(self):
        self.assertEqual(entries_with_confirmation([{1: False}, {1: True}], 1), [1])

    def test_frames_where_the_track_is_missing_neither_reset_nor_advance_confirmation(self):
        frames = [{1: False}, {1: True}, {1: True}, {}, {}, {1: True}]
        self.assertEqual(entries_with_confirmation(frames, 3), [1])

    def test_each_person_is_confirmed_independently(self):
        frames = [{1: False, 2: False}, {1: True, 2: True}, {1: True, 2: False}, {1: True, 2: False}]
        self.assertEqual(entries_with_confirmation(frames, 3), [1])


class LogEntryTests(unittest.TestCase):
    """log_entry(): creates the folder, file and header; appends rows; reports a write failure without raising."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.csv_path = Path(self.tmp.name) / "outputs" / "events.csv"  # folder does not exist yet

    def read_rows(self):
        with open(self.csv_path, newline="") as file:
            return list(csv.DictReader(file))

    def test_creates_folder_and_file_with_header_and_one_entry_row(self):
        self.assertTrue(main.log_entry(self.csv_path, 5))
        with open(self.csv_path, newline="") as file:
            lines = file.read().splitlines()
        self.assertEqual(lines[0], "timestamp,tracking_id,event_type")
        self.assertEqual(len(lines), 2)
        rows = self.read_rows()
        self.assertEqual((rows[0]["tracking_id"], rows[0]["event_type"]), ("5", "ENTRY"))

    def test_timestamp_is_local_date_and_time_to_the_second(self):
        main.log_entry(self.csv_path, 1)
        timestamp = self.read_rows()[0]["timestamp"]
        parsed = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        self.assertLess(abs((datetime.now() - parsed).total_seconds()), 5)

    def test_second_event_is_appended_without_a_second_header(self):
        main.log_entry(self.csv_path, 1)
        main.log_entry(self.csv_path, 2)
        self.assertEqual([row["tracking_id"] for row in self.read_rows()], ["1", "2"])
        self.assertEqual(self.csv_path.read_text().count("timestamp"), 1)

    def test_existing_empty_file_gets_a_header(self):
        self.csv_path.parent.mkdir(parents=True)
        self.csv_path.touch()
        main.log_entry(self.csv_path, 3)
        self.assertEqual(self.read_rows()[0]["tracking_id"], "3")

    def test_write_failure_is_reported_and_does_not_raise(self):
        blocker = Path(self.tmp.name) / "blocker"
        blocker.write_text("a file, so it cannot be used as a folder")
        output = io.StringIO()
        with redirect_stdout(output):
            result = main.log_entry(blocker / "events.csv", 1)
        self.assertFalse(result)
        self.assertIn("Could not write", output.getvalue())


class PolygonSelfIntersectionTests(unittest.TestCase):
    """Any shape is allowed (including concave ones) as long as its edges do not cross or touch."""

    def test_square_does_not_cross_itself(self):
        self.assertFalse(main.polygon_self_intersects([(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]))

    def test_triangle_does_not_cross_itself(self):
        self.assertFalse(main.polygon_self_intersects([(0.5, 0.1), (0.9, 0.9), (0.1, 0.9)]))

    def test_concave_l_shape_is_allowed(self):
        l_shape = [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.9, 0.5), (0.9, 0.9), (0.1, 0.9)]
        self.assertFalse(main.polygon_self_intersects(l_shape))

    def test_concave_star_is_allowed(self):
        star = []
        for i in range(10):
            radius = 0.4 if i % 2 == 0 else 0.17
            angle = math.pi * i / 5 - math.pi / 2
            star.append((0.5 + radius * math.cos(angle), 0.5 + radius * math.sin(angle)))
        self.assertFalse(main.polygon_self_intersects(star))

    def test_bow_tie_crosses_itself(self):
        self.assertTrue(main.polygon_self_intersects([(0.1, 0.1), (0.9, 0.9), (0.9, 0.1), (0.1, 0.9)]))

    def test_polygon_that_touches_itself_at_a_point_is_rejected(self):
        figure_eight = [(0.1, 0.1), (0.9, 0.1), (0.5, 0.5), (0.9, 0.9), (0.1, 0.9), (0.5, 0.5)]
        self.assertTrue(main.polygon_self_intersects(figure_eight))

    def test_one_edge_pushed_across_another_is_rejected(self):
        # a rectangle whose top-left corner was dragged past the bottom edge
        self.assertTrue(main.polygon_self_intersects([(0.5, 0.95), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]))


class ValidateZoneTests(unittest.TestCase):
    """validate_zone(): a self-crossing shape is refused, and the message names the zone it is about."""
    def test_zone_that_crosses_itself_is_rejected_with_a_clear_message(self):
        with self.assertRaises(RuntimeError) as context:
            main.validate_zone([(0.1, 0.1), (0.9, 0.9), (0.9, 0.1), (0.1, 0.9)])
        self.assertIn("crosses itself", str(context.exception))

    def test_message_uses_the_given_label(self):
        with self.assertRaises(RuntimeError) as context:
            main.validate_zone([(0.1, 0.1), (0.9, 0.9)], label="The zone")
        self.assertIn("The zone", str(context.exception))
        self.assertNotIn("ZONE_POLYGON", str(context.exception))

    def test_validate_config_also_rejects_a_zone_that_crosses_itself(self):
        with self.assertRaises(RuntimeError) as context:
            main.validate_config([(0.1, 0.1), (0.9, 0.9), (0.9, 0.1), (0.1, 0.9)], 0.4, 3)
        self.assertIn("crosses itself", str(context.exception))


class ZoneFileTests(unittest.TestCase):
    """load_zone() / save_zone() / clear_zone(): the saved zone round-trips, and bad or missing files fall back safely."""
    DEFAULT = [(0.1, 0.1), (0.5, 0.1), (0.3, 0.5)]
    CUSTOM = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.6), (0.5, 0.9), (0.2, 0.6)]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "zone.json"

    def load_quietly(self):
        output = io.StringIO()
        with redirect_stdout(output):
            zone = main.load_zone(self.path, self.DEFAULT)
        return zone, output.getvalue()

    def test_missing_file_gives_the_default_zone(self):
        self.assertEqual(main.load_zone(self.path, self.DEFAULT), self.DEFAULT)

    def test_saved_zone_is_loaded_back(self):
        main.save_zone(self.path, self.CUSTOM)
        self.assertEqual(main.load_zone(self.path, self.DEFAULT), self.CUSTOM)

    def test_saved_file_is_plain_json_with_a_points_list(self):
        main.save_zone(self.path, self.CUSTOM)
        self.assertEqual(json.loads(self.path.read_text()), {"points": [list(p) for p in self.CUSTOM]})

    def test_points_are_rounded_to_four_decimals(self):
        main.save_zone(self.path, [(0.123456, 0.2), (0.8, 0.2), (0.5, 0.9)])
        self.assertEqual(main.load_zone(self.path, self.DEFAULT)[0], (0.1235, 0.2))

    def test_unreadable_file_falls_back_to_the_default_and_says_why(self):
        self.path.write_text("{ this is not json")
        zone, output = self.load_quietly()
        self.assertEqual(zone, self.DEFAULT)
        self.assertIn("Ignoring the saved zone", output)

    def test_file_holding_an_invalid_zone_falls_back_to_the_default(self):
        self.path.write_text(json.dumps({"points": [[100, 100], [300, 100], [300, 300]]}))
        zone, output = self.load_quietly()
        self.assertEqual(zone, self.DEFAULT)
        self.assertIn("between 0 and 1", output)

    def test_file_with_the_wrong_structure_falls_back_to_the_default(self):
        for content in ('{"shape": 1}', "[1, 2, 3]", '{"points": 5}', '{"points": [1, 2, 3]}'):
            with self.subTest(content=content):
                self.path.write_text(content)
                self.assertEqual(self.load_quietly()[0], self.DEFAULT)

    def test_saving_an_invalid_zone_is_rejected_and_writes_nothing(self):
        with self.assertRaises(RuntimeError) as context:
            main.save_zone(self.path, [(0.1, 0.1), (0.9, 0.9), (0.9, 0.1), (0.1, 0.9)])
        self.assertIn("crosses itself", str(context.exception))
        self.assertFalse(self.path.exists())

    def test_save_failure_is_reported(self):
        blocker = Path(self.tmp.name) / "blocker"
        blocker.write_text("a file, so it cannot be used as a folder")
        with self.assertRaises(RuntimeError) as context:
            main.save_zone(blocker / "zone.json", self.CUSTOM)
        self.assertIn("Could not save", str(context.exception))

    def test_clearing_removes_the_file_and_is_safe_when_there_is_none(self):
        main.save_zone(self.path, self.CUSTOM)
        main.clear_zone(self.path)
        self.assertFalse(self.path.exists())
        main.clear_zone(self.path)  # no file: must not raise


class ValidateConfigTests(unittest.TestCase):
    """validate_config(): zone shape, confidence threshold and confirmation window are checked up front."""
    GOOD_ZONE = [(0.2, 0.3), (0.6, 0.3), (0.6, 0.8), (0.2, 0.8)]

    def assert_rejected(self, expected_text, zone=None, confidence=0.4, confirm_frames=3):
        with self.assertRaises(RuntimeError) as context:
            main.validate_config(self.GOOD_ZONE if zone is None else zone, confidence, confirm_frames)
        self.assertIn(expected_text, str(context.exception))

    def test_valid_configuration_is_accepted(self):
        main.validate_config(self.GOOD_ZONE, 0.4, 3)

    def test_shipped_default_configuration_is_valid(self):
        main.validate_config(main.ZONE_POLYGON, main.CONFIDENCE_THRESHOLD, main.ZONE_CONFIRM_FRAMES)

    def test_zone_with_fewer_than_three_points_is_rejected(self):
        self.assert_rejected("at least 3 points", zone=[(0.1, 0.1), (0.9, 0.9)])

    def test_zone_that_is_not_a_list_is_rejected(self):
        self.assert_rejected("ZONE_POLYGON", zone="not a zone")

    def test_zone_given_in_pixels_instead_of_fractions_is_rejected(self):
        self.assert_rejected("between 0 and 1", zone=[(100, 100), (300, 100), (300, 300)])

    def test_negative_zone_coordinate_is_rejected(self):
        self.assert_rejected("between 0 and 1", zone=[(-0.1, 0.1), (0.5, 0.1), (0.5, 0.5)])

    def test_non_numeric_zone_coordinate_is_rejected(self):
        self.assert_rejected("numbers", zone=[(0.1, "a"), (0.5, 0.1), (0.5, 0.5)])

    def test_zone_point_that_is_not_an_x_y_pair_is_rejected(self):
        self.assert_rejected("(x, y) pair", zone=[(0.1, 0.1, 0.1), (0.5, 0.1), (0.5, 0.5)])

    def test_zone_whose_points_lie_on_a_line_is_rejected(self):
        self.assert_rejected("no area", zone=[(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)])

    def test_confidence_threshold_outside_zero_to_one_is_rejected(self):
        for bad in (0, -0.2, 1.5, "0.5", None):
            with self.subTest(confidence=bad):
                self.assert_rejected("CONFIDENCE_THRESHOLD", confidence=bad)

    def test_confirm_frames_must_be_a_whole_number_of_at_least_one(self):
        for bad in (0, -1, 2.5, "3", True):
            with self.subTest(confirm_frames=bad):
                self.assert_rejected("ZONE_CONFIRM_FRAMES", confirm_frames=bad)


def person(track_id, foot_x, foot_y, conf=0.9):
    """A tracked person (track_id, x1, y1, x2, y2, conf) whose box bottom-centre is (foot_x, foot_y)."""
    return (track_id, foot_x - 10, foot_y - 40, foot_x + 10, foot_y, conf)


INSIDE_Y, OUTSIDE_Y = 150, 50  # foot positions relative to SQUARE (x 100-200, y 100-200)


class UpdateMonitoringTests(unittest.TestCase):
    """Scripted people-per-frame scenarios through the real zone, entry and CSV logic."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.csv_path = Path(self.tmp.name) / "outputs" / "events.csv"
        self.zone_states = {}
        self.entry_count = 0

    def run_frame(self, people):
        inside_by_id, entered = main.update_monitoring(people, SQUARE, self.zone_states, self.csv_path)
        self.entry_count += len(entered)
        return inside_by_id

    def logged_ids(self):
        if not self.csv_path.exists():
            return []
        with open(self.csv_path, newline="") as file:
            return [int(row["tracking_id"]) for row in csv.DictReader(file)]

    def test_check_zone_maps_each_track_id_to_inside_or_outside(self):
        inside_by_id = main.check_zone([person(1, 150, INSIDE_Y), person(2, 150, OUTSIDE_Y)], SQUARE)
        self.assertEqual(inside_by_id, {1: True, 2: False})

    def test_person_outside_zone_records_no_entry(self):
        for _ in range(3):
            self.run_frame([person(1, 150, OUTSIDE_Y)])
        self.assertEqual((self.entry_count, self.logged_ids()), (0, []))
        self.assertFalse(self.csv_path.exists())

    def test_person_entering_increments_counter_once_and_logs_one_row(self):
        self.run_frame([person(1, 150, OUTSIDE_Y)])
        self.run_frame([person(1, 150, INSIDE_Y)])
        self.assertEqual((self.entry_count, self.logged_ids()), (1, [1]))

    def test_person_remaining_inside_adds_nothing(self):
        self.run_frame([person(1, 150, OUTSIDE_Y)])
        for _ in range(5):
            self.run_frame([person(1, 150, INSIDE_Y)])
        self.assertEqual((self.entry_count, self.logged_ids()), (1, [1]))

    def test_person_exiting_records_no_entry(self):
        self.run_frame([person(1, 150, OUTSIDE_Y)])
        self.run_frame([person(1, 150, INSIDE_Y)])
        self.run_frame([person(1, 150, OUTSIDE_Y)])
        self.assertEqual((self.entry_count, self.logged_ids()), (1, [1]))

    def test_person_reentering_increments_counter_again(self):
        for y in (OUTSIDE_Y, INSIDE_Y, OUTSIDE_Y, INSIDE_Y):
            self.run_frame([person(1, 150, y)])
        self.assertEqual((self.entry_count, self.logged_ids()), (2, [1, 1]))

    def test_multiple_people_entering_are_each_counted_and_logged(self):
        self.run_frame([person(1, 150, OUTSIDE_Y), person(2, 120, OUTSIDE_Y), person(3, 180, OUTSIDE_Y)])
        self.run_frame([person(1, 150, INSIDE_Y), person(2, 120, OUTSIDE_Y), person(3, 180, INSIDE_Y)])
        self.run_frame([person(1, 150, INSIDE_Y), person(2, 120, INSIDE_Y), person(3, 180, INSIDE_Y)])
        self.assertEqual(self.entry_count, 3)
        self.assertEqual(sorted(self.logged_ids()), [1, 2, 3])

    def test_zone_check_uses_feet_not_box_centre(self):
        # Box centre (150, 170) is inside SQUARE, but the bottom-centre (150, 220) is outside.
        self.run_frame([person(1, 150, OUTSIDE_Y)])
        self.run_frame([(1, 140, 120, 160, 220, 0.9)])
        self.assertEqual(self.entry_count, 0)

    def test_frame_with_no_people_is_handled(self):
        self.assertEqual(main.update_monitoring([], SQUARE, self.zone_states, self.csv_path), ({}, []))

    def test_flicker_across_the_edge_is_neither_counted_nor_logged_when_confirmation_is_on(self):
        for y in (OUTSIDE_Y, INSIDE_Y, INSIDE_Y, OUTSIDE_Y, OUTSIDE_Y):
            main.update_monitoring([person(1, 150, y)], SQUARE, self.zone_states, self.csv_path, confirm_frames=3)
        self.assertEqual(self.logged_ids(), [])
        self.assertFalse(self.csv_path.exists())

    def test_sustained_entry_is_logged_once_when_confirmation_is_on(self):
        for y in [OUTSIDE_Y] + [INSIDE_Y] * 6:
            main.update_monitoring([person(1, 150, y)], SQUARE, self.zone_states, self.csv_path, confirm_frames=3)
        self.assertEqual(self.logged_ids(), [1])


class TrackPeopleTests(unittest.TestCase):
    """Uses the real YOLO model and tracker."""

    @classmethod
    def setUpClass(cls):
        import cv2
        import ultralytics
        cls.bus = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))

    def setUp(self):
        self.model = main.load_model(main.MODEL_PATH)  # fresh model = fresh tracker state

    def test_frame_without_people_returns_empty_list(self):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertEqual(main.track_people(self.model, blank), [])

    def test_tracks_only_people_with_integer_ids(self):
        people = main.track_people(self.model, self.bus)  # bus.jpg has 3 people at conf >= 0.4 plus a bus
        self.assertEqual(len(people), 3)
        track_ids = [p[0] for p in people]
        self.assertTrue(all(isinstance(i, int) for i in track_ids))
        self.assertEqual(len(set(track_ids)), 3)
        for _, x1, y1, x2, y2, conf in people:
            self.assertTrue(x1 < x2 and y1 < y2 and main.CONFIDENCE_THRESHOLD <= conf <= 1)

    def test_same_person_keeps_id_across_consecutive_frames(self):
        first = {p[0] for p in main.track_people(self.model, self.bus)}
        second = {p[0] for p in main.track_people(self.model, self.bus)}
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
