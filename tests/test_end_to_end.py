"""End-to-end test: real YOLO model, real tracker, real video frames, real zone logic, drawing and CSV.

Only the OpenCV window calls are replaced, so no window opens. Uses the first 300 frames of the sample
video (about 30-40 s on CPU). Run: python -m unittest tests.test_end_to_end
"""
import csv
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main

SAMPLE_VIDEO = ROOT / "inputs" / "videos" / "vtest.mp4"
FRAME_LIMIT = 300


class FrameLimitedCapture:
    """A real video capture that reports end-of-video after `limit` frames."""

    def __init__(self, capture, limit):
        self.capture, self.limit, self.frames = capture, limit, 0

    def read(self):
        if self.frames >= self.limit:
            return False, None
        self.frames += 1
        return self.capture.read()

    def release(self):
        self.capture.release()


class SampleVideoEndToEndTests(unittest.TestCase):
    """Runs the real model over the start of inputs/videos/vtest.mp4 and checks the whole chain: every frame processed, entries counted, and the CSV matching the output."""
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.csv_path = Path(cls.tmp.name) / "outputs" / "events.csv"
        model = main.load_model(main.MODEL_PATH)
        capture = FrameLimitedCapture(main.open_source(SAMPLE_VIDEO), FRAME_LIMIT)
        output = io.StringIO()
        with mock.patch.object(main, "EVENTS_CSV", cls.csv_path), \
                mock.patch.object(main.cv2, "imshow") as imshow, \
                mock.patch.object(main.cv2, "waitKey", return_value=-1), \
                mock.patch.object(main.cv2, "getWindowProperty", return_value=1.0), \
                redirect_stdout(output):
            main.process_video(model, capture, str(SAMPLE_VIDEO))
        capture.release()
        cls.output = output.getvalue()
        cls.frames_shown = imshow.call_count
        with open(cls.csv_path, newline="") as file:
            cls.rows = list(csv.DictReader(file))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def printed_entries(self):
        return [line for line in self.output.splitlines() if line.startswith("ENTRY:")]

    def test_every_frame_was_processed_and_displayed(self):
        self.assertEqual(self.frames_shown, FRAME_LIMIT)
        self.assertIn("End of video.", self.output)

    def test_people_enter_the_zone_and_are_counted(self):
        self.assertGreaterEqual(len(self.rows), 5)
        self.assertGreaterEqual(len({row["tracking_id"] for row in self.rows}), 4)

    def test_csv_rows_match_the_printed_entries_and_the_final_count(self):
        self.assertEqual(len(self.rows), len(self.printed_entries()))
        self.assertIn(f"Entries counted this run: {len(self.rows)}", self.output)

    def test_csv_has_the_required_columns_and_only_entry_events(self):
        self.assertEqual(list(self.rows[0].keys()), ["timestamp", "tracking_id", "event_type"])
        self.assertEqual({row["event_type"] for row in self.rows}, {"ENTRY"})

    def test_csv_tracking_ids_are_positive_integers_and_timestamps_are_in_order(self):
        for row in self.rows:
            self.assertGreaterEqual(int(row["tracking_id"]), 1)
        stamps = [datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S") for row in self.rows]
        self.assertEqual(stamps, sorted(stamps))

    def test_printed_entry_numbers_count_up_without_gaps(self):
        totals = [int(line.rsplit(":", 1)[1].strip(" )")) for line in self.printed_entries()]
        self.assertEqual(totals, list(range(1, len(totals) + 1)))


if __name__ == "__main__":
    unittest.main()
