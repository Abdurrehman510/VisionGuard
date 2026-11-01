"""Tests for the VisionGuard web portal (portal.py): session statistics, zone editing, the monitoring
thread and the HTTP API. Frames and detections are scripted so the tests are fast; everything else
(zone logic, entry detection, CSV logging, JPEG encoding, the HTTP server) is real.
Run: python -m unittest tests.test_portal
"""
import csv
import http.client
import io
import json
import re
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main
import portal

DEFAULT_ZONE = [(0.1, 0.1), (0.4, 0.1), (0.25, 0.5)]
SQUARE_ZONE = [(0.4, 0.4), (0.9, 0.4), (0.9, 0.9), (0.4, 0.9)]  # pixels 64-144 x 48-108 on a 160x120 frame
FRAME_SIZE = (160, 120)
BLANK_FRAME = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
INSIDE_SQUARE, OUTSIDE_SQUARE = (100, 80), (20, 80)  # feet positions relative to SQUARE_ZONE


def person(track_id, feet):
    """A tracked person (track_id, x1, y1, x2, y2, conf) whose box bottom-centre is `feet`."""
    foot_x, foot_y = feet
    return (track_id, foot_x - 10, foot_y - 40, foot_x + 10, foot_y, 0.9)


class FakeClock:
    """A clock that only moves when the test says so: set .now, or let feed(advance=...) move it."""
    def __init__(self, now=1_700_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def wait_until(predicate, timeout=10.0):
    """Poll predicate() until it is true or `timeout` seconds pass; returns whether it became true."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class SessionTestCase(unittest.TestCase):
    """Base class: a Session with a fake clock and a temporary CSV file and zone file. feed() sends it one frame of people."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clock = FakeClock()
        self.csv_path = Path(self.tmp.name) / "outputs" / "events.csv"
        self.zone_file = Path(self.tmp.name) / "zone.json"
        self.session = self.make_session()

    def make_session(self, zone=None, confirm_frames=1, **options):
        return portal.Session(zone or SQUARE_ZONE, confirm_frames, self.csv_path, self.zone_file,
                              DEFAULT_ZONE, clock=self.clock, **options)

    def feed(self, *people, advance=0.0):
        self.clock.now += advance
        return self.session.process(list(people), FRAME_SIZE)

    def logged_ids(self):
        if not self.csv_path.exists():
            return []
        with open(self.csv_path, newline="") as file:
            return [int(row["tracking_id"]) for row in csv.DictReader(file)]


class SessionCountingTests(SessionTestCase):
    """Session.process(): entries are counted, logged and listed; the live counts and peaks are right."""
    def test_person_walking_into_the_zone_is_counted_logged_and_listed(self):
        self.feed(person(7, OUTSIDE_SQUARE))
        inside_by_id, entered = self.feed(person(7, INSIDE_SQUARE), advance=1.0)
        self.assertEqual((inside_by_id, entered), ({7: True}, [7]))
        snapshot = self.session.snapshot()
        self.assertEqual(snapshot["counts"]["entries"], 1)
        self.assertEqual(self.logged_ids(), [7])
        event = snapshot["events"][0]
        self.assertEqual((event["n"], event["id"]), (1, 7))
        self.assertEqual(event["time"], datetime.fromtimestamp(self.clock.now).strftime("%H:%M:%S"))

    def test_staying_inside_does_not_count_again(self):
        self.feed(person(1, OUTSIDE_SQUARE))
        for _ in range(5):
            self.feed(person(1, INSIDE_SQUARE))
        self.assertEqual(self.session.snapshot()["counts"]["entries"], 1)

    def test_live_counts_and_peaks(self):
        self.feed(person(1, OUTSIDE_SQUARE), person(2, OUTSIDE_SQUARE))
        self.feed(person(1, INSIDE_SQUARE), person(2, INSIDE_SQUARE), person(3, OUTSIDE_SQUARE))
        self.feed(person(3, OUTSIDE_SQUARE))
        counts = self.session.snapshot()["counts"]
        self.assertEqual(counts["inView"], 1)
        self.assertEqual(counts["inside"], 0)
        self.assertEqual(counts["uniqueIds"], 3)
        self.assertEqual(counts["peakInView"], 3)
        self.assertEqual(counts["peakInside"], 2)

    def test_confirmation_window_is_respected(self):
        session = self.make_session(confirm_frames=3)
        for feet in (OUTSIDE_SQUARE, INSIDE_SQUARE, INSIDE_SQUARE):
            session.process([person(1, feet)], FRAME_SIZE)
        self.assertEqual(session.snapshot()["counts"]["entries"], 0)
        session.process([person(1, INSIDE_SQUARE)], FRAME_SIZE)
        self.assertEqual(session.snapshot()["counts"]["entries"], 1)

    def test_only_the_newest_fifty_events_are_in_the_snapshot_but_the_report_has_all(self):
        for track_id in range(1, 61):
            self.feed(person(track_id, OUTSIDE_SQUARE))
            self.feed(person(track_id, INSIDE_SQUARE))
        snapshot = self.session.snapshot()
        self.assertEqual(len(snapshot["events"]), 50)
        self.assertEqual(snapshot["events"][0]["n"], 60)  # newest first
        self.assertEqual(len(self.session.report()["events"]), 60)

    def test_snapshot_can_be_sent_as_json(self):
        self.feed(person(1, OUTSIDE_SQUARE))
        json.dumps(self.session.snapshot())
        json.dumps(self.session.report())


class SessionHistoryTests(SessionTestCase):
    """Session history: one sample per second, a 3-minute window, and the per-minute entry buckets."""
    def test_one_history_sample_per_elapsed_second(self):
        for advance in (0, 0.4, 0.9, 0.2, 1.8, 3.0):  # clock reads 0, .4, 1.3, 1.5, 3.3, 6.3 s after the start
            self.feed(person(1, OUTSIDE_SQUARE), advance=advance)
        self.assertEqual([sample["t"] for sample in self.session.snapshot()["history"]], [0, 1, 3, 6])

    def test_history_keeps_only_the_most_recent_samples(self):
        for _ in range(portal.Session.HISTORY_SAMPLES + 20):
            self.feed(person(1, OUTSIDE_SQUARE), advance=1.0)
        history = self.session.snapshot()["history"]
        self.assertEqual(len(history), portal.Session.HISTORY_SAMPLES)
        self.assertEqual(history[-1]["t"], portal.Session.HISTORY_SAMPLES + 20)  # one frame per second, 200 frames

    def test_history_records_how_many_are_in_view_and_inside(self):
        self.feed(person(1, INSIDE_SQUARE), person(2, OUTSIDE_SQUARE))
        sample = self.session.snapshot()["history"][0]
        self.assertEqual((sample["inView"], sample["inside"]), (2, 1))

    def test_entries_per_minute_are_bucketed_relative_to_now(self):
        self.feed(person(1, OUTSIDE_SQUARE), person(2, OUTSIDE_SQUARE), person(3, OUTSIDE_SQUARE))
        self.feed(person(1, INSIDE_SQUARE))                                      # entry now
        self.feed(person(2, INSIDE_SQUARE), advance=100)                         # entry now, first is 100 s old
        self.feed(person(3, INSIDE_SQUARE), advance=61)                          # entry now, others 161 s and 61 s old
        buckets = self.session.snapshot()["perMinute"]
        self.assertEqual(len(buckets), 10)
        self.assertEqual((buckets[-1]["label"], buckets[-1]["count"]), ("now", 1))
        self.assertEqual(buckets[-2]["count"], 1)   # 61 s old
        self.assertEqual(buckets[-3]["count"], 1)   # 161 s old (2-3 minutes)
        self.assertEqual(sum(bucket["count"] for bucket in buckets), 3)

    def test_old_entries_fall_out_of_the_ten_minute_window(self):
        self.feed(person(1, OUTSIDE_SQUARE))
        self.feed(person(1, INSIDE_SQUARE))
        self.clock.now += 11 * 60
        self.feed()
        self.assertEqual(sum(b["count"] for b in self.session.snapshot()["perMinute"]), 0)


class SessionZoneTests(SessionTestCase):
    """Session.set_zone() / reset_zone(): a new zone is validated, saved and applied from the next frame."""
    def test_new_zone_is_applied_saved_and_reported(self):
        new_zone = [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)]
        self.session.set_zone(new_zone)
        zone = self.session.snapshot()["zone"]
        self.assertEqual([tuple(p) for p in zone["points"]], new_zone)
        self.assertFalse(zone["isDefault"])
        self.assertAlmostEqual(zone["areaPct"], 16.0, places=1)
        self.assertEqual(main.load_zone(self.zone_file, DEFAULT_ZONE), new_zone)

    def test_entries_are_judged_against_the_new_zone(self):
        self.feed(person(1, (20, 20)))
        self.session.set_zone([(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)])  # covers pixels 0-80 x 0-60
        self.feed(person(1, (20, 100)))
        _, entered = self.feed(person(1, (20, 50)))
        self.assertEqual(entered, [1])

    def test_person_already_inside_when_the_zone_changes_is_not_counted(self):
        self.feed(person(1, INSIDE_SQUARE))
        self.session.set_zone([(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)])  # still contains (100, 80)
        _, entered = self.feed(person(1, INSIDE_SQUARE))
        self.assertEqual(entered, [])
        self.assertEqual(self.session.snapshot()["counts"]["entries"], 0)

    def test_invalid_zone_is_rejected_and_the_old_zone_stays(self):
        for bad in ([(0.1, 0.1), (0.9, 0.9), (0.9, 0.1), (0.1, 0.9)], [(0.1, 0.1), (0.5, 0.5)], [(10, 10), (300, 10), (300, 300)]):
            with self.subTest(zone=bad), self.assertRaises(RuntimeError):
                self.session.set_zone(bad)
        self.assertEqual([tuple(p) for p in self.session.snapshot()["zone"]["points"]], SQUARE_ZONE)
        self.assertFalse(self.zone_file.exists())

    def test_reset_zone_restores_the_default_and_removes_the_saved_file(self):
        self.session.set_zone([(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)])
        self.session.reset_zone()
        zone = self.session.snapshot()["zone"]
        self.assertEqual([tuple(p) for p in zone["points"]], DEFAULT_ZONE)
        self.assertTrue(zone["isDefault"])
        self.assertFalse(self.zone_file.exists())


class SessionResetAndReportTests(SessionTestCase):
    """Session.reset() keeps the zone but clears the numbers; report() summarises the session."""
    def test_reset_clears_the_numbers_but_keeps_the_zone(self):
        self.feed(person(1, OUTSIDE_SQUARE))
        self.feed(person(1, INSIDE_SQUARE))
        self.session.reset()
        snapshot = self.session.snapshot()
        self.assertEqual(snapshot["counts"], {"entries": 0, "inside": 0, "inView": 0, "uniqueIds": 0,
                                              "peakInView": 0, "peakInside": 0})
        self.assertEqual((snapshot["events"], snapshot["history"]), ([], []))
        self.assertEqual([tuple(p) for p in snapshot["zone"]["points"]], SQUARE_ZONE)

    def test_person_inside_after_a_reset_is_not_counted_as_an_entry(self):
        self.feed(person(1, INSIDE_SQUARE))
        self.session.reset()
        _, entered = self.feed(person(1, INSIDE_SQUARE))
        self.assertEqual(entered, [])

    def test_report_summarises_the_session(self):
        self.feed(person(1, OUTSIDE_SQUARE))
        self.feed(person(1, INSIDE_SQUARE), advance=30)
        self.feed(person(1, OUTSIDE_SQUARE), advance=30)
        self.feed(person(1, INSIDE_SQUARE), advance=60)
        report = self.session.report()
        self.assertEqual(report["counts"]["entries"], 2)
        self.assertEqual(report["durationSeconds"], 120)
        self.assertEqual(report["entriesPerMinute"], 1.0)
        self.assertEqual([event["n"] for event in report["events"]], [1, 2])  # oldest first
        self.assertEqual(report["firstEntry"], report["events"][0]["time"])
        self.assertEqual(report["lastEntry"], report["events"][1]["time"])

    def test_report_for_a_session_without_entries(self):
        report = self.session.report()
        self.assertEqual((report["firstEntry"], report["lastEntry"], report["entriesPerMinute"]), (None, None, 0.0))


class FakeCapture:
    """A stand-in for cv2.VideoCapture that serves blank frames: read() and grab() count down from frame_count (None = endless); release() is recorded."""
    def __init__(self, frame_count=None, delay=0.0, fps=None):
        self.remaining, self.delay, self.released = frame_count, delay, False
        self.total, self.fps = frame_count, fps

    def read(self):
        if self.delay:
            time.sleep(self.delay)
        if self.remaining is not None:
            if self.remaining <= 0:
                return False, None
            self.remaining -= 1
        return True, BLANK_FRAME.copy()

    def grab(self):
        if self.remaining is not None:
            if self.remaining <= 0:
                return False
            self.remaining -= 1
        return True

    def get(self, prop):
        if prop == main.cv2.CAP_PROP_FPS:
            return float(self.fps or 0)
        if prop == main.cv2.CAP_PROP_FRAME_COUNT:
            return float(self.total or 0)
        return 0.0

    def release(self):
        self.released = True


class FakeTime:
    """A clock the test controls; sleeping just moves it forward."""

    def __init__(self, now=1000.0):
        self.start = self.now = now

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    @property
    def elapsed(self):
        return self.now - self.start


class TimedDetector:
    """A detector that takes `seconds` of fake time per frame, finds nobody, and counts its calls."""

    def __init__(self, fake_time, seconds):
        self.time, self.seconds, self.calls = fake_time, seconds, 0

    def __call__(self, frame):
        self.calls += 1
        self.time.sleep(self.seconds)
        return []


class ScriptedDetector:
    """Returns the scripted list of people for each frame, then nobody."""

    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def __call__(self, frame):
        index, self.calls = self.calls, self.calls + 1
        return self.script[index] if index < len(self.script) else []


def make_monitor(session, capture_factory, detector, is_webcam=False, source_label="test source", **options):
    """A Monitor wired to a scripted capture and detector instead of a real camera and model."""
    return portal.Monitor(session, open_source=capture_factory, make_detector=lambda: detector,
                          source_label=source_label, is_webcam=is_webcam, **options)


class MonitorTests(SessionTestCase):
    """Monitor (the background thread): processing, JPEG frames, error states, pause, restart and stop, all with scripted frames and detections."""
    def tearDown(self):
        if getattr(self, "monitor", None):
            self.monitor.stop()

    def start(self, capture_factory, detector, **kwargs):
        self.monitor = make_monitor(self.session, capture_factory, detector, **kwargs)
        self.monitor.start()
        return self.monitor

    def test_scripted_video_is_processed_until_it_ends(self):
        script = [[person(1, OUTSIDE_SQUARE)], [person(1, INSIDE_SQUARE)], [person(1, INSIDE_SQUARE)]]
        captures = []
        monitor = self.start(lambda: captures.append(FakeCapture(frame_count=3)) or captures[-1], ScriptedDetector(script))
        self.assertTrue(wait_until(lambda: monitor.state()["status"] == "ended"))
        state = monitor.state()
        self.assertEqual(state["counts"]["entries"], 1)
        self.assertEqual(state["frame"], {"w": FRAME_SIZE[0], "h": FRAME_SIZE[1]})
        self.assertIn("ended", state["message"].lower())
        self.assertTrue(captures[0].released)

    def test_frames_are_published_as_jpeg_images(self):
        monitor = self.start(lambda: FakeCapture(frame_count=2), ScriptedDetector([[person(1, INSIDE_SQUARE)]]))
        self.assertTrue(wait_until(lambda: monitor.state()["status"] == "ended"))
        frame_id, jpeg = monitor.wait_for_frame(0, timeout=1)
        self.assertGreater(frame_id, 0)
        self.assertEqual(jpeg[:2], b"\xff\xd8")

    def test_unavailable_source_gives_an_error_state_with_the_reason(self):
        def broken_source():
            raise RuntimeError("Webcam 0 is unavailable (not connected or in use).")
        monitor = self.start(broken_source, ScriptedDetector([]), is_webcam=True)
        self.assertTrue(wait_until(lambda: monitor.state()["status"] == "error"))
        self.assertIn("Webcam 0 is unavailable", monitor.state()["message"])

    def test_detector_that_cannot_start_gives_an_error_state(self):
        monitor = portal.Monitor(self.session, open_source=lambda: FakeCapture(frame_count=1),
                                 make_detector=lambda: (_ for _ in ()).throw(RuntimeError("Could not load YOLO model")),
                                 source_label="x", is_webcam=False)
        self.monitor = monitor
        monitor.start()
        self.assertTrue(wait_until(lambda: monitor.state()["status"] == "error"))
        self.assertIn("Could not load YOLO model", monitor.state()["message"])

    def test_crash_while_processing_is_reported_and_the_source_is_released(self):
        captures = []

        def crashing_detector(frame):
            raise ValueError("boom")
        monitor = self.start(lambda: captures.append(FakeCapture()) or captures[-1], crashing_detector)
        self.assertTrue(wait_until(lambda: monitor.state()["status"] == "error"))
        self.assertIn("boom", monitor.state()["message"])
        self.assertTrue(wait_until(lambda: captures[0].released))

    def test_webcam_that_stops_sending_frames_is_an_error_not_an_end(self):
        monitor = self.start(lambda: FakeCapture(frame_count=1), ScriptedDetector([]), is_webcam=True)
        self.assertTrue(wait_until(lambda: monitor.state()["status"] == "error"))
        self.assertIn("camera", monitor.state()["message"].lower())

    def test_pausing_freezes_the_numbers_and_resuming_continues(self):
        script = [[person(1, OUTSIDE_SQUARE)]] + [[person(1, INSIDE_SQUARE)]] * 1000
        monitor = self.start(lambda: FakeCapture(delay=0.01), ScriptedDetector(script))
        self.assertTrue(wait_until(lambda: monitor.state()["status"] == "monitoring"))
        monitor.pause(True)
        self.assertTrue(wait_until(lambda: monitor.state()["status"] == "paused"))
        time.sleep(0.2)
        frozen = monitor.state()["counts"]
        time.sleep(0.3)
        self.assertEqual(monitor.state()["counts"], frozen)
        monitor.pause(False)
        self.assertTrue(wait_until(lambda: monitor.state()["status"] == "monitoring"))

    def test_restart_reopens_the_source_and_resets_the_numbers(self):
        script = [[person(1, OUTSIDE_SQUARE)], [person(1, INSIDE_SQUARE)]]
        opened = []
        monitor = self.start(lambda: opened.append(FakeCapture(frame_count=2)) or opened[-1], ScriptedDetector(script))
        self.assertTrue(wait_until(lambda: monitor.state()["status"] == "ended"))
        self.assertEqual(monitor.state()["counts"]["entries"], 1)
        monitor.restart()
        self.assertTrue(wait_until(lambda: len(opened) == 2))
        self.assertTrue(wait_until(lambda: monitor.state()["status"] == "ended"))
        self.assertEqual(monitor.state()["counts"]["entries"], 0)  # the detector script is used up, so nobody is found

    def test_state_reports_source_and_rate(self):
        monitor = self.start(lambda: FakeCapture(delay=0.01), ScriptedDetector([]), is_webcam=True, source_label="Webcam 0")
        self.assertTrue(wait_until(lambda: monitor.state()["fps"] > 0))
        state = monitor.state()
        self.assertEqual((state["source"], state["isWebcam"]), ("Webcam 0", True))
        json.dumps(state)

    def test_stop_ends_the_thread_and_releases_the_source(self):
        captures = []
        monitor = self.start(lambda: captures.append(FakeCapture(delay=0.01)) or captures[-1], ScriptedDetector([]))
        self.assertTrue(wait_until(lambda: monitor.state()["status"] == "monitoring"))
        monitor.stop()
        self.assertTrue(captures[0].released)


class MonitorSpeedTests(SessionTestCase):
    """Playing a video at a chosen speed. Time is fake, so a 10-second video 'plays' instantly."""

    def tearDown(self):
        if getattr(self, "monitor", None):
            self.monitor.stop()

    def play(self, frames, fps, detect_seconds, speed):
        self.fake_time = FakeTime()
        self.detector = TimedDetector(self.fake_time, detect_seconds)
        self.monitor = make_monitor(self.session, lambda: FakeCapture(frame_count=frames, fps=fps), self.detector,
                                    speed=speed, clock=self.fake_time.clock, sleep=self.fake_time.sleep)
        self.monitor.start()
        self.assertTrue(wait_until(lambda: self.monitor.state()["status"] == "ended"))
        return self.monitor.state()

    def test_speed_zero_uses_every_frame(self):
        state = self.play(frames=600, fps=60, detect_seconds=0.5, speed=0)
        self.assertEqual(self.detector.calls, 600)
        self.assertEqual(state["video"]["position"], 10.0)

    def test_normal_speed_skips_frames_a_slow_detector_cannot_keep_up_with(self):
        state = self.play(frames=600, fps=60, detect_seconds=0.5, speed=1.0)
        self.assertTrue(18 <= self.detector.calls <= 23, f"detector ran {self.detector.calls} times")
        self.assertEqual(state["video"]["position"], 10.0)  # the whole 10-second video was still played through

    def test_double_speed_processes_even_fewer_frames(self):
        self.play(frames=600, fps=60, detect_seconds=0.1, speed=2.0)
        self.assertTrue(48 <= self.detector.calls <= 53, f"detector ran {self.detector.calls} times")

    def test_normal_speed_waits_when_the_detector_is_faster_than_the_video(self):
        self.play(frames=20, fps=10, detect_seconds=0.0, speed=1.0)
        self.assertEqual(self.detector.calls, 20)
        self.assertTrue(1.7 <= self.fake_time.elapsed <= 2.3, f"took {self.fake_time.elapsed:.2f} s for a 2 s video")

    def test_state_reports_position_length_and_speed(self):
        state = self.play(frames=300, fps=30, detect_seconds=0.0, speed=4.0)
        self.assertEqual(state["video"], {"position": 10.0, "duration": 10.0, "speed": 4.0, "fps": 30.0})
        json.dumps(state)

    def test_a_webcam_has_no_video_position_and_refuses_a_speed(self):
        self.monitor = make_monitor(self.session, lambda: FakeCapture(delay=0.01, fps=30), ScriptedDetector([]), is_webcam=True)
        self.monitor.start()
        self.assertTrue(wait_until(lambda: self.monitor.state()["status"] == "monitoring"))
        self.assertIsNone(self.monitor.state()["video"])
        with self.assertRaises(RuntimeError) as context:
            self.monitor.set_speed(2)
        self.assertIn("video", str(context.exception).lower())

    def test_a_video_whose_frame_rate_is_unknown_has_no_position(self):
        self.monitor = make_monitor(self.session, lambda: FakeCapture(frame_count=5), ScriptedDetector([]), speed=1.0)
        self.monitor.start()
        self.assertTrue(wait_until(lambda: self.monitor.state()["status"] == "ended"))
        self.assertIsNone(self.monitor.state()["video"])

    def test_speed_can_be_changed_while_the_video_plays(self):
        self.monitor = make_monitor(self.session, lambda: FakeCapture(delay=0.01, fps=30), ScriptedDetector([]))
        self.monitor.start()
        self.assertTrue(wait_until(lambda: self.monitor.state()["video"] is not None))
        self.assertEqual(self.monitor.state()["video"]["speed"], 0.0)
        self.monitor.set_speed(2)
        self.assertEqual(self.monitor.state()["video"]["speed"], 2.0)

    def test_bad_speeds_are_refused_and_the_speed_is_unchanged(self):
        self.monitor = make_monitor(self.session, lambda: FakeCapture(delay=0.01, fps=30), ScriptedDetector([]), speed=1.0)
        self.monitor.start()
        self.assertTrue(wait_until(lambda: self.monitor.state()["video"] is not None))
        for bad in (-1, 99, "fast", None, True):
            with self.subTest(speed=bad), self.assertRaises(RuntimeError):
                self.monitor.set_speed(bad)
        self.assertEqual(self.monitor.state()["video"]["speed"], 1.0)

    def test_pausing_does_not_make_the_video_race_ahead_afterwards(self):
        self.monitor = make_monitor(self.session, lambda: FakeCapture(delay=0.02, fps=50), ScriptedDetector([]), speed=1.0)
        self.monitor.start()
        self.assertTrue(wait_until(lambda: self.monitor.state()["video"] is not None and self.monitor.state()["video"]["position"] > 0.3))
        self.monitor.pause(True)
        self.assertTrue(wait_until(lambda: self.monitor.state()["status"] == "paused"))
        position_when_paused = self.monitor.state()["video"]["position"]
        time.sleep(1.0)
        self.monitor.pause(False)
        self.assertTrue(wait_until(lambda: self.monitor.state()["status"] == "monitoring"))
        time.sleep(0.4)
        jump = self.monitor.state()["video"]["position"] - position_when_paused
        self.assertLess(jump, 1.0, f"the video jumped {jump:.1f} s after resuming")


class PortalSpeedArgumentTests(unittest.TestCase):
    """The portal's --speed command-line option: default, parsing, and a clear error for a bad value."""
    def test_default_is_every_frame(self):
        self.assertEqual(portal.parse_args(["video.mp4"]).speed, 0.0)

    def test_speed_option_is_read(self):
        args = portal.parse_args(["video.mp4", "--speed", "2"])
        self.assertEqual((args.source, args.speed), ("video.mp4", 2.0))

    def test_a_bad_speed_stops_with_a_message(self):
        for bad in ("-1", "99", "fast"):
            with self.subTest(speed=bad), self.assertRaises(SystemExit), redirect_stderr(io.StringIO()) as errors:
                portal.parse_args(["video.mp4", "--speed", bad])
            self.assertIn("speed", errors.getvalue().lower())


class ReportTests(SessionTestCase):
    """render_report() and session_events_csv(): the printable report and the downloadable CSV."""
    def test_report_page_contains_the_numbers_and_every_event(self):
        self.feed(person(1, OUTSIDE_SQUARE))
        self.feed(person(1, INSIDE_SQUARE))
        self.feed(person(2, OUTSIDE_SQUARE))
        self.feed(person(2, INSIDE_SQUARE))
        page = portal.render_report(self.session.report(), source="Webcam 0", frame={"w": 640, "h": 480})
        self.assertIn("<!doctype html>", page.lower())
        self.assertIn("Session report", page)
        self.assertIn("Webcam 0", page)
        self.assertEqual(page.count(">ENTRY<"), 2)  # one table cell per event
        self.assertIn("temporary", page)  # the honest note about tracking IDs

    def test_report_page_escapes_the_source_name(self):
        page = portal.render_report(self.session.report(), source="<script>alert(1)</script>", frame=None)
        self.assertNotIn("<script>alert(1)</script>", page)
        self.assertIn("&lt;script&gt;", page)

    @staticmethod
    def report_css(page):
        return page[page.index("<style>"):page.index("</style>")]

    def test_each_heading_stays_with_its_content_when_printed(self):
        page = portal.render_report(self.session.report(), source="x", frame=None)
        headings = page.count("<h2>")
        self.assertGreaterEqual(headings, 5)
        self.assertEqual(len(re.findall(r"<section[^>]*>\s*<h2>", page)), headings)   # every heading opens its own section
        css = self.report_css(page)
        self.assertRegex(css, r"section\s*\{[^}]*break-inside:\s*avoid")               # a section is not split across pages
        self.assertRegex(css, r"h2\s*\{[^}]*break-after:\s*avoid")                     # a heading is never left at the page bottom

    def test_a_long_entry_log_can_still_run_over_several_pages_without_splitting_rows(self):
        css = self.report_css(portal.render_report(self.session.report(), source="x", frame=None))
        self.assertRegex(css, r"section\.log\s*\{[^}]*break-inside:\s*auto")
        self.assertRegex(css, r"tr\s*\{[^}]*break-inside:\s*avoid")
        self.assertRegex(css, r"thead\s*\{[^}]*display:\s*table-header-group")         # column titles repeat on each page

    def test_report_for_an_empty_session_says_so(self):
        page = portal.render_report(self.session.report(), source="x", frame=None)
        self.assertIn("No entries", page)

    def test_session_csv_has_the_same_columns_as_the_event_log(self):
        self.feed(person(4, OUTSIDE_SQUARE))
        self.feed(person(4, INSIDE_SQUARE))
        rows = list(csv.DictReader(io.StringIO(portal.session_events_csv(self.session.report()))))
        self.assertEqual(list(rows[0].keys()), ["timestamp", "tracking_id", "event_type"])
        self.assertEqual((rows[0]["tracking_id"], rows[0]["event_type"]), ("4", "ENTRY"))
        datetime.strptime(rows[0]["timestamp"], "%Y-%m-%d %H:%M:%S")


class RunningPortal:
    """A real HTTP server on a free port, backed by a real Session and Monitor fed with scripted frames."""

    def __init__(self, test_case, script=None, frame_count=None, capture_factory=None, is_webcam=False):
        tmp = tempfile.TemporaryDirectory()
        test_case.addCleanup(tmp.cleanup)
        self.csv_path = Path(tmp.name) / "events.csv"
        self.zone_file = Path(tmp.name) / "zone.json"
        self.session = portal.Session(SQUARE_ZONE, 1, self.csv_path, self.zone_file, DEFAULT_ZONE)
        factory = capture_factory or (lambda: FakeCapture(frame_count=frame_count, delay=0.01))
        self.monitor = make_monitor(self.session, factory, ScriptedDetector(script or []), is_webcam=is_webcam)
        self.server = portal.make_server(self.monitor, self.session, "127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.monitor.start()
        test_case.addCleanup(self.close)

    def close(self):
        self.monitor.stop()
        self.server.shutdown()
        self.server.server_close()

    def request(self, method, path, body=None, headers=None, raw_body=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = dict(headers or {})
        data = raw_body
        if body is not None:
            data = json.dumps(body).encode()
            headers.setdefault("Content-Type", "application/json")
        connection.request(method, path, body=data, headers=headers)
        response = connection.getresponse()
        payload = response.read()
        connection.close()
        return response.status, response.getheader("Content-Type"), payload

    def get_json(self, path):
        status, _, payload = self.request("GET", path)
        return status, json.loads(payload)

    def post_json(self, path, body=None, headers=None):
        status, _, payload = self.request("POST", path, body={} if body is None else body, headers=headers)
        return status, json.loads(payload)


class HttpApiTests(unittest.TestCase):
    """The real HTTP server on a free port: pages, state, MJPEG stream and every POST action."""
    def test_state_endpoint_returns_the_live_numbers(self):
        script = [[person(1, OUTSIDE_SQUARE)]] + [[person(1, INSIDE_SQUARE)]] * 500
        portal_ = RunningPortal(self, script)
        self.assertTrue(wait_until(lambda: portal_.get_json("/api/state")[1]["counts"]["entries"] == 1))
        status, state = portal_.get_json("/api/state")
        self.assertEqual(status, 200)
        for key in ("status", "message", "source", "isWebcam", "frame", "fps", "uptime", "counts", "zone",
                    "history", "perMinute", "events", "confirmFrames"):
            self.assertIn(key, state)
        self.assertEqual(state["events"][0]["id"], 1)

    def test_page_and_report_and_csv_are_served(self):
        script = [[person(1, OUTSIDE_SQUARE)]] + [[person(1, INSIDE_SQUARE)]] * 500
        portal_ = RunningPortal(self, script)
        self.assertTrue(wait_until(lambda: portal_.get_json("/api/state")[1]["counts"]["entries"] == 1))
        status, content_type, page = portal_.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        self.assertIn(b"VisionGuard", page)
        status, content_type, report = portal_.request("GET", "/report")
        self.assertEqual(status, 200)
        self.assertIn(b"Session report", report)
        status, content_type, payload = portal_.request("GET", "/api/events.csv")
        self.assertEqual(status, 200)
        self.assertIn("text/csv", content_type)
        self.assertTrue(payload.decode().startswith("timestamp,tracking_id,event_type"))

    def test_mjpeg_stream_delivers_jpeg_frames(self):
        portal_ = RunningPortal(self, [])
        connection = http.client.HTTPConnection("127.0.0.1", portal_.port, timeout=10)
        connection.request("GET", "/stream.mjpg")
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertIn("multipart/x-mixed-replace", response.getheader("Content-Type"))
        chunk = response.read(4000)
        connection.close()
        self.assertIn(b"Content-Type: image/jpeg", chunk)
        self.assertIn(b"\xff\xd8", chunk)

    def test_zone_can_be_changed_to_any_valid_shape(self):
        portal_ = RunningPortal(self, [])
        l_shape = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.9, 0.5], [0.9, 0.9], [0.1, 0.9]]
        status, body = portal_.post_json("/api/zone", {"points": l_shape})
        self.assertEqual((status, body["ok"]), (200, True))
        self.assertEqual(body["zone"]["points"], l_shape)
        self.assertEqual(portal_.get_json("/api/state")[1]["zone"]["points"], l_shape)
        self.assertEqual(main.load_zone(portal_.zone_file, DEFAULT_ZONE), [tuple(p) for p in l_shape])

    def test_invalid_zone_is_refused_with_a_message_and_nothing_changes(self):
        portal_ = RunningPortal(self, [])
        status, body = portal_.post_json("/api/zone", {"points": [[0.1, 0.1], [0.9, 0.9], [0.9, 0.1], [0.1, 0.9]]})
        self.assertEqual((status, body["ok"]), (400, False))
        self.assertIn("crosses itself", body["error"])
        self.assertEqual([tuple(p) for p in portal_.get_json("/api/state")[1]["zone"]["points"]], SQUARE_ZONE)

    def test_zone_request_without_points_is_refused(self):
        portal_ = RunningPortal(self, [])
        self.assertEqual(portal_.post_json("/api/zone", {"shape": 1})[0], 400)
        self.assertEqual(portal_.post_json("/api/zone", {"points": "not a list"})[0], 400)

    def test_zone_reset_restores_the_default(self):
        portal_ = RunningPortal(self, [])
        portal_.post_json("/api/zone", {"points": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]})
        status, body = portal_.post_json("/api/zone/reset")
        self.assertEqual((status, body["zone"]["isDefault"]), (200, True))
        self.assertFalse(portal_.zone_file.exists())

    def test_pause_and_resume(self):
        portal_ = RunningPortal(self, [])
        self.assertTrue(wait_until(lambda: portal_.get_json("/api/state")[1]["status"] == "monitoring"))
        self.assertEqual(portal_.post_json("/api/pause", {"paused": True})[0], 200)
        self.assertTrue(wait_until(lambda: portal_.get_json("/api/state")[1]["status"] == "paused"))
        portal_.post_json("/api/pause", {"paused": False})
        self.assertTrue(wait_until(lambda: portal_.get_json("/api/state")[1]["status"] == "monitoring"))

    def test_reset_counters(self):
        script = [[person(1, OUTSIDE_SQUARE)]] + [[person(1, INSIDE_SQUARE)]] * 3
        portal_ = RunningPortal(self, script)
        self.assertTrue(wait_until(lambda: portal_.get_json("/api/state")[1]["counts"]["entries"] == 1))
        portal_.post_json("/api/reset")
        self.assertEqual(portal_.get_json("/api/state")[1]["counts"]["entries"], 0)

    def test_restart_reopens_the_source(self):
        opened = []
        portal_ = RunningPortal(self, [], capture_factory=lambda: opened.append(FakeCapture(frame_count=3)) or opened[-1])
        self.assertTrue(wait_until(lambda: portal_.get_json("/api/state")[1]["status"] == "ended"))
        self.assertEqual(portal_.post_json("/api/restart")[0], 200)
        self.assertTrue(wait_until(lambda: len(opened) == 2))

    def test_playback_speed_can_be_changed_and_is_reported(self):
        portal_ = RunningPortal(self, [], capture_factory=lambda: FakeCapture(delay=0.01, fps=30))
        self.assertTrue(wait_until(lambda: portal_.get_json("/api/state")[1]["video"] is not None))
        self.assertEqual(portal_.get_json("/api/state")[1]["video"]["speed"], 0.0)
        status, body = portal_.post_json("/api/speed", {"speed": 2})
        self.assertEqual((status, body["ok"], body["speed"]), (200, True, 2.0))
        self.assertEqual(portal_.get_json("/api/state")[1]["video"]["speed"], 2.0)

    def test_bad_playback_speeds_are_refused_with_a_message(self):
        portal_ = RunningPortal(self, [], capture_factory=lambda: FakeCapture(delay=0.01, fps=30))
        self.assertTrue(wait_until(lambda: portal_.get_json("/api/state")[1]["video"] is not None))
        for bad in (-1, 99, "fast", None, True):
            with self.subTest(speed=bad):
                status, body = portal_.post_json("/api/speed", {"speed": bad})
                self.assertEqual((status, body["ok"]), (400, False))
                self.assertIn("speed", body["error"].lower())
        self.assertEqual(portal_.post_json("/api/speed", {})[0], 400)
        self.assertEqual(portal_.get_json("/api/state")[1]["video"]["speed"], 0.0)

    def test_playback_speed_is_refused_for_a_webcam(self):
        portal_ = RunningPortal(self, [], is_webcam=True)
        status, body = portal_.post_json("/api/speed", {"speed": 2})
        self.assertEqual((status, body["ok"]), (400, False))
        self.assertIn("video", body["error"].lower())

    def test_unknown_page_is_a_404_and_unknown_method_is_refused(self):
        portal_ = RunningPortal(self, [])
        self.assertEqual(portal_.request("GET", "/nope")[0], 404)
        self.assertEqual(portal_.request("POST", "/nope", body={})[0], 404)


class HttpSecurityTests(unittest.TestCase):
    """The portal is for this computer only: it must refuse requests that did not come from its own page."""

    def test_request_with_a_foreign_host_header_is_refused(self):
        portal_ = RunningPortal(self, [])
        self.assertEqual(portal_.request("GET", "/api/state", headers={"Host": "evil.example"})[0], 403)
        self.assertEqual(portal_.request("GET", "/stream.mjpg", headers={"Host": "evil.example:80"})[0], 403)

    def test_localhost_name_is_accepted(self):
        portal_ = RunningPortal(self, [])
        status, _, _ = portal_.request("GET", "/api/state", headers={"Host": f"localhost:{portal_.port}"})
        self.assertEqual(status, 200)

    def test_post_from_another_website_is_refused(self):
        portal_ = RunningPortal(self, [])
        status, body = portal_.post_json("/api/pause", {"paused": True}, headers={"Origin": "https://evil.example"})
        self.assertEqual(status, 403)
        self.assertEqual(portal_.get_json("/api/state")[1]["status"] != "paused", True)

    def test_post_from_the_portal_page_itself_is_accepted(self):
        portal_ = RunningPortal(self, [])
        status, _ = portal_.post_json("/api/pause", {"paused": False}, headers={"Origin": f"http://127.0.0.1:{portal_.port}"})
        self.assertEqual(status, 200)

    def test_post_that_is_not_json_is_refused(self):
        portal_ = RunningPortal(self, [])
        status, _, _ = portal_.request("POST", "/api/reset", raw_body=b"a=1", headers={"Content-Type": "application/x-www-form-urlencoded"})
        self.assertEqual(status, 415)

    def test_broken_json_is_a_400_not_a_crash(self):
        portal_ = RunningPortal(self, [])
        status, _, _ = portal_.request("POST", "/api/zone", raw_body=b"{not json", headers={"Content-Type": "application/json"})
        self.assertEqual(status, 400)
        self.assertEqual(portal_.get_json("/api/state")[0], 200)

    def test_oversized_request_is_refused(self):
        portal_ = RunningPortal(self, [])
        status, _, _ = portal_.request("POST", "/api/zone", raw_body=b"x" * 200_000, headers={"Content-Type": "application/json"})
        self.assertEqual(status, 413)


# --- Zone insights: counting rule, manual-count check, foot heat-map ------------------------------------

FEET_BELOW_ZONE = (100, 125)  # the body reaches the zone but the feet are below it (SQUARE_ZONE ends at y=108)
CLIPPING_ONLY = (150, 50)     # feet and middle are outside SQUARE_ZONE, but the box clips its top-right corner


class SessionRuleTests(SessionTestCase):
    """Session counting rules: the choice of "feet", "centre" or "overlap", and the first-seen-inside switch."""

    def entries(self):
        return self.session.snapshot()["counts"]["entries"]

    def test_defaults_are_the_original_behaviour(self):
        self.assertEqual(self.session.rules(), {"rule": "feet", "countFirstSeenInside": False})
        self.assertEqual(self.session.snapshot()["rules"], {"rule": "feet", "countFirstSeenInside": False})

    def test_feet_rule_does_not_count_feet_below_the_zone(self):
        self.feed(person(1, OUTSIDE_SQUARE))
        self.feed(person(1, FEET_BELOW_ZONE))
        self.assertEqual(self.entries(), 0)

    def test_centre_rule_counts_a_body_that_reaches_the_zone(self):
        self.session = self.make_session(rule="centre")
        self.feed(person(1, OUTSIDE_SQUARE))
        self.feed(person(1, FEET_BELOW_ZONE))
        self.assertEqual(self.entries(), 1)
        self.assertEqual(self.logged_ids(), [1])

    def test_overlap_rule_counts_a_box_that_only_clips_the_zone(self):
        self.session = self.make_session(rule="overlap")
        self.feed(person(1, OUTSIDE_SQUARE))
        self.feed(person(1, CLIPPING_ONLY))
        self.assertEqual(self.entries(), 1)

    def test_the_rule_can_be_changed_while_running(self):
        self.feed(person(1, OUTSIDE_SQUARE))
        self.session.set_rules("centre", False)
        self.feed(person(1, OUTSIDE_SQUARE))  # the first frame after a change only re-learns where everyone is
        self.feed(person(1, FEET_BELOW_ZONE))
        self.assertEqual(self.entries(), 1)
        self.assertEqual(self.session.rules()["rule"], "centre")

    def test_an_unknown_rule_is_refused_and_nothing_changes(self):
        with self.assertRaises(RuntimeError) as context:
            self.session.set_rules("head", True)
        self.assertIn("COUNT_RULE", str(context.exception))
        self.assertEqual(self.session.rules(), {"rule": "feet", "countFirstSeenInside": False})

    def test_the_switch_must_be_true_or_false(self):
        with self.assertRaises(RuntimeError):
            self.session.set_rules("feet", "yes")
        self.assertEqual(self.session.rules()["countFirstSeenInside"], False)

    def test_a_new_session_with_a_bad_rule_is_refused(self):
        with self.assertRaises(RuntimeError):
            self.make_session(rule="head")

    def test_switch_on_counts_a_person_who_appears_inside(self):
        self.session = self.make_session(count_first_seen_inside=True)
        self.feed(person(1, INSIDE_SQUARE))
        self.assertEqual(self.entries(), 1)
        self.assertEqual(self.logged_ids(), [1])

    def test_switch_off_does_not_count_a_person_who_appears_inside(self):
        self.feed(person(1, INSIDE_SQUARE))
        self.assertEqual(self.entries(), 0)

    def test_changing_the_rule_does_not_count_people_who_are_already_inside(self):
        self.session = self.make_session(count_first_seen_inside=True)
        self.feed(person(1, INSIDE_SQUARE))
        self.assertEqual(self.entries(), 1)
        self.session.set_rules("feet", True)
        self.feed(person(1, INSIDE_SQUARE))
        self.assertEqual(self.entries(), 1)

    def test_editing_the_zone_does_not_count_people_who_are_already_inside(self):
        self.session = self.make_session(count_first_seen_inside=True)
        self.feed(person(1, INSIDE_SQUARE))
        self.session.set_zone([(0.5, 0.5), (0.95, 0.5), (0.95, 0.95), (0.5, 0.95)])  # still contains the person
        self.feed(person(1, INSIDE_SQUARE))
        self.assertEqual(self.entries(), 1)

    def test_a_newcomer_after_a_zone_edit_is_counted_with_the_switch_on(self):
        self.session = self.make_session(count_first_seen_inside=True)
        self.session.set_zone([(0.4, 0.4), (0.95, 0.4), (0.95, 0.95), (0.4, 0.95)])
        self.feed(person(1, OUTSIDE_SQUARE))  # the first frame after the edit only re-learns positions
        self.feed(person(2, INSIDE_SQUARE))
        self.assertEqual(self.entries(), 1)

    def test_a_reset_keeps_the_rules_but_counts_people_inside_again(self):
        self.session = self.make_session(rule="centre", count_first_seen_inside=True)
        self.feed(person(1, INSIDE_SQUARE))
        self.session.reset()
        self.assertEqual(self.session.rules(), {"rule": "centre", "countFirstSeenInside": True})
        self.feed(person(1, INSIDE_SQUARE))
        self.assertEqual(self.entries(), 1)  # a new session starts with nothing remembered, so they count again


class ManualCheckTextTests(unittest.TestCase):
    """describe_manual_check(): the wording is the same on the page and in the report."""

    def test_equal_counts_match(self):
        difference, percent, summary = portal.describe_manual_check(system=4, manual=4)
        self.assertEqual((difference, percent), (0, 0))
        self.assertIn("matches", summary)

    def test_system_below_the_manual_count(self):
        difference, percent, summary = portal.describe_manual_check(system=3, manual=5)
        self.assertEqual((difference, percent), (-2, 40))
        self.assertIn("2 fewer", summary)
        self.assertIn("3 vs 5", summary)
        self.assertIn("40%", summary)

    def test_system_above_the_manual_count(self):
        difference, percent, summary = portal.describe_manual_check(system=7, manual=4)
        self.assertEqual((difference, percent), (3, 75))
        self.assertIn("3 more", summary)

    def test_a_manual_count_of_zero_has_no_percentage(self):
        difference, percent, summary = portal.describe_manual_check(system=2, manual=0)
        self.assertEqual((difference, percent), (2, None))
        self.assertNotIn("%", summary)
        self.assertIn("2 more", summary)

    def test_both_zero_match(self):
        self.assertIn("matches", portal.describe_manual_check(system=0, manual=0)[2])


class SessionManualCheckTests(SessionTestCase):
    """Session.set_manual_count(): compares the system's count with a count the user made by eye."""

    def enter(self, track_id):
        self.feed(person(track_id, OUTSIDE_SQUARE))
        self.feed(person(track_id, INSIDE_SQUARE))

    def test_there_is_no_check_until_a_count_is_entered(self):
        self.assertIsNone(self.session.snapshot()["manualCheck"])

    def test_the_check_compares_with_the_system_count(self):
        self.enter(1)
        self.enter(2)
        self.enter(3)
        self.session.set_manual_count(5)
        check = self.session.snapshot()["manualCheck"]
        self.assertEqual((check["manual"], check["system"], check["difference"], check["percent"]), (5, 3, -2, 40))
        self.assertIn("2 fewer", check["summary"])

    def test_the_check_follows_the_live_count(self):
        self.session.set_manual_count(2)
        self.assertEqual(self.session.snapshot()["manualCheck"]["system"], 0)
        self.enter(1)
        self.assertEqual(self.session.snapshot()["manualCheck"]["system"], 1)

    def test_zero_is_a_valid_manual_count(self):
        self.session.set_manual_count(0)
        self.assertEqual(self.session.snapshot()["manualCheck"]["manual"], 0)

    def test_the_count_can_be_cleared(self):
        self.session.set_manual_count(4)
        self.session.set_manual_count(None)
        self.assertIsNone(self.session.snapshot()["manualCheck"])

    def test_bad_values_are_refused_and_the_old_count_stays(self):
        self.session.set_manual_count(4)
        for bad in (-1, 1.5, "3", True, [3], 1_000_001):
            with self.assertRaises(RuntimeError, msg=repr(bad)) as context:
                self.session.set_manual_count(bad)
            self.assertIn("whole number", str(context.exception))
        self.assertEqual(self.session.snapshot()["manualCheck"]["manual"], 4)

    def test_a_reset_clears_the_manual_count(self):
        self.session.set_manual_count(4)
        self.session.reset()
        self.assertIsNone(self.session.snapshot()["manualCheck"])

    def test_the_report_carries_the_check(self):
        self.enter(1)
        self.session.set_manual_count(1)
        self.assertEqual(self.session.report()["manualCheck"]["difference"], 0)


class SessionHeatmapTests(SessionTestCase):
    """Session.heatmap(): where feet have been, as counts on a coarse grid (no footage, no per-person paths)."""

    @property
    def COLS(self):
        return portal.HEATMAP_COLS

    @property
    def ROWS(self):
        return portal.HEATMAP_ROWS

    def cell(self, heat, col, row):
        return heat["cells"][row * heat["cols"] + col]

    def test_an_empty_session_has_an_empty_grid(self):
        heat = self.session.heatmap()
        self.assertEqual((heat["cols"], heat["rows"]), (self.COLS, self.ROWS))
        self.assertEqual(len(heat["cells"]), self.COLS * self.ROWS)
        self.assertEqual((heat["max"], heat["total"]), (0, 0))

    def test_a_persons_feet_are_added_to_the_cell_they_stand_in(self):
        self.feed(person(1, (80, 60)))  # the exact middle of a 160 x 120 picture
        heat = self.session.heatmap()
        self.assertEqual(self.cell(heat, self.COLS // 2, self.ROWS // 2), 1)
        self.assertEqual((heat["max"], heat["total"]), (1, 1))

    def test_standing_still_builds_up_and_everyone_counts(self):
        self.feed(person(1, (80, 60)), person(2, (10, 10)))
        self.feed(person(1, (80, 60)))
        heat = self.session.heatmap()
        self.assertEqual(self.cell(heat, self.COLS // 2, self.ROWS // 2), 2)
        self.assertEqual(self.cell(heat, int(10 / 160 * self.COLS), int(10 / 120 * self.ROWS)), 1)
        self.assertEqual((heat["max"], heat["total"]), (2, 3))

    def test_feet_on_or_beyond_the_picture_edge_stay_in_the_grid(self):
        self.feed(person(1, (160, 120)), person(2, (-5, -3)))
        heat = self.session.heatmap()
        self.assertEqual(self.cell(heat, self.COLS - 1, self.ROWS - 1), 1)
        self.assertEqual(self.cell(heat, 0, 0), 1)

    def test_it_records_feet_whichever_counting_rule_is_chosen(self):
        self.session = self.make_session(rule="centre")
        self.feed(person(1, (80, 60)))
        self.assertEqual(self.cell(self.session.heatmap(), self.COLS // 2, self.ROWS // 2), 1)

    def test_a_reset_clears_it(self):
        self.feed(person(1, (80, 60)))
        self.session.reset()
        self.assertEqual(self.session.heatmap()["total"], 0)

    def test_it_is_not_part_of_the_polled_state(self):
        self.assertNotIn("heatmap", self.session.snapshot())

    def test_it_can_be_sent_as_json(self):
        self.feed(person(1, (80, 60)))
        self.assertEqual(json.loads(json.dumps(self.session.heatmap()))["total"], 1)


class InsightReportTests(SessionTestCase):
    """The printable report states the counting rule, the manual check and the foot heat-map."""

    def page(self, session=None):
        session = session or self.session
        return portal.render_report(session.report(), "clip.mp4", {"w": 160, "h": 120})

    def test_the_default_rule_is_described_as_feet(self):
        page = self.page()
        self.assertIn("feet", page)
        self.assertNotIn("first seen already inside", page)

    def test_the_other_rules_are_described(self):
        self.assertIn("middle of their box", self.page(self.make_session(rule="centre")))
        self.assertIn("any part of their box", self.page(self.make_session(rule="overlap")))

    def test_the_first_seen_switch_is_described_when_on(self):
        self.assertIn("first seen already inside", self.page(self.make_session(count_first_seen_inside=True)))

    def test_no_manual_section_without_a_manual_count(self):
        self.assertNotIn("Check against a manual count", self.page())

    def test_the_manual_check_is_reported_with_a_caution(self):
        self.feed(person(1, OUTSIDE_SQUARE))
        self.feed(person(1, INSIDE_SQUARE))
        self.session.set_manual_count(3)
        page = self.page()
        self.assertIn("Check against a manual count", page)
        self.assertIn("2 fewer", page)
        self.assertIn("cancel", page)  # equal totals can hide offsetting mistakes

    def test_the_heatmap_section_says_so_when_nobody_was_seen(self):
        page = self.page()
        self.assertIn("Where people walked", page)
        self.assertIn("No one has been seen yet", page)
        self.assertNotIn('aria-label="Foot heat-map', page)

    def test_the_heatmap_is_drawn_once_someone_has_been_seen(self):
        self.feed(person(1, (80, 60)))
        page = self.page()
        self.assertIn('aria-label="Foot heat-map', page)
        self.assertIn("<rect", page)


class InsightArgumentTests(unittest.TestCase):
    """The portal's --rule and --count-first-seen command-line options."""

    def test_defaults_follow_the_settings_in_main(self):
        args = portal.parse_args([])
        self.assertEqual((args.rule, args.count_first_seen), (main.COUNT_RULE, main.COUNT_FIRST_SEEN_INSIDE))

    def test_options_are_read(self):
        args = portal.parse_args(["--rule", "overlap", "--count-first-seen"])
        self.assertEqual((args.rule, args.count_first_seen), ("overlap", True))

    def test_an_unknown_rule_stops_with_a_message(self):
        with redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit):
            portal.parse_args(["--rule", "head"])
        self.assertIn("feet", errors.getvalue())


class HttpInsightsTests(unittest.TestCase):
    """The HTTP interface for the three insight features."""

    def test_rules_can_be_changed_and_are_reported_in_the_state(self):
        portal_ = RunningPortal(self, [])
        status, body = portal_.post_json("/api/rules", {"rule": "centre", "countFirstSeenInside": True})
        self.assertEqual(status, 200)
        expected = {"rule": "centre", "countFirstSeenInside": True}
        self.assertEqual(body["rules"], expected)
        self.assertEqual(portal_.get_json("/api/state")[1]["rules"], expected)

    def test_a_field_left_out_keeps_its_value(self):
        portal_ = RunningPortal(self, [])
        portal_.post_json("/api/rules", {"rule": "overlap"})
        status, body = portal_.post_json("/api/rules", {"countFirstSeenInside": True})
        self.assertEqual((status, body["rules"]), (200, {"rule": "overlap", "countFirstSeenInside": True}))

    def test_a_bad_rule_is_refused_with_a_message(self):
        portal_ = RunningPortal(self, [])
        status, body = portal_.post_json("/api/rules", {"rule": "head"})
        self.assertEqual(status, 400)
        self.assertIn("COUNT_RULE", body["error"])
        self.assertEqual(portal_.get_json("/api/state")[1]["rules"]["rule"], "feet")

    def test_a_switch_that_is_not_true_or_false_is_refused(self):
        portal_ = RunningPortal(self, [])
        status, body = portal_.post_json("/api/rules", {"countFirstSeenInside": "yes"})
        self.assertEqual(status, 400)
        self.assertIn("True or False", body["error"])

    def test_manual_count_can_be_set_read_and_cleared(self):
        portal_ = RunningPortal(self, [])
        self.assertIsNone(portal_.get_json("/api/state")[1]["manualCheck"])
        status, body = portal_.post_json("/api/manual", {"count": 4})
        self.assertEqual(status, 200)
        self.assertEqual(body["manualCheck"]["manual"], 4)
        self.assertEqual(portal_.get_json("/api/state")[1]["manualCheck"]["manual"], 4)
        status, body = portal_.post_json("/api/manual", {"count": None})
        self.assertEqual((status, body["manualCheck"]), (200, None))

    def test_a_bad_manual_count_is_refused(self):
        portal_ = RunningPortal(self, [])
        for bad in (-2, 2.5, "many"):
            status, body = portal_.post_json("/api/manual", {"count": bad})
            self.assertEqual(status, 400, bad)
            self.assertIn("whole number", body["error"])

    def test_a_manual_request_without_a_count_is_refused(self):
        portal_ = RunningPortal(self, [])
        status, body = portal_.post_json("/api/manual", {})
        self.assertEqual(status, 400)

    def test_heatmap_endpoint_returns_the_grid(self):
        script = [[person(1, (80, 60))]] * 500
        portal_ = RunningPortal(self, script)
        self.assertTrue(wait_until(lambda: portal_.get_json("/api/heatmap")[1]["total"] > 0))
        status, heat = portal_.get_json("/api/heatmap")
        self.assertEqual(status, 200)
        self.assertEqual(len(heat["cells"]), heat["cols"] * heat["rows"])
        self.assertGreater(heat["max"], 0)

    def test_the_new_posts_refuse_other_websites(self):
        portal_ = RunningPortal(self, [])
        evil = {"Origin": "https://evil.example"}
        self.assertEqual(portal_.post_json("/api/rules", {"rule": "centre"}, headers=evil)[0], 403)
        self.assertEqual(portal_.post_json("/api/manual", {"count": 1}, headers=evil)[0], 403)
        self.assertEqual(portal_.get_json("/api/state")[1]["rules"]["rule"], "feet")
        self.assertIsNone(portal_.get_json("/api/state")[1]["manualCheck"])

    def test_the_report_page_shows_the_new_sections(self):
        portal_ = RunningPortal(self, [])
        portal_.post_json("/api/manual", {"count": 2})
        status, _, page = portal_.request("GET", "/report")
        self.assertEqual(status, 200)
        self.assertIn(b"Check against a manual count", page)
        self.assertIn(b"Where people walked", page)


if __name__ == "__main__":
    unittest.main()
