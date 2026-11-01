"""Tests for playing a video at a chosen speed: the pacing logic, speed validation, frame skipping and
the command-line option. Run: python -m unittest tests.test_speed
"""
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main


class FakeTime:
    """A clock the test controls; sleeping just moves it forward."""

    def __init__(self, now=100.0):
        self.start = self.now = now

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class PacerTests(unittest.TestCase):
    """The Pacer says how many frames to skip (when behind) or how long to wait (when ahead)."""

    def setUp(self):
        self.time = FakeTime()

    def pacer(self, fps=60, speed=1.0, start_frames=0):
        return main.Pacer(fps, speed, start_frames, clock=self.time.clock)

    def test_when_detection_is_slow_the_skipped_frames_keep_the_video_on_time(self):
        pacer = self.pacer(fps=60, speed=1.0)
        self.time.now += 0.5                      # one frame took half a second to process
        self.assertEqual(pacer.frames_to_skip(frames_consumed=1), 29)   # 0.5 s of video = 30 frames, 1 already used
        self.assertEqual(pacer.wait_time(frames_consumed=1), 0.0)

    def test_double_speed_skips_twice_as_many_frames(self):
        pacer = self.pacer(fps=60, speed=2.0)
        self.time.now += 0.5
        self.assertEqual(pacer.frames_to_skip(frames_consumed=1), 59)

    def test_half_speed_skips_half_as_many_frames(self):
        pacer = self.pacer(fps=60, speed=0.5)
        self.time.now += 0.5
        self.assertEqual(pacer.frames_to_skip(frames_consumed=1), 14)

    def test_when_ahead_of_the_clock_it_waits_instead_of_skipping(self):
        pacer = self.pacer(fps=10, speed=1.0)
        self.time.now += 0.01                     # three frames used in 0.01 s: far ahead of a 10 fps video
        self.assertEqual(pacer.frames_to_skip(frames_consumed=3), 0)
        self.assertAlmostEqual(pacer.wait_time(frames_consumed=3), 0.29, places=6)

    def test_exactly_on_time_neither_skips_nor_waits(self):
        pacer = self.pacer(fps=10, speed=1.0)
        self.time.now += 0.3
        self.assertEqual((pacer.frames_to_skip(3), pacer.wait_time(3)), (0, 0.0))

    def test_speed_zero_means_every_frame_and_never_skips_or_waits(self):
        pacer = self.pacer(fps=60, speed=0)
        self.time.now += 10
        self.assertEqual((pacer.frames_to_skip(1), pacer.wait_time(1)), (0, 0.0))

    def test_counting_starts_from_where_the_pacer_was_created(self):
        pacer = self.pacer(fps=60, speed=1.0, start_frames=600)   # e.g. the speed was changed 10 s into the video
        self.time.now += 0.5
        self.assertEqual(pacer.frames_to_skip(frames_consumed=601), 29)


class SkipFramesTests(unittest.TestCase):
    """skip_frames(): drops the requested number of frames and stops cleanly at the end of the video."""
    class Capture:
        def __init__(self, frames):
            self.frames = frames

        def grab(self):
            if self.frames <= 0:
                return False
            self.frames -= 1
            return True

    def test_skips_the_requested_number_of_frames(self):
        capture = self.Capture(100)
        self.assertEqual(main.skip_frames(capture, 29), 29)
        self.assertEqual(capture.frames, 71)

    def test_stops_early_at_the_end_of_the_video(self):
        capture = self.Capture(5)
        self.assertEqual(main.skip_frames(capture, 29), 5)

    def test_skipping_nothing_does_nothing(self):
        capture = self.Capture(5)
        self.assertEqual(main.skip_frames(capture, 0), 0)
        self.assertEqual(capture.frames, 5)


class VideoInfoTests(unittest.TestCase):
    """video_info(): frame rate and length of a video, with None for anything the file cannot report."""
    class Capture:
        def __init__(self, fps, count):
            self.values = {main.cv2.CAP_PROP_FPS: fps, main.cv2.CAP_PROP_FRAME_COUNT: count}

        def get(self, prop):
            return self.values[prop]

    def test_reads_frame_rate_and_length(self):
        self.assertEqual(main.video_info(self.Capture(59.94, 4548)), (59.94, 4548))

    def test_unknown_values_become_none(self):
        self.assertEqual(main.video_info(self.Capture(0.0, 0.0)), (None, None))

    def test_a_capture_that_cannot_report_anything_gives_none(self):
        self.assertEqual(main.video_info(object()), (None, None))


class ValidateSpeedTests(unittest.TestCase):
    """validate_speed(): 0 and sensible speeds pass; anything else gives a clear message."""
    def test_zero_and_sensible_speeds_are_accepted(self):
        for speed in (0, 0.5, 1, 2, 4, 8, 16, 2.5):
            with self.subTest(speed=speed):
                main.validate_speed(speed)

    def test_bad_speeds_are_rejected_with_a_clear_message(self):
        for bad in (-1, 16.5, 100, "fast", None, True, float("nan"), float("inf")):
            with self.subTest(speed=bad), self.assertRaises(RuntimeError) as context:
                main.validate_speed(bad)
            self.assertIn("speed", str(context.exception).lower())


class ProcessVideoSpeedTests(unittest.TestCase):
    """The OpenCV window's video loop with a chosen speed. Time is fake, so a 10-second video 'plays' instantly."""

    class VideoCapture:
        def __init__(self, frames, fps):
            self.remaining, self.fps = frames, fps

        def read(self):
            if self.remaining <= 0:
                return False, None
            self.remaining -= 1
            return True, np.zeros((120, 160, 3), dtype=np.uint8)

        def grab(self):
            if self.remaining <= 0:
                return False
            self.remaining -= 1
            return True

        def get(self, prop):
            return float(self.fps) if prop == main.cv2.CAP_PROP_FPS else 0.0

        def release(self):
            pass

    def play(self, frames, fps, detect_seconds, speed):
        fake, calls, real_pacer = FakeTime(), [], main.Pacer

        def slow_detector(model, frame):
            calls.append(1)
            fake.sleep(detect_seconds)
            return []

        with mock.patch.object(main, "Pacer", lambda video_fps, video_speed: real_pacer(video_fps, video_speed, clock=fake.clock)), \
                mock.patch.object(main.time, "sleep", fake.sleep), \
                mock.patch.object(main, "track_people", slow_detector), \
                mock.patch.object(main.cv2, "imshow"), mock.patch.object(main.cv2, "waitKey", return_value=-1), \
                mock.patch.object(main.cv2, "getWindowProperty", return_value=1.0), \
                redirect_stdout(io.StringIO()):
            main.process_video(object(), self.VideoCapture(frames, fps), "video.mp4", speed=speed)
        return len(calls), fake.now - fake.start

    def test_default_uses_every_frame(self):
        calls, _ = self.play(frames=600, fps=60, detect_seconds=0.5, speed=0)
        self.assertEqual(calls, 600)

    def test_normal_speed_skips_frames_the_detector_cannot_keep_up_with(self):
        calls, _ = self.play(frames=600, fps=60, detect_seconds=0.5, speed=1.0)
        self.assertTrue(18 <= calls <= 23, f"detector ran {calls} times")

    def test_normal_speed_waits_when_the_detector_is_faster_than_the_video(self):
        calls, elapsed = self.play(frames=20, fps=10, detect_seconds=0.0, speed=1.0)
        self.assertEqual(calls, 20)
        self.assertTrue(1.7 <= elapsed <= 2.3, f"took {elapsed:.2f} s for a 2 s video")

    def test_speed_is_ignored_for_a_webcam(self):
        with mock.patch.object(main.time, "sleep") as sleep, mock.patch.object(main, "track_people", return_value=[]), \
                mock.patch.object(main.cv2, "imshow"), mock.patch.object(main.cv2, "waitKey", return_value=-1), \
                mock.patch.object(main.cv2, "getWindowProperty", return_value=1.0), redirect_stdout(io.StringIO()):
            main.process_video(object(), self.VideoCapture(5, 30), 0, speed=2.0)   # source 0 = webcam
        sleep.assert_not_called()


class CommandLineSpeedTests(unittest.TestCase):
    """--speed on the command line: parsing, how it combines with the video argument, and main() using it."""
    def test_default_speed_is_every_frame(self):
        self.assertEqual(main.get_speed(["main.py", "video.mp4"]), 0.0)

    def test_speed_option_is_read(self):
        self.assertEqual(main.get_speed(["main.py", "video.mp4", "--speed", "2"]), 2.0)
        self.assertEqual(main.get_speed(["main.py", "--speed", "0.5", "video.mp4"]), 0.5)

    def test_speed_option_does_not_disturb_the_video_argument(self):
        self.assertEqual(main.get_source(["main.py", "video.mp4", "--speed", "2"]), "video.mp4")
        self.assertEqual(main.get_source(["main.py", "--speed", "2", "video.mp4"]), "video.mp4")
        self.assertEqual(main.get_source(["main.py", "--speed", "2"]), main.VIDEO_SOURCE)
        self.assertEqual(main.get_source(["main.py", "1", "--speed", "2"]), 1)

    def test_speed_without_a_number_or_with_text_is_a_clear_error(self):
        for argv in (["main.py", "v.mp4", "--speed"], ["main.py", "v.mp4", "--speed", "quick"]):
            with self.subTest(argv=argv), self.assertRaises(RuntimeError) as context:
                main.get_speed(argv)
            self.assertIn("--speed", str(context.exception))

    def test_main_passes_the_speed_to_the_video_loop(self):
        with mock.patch.object(main, "load_model", return_value=object()), \
                mock.patch.object(main, "process_video") as process_video, \
                mock.patch.object(main.cv2, "destroyAllWindows"), \
                mock.patch.object(sys, "argv", ["main.py", str(ROOT / "inputs" / "videos" / "vtest.mp4"), "--speed", "2"]):
            code = main.main()
        self.assertEqual(code, 0)
        self.assertEqual(process_video.call_args.kwargs["speed"], 2.0)

    def test_main_refuses_an_invalid_speed_before_opening_anything(self):
        with mock.patch.object(main, "open_source") as open_source, \
                mock.patch.object(sys, "argv", ["main.py", "v.mp4", "--speed", "99"]), \
                mock.patch("builtins.print") as printed:
            code = main.main()
        self.assertEqual(code, 1)
        open_source.assert_not_called()
        self.assertIn("speed", str(printed.call_args_list).lower())


if __name__ == "__main__":
    unittest.main()
