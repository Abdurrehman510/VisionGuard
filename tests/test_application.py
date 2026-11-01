"""Application-level tests: quit keys, startup errors, resource release and the status panel text.

The OpenCV display calls (imshow, waitKey, getWindowProperty) are replaced so no window opens;
video capture, model files, the CSV and all application logic are real.
Run: python -m unittest discover -s tests
"""
import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main

SAMPLE_VIDEO = ROOT / "inputs" / "videos" / "vtest.mp4"


class ShouldQuitTests(unittest.TestCase):
    """should_quit(): Q, q and ESC stop the program, as does closing the window; no other key does."""
    def quit_for(self, key, window_visible=1.0):
        with mock.patch.object(main.cv2, "waitKey", return_value=key), \
                mock.patch.object(main.cv2, "getWindowProperty", return_value=window_visible):
            return main.should_quit()

    def test_q_uppercase_q_and_escape_quit(self):
        for key in (ord("q"), ord("Q"), 27):
            with self.subTest(key=key):
                self.assertTrue(self.quit_for(key))

    def test_other_keys_and_no_key_do_not_quit(self):
        for key in (-1, 255, ord("a"), ord(" ")):
            with self.subTest(key=key):
                self.assertFalse(self.quit_for(key))

    def test_closing_the_window_quits(self):
        self.assertTrue(self.quit_for(-1, window_visible=0.0))


class TextColorTests(unittest.TestCase):
    """Label text must stay readable on any box colour (colours are BGR)."""

    def test_light_backgrounds_get_dark_text(self):
        for background in ((0, 255, 0), (240, 176, 95), (26, 107, 255), (255, 255, 255)):
            with self.subTest(background=background):
                self.assertEqual(main.text_color_for(background), (0, 0, 0))

    def test_dark_backgrounds_get_white_text(self):
        for background in ((0, 0, 255), (43, 32, 20), (0, 0, 0)):
            with self.subTest(background=background):
                self.assertEqual(main.text_color_for(background), (255, 255, 255))

    def test_original_colours_keep_their_original_text(self):
        self.assertEqual(main.text_color_for(main.BOX_COLOR), (0, 0, 0))        # green box, black text
        self.assertEqual(main.text_color_for(main.ZONE_COLOR), (255, 255, 255))  # red box, white text


class StatusLinesTests(unittest.TestCase):
    """status_lines(): the text of the on-screen status panel (name, entries, detections, zone state)."""
    def test_shows_project_name_monitoring_status_entries_and_detection(self):
        self.assertEqual(
            main.status_lines(entry_count=3, people_count=5, inside_count=2),
            ["VisionGuard | MONITORING", "Entries: 3", "Detected: 5 people", "Zone: OCCUPIED (2 inside)"],
        )

    def test_clear_zone_and_singular_wording(self):
        self.assertEqual(
            main.status_lines(entry_count=0, people_count=1, inside_count=0),
            ["VisionGuard | MONITORING", "Entries: 0", "Detected: 1 person", "Zone: CLEAR"],
        )

    def test_no_people_detected(self):
        self.assertEqual(main.status_lines(0, 0, 0)[2], "Detected: 0 people")


class MainErrorHandlingTests(unittest.TestCase):
    """main() must report problems clearly, return 1, and always release the video source."""

    def setUp(self):
        self.captures = []
        real_open = main.open_source

        def recording_open(source):
            capture = real_open(source)
            self.captures.append(capture)
            return capture

        patches = [
            mock.patch.object(main, "open_source", side_effect=recording_open),
            mock.patch.object(main.cv2, "imshow"),
            mock.patch.object(main.cv2, "waitKey", return_value=-1),
            mock.patch.object(main.cv2, "getWindowProperty", return_value=1.0),
            mock.patch.object(main.cv2, "destroyAllWindows"),
        ]
        self.mocks = [patch.start() for patch in patches]
        for patch in patches:
            self.addCleanup(patch.stop)
        self.destroy_windows = self.mocks[-1]
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        csv_patch = mock.patch.object(main, "EVENTS_CSV", Path(self.tmp.name) / "outputs" / "events.csv")
        csv_patch.start()
        self.addCleanup(csv_patch.stop)
        self.zone_file = Path(self.tmp.name) / "zone.json"  # does not exist unless a test saves one
        zone_patch = mock.patch.object(main, "ZONE_FILE", self.zone_file)
        zone_patch.start()
        self.addCleanup(zone_patch.stop)

    def run_main(self, *args):
        output = io.StringIO()
        with mock.patch.object(sys, "argv", ["main.py", *args]), redirect_stdout(output):
            code = main.main()
        return code, output.getvalue()

    def assert_all_captures_released(self):
        self.assertTrue(self.captures, "the video source was never opened")
        self.assertTrue(all(not capture.isOpened() for capture in self.captures))

    def test_missing_video_file_gives_a_clear_error_before_loading_the_model(self):
        with mock.patch.object(main, "load_model") as load_model:
            code, output = self.run_main(str(ROOT / "inputs" / "videos" / "does_not_exist.mp4"))
        self.assertEqual(code, 1)
        self.assertIn("Video file not found", output)
        load_model.assert_not_called()

    def test_file_that_is_not_a_video_gives_a_clear_error(self):
        code, output = self.run_main(str(ROOT / "requirements.txt"))
        self.assertEqual(code, 1)
        self.assertIn("Could not open video file", output)

    def test_unavailable_webcam_gives_a_clear_error_before_loading_the_model(self):
        with mock.patch.object(main, "load_model") as load_model:
            code, output = self.run_main("9")
        self.assertEqual(code, 1)
        self.assertIn("Webcam 9 is unavailable", output)
        load_model.assert_not_called()

    def test_invalid_zone_configuration_is_reported_and_nothing_is_opened(self):
        with mock.patch.object(main, "ZONE_POLYGON", [(100, 100), (300, 100), (300, 300)]):
            code, output = self.run_main(str(SAMPLE_VIDEO))
        self.assertEqual(code, 1)
        self.assertIn("ZONE_POLYGON", output)
        self.assertEqual(self.captures, [])

    def test_zone_saved_by_the_portal_is_used_instead_of_the_default(self):
        saved = [(0.2, 0.2), (0.8, 0.2), (0.5, 0.9)]
        main.save_zone(self.zone_file, saved)
        with mock.patch.object(main, "load_model", return_value=object()), \
                mock.patch.object(main, "process_video") as process_video:
            code, _ = self.run_main(str(SAMPLE_VIDEO))
        self.assertEqual(code, 0)
        self.assertEqual(process_video.call_args.kwargs["zone_fractions"], saved)

    def test_default_zone_is_used_when_nothing_is_saved(self):
        with mock.patch.object(main, "load_model", return_value=object()), \
                mock.patch.object(main, "process_video") as process_video:
            self.run_main(str(SAMPLE_VIDEO))
        self.assertEqual(process_video.call_args.kwargs["zone_fractions"], main.ZONE_POLYGON)

    def test_unusable_saved_zone_falls_back_to_the_default_and_the_app_still_runs(self):
        self.zone_file.write_text('{"points": [[100, 100], [300, 100], [300, 300]]}')
        with mock.patch.object(main, "load_model", return_value=object()), \
                mock.patch.object(main, "process_video") as process_video:
            code, output = self.run_main(str(SAMPLE_VIDEO))
        self.assertEqual(code, 0)
        self.assertIn("Ignoring the saved zone", output)
        self.assertEqual(process_video.call_args.kwargs["zone_fractions"], main.ZONE_POLYGON)

    def test_unloadable_model_is_reported_and_the_video_is_released(self):
        broken_model = Path(self.tmp.name) / "broken.pt"
        broken_model.write_bytes(b"this is not a model")
        with mock.patch.object(main, "MODEL_PATH", broken_model):
            code, output = self.run_main(str(SAMPLE_VIDEO))
        self.assertEqual(code, 1)
        self.assertIn("Could not load YOLO model", output)
        self.assert_all_captures_released()

    def test_error_during_processing_is_reported_and_resources_are_released(self):
        with mock.patch.object(main, "load_model", return_value=object()), \
                mock.patch.object(main, "track_people", side_effect=ValueError("boom")):
            code, output = self.run_main(str(SAMPLE_VIDEO))
        self.assertEqual(code, 1)
        self.assertIn("unexpected error", output)
        self.assertIn("boom", output)
        self.assert_all_captures_released()
        self.destroy_windows.assert_called()

    def test_missing_package_at_run_time_names_the_fix(self):
        missing = ModuleNotFoundError("No module named 'lap'", name="lap")
        with mock.patch.object(main, "load_model", return_value=object()), \
                mock.patch.object(main, "track_people", side_effect=missing):
            code, output = self.run_main(str(SAMPLE_VIDEO))
        self.assertEqual(code, 1)
        self.assertIn("lap", output)
        self.assertIn("pip install -r requirements.txt", output)
        self.assert_all_captures_released()

    def test_ctrl_c_stops_cleanly_and_releases_resources(self):
        with mock.patch.object(main, "load_model", return_value=object()), \
                mock.patch.object(main, "track_people", side_effect=KeyboardInterrupt):
            code, output = self.run_main(str(SAMPLE_VIDEO))
        self.assertEqual(code, 0)
        self.assertIn("Interrupted", output)
        self.assert_all_captures_released()
        self.destroy_windows.assert_called()

    def test_quit_key_stops_normally_and_releases_resources(self):
        self.mocks[2].return_value = ord("q")  # waitKey
        with mock.patch.object(main, "load_model", return_value=object()), \
                mock.patch.object(main, "track_people", return_value=[]):
            code, output = self.run_main(str(SAMPLE_VIDEO))
        self.assertEqual(code, 0)
        self.assertIn("Stopped by user", output)
        self.assertIn("Entries counted this run: 0", output)
        self.assert_all_captures_released()
        self.destroy_windows.assert_called()


class ProcessVideoTests(unittest.TestCase):
    """process_video(): a source that opens but never returns a frame is reported as an error."""
    def test_source_that_opens_but_returns_no_frames_is_reported(self):
        class SilentCapture:
            def read(self):
                return False, None

        with self.assertRaises(RuntimeError) as context:
            main.process_video(object(), SilentCapture(), "camera-or-file")
        self.assertIn("no frames", str(context.exception))


class MissingDependencyTests(unittest.TestCase):
    """A missing package gives install instructions instead of a traceback."""
    def test_missing_opencv_prints_install_instructions_instead_of_a_traceback(self):
        code = (
            "import runpy, sys\n"
            "sys.modules['cv2'] = None  # makes 'import cv2' fail as if it were not installed\n"
            f"runpy.run_path(r'{ROOT / 'main.py'}', run_name='__not_main__')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing dependency", result.stderr)
        self.assertIn("pip install -r requirements.txt", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
