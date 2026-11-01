"""VisionGuard web portal: a live monitoring page on this computer (localhost) with an editable
restricted zone, live numbers and a session report.

Run:  python portal.py                 (default webcam)
      python portal.py video.mp4       (a video file)
Then open http://127.0.0.1:8000 (the browser opens by itself). Uses only the Python standard library
plus the packages the main application already needs.
"""
import argparse
import csv
import html
import io
import json
import sys
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import main

# --- Configuration -----------------------------------------------------------------------------------
UI_FILE = main.BASE_DIR / "portal_ui" / "index.html"  # the single-page web interface served at "/"
DEFAULT_PORT = 8000
MAX_BODY_BYTES = 64 * 1024  # largest POST body accepted; real requests are a few hundred bytes
JPEG_QUALITY = 80           # quality of the frames streamed to the page (smaller = faster, blurrier)
STREAM_BOUNDARY = "frame"   # separator between JPEG frames in the multipart/x-mixed-replace (MJPEG) stream
OUTSIDE_BOX_COLOR = (240, 176, 95)  # BGR: light blue, matches the "in view" series in the page
INSIDE_BOX_COLOR = (26, 107, 255)   # BGR: signage orange, matches the zone and the "inside" series


# --- Session: everything that changes while monitoring ---------------------------------------------

class Session:
    """Counts, history, events and the zone for one monitoring session. Thread-safe."""

    HISTORY_SAMPLES = 180  # one sample per second -> the last 3 minutes
    RECENT_EVENTS = 50     # how many of the newest entries the live page lists
    MINUTES_SHOWN = 10     # how many one-minute bars the "entries per minute" chart has

    def __init__(self, zone_fractions, confirm_frames, csv_path, zone_file, default_zone, clock=time.time):
        """zone_fractions = the zone in use now; default_zone = what "Reset to default" restores;
        clock = time source (a parameter so tests can control time)."""
        # The monitor thread writes (process) while HTTP threads read (snapshot), so every access takes
        # this lock. It is re-entrant (RLock) so a method holding it can safely call another one that needs it.
        self._lock = threading.RLock()
        self._zone = [tuple(point) for point in zone_fractions]
        self._default_zone = [tuple(point) for point in default_zone]
        self._confirm_frames = confirm_frames
        self._csv_path = csv_path
        self._zone_file = zone_file
        self._clock = clock
        self.reset()

    def reset(self):
        """Start a new session: zero the numbers and forget who was where. The zone is kept."""
        with self._lock:
            self._started = self._clock()
            self._zone_states = {}  # track ID -> (confirmed state, frames in the opposite state); see main.find_entries
            self._entry_count = 0
            self._events = []       # every entry this session: {"n": running number, "id": track ID, "ts": time}
            self._history = deque(maxlen=self.HISTORY_SAMPLES)  # old samples fall off the left automatically
            self._last_sample_second = -1
            self._in_view = self._inside = 0
            self._unique_ids = set()  # every track ID seen (tracker labels, not distinct people)
            self._peak_in_view = self._peak_inside = 0

    def process(self, people, frame_size):
        """Update the session with one frame's people. Returns (inside_by_id, entered_ids)."""
        width, height = frame_size
        with self._lock:
            zone_pixels = main.scale_zone(self._zone, width, height)
            # The counting rules live in main.py, shared with the desktop app, so the portal can never
            # count differently (Scope Rule 6). This also appends each entry to outputs/events.csv.
            inside_by_id, entered_ids = main.update_monitoring(
                people, zone_pixels, self._zone_states, self._csv_path, self._confirm_frames)
            now = self._clock()
            for track_id in entered_ids:
                self._entry_count += 1
                self._events.append({"n": self._entry_count, "id": track_id, "ts": now})
            self._unique_ids.update(inside_by_id)
            self._in_view = len(people)
            self._inside = sum(inside_by_id.values())
            self._peak_in_view = max(self._peak_in_view, self._in_view)
            self._peak_inside = max(self._peak_inside, self._inside)
            self._sample_history(now)
        return inside_by_id, entered_ids

    def _sample_history(self, now):
        """Record people-in-view / people-inside once per elapsed second for the activity chart."""
        second = int(now - self._started)
        if second > self._last_sample_second:  # frames arrive many times a second; keep one sample per second
            self._history.append({"t": second, "inView": self._in_view, "inside": self._inside})
            self._last_sample_second = second

    def set_zone(self, points):
        """Validate, save and apply a new zone shape. Raises RuntimeError (zone unchanged) if unusable."""
        main.save_zone(self._zone_file, points)  # validates first, then writes
        saved = main.load_zone(self._zone_file, self._default_zone)  # read back, so we use exactly what was stored (rounded)
        with self._lock:
            self._zone = saved
            self._zone_states = {}  # a person's inside/outside state means something new now

    def reset_zone(self):
        """Delete the saved zone file and go back to the default zone."""
        main.clear_zone(self._zone_file)
        with self._lock:
            self._zone = list(self._default_zone)
            self._zone_states = {}  # same reason as in set_zone: old inside/outside states no longer apply

    def _zone_info(self):
        """The zone as the page needs it: its points, whether it is the default, and its % of the picture."""
        points = [[round(x, 4), round(y, 4)] for x, y in self._zone]
        default = [[round(x, 4), round(y, 4)] for x, y in self._default_zone]
        # Points are 0-1 fractions, so the polygon area is already the fraction of the picture; x100 = percent.
        area = abs(main.cv2.contourArea(main.np.array(self._zone, dtype=main.np.float32))) * 100
        return {"points": points, "isDefault": points == default, "areaPct": round(float(area), 1)}

    @staticmethod
    def _describe(event):
        """Turn a stored event into the display form (short time and full timestamp) used by the page and report."""
        moment = datetime.fromtimestamp(event["ts"])
        return {"n": event["n"], "id": event["id"], "ts": event["ts"],
                "time": moment.strftime("%H:%M:%S"), "timestamp": moment.strftime("%Y-%m-%d %H:%M:%S")}

    def _entries_per_minute(self, now):
        """Entries in each of the last MINUTES_SHOWN minutes, oldest first, ready to draw as bars."""
        buckets = [0] * self.MINUTES_SHOWN  # buckets[0] = the most recent minute, buckets[1] = the one before, ...
        for event in self._events:
            age_minutes = int((now - event["ts"]) // 60)
            if 0 <= age_minutes < self.MINUTES_SHOWN:
                buckets[age_minutes] += 1
        # Reverse the order (oldest first) so the chart reads left to right in time.
        return [{"label": "now" if age == 0 else f"-{age}m", "count": buckets[age]}
                for age in range(self.MINUTES_SHOWN - 1, -1, -1)]

    def _counts(self):
        """The headline numbers shared by the live page and the report."""
        return {"entries": self._entry_count, "inside": self._inside, "inView": self._in_view,
                "uniqueIds": len(self._unique_ids), "peakInView": self._peak_in_view,
                "peakInside": self._peak_inside}

    def snapshot(self):
        """Everything the live page needs, as plain JSON-ready data."""
        with self._lock:
            now = self._clock()
            return {
                "counts": self._counts(),
                "uptime": round(now - self._started, 1),
                "zone": self._zone_info(),
                "history": list(self._history),
                "perMinute": self._entries_per_minute(now),
                "events": [self._describe(e) for e in reversed(self._events[-self.RECENT_EVENTS:])],  # newest first
                "confirmFrames": self._confirm_frames,
            }

    def report(self):
        """The whole session summarised for the report page and the CSV download."""
        with self._lock:
            now = self._clock()
            duration = round(now - self._started)
            events = [self._describe(e) for e in self._events]
            per_minute = round(self._entry_count / (duration / 60), 2) if duration >= 1 else 0.0  # avoid dividing by zero right after a reset
            return {
                "started": datetime.fromtimestamp(self._started).strftime("%Y-%m-%d %H:%M:%S"),
                "generated": datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
                "durationSeconds": duration,
                "counts": self._counts(),
                "entriesPerMinute": per_minute,
                "firstEntry": events[0]["time"] if events else None,
                "lastEntry": events[-1]["time"] if events else None,
                "events": events,
                "zone": self._zone_info(),
                "confirmFrames": self._confirm_frames,
                "perMinute": self._entries_per_minute(now),
            }


# --- Monitor: the thread that reads the camera and runs detection -----------------------------------

class Monitor:
    """Reads frames, runs the detector, updates the Session and publishes JPEG frames for the page."""

    def __init__(self, session, open_source, make_detector, source_label, is_webcam,
                 speed=0.0, clock=time.monotonic, sleep=None):
        """open_source and make_detector are factories (called on the monitor thread, and again on every
        restart); clock and sleep are parameters so tests can replace real time. Nothing runs until start()."""
        self._session = session
        self._open_source = open_source
        self._make_detector = make_detector
        self._source_label = source_label
        self._is_webcam = is_webcam
        self._speed = float(speed)  # video files only: 0 = every frame, 1 = normal speed
        self._clock = clock
        self._stop = threading.Event()
        # Sleeping on the stop event (instead of time.sleep) lets stop() interrupt a long pacing wait at once.
        self._sleep = sleep or self._stop.wait
        self._video_fps = self._video_frames = None
        self._video_position = 0.0
        # Events are thread-safe flags that the HTTP threads set and the monitor thread polls.
        self._restart = threading.Event()
        self._paused = threading.Event()
        self._lock = threading.Lock()              # guards the status / frame-size / fps / video fields
        self._frame_cond = threading.Condition()   # lets stream threads sleep until a new JPEG is published
        self._status, self._message = "starting", ""
        self._frame_size = None
        self._fps = 0.0
        self._frame_id, self._jpeg = 0, None       # newest published frame: a counter and its JPEG bytes
        self._thread = None

    # control (called from the HTTP threads and from cli())
    def start(self):
        """Start the monitoring thread. It is a daemon thread, so it can never keep the program alive."""
        self._thread = threading.Thread(target=self._run, name="visionguard-monitor", daemon=True)
        self._thread.start()

    def stop(self):
        """Ask the thread to finish, wake anything waiting for a frame, and wait up to 10 s for it to end."""
        self._stop.set()
        self._restart.set()  # also releases the thread if it is idling after an error or the end of the video
        with self._frame_cond:
            self._frame_cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def pause(self, paused):
        """Pause (True) or resume (False) analysis. The source stays open while paused."""
        (self._paused.set if paused else self._paused.clear)()

    def restart(self):
        """Reopen the source and start a new session (used by Retry / Play again / Restart)."""
        self._restart.set()

    def set_speed(self, speed):
        """Play a video file at `speed` times normal speed (0 = every frame). Takes effect at once."""
        if self._is_webcam:
            raise RuntimeError("Playback speed applies to video files only; the webcam is always live.")
        main.validate_speed(speed)
        with self._lock:
            self._speed = float(speed)

    @property
    def stopped(self):
        """True once stop() has been called."""
        return self._stop.is_set()

    # reading state
    def state(self):
        """Everything the page polls for: the session's numbers plus the monitor's status, FPS and video progress."""
        with self._lock:
            status, message, fps = self._status, self._message, round(self._fps, 1)
            frame = {"w": self._frame_size[0], "h": self._frame_size[1]} if self._frame_size else None
            video = None
            if self._video_fps:
                duration = round(self._video_frames / self._video_fps, 1) if self._video_frames else None
                video = {"position": round(self._video_position, 1), "duration": duration, "speed": self._speed,
                         "fps": round(self._video_fps, 2)}
        state = self._session.snapshot()
        state.update(status=status, message=message, source=self._source_label,
                     isWebcam=self._is_webcam, frame=frame, fps=fps, video=video)
        return state

    def source_info(self):
        """(source name, frame size or None) - used by the printable report."""
        with self._lock:
            return self._source_label, ({"w": self._frame_size[0], "h": self._frame_size[1]} if self._frame_size else None)

    def wait_for_frame(self, last_id, timeout=2.0):
        """Return (frame_id, jpeg) of the newest frame, waiting up to `timeout` s for one newer than last_id."""
        with self._frame_cond:
            if self._frame_id <= last_id and not self._stop.is_set():
                self._frame_cond.wait(timeout)
            return self._frame_id, self._jpeg

    # the thread
    def _set_status(self, status, message=""):
        """Publish the monitor's state ("starting", "monitoring", "paused", "error" or "ended") and a message for the page."""
        with self._lock:
            self._status, self._message = status, message

    def _run(self):
        """Thread body: open the source, run the loop, and when it ends (error, end of video, restart)
        wait for a restart or stop. Each pass starts a fresh session."""
        while not self._stop.is_set():
            self._restart.clear()
            self._set_status("starting", "Starting the camera and loading the model...")
            capture = None
            try:
                detect = self._make_detector()
                capture = self._open_source()
                self._loop(capture, detect)
            except RuntimeError as error:
                self._set_status("error", str(error))
            except Exception as error:  # keep the portal alive and tell the user what happened
                self._set_status("error", f"Stopped because of an unexpected error: {error}")
            finally:  # always give the camera / file back, whatever happened
                if capture is not None:
                    capture.release()
            # Idle here until the user presses Retry/Restart (or the portal is closing).
            while not (self._stop.is_set() or self._restart.is_set()):
                self._restart.wait(0.2)
            if self._restart.is_set() and not self._stop.is_set():
                self._session.reset()  # a restart means a clean slate: zero counts, no remembered positions

    def _loop(self, capture, detect):
        """Read, detect, count, draw and publish frames until the source ends, an error occurs, or stop/restart is requested."""
        last_frame_time = None
        fps, total_frames = (None, None) if self._is_webcam else main.video_info(capture)
        with self._lock:
            self._video_fps, self._video_frames, self._video_position = fps, total_frames, 0.0
        frames_consumed, pacer = 0, None  # frames taken from the video so far, including skipped ones
        while not (self._stop.is_set() or self._restart.is_set()):
            if self._paused.is_set():
                self._set_status("paused", "Monitoring is paused.")
                last_frame_time = pacer = None  # restart the pacing clock after a pause, so the video does not race ahead
                time.sleep(0.05)  # short nap, so Resume, Restart and Stop still react quickly
                continue
            self._set_status("monitoring")
            grabbed, frame = capture.read()
            if grabbed:
                frames_consumed += 1
            if not grabbed:  # a webcam that stops is a problem; a video that stops has simply ended
                if self._is_webcam:
                    self._set_status("error", "The camera stopped sending frames. Check that it is connected "
                                              "and not used by another app, then press Retry.")
                else:
                    self._set_status("ended", "The video has ended. Press Restart to play it again.")
                return
            height, width = frame.shape[:2]
            people = detect(frame)
            inside_by_id, _ = self._session.process(people, (width, height))
            # Only the people are drawn here; the zone itself is drawn by the page on a canvas over the video,
            # which is what makes it editable.
            main.draw_people(frame, people, inside_by_id, outside_color=OUTSIDE_BOX_COLOR,
                             inside_color=INSIDE_BOX_COLOR, compact=True)
            encoded, buffer = main.cv2.imencode(".jpg", frame, [int(main.cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not encoded:
                raise RuntimeError("Could not encode a video frame.")
            now = time.time()
            with self._lock:
                self._frame_size = (width, height)
                if last_frame_time is not None and now > last_frame_time:
                    instant = 1.0 / (now - last_frame_time)
                    # Smoothed FPS (exponential moving average: 80% old, 20% new) so the number does not jitter.
                    self._fps = instant if self._fps == 0 else 0.8 * self._fps + 0.2 * instant
            last_frame_time = now
            with self._frame_cond:  # publish the new JPEG and wake every stream thread waiting for it
                self._frame_id += 1
                self._jpeg = buffer.tobytes()
                self._frame_cond.notify_all()
            speed = self._speed if fps else 0.0  # no known frame rate (e.g. a webcam) means pacing is impossible
            if speed:  # keep the video on time: skip frames if detection is slow, wait if it is fast
                if pacer is None or pacer.speed != speed:
                    pacer = main.Pacer(fps, speed, frames_consumed, clock=self._clock)
                wait = pacer.wait_time(frames_consumed)
                if wait:
                    self._sleep(wait)
                frames_consumed += main.skip_frames(capture, pacer.frames_to_skip(frames_consumed))
            else:
                pacer = None
            if fps:
                with self._lock:
                    self._video_position = frames_consumed / fps


# --- Report and CSV ---------------------------------------------------------------------------------

def session_events_csv(report):
    """The session's entries in the same format as outputs/events.csv (this is the "Download CSV" file)."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(main.CSV_HEADER)
    for event in report["events"]:
        writer.writerow([event["timestamp"], event["id"], main.ENTRY_EVENT])
    return output.getvalue()


def _format_duration(seconds):
    """Seconds as "12m 05s", or "1h 02m 05s" once it passes an hour."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s" if hours else f"{minutes}m {secs:02d}s"


def _minute_bars_svg(per_minute):
    """The report's "entries per minute" bar chart as an inline SVG string (no JavaScript or libraries,
    so the report stays a single self-contained page that prints correctly)."""
    peak = max([bucket["count"] for bucket in per_minute] + [1])  # "+ [1]" keeps the scale valid when every bar is 0
    bar_width, gap, height = 34, 8, 110
    width = len(per_minute) * (bar_width + gap)
    parts = [f'<svg viewBox="0 0 {width} {height + 24}" role="img" aria-label="Entries per minute, last 10 minutes" '
             f'width="100%" style="max-width:560px">']
    for index, bucket in enumerate(per_minute):
        x = index * (bar_width + gap)
        bar_height = round((bucket["count"] / peak) * (height - 18))
        parts.append(f'<rect x="{x}" y="{height - bar_height}" width="{bar_width}" height="{max(bar_height, 1)}" '
                     f'rx="3" fill="#E2570B"/>')
        if bucket["count"]:
            parts.append(f'<text x="{x + bar_width / 2}" y="{height - bar_height - 4}" text-anchor="middle" '
                         f'font-size="11" fill="#14202B">{bucket["count"]}</text>')
        parts.append(f'<text x="{x + bar_width / 2}" y="{height + 16}" text-anchor="middle" font-size="10" '
                     f'fill="#5B6B78">{html.escape(bucket["label"])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def render_report(report, source, frame):
    """A stand-alone, printable HTML report of the session (opened at /report; "Print or save as PDF" uses the browser)."""
    esc = html.escape  # every value that came from outside the template is escaped before it enters the HTML
    counts = report["counts"]
    frame_text = f"{frame['w']} x {frame['h']} px" if frame else "not known yet"
    zone = report["zone"]
    if report["events"]:
        rows = "".join(
            f"<tr><td>{event['n']}</td><td>{esc(event['timestamp'])}</td><td>{event['id']}</td><td>ENTRY</td></tr>"
            for event in report["events"])
        log = ("<table><thead><tr><th>#</th><th>Time</th><th>Tracking ID</th><th>Event</th></tr></thead>"
               f"<tbody>{rows}</tbody></table>")
    else:
        log = "<p>No entries were recorded in this session.</p>"
    summary = [
        ("Entries counted", counts["entries"]),
        ("Entries per minute", report["entriesPerMinute"]),
        ("Different tracking IDs seen", counts["uniqueIds"]),
        ("Most people in view at once", counts["peakInView"]),
        ("Most people inside the zone at once", counts["peakInside"]),
        ("First entry", report["firstEntry"] or "-"),
        ("Last entry", report["lastEntry"] or "-"),
    ]
    summary_rows = "".join(f"<tr><th scope='row'>{esc(str(label))}</th><td>{esc(str(value))}</td></tr>"
                           for label, value in summary)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>VisionGuard session report</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ margin: 0; padding: 32px 20px 56px; background: #EEF1EE; color: #14202B;
         font: 15px/1.5 "Segoe UI", system-ui, sans-serif; }}
  main {{ max-width: 760px; margin: 0 auto; background: #fff; padding: 36px 40px; border: 1px solid #D3DAD6; }}
  h1 {{ font: 700 30px/1.1 Bahnschrift, "Arial Narrow", sans-serif; letter-spacing: .04em; text-transform: uppercase; margin: 0 0 4px; }}
  h2 {{ font: 600 16px/1.2 Bahnschrift, "Arial Narrow", sans-serif; letter-spacing: .08em; text-transform: uppercase;
        margin: 30px 0 10px; padding-top: 14px; border-top: 2px solid #14202B; break-after: avoid; }}
  section {{ break-inside: avoid; }}
  section.log {{ break-inside: auto; }}  /* a long entry log may run over several pages... */
  tr {{ break-inside: avoid; }}          /* ...but a row is never split, */
  thead {{ display: table-header-group; }} /* and the column titles repeat on each page */
  .meta {{ color: #5B6B78; margin: 0 0 6px; }}
  table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #D3DAD6; }}
  thead th {{ font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: #5B6B78; }}
  tbody th {{ font-weight: 500; color: #5B6B78; width: 55%; }}
  .note {{ background: #FFF3E8; border-left: 4px solid #E2570B; padding: 10px 14px; }}
  button {{ font: inherit; padding: 8px 16px; border: 1px solid #14202B; background: #14202B; color: #fff; cursor: pointer; }}
  @media print {{ body {{ background: #fff; padding: 0; }} main {{ border: 0; padding: 0; max-width: none; }}
                  button {{ display: none; }} }}
</style></head><body><main>
<h1>VisionGuard</h1>
<p class="meta"><strong>Session report</strong> &middot; generated {esc(report['generated'])}</p>
<p class="meta">Source: {esc(str(source))} &middot; frame {esc(frame_text)} &middot; started {esc(report['started'])}
&middot; duration {esc(_format_duration(report['durationSeconds']))}</p>
<p><button onclick="window.print()">Print or save as PDF</button></p>
<section><h2>Summary</h2>
<table><tbody>{summary_rows}</tbody></table></section>
<section><h2>Entries per minute (last 10 minutes)</h2>
{_minute_bars_svg(report['perMinute'])}</section>
<section><h2>Restricted zone</h2>
<p>{len(zone['points'])} corner points, covering {zone['areaPct']}% of the picture
({'the default shape' if zone['isDefault'] else 'a custom shape drawn in the portal'}). An entry is counted when a
tracked person's feet move from outside to inside and stay there for {report['confirmFrames']} frame(s) in a row.</p></section>
<section class="log"><h2>Entry log</h2>
{log}</section>
<section><h2>How to read these numbers</h2>
<p class="note">Tracking IDs are temporary labels given by the tracker. The same person can receive a new ID after
being hidden or leaving the picture, and then counts as a new person, so these numbers are not guaranteed to be
exact and are not a count of different people. Detection confidence is not accuracy.</p></section>
</main></body></html>"""


# --- HTTP server ------------------------------------------------------------------------------------

class PortalServer(ThreadingHTTPServer):
    """HTTP server with one thread per request (needed because the video stream keeps a connection open).
    It carries the monitor and session so the request handler can reach them through self.server."""
    daemon_threads = True  # request threads never block shutdown

    def __init__(self, address, handler, monitor, session):
        super().__init__(address, handler)
        self.monitor, self.session = monitor, session
        port = self.server_address[1]  # read back from the socket, because port 0 means "pick a free one"
        # The only Host and Origin values accepted (Technical Rule 11); see PortalHandler._allowed.
        self.allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        self.allowed_origins = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}


class PortalHandler(BaseHTTPRequestHandler):
    """Routes the portal's pages and JSON API. GET reads state; POST changes it."""
    server_version = "VisionGuardPortal"

    def log_message(self, format, *args):  # keep the console quiet
        pass

    # responses
    def _send(self, status, content_type, data, headers=None):
        """Send a complete response. no-store stops the browser showing stale numbers from its cache."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status, payload):
        """Send `payload` as JSON."""
        self._send(status, "application/json; charset=utf-8", json.dumps(payload).encode())

    def _error(self, status, message):
        """Send an error in the one shape the page expects: {"ok": false, "error": "..."}."""
        self._json(status, {"ok": False, "error": message})

    # safety: this server is for this computer only
    def _allowed(self, is_post):
        """True if the request may proceed; otherwise send a 403 and return False.

        Checking Host blocks DNS-rebinding (a web page on another site pointing its own name at 127.0.0.1).
        Checking Origin on POST blocks cross-site requests (CSRF) from other pages open in the browser.
        """
        if self.headers.get("Host", "") not in self.server.allowed_hosts:
            self._error(403, "Request refused: this portal only answers requests addressed to localhost.")
            return False
        origin = self.headers.get("Origin")  # browsers always send it on cross-site POSTs; scripts such as curl may omit it
        if is_post and origin is not None and origin not in self.server.allowed_origins:
            self._error(403, "Request refused: it did not come from the VisionGuard page.")
            return False
        return True

    def _read_json(self):
        """Read and check a POST body. Returns the parsed JSON object, or None after sending the error reply."""
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            data = self.rfile.read(min(length, 1_000_000))  # always read the body so the reply is not cut off
        else:
            data = b""
        if length > MAX_BODY_BYTES:  # checked after reading (capped above), so an oversized upload cannot exhaust memory
            self._error(413, "Request too large.")
            return None
        if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
            self._error(415, "Send the request as application/json.")
            return None
        try:
            body = json.loads(data or b"{}")
        except ValueError:
            self._error(400, "The request body is not valid JSON.")
            return None
        if not isinstance(body, dict):
            self._error(400, "The request body must be a JSON object.")
            return None
        return body

    # GET
    def do_GET(self):
        """Pages and data: / (the portal), /api/state, /api/events.csv, /report, /stream.mjpg."""
        if not self._allowed(is_post=False):
            return
        path = urlsplit(self.path).path
        server = self.server
        if path == "/":
            self._serve_page()
        elif path == "/api/state":
            self._json(200, server.monitor.state())
        elif path == "/api/events.csv":
            data = session_events_csv(server.session.report()).encode()
            self._send(200, "text/csv; charset=utf-8", data,
                       {"Content-Disposition": 'attachment; filename="visionguard_session_events.csv"'})
        elif path == "/report":
            source, frame = server.monitor.source_info()
            self._send(200, "text/html; charset=utf-8", render_report(server.session.report(), source, frame).encode())
        elif path == "/stream.mjpg":
            self._stream()
        elif path == "/favicon.ico":
            self._send(204, "image/x-icon", b"")
        else:
            self._error(404, "Not found.")

    def _serve_page(self):
        """Serve portal_ui/index.html. It is read on every request so edits show up on refresh."""
        try:
            data = UI_FILE.read_bytes()
        except OSError:
            self._error(500, f"The portal page is missing: {UI_FILE}")
            return
        self._send(200, "text/html; charset=utf-8", data)

    def _stream(self):
        """MJPEG stream: one long response that keeps sending JPEG frames. The browser replaces the
        previous picture with each new part, so a plain <img> tag shows live video with no JavaScript."""
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={STREAM_BOUNDARY}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        monitor, last_id = self.server.monitor, 0
        try:
            while not monitor.stopped:
                frame_id, jpeg = monitor.wait_for_frame(last_id)
                if jpeg is None:
                    continue
                last_id = frame_id  # a repeat frame every 2 s also tells us when the viewer has left
                self.wfile.write(f"--{STREAM_BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                                 f"Content-Length: {len(jpeg)}\r\n\r\n".encode() + jpeg + b"\r\n")
                self.wfile.flush()
        except OSError:
            pass  # the browser closed the stream

    # POST
    def do_POST(self):
        """Actions: set or reset the zone, pause/resume, reset counters, restart the source, set playback speed."""
        if not self._allowed(is_post=True):
            return
        path = urlsplit(self.path).path
        body = self._read_json()
        if body is None:
            return
        server = self.server
        try:
            if path == "/api/zone":
                server.session.set_zone(body.get("points"))
                self._json(200, {"ok": True, "zone": server.session.snapshot()["zone"]})
            elif path == "/api/zone/reset":
                server.session.reset_zone()
                self._json(200, {"ok": True, "zone": server.session.snapshot()["zone"]})
            elif path == "/api/pause":
                if not isinstance(body.get("paused"), bool):
                    self._error(400, "Send {\"paused\": true} or {\"paused\": false}.")
                    return
                server.monitor.pause(body["paused"])
                self._json(200, {"ok": True})
            elif path == "/api/reset":
                server.session.reset()
                self._json(200, {"ok": True})
            elif path == "/api/restart":
                server.monitor.restart()
                self._json(200, {"ok": True})
            elif path == "/api/speed":
                server.monitor.set_speed(body.get("speed"))
                self._json(200, {"ok": True, "speed": float(body["speed"])})  # set_speed has checked it is a number
            else:
                self._error(404, "Not found.")
        except RuntimeError as error:  # validation failures from main.py (bad zone, bad speed, ...) become a 400 with the message
            self._error(400, str(error))


def make_server(monitor, session, host, port):
    """Create the HTTP server (port 0 picks a free port)."""
    return PortalServer((host, port), PortalHandler, monitor, session)


# --- Command line -----------------------------------------------------------------------------------

def make_yolo_detector():
    """Load the YOLO model and return detect(frame) -> list of tracked people. Called on the monitor thread
    (not at import), so the page can open and show "Starting" while the model loads."""
    model = main.load_model(main.MODEL_PATH)
    return lambda frame: main.track_people(model, frame)


def parse_args(argv):
    """Read the command line. Returns the parsed arguments with `source` already converted
    (webcam number -> int, otherwise the file path string, default main.VIDEO_SOURCE)."""
    parser = argparse.ArgumentParser(description="VisionGuard web portal (opens on localhost).")
    parser.add_argument("source", nargs="?", default=None,
                        help="webcam number (default 0) or a video file path")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port to listen on (default {DEFAULT_PORT})")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    parser.add_argument("--speed", type=float, default=0.0,
                        help="video files only: 0 = use every frame (default), 1 = normal speed, 2 = twice as fast, "
                             f"up to {main.MAX_SPEED}; frames the detector cannot keep up with are skipped")
    args = parser.parse_args(argv)
    try:
        main.validate_speed(args.speed)
    except RuntimeError as error:
        parser.error(str(error))
    if args.source is None:
        args.source = main.VIDEO_SOURCE
    elif args.source.isdigit():
        args.source = int(args.source)
    return args


def cli(argv=None):
    """Start the portal: check the configuration, start the monitor thread, serve until Ctrl+C. Returns an exit code."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    zone = main.load_zone(main.ZONE_FILE, main.ZONE_POLYGON)
    try:
        main.validate_config(zone, main.CONFIDENCE_THRESHOLD, main.ZONE_CONFIRM_FRAMES)
    except RuntimeError as error:
        print(error)
        return 1
    session = Session(zone, main.ZONE_CONFIRM_FRAMES, main.EVENTS_CSV, main.ZONE_FILE, main.ZONE_POLYGON)
    is_webcam = main.is_webcam(args.source)
    if args.speed and is_webcam:
        print("Note: --speed only applies to video files; the webcam is always live.")
        args.speed = 0.0
    label = f"Webcam {args.source}" if is_webcam else Path(args.source).name
    monitor = Monitor(session, open_source=lambda: main.open_source(args.source),
                      make_detector=make_yolo_detector, source_label=label, is_webcam=is_webcam, speed=args.speed)
    try:
        server = make_server(monitor, session, "127.0.0.1", args.port)  # 127.0.0.1 = this computer only (Scope Rule 6)
    except OSError as error:  # typically "address already in use"
        print(f"Could not start the portal on port {args.port}: {error}\n"
              f"Close the other program using it, or choose another port: python portal.py --port {args.port + 1}")
        return 1
    url = f"http://127.0.0.1:{server.server_address[1]}"
    monitor.start()
    print(f"VisionGuard portal is running at {url}  (press Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:  # stop the monitor first so the camera or file is released, then close the listening socket
        monitor.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(cli())
