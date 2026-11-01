"""VisionGuard: track people in a webcam feed or local video and count restricted-zone entries."""
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
except ImportError as error:
    sys.exit(f"Missing dependency '{error.name or error}'. Install the requirements with: pip install -r requirements.txt")

# --- Configuration (edit these values) ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "yolov8n.pt"
# 0 = default webcam, or a local video such as BASE_DIR / "inputs" / "videos" / "vtest.mp4".
# A path given on the command line overrides this value.
VIDEO_SOURCE = 0
CONFIDENCE_THRESHOLD = 0.4
PERSON_CLASS_ID = 0  # COCO class ID for "person"
WINDOW_NAME = "VisionGuard - Press Q or ESC to quit"
BOX_COLOR = (0, 255, 0)  # BGR green
FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.5
QUIT_KEYS = (ord("q"), ord("Q"), 27)  # 27 = ESC

# --- Tracking, zone and event-log configuration ---
EVENTS_CSV = BASE_DIR / "outputs" / "events.csv"
CSV_HEADER = ["timestamp", "tracking_id", "event_type"]
ENTRY_EVENT = "ENTRY"
TRACKER = "bytetrack.yaml"  # tracker config bundled with Ultralytics
# Restricted zone as (x, y) points given as FRACTIONS of the frame width/height (0-1), so it
# works for any resolution. The default trapezoid suits inputs/videos/vtest.mp4; change it to
# match your own camera view.
ZONE_POLYGON = [(0.20, 0.36), (0.44, 0.36), (0.48, 0.66), (0.12, 0.66)]
# A zone drawn in the web portal (portal.py) is saved here and used instead of ZONE_POLYGON.
# Delete the file, or use "Reset to default" in the portal, to go back to ZONE_POLYGON.
ZONE_FILE = BASE_DIR / "zone.json"
ZONE_COLOR = (0, 0, 255)  # BGR red
# A change of inside/outside state must last this many frames in a row before it counts. This
# filters out one- or two-frame flicker of the bounding box along the zone edge. 1 = count at once.
ZONE_CONFIRM_FRAMES = 3

# --- Counting rule (what "inside the zone" means) ---
# "feet"    the bottom-centre of the person's box, roughly where they stand (the default; Technical Rule 4)
# "centre"  the middle of the box; for cameras that look steeply down, or a zone at body height
# "overlap" any part of the box touching the zone; the most sensitive choice
COUNT_RULES = ("feet", "centre", "overlap")
COUNT_RULE = "feet"
# False (default): a person whose ID is first seen already inside is NOT counted, because no outside-to-inside
# move was observed (Technical Rule 5). True: such a person is counted once (after the confirmation window), which
# catches people who appear inside the zone but can also count the same person again if the tracker gives them a
# new ID while they stand inside.
COUNT_FIRST_SEEN_INSIDE = False


# --- Zone and configuration validation ---------------------------------------------------------------

def is_number(value):
    """True for an int or float. bool is excluded on purpose: True/False are ints in Python but are
    never a sensible coordinate, confidence or speed."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _cross(origin, a, b):
    """Cross product of the vectors origin->a and origin->b. Its sign tells which side of the line
    origin->a the point b lies on (positive = left, negative = right, 0 = exactly on the line)."""
    return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])


def _within_segment_box(p, q, r):
    """True if point r (already known to be on the line through p and q) lies between p and q."""
    return min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1])


def segments_intersect(p1, p2, p3, p4):
    """True if segment p1-p2 and segment p3-p4 cross or touch."""
    # d1/d2: which side of line p3-p4 the ends of the first segment are on; d3/d4: the reverse.
    d1, d2 = _cross(p3, p4, p1), _cross(p3, p4, p2)
    d3, d4 = _cross(p1, p2, p3), _cross(p1, p2, p4)
    if d1 * d2 < 0 and d3 * d4 < 0:  # each segment's ends lie on opposite sides of the other: a proper crossing
        return True
    # A zero means an end point lies on the other segment's line; it only touches if it is also between its ends.
    return ((d1 == 0 and _within_segment_box(p3, p4, p1)) or (d2 == 0 and _within_segment_box(p3, p4, p2))
            or (d3 == 0 and _within_segment_box(p1, p2, p3)) or (d4 == 0 and _within_segment_box(p1, p2, p4)))


def polygon_self_intersects(points):
    """True if any two edges that are not neighbours cross or touch. Concave shapes are fine."""
    count = len(points)
    for i in range(count):
        for j in range(i + 1, count):
            if j == i + 1 or (i == 0 and j == count - 1):
                continue  # neighbouring edges share a corner by design
            if segments_intersect(points[i], points[(i + 1) % count], points[j], points[(j + 1) % count]):
                return True
    return False


def validate_zone(zone_fractions, label="ZONE_POLYGON"):
    """Raise RuntimeError with a clear message if the zone shape cannot be used.

    Rules (see docs/rules.md, Technical Rule 10): at least 3 corners, every coordinate a number from
    0 to 1, no edges crossing or touching, and a real (non-zero) area. `label` names the zone in the message.
    """
    if not isinstance(zone_fractions, (list, tuple)) or len(zone_fractions) < 3:
        raise RuntimeError(f"{label} must be a list of at least 3 points, e.g. [(0.2, 0.3), (0.6, 0.3), (0.4, 0.8)].")
    for point in zone_fractions:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise RuntimeError(f"{label} point {point!r} must be an (x, y) pair.")
        if not all(is_number(value) for value in point):
            raise RuntimeError(f"{label} point {point!r} must contain numbers.")
        if not all(0 <= value <= 1 for value in point):
            raise RuntimeError(f"{label} point {point!r} must be between 0 and 1 "
                               "(fractions of the frame width and height, not pixels).")
    if polygon_self_intersects([tuple(point) for point in zone_fractions]):  # before the area check: a bow-tie has zero signed area
        raise RuntimeError(f"{label} crosses itself. Move a point so that no edges cross or touch.")
    if cv2.contourArea(np.array(zone_fractions, dtype=np.float32)) < 1e-6:
        raise RuntimeError(f"{label} has no area (its points lie on a line).")


def validate_rule(rule):
    """Raise RuntimeError unless `rule` is one of COUNT_RULES."""
    if not isinstance(rule, str) or rule not in COUNT_RULES:
        raise RuntimeError(f"COUNT_RULE must be one of {', '.join(repr(name) for name in COUNT_RULES)}, not {rule!r}.")


def validate_count_first_seen(value):
    """Raise RuntimeError unless the switch is exactly True or False (1, 0 and "yes" are refused on purpose)."""
    if not isinstance(value, bool):
        raise RuntimeError(f"COUNT_FIRST_SEEN_INSIDE must be True or False, not {value!r}.")


def validate_config(zone_fractions, confidence, confirm_frames, rule=COUNT_RULE, count_first_seen_inside=COUNT_FIRST_SEEN_INSIDE):
    """Raise RuntimeError with a clear message if a configuration value cannot be used."""
    validate_zone(zone_fractions)
    if not is_number(confidence) or not 0 < confidence <= 1:
        raise RuntimeError(f"CONFIDENCE_THRESHOLD must be a number above 0 and at most 1, not {confidence!r}.")
    if not isinstance(confirm_frames, int) or isinstance(confirm_frames, bool) or confirm_frames < 1:
        raise RuntimeError(f"ZONE_CONFIRM_FRAMES must be a whole number of at least 1, not {confirm_frames!r}.")
    validate_rule(rule)
    validate_count_first_seen(count_first_seen_inside)


def load_zone(zone_file, default):
    """Return the zone saved in zone_file, or the default if there is no file or it cannot be used."""
    zone_file = Path(zone_file)
    if not zone_file.exists():
        return default
    try:
        points = [tuple(point) for point in json.loads(zone_file.read_text())["points"]]
        validate_zone(points, label="The saved zone")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        # A damaged or hand-edited file must never stop the application: fall back to the default zone.
        print(f"Ignoring the saved zone in {zone_file} ({error}). Using the default zone.")
        return default
    return points


def save_zone(zone_file, points):
    """Validate the zone and save it as JSON; raises RuntimeError if it is invalid or cannot be written."""
    validate_zone(points, label="The zone")  # first, so a malformed shape gives a clear message
    points = [(round(x, 4), round(y, 4)) for x, y in points]
    validate_zone(points, label="The zone")  # again, because rounding can change a very thin shape
    try:  # the JSON layout is {"points": [[x, y], ...]} with x, y as 0-1 fractions of the frame
        Path(zone_file).write_text(json.dumps({"points": [list(point) for point in points]}, indent=2))
    except OSError as error:
        raise RuntimeError(f"Could not save the zone to {zone_file}: {error}") from error


def clear_zone(zone_file):
    """Delete the saved zone so the default zone is used again."""
    try:
        Path(zone_file).unlink(missing_ok=True)
    except OSError as error:
        raise RuntimeError(f"Could not remove the saved zone {zone_file}: {error}") from error


# --- Playback speed (video files only) ---------------------------------------------------------------

MAX_SPEED = 16  # highest playback speed accepted (times normal speed)


def validate_speed(speed):
    """Raise RuntimeError unless speed is 0 (every frame) or a number up to MAX_SPEED (1 = normal speed)."""
    if not is_number(speed) or not math.isfinite(speed) or not 0 <= speed <= MAX_SPEED:
        raise RuntimeError(f"The playback speed must be a number from 0 to {MAX_SPEED} "
                           f"(0 = every frame, 1 = normal speed), not {speed!r}.")


class Pacer:
    """Keeps a video playing at `speed` times its normal speed.

    When the detector is slower than the video, frames_to_skip() says how many frames to drop so the
    video keeps up; when it is faster, wait_time() says how long to pause. speed 0 turns pacing off,
    so every frame is used (the default).
    """

    def __init__(self, fps, speed, start_frames=0, clock=time.monotonic):
        """fps = frame rate of the video; start_frames = frames already consumed when pacing begins;
        clock = time source (a parameter so tests can use a fake clock)."""
        self.fps, self.speed, self.start_frames, self._clock = fps, speed, start_frames, clock
        self._start_time = clock()

    def _target_frames(self):
        """How many frames should have been consumed by now to play at the requested speed."""
        return self.start_frames + (self._clock() - self._start_time) * self.fps * self.speed

    def frames_to_skip(self, frames_consumed):
        """Frames to drop right now because the detector has fallen behind the video (0 if on time or ahead)."""
        if not self.speed:
            return 0
        return max(0, int(self._target_frames() - frames_consumed))

    def wait_time(self, frames_consumed):
        """Seconds to pause because the detector is ahead of the video (0 if on time or behind)."""
        if not self.speed:
            return 0.0
        wait = (frames_consumed - self._target_frames()) / (self.fps * self.speed)
        return wait if wait >= 0.001 else 0.0  # ignore waits too short to be worth a sleep call


def skip_frames(capture, count):
    """Drop `count` frames without decoding them fully; returns how many were really skipped."""
    skipped = 0
    while skipped < count and capture.grab():
        skipped += 1
    return skipped


def video_info(capture):
    """Return (frames per second, number of frames) of a video file, with None for anything unknown."""
    try:
        fps, count = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
    except Exception:  # some capture backends cannot report these values at all
        return None, None
    # OpenCV reports 0 (or a negative number) for "unknown", so turn those into None.
    return (fps if fps and fps > 0 else None), (int(count) if count and count > 0 else None)


# --- Command line and input source -------------------------------------------------------------------

def split_arguments(argv):
    """Split argv into (positional arguments, text after --speed or None)."""
    positional, speed_text, items, index = [], None, argv[1:], 0
    while index < len(items):
        if items[index] == "--speed":
            if index + 1 >= len(items):
                raise RuntimeError("--speed needs a number after it, for example: --speed 1")
            speed_text, index = items[index + 1], index + 2
        else:
            positional.append(items[index])
            index += 1
    return positional, speed_text


def get_source(argv):
    """Use the first command-line argument as the source if given, else VIDEO_SOURCE."""
    positional, _ = split_arguments(argv)
    if not positional:
        return VIDEO_SOURCE
    # A bare number such as "0" is a webcam index; anything else is treated as a video file path.
    return int(positional[0]) if positional[0].isdigit() else positional[0]


def get_speed(argv):
    """The playback speed from --speed (0 = every frame, 1 = normal speed); 0 if not given."""
    _, speed_text = split_arguments(argv)
    if speed_text is None:
        return 0.0
    try:
        return float(speed_text)
    except ValueError:
        raise RuntimeError(f"--speed must be a number, not {speed_text!r}.") from None


def is_webcam(source):
    """True if the source is a webcam index (an int); a video file is given as a path string."""
    return isinstance(source, int)


def load_model(model_path):
    """Load the pretrained YOLO model; raise RuntimeError with a clear message on failure."""
    try:
        return YOLO(str(model_path))
    except Exception as error:
        raise RuntimeError(f"Could not load YOLO model '{model_path}': {error}") from error


def open_source(source):
    """Open a webcam index or a local video file; raise RuntimeError if unavailable."""
    if not is_webcam(source) and not Path(source).is_file():
        raise RuntimeError(f"Video file not found: {source}")
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()  # release even a failed capture so no device handle is left behind
        if is_webcam(source):
            raise RuntimeError(f"Webcam {source} is unavailable (not connected or in use).")
        raise RuntimeError(f"Could not open video file: {source}")
    return capture


# --- Zone geometry and entry counting ----------------------------------------------------------------

def bottom_centre(box):
    """Reference point of a person: the middle of the bottom edge of the box (x1, y1, x2, y2).

    This is roughly where the feet touch the ground, which is what decides "inside the zone"
    (Technical Rule 4). The box centre would wrongly count people standing just in front of the zone.
    """
    x1, _, x2, y2 = box
    return ((x1 + x2) / 2, float(y2))


def centre_point(box):
    """The middle of the box (x1, y1, x2, y2); the reference point of the "centre" counting rule."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def scale_zone(zone_fractions, width, height):
    """Convert zone points given as 0-1 fractions of the frame into pixel coordinates."""
    return np.array([(x * width, y * height) for x, y in zone_fractions], dtype=np.float32)


def point_in_zone(point, zone):
    """True if the point is inside the zone polygon or exactly on its boundary."""
    # measureDist=False makes this return +1 (inside), 0 (on the edge) or -1 (outside).
    return cv2.pointPolygonTest(zone, (float(point[0]), float(point[1])), False) >= 0


def find_entries(zone_states, inside_by_id, confirm_frames=1, count_first_seen_inside=False):
    """Return the track IDs that moved from outside to inside the zone in this frame.

    zone_states maps track ID -> (confirmed inside/outside state, consecutive frames seen in the
    opposite state) and is updated in place. A change of state is confirmed only after it has lasted
    confirm_frames frames in a row, which filters out one- or two-frame flicker of the bounding box
    at the zone edge. With confirm_frames=1 every change counts immediately.
    A person first seen already inside has no known previous state, so by default no entry is counted.
    With count_first_seen_inside=True such a person is treated as having been outside just before, so they
    are counted once, after the same confirmation window.
    """
    entered = []
    for track_id, is_inside in inside_by_id.items():
        if track_id not in zone_states:
            if count_first_seen_inside and is_inside:
                zone_states[track_id] = (False, 0)  # pretend "outside until now", then judge this frame below
            else:
                zone_states[track_id] = (is_inside, 0)  # first sighting: remember the state, never count an entry
                continue
        confirmed, differing_frames = zone_states[track_id]
        # Count consecutive frames that disagree with the confirmed state; any agreeing frame resets the count.
        differing_frames = differing_frames + 1 if is_inside != confirmed else 0
        if differing_frames >= confirm_frames:
            confirmed, differing_frames = is_inside, 0  # the change has lasted long enough: accept it
            if is_inside:  # only outside -> inside is an entry; inside -> outside is accepted silently
                entered.append(track_id)
        zone_states[track_id] = (confirmed, differing_frames)
    return entered


def log_entry(csv_path, track_id):
    """Append one ENTRY event (timestamp, tracking_id, event_type) to the CSV file.

    Creates the folder and the header row if needed. If the file cannot be written, prints
    the reason and returns False instead of stopping the application.
    """
    csv_path = Path(csv_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not csv_path.exists() or csv_path.stat().st_size == 0  # also covers an empty leftover file
        with open(csv_path, "a", newline="") as file:  # append, so earlier runs' entries are kept; newline="" avoids blank rows on Windows
            writer = csv.writer(file)
            if needs_header:
                writer.writerow(CSV_HEADER)
            writer.writerow([timestamp, track_id, ENTRY_EVENT])
    except OSError as error:
        print(f"Could not write to {csv_path}: {error}")
        return False
    return True


def box_overlaps_zone(box, zone):
    """True if any part of the box (x1, y1, x2, y2) touches or lies inside the zone polygon (pixels).

    Two shapes overlap if a box corner is inside the zone, a zone corner is inside the box, or an edge of one
    crosses an edge of the other; if none of those holds they are apart. This works for concave zones too.
    """
    x1, y1, x2, y2 = box
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    zone_points = [(float(x), float(y)) for x, y in zone]
    if any(point_in_zone(corner, zone) for corner in corners):
        return True
    if any(x1 <= x <= x2 and y1 <= y <= y2 for x, y in zone_points):
        return True
    for index in range(4):
        start, end = corners[index], corners[(index + 1) % 4]
        for j, zone_start in enumerate(zone_points):
            if segments_intersect(start, end, zone_start, zone_points[(j + 1) % len(zone_points)]):
                return True
    return False


def person_in_zone(box, zone, rule=COUNT_RULE):
    """Is this person inside the zone under the chosen counting rule ("feet", "centre" or "overlap")?"""
    if rule == "feet":
        return point_in_zone(bottom_centre(box), zone)
    if rule == "centre":
        return point_in_zone(centre_point(box), zone)
    if rule == "overlap":
        return box_overlaps_zone(box, zone)
    raise ValueError(f"Unknown counting rule {rule!r}; use one of {COUNT_RULES}.")


def check_zone(people, zone, rule=COUNT_RULE):
    """Map each tracked person's ID to True/False: are they inside the zone under the counting rule?"""
    return {
        track_id: person_in_zone((x1, y1, x2, y2), zone, rule)
        for track_id, x1, y1, x2, y2, _ in people
    }


def update_monitoring(people, zone, zone_states, csv_path, confirm_frames=1, rule=COUNT_RULE,
                      count_first_seen_inside=False):
    """Check the zone for this frame, detect entries and log each one to the CSV file.

    Returns (inside_by_id, entered_ids).
    """
    inside_by_id = check_zone(people, zone, rule)
    entered_ids = find_entries(zone_states, inside_by_id, confirm_frames, count_first_seen_inside)
    for track_id in entered_ids:
        log_entry(csv_path, track_id)
    return inside_by_id, entered_ids


# --- Detection and tracking --------------------------------------------------------------------------

def track_people(model, frame):
    """Detect and track people; return a list of (track_id, x1, y1, x2, y2, confidence).

    The tracker may return no IDs (for example when nobody is visible); that gives an empty list.
    """
    result = model.track(
        # persist=True keeps the tracker's memory between calls, so IDs stay stable from frame to frame.
        # classes limits detection to people only (Scope Rule 3); conf drops low-confidence detections.
        frame, persist=True, classes=[PERSON_CLASS_ID], conf=CONFIDENCE_THRESHOLD,
        tracker=TRACKER, verbose=False,
    )[0]  # one frame in, so take the first (only) result
    if result.boxes.id is None:  # no tracked person this frame
        return []
    track_ids = result.boxes.id.int().tolist()
    boxes = result.boxes.xyxy.int().tolist()
    confidences = result.boxes.conf.tolist()
    return [(track_id, *box, conf) for track_id, box, conf in zip(track_ids, boxes, confidences)]


# --- Drawing on the video frame ----------------------------------------------------------------------

def blend_rectangle(frame, top_left, bottom_right, color, opacity):
    """Fill a rectangle with a semi-transparent colour."""
    overlay = frame.copy()  # draw the solid shape on a copy, then mix the copy back in at `opacity`
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)  # thickness -1 = filled
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)


def draw_label(frame, text, left, top, background, text_color=(255, 255, 255)):
    """Draw text on a filled background so it stays readable on any video, ending at y=top."""
    (text_width, text_height), baseline = cv2.getTextSize(text, FONT, LABEL_SCALE, 1)
    box_top = max(top - text_height - baseline - 4, 0)  # keep the label on screen when the box touches the top edge
    cv2.rectangle(frame, (left, box_top), (left + text_width + 6, box_top + text_height + baseline + 4),
                  background, -1)
    cv2.putText(frame, text, (left + 3, box_top + text_height + 1), FONT, LABEL_SCALE, text_color, 1)


def draw_zone(frame, zone):
    """Draw the restricted zone as a lightly shaded polygon with its boundary and name."""
    points = zone.astype(np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [points], ZONE_COLOR)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.polylines(frame, [points], True, ZONE_COLOR, 2)
    left, top = points.min(axis=0)
    draw_label(frame, "RESTRICTED ZONE", int(left), int(top), ZONE_COLOR)


def text_color_for(background):
    """Black or white text, whichever is easier to read on the given BGR background colour."""
    blue, green, red = background
    # Perceived brightness (luma) weights green most and blue least; above 140/255 the colour is "light".
    return (0, 0, 0) if 0.114 * blue + 0.587 * green + 0.299 * red > 140 else (255, 255, 255)


def draw_people(frame, people, inside_by_id, outside_color=BOX_COLOR, inside_color=ZONE_COLOR, compact=False,
                rule=COUNT_RULE):
    """Draw each person's box, 'ID <n> Person <confidence>' label and the point that decides inside/outside.

    People inside the zone get their own box colour and an 'INSIDE' tag so the state is not colour-only.
    compact=True drops the word 'Person' to keep labels short in crowded scenes.
    The dot marks the bottom-centre ("feet" rule) or the middle of the box ("centre" rule); the "overlap" rule
    has no single point, so no dot is drawn and the whole box is what counts.
    """
    for track_id, x1, y1, x2, y2, confidence in people:
        is_inside = inside_by_id[track_id]
        color = inside_color if is_inside else outside_color
        name = "" if compact else "Person "
        label = f"ID {track_id} {name}{confidence:.2f}" + (" INSIDE" if is_inside else "")
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_label(frame, label, x1, y1, color, text_color_for(color))
        if rule != "overlap":
            dot_x, dot_y = centre_point((x1, y1, x2, y2)) if rule == "centre" else bottom_centre((x1, y1, x2, y2))
            cv2.circle(frame, (int(dot_x), int(dot_y)), 4, color, -1)


def status_lines(entry_count, people_count, inside_count):
    """Text for the status panel: project name, monitoring status, entries, detections, zone."""
    noun = "person" if people_count == 1 else "people"
    zone_status = f"OCCUPIED ({inside_count} inside)" if inside_count else "CLEAR"
    return [
        "VisionGuard | MONITORING",
        f"Entries: {entry_count}",
        f"Detected: {people_count} {noun}",
        f"Zone: {zone_status}",
    ]


def draw_status(frame, lines):
    """Draw the status panel (top-left, semi-transparent); the zone line is red while occupied."""
    line_height = 24
    blend_rectangle(frame, (0, 0), (290, line_height * len(lines) + 8), (0, 0, 0), 0.65)
    for index, text in enumerate(lines):
        occupied = text.startswith("Zone: OCCUPIED")
        color = ZONE_COLOR if occupied else (255, 255, 255)  # white by default, red while someone is inside
        if text.startswith("Zone:") and not occupied:
            color = BOX_COLOR  # "Zone: CLEAR" is green
        scale, thickness = (0.6, 2) if index < 2 else (0.55, 1)  # the first two lines (name, entries) are bolder
        cv2.putText(frame, text, (8, 22 + index * line_height), FONT, scale, color, thickness)


# --- Main loop ---------------------------------------------------------------------------------------

def should_quit():
    """True if Q/ESC was pressed or the window was closed."""
    # waitKey also gives OpenCV time to redraw the window; & 0xFF keeps just the key code (-1 becomes 255).
    key = cv2.waitKey(1) & 0xFF
    return key in QUIT_KEYS or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1


def process_video(model, capture, source, zone_fractions=None, speed=0.0):
    """Track people, count zone entries and display the result until the video ends or the user quits.

    speed 0 uses every frame. A positive speed plays a video file at that many times its normal speed by
    skipping the frames the detector cannot keep up with (1 = normal speed).
    """
    if zone_fractions is None:
        zone_fractions = ZONE_POLYGON
    fps, _ = video_info(capture) if speed and not is_webcam(source) else (None, None)
    pacer = Pacer(fps, speed) if fps else None
    frames_consumed = 0  # frames taken from the video so far, including skipped ones
    zone = None  # pixel polygon, built from the first frame's size
    zone_states = {}  # track ID -> (confirmed inside/outside state, frames seen in the opposite state)
    entry_count = 0
    frames_read = 0
    if COUNT_RULE != "feet" or COUNT_FIRST_SEEN_INSIDE:  # say so when the counting differs from the default
        print(f"Counting rule: {COUNT_RULE}; people first seen inside the zone "
              f"{'ARE' if COUNT_FIRST_SEEN_INSIDE else 'are not'} counted.")
    while True:
        grabbed, frame = capture.read()
        if not grabbed:
            if frames_read == 0:  # never got a single frame: that is a failure, not a normal end of video
                raise RuntimeError(f"The video source opened but returned no frames: {source}")
            print("Webcam stopped sending frames." if is_webcam(source) else "End of video.")
            break
        frames_read += 1
        frames_consumed += 1
        if zone is None:  # build the pixel polygon once, from the real frame size
            height, width = frame.shape[:2]
            zone = scale_zone(zone_fractions, width, height)

        # 1. detect and track, 2. decide inside/outside and count entries, 3. draw, 4. show.
        people = track_people(model, frame)
        inside_by_id, entered_ids = update_monitoring(
            people, zone, zone_states, EVENTS_CSV, ZONE_CONFIRM_FRAMES, COUNT_RULE, COUNT_FIRST_SEEN_INSIDE)
        for track_id in entered_ids:
            entry_count += 1
            print(f"ENTRY: ID {track_id} (total entries: {entry_count})")

        draw_zone(frame, zone)
        draw_people(frame, people, inside_by_id, rule=COUNT_RULE)
        draw_status(frame, status_lines(entry_count, len(people), sum(inside_by_id.values())))
        cv2.imshow(WINDOW_NAME, frame)
        if should_quit():
            print("Stopped by user.")
            break
        if pacer:  # --speed was given: wait if we are ahead of the video, drop frames if we are behind
            time.sleep(pacer.wait_time(frames_consumed))
            frames_consumed += skip_frames(capture, pacer.frames_to_skip(frames_consumed))
    print(f"Entries counted this run: {entry_count} (events file: {EVENTS_CSV})")


def main():
    """Validate the configuration and input first, then run; always release the video and windows."""
    capture = None
    try:
        source, speed = get_source(sys.argv), get_speed(sys.argv)
        validate_speed(speed)
        zone_fractions = load_zone(ZONE_FILE, ZONE_POLYGON)
        validate_config(zone_fractions, CONFIDENCE_THRESHOLD, ZONE_CONFIRM_FRAMES, COUNT_RULE, COUNT_FIRST_SEEN_INSIDE)
        capture = open_source(source)  # checked before the slower model load
        if speed and is_webcam(source):
            print("Note: --speed only applies to video files; the webcam is always live.")
            speed = 0.0
        model = load_model(MODEL_PATH)
        process_video(model, capture, source, zone_fractions=zone_fractions, speed=speed)
    except RuntimeError as error:  # expected, user-facing problems (bad zone, missing file, no webcam, ...)
        print(error)
        return 1
    except KeyboardInterrupt:  # Ctrl+C is a normal way to stop, not an error
        print("Interrupted.")
    except ImportError as error:
        print(f"Missing dependency '{error.name or error}'. Install the requirements with: pip install -r requirements.txt")
        return 1
    except Exception as error:
        print(f"Stopped because of an unexpected error: {error}")
        return 1
    finally:  # runs on normal exit, on quitting and on every error (Technical Rule 7)
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
