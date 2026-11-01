# Architecture --- VisionGuard

## Architecture Style

A small, local Python application with two entry points that share the
same logic:

-   `main.py` --- the core application: one process, one OpenCV window.
-   `portal.py` --- an optional web portal: one process that runs a small
    HTTP server bound to `127.0.0.1` (this computer only) plus one
    monitoring thread. It was added later at the project owner's request
    (see "Web Portal" below and the change notes in the SRS and PRD).

No database and no cloud service are used by either.

## Processing Flow

`Video Source → Frame Capture → YOLO Person Detection + Tracking → Zone Membership Check → Confirmed Entry Event → CSV Log + Display`

The display is either the OpenCV window (`main.py`) or the browser page
(`portal.py`). Both use the same functions for tracking, zone checking,
entry detection and CSV logging, all defined in `main.py`.

## Main Components

### 1. Video Input

-   Opens either the default webcam or one local video file.
-   Reads frames sequentially.
-   Reports a clear error if the source cannot be opened (missing file,
    file that is not a video, unavailable webcam, source that returns no
    frames).
-   Releases the source on exit, including after errors.
-   **Playback speed (video files only).** By default every frame is
    analysed, so a large video can play far slower than real time. With
    `--speed N` (or the portal's Playback speed dropdown) a `Pacer`
    compares the number of frames used so far with the number the video
    clock says should have been played, then skips the difference with
    `grab()` (when the detector is slower than the video) or waits (when
    it is faster). `0` means every frame, `1` normal speed, up to 16.
    Pacing restarts after a pause so the video does not race ahead. It
    does not apply to a webcam. Skipped frames are never analysed, so the
    confirmation window counts analysed frames.

### 2. Detection and Tracking

-   Loads a small pretrained Ultralytics YOLO model (`yolov8n.pt`).
-   Detects the person class only.
-   Uses the model's tracking functionality (ByteTrack, which needs the
    `lap` package) to associate detections with temporary IDs across
    frames.
-   Does not identify a person's real-world identity.

### 3. Restricted-Zone Logic

-   Stores one polygon as a list of `(x, y)` points given as **fractions
    (0 to 1) of the frame width and height**, so the same zone works at
    any resolution. The built-in default is `ZONE_POLYGON`; a zone saved
    by the portal in `zone.json` is used instead when present and valid.
-   The shape can be any polygon with three or more corners, including
    concave shapes. A shape whose edges cross or touch, has no area, or
    has points outside 0 to 1 is rejected with a clear message.
-   Decides "inside" with the configured **counting rule** (`COUNT_RULE`):
    `"feet"` (default) tests the bottom-centre point of each person's
    bounding box; `"centre"` tests the middle of the box; `"overlap"` is
    true when any part of the box touches the polygon (box corner inside
    the zone, zone corner inside the box, or any two edges crossing, so
    concave zones work).
-   Checks whether the point is inside or outside the polygon (a point on
    the boundary counts as inside).
-   Maintains the confirmed inside/outside state for each tracking ID.

### 4. Entry Event Logic

-   Generate an `ENTRY` event only when a tracked ID changes from
    outside to inside.
-   A change of state is confirmed only after it has lasted
    `ZONE_CONFIRM_FRAMES` frames in a row (default 3), which filters out
    one- or two-frame flicker of the bounding box along the zone edge.
    `1` counts immediately.
-   Do not increment the count on every frame while the person remains
    inside.
-   A person first seen already inside has no known previous state and
    is not counted. With `COUNT_FIRST_SEEN_INSIDE = True` the person is
    instead treated as having been outside just before, so they are
    counted once, after the same confirmation window (the trade-off:
    someone who gets a new ID while standing inside is counted again).
    In the portal, the first frame after a zone or rule change only
    re-learns who is where, so a change never counts anyone.
-   A later exit followed by re-entry may generate another event if the
    tracker preserves the same ID and the transition is observed.
-   Tracking loss or ID reassignment can affect counts; this is an
    acknowledged prototype limitation.

### 5. Event Storage

-   Append local records to `outputs/events.csv`.
-   Columns: `timestamp`, `tracking_id`, `event_type`.
-   Use `ENTRY` as the event type.
-   The file is appended to across runs and sessions; the header is
    written once. No database is needed.

### 6. Display (OpenCV window)

-   Show video frames with the shaded zone, boxes, IDs, confidence
    values, entry count, number of people detected and the status
    `VisionGuard | MONITORING`.
-   `Q`, `ESC` or closing the window stops the program.

### 7. Web Portal (optional, `portal.py`)

-   **Runs on this computer only.** The HTTP server binds to
    `127.0.0.1`. It refuses requests whose `Host` header is not its own
    address and `POST` requests whose `Origin` is not its own page. It
    has no login.
-   **Uses only the Python standard library** (`http.server`,
    `threading`) plus the packages the core application already needs.
    The page (`portal_ui/index.html`) is plain HTML, CSS and JavaScript
    with hand-drawn SVG charts and no external libraries, fonts or
    network access.
-   **Session** (class `Session`): holds the live counts, peaks, history,
    entry events and the zone for one monitoring session. Thread-safe.
-   **Monitor** (class `Monitor`): a thread that reads frames, runs the
    detector, updates the session, draws person boxes, encodes JPEG
    frames and publishes them. Status values: `starting`, `monitoring`,
    `paused`, `error`, `ended`. Errors in opening the source or loading
    the model are shown on the page with a Retry button.
-   **HTTP interface** (used only by the portal page):

    | Method and path | Purpose |
    |---|---|
    | `GET /` | the portal page |
    | `GET /stream.mjpg` | the live video as a motion-JPEG stream |
    | `GET /api/state` | all live numbers, history, events and zone as JSON |
    | `GET /api/events.csv` | this session's entries as CSV |
    | `GET /report` | printable session report page |
    | `POST /api/zone` | apply and save a new zone shape |
    | `POST /api/zone/reset` | restore the default zone |
    | `POST /api/pause`, `/api/reset`, `/api/restart` | pause or resume, reset counters, restart the source |
    | `POST /api/speed` | change the playback speed of a video file (refused for a webcam) |
    | `POST /api/rules` | change the counting rule and/or the first-seen-inside switch (`{"rule": ..., "countFirstSeenInside": ...}`, either may be left out) |
    | `POST /api/manual` | set (`{"count": 5}`) or clear (`{"count": null}`) the manual count to compare with |
    | `GET /api/heatmap` | the foot heat-map: grid size, one count per cell, the maximum and the total |

-   `GET /api/state` includes a `video` object for video files
    (`position`, `duration`, `speed`, `fps`, in seconds and times
    normal speed); it is `null` for a webcam.
-   `GET /api/state` also carries `rules` (the counting rule in use) and
    `manualCheck` (`null`, or your count, the system's count, the
    difference and a one-sentence summary). The heat-map is deliberately
    not in the polled state; the page fetches `/api/heatmap` only while
    the heat-map is switched on.
-   **Foot heat-map.** `Session` adds one count to a 64 × 48 grid cell for
    each person's feet (bottom-centre) on every analysed frame, whichever
    counting rule is active. Only the anonymous cell counts are kept:
    no frame, no track ID and no path. It is cleared with the session.
    The page draws it on the overlay canvas under the zone; the report
    draws it as inline SVG with the zone outline.
-   **Manual-count check.** `describe_manual_check()` turns the system's
    count and a hand count into a difference, a percentage of the hand
    count (none when it is 0) and a sentence. It is a difference between
    two totals, not an accuracy figure, and the report says so.
-   `POST` requests must be `application/json` and at most 64 KB.
-   Changing the zone clears the remembered inside/outside states, so a
    person already inside when the zone changes is not counted as
    entering.

## Code Organization

-   `main.py` --- configuration, zone geometry and validation, saving and
    loading the zone, tracking, entry logic, CSV logging, OpenCV drawing
    and the application loop.
-   `portal.py` --- `Session`, `Monitor`, report and CSV builders, the
    HTTP server and the command line.
-   `portal_ui/index.html` --- the portal page (layout, zone editor,
    charts, controls).
-   `tests/` --- `test_zone_monitoring.py`, `test_application.py`,
    `test_portal.py`, `test_speed.py`, `test_counting_rules.py`,
    `test_end_to_end.py`.

The earlier suggestion to split `main.py` into `video_source.py`,
`zone.py`, `event_logger.py` and `config.py` was not needed; the
functions stay small and named by purpose inside `main.py`.

## Data

The heat-map and the manual count live in memory for one portal session
and are not written to disk.

The application processes frames in memory. Persistent files are
`outputs/events.csv` (the event log) and `zone.json` (only if a zone was
saved from the portal). Video footage is not uploaded or stored by the
application.

## Failure Handling

-   Invalid or unavailable video source: show a clear message and stop
    cleanly (exit code 1); the portal shows the message on the page.
-   Invalid configuration (zone, confidence, confirmation window): report
    which setting is wrong before opening anything. A damaged `zone.json`
    is ignored with a message and the default zone is used.
-   Missing Python package: print `pip install -r requirements.txt`
    instead of a traceback.
-   Model load/inference error: report the error; do not silently
    continue.
-   CSV write error: report that logging failed and keep running.
-   User stop or end of video: release capture resources and close
    display windows.
-   Portal: a refused request returns a JSON error with a clear message;
    a crash inside the monitor thread is reported on the page and the
    source is released.

## Key Technical Assumptions

-   The selected video view is sufficiently clear.
-   The zone is drawn or configured for that view.
-   The person class is available in the pretrained model.
-   Tracking IDs are temporary, not permanent identities.
-   Only one person uses the portal at a time on the same computer.
