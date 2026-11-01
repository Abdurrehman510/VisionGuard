# VisionGuard — Real-Time Object Detection and Restricted-Zone Monitoring

## 1. Introduction

VisionGuard is a small, local Python application for a college project. It watches a webcam or a video
file, detects **people only** with the pretrained YOLOv8n model, gives each person a temporary tracking
ID, and counts every time a tracked person moves from **outside to inside** one restricted zone. Each
entry is saved to a CSV file with a timestamp, the tracking ID and the event type (`ENTRY`).

Everything runs on the local machine. No model is trained, no video is saved or uploaded, and there is
no database or cloud service.

There are two ways to use it. `main.py` is the core application with a plain OpenCV window. `portal.py` is an
**optional web portal** that shows the same monitoring in your browser at `http://127.0.0.1:8000`, with a live
view, a zone you can draw and reshape freely with the mouse, live numbers and charts, an entry log and a
printable session report. The portal is a small server that only answers requests from your own computer.

## 2. Problem statement

Watching a camera feed continuously to see whether anyone enters a restricted area is tedious and
error-prone for a person to do. A simple system that can look at the video, recognise people, notice
when one of them crosses into a marked area and keep a record of when it happened would reduce that
effort. VisionGuard is a working academic prototype of that idea. It is **not** a security product and
makes no guarantee of detecting every entry (see [Limitations](#11-limitations-and-future-improvements)).

## 3. Objectives

1. Open a webcam or a local video and detect people with a pretrained model.
2. Track each detected person with a temporary ID.
3. Define one restricted zone and decide whether each person is inside or outside it.
4. Detect and count each outside-to-inside transition without counting a person repeatedly while they
   stay inside.
5. Record each entry in `outputs/events.csv`.
6. Show the result clearly on the video and exit safely, releasing the camera/video and windows.

## 4. Features

- Webcam or local video input (MP4 tested), chosen on the command line.
- Person-only detection (COCO class 0) with `yolov8n.pt`, a small CPU-friendly model.
- Temporary tracking IDs from the Ultralytics ByteTrack tracker.
- One restricted zone of **any shape you want**: three or more corners, concave shapes such as an L or a star
  are fine, and a shape whose edges cross each other is refused with a clear message. By default the
  reference point is the bottom-centre of each person's box (roughly where their feet are); you can choose
  another rule (see *Counting rule* below).
- Entry counting on observed outside-to-inside transitions only, with a short confirmation window
  (3 frames by default) so a box that flickers across the zone edge for a frame or two is not counted.
- **Counting rule (optional, off by default):** choose what counts as "inside" (feet, middle of the body, or any
  overlap of the box with the zone) and whether to also count a person who is first seen already inside. The
  defaults reproduce the original behaviour exactly.
- CSV event log: `timestamp`, `tracking_id`, `event_type`.
- On-screen display: project name and `MONITORING` status, boxes with `ID` / `Person` / confidence,
  the shaded restricted zone, the entry counter, the number of people detected and the zone status
  (`CLEAR` / `OCCUPIED`). People inside the zone get a red box and an `INSIDE` tag, so the state is not
  shown by colour alone.
- Input and configuration checks with clear messages: missing video file, unavailable webcam, a file that
  is not a video, a video that returns no frames, an invalid zone / confidence / confirmation setting,
  a model file that cannot be loaded (a missing `yolov8n.pt` is downloaded again), missing Python packages.
- Safe exit with `Q`, `ESC`, closing the window, `Ctrl+C`, the end of the video, or an error: the
  video source and windows are always released.
- **Optional web portal** (`portal.py`, opens in your browser):
  - live camera view with the zone drawn over it, in a light or dark theme, on a computer or a phone screen;
  - **zone editor**: drag corners, add corners with the `+` handles, move the whole shape, draw a new shape
    corner by corner, pick a ready-made shape, resize with a slider, type exact percentages, undo and redo;
    every change applies to the next frame and is saved;
  - live numbers (entries, people inside, people in view, most at once, tracking IDs), a 3-minute activity
    chart, an entries-per-minute chart and a live entry log;
  - session report page (printable, can be saved as PDF) and a CSV download;
  - pause, reset counters and restart-the-source buttons, and clear messages if the camera has a problem;
  - for video files: a progress bar with the video's own clock and a **playback speed** control (every frame,
    0.5×, 1×, 2×, 4×, 8×), also available as `--speed` on the command line (see *Using a video file*);
  - **foot heat-map**: shade the picture by where people's feet have been, to see where the zone should go
    (anonymous counts per area only; no picture or per-person path is kept);
  - **counting rule** controls (what counts as inside, and the first-seen-inside switch);
  - **manual-count check**: type how many entries you counted yourself and see the difference from the
    system's count, in the page and in the report.

## 5. Technology stack

| Component | Use |
|---|---|
| Python 3.11 (specified); **tested on 3.12.7** | Language |
| OpenCV (`opencv-python` 5.0.0.93) | Video input, drawing, display, point-in-polygon test |
| Ultralytics YOLO (8.4.193) with `yolov8n.pt` | Pretrained person detection and ByteTrack tracking |
| NumPy (2.5.4) | Zone coordinates |
| `lap` (0.5.13) | Required by the Ultralytics tracker |
| Python `csv`, `unittest` (standard library) | Event log and tests |
| Python `http.server` (standard library) with plain HTML, CSS and JavaScript | Optional web portal; no extra packages, fonts or libraries are downloaded, so it works offline |

Versions are pinned in `requirements.txt`. PyTorch is installed automatically as a dependency of
Ultralytics. The web portal adds **no** new dependencies.

## 6. Project structure

```
VisionGuard/
├── main.py                        # the core application (OpenCV window)
├── portal.py                      # optional web portal server (localhost only)
├── portal_ui/index.html           # the portal page (HTML, CSS and JavaScript in one file)
├── requirements.txt               # pinned dependencies
├── README.md
├── models/yolov8n.pt              # pretrained YOLOv8n weights (re-downloaded automatically if missing)
├── inputs/videos/vtest.mp4        # sample video (the default zone is made for this one)
├── inputs/videos/worker-zone-detection.mp4   # Intel sample video, optional extra for manual testing (12 MB)
├── inputs/videos/worker-zone-detection-small.mp4   # lighter copy of it (960x540, 15 fps, 5 MB), made with ffmpeg
├── outputs/events.csv             # event log, created on the first entry
├── zone.json                      # your custom zone, created when you change the zone in the portal
├── tests/
│   ├── test_zone_monitoring.py    # zone shapes, entry, confirmation window, CSV, zone file, configuration
│   ├── test_application.py        # quit keys, error handling, resource release, status text
│   ├── test_portal.py             # portal: session numbers, monitor thread, HTTP API, security checks
│   ├── test_speed.py              # video playback speed: pacing, frame skipping, --speed
│   ├── test_counting_rules.py     # counting rules (feet / centre / overlap) and the first-seen-inside switch
│   └── test_end_to_end.py         # real model + real video frames (slow)
└── docs/
    ├── example_events.csv         # event log produced by a full run on the sample video
    ├── screenshots/               # screenshots of the application and of the portal
    └── (project documents)
```

## 7. Installation

Requires Python on Windows (the specification says 3.11; the project was tested on 3.12.7) and a
webcam or a video file.

```powershell
cd VisionGuard
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The first install is large because PyTorch is included. If `models/yolov8n.pt` is missing, the first run
downloads it (about 6 MB), which needs an internet connection once.

## 8. How to run

```powershell
python main.py                                # default webcam (index 0)
python main.py 1                              # a different webcam index
python main.py inputs\videos\vtest.mp4        # a local video file
```

Press **Q** or **ESC** (or close the window) to stop. When a video ends, the program closes the window,
prints `Entries counted this run: N` and exits. Start with a fresh log by deleting `outputs/events.csv`;
otherwise new events are appended to it.

### Web portal (optional)

```powershell
python portal.py                              # your default webcam; the page opens in your browser
python portal.py inputs\videos\vtest.mp4      # the same page, fed from a video file
python portal.py --port 8001                  # another port, if 8000 is busy
python portal.py --no-browser                 # do not open the browser by itself
python portal.py --rule overlap               # start with another counting rule (feet | centre | overlap)
python portal.py --count-first-seen           # start by also counting people first seen already inside
```

Open **http://127.0.0.1:8000**. The first start takes anywhere from a few seconds to about a minute (the model
and the camera have to start; it is slower when the computer is short of free memory), and the page shows
"Starting" until then. Press `Ctrl+C` in the terminal to stop it. Closing the browser tab does not stop the
monitoring; the numbers keep counting while `portal.py` runs.

**Drawing and reshaping the zone** (this is also how you adapt the zone to your own camera):

| Do this | To get this |
|---|---|
| **Edit shape** (`E`), then drag a square corner | Move that corner |
| Drag a **+** on an edge | Add a new corner exactly where you drop it |
| Drag inside the shape | Move the whole zone |
| Double-click (or right-click) a corner, or select it and press `Del` | Remove that corner (at least 3 must stay) |
| **Draw new** (`D`), then click out the corners; click the first corner or press `Enter` to finish | Draw any shape from scratch (`Backspace` removes the last corner, `Esc` cancels) |
| **Shapes…** menu | Replace the zone with a rectangle, trapezoid, triangle, pentagon, L-shape or star |
| **Size** slider | Make the zone smaller or larger around its centre |
| **Edit points by numbers** | Type each corner's position as a percentage of the picture (also works without a mouse) |
| **Undo** / **Redo** (`Z` / `Y`) | Step back and forward through your changes |
| **Reset to default** | Go back to the built-in zone |

A change takes effect on the next frame and is saved to `zone.json`. If you drag a corner so the edges cross, the
red edges and a message tell you, and nothing is changed until the shape is valid. **`python main.py` uses the
same saved zone**, so a zone you drew in the portal is also used by the OpenCV window.

**Other controls:** `Space` or **Pause** freezes monitoring, **Reset counters** starts a fresh session (click twice
to confirm; the CSV log is kept), **Restart source** re-opens the camera (or plays the video again) and also starts a fresh session. The moon button
switches between light and dark. Every entry is also appended to `outputs/events.csv`, exactly as in the OpenCV
version.

**Session report:** the **Session report** card shows the length of the session, entries, entries per minute and
peaks. **Open full report** opens a printable page that breaks cleanly across pages (use *Print* and choose
*Save as PDF* if you want a PDF) and
**Download CSV** saves this session's entries. The report page repeats the warning that tracking IDs are temporary.

**Zone insights: where to put the zone, what counts as inside, and a check of the count.** Three tools that help
you fit the zone and the counting to your own camera view. None of them changes the default behaviour.

| Tool | What it does | How to use it |
|---|---|---|
| **Foot heat-map** (`H`, or the *Foot heat-map* button) | Shades the live picture from blue to orange by how much time feet have spent in each area during this session, and adds a *Where people walked* picture to the report | Let people walk through the picture, then compare the hot areas with your zone. Move the zone where people really walk. *Reset counters* clears it |
| **What counts as inside** (card under the video) | **Feet** (default): the bottom-centre of the box. **Middle of the body**: the centre of the box. **Any overlap**: any part of the box touching the zone | Pick one; it applies from the next frame. The dot drawn on each person shows the point that decides (no dot for *Any overlap*, because the whole box counts) |
| **Also count people who first appear already inside** | A person whose ID is first seen inside the zone is counted once, after the same confirmation window | Off by default. Turn it on when people can appear inside the zone (a doorway, a frame edge). See the limitations for the trade-off |
| **Check against your count** | You type how many entries you counted by eye; the page and the report show the system's count, yours and the difference | Count while watching the video, then press *Compare*. It follows the live count, so you can enter your number first |

Changing the zone, or the rule, never counts the people who are already in view: the next frame only re-learns where
everyone is. The portal keeps the choices for the current run only; start the portal with `--rule` or
`--count-first-seen`, or change `COUNT_RULE` / `COUNT_FIRST_SEEN_INSIDE` in `main.py`, to make them the starting
values (the OpenCV window, `python main.py`, uses those two constants and has no on-screen switch).

**What the numbers mean:** the manual check compares two totals for one session. It is *not* an accuracy score:
equal totals can hide a missed entry and a false one that cancel out, and the report says so. The heat-map counts
foot positions (frames), not people, so someone standing still looks hot; it uses the foot position whichever
counting rule is selected.

**Privacy and safety:** the portal listens on `127.0.0.1` only, so other computers cannot reach it, and it refuses
requests whose address or origin does not match its own page, which is designed to stop other websites you may
have open from controlling it. It has no login, because it is meant for one person on one computer, so anyone
using the same computer and browser can open it while it runs. Nothing is uploaded and no video is stored.

### Using a video file instead of the webcam

Both programs take the video as an argument, so you can run either of them on a recording:

```powershell
python portal.py inputs\videos\vtest.mp4                  # web portal on a video
python main.py   inputs\videos\vtest.mp4                  # OpenCV window on a video
python portal.py "C:\Users\You\Videos\my clip.mp4"        # any path; use quotes if it contains spaces
python portal.py inputs\videos\worker-zone-detection.mp4 --speed 1    # play at normal speed (see below)
python portal.py 1                                        # (for comparison) a second webcam, not a file
```

**Playback speed (video files only).** By default every single frame is analysed, so a long or high-resolution
video can play very slowly: the 1080p, 60-frames-per-second Intel clip played at about 3% of normal speed on this
laptop (about 1.8 pictures per second), which would take about 40 minutes. To keep a video on time, give a
speed: `--speed 1` plays at normal speed by skipping the frames the detector cannot keep up with, `--speed 2` is
twice as fast, `--speed 0.5` half speed, and no option (or `0`) uses every frame. The portal also has a
**Playback speed** dropdown under the video that you can change at any time, and a progress bar with the video's
own clock (for example `0:36 / 1:16`), which is separate from the page's *Session* clock. The OpenCV window takes
the same `--speed` option. Speed does not apply to a webcam, which is always live.

Skipping frames has a cost: the tracker sees fewer pictures, and a very fast person crossing a thin zone could be
missed. In my replay test on the Intel video, analysing every 4th frame still counted all 6 workers correctly.
Another way to make a video faster to analyse is a lighter copy, for example with ffmpeg (about 4 times fewer
frames, each a quarter of the size; the zone still fits because zones are fractions of the picture):

```powershell
ffmpeg -i inputs\videos\worker-zone-detection.mp4 -vf "scale=960:-2,fps=15" -an inputs\videos\worker-zone-detection-small.mp4
```

1. Copy the video into `inputs\videos\` (or just use its full path).
2. Start the program with that path. In the portal, the **Source** at the top shows the file name.
3. **Draw a zone for that video.** The built-in zone only fits `vtest.mp4`. In the portal use *Edit shape* or
   *Draw new*; the zone is saved to `zone.json` and used the next time too, so switch back with
   *Reset to default* when you go back to `vtest.mp4`.
4. When the video ends the portal shows **Video ended** with a *Play again* button (this restarts the session and
   resets the counters); the OpenCV window simply closes and prints the total.

Good to know:

- **Format.** MP4 (H.264) is what was tested; the original OpenCV sample was an AVI and opened too. Other
  formats depend on the codecs inside OpenCV. If you see `Could not open video file` or `returned no frames`,
  convert the video to MP4 with H.264 (for example with HandBrake or ffmpeg) and try again.
- **Size.** Smaller is smoother on a laptop: 720p or lower is a good choice. Full HD (1920×1080) should work
  because the picture is handled at its own size, but it was not tested and will be slower.
- **Speed.** Without `--speed`, the video is processed as fast as the computer can detect people, so it usually
  plays slower than real time (see *Playback speed* above to change that). The counts follow frames, not the clock.
- **Fixed camera.** The zone is fixed on the picture, so use footage from a camera that does not move. Handheld or
  panning shots make the zone meaningless.
- **Log file.** Every entry is added to `outputs/events.csv`. Delete that file before a test if you want a clean log.

### Where to find sample videos for manual testing

| Source | What you get | Terms and notes |
|---|---|---|
| **Already in this project:** `inputs/videos/vtest.mp4` | OpenCV's pedestrian sample (people walking across a campus path, 768×576, 795 frames). The default zone is made for it. | Original: [`vtest.avi` in the OpenCV samples](https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi), converted to MP4. |
| **Intel sample video** `worker-zone-detection.mp4` (**downloaded into this project**: `inputs/videos/worker-zone-detection.mp4`) | A warehouse floor with red lines taped out as zones, and two workers in hard hats walking close to the camera. 1920×1080, 60 fps, 4,548 frames (about 76 s), 12 MB. Source: [worker-zone-detection.mp4](https://github.com/intel-iot-devkit/sample-videos/raw/master/worker-zone-detection.mp4). | Linked from Intel's [restricted-zone-notifier](https://github.com/intel-iot-devkit/restricted-zone-notifier-python) project (BSD-3-Clause for that project's code; no separate licence for the video was stated there, and I did not check the video repository's own licence, so use it for private testing and check before redistributing it). **What I checked:** it opens in OpenCV, and in 31 frames sampled every 2.5 s a person was found in 23 of them (up to 2 at once; the scene starts empty), with confidences around 0.87 to 0.91 in the frame I looked at. **What I did not check:** entry counting on it. |
| **Pexels** ([pexels.com/videos](https://www.pexels.com/videos/)) | Free stock footage. Search for *pedestrians*, *people walking*, *street*, *crowd* or *CCTV*, and choose a clip filmed from a fixed spot. Download the HD (720p) size. | The [Pexels licence](https://www.pexels.com/license/) allows free use without attribution, but you may not imply that the people or brands endorse anything, show identifiable people in a bad light, or sell the unaltered video. |
| **MOTChallenge MOT17** ([motchallenge.net](https://motchallenge.net/data/MOT17/)) | Research pedestrian footage. Individual videos can be downloaded without the 5.5 GB archive, for example [MOT17-02 (people walking around a large square)](https://motchallenge.net/sequenceVideos/MOT17-02-FRCNN-raw.mp4): 1920×1080, 30 fps, about 20 s. | Licence CC BY-NC-SA 3.0 (non-commercial, give credit, share alike); the page says some sequences come from other sources whose conditions may also apply. Full HD, so expect it to run slowly. The site is a static archive and the links could change. I have not downloaded these videos myself. |
| **Your own phone video** | Best match for your real use. Prop the phone still (landscape), mark an area on the floor with tape, and film people walking in, standing, leaving and coming back. | Film only people who agree to it, and do not publish the footage. |

The first two rows are already in `inputs/videos/`. I read the terms for the other sources from their own pages
but have not downloaded or tested those files. Keep large downloads out of the submission folder.

**Trying the Intel video** (it is 1080p at 60 frames per second, so use a playback speed, or the lighter copy
made with the ffmpeg command above, `worker-zone-detection-small.mp4`):

```powershell
python portal.py inputs\videos\worker-zone-detection.mp4 --speed 1
python portal.py inputs\videos\worker-zone-detection-small.mp4 --speed 1
```

With `--speed 1` the 76-second clip played through in about 76 seconds in a real browser (video clock
`0:00` to `1:16`), with the detector managing only 1 to 2.4 pictures per second; the page then showed *Video ended*.

The default zone does not fit this video, so draw one first (*Draw new*). The workers walk close to the camera and
their feet are cut off by the bottom edge of the picture, so the foot point sits on that edge: make the zone reach
the bottom of the picture (for example one of the red-lined floor squares in the lower half) or people will rarely
register inside it. Use *Reset to default* when you go back to `vtest.mp4`.

**"The worker is standing in the zone but nothing is counted."** VisionGuard decides inside or outside from the
point at the bottom-centre of the person's box, roughly their **feet**, not from their body. In this video the
workers are close to the camera, so a zone drawn on the floor farther back appears *behind* them: their bodies and
heads overlap the shaded area, but their feet are still in front of it (and at the bottom edge of the picture), so
nothing is counted. This was seen in a real browser run with a small zone drawn on the floor in the middle of the
picture: two workers overlapped it for several seconds and *Entries* stayed at 0. Make the zone cover the floor
where their **feet** actually are.

**A simple manual test to try on any video:**

1. Start the portal on the video and wait for `Monitoring`.
2. Draw the zone over a floor area people walk across.
3. Watch for the six situations: someone outside the zone (no entry), someone stepping in (+1), someone standing
   inside (no more entries), someone leaving (no entry), the same person coming back (+1 if the ID is the same),
   and two people entering together (+2).
4. Count the entries yourself and compare with the page. Where they differ, look at the tracking IDs on the video:
   a person who gets a new ID, or who is hidden behind someone, is the usual reason. This is how you find the
   limits described below.
5. Open `outputs/events.csv` (or *Download CSV*) and check that each row matches an entry you saw.

### Settings

Edit the constants at the top of `main.py`:

| Constant | Default | Meaning |
|---|---|---|
| `VIDEO_SOURCE` | `0` | Webcam index or video path (a path on the command line overrides it) |
| `CONFIDENCE_THRESHOLD` | `0.4` | Detections below this confidence are dropped (above 0, at most 1) |
| `ZONE_POLYGON` | trapezoid | The built-in zone: corners as `(x, y)` **fractions of the frame width/height** (0–1), at least 3 points, edges must not cross |
| `ZONE_FILE` | `zone.json` | A zone saved from the portal. If this file exists and is valid it is used instead of `ZONE_POLYGON`; delete it, or use **Reset to default**, to go back |
| `ZONE_CONFIRM_FRAMES` | `3` | A change inside/outside must last this many frames to count (`1` = count at once) |
| `COUNT_RULE` | `"feet"` | What counts as inside: `"feet"` (bottom-centre of the box), `"centre"` (middle of the box) or `"overlap"` (any part of the box touching the zone) |
| `COUNT_FIRST_SEEN_INSIDE` | `False` | `True` also counts a person whose ID is first seen already inside the zone (once, after the confirmation window) |
| `MODEL_PATH` | `models/yolov8n.pt` | Model weights |
| `TRACKER` | `bytetrack.yaml` | Ultralytics tracker configuration |
| `EVENTS_CSV` | `outputs/events.csv` | Event log file |

**Zone coordinates** are fractions, so `(0.5, 0.5)` is the centre of any frame and the same setting works
at any resolution. The default trapezoid was chosen for `inputs/videos/vtest.mp4`. **For your own camera
or video, draw the zone in the portal (easiest) or edit `ZONE_POLYGON` to match the view.** Entering pixel
values (for example `(300, 200)`) is caught and reported. A `zone.json` that has been damaged or edited into an
invalid shape is ignored with a message, and the default zone is used.

### How an entry is counted

1. YOLO detects people and the tracker gives each a temporary ID.
2. With the default rule, the bottom-centre of each person's box is tested against the zone polygon (a point
   exactly on the boundary counts as inside). `COUNT_RULE = "centre"` tests the middle of the box instead, and
   `"overlap"` asks whether any part of the box touches the zone (a box that only brushes the boundary counts).
3. The confirmed inside/outside state is remembered per ID. A change is confirmed once it has lasted
   `ZONE_CONFIRM_FRAMES` frames in a row.
4. An **entry** is counted when a confirmed state changes from outside to inside. Staying inside adds
   nothing; leaving and later coming back with the same ID adds another entry.
5. A person whose ID is first seen already inside the zone is not counted, because no outside-to-inside
   move was observed. (With `COUNT_FIRST_SEEN_INSIDE = True` they are counted once instead, after the same
   `ZONE_CONFIRM_FRAMES` window.)

## 9. Sample output

**Screenshots** (frames saved directly from the running application on `inputs/videos/vtest.mp4`):

| File | Shows |
|---|---|
| `docs/screenshots/01_zone_clear.png` | Shaded restricted zone, people detected with IDs, everyone outside: `Entries: 0`, `Zone: CLEAR` |
| `docs/screenshots/02_entry_detected.png` | ID 16 has entered: red box with `INSIDE`, `Entries: 1`, `Zone: OCCUPIED (1 inside)` |
| `docs/screenshots/03_multiple_entries.png` | `Entries: 6`, seven people detected, two currently inside |
| `docs/screenshots/04_non_person_ignored.png` | Ultralytics' `bus.jpg`: the three people are boxed, the bus is not |

**Web portal screenshots** (taken in a real browser while the portal played the sample video, so they contain no
camera image of a person):

| File | Shows |
|---|---|
| `docs/screenshots/portal_live_light.png` | The whole portal: live view with the hatched zone and a `1 INSIDE` tag, entry counter, tiles, both charts, entry log and report card |
| `docs/screenshots/portal_live_dark.png` | The same page in the dark theme |
| `docs/screenshots/portal_star_zone.png` | A concave ten-corner star zone, with the square corner handles and the `+` handles in edit mode |
| `docs/screenshots/portal_drawing_zone.png` | Drawing a new free-form shape corner by corner (five placed so far, the first corner ringed; click it or press `Enter` to finish) |
| `docs/screenshots/portal_entry_stamp.png` | The moment a new entry arrives: the `ENTRY #4 · ID 9` stamp over the video, the message at the bottom, the counter at 4 and the hatched `Inside now` tile. The zone in this image is a free-form 13-corner shape. |
| `docs/screenshots/portal_video_playback.png` | A video file at 1× normal speed: the progress bar and video clock (`0:36 / 1:16`), the **Playback speed** dropdown and its explanation. Two workers overlap the small zone on the floor, but their feet are in front of it, so *Entries* stays at 0 (see the note under *Trying the Intel video*). |
| `docs/screenshots/portal_video_every_frame.png` | The same video with *Every frame* selected: the page states that the 60-frames-per-second clip plays at about 1.5% of normal speed on this computer |
| `docs/screenshots/portal_crossing_refused.png` | A corner dragged so the shape crosses itself: red edges, a warning and a message, and the zone is left unchanged |
| `docs/screenshots/portal_phone.png` | The same page on a phone-width screen |

**Console output** while the sample video runs:

```
ENTRY: ID 16 (total entries: 1)
ENTRY: ID 29 (total entries: 2)
...
ENTRY: ID 91 (total entries: 19)
End of video.
Entries counted this run: 19 (events file: ...\outputs\events.csv)
```

**Event log** (`docs/example_events.csv` is the complete log of that run — 19 rows, 17 different IDs):

```
timestamp,tracking_id,event_type
2026-10-30 18:48:11,16,ENTRY
2026-10-30 18:48:39,29,ENTRY
2026-10-30 18:48:44,22,ENTRY
2026-10-30 18:48:55,12,ENTRY
2026-10-30 18:49:04,9,ENTRY
```

Timestamps are the local clock time when the entry was processed, not positions in the video.

## 10. Testing and results

All results below come from runs on one Windows 10 laptop (CPU only, Python 3.12.7) with the final code.
They are functional checks against **one sample video**, not an accuracy evaluation: no ground-truth count
of who really entered the zone was made.

Run the tests yourself:

```powershell
python -m unittest tests.test_zone_monitoring tests.test_application tests.test_portal tests.test_speed tests.test_counting_rules   # logic, errors, portal, speed, counting rules
python -m unittest discover -s tests                                                                     # everything, incl. the end-to-end test
```

**Result of the latest run (branch `feature/zone-insights`): 289 tests, all passing** (72 in `test_zone_monitoring`,
23 in `test_application`, 125 in `test_portal`, 25 in `test_speed`, 38 in `test_counting_rules`, 6 in
`test_end_to_end`). That run took about 55 seconds; an earlier full run on the previous 196 tests took about 3½
minutes when the laptop was short of free memory. The end-to-end test runs the real model over 300 video frames, so on a slow or
busy machine it can take a few minutes. The unit tests were written before the code they cover; the end-to-end
test was added afterwards to check the pieces work together. The 93 tests added for the zone insights were
written first and seen failing before the code was written; the end-to-end test still passes with the default settings,
which shows the defaults behave as before.

The by-hand checks in the table below (full run on the sample video, real key presses, error messages from the
command line) were done on the code as it stood at the end of Phase 3. `main.py` has since gained the saved-zone
support, the shape-crossing check and optional box colours for the portal; those changes are covered by the tests
above, which were all re-run, but the long by-hand checks were not repeated.

### Required scenarios

| Test | Expected | Result and evidence |
|---|---|---|
| Person outside zone | No entry recorded | **Pass.** Unit tests (scripted people per frame). Sample video: first frames show `Entries: 0`, `Zone: CLEAR` (screenshot 01). |
| Person enters zone | Counter increases | **Pass.** Unit tests. Sample video: ID 16 enters, counter 0 → 1, first CSV row (screenshot 02). |
| Person remains inside | No duplicate count | **Pass.** Unit tests (several frames inside add nothing). Sample video: the log has 19 rows for 17 IDs; the only repeated IDs are 12 and 52 (see re-entry). |
| Person exits zone | No new entry | **Pass.** Unit tests. Sample video: the log contains only `ENTRY` rows and the row count equals the counter, so leaving the zone wrote nothing. |
| Person re-enters | New entry recorded | **Pass.** Unit tests. Sample video: IDs 12 and 52 each left the zone and came back with the same ID, and were logged twice. |
| Multiple people enter | Entries counted individually | **Pass.** Unit tests (3 people, including in the same frame). Sample video: 17 different IDs, 19 entries (screenshot 03). |
| CSV logging | Correct event records saved | **Pass.** Unit tests (header, append, timestamp format, missing folder, write failure) and the end-to-end test. The full run's CSV was checked: correct header, 19 rows, all `ENTRY`, integer IDs, valid non-decreasing timestamps, and identical IDs in identical order to an independent replay of the recorded tracker output. |
| Invalid video path | Clear error displayed | **Pass.** `python main.py inputs\videos\nope.mp4` printed `Video file not found: ...` and exited with code 1, before loading the model. A file that is not a video printed `Could not open video file: ...`. Also unit-tested. |
| Webcam unavailable | Graceful error handling | **Pass.** `python main.py 9` printed `Webcam 9 is unavailable (not connected or in use).` and exited with code 1. Also unit-tested. |
| Application exit | Resources released safely | **Pass.** Real key presses `q`, `Shift+Q` and `ESC` each ended the app with exit code 0, a released capture and a closed window. The full sample video also ended with exit code 0. Unit tests confirm the capture is released and windows are closed after `Ctrl+C`, a crash while processing, a missing package and a model that cannot be loaded. |

### Other checks

- **Boundary flicker (bug fix).** On the sample video the old logic counted one person (ID 63) as entering after
  being inside the zone for only 2 frames. With `ZONE_CONFIRM_FRAMES = 3` that flicker is ignored: the replay of
  the recorded tracker output gives 20 entries with a 1-frame window and 19 with 3 to 8 frames, and the real
  run produced the 19.
- **Input and configuration validation.** Unit-tested: zone with fewer than 3 points, zone in pixels instead of
  fractions, negative or non-numeric coordinates, points on a line, confidence outside 0–1, a bad confirmation
  setting, a source that opens but returns no frames, a model file that cannot be loaded, a missing package
  (clear `pip install -r requirements.txt` message, no traceback).
- **Non-person objects are ignored.** On `bus.jpg` the unfiltered model finds a bus and 3 people; the app
  returns only the 3 people (unit test and screenshot 04).
- **Installation from scratch.** A brand-new virtual environment was built using only `pip install -r
  requirements.txt`: `pip check` reported no problems, nothing extra was installed at run time, and the 69 logic
  and application tests that existed then passed in it (the 6 end-to-end tests were run in the main environment).
  This was done before the web portal was added; the portal uses only the standard library, so the package list
  did not change, but the fresh-environment check was not repeated afterwards.
- **Webcam.** Webcam 0 opens at 640×480. A first automated 40-frame check ran cleanly but nobody was in front of
  the camera. In a later live session with a person in view, the app tracked them, counted 1 entry (tracking ID
  10), wrote the row to `outputs/events.csv`, and exited with code 0 when stopped with the keyboard. That was a
  single short session, run by the project owner; the screen was not recorded, so it is not shown in a
  screenshot, and nothing more is claimed about how accurate the live tracking was. The tracking ID of 10 shows
  the tracker had already issued earlier IDs in that session (see the ID limitation below).
- **Speed.** A full run of the 795-frame sample video took about 100 s in an earlier test (about 8 frames per
  second) and 233 s in the final run (about 3 frames per second). While the laptop was low on free memory
  tracking slowed to about 1 frame per second, and with free memory the 300-frame end-to-end test finished in
  well under 35 s (roughly 10 frames per second or better). It is **not real-time**, and the speed depends heavily
  on the machine and its free memory.

### Web portal checks

- **Automated tests (52 in `tests/test_portal.py`, all passing).** They use scripted frames and detections so
  they are fast, but everything else is real, including the HTTP server on a free port, the JPEG video stream and
  the CSV file. They cover: counting entries, exits and re-entries; the live numbers and peaks; the history and
  entries-per-minute buckets; changing the zone (the new shape is used, saved and reported, people already inside
  are not counted as entering, invalid shapes are refused and leave the old zone in place, reset to default);
  the monitor thread (video ending, camera missing, a webcam that stops sending frames, a crash while processing,
  pause, restart, stop); the report page and CSV (including that the source name is escaped); and the security
  checks (a wrong `Host`, a request from another website, a request that is not JSON, broken JSON and oversized
  requests are all refused).
- **Real browser (30 of 31 checks passed in the main run; the 31st was confirmed in a separate run).** I drove the portal in headless Microsoft Edge with real mouse and
  keyboard events while it played the sample video, and took the screenshots from that run. Passing: the page
  loads and shows `Monitoring`; the video stream is displayed; the entry counter and entry log follow the server;
  both charts draw; the light/dark switch works; `E` enters edit mode; dragging a corner, dragging a `+` handle,
  dragging inside the shape and double-clicking a corner each change the zone as described and the server
  reports the new shape; the star (10 corners) and L-shape presets are accepted; `Z` and `Y` undo and redo; `D`
  and clicks draw a five-corner shape that is saved; dragging a corner so the shape crosses itself is **not**
  applied and the page explains why; `Esc` leaves edit mode; typing a number moves a corner; the size slider
  shrinks the zone; *Reset to default* restores it; Pause and Resume work; a phone-width screen has no sideways
  scrolling; and the page logged no JavaScript errors. The one check that could not be confirmed in that run was
  the animated `ENTRY` stamp over the video: the test browser reports "reduce motion" as switched on, so the page
  skips the animation by design and shows a message instead (the message did appear). **A second run with
  reduced motion forced off confirmed the stamp**: it started playing when entry #1 arrived and was caught at
  full opacity for entry #4 (`docs/screenshots/portal_entry_stamp.png`), so all 31 checks are now accounted for.
  The browser scripts are not part of the project folder. These checks used the sample video, not the webcam.
- **Bugs found and fixed while building the portal.** (1) A bow-tie shape was reported as "no area" instead of
  "crosses itself" because its two halves cancel out; found by a unit test and fixed. (2) The first colour choices
  for the charts failed the colour checks (one blue too grey, the dark-mode colours too light); they were changed
  until the palette passed all five checks (lightness, colourfulness, colour-blind separation, separation for
  normal vision, contrast) in both light and dark mode. (3) Typing a number that would make the shape cross itself
  was sent to the server before being refused; the page now catches it first (this new check was not run again
  in the browser). (4) The video frame could have been
  stretched on tall screens, which would misalign the zone; the frame now keeps its true proportions.
  (5) **Printed report page break.** A report saved as PDF by the project owner (A4 pages) had its last heading
  ("How to read these numbers") stranded at the bottom of page 1 with its text alone on page 2. I reproduced it by
  printing a report of the same shape on A4 from a browser, then fixed it: each heading and its content now form
  a block that is not split across pages, a long entry log may run over several pages but its rows are never cut
  and its column titles repeat on each page. After the fix, the same A4 print has the heading and text together,
  and a 45-entry report runs cleanly over three pages. This was checked on the printed pages (A4) and by two new
  tests; it was not checked on other paper sizes or other browsers.
- **Zone insights (counting rule, manual check, heat-map).** Automated: 38 tests for the rules in `main.py` (the
  box-overlap geometry including concave zones, each rule on the same people, the first-seen switch with the
  confirmation window and flicker, validation, and that the defaults are unchanged) and 55 more in
  `tests/test_portal.py` (rule changes not counting anyone already in view, the manual comparison and its wording,
  heat-map cells and edge clamping, the new HTTP endpoints, the report). By hand, in a real browser (Chrome) on
  `worker-zone-detection-small.mp4` with the saved zone: the heat-map toggle drew a trail along the path the
  worker walked, the legend showed, choosing *Any overlap* changed the server's rule, the controls and showed a
  message, typing 3 in the manual check showed `0 vs 3, difference -3`, the report page showed the rule sentence,
  the manual check and the *Where people walked* picture, and the page logged no JavaScript errors. This was one
  short session, not the full checklist. **Not measured:** how many more real entries the other rules find. I did
  not count the people in any video, and I did not replay a video under the other rules. In that run the saved zone
  lay away from where the worker walked (which is what the heat-map showed) and the default rule counted 0 entries;
  *Any overlap* was only switched on after the video had ended, so it was not tried on the footage.
- **Video playback speed.** Tested with fake time (so a long video "plays" instantly): at normal speed a slow
  detector skips the right number of frames and a fast one waits, double speed skips more, speed 0 uses every
  frame, pausing does not make the video race ahead afterwards, and a webcam refuses a speed. In a real browser on
  the 1080p, 60 fps Intel clip: **before** the option existed, the page showed about 1.8 new pictures per second,
  which is about 3% of normal speed (after 100 s on the page's clock the video had reached about 3 s, so the
  counters still read 0); **with `--speed 1`** the video clock followed the real clock (`0:05` at 10 s, `0:36` at
  42 s, `1:13` at 78 s), the clip finished in about 83 s and the page showed *Video ended*. Changing the speed
  from the page's dropdown (4×, every frame, 2×) was checked in the browser: the server changed speed each time,
  the explanatory text updated, and the page logged no JavaScript errors (8 of 8 checks). The detector itself
  stayed slow (1 to 2.4 pictures per second); the option only keeps the video on time by skipping frames.
- **Live webcam.** The portal started on the webcam (640×480) and showed `Starting` for about a minute while the
  laptop had only about 0.8 GB of free memory, then switched to `Monitoring` at about 1.8 frames per second with
  one person detected. The project owner then used it on their own webcam and reported that it works; no further
  detail was recorded and no screenshot of it is included.

## 11. Limitations and future improvements

### Limitations

- **Tracking IDs are temporary and can change.** The tracker can lose a person (occlusion, people crossing or
  walking close together, leaving and re-entering the frame) and give them a new ID. A new ID is counted as a
  new person if it first appears outside and then enters, and is not counted if it first appears inside.
  Identity continuity is **not guaranteed**.
- **Counts are not guaranteed accurate.** The tracker produced 33 IDs on the sample video (4 of them visible for
  fewer than 10 frames). I did not count the real number of people, so I cannot say how many IDs are duplicates
  or how many entries are real. "19 entries" is what the system output, not a verified count.
- **The confirmation window is a trade-off.** `ZONE_CONFIRM_FRAMES = 3` removes one- and two-frame flicker at the
  zone edge, but it also ignores a genuine crossing shorter than 3 frames, and counts an entry 2 frames late. On a
  slow machine 3 frames last longer in real time. Set it to `1` for immediate counting.
- **First seen inside is not counted (by default).** Keep the zone away from places where people first appear (for
  example frame edges). The tracker may need a frame or more before it gives a new person an ID. The
  *first-seen-inside* switch counts them, but it cannot tell a new person from a known person who got a new ID
  while standing inside, so it can count one person twice; it also counts everyone who is inside when monitoring
  starts.
- **The other counting rules trade accuracy for sensitivity.** *Any overlap* counts people who only brush past the
  zone edge or whose box reaches into the zone (an arm or a bag), and a large box near the camera touches more.
  *Middle of the body* assumes the box centre is meaningful, which fails for a sitting or partly hidden person.
  None of the rules was measured against a real count.
- **The heat-map is a rough picture.** It uses a 64 × 48 grid, so it is coarse; it counts foot positions per frame
  (a person standing still stands out); it shows nothing until people have been seen; and frames skipped by the
  playback speed are not in it. It is not a measure of how safe or busy an area is.
- **The manual check is only as good as the person counting.** It compares two totals; it does not match
  individual entries.
- **The default zone fits the sample video only.** Draw your own zone in the portal (or edit `ZONE_POLYGON`) for
  any other camera view. Shapes whose edges cross or touch are refused; a very thin zone is allowed but people
  will rarely register inside it.
- **The foot point assumes the feet are visible.** A person cut off at the bottom of the frame, a sitting person
  or a steep camera angle shifts the reference point.
- **Timestamps are wall-clock time of processing**, not video time, and processing is slower than playback.
- **Overlapping labels** when people stand close together (the portal uses shorter labels to reduce this), and the
  status panel in the OpenCV window covers the top-left of the video.
- **Portal limits.** It is for one person on one computer and has no login; it was checked in one browser
  engine (Microsoft Edge/Chromium) and by one live webcam session run by the project owner, not on other
  browsers or phones. The video stream is a plain motion-JPEG image, so on a slow computer it is as slow as the
  detection (see Speed). The animated "ENTRY" stamp over the video is skipped when the computer asks for reduced
  motion; a message appears instead. It was checked with a screenshot in a test browser, not by eye on a live
  webcam.
- **The portal's numbers reset, the CSV does not.** *Reset counters* and *Restart source* start a new session on
  the page, but `outputs/events.csv` keeps every entry from every session, so it can hold rows from earlier runs.
- **Slow on a busy machine.** At 1–3 frames per second people move further between frames, which may make
  tracking less reliable; this was not measured.
- **The window closes when a video ends**, so the final count is only visible in the console and the log.
- **Confidence is not accuracy.** It is the model's estimate for one box, not how often the system is right.
- **Environment.** Specified for Python 3.11 but tested only on 3.12.7; Windows only; only MP4 video was tested;
  only the first command-line argument is used; live webcam use was tested in one short session only.
- Results from one sample video do not show reliable behaviour in other lighting, camera angles or crowds.

### Possible future improvements

- Keep the last frame on screen at the end of a video in the OpenCV window (the portal already does this).
- A tracker with appearance matching to reduce ID changes, and a labelled test video to measure real accuracy.
- Video-time timestamps and saving an annotated output video.
- Checking the portal on more browsers and on a phone, and adding a low-memory warning when the detection is
  running very slowly.

## 12. Out of scope

One zone and one video source only. No multiple zones or cameras, face recognition, database, cloud
deployment, notifications, mobile app, accounts or model training.

The SRS and PRD allowed an optional, *basic* browser interface (they named Streamlit) and the design notes asked
for no graphical zone editor, no themes or complex charts, no automatic report generation and no backend API.
The web portal was added afterwards **at the project owner's request** and goes beyond that in five ways: a
graphical zone editor, a session report page, a light/dark theme with two small charts, a small local HTTP
interface used only by its own page, and the use of Python's built-in web server instead of Streamlit. It stays
inside the other limits (one source, one zone, local only, no database, no login, no new packages), is separate
from `main.py`, and `main.py` works without it. The change is recorded in the project documents: as tracked
changes (an "Appendix A. Change note") at the end of `docs/VisionGuard_SRS.docx` and
`docs/VisionGuard_PRD.docx`, and in `docs/architecture.md`, `design.md`, `phases.md` (Phase 4), `rules.md`,
`skills.md` and `role.md`.

## 13. Submission checklist

Include: `main.py`, `portal.py`, `portal_ui/`, `requirements.txt`, `README.md`, `models/`, `inputs/`, `tests/`,
`docs/` (including `docs/example_events.csv` and `docs/screenshots/`). Leave out `.venv/` (recreate it with the
installation steps), `outputs/` (created when the program runs) and `zone.json` (your personal zone; delete it
so the submitted project starts with the default zone). `inputs/videos/worker-zone-detection.mp4` is an optional
extra from Intel's sample-video collection; leave it out of the submission if you are unsure about its licence.
