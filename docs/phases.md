# Phases --- VisionGuard

## Delivery Target

Complete a demonstrable academic prototype in 2--3 days. Work in the
order below. Do not start optional interface work before the core
pipeline is functional.

## Phase 1 --- Environment and Video Detection

**Target:** Day 1

Tasks: - Install Python and create a virtual environment. - Install
Ultralytics, OpenCV, and NumPy. - Load a small pretrained YOLO model. -
Open a local sample video or webcam. - Detect people and draw bounding
boxes and confidence labels.

Exit criteria: - A video source opens successfully. - At least one
visible person can be detected in a suitable test clip. - The
application can be stopped cleanly.

**Status: done.**

## Phase 2 --- Zone Monitoring and Event Logging

**Target:** Day 2

Tasks: - Define one polygonal restricted zone. - Add object tracking and
temporary IDs. - Calculate each person's bottom-centre reference
point. - Check point-in-polygon membership. - Detect outside-to-inside
transitions. - Increment the entry counter once per transition. - Append
timestamp, tracking ID, and `ENTRY` to `events.csv`.

Exit criteria: - The zone is visible. - A controlled entry causes one
count increment and one CSV row. - Remaining inside does not repeatedly
increment the count.

**Status: done.**

## Phase 3 --- Testing and Submission Preparation

**Target:** Day 3

Tasks: - Test person outside zone, entry, remaining inside, and
exit/re-entry. - Test invalid video path and stop behaviour. - Fix
critical issues only. - Add minimal status/count display. - Add
Streamlit only if all core requirements are already working. - Capture
screenshots and write concise run instructions. - Prepare a short
demonstration.

Exit criteria: - The project runs from documented instructions. - Core
acceptance tests pass on the chosen demo video. - Source code,
dependency list, event log example, and screenshots are ready.

**Status: done.** Streamlit was not used (see Phase 4). Fixes made in this
phase: a short confirmation window against zone-edge flicker, input and
configuration checks, and safe release of resources on every exit path.

## Phase 4 --- Optional Web Portal (owner-approved extension)

**Added after Phase 3, at the project owner's request.** This phase is
outside the original 2--3 day plan and the SRS/PRD baseline; it is
recorded in the change notes at the end of the SRS and PRD. The core
application (`main.py`) does not depend on it.

Tasks: - Let the user draw and reshape the zone with the mouse (any
simple polygon) and save it. - Serve a local page with the live view, the
zone, live numbers, two charts, an entry log and a session report. - Keep
it local-only, with no new packages. - Test it, including in a real
browser. - Add playback-speed control for video files (`--speed` and a
dropdown with a progress bar), after a real-browser run on a 1080p, 60 fps
clip showed it playing at about 3% of normal speed.

Exit criteria: - The portal runs on a webcam or a video and shows live
numbers. - A zone change applies on the next frame and is saved; a shape
that crosses itself is refused. - Entries are still written to
`events.csv`. - The core application and its tests are unaffected.

**Status: done, with limits.** Automated tests cover the logic and the
HTTP interface; the page was checked in a real browser on the sample
video (30 of 31 checks in the main run; the 31st, the animated entry
stamp, was confirmed in a second run with reduced motion switched off)
and in one short session on the owner's webcam. See the README for
details.

## Phase 5 --- Zone Insights (owner-approved extension)

**Added after Phase 4, at the project owner's request. It was built on the
separate branch `feature/zone-insights` so the working `main` code stayed
as it was while it was tested, and merged into `main` on 2026-11-01.**
Not needed for the SRS acceptance criteria, so it is optional and kept
apart from the core flow: every new setting defaults to the original
behaviour.

Tasks: - **Counting rule:** choose what counts as inside (feet, middle of
the body, any overlap of the box) and whether a person first seen inside
is counted; shared by `main.py` and the portal. - **Manual-count check:**
type the number of entries counted by eye; show the difference from the
system's count in the page and the report, with a caution that it is not
an accuracy score. - **Foot heat-map:** an optional overlay and a report
picture of where feet have been, to help place the zone; anonymous
counts only. - Test first, then check the page in a real browser.

Exit criteria: - With default settings, behaviour and the existing tests
are unchanged. - A zone or rule change never counts people already in
view. - The report states the rule in use. - No footage and no
per-person path is kept.

**Status: built, tested and merged into `main` (2026-11-01).** 289
automated tests pass on the merged `main` (93 new). Checked in a real
browser on the worker video, and once more on 2026-11-01 on an export of
the same code playing the sample video (heat-map, rule switch and manual
check all worked). The other counting rules were not replayed against a
real count (see the README).

## Priority Rule

If time is running out, deliver Phase 1 and Phase 2 reliably. Skip
optional dashboard styling and all features outside the MVP.

## Change Control

Any proposed feature must be assessed against the 2--3 day limit. If it
is not necessary for the SRS acceptance criteria, defer it.

Change record:

-   **2026-10-30 --- Optional web portal (Phase 4).** Requested by the
    project owner. Not needed for the SRS acceptance criteria, so it is
    optional and separate from the core application. Goes beyond the SRS
    and PRD in four ways: a graphical zone editor, a session report page,
    a light/dark theme with two small charts, and a small local HTTP
    interface. Kept within the other limits: one source, one zone, local
    only, no database, no login, no new dependencies. Recorded as
    tracked-change notes in `VisionGuard_SRS.docx` and
    `VisionGuard_PRD.docx`.
-   **2026-11-01 --- Zone insights (Phase 5).** Requested by the project
    owner after testing showed two gaps: a zone drawn where nobody
    actually walks, and people who appear inside the zone or overlap it
    without their feet inside. Adds a choice of counting rule
    (including counting a person first seen inside), a manual-count check
    and a foot heat-map. Goes beyond the SRS and PRD baseline and
    relaxes Technical Rules 4 and 5 and Scope Rule 6 only as opt-in
    options; defaults are unchanged. Built on branch
    `feature/zone-insights` and merged into `main` on 2026-11-01.
    Recorded, together with the playback-speed control, as tracked-change
    notes in `VisionGuard_SRS.docx` and `VisionGuard_PRD.docx`
    (Appendix A).
