# Rules --- VisionGuard

## Scope Rules

1.  Build only the features listed as in-scope in the SRS and MVP in the
    PRD.
2.  Support one video source at a time and one predefined restricted
    zone.
3.  Detect people only; do not add other object classes unless
    explicitly approved.
4.  Use pretrained model weights. Do not train or fine-tune a model.
5.  Keep processing local.
6.  One owner-approved extension exists: the optional web portal
    (`portal.py`, see Phase 4 in `phases.md`). It must stay local-only
    (`127.0.0.1`), use no new dependencies, remain separate from the
    core application, and never change the counting rules on its own.
    It does not relax any other rule in this file.
7.  One further owner-approved extension exists: the zone insights
    (Phase 5 in `phases.md`): a choice of counting rule, a manual-count
    check and a foot heat-map. They must default to the original
    behaviour, stay local, use no new dependencies and keep no footage
    and no per-person path.

## Technical Rules

1.  Use Python, OpenCV, Ultralytics YOLO, and NumPy as the core stack.
2.  Use a small pretrained model suitable for CPU execution.
3.  Use tracking IDs for entry-transition counting; raw per-frame
    detections are not sufficient for unique transition events.
4.  Use the bottom-centre of the bounding box as the zone reference
    point. (Default. The owner-approved options `"centre"` and
    `"overlap"` in `COUNT_RULE` are opt-in; see Phase 5.)
5.  Count only an observed outside-to-inside transition. (Default. The
    owner-approved switch `COUNT_FIRST_SEEN_INSIDE`, off by default, also
    counts a person first seen inside, once; see Phase 5.)
6.  Write only the required event fields to CSV: timestamp, tracking_id,
    event_type.
7.  Release the video capture and close OpenCV windows on normal stop
    and errors.
8.  Keep configuration values, such as model name, confidence threshold,
    source, and polygon coordinates, easy to find.
9.  Check configuration and input before running (zone shape, confidence,
    confirmation window, video path, webcam) and report problems with a
    clear message.
10. Accept a zone only if it has three or more corners, stays inside the
    picture and does not cross or touch itself.
11. Portal requests must be refused unless they are addressed to
    localhost and, for changes, come from the portal's own page.
12. Analyse every frame by default. Skipping frames to keep a video on
    time (`--speed`, the portal's Playback speed) must be an explicit
    choice, apply to video files only, and be documented as a trade-off.
13. Anything that changes how entries are counted must default to the
    original behaviour, be validated like the other settings, and be
    stated on the session report. A change of zone or rule must never
    count the people who are already in view.

## Code Quality Rules

-   Prefer small functions with clear names.
-   Avoid duplicated logic and unnecessary abstractions.
-   Handle expected failures with clear messages.
-   Do not hard-code machine-specific absolute paths.
-   Keep dependencies limited to those actually used.
-   Do not commit the virtual environment or large generated files
    unless required by the college.

## Honesty and Testing Rules

-   Never claim guaranteed accuracy, real-time performance, or
    production readiness.
-   Record actual test outcomes; do not fabricate metrics.
-   Distinguish detection confidence from system accuracy.
-   A successful demo on one video does not prove reliable performance
    in all environments.
-   Document tracking-ID loss, occlusion, lighting, and camera-angle
    limitations.

## Documentation Rules

-   Update documentation when implementation changes.
-   Do not mark planned features as completed.
-   Keep setup commands reproducible.
-   Keep screenshots and sample logs representative of actual runs.

## Explicitly Prohibited Scope Creep

Do not add face recognition, identity tracking, authentication,
multi-camera support, cloud hosting, databases, email/SMS/WhatsApp
alerts, mobile apps, advanced analytics, or custom model training for
this submission. The web portal does not change this list: it has no
login, is not a mobile app and is not for remote use.
