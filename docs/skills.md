# Skills --- VisionGuard

## Purpose

This file defines the practical skills and implementation knowledge
needed to build and maintain VisionGuard. The project is a small
academic prototype; use the simplest reliable approach.

## Required Skills

1.  **Python fundamentals**
    -   Functions, modules, exceptions, file paths, and virtual
        environments.
    -   Readable code with small, focused functions.
2.  **OpenCV**
    -   Open webcam/video files with `cv2.VideoCapture`.
    -   Read and display frames.
    -   Draw bounding boxes, labels, polygons, and counters.
    -   Release the video source and close windows safely.
3.  **YOLO with Ultralytics**
    -   Load a small pretrained model.
    -   Run inference on video frames.
    -   Filter detections to the `person` class.
    -   Use the tracking API to obtain temporary track IDs.
    -   Understand confidence thresholds and basic model performance
        trade-offs.
4.  **Basic geometry**
    -   Represent the restricted zone as a polygon.
    -   Use a point-in-polygon check, such as OpenCV's
        `pointPolygonTest`.
    -   Use the bottom-centre of a person's bounding box as the
        reference point.
5.  **CSV file handling**
    -   Create a CSV file with a header when needed.
    -   Append event timestamp, tracking ID, and event type.
    -   Handle file-write errors clearly.

## Optional Skills

-   **Basic testing/debugging:** test valid and invalid video sources,
    zone entry, remaining inside, exit, and re-entry.
-   **Web portal (optional extension):** Python's built-in `http.server`
    and `threading`; plain HTML, CSS and JavaScript; the HTML canvas and
    pointer events for the zone editor; SVG for charts; JPEG
    (motion-JPEG) video streaming; basic localhost security (check the
    `Host` and `Origin` headers). Streamlit was considered in the first
    plan and not used.
-   **Testing a web page:** `unittest` with a real HTTP server on a free
    port for the interface, and a scripted real browser (mouse and
    keyboard events) for the page itself.

## Implementation Guidance

-   Prefer built-in capabilities of the chosen libraries over custom
    algorithms.
-   Do not train or fine-tune a model.
-   Do not introduce a database, authentication, cloud services, or
    notifications. Do not introduce a separate API server; the only
    exception is the small JSON interface inside the optional web
    portal, which listens on this computer only and is used only by the
    portal's own page.
-   Keep the project runnable locally on a standard Windows laptop.
-   If a feature threatens the 2--3 day timeline, omit it and document
    the limitation.

## Definition of Done

A feature is complete when it runs, has been tested with a relevant
example, handles its basic failure case, and does not break the existing
demo.
