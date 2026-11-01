# Design --- VisionGuard

## Design Goal

Create a clear, minimal academic demonstration of person detection and
restricted-zone entry monitoring. Prioritize visible, understandable
behaviour over visual polish.

There are two screens. The OpenCV window (`main.py`) is the core, minimal
design and follows the conventions below as written. The web portal
(`portal.py`) is an optional, owner-approved extra with a richer design,
described under "Web Portal Screen".

## Main Screen (OpenCV window)

The application should show: - The selected webcam or video feed. - One
clearly outlined restricted polygon. - Person bounding boxes and
confidence labels. - A temporary tracking ID for each detected person. -
A visible total entry count. - A simple monitoring status: the panel
shows `VisionGuard | MONITORING` while running (the window closes when
monitoring stops).

## Visual Conventions (OpenCV window)

-   Use a consistent colour for the restricted-zone boundary (red,
    lightly shaded).
-   Use a contrasting colour for person bounding boxes (green outside the
    zone, red inside it).
-   Display labels in a readable size, on a filled background.
-   Keep the entry count separate from detection labels (status panel at
    the top-left).
-   Avoid decorative panels, animations, themes, login screens, and
    complex charts.

## Interaction (OpenCV window)

1.  Select webcam or local video (command-line argument).
2.  Start monitoring.
3.  Observe detections and zone events.
4.  Stop monitoring (`Q`, `ESC` or close the window).
5.  Inspect the generated `events.csv` file.

## Zone Configuration

-   Use one polygon for one camera/video view.
-   The default polygon is stored in one clearly named configuration
    variable (`ZONE_POLYGON`), as fractions (0 to 1) of the frame size.
-   A zone saved from the portal (`zone.json`) replaces the default; the
    OpenCV window uses it too.
-   The OpenCV window has no zone editor. Zones are edited in the portal.
-   A zone must have three or more corners, stay inside the picture, and
    must not cross or touch itself. Concave shapes are allowed.

## Event Feedback

When a tracked person moves from outside to inside the zone (and stays
there for the confirmation window): - Increase the entry count once for
that transition. - Show the new count on screen. - Append one row to
`events.csv`.

A person remaining inside must not trigger a new event every frame.

## Accessibility and Clarity

-   Use concise labels such as `Person`, `ID`, `Entries`, and
    `Monitoring`.
-   Ensure text and boundary lines are visible against the video.
-   Do not rely on colour alone; include labels or status text (people
    inside the zone carry an `INSIDE` tag as well as a different colour).

## Web Portal Screen (optional)

Goal: let one person watch the camera, shape the zone to fit what the
camera sees, and read the numbers without opening the terminal.

**Look.** A light, drafting-table feel: pale paper background with a faint
grid, deep ink text, and one signal colour taken from restricted-area
signage (safety orange) used for the zone, people inside the zone and the
"inside" series in charts. Lettering is DIN-style signage type
(Bahnschrift, with fallbacks) for headings and a monospace face for
numbers. A dark theme is available. The zone is drawn as hazard-hatched
tape that moves only while someone is inside it.

**Layout.** Left column: the live view with its controls, the zone tools
and an entry log. Right column: the entry counter with four tiles (people
inside now, in view, most at once, tracking IDs), a three-minute activity
chart, an entries-per-minute chart and the session report card. On a
narrow (phone-width) screen the columns stack.

**Zone editing.** Edit shape (move corners, drag `+` handles to add
corners, drag inside to move the shape, double-click a corner to remove
it), Draw new (click out any shape), a menu of ready-made shapes, a size
slider, a table to type exact percentages, Undo and Redo, and Reset to
default. Every change applies to the next frame and is saved. A shape that
crosses itself is shown with red edges and a message and is not applied.

**Video files.** When the source is a video file, a strip under the video
shows a progress bar with the video's own clock (for example `0:36 /
1:16`, separate from the page's Session clock) and a Playback speed
dropdown (every frame, 0.5x, 1x, 2x, 4x, 8x). A line of text says what
the current choice means: for "every frame" it states how slowly the
video really plays on this computer; for a speed it warns that skipped
frames can make a very fast walker crossing a thin zone be missed. The
strip is hidden for a webcam. When the video ends the page says "Video
ended" and offers Play again.

**Zone insights.** Three optional aids that help place the zone and judge
the count; none of them changes anything until the person uses it.

-   *Foot heat-map.* A button above the video (key `H`) shades the picture
    from blue (less time) to orange (more time) by where people's feet
    have been, with a small colour bar ("Less time" to "More time") and a
    one-line note that only counts per area are kept, with no picture and
    no path of any person. It is drawn underneath the zone and the boxes,
    so the zone and labels stay readable. The shading is a hint, not a
    measurement: it is coarse and says nothing until people have been
    seen.
-   *What counts as inside.* A card under the zone tools with three
    choices, each with a one-line plain explanation: *Feet* (the default),
    *Middle of the body* and *Any overlap*, plus a tick box *Also count
    people who first appear already inside* (off by default, with its
    drawback written beside it). The chosen rule is shown as a short label
    in the card title and in the report. A small dot on each person shows
    the point being tested (the feet point, or the middle of the box); with
    *Any overlap* there is no dot, because the whole box counts. The same
    dot appears in the OpenCV window.
-   *Check against your count.* A card between the counter tiles and the
    activity chart: one number box and a Compare button. The answer is
    three figures (You counted, System counted, System minus you) and one
    plain sentence such as "The system counted 2 fewer than you (5 vs 7),
    29% below your count." A fixed caution says it compares two totals, is
    not an accuracy score, and that one video says little about other
    cameras. Clear removes it. The same figures and caution appear in the
    report.

**Colour and charts.** The two chart colours were checked with the data
visualisation palette validator (lightness, colourfulness, colour-blind
separation, normal-vision separation and contrast) in both themes. Counts
are drawn as step lines because they are whole numbers. Both charts have
a hover read-out; the activity chart also has a legend, direct labels at
the end of each line and a table view.

**Accessibility.** Real buttons with visible keyboard focus, keyboard
shortcuts (`E`, `D`, `Z`, `Y`, `H`, `Space`, `Esc`, `Del`), an alternative
numeric way to edit every corner, status shown by text as well as colour,
and animations that switch off when the system asks for reduced motion
(a message appears instead).

**Feedback.** A toast message for saves, undo, errors and each new entry;
the entry counter, the entry log and the zone label update live. A banner
appears if the page loses contact with the program.

## Out of Scope

No mobile app, user accounts, multi-camera control, remote access,
custom branding system, advanced analytics, or production-security
interface. (The portal page adapts to a phone-width screen, but it is
still a local page, not a mobile app, and is not meant for remote use.)
