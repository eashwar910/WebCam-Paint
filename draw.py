"""AI Finger-Paint: webcam + hand-tracking virtual paint app.

Controls (no mouse/keyboard needed after launch):
  - Point with index finger to draw.
  - Pinch (index tip + thumb tip together) to click UI buttons.
  - Open palm over the canvas to erase.
  - Press 'q' or Esc to quit.
"""

import os
import time
import urllib.request
from collections import deque

import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# ---------------------------------------------------------------------------
# Constants / configuration
# ---------------------------------------------------------------------------

FRAME_W, FRAME_H = 1280, 720
TOPBAR_H = 100

# The drawing box: painting only happens inside this rectangle. The toolbar
# panel lives above it (outside the box), so moving the cursor up to pick a
# color never leaves a stray stroke.
CANVAS_MARGIN = 20
CANVAS_RECT = (
    CANVAS_MARGIN,
    TOPBAR_H + CANVAS_MARGIN,
    FRAME_W - CANVAS_MARGIN,
    FRAME_H - CANVAS_MARGIN,
)

# Webcam device index. 1 selects the built-in MacBook camera (0 was grabbing
# the iPhone Continuity Camera).
CAMERA_INDEX = 1

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

CURSOR_COLOR = (0, 255, 255)     # bright cyan-yellow, always distinct from strokes
HOVER_COLOR = (255, 255, 255)
PINCH_COLOR = (0, 140, 255)

PINCH_THRESHOLD = 0.55           # pinch distance, relative to hand size
SMOOTHING_WINDOW = 5             # frames averaged for cursor/stroke position

BRUSH_SIZES = {"S": 4, "M": 10, "L": 20}
ERASER_MULT = 3                  # eraser thickness = brush size * this
PALM_ERASE_RADIUS = 55

COLORS = [
    ("Black", (30, 30, 30)),
    ("White", (255, 255, 255)),
    ("Red", (0, 0, 255)),
    ("Green", (0, 200, 0)),
    ("Blue", (255, 80, 0)),
    ("Yellow", (0, 220, 255)),
    ("Cyan", (255, 255, 0)),
    ("Magenta", (255, 0, 220)),
]

FINGER_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_PIPS = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}
THUMB_TIP, THUMB_MCP = 4, 2
WRIST, MIDDLE_MCP = 0, 9


# ---------------------------------------------------------------------------
# UI model
# ---------------------------------------------------------------------------

class Button:
    """A rectangular UI element that can be hovered and pinch-clicked."""

    def __init__(self, rect, kind, value, label="", swatch_color=None):
        self.rect = rect  # (x1, y1, x2, y2)
        self.kind = kind  # "tool" | "color" | "size" | "clear"
        self.value = value
        self.label = label
        self.swatch_color = swatch_color

    def contains(self, x, y):
        x1, y1, x2, y2 = self.rect
        return x1 <= x <= x2 and y1 <= y <= y2


def build_ui():
    """Lay out all toolbar buttons and return them as a flat list."""
    buttons = []

    # Tool buttons: Pen, Fill, Eraser
    tools = ["Pen", "Fill", "Eraser"]
    x = 10
    for name in tools:
        rect = (x, 10, x + 90, 90)
        buttons.append(Button(rect, "tool", name.lower(), label=name))
        x += 100

    # Color swatches
    x += 20
    for name, color in COLORS:
        rect = (x, 10, x + 50, 60)
        buttons.append(Button(rect, "color", color, label=name, swatch_color=color))
        x += 60

    # Brush size buttons
    x += 20
    for name, size in BRUSH_SIZES.items():
        rect = (x, 10, x + 60, 90)
        buttons.append(Button(rect, "size", size, label=name))
        x += 70

    # Clear canvas button (pinned to the right edge)
    clear_w = 150
    rect = (FRAME_W - clear_w - 10, 10, FRAME_W - 10, 90)
    buttons.append(Button(rect, "clear", None, label="Clear All"))

    return buttons


def draw_ui(display, buttons, state, cursor_pt):
    """Render the toolbar, highlighting whichever button the cursor hovers."""
    overlay = display.copy()
    cv.rectangle(overlay, (0, 0), (FRAME_W, TOPBAR_H), (25, 25, 25), -1)
    cv.addWeighted(overlay, 0.75, display, 0.25, 0, dst=display)

    hover_x, hover_y = cursor_pt if cursor_pt else (None, None)

    for btn in buttons:
        x1, y1, x2, y2 = btn.rect
        hovering = cursor_pt is not None and btn.contains(hover_x, hover_y)

        if btn.kind == "tool":
            selected = state["tool"] == btn.value
            base = (90, 90, 90) if not selected else (0, 120, 220)
            cv.rectangle(display, (x1, y1), (x2, y2), base, -1)
            border = HOVER_COLOR if hovering else (0, 0, 0)
            cv.rectangle(display, (x1, y1), (x2, y2), border, 3 if hovering else 1)
            cv.putText(display, btn.label, (x1 + 8, y1 + 45),
                       cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        elif btn.kind == "color":
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            radius = (x2 - x1) // 2
            cv.circle(display, (cx, cy), radius, btn.swatch_color, -1)
            selected = state["color"] == btn.swatch_color
            ring_color = (0, 255, 0) if selected else (200, 200, 200)
            ring_thick = 3 if (selected or hovering) else 1
            cv.circle(display, (cx, cy), radius, ring_color, ring_thick)

        elif btn.kind == "size":
            selected = state["brush_size"] == btn.value
            base = (90, 90, 90) if not selected else (0, 120, 220)
            cv.rectangle(display, (x1, y1), (x2, y2), base, -1)
            border = HOVER_COLOR if hovering else (0, 0, 0)
            cv.rectangle(display, (x1, y1), (x2, y2), border, 3 if hovering else 1)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 + 10
            preview_r = max(2, btn.value // 2)
            cv.circle(display, (cx, cy), preview_r, (255, 255, 255), -1)
            cv.putText(display, btn.label, (x1 + 18, y1 + 22),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        elif btn.kind == "clear":
            base = (0, 0, 160)
            cv.rectangle(display, (x1, y1), (x2, y2), base, -1)
            border = HOVER_COLOR if hovering else (0, 0, 0)
            cv.rectangle(display, (x1, y1), (x2, y2), border, 3 if hovering else 1)
            cv.putText(display, btn.label, (x1 + 10, y1 + 45),
                       cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


INSTRUCTIONS_TIMEOUT = 12.0  # seconds the first-launch popup stays up


def draw_instructions_overlay(display, elapsed, dismissed):
    """Show the first-launch instruction popup until dismissed or timed out."""
    if dismissed or elapsed > INSTRUCTIONS_TIMEOUT:
        return
    title = "AI Finger-Paint"
    lines = [
        "Pinch index finger + thumb together to select a tool or color.",
        "Point with your index finger inside the box to draw.",
        "Wave your open palm over the box to erase.",
    ]
    remaining = max(0, int(INSTRUCTIONS_TIMEOUT - elapsed) + 1)
    footer = f"Pinch anywhere to dismiss  (auto-closes in {remaining}s)"

    box_w, box_h = 940, 240
    x1 = (FRAME_W - box_w) // 2
    y1 = (FRAME_H - box_h) // 2
    overlay = display.copy()
    cv.rectangle(overlay, (x1, y1), (x1 + box_w, y1 + box_h), (20, 20, 20), -1)
    cv.addWeighted(overlay, 0.85, display, 0.15, 0, dst=display)
    cv.rectangle(display, (x1, y1), (x1 + box_w, y1 + box_h), (0, 220, 255), 3)

    cv.putText(display, title, (x1 + 30, y1 + 50),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2)
    for i, line in enumerate(lines):
        cv.putText(display, line, (x1 + 30, y1 + 100 + i * 38),
                   cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv.putText(display, footer, (x1 + 30, y1 + box_h - 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)


def draw_cursor(display, pt, mode):
    """Render the persistent index-fingertip cursor marker."""
    if pt is None:
        return
    x, y = pt
    color = {"pinch": PINCH_COLOR, "draw": (0, 255, 0)}.get(mode, CURSOR_COLOR)

    cv.circle(display, (x, y), 14, color, 2)
    cv.circle(display, (x, y), 3, color, -1)
    cv.line(display, (x - 20, y), (x - 8, y), color, 2)
    cv.line(display, (x + 8, y), (x + 20, y), color, 2)
    cv.line(display, (x, y - 20), (x, y - 8), color, 2)
    cv.line(display, (x, y + 8), (x, y + 20), color, 2)


# ---------------------------------------------------------------------------
# Hand detection helpers
# ---------------------------------------------------------------------------

def ensure_model():
    """Download the hand-landmarker model asset on first run if it's missing."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def create_hand_detector():
    options = mp_vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=ensure_model()),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def detect_hand(detector, rgb_frame, timestamp_ms):
    """Run detection for one frame; return pixel-space landmarks or None."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect_for_video(mp_image, timestamp_ms)
    if not result.hand_landmarks:
        return None
    h, w = rgb_frame.shape[:2]
    return [(int(lm.x * w), int(lm.y * h)) for lm in result.hand_landmarks[0]]


# ---------------------------------------------------------------------------
# Gesture classification
# ---------------------------------------------------------------------------

def hand_scale(pts):
    """Rough hand size (wrist to middle-finger MCP) used to normalize thresholds."""
    (wx, wy), (mx, my) = pts[WRIST], pts[MIDDLE_MCP]
    return max(1.0, np.hypot(mx - wx, my - wy))


def is_finger_extended(pts, name):
    tip = pts[FINGER_TIPS[name]]
    pip = pts[FINGER_PIPS[name]]
    wrist = pts[WRIST]
    # Extended if the tip is farther from the wrist than the pip joint.
    return np.hypot(tip[0] - wrist[0], tip[1] - wrist[1]) > np.hypot(
        pip[0] - wrist[0], pip[1] - wrist[1]
    )


def is_thumb_extended(pts):
    tip, mcp, wrist = pts[THUMB_TIP], pts[THUMB_MCP], pts[WRIST]
    return np.hypot(tip[0] - wrist[0], tip[1] - wrist[1]) > np.hypot(
        mcp[0] - wrist[0], mcp[1] - wrist[1]
    )


def classify_gesture(pts):
    """Return (mode, palm_center) where mode is one of

    "pinch", "draw", "palm", or "idle".
    """
    scale = hand_scale(pts)
    thumb_tip, index_tip = pts[THUMB_TIP], pts[FINGER_TIPS["index"]]
    pinch_dist = np.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1])
    pinching = pinch_dist < PINCH_THRESHOLD * scale

    index_ext = is_finger_extended(pts, "index")
    middle_ext = is_finger_extended(pts, "middle")
    ring_ext = is_finger_extended(pts, "ring")
    pinky_ext = is_finger_extended(pts, "pinky")
    thumb_ext = is_thumb_extended(pts)

    open_palm = index_ext and middle_ext and ring_ext and pinky_ext and thumb_ext and not pinching

    if pinching:
        mode = "pinch"
    elif open_palm:
        mode = "palm"
    elif index_ext and not middle_ext:
        mode = "draw"
    else:
        mode = "idle"

    palm_center = (
        (pts[WRIST][0] + pts[MIDDLE_MCP][0]) // 2,
        (pts[WRIST][1] + pts[MIDDLE_MCP][1]) // 2,
    )
    return mode, palm_center


# ---------------------------------------------------------------------------
# Drawing logic
# ---------------------------------------------------------------------------

def in_canvas(pt):
    """True if a point lies inside the drawing box."""
    x1, y1, x2, y2 = CANVAS_RECT
    return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2


def draw_canvas_frame(display):
    """Draw the border that marks the paintable region."""
    x1, y1, x2, y2 = CANVAS_RECT
    cv.rectangle(display, (x1, y1), (x2, y2), (120, 120, 120), 2)
    cv.putText(display, "Drawing area", (x1 + 8, y1 - 8),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)


def flood_fill_canvas(canvas, mask, seed, color):
    h, w = mask.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    # Block the fill from spreading outside the drawing box by pre-marking
    # everything beyond it as already filled (non-zero) in the flood mask.
    x1, y1, x2, y2 = CANVAS_RECT
    flood_mask[:] = 1
    flood_mask[y1 + 1:y2 + 1, x1 + 1:x2 + 1] = 0
    diff = (10, 10, 10)
    try:
        cv.floodFill(canvas, flood_mask, seed, color, diff, diff,
                      flags=cv.FLOODFILL_FIXED_RANGE)
    except cv.error:
        return
    filled = flood_mask[1:-1, 1:-1] > 0
    filled[:y1, :] = False
    filled[y2:, :] = False
    filled[:, :x1] = False
    filled[:, x2:] = False
    mask[filled] = 255


def apply_palm_erase(canvas, mask, center):
    cv.circle(mask, center, PALM_ERASE_RADIUS, 0, -1)
    cv.circle(canvas, center, PALM_ERASE_RADIUS, (0, 0, 0), -1)


def apply_stroke(canvas, mask, prev_pt, curr_pt, color, thickness, erasing):
    if erasing:
        cv.line(mask, prev_pt, curr_pt, 0, thickness)
        cv.line(canvas, prev_pt, curr_pt, (0, 0, 0), thickness)
    else:
        cv.line(canvas, prev_pt, curr_pt, color, thickness)
        cv.line(mask, prev_pt, curr_pt, 255, thickness)


def composite(camera_frame, canvas, mask):
    """Blend the persistent canvas layer over the live camera feed using mask."""
    display = camera_frame.copy()
    mask_3c = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    np.copyto(display, canvas, where=mask_3c.astype(bool))
    return display


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    cap = cv.VideoCapture(CAMERA_INDEX)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    detector = create_hand_detector()
    buttons = build_ui()

    canvas = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    draw_mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)

    state = {"tool": "pen", "color": COLORS[2][1], "brush_size": BRUSH_SIZES["M"]}

    pos_history = deque(maxlen=SMOOTHING_WINDOW)
    prev_draw_pt = None
    prev_pinching = False
    start_time = time.time()
    overlay_start = None  # set on the first rendered frame (post camera warmup)
    instructions_dismissed = False

    window_name = "AI Finger-Paint"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if overlay_start is None:
                overlay_start = time.time()

            frame = cv.resize(frame, (FRAME_W, FRAME_H))
            frame = cv.flip(frame, 1)

            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            timestamp_ms = int((time.time() - start_time) * 1000)
            pts = detect_hand(detector, rgb, timestamp_ms)

            cursor_pt = None
            mode = "idle"

            if pts is not None:
                mode, palm_center = classify_gesture(pts)

                raw_pt = pts[FINGER_TIPS["index"]]
                pos_history.append(raw_pt)
                avg_x = int(sum(p[0] for p in pos_history) / len(pos_history))
                avg_y = int(sum(p[1] for p in pos_history) / len(pos_history))
                cursor_pt = (avg_x, avg_y)

                in_box = in_canvas(cursor_pt)

                if mode == "pinch":
                    pinch_edge = not prev_pinching
                    if pinch_edge:
                        # Any pinch dismisses the first-launch popup.
                        instructions_dismissed = True
                        clicked = next((b for b in buttons if b.contains(*cursor_pt)), None)
                        if clicked:
                            if clicked.kind == "tool":
                                state["tool"] = clicked.value
                            elif clicked.kind == "color":
                                state["color"] = clicked.value
                            elif clicked.kind == "size":
                                state["brush_size"] = clicked.value
                            elif clicked.kind == "clear":
                                canvas[:] = 0
                                draw_mask[:] = 0
                        elif in_box and state["tool"] == "fill":
                            flood_fill_canvas(canvas, draw_mask, cursor_pt, state["color"])
                    prev_pinching = True
                    prev_draw_pt = None

                elif mode == "draw" and in_box:
                    prev_pinching = False
                    erasing = state["tool"] == "eraser"
                    thickness = (state["brush_size"] * ERASER_MULT if erasing
                                 else state["brush_size"])
                    if prev_draw_pt is not None:
                        apply_stroke(canvas, draw_mask, prev_draw_pt, cursor_pt,
                                     state["color"], thickness, erasing)
                    prev_draw_pt = cursor_pt

                elif mode == "palm" and in_box:
                    prev_pinching = False
                    prev_draw_pt = None
                    apply_palm_erase(canvas, draw_mask, palm_center)

                else:
                    # Cursor left the box or gesture isn't drawing: break the
                    # stroke so no line is drawn on the way to the toolbar.
                    prev_pinching = False
                    prev_draw_pt = None
            else:
                pos_history.clear()
                prev_pinching = False
                prev_draw_pt = None

            display = composite(frame, canvas, draw_mask)
            draw_canvas_frame(display)
            draw_ui(display, buttons, state, cursor_pt)
            draw_cursor(display, cursor_pt, mode)
            draw_instructions_overlay(display, time.time() - overlay_start,
                                       instructions_dismissed)

            cv.imshow(window_name, display)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        cap.release()
        detector.close()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
