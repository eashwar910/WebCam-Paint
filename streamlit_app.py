"""AI Finger-Paint — Streamlit / WebRTC web version.

This is the browser-deployable version of the desktop app in draw.py.

Why it's different from draw.py:
  On Streamlit Community Cloud the Python code runs on a REMOTE server that has
  no webcam. `cv2.VideoCapture` cannot see the visitor's camera. Instead, the
  browser captures the webcam and streams frames to the server over WebRTC via
  `streamlit-webrtc`; each frame is processed server-side in VideoProcessor.recv
  and streamed back. This is an inherently DYNAMIC, stateful, server-side app —
  there is no static-hosting option for it.

Controls:
  - Point your index finger inside the box to draw.
  - Open palm over the box to erase.
  - Pinch (index + thumb) to fill, when the Fill tool is selected.
  - Tool / color / brush size / clear are in the sidebar.
"""

import threading
import time
from collections import deque

import av
import cv2 as cv
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

import draw  # reuse gesture + drawing helpers from the desktop version

st.set_page_config(page_title="AI Finger-Paint", page_icon="🎨", layout="wide")


# ---------------------------------------------------------------------------
# ICE / TURN configuration
# ---------------------------------------------------------------------------

def get_ice_servers():
    """Return ICE servers for WebRTC.

    Community Cloud sits behind NAT/firewall, so a TURN server is usually
    required for the video stream to connect. Twilio is the most reliable
    option; add TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN in the app's Secrets to
    enable it. Without them we fall back to a public STUN server, which works
    on some networks but often fails on Community Cloud.
    """
    try:
        account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
        from twilio.rest import Client

        token = Client(account_sid, auth_token).tokens.create()
        return token.ice_servers
    except Exception:
        return [{"urls": ["stun:stun.l.google.com:19302"]}]


# ---------------------------------------------------------------------------
# Video processing (runs in a forked WebRTC worker thread, once per frame)
# ---------------------------------------------------------------------------

class VideoProcessor:
    def __init__(self):
        self.detector = draw.create_hand_detector()
        self.canvas = np.zeros((draw.FRAME_H, draw.FRAME_W, 3), dtype=np.uint8)
        self.mask = np.zeros((draw.FRAME_H, draw.FRAME_W), dtype=np.uint8)

        self.pos_history = deque(maxlen=draw.SMOOTHING_WINDOW)
        self.prev_draw_pt = None
        self.prev_pinching = False
        self.frame_ts = 0  # strictly increasing timestamp for MediaPipe VIDEO mode

        # Controls pushed in from the sidebar (Streamlit main thread).
        self.tool = "pen"
        self.color = draw.COLORS[2][1]  # red (BGR)
        self.brush_size = draw.BRUSH_SIZES["M"]
        self.clear_requested = False

        self.lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv.resize(img, (draw.FRAME_W, draw.FRAME_H))
        img = cv.flip(img, 1)

        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.frame_ts += 33  # ~30 fps, monotonically increasing
        pts = draw.detect_hand(self.detector, rgb, self.frame_ts)

        cursor_pt = None
        mode = "idle"

        with self.lock:
            if self.clear_requested:
                self.canvas[:] = 0
                self.mask[:] = 0
                self.clear_requested = False

            if pts is not None:
                mode, palm_center = draw.classify_gesture(pts)

                raw_pt = pts[draw.FINGER_TIPS["index"]]
                self.pos_history.append(raw_pt)
                avg_x = int(sum(p[0] for p in self.pos_history) / len(self.pos_history))
                avg_y = int(sum(p[1] for p in self.pos_history) / len(self.pos_history))
                cursor_pt = (avg_x, avg_y)
                in_box = draw.in_canvas(cursor_pt)

                if mode == "pinch":
                    if not self.prev_pinching and in_box and self.tool == "fill":
                        draw.flood_fill_canvas(self.canvas, self.mask, cursor_pt, self.color)
                    self.prev_pinching = True
                    self.prev_draw_pt = None

                elif mode == "draw" and in_box:
                    self.prev_pinching = False
                    erasing = self.tool == "eraser"
                    thickness = (self.brush_size * draw.ERASER_MULT if erasing
                                 else self.brush_size)
                    if self.prev_draw_pt is not None:
                        draw.apply_stroke(self.canvas, self.mask, self.prev_draw_pt,
                                          cursor_pt, self.color, thickness, erasing)
                    self.prev_draw_pt = cursor_pt

                elif mode == "palm" and in_box:
                    self.prev_pinching = False
                    self.prev_draw_pt = None
                    draw.apply_palm_erase(self.canvas, self.mask, palm_center)

                else:
                    self.prev_pinching = False
                    self.prev_draw_pt = None
            else:
                self.pos_history.clear()
                self.prev_pinching = False
                self.prev_draw_pt = None

            display = draw.composite(img, self.canvas, self.mask)

        draw.draw_canvas_frame(display)
        draw.draw_cursor(display, cursor_pt, mode)
        return av.VideoFrame.from_ndarray(display, format="bgr24")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🎨 AI Finger-Paint")
st.caption(
    "Point your index finger inside the box to draw · open palm to erase · "
    "pinch to fill (Fill tool). Allow camera access when prompted."
)

with st.sidebar:
    st.header("Tools")
    tool = st.radio("Tool", ["pen", "fill", "eraser"], index=0,
                    format_func=str.capitalize)
    color_name = st.selectbox("Color", [c[0] for c in draw.COLORS], index=2)
    brush_label = st.select_slider("Brush size", options=["S", "M", "L"], value="M")
    clear_clicked = st.button("🗑️ Clear canvas", use_container_width=True)

ctx = webrtc_streamer(
    key="finger-paint",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": get_ice_servers()},
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Push sidebar selections into the running video processor.
if ctx.video_processor:
    ctx.video_processor.tool = tool
    ctx.video_processor.color = dict(draw.COLORS)[color_name]
    ctx.video_processor.brush_size = draw.BRUSH_SIZES[brush_label]
    if clear_clicked:
        ctx.video_processor.clear_requested = True

with st.expander("Deployment note / performance"):
    st.markdown(
        "- MediaPipe runs **on the server**, so frame rate depends on the host. "
        "On Streamlit Community Cloud (shared CPU) expect a laggy ~5–15 fps.\n"
        "- If the video never connects on Community Cloud, you almost certainly "
        "need a **TURN server** — add Twilio credentials in *Settings → Secrets*."
    )
