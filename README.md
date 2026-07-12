# 🎨 AI Finger-Paint

A webcam-based virtual paint app controlled entirely by hand gestures. Track one
hand with [MediaPipe](https://developers.google.com/mediapipe), draw with your
index finger, select tools by pinching, and erase with an open palm — no mouse or
keyboard needed after launch.

Comes in two flavors:

| File | Runs where | Camera |
| --- | --- | --- |
| [`draw.py`](draw.py) | Desktop (OpenCV window) | Local webcam |
| [`streamlit_app.py`](streamlit_app.py) | Browser (Streamlit + WebRTC) | Visitor's webcam via the browser |

## Gestures

- **Draw** — point your index finger inside the drawing box and move it.
- **Pinch** (index tip + thumb tip together) — "click" to select a tool/color, or
  fill an area when the Fill tool is active.
- **Open palm** — wave it over the box to erase like a large soft eraser.

## Tools

Pen · Fill · Eraser · color palette · brush size (S/M/L) · clear canvas.

---

## Desktop version (`draw.py`)

### Requirements

- Python 3.10+
- `opencv-python`, `mediapipe`, `numpy`

```bash
pip install opencv-python mediapipe numpy
python draw.py
```

The hand-landmarker model (`models/hand_landmarker.task`, ~7.8 MB) downloads
automatically on first run. Press **`q`** or **Esc** to quit.

> **Camera note:** the webcam index is set by `CAMERA_INDEX` in
> [`draw.py`](draw.py). It defaults to `1` (built-in MacBook camera). If it opens
> the wrong device, try `0` or `2`.

---

## Web version (`streamlit_app.py`)

Because a cloud server has no webcam, the browser captures the camera and streams
frames to the server over WebRTC ([`streamlit-webrtc`](https://github.com/whitphx/streamlit-webrtc)),
where each frame is processed and streamed back. Tool/color/brush/clear are in the
sidebar; drawing, erasing, and fill stay gesture-driven.

### Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Deploy to Streamlit Community Cloud

1. Push the repo to GitHub (`streamlit_app.py`, `requirements.txt`, `packages.txt`,
   `draw.py`).
2. On [share.streamlit.io](https://share.streamlit.io), create an app from the repo
   with **Main file path** = `streamlit_app.py` and **Python 3.11 or 3.12**.
3. Open the app, click **START**, and allow camera access.

**TURN server (usually required):** Community Cloud sits behind NAT, so the video
stream often won't connect with STUN alone. This app uses
[Metered's free Open Relay TURN service](https://www.metered.ca/tools/openrelay/).
Add your credentials under **Settings → Secrets**:

```toml
METERED_API_KEY   = "your_api_key"
METERED_SUBDOMAIN = "your-app-name"   # the xxx in xxx.metered.live
```

Without them the app falls back to public STUN, which may not connect.

> **Performance:** MediaPipe runs on the server's shared CPU, so expect a laggy
> ~5–15 fps on Community Cloud — much smoother when run locally.

---

## Project layout

```
draw.py            Desktop app + all gesture/drawing helpers
streamlit_app.py   Browser app (reuses helpers from draw.py)
requirements.txt   Python deps for the web version
packages.txt       apt libs for Streamlit Cloud (libgl1, libglib2.0-0t64)
models/            Hand-landmarker model (auto-downloaded)
```

## Tech stack

OpenCV · MediaPipe Hands (Tasks API) · NumPy · Streamlit · streamlit-webrtc
