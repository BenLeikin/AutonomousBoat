#!/usr/bin/env python3
"""
annotator.py — draw overlays and runtime stats onto frames (headless-safe)

Features
- Color-coded detections: edge | hough | motion | flow | dnn
- Forward-ROI band/corridor box
- Status panel: FPS, counts per kind, blocked flag, action, frame index
- Optional heading arrow, IMU axes, simple occupancy histogram
- Tuned entirely from config.yaml (annotator.*, and recording.annotate.*)

Inputs (typical):
  - detections: list of dicts produced by vision:
      {'bbox':(x1,y1,x2,y2), 'kind':'edge|hough|motion|flow|dnn', ...}
      (for classical kinds, 'contour' may be present; we'll draw bbox either way)
  - stats: dict with keys you want shown, e.g.:
      {
        'frame_idx': int,
        'fps': float,
        'mode': 'classical'|'dnn'|'bg',
        'counts': {'edge':N, 'hough':N, 'motion':N, 'flow':N, 'dnn':N, 'total':N},
        'blocked': bool,
        'action': 'Forward'|'Turn 90° left'|...,
        'motors': 'L=F R=F' (optional),
        'thoughts': [str, ...] (optional)
      }
  - roi: dict with forward zone geometry (optional):
      {'x0': int, 'x1': int, 'y0': int}  # forward band = [y0:h, x0:x1]
"""

import os
import math
import yaml
import cv2
import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def _load_cfg():
    try:
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

_CFG = _load_cfg()
_ANN = _CFG.get("annotator", {}) or {}
_REC = _CFG.get("recording", {}) or {}
_FLAGS = (_REC.get("annotate") or {
    "detections": True, "heading": True, "histogram": False, "action": True, "axes": False
})

# ------- style helpers from config -------
def _get(path, default=None):
    cur = _CFG
    try:
        for k in path.split("."):
            cur = cur[k]
        return default if cur is None else cur
    except Exception:
        return default

_TEXT_FONT   = getattr(cv2, _ANN.get("text", {}).get("font", "FONT_HERSHEY_SIMPLEX"), cv2.FONT_HERSHEY_SIMPLEX)
_TEXT_SCALE  = float(_ANN.get("text", {}).get("scale", 0.6))
_TEXT_COLOR  = tuple(int(c) for c in _ANN.get("text", {}).get("color", [255, 255, 255]))
_TEXT_THICK  = int(_ANN.get("text", {}).get("thickness", 2))
_TEXT_POS    = tuple(int(v) for v in _ANN.get("text", {}).get("position", [10, 30]))

_ACTION_POS  = tuple(int(v) for v in _ANN.get("action", {}).get("position", [10, 80]))
_ACTION_COLOR= tuple(int(c) for c in _ANN.get("action", {}).get("color", [0, 255, 255]))
_ACTION_SCALE= float(_ANN.get("action", {}).get("scale", 0.6))
_ACTION_THICK= int(_ANN.get("action", {}).get("thickness", 2))

_BOX_COLOR   = tuple(int(c) for c in _ANN.get("det", {}).get("box_color", [0, 255, 255]))
_BOX_THICK   = int(_ANN.get("det", {}).get("box_thickness", 2))
_CNT_COLOR   = tuple(int(c) for c in _ANN.get("det", {}).get("contour_color", [0, 255, 255]))
_DET_TXT_COL = tuple(int(c) for c in _ANN.get("det", {}).get("text_color", [0, 255, 255]))

_HEAD_COLOR  = tuple(int(c) for c in _ANN.get("heading", {}).get("color", [255, 255, 255]))
_HEAD_THICK  = int(_ANN.get("heading", {}).get("thickness", 2))
_HEAD_TIP    = float(_ANN.get("heading", {}).get("tip_length", 0.2))
_HEAD_LENR   = float(_ANN.get("heading", {}).get("length_ratio", 0.25))

_AX_LENR     = float(_ANN.get("axes", {}).get("length_ratio", 0.2))
_AX_THICK    = int(_ANN.get("axes", {}).get("thickness", 2))
_AX_TIP      = float(_ANN.get("axes", {}).get("tip_length", 0.2))
_AX_XCOL     = tuple(int(c) for c in _ANN.get("axes", {}).get("x_color", [0, 0, 255]))
_AX_YCOL     = tuple(int(c) for c in _ANN.get("axes", {}).get("y_color", [0, 255, 0]))
_AX_ZCOL     = tuple(int(c) for c in _ANN.get("axes", {}).get("z_color", [255, 0, 0]))

_HIST_H      = int(_ANN.get("histogram", {}).get("height", 10))
_HIST_FREE   = tuple(int(c) for c in _ANN.get("histogram", {}).get("free_color", [0, 255, 0]))
_HIST_OCC    = tuple(int(c) for c in _ANN.get("histogram", {}).get("occ_color", [0, 0, 255]))

# Per-kind colors (overrideable by config under annotator.det.kind_colors.<kind>)
_KIND_COLORS = {
    "edge":   (0, 255, 255),  # yellow
    "hough":  (255, 255, 0),  # cyan
    "motion": (0, 0, 255),    # red
    "flow":   (255, 0, 0),    # blue
    "dnn":    _BOX_COLOR,
}
_kind_cfg = (_ANN.get("det", {}).get("kind_colors") or {})
for k, v in _kind_cfg.items():
    try:
        _KIND_COLORS[str(k).lower()] = tuple(int(c) for c in v)
    except Exception:
        pass

# Thoughts overlay lines (from logging.thoughts.max_overlay_lines)
_MAX_THOUGHTS = int(_get("logging.thoughts.max_overlay_lines", 4))
_SHOW_THOUGHTS = bool(_get("logging.thoughts.overlay", True))

class Annotator:
    def __init__(self):
        # Nothing heavy; all styling pulled from config above
        pass

    # ---------- basic text ----------
    def _put(self, img, text, org, color=None, scale=None, thick=None):
        cv2.putText(
            img, str(text),
            org,
            _TEXT_FONT,
            scale if scale is not None else _TEXT_SCALE,
            color if color is not None else _TEXT_COLOR,
            thick if thick is not None else _TEXT_THICK,
            cv2.LINE_AA,
        )

    # ---------- detections ----------
    def draw_detections(self, img, detections):
        if not _FLAGS.get("detections", True):
            return img
        for d in detections or []:
            bbox = d.get("bbox")
            if not bbox: 
                continue
            x1, y1, x2, y2 = map(int, bbox)
            kind = str(d.get("kind", "dnn")).lower()
            color = _KIND_COLORS.get(kind, _BOX_COLOR)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, _BOX_THICK)
            # Label
            label = kind
            if "confidence" in d:
                try:
                    label += f" {d['confidence']:.2f}"
                except Exception:
                    pass
            elif "score" in d:
                try:
                    label += f" {d['score']:.0f}"
                except Exception:
                    pass
            ty = max(0, y1 - 5)
            cv2.putText(img, label, (x1, ty), _TEXT_FONT, 0.5, color, 1, cv2.LINE_AA)
        return img

    # ---------- status panel ----------
    def draw_status(self, img, stats: dict | None):
        if not stats:
            return img
        x, y = _TEXT_POS
        lines = []

        # First line: frame/fps/mode
        frame_idx = stats.get("frame_idx", None)
        fps = stats.get("fps", None)
        mode = stats.get("mode", None)
        parts = []
        if frame_idx is not None: parts.append(f"f={frame_idx}")
        if fps is not None:       parts.append(f"fps={fps:.1f}")
        if mode:                  parts.append(f"mode={mode}")
        if parts: lines.append(" | ".join(parts))

        # Second: counts
        cnts = stats.get("counts", {})
        if cnts:
            c_parts = []
            for k in ("edge","hough","motion","flow","dnn","total"):
                if k in cnts:
                    c_parts.append(f"{k[0]}={cnts[k]}")
            if c_parts:
                lines.append("cnts: " + " ".join(c_parts))

        # Third: blocked + action
        blk = stats.get("blocked", None)
        act = stats.get("action", None)
        mtr = stats.get("motors", None)
        parts = []
        if blk is not None: parts.append(f"blocked={'YES' if blk else 'no'}")
        if act: parts.append(act)
        if mtr: parts.append(mtr)
        if parts: lines.append(" | ".join(parts))

        # Thoughts (truncated)
        if _SHOW_THOUGHTS:
            thoughts = stats.get("thoughts") or []
            for t in thoughts[:_MAX_THOUGHTS]:
                lines.append(t)

        # Draw lines
        ly = y
        for i, line in enumerate(lines):
            self._put(img, line, (x, ly))
            ly += int(18 * _TEXT_SCALE + 8)

        return img

    # ---------- forward ROI / band ----------
    def draw_forward_roi(self, img, roi: dict | None, blocked: bool | None = None):
        if not roi:
            return img
        h, w = img.shape[:2]
        x0, x1 = int(roi.get("x0", 0)), int(roi.get("x1", w))
        y0 = int(roi.get("y0", int(0.6 * h)))
        color = (0, 255, 0) if not blocked else (0, 0, 255)
        cv2.rectangle(img, (x0, y0), (x1, h-1), color, 2)
        return img

    # ---------- action text ----------
    def draw_action(self, img, action: str | None):
        if not _FLAGS.get("action", True) or not action:
            return img
        self._put(img, action, _ACTION_POS, color=_ACTION_COLOR, scale=_ACTION_SCALE, thick=_ACTION_THICK)
        return img

    # ---------- heading arrow ----------
    def draw_heading(self, img, yaw_deg: float | None = None):
        if not _FLAGS.get("heading", True) or yaw_deg is None:
            return img
        h, w = img.shape[:2]
        L = int(min(h, w) * _HEAD_LENR)
        cx, cy = w // 2, int(h * 0.65)  # slightly below center
        rad = math.radians(yaw_deg)
        x2 = int(cx + L * math.cos(rad))
        y2 = int(cy - L * math.sin(rad))
        cv2.arrowedLine(img, (cx, cy), (x2, y2), _HEAD_COLOR, _HEAD_THICK, tipLength=_HEAD_TIP)
        return img

    # ---------- axes (optional IMU viz) ----------
    def draw_axes(self, img, roll_deg: float, pitch_deg: float):
        if not _FLAGS.get("axes", False):
            return img
        h, w = img.shape[:2]
        L = int(min(h, w) * _AX_LENR)
        cx, cy = int(w * 0.1), int(h * 0.85)
        # Simple 2D projection
        r = math.radians(roll_deg); p = math.radians(pitch_deg)
        # X axis (roll)
        cv2.arrowedLine(img, (cx, cy), (cx + int(L * math.cos(r)), cy - int(L * math.sin(r))),
                        _AX_XCOL, _AX_THICK, tipLength=_AX_TIP)
        # Y axis (pitch)
        cv2.arrowedLine(img, (cx, cy), (cx, cy - int(L * math.cos(p))),
                        _AX_YCOL, _AX_THICK, tipLength=_AX_TIP)
        return img

    # ---------- simple occupancy histogram (visual aid only) ----------
    def draw_histogram(self, img, occ_ratio: float | None):
        if not _FLAGS.get("histogram", False) or occ_ratio is None:
            return img
        h, w = img.shape[:2]
        bar_h = _HIST_H
        filled_w = int(max(0.0, min(1.0, occ_ratio)) * w)
        y0 = 5
        cv2.rectangle(img, (0, y0), (w-1, y0 + bar_h), _HIST_FREE, -1)
        cv2.rectangle(img, (0, y0), (filled_w, y0 + bar_h), _HIST_OCC, -1)
        cv2.rectangle(img, (0, y0), (w-1, y0 + bar_h), (0,0,0), 1)
        return img

    # ---------- one-stop shop ----------
    def annotate_frame(self, img, detections=None, stats=None, roi=None,
                       yaw_deg=None, roll_deg=None, pitch_deg=None, occ_ratio=None):
        """
        Draw everything according to config flags.
        """
        if detections:
            self.draw_detections(img, detections)
        if roi:
            self.draw_forward_roi(img, roi, blocked=stats.get("blocked") if stats else None)
        if stats:
            self.draw_status(img, stats)
            self.draw_action(img, stats.get("action"))
        if yaw_deg is not None:
            self.draw_heading(img, yaw_deg)
        if roll_deg is not None and pitch_deg is not None:
            self.draw_axes(img, roll_deg, pitch_deg)
        if occ_ratio is not None:
            self.draw_histogram(img, occ_ratio)
        return img

# ---------- headless self-test ----------
if __name__ == "__main__":
    # Build a dummy frame and show a sample overlay (saved to logs/)
    h, w = 480, 640
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # fake detections
    detections = [
        {"bbox": (50, 320, 200, 470), "kind": "edge", "score": 1234},
        {"bbox": (260, 340, 380, 470), "kind": "motion", "score": 900},
        {"bbox": (420, 300, 620, 460), "kind": "hough", "score": 300},
    ]
    # fake stats
    stats = {
        "frame_idx": 42,
        "fps": 18.7,
        "mode": "classical",
        "counts": {"edge": 3, "hough": 1, "motion": 2, "flow": 0, "dnn": 0, "total": 6},
        "blocked": True,
        "action": "Turn 90° right",
        "motors": "L=REV R=FWD",
        "thoughts": ["edge>thresh in ROI", "choose right (free gap wider)"]
    }
    roi = {"x0": int(w*0.125), "x1": int(w*0.875), "y0": int(h*0.4)}

    ann = Annotator()
    ann.annotate_frame(frame, detections=detections, stats=stats, roi=roi, yaw_deg=15.0, roll_deg=2.0, pitch_deg=-1.0, occ_ratio=0.35)

    out_dir = "/mnt/usb" if os.path.isdir("/mnt/usb") else "logs"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "annotator_smoke.jpg")
    cv2.imwrite(path, frame)
    print("Saved", path)
