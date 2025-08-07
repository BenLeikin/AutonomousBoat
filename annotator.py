#!/usr/bin/env python3
"""
Frame annotation utilities (heading arrow, histogram, text overlays, action, axes)

Refactored to use config-driven styles, instance methods, structured logging,
and decoupled helper functions. Self-test is now headless (no cv2.imshow).
"""
import os
import yaml
import logging
import cv2
import numpy as np
import math
from typing import Any, Dict, List, Tuple, Optional

# Load annotator configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
try:
    with open(CONFIG_PATH, 'r') as f:
        _full_cfg = yaml.safe_load(f) or {}
    _cfg: Dict[str, Any] = _full_cfg.get('annotator', {}) or {}
except Exception:
    _cfg = {}

# Set up module logger
logger = logging.getLogger(__name__)

class Annotator:
    """
    Provides methods to draw overlays on frames:
      - Heading arrow
      - VFH histogram
      - Text overlays
      - Motor action
      - Boat axes (XYZ)
    Styles and parameters driven by 'annotator' section of config.yaml.
    """
    def __init__(self) -> None:
        # Text defaults
        txt = _cfg.get('text', {})
        self.text_font = getattr(cv2, txt.get('font', 'FONT_HERSHEY_SIMPLEX'))
        self.text_scale = float(txt.get('scale', 0.6))
        self.text_color = tuple(txt.get('color', [255,255,255]))
        self.text_thickness = int(txt.get('thickness', 2))
        self.text_pos = tuple(txt.get('position', [10, 30]))

        # Heading style
        hd = _cfg.get('heading', {})
        self.heading_color = tuple(hd.get('color', [255,255,255]))
        self.heading_thickness = int(hd.get('thickness', 2))
        self.heading_tip = float(hd.get('tip_length', 0.2))
        self.heading_len = float(hd.get('length_ratio', 0.25))

        # Action style
        act = _cfg.get('action', {})
        self.action_pos = tuple(act.get('position', [10, 80]))
        self.action_color = tuple(act.get('color', [0,255,255]))
        self.action_scale = float(act.get('scale', 0.6))
        self.action_thickness = int(act.get('thickness', 2))

        # Axes style
        ax = _cfg.get('axes', {})
        self.ax_len = float(ax.get('length_ratio', 0.2))
        self.ax_th = int(ax.get('thickness', 2))
        self.ax_tip = float(ax.get('tip_length', 0.2))
        self.x_color = tuple(ax.get('x_color', [0,0,255]))
        self.y_color = tuple(ax.get('y_color', [0,255,0]))
        self.z_color = tuple(ax.get('z_color', [255,0,0]))

        # Histogram style
        hg = _cfg.get('histogram', {})
        self.hist_height = int(hg.get('height', 10))
        self.free_color = tuple(hg.get('free_color', [0,255,0]))
        self.occ_color = tuple(hg.get('occ_color', [0,0,255]))

    def draw_text(self, frame: np.ndarray, text: str,
                  position: Optional[Tuple[int,int]] = None,
                  color: Optional[Tuple[int,int,int]] = None,
                  scale: Optional[float] = None,
                  thickness: Optional[int] = None) -> np.ndarray:
        """Draw a line of text on the frame at the given position."""
        pos   = position or self.text_pos
        col   = color or self.text_color
        sc    = scale or self.text_scale
        th    = thickness or self.text_thickness
        cv2.putText(frame, text, pos, self.text_font, sc, col, th)
        return frame

    def draw_heading(self, frame: np.ndarray, heading_deg: float) -> np.ndarray:
        """Draw an arrow indicating heading in degrees from frame center."""
        h, w = frame.shape[:2]
        center = (w//2, h//2)
        rad = math.radians(heading_deg)
        length = int(h * self.heading_len)
        dx = int(length * math.sin(rad))
        dy = int(-length * math.cos(rad))
        end = (center[0]+dx, center[1]+dy)
        cv2.arrowedLine(frame, center, end, self.heading_color,
                        self.heading_thickness, tipLength=self.heading_tip)
        label = f"{heading_deg:.1f}°"
        txt_pos = (end[0], end[1] - 10)
        self.draw_text(frame, label, txt_pos, self.heading_color,
                       self.text_scale, self.heading_thickness)
        return frame

    def draw_histogram(self, frame: np.ndarray, hist: List[bool]) -> np.ndarray:
        """Draw VFH occupancy histogram at bottom of frame."""
        h, w = frame.shape[:2]
        sectors = len(hist)
        bar_w = w / sectors
        y1 = h - self.hist_height*2
        y2 = h - self.hist_height
        for i, occ in enumerate(hist):
            x1 = int(i * bar_w)
            x2 = int((i+1) * bar_w)
            color = self.occ_color if occ else self.free_color
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        return frame

    def draw_action(self, frame: np.ndarray, action_str: str) -> np.ndarray:
        """Overlay current motor action label."""
        text = f"Action: {action_str}"
        return self.draw_text(frame, text, self.action_pos,
                              self.action_color, self.action_scale,
                              self.action_thickness)

    def draw_axes(self, frame: np.ndarray,
                  pitch_deg: float, roll_deg: float) -> np.ndarray:
        """Overlay boat body-fixed axes (X, Y, Z)."""
        h, w = frame.shape[:2]
        origin = (w//2, h//2)
        length = int(h * self.ax_len)

        # X axis (roll)
        rad_r = math.radians(-roll_deg)
        x_end = (origin[0] + int(length*math.cos(rad_r)),
                 origin[1] + int(length*math.sin(rad_r)))
        cv2.arrowedLine(frame, origin, x_end, self.x_color,
                        self.ax_th, tipLength=self.ax_tip)
        self.draw_text(frame, 'X', (x_end[0]+5, x_end[1]),
                       self.x_color, self.text_scale, self.ax_th)

        # Y axis (pitch)
        rad_p = math.radians(-pitch_deg + 90)
        y_end = (origin[0] + int(length*math.cos(rad_p)),
                 origin[1] + int(length*math.sin(rad_p)))
        cv2.arrowedLine(frame, origin, y_end, self.y_color,
                        self.ax_th, tipLength=self.ax_tip)
        self.draw_text(frame, 'Y', (y_end[0]+5, y_end[1]),
                       self.y_color, self.text_scale, self.ax_th)

        # Z axis (vertical)
        z_end = (origin[0], origin[1] - length)
        cv2.arrowedLine(frame, origin, z_end, self.z_color,
                        self.ax_th, tipLength=self.ax_tip)
        self.draw_text(frame, 'Z', (z_end[0]+5, z_end[1]),
                       self.z_color, self.text_scale, self.ax_th)
        return frame

# Headless self-test
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    annot = Annotator()
    # Build dummy frame based on config
    try:
        cam_cfg = yaml.safe_load(open(CONFIG_PATH))['camera']
        w = int(cam_cfg.get('frame_width', 640))
        h = int(cam_cfg.get('frame_height', 480))
    except Exception:
        w, h = 640, 480
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Run through all draws
    frame = annot.draw_heading(frame, 30)
    frame = annot.draw_axes(frame, pitch_deg=10, roll_deg=15)
    frame = annot.draw_histogram(frame, [False, True, False, True, False])
    frame = annot.draw_action(frame, 'Forward')
    logger.info("Annotator self-test completed on %dx%d frame", w, h)
