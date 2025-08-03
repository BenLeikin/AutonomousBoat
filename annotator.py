# annotator.py
# Module 7: Frame annotation utilities (heading arrow, histogram, text overlays)

import cv2
import numpy as np
import math

class Annotator:
    """
    Annotator provides static methods to draw various overlays on frames:
    - draw_heading: an arrow indicating the chosen heading
    - draw_histogram: VFH occupancy histogram at bottom of frame
    - draw_text: general text overlay
    """
    @staticmethod
    def draw_heading(frame, heading_deg, length_ratio=0.25, color=(255, 255, 255), thickness=2):
        """
        Draw an arrow from the center of the frame pointing at heading_deg relative to forward.
        heading_deg: angle in degrees, where 0 is straight ahead (upwards), positive is to the right.
        length_ratio: fraction of frame height for arrow length.
        """
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        # Convert to radians: 0 deg => -Y axis, so subtract 90°
        rad = math.radians(heading_deg)
        # Arrow length
        length = int(h * length_ratio)
        # Compute end point
        dx = int(length * math.sin(rad))
        dy = int(-length * math.cos(rad))
        end_point = (center[0] + dx, center[1] + dy)
        cv2.arrowedLine(frame, center, end_point, color, thickness, tipLength=0.2)
        # Optionally label the arrow
        cv2.putText(frame, f"{heading_deg:.1f}°", (center[0] + dx, center[1] + dy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    @staticmethod
    def draw_histogram(frame, hist, height=10, free_color=(0,255,0), occ_color=(0,0,255)):
        """
        Draw a VFH occupancy histogram at the bottom of the frame.
        hist: boolean array where True indicates occupied.
        height: pixel height of the histogram bar band.
        """
        h, w = frame.shape[:2]
        sectors = len(hist)
        bar_w = w / sectors
        y1 = h - height*2
        y2 = h - height
        for i, occupied in enumerate(hist):
            x1 = int(i * bar_w)
            x2 = int((i + 1) * bar_w)
            color = occ_color if occupied else free_color
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        return frame

    @staticmethod
    def draw_text(frame, text, position=(10, 30), color=(255,255,255), scale=0.6, thickness=2):
        """
        Draw a line of text on the frame at the given position.
        """
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
        return frame
