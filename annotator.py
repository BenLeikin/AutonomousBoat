# annotator.py
# Module 7: Frame annotation utilities (heading arrow, histogram, text overlays, action, axes)

import cv2
import numpy as np
import math

class Annotator:
    """
    Annotator provides static methods to draw various overlays on frames:
    - draw_heading: an arrow indicating the chosen heading
    - draw_histogram: VFH occupancy histogram at bottom of frame
    - draw_text: general text overlay
    - draw_action: overlay current motor action
    - draw_axes: overlay boat-oriented XYZ axes
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
        rad = math.radians(heading_deg)
        length = int(h * length_ratio)
        dx = int(length * math.sin(rad))
        dy = int(-length * math.cos(rad))
        end = (center[0] + dx, center[1] + dy)
        cv2.arrowedLine(frame, center, end, color, thickness, tipLength=0.2)
        cv2.putText(frame, f"{heading_deg:.1f}°", (end[0], end[1] - 10),
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

    @staticmethod
    def draw_action(frame, action_str, position=(10, 80), color=(0, 255, 255), scale=0.6, thickness=2):
        """
        Draw the current motor action on the frame.
        action_str: human-readable action label (e.g. 'Forward', 'Pivot Left')
        """
        return Annotator.draw_text(frame, f"Action: {action_str}", position, color, scale, thickness)

    @staticmethod
    def draw_axes(frame, pitch_deg, roll_deg, length_ratio=0.2, thickness=2):
        """
        Draw the boat's body-fixed axes at the frame center.
        X-axis (red): forward direction rotated by roll
        Y-axis (green): starboard direction rotated by pitch
        Z-axis (blue): upward
        """
        h, w = frame.shape[:2]
        origin = (w // 2, h // 2)
        length = int(h * length_ratio)

        # X-axis: red, based on roll
        rad_roll = math.radians(-roll_deg)
        x_end = (
            int(origin[0] + length * math.cos(rad_roll)),
            int(origin[1] + length * math.sin(rad_roll))
        )
        cv2.arrowedLine(frame, origin, x_end, (0, 0, 255), thickness, tipLength=0.2)
        cv2.putText(frame, 'X', (x_end[0]+5, x_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)

        # Y-axis: green, based on pitch
        rad_pitch = math.radians(-pitch_deg)
        y_end = (
            int(origin[0] + length * math.cos(rad_pitch + math.pi/2)),
            int(origin[1] + length * math.sin(rad_pitch + math.pi/2))
        )
        cv2.arrowedLine(frame, origin, y_end, (0, 255, 0), thickness, tipLength=0.2)
        cv2.putText(frame, 'Y', (y_end[0]+5, y_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)

        # Z-axis: blue, vertical up
        z_end = (origin[0], origin[1] - length)
        cv2.arrowedLine(frame, origin, z_end, (255, 0, 0), thickness, tipLength=0.2)
        cv2.putText(frame, 'Z', (z_end[0]+5, z_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),1)

        return frame
