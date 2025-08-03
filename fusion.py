# fusion.py
# Module 4: LiDAR overlay logic

import cv2

class FusionProcessor:
    """
    FusionProcessor handles overlaying LiDAR distance information on video frames.
    - Draws a centered box indicating the LiDAR beam area
    - Colors the box red if an obstacle is within threshold, otherwise green
    - Annotates the measured distance in meters
    """
    def __init__(self, threshold=0.25, box_ratio=0.3, box_color_free=(0,255,0), box_color_alert=(0,0,255)):
        """
        threshold: distance in meters below which the alert color is used
        box_ratio: fraction of frame width/height for the center box
        box_color_free: BGR color when no obstacle within threshold
        box_color_alert: BGR color when distance < threshold
        """
        self.threshold = threshold
        self.box_ratio = box_ratio
        self.box_color_free = box_color_free
        self.box_color_alert = box_color_alert

    def annotate(self, frame, distance):
        """
        Draw the LiDAR overlay on the frame.
        frame: BGR image
        distance: float in meters or None
        Returns the annotated frame.
        """
        h, w = frame.shape[:2]
        bw = int(w * self.box_ratio)
        bh = int(h * self.box_ratio)
        x1 = (w - bw) // 2
        y1 = (h - bh) // 2
        x2 = x1 + bw
        y2 = y1 + bh

        # choose color based on threshold
        if distance is not None and distance < self.threshold:
            color = self.box_color_alert
        else:
            color = self.box_color_free

        # draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # annotate distance
        if distance is not None:
            text = f"{distance:.2f} m"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame
