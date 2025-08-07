# vision.py
# Module 3: Background-subtraction + contour smoothing for obstacle detection

import cv2
import numpy as np

class VisionProcessor:
    """
    VisionProcessor uses background subtraction (MOG2) and contour processing
    to detect obstacles in a video frame.

    Methods:
      - detect(frame): returns a list of smoothed contours for obstacles
      - annotate(frame, contours): draws obstacle outlines on the frame
    """
    def __init__(self, min_area=5000, history=500, var_threshold=16):
        """
        min_area: contours smaller than this (in pixels) are ignored
        history: number of frames for MOG2 background model
        var_threshold: threshold on squared Mahalanobis distance for background
        """
        # Background subtractor
        self.backsub = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False
        )
        self.min_area = min_area
        # Morphological kernel for noise removal
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def detect(self, frame):
        """
        Detect obstacles via background subtraction and return filtered contours.
        """
        # 1) Grayscale + blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2) Background subtraction
        mask = self.backsub.apply(blur)

        # 3) Morphological cleaning
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        # 4) Threshold to binary image
        _, thresh = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        # 5) Find raw contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 6) Filter and smooth contours
        processed = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            # Approximate polygon to smooth edges
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # Take convex hull for cleaner shape
            hull = cv2.convexHull(approx)
            processed.append(hull)
        return processed

    def annotate(self, frame, contours):
        """
        Draw convex hulls of detected contours on the frame.
        """
        for cnt in contours:
            cv2.drawContours(frame, [cnt], -1, (255, 0, 0), 2)
        return frame
