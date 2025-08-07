# vfh.py
# Module 5: Vector Field Histogram (VFH) navigator

import math
import numpy as np
import cv2

class VFHNavigator:
    """
    VFHNavigator builds a local polar occupancy histogram from LiDAR and vision
    data and selects the best free direction to steer the boat.

    Methods:
      - compute_histogram(contours, lidar_dist): returns a boolean array of occupied sectors
      - find_best_gap(hist): returns the heading angle (radians) of the largest free sector
    """
    def __init__(self, sectors=72, fov_deg=90, threshold=0.25, frame_width=1280):
        """
        sectors: number of angular bins in the histogram
        fov_deg: horizontal field of view of camera in degrees
        threshold: LiDAR distance threshold (m) to mark center sector as occupied
        frame_width: width of image frame in pixels (for contour angle mapping)
        """
        self.sectors = sectors
        self.threshold = threshold
        self.fov = math.radians(fov_deg)
        # precompute angles for each sector center
        self.angles = np.linspace(-self.fov/2, self.fov/2, sectors)
        self.frame_width = frame_width

    def compute_histogram(self, contours, lidar_dist):
        """
        Build a boolean histogram where True indicates an obstacle in that sector.
        Contours from vision and a single LiDAR distance are binned into sectors.
        """
        hist = np.zeros(self.sectors, dtype=bool)
        center_idx = self.sectors // 2
        # Mark center sector if LiDAR sees close object
        if lidar_dist is not None and lidar_dist < self.threshold:
            hist[center_idx] = True
        # Bin vision contours by their centroid angle
        for cnt in contours:
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            # Map pixel x-coordinate to angle
            angle = (cx - self.frame_width/2) / (self.frame_width/2) * (self.fov/2)
            # Find closest sector
            idx = np.argmin(np.abs(self.angles - angle))
            hist[idx] = True
        return hist

    def find_best_gap(self, hist):
        """
        Identify the largest contiguous run of False (free) sectors and return
        its center angle in radians. If no free gap, returns 0.0.
        """
        free = ~hist
        best_start = best_len = 0
        curr_start = curr_len = None, 0
        best_start = 0
        best_len = 0
        curr_start = None
        curr_len = 0
        for i, is_free in enumerate(free):
            if is_free:
                if curr_start is None:
                    curr_start, curr_len = i, 1
                else:
                    curr_len += 1
            else:
                if curr_start is not None and curr_len > best_len:
                    best_start, best_len = curr_start, curr_len
                curr_start, curr_len = None, 0
        # Final check at end
        if curr_start is not None and curr_len > best_len:
            best_start, best_len = curr_start, curr_len
        if best_len == 0:
            return 0.0
        mid = best_start + best_len // 2
        return self.angles[mid]
