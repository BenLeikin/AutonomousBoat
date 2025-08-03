# logger.py
# Module 6: CSV + console logging utilities for LiDAR and obstacle counts

import csv
import sys
from datetime import datetime

class Logger:
    """
    Logger handles writing LiDAR distance readings to a CSV file and
    printing obstacle counts or other information to the console.
    """
    def __init__(self, csv_path):
        """
        Initialize the CSV writer and write the header.
        csv_path: filesystem path to output CSV log
        """
        self.csv_path = csv_path
        try:
            self.csv_file = open(csv_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            # Write CSV header
            self.csv_writer.writerow(['timestamp', 'lidar_distance_m'])
        except Exception as e:
            print(f"Error: could not open CSV log at {csv_path}: {e}", file=sys.stderr)
            self.csv_file = None
            self.csv_writer = None

    def log_lidar(self, distance):
        """
        Log a LiDAR distance reading to the CSV file, timestamped.
        distance: float in meters (or None)
        """
        if self.csv_writer:
            timestamp = datetime.now().isoformat()
            self.csv_writer.writerow([timestamp, '' if distance is None else f"{distance:.3f}"])
            self.csv_file.flush()

    def log_console(self, obstacle_count):
        """
        Print obstacle count information to the console.
        obstacle_count: integer number of detected obstacles
        """
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] Obstacles detected: {obstacle_count}")

    def close(self):
        """
        Close the CSV file if open.
        """
        if self.csv_file:
            try:
                self.csv_file.close()
            except Exception:
                pass
