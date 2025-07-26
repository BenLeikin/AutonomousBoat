class VisionPipeline:
    """Stub for vision processing."""
    def __init__(self, resolution, fps):
        self.resolution = resolution
        self.fps = fps

    def next_frame(self):
        """Capture and return next frame. (stub)"""
        return None

    def analyze(self, frame):
        """Detect walls/obstacles and return event dict. (stub)"""
        return None
