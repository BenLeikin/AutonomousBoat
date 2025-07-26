# tests/test_vision.py
from autoboat.vision import VisionPipeline

def test_vision_pipeline_interface():
    vp = VisionPipeline((320, 240), 30)
    # Methods should exist and return None by default
    assert hasattr(vp, 'next_frame')
    assert vp.next_frame() is None

    assert hasattr(vp, 'analyze')
    assert vp.analyze(None) is None
