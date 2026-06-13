"""AutoBoat control subsystem.

Hardware-free reactive avoidance logic. The dashboard imports the controller from
here and feeds it each frame's vision result. Kept as a package (alongside the
vision and sensors packages) so related control code can grow here later.
"""
from .controller import Controller, params, analyze_zones  # noqa: F401
