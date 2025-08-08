#!/usr/bin/env python3
import os, time, logging, yaml
import numpy as np
import cv2
from picamera2 import Picamera2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Try to get real libcamera enums (best path)
try:
    import libcamera
    AWB_ENUM = libcamera.controls.AwbModeEnum
    NR_ENUM  = libcamera.controls.draft.NoiseReductionModeEnum if hasattr(libcamera.controls, "draft") else None
except Exception:
    AWB_ENUM = None
    NR_ENUM  = None

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
_CFG = yaml.safe_load(open(CONFIG_PATH, "r")) if os.path.exists(CONFIG_PATH) else {}
_CAM = _CFG.get("camera", {}) or {}
_AWB = (_CAM.get("awb") or {})
_CLR = (_CAM.get("color") or {})
_DENOISE = (_CAM.get("denoise", "auto") or "auto").strip().lower()

# Fallback maps if libcamera enums are unavailable
_FALLBACK_AWB = {
    "auto": None,  # None => let camera pick
    "daylight": 1,     # Typical libcamera ordering; will only be used if AWB_ENUM missing
    "cloudy": 2,
    "tungsten": 3,
    "fluorescent": 4,
    "indoor": 5,
}
_FALLBACK_NR = {
    "off": 0, "fast": 1, "high_quality": 2, "hq": 2, "minimal": 3, "zsl": 4
}

def _grayworld_gains(bgr: np.ndarray) -> tuple[float, float]:
    img = bgr.astype(np.float32)
    b, g, r = cv2.split(img)
    # avoid div by zero
    mb = float(b.mean() + 1e-6); mg = float(g.mean() + 1e-6); mr = float(r.mean() + 1e-6)
    k = (mb + mg + mr) / 3.0
    r_gain = max(0.25, min(8.0, k / mr))
    b_gain = max(0.25, min(8.0, k / mb))
    return (r_gain, b_gain)

def _map_awb(mode: str):
    m = (mode or "auto").strip().lower()
    if AWB_ENUM:
        table = {
            "auto": AWB_ENUM.Auto,
            "daylight": AWB_ENUM.Daylight,
            "cloudy": AWB_ENUM.Cloudy,
            "tungsten": AWB_ENUM.Tungsten,
            "fluorescent": AWB_ENUM.Fluorescent,
            "indoor": AWB_ENUM.Indoor,
        }
        return table.get(m, AWB_ENUM.Auto)
    # fallback integer (only if not auto)
    return _FALLBACK_AWB.get(m, None)

def _map_nr(mode: str):
    m = (mode or "auto").strip().lower()
    if NR_ENUM:
        table = {
            "off": NR_ENUM.Off,
            "fast": NR_ENUM.Fast,
            "high_quality": NR_ENUM.HighQuality,
            "hq": NR_ENUM.HighQuality,
            "minimal": NR_ENUM.Minimal,
            "zsl": NR_ENUM.ZSL,
        }
        return table.get(m, None)  # None => don't set
    return _FALLBACK_NR.get(m, None)

class CameraReader:
    def __init__(self):
        self.picam2 = None
        self.started = False
        self.width = int(_CAM.get("frame_width", 640))
        self.height = int(_CAM.get("frame_height", 480))
        self.fps = int(_CAM.get("fps", 30))
        self.warmup = float(_CAM.get("warmup", 2.0))

    def _apply_misc_controls(self):
        # Optional noise reduction (skip if "auto")
        nr_val = _map_nr(_DENOISE)
        if nr_val is not None:
            try:
                self.picam2.set_controls({"NoiseReductionMode": int(nr_val)})
                logger.info("NoiseReductionMode set to %s (%s)", _DENOISE, nr_val)
            except Exception as e:
                logger.debug("NoiseReductionMode not supported: %s", e)

        # Generic image adjustments
        for k_src, k_cam in [
            ("brightness", "Brightness"),
            ("contrast",   "Contrast"),
            ("saturation", "Saturation"),
            ("sharpness",  "Sharpness"),
        ]:
            v = _CLR.get(k_src, None)
            if v is not None:
                try:
                    self.picam2.set_controls({k_cam: float(v)})
                except Exception as e:
                    logger.debug("%s not supported: %s", k_cam, e)

    def _apply_awb(self, first_frame=None):
        mode = (_AWB.get("mode", "auto") or "auto").lower()

        try:
            if mode == "manual":
                gains = _AWB.get("gains", [1.5, 1.7])
                r_gain = float(gains[0]); b_gain = float(gains[1])
                self.picam2.set_controls({"AwbEnable": False, "ColourGains": (r_gain, b_gain)})
                logger.info("AWB manual gains set R=%.2f B=%.2f", r_gain, b_gain)
                return

            if mode == "grayworld":
                if first_frame is None:
                    # Temporarily leave AWB enabled until we have a frame
                    self.picam2.set_controls({"AwbEnable": True})
                    logger.info("AWB temp Auto (grayworld pending first frame)")
                else:
                    r_gain, b_gain = _grayworld_gains(first_frame)
                    self.picam2.set_controls({"AwbEnable": False, "ColourGains": (r_gain, b_gain)})
                    logger.info("AWB grayworld gains -> R=%.2f B=%.2f", r_gain, b_gain)
                return

            # Auto or fixed mode via enum (use integer, not string)
            if mode == "auto":
                # safest: enable AWB and don't send AwbMode at all
                self.picam2.set_controls({"AwbEnable": True})
                logger.info("AWB mode: Auto")
                return

            # daylight/cloudy/...: set AwbMode as INT if we can resolve it
            awb_val = _map_awb(mode)
            if awb_val is None:
                # Could not resolve to int; fall back to Auto to avoid string→int crash
                self.picam2.set_controls({"AwbEnable": True})
                logger.warning("AWB '%s' not supported on this build; using Auto", mode)
            else:
                self.picam2.set_controls({"AwbEnable": True, "AwbMode": int(awb_val)})
                logger.info("AWB mode set to '%s' (enum=%s)", mode, awb_val)

        except Exception as e:
            logger.warning("AWB control failed (%s); enabling Auto", e)
            try:
                self.picam2.set_controls({"AwbEnable": True})
            except Exception:
                pass

    def _init_camera(self):
        logger.info("Initializing Picamera2 (resolution=(%d, %d), fps=%d, warmup=%.1fs)",
                    self.width, self.height, self.fps, self.warmup)
        self.picam2 = Picamera2()
        cfg = self.picam2.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            buffer_count=4,
        )
        self.picam2.configure(cfg)

        # Apply controls that don't depend on an image
        self._apply_misc_controls()
        self._apply_awb(first_frame=None)

        # Start and warm up
        self.picam2.start()
        self.started = True
        time.sleep(self.warmup)

        # If grayworld, compute from a real frame and lock gains
        if (_AWB.get("mode", "auto") or "auto").lower() == "grayworld":
            frame = self.capture_array()
            self._apply_awb(first_frame=frame)

        logger.info("Camera ready at %d fps", self.fps)

    def capture_array(self):
        return self.picam2.capture_array("main")

    def read(self):
        if not self.started:
            self._init_camera()
        return self.capture_array()

    def release(self):
        if self.started and self.picam2:
            logger.info("Releasing camera resources")
            try:
                self.picam2.stop()
                self.picam2.close()
                logger.info("Camera closed successfully.")
            except Exception as e:
                logger.warning("Camera close error: %s", e)
        self.started = False
        self.picam2 = None

    def __enter__(self):
        self._init_camera()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

if __name__ == "__main__":
    out_dir = "/mnt/usb" if os.path.isdir("/mnt/usb") else "logs"
    os.makedirs(out_dir, exist_ok=True)
    with CameraReader() as cam:
        img = cam.read()
        path = os.path.join(out_dir, "awb_check.jpg")
        cv2.imwrite(path, img)
        logger.info("Saved test frame to %s", path)
