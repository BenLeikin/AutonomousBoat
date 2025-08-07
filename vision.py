#!/usr/bin/env python3
"""
VisionProcessor with robust ONNX detection, dynamic input-size handling,
and scalar conversion fix. Always resizes to model's discovered input dimensions.
"""
import os
import yaml
import logging
import cv2
import numpy as np
import time
import onnxruntime as ort
from typing import List, Dict, Any, Optional, Any as AnyType

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
try:
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f) or {}
except Exception:
    cfg = {}

vision_cfg = cfg.get('vision', {}) or {}
detect_cfg = cfg.get('detection', {}) or {}

logger = logging.getLogger(__name__)

class VisionProcessor:
    """
    Processes frames to detect obstacles using either background subtraction or ONNX DNN.
    Always resizes input to the model's actual expected dimensions.
    """
    def __init__(self) -> None:
        # Timing and mode
        self.method       = vision_cfg.get('method', 'bg')
        self.input_scale  = float(vision_cfg.get('input_scale', 1.0))
        self.target_fps   = float(vision_cfg.get('run_fps', 10))
        self.min_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        self.last_time    = 0.0

        # Background-subtractor settings
        bg              = vision_cfg.get('bg', {})
        self.bg_history = int(bg.get('history', 500))
        self.bg_threshold = int(bg.get('var_threshold', 16))
        self.bg_min_area = int(bg.get('min_area', 5000))
        self.bg_sub      = None

        # DNN/ONNX settings
        dnn             = vision_cfg.get('dnn', {})
        self.onnx_path  = dnn.get('onnx_model') or detect_cfg.get('model_path', '')
        self.conf_th    = float(dnn.get('conf_threshold', 0.5))
        self.nms_th     = float(dnn.get('nms_threshold', 0.4))
        self.use_gpu    = bool(dnn.get('use_gpu', False))

        # ONNX runtime placeholders
        self.sess           : Optional[ort.InferenceSession] = None
        self.input_name     : Optional[str] = None
        self.layout         : Optional[str] = None  # 'NCHW' or 'NHWC'
        self.model_input_hw : Optional[tuple[int,int]] = None  # (height, width)

        # Initialize pipeline
        self._setup()

    def _setup(self) -> None:
        if self.method == 'dnn' and self.onnx_path:
            self._init_onnx()
        else:
            self.method = 'bg'
            self._init_bg()

    def _init_bg(self) -> None:
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=self.bg_history, varThreshold=self.bg_threshold
        )
        logger.info("BG init: history=%d, varThreshold=%d, minArea=%d",
                    self.bg_history, self.bg_threshold, self.bg_min_area)

    def _init_onnx(self) -> None:
        try:
            providers = ['CPUExecutionProvider']
            if self.use_gpu:
                providers.insert(0, 'CUDAExecutionProvider')
            self.sess = ort.InferenceSession(self.onnx_path, providers=providers)
            inp = self.sess.get_inputs()[0]
            self.input_name = inp.name
            shp = inp.shape  # e.g. [None,3,640,640] or [None,640,640,3]
            # Determine layout and dims
            if len(shp) == 4 and shp[1] == 3:
                self.layout = 'NCHW'
                h_idx, w_idx = 2, 3
            elif len(shp) == 4 and shp[3] == 3:
                self.layout = 'NHWC'
                h_idx, w_idx = 1, 2
            else:
                raise ValueError(f"Unsupported ONNX input shape {shp}")
            # Store model's expected height/width
            h = int(shp[h_idx]); w = int(shp[w_idx])
            self.model_input_hw = (h, w)
            logger.info("Loaded ONNX '%s' layout=%s inputHW=%s",
                        self.onnx_path, self.layout, self.model_input_hw)
        except Exception as e:
            logger.warning("ONNX init failed (%s), falling back to BG", e)
            self.method = 'bg'
            self._init_bg()

    def detect(self, frame: AnyType) -> List[Dict[str, Any]]:
        """Detect obstacles; dispatch to BG or ONNX path."""
        now = time.time()
        if now - self.last_time < self.min_interval:
            return []
        self.last_time = now

        # Downscale if requested
        if self.input_scale != 1.0:
            frame = cv2.resize(frame, None,
                               fx=self.input_scale,
                               fy=self.input_scale,
                               interpolation=cv2.INTER_LINEAR)

        if self.method == 'bg':
            return self._detect_bg(frame)
        else:
            return self._detect_onnx(frame)

    def _detect_bg(self, frame: AnyType) -> List[Dict[str, Any]]:
        mask = self.bg_sub.apply(frame)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [{'contour': c} for c in cnts if cv2.contourArea(c) >= self.bg_min_area]

    def _detect_onnx(self, frame: AnyType) -> List[Dict[str, Any]]:
        h0, w0 = frame.shape[:2]
        # Ensure model_input_hw is set
        if self.model_input_hw is None:
            logger.warning("Model input size unknown, defaulting to 416x416")
            self.model_input_hw = (416, 416)
        h, w = self.model_input_hw

        # Convert and resize to model size
        blob = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        blob = cv2.resize(blob, (w, h))
        blob = blob.astype(np.float32) / 255.0
        if self.layout == 'NHWC':
            blob = np.expand_dims(blob, axis=0)
        else:
            blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        # Inference
        outs = self.sess.run(None, {self.input_name: blob})
        preds = np.array(outs[0][0])

        # Post-process
        boxes, scores = [], []
        for det in preds:
            # extract scalar values robustly
            # confidence at index 4
            conf_arr = det[4]
            conf = float(conf_arr.item()) if isinstance(conf_arr, np.ndarray) else float(conf_arr)
            if conf < self.conf_th:
                continue
            # bounding box center/size
            vals = [det[i] for i in range(4)]
            cx, cy, wr, hr = [float(v.item()) if isinstance(v, np.ndarray) else float(v) for v in vals]
            x1 = int((cx - wr/2) * w0)
            y1 = int((cy - hr/2) * h0)
            x2 = int((cx + wr/2) * w0)
            y2 = int((cy + hr/2) * h0)
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(conf)

        idxs = cv2.dnn.NMSBoxes(boxes, scores, self.conf_th, self.nms_th)
        return [
            {
                'bbox': (boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]),
                'confidence': scores[i]
            }
            for i in idxs.flatten()
        ]

# End of module
