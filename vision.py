#!/usr/bin/env python3
"""
VisionProcessor with robust ONNX handling:
- Detects NHWC/NCHW inputs and resizes accordingly
- Supports both YOLOv5 "raw" and "NMS-export" ONNX outputs
- Flattens multi-head raw outputs to (N, 85)
- Computes conf = obj * max(class_probs) and applies OpenCV NMS
"""
import os
import yaml
import logging
import cv2
import numpy as np
import time
import onnxruntime as ort
from typing import List, Dict, Any, Optional, Tuple

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
    def __init__(self) -> None:
        # Timing & mode
        self.method       = vision_cfg.get('method', 'bg')
        self.input_scale  = float(vision_cfg.get('input_scale', 1.0))
        self.target_fps   = float(vision_cfg.get('run_fps', 10))
        self.min_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        self.last_time    = 0.0

        # BG params
        bg = vision_cfg.get('bg', {})
        self.bg_history   = int(bg.get('history', 500))
        self.bg_threshold = int(bg.get('var_threshold', 16))
        self.bg_min_area  = int(bg.get('min_area', 5000))
        self.bg_sub       = None

        # DNN params
        dnn = vision_cfg.get('dnn', {})
        self.onnx_path    = dnn.get('onnx_model') or detect_cfg.get('model_path', '')
        self.conf_th      = float(dnn.get('conf_threshold', 0.5))
        self.nms_th       = float(dnn.get('nms_threshold', 0.4))
        self.use_gpu      = bool(dnn.get('use_gpu', False))

        # ONNX placeholders
        self.sess: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.layout: Optional[str] = None         # 'NHWC' or 'NCHW'
        self.model_input_hw: Optional[Tuple[int,int]] = None  # (H, W)

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
            shp = inp.shape  # e.g. [None, 640, 640, 3] or [None, 3, 640, 640]
            if len(shp) == 4 and shp[1] == 3:
                self.layout = 'NCHW'; h_idx, w_idx = 2, 3
            elif len(shp) == 4 and shp[3] == 3:
                self.layout = 'NHWC'; h_idx, w_idx = 1, 2
            else:
                raise ValueError(f"Unsupported ONNX input shape {shp}")
            self.model_input_hw = (int(shp[h_idx]), int(shp[w_idx]))
            logger.info("Loaded ONNX '%s' layout=%s inputHW=%s",
                        self.onnx_path, self.layout, self.model_input_hw)
        except Exception as e:
            logger.warning("ONNX init failed (%s), falling back to BG", e)
            self.method = 'bg'
            self._init_bg()

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        now = time.time()
        if now - self.last_time < self.min_interval:
            return []
        self.last_time = now

        if self.input_scale != 1.0:
            frame = cv2.resize(frame, None,
                               fx=self.input_scale, fy=self.input_scale,
                               interpolation=cv2.INTER_LINEAR)

        return self._detect_bg(frame) if self.method == 'bg' else self._detect_onnx(frame)

    def _detect_bg(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        mask = self.bg_sub.apply(frame)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [{'contour': c} for c in cnts if cv2.contourArea(c) >= self.bg_min_area]

    # ---------- ONNX helpers ----------
    @staticmethod
    def _flatten_raw_output(arr: np.ndarray) -> np.ndarray:
        """
        Normalize any raw YOLOv5 output to shape (N, 85).
        Handles shapes like (1, 25200, 85) or (1, 3, 80, 80, 85), etc.
        """
        if arr.ndim == 3:
            # (B, N, 85)
            return arr[0]
        elif arr.ndim >= 4:
            # (B, ..., 85) -> (B, N, 85)
            b = arr.shape[0]
            feat = arr.shape[-1]
            return arr.reshape(b, -1, feat)[0]
        elif arr.ndim == 2:
            # (N, 85)
            return arr
        else:
            raise ValueError(f"Unexpected raw output shape: {arr.shape}")

    def _decode_raw_preds(self, preds: np.ndarray, img_w: int, img_h: int
                          ) -> Tuple[List[List[int]], List[float]]:
        """
        Convert YOLOv5 raw predictions (N,85) -> boxes, scores
        box format expected by OpenCV NMSBoxes: [x, y, w, h]
        """
        boxes: List[List[int]] = []
        scores: List[float] = []
        if preds.size == 0:
            return boxes, scores

        # preds: [cx, cy, w, h, obj, cls0..cls79]
        obj = preds[:, 4]
        cls_probs = preds[:, 5:]
        # best class prob per row
        best_cls = cls_probs.max(axis=1, keepdims=True)  # (N,1)
        conf = (obj.reshape(-1, 1) * best_cls).reshape(-1)  # (N,)

        # Filter by conf threshold early
        keep = conf >= self.conf_th
        if not np.any(keep):
            return boxes, scores
        sel = preds[keep]
        conf_sel = conf[keep]

        cx = sel[:, 0] * img_w
        cy = sel[:, 1] * img_h
        w  = sel[:, 2] * img_w
        h  = sel[:, 3] * img_h

        x1 = (cx - w / 2).astype(np.int32)
        y1 = (cy - h / 2).astype(np.int32)
        ww = w.astype(np.int32)
        hh = h.astype(np.int32)

        for i in range(len(conf_sel)):
            boxes.append([int(x1[i]), int(y1[i]), int(ww[i]), int(hh[i])])
            scores.append(float(conf_sel[i]))
        return boxes, scores

    # ---------- Main ONNX path ----------
    def _detect_onnx(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        h0, w0 = frame.shape[:2]
        if self.model_input_hw is None:
            logger.warning("Model input size unknown, defaulting to 416x416")
            self.model_input_hw = (416, 416)
        in_h, in_w = self.model_input_hw

        # Preprocess
        blob = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        blob = cv2.resize(blob, (in_w, in_h)).astype(np.float32) / 255.0
        if self.layout == 'NHWC':
            blob = np.expand_dims(blob, axis=0)  # [1,H,W,3]
        else:
            blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]  # [1,3,H,W]

        outs = self.sess.run(None, {self.input_name: blob})

        # Case A: NMS-export (boxes, scores, classes[, num])
        if len(outs) >= 3 and outs[0].ndim == 3 and outs[0].shape[-1] == 4:
            boxes_arr = outs[0][0]  # (N,4) in x1y1x2y2 or xywh? Ultralytics NMS is x1y1x2y2
            scores_arr = outs[1][0]  # (N,)
            # Convert boxes to [x,y,w,h]
            x1y1x2y2 = boxes_arr
            xywh = np.zeros_like(x1y1x2y2)
            xywh[:, 0] = x1y1x2y2[:, 0]
            xywh[:, 1] = x1y1x2y2[:, 1]
            xywh[:, 2] = x1y1x2y2[:, 2] - x1y1x2y2[:, 0]
            xywh[:, 3] = x1y1x2y2[:, 3] - x1y1x2y2[:, 1]
            # Scale to original frame if needed (most NMS-export models already output in input scale)
            # Clamp and cast
            boxes = [[int(x), int(y), int(w), int(h)] for x, y, w, h in xywh]
            scores = [float(s) for s in scores_arr.tolist()]
        else:
            # Case B: raw export — flatten to (N,85) then decode
            raw = outs[0]
            preds = self._flatten_raw_output(np.array(raw))
            boxes, scores = self._decode_raw_preds(preds, w0, h0)

        if not boxes:
            return []

        idxs = cv2.dnn.NMSBoxes(boxes, scores, self.conf_th, self.nms_th)
        if len(idxs) == 0:
            return []
        # idxs can be a list of lists or a numpy array with shape (K,1)
        idxs = np.array(idxs).reshape(-1)

        results: List[Dict[str, Any]] = []
        for i in idxs:
            x, y, w, h = boxes[i]
            results.append({
                'bbox': (x, y, x + w, y + h),
                'confidence': scores[i]
            })
        return results
