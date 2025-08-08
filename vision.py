#!/usr/bin/env python3
"""
vision.py — Classical stack default (A1+A2+A3, optional A4) with DNN/BG fallback.

Methods:
  - 'classical' (default): edges/contours + optional Hough + motion + optional optical flow
  - 'dnn': YOLO ONNX (kept; off by default)
  - 'bg': background subtractor only

Detections returned as a list of dicts:
  {'contour': np.ndarray, 'bbox': (x1,y1,x2,y2), 'kind': 'edge'|'motion'|'flow'|'hough', 'score': float}
or, for DNN:
  {'bbox': (x1,y1,x2,y2), 'confidence': float, 'class_id': int, 'kind': 'dnn'}
"""

import os, logging, yaml, math
import numpy as np
import cv2

# Optional ONNX runtime for 'dnn' mode
try:
    import onnxruntime as ort
    _HAS_ORT = True
except Exception:
    _HAS_ORT = False

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')

def _load_cfg():
    try:
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("vision: could not load config.yaml (%s); using defaults", e)
        return {}

def _get(d, path, default):
    cur = d
    try:
        for k in path.split('.'):
            cur = cur[k]
        return default if cur is None else cur
    except Exception:
        return default

# ---------- Letterbox helpers (for DNN) ----------
def _letterbox(img, new_shape=(640, 640), color=(114,114,114)):
    h, w = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    nh, nw = int(new_shape[1]), int(new_shape[0])
    r = min(nh / h, nw / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = nw - new_unpad[0], nh - new_unpad[1]
    dw /= 2; dh /= 2
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)

def _nms_boxes(boxes_xyxy, scores, conf_thres, iou_thres):
    if not boxes_xyxy: return []
    boxes_xywh = [[int(x1),int(y1),int(max(0,x2-x1)),int(max(0,y2-y1))] for (x1,y1,x2,y2) in boxes_xyxy]
    idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores, conf_thres, iou_thres)
    if len(idxs) == 0: return []
    return np.array(idxs).reshape(-1).tolist()

# ---------- Vision ----------
class VisionProcessor:
    def __init__(self, cfg: dict | None = None):
        C = _load_cfg()
        V = (cfg if cfg is not None else (C.get('vision') or {}))

        # MODE
        self.method = str(V.get('method', 'classical')).lower()
        self.debug  = bool(V.get('debug', False))
        self.input_scale = float(V.get('input_scale', 1.0))
        self.run_fps = float(V.get('run_fps', 10))

        # CLASSICAL params
        CL = V.get('classical', {}) or {}
        # Preprocess
        self.clahe_clip = float(_get(CL, 'clahe.clip', 2.0))
        self.clahe_grid = int(_get(CL, 'clahe.grid', 8))
        self.blur_ksize = int(_get(CL, 'blur_ksize', 3))

        # Canny (adaptive by default)
        self.canny_adaptive = bool(_get(CL, 'canny.adaptive', True))
        self.canny_sigma = float(_get(CL, 'canny.sigma', 0.33))
        self.canny_low = int(_get(CL, 'canny.low', 60))
        self.canny_high = int(_get(CL, 'canny.high', 180))

        # Morphology
        self.open_k = int(_get(CL, 'morph.open', 3))
        self.close_k = int(_get(CL, 'morph.close', 0))
        self.open_iter = int(_get(CL, 'morph.open_iter', 1))
        self.close_iter = int(_get(CL, 'morph.close_iter', 0))

        # Contour filter
        self.min_area_px = int(_get(CL, 'contour.min_area_px', 2500))
        self.min_aspect = float(_get(CL, 'contour.min_aspect', 0.1))  # w/h or h/w min

        # Hough
        self.hough_enabled = bool(_get(CL, 'hough.enabled', True))
        self.hough_min_line_frac = float(_get(CL, 'hough.min_line_frac', 0.6))
        self.hough_theta_tol = float(_get(CL, 'hough.theta_tol_deg', 15.0))
        self.hough_thresh = int(_get(CL, 'hough.threshold', 30))
        self.hough_max_gap = int(_get(CL, 'hough.max_gap', 10))
        self.hough_thickness = int(_get(CL, 'hough.thickness', 3))

        # Motion (MOG2)
        self.motion_enabled = bool(_get(CL, 'motion.enabled', True))
        self.motion_history = int(_get(CL, 'motion.history', 300))
        self.motion_varth = float(_get(CL, 'motion.var_threshold', 16.0))
        self.motion_shadows = bool(_get(CL, 'motion.detect_shadows', False))
        self.motion_lr = float(_get(CL, 'motion.learning_rate', 0.005))
        self.motion_open_k = int(_get(CL, 'motion.open', 3))
        self.motion_open_iter = int(_get(CL, 'motion.open_iter', 2))
        self.motion_min_area = int(_get(CL, 'motion.min_area_px', 2000))

        # Optical flow (A4)
        self.flow_enabled = bool(_get(CL, 'flow.enabled', True))
        self.flow_points = int(_get(CL, 'flow.max_corners', 200))
        self.flow_quality = float(_get(CL, 'flow.quality_level', 0.01))
        self.flow_min_dist = int(_get(CL, 'flow.min_distance', 8))
        self.flow_winsize = int(_get(CL, 'flow.win_size', 21))
        self.flow_levels = int(_get(CL, 'flow.levels', 3))
        self.flow_pyr_scale = float(_get(CL, 'flow.pyr_scale', 0.5))
        self.flow_mag_thresh = float(_get(CL, 'flow.mag_thresh_px', 2.0))
        # flow ROI (duplicated here so we don't depend on navigation.yaml)
        self.flow_corridor_frac = float(_get(CL, 'flow.roi_corridor_frac', 0.60))
        self.flow_band_frac = float(_get(CL, 'flow.roi_band_frac', 0.55))

        # BG-only fallback settings (if method='bg')
        BG = V.get('bg', {}) or {}
        self.bg_min_area = int(BG.get('min_area', 5000))
        self.bg_history  = int(BG.get('history', 500))
        self.bg_var_threshold = float(BG.get('var_threshold', 16))
        self.bg_detect_shadows = bool(BG.get('detect_shadows', False))
        self.bg_kernel = int(BG.get('kernel', 3))
        self.bg_open_iter = int(BG.get('morph_open_iter', 2))

        # DNN (kept as option)
        D = V.get('dnn', {}) or {}
        self.model_path = D.get('onnx_model', 'models/yolov5s.onnx')
        self.conf_thres = float(D.get('conf_threshold', 0.25))
        self.nms_thres = float(D.get('nms_threshold', 0.5))
        self.use_gpu   = bool(D.get('use_gpu', False))
        self.model_input_size = tuple(D.get('input_size', [640, 640]))
        self.pad_color = tuple(V.get('preprocess', {}).get('pad_color', [114,114,114]))

        # Internals
        self.prev_gray = None
        self.mog2 = None
        self.sess = None
        self.input_name = None
        self.is_nchw = True
        self.model_hw = (640, 640)

        # Init paths
        if self.method == 'classical':
            self._init_classical()
        elif self.method == 'dnn':
            self._init_dnn()
            if self.sess is None:
                logger.warning("DNN init failed → using BG")
                self.method = 'bg'
                self._init_bg()
        elif self.method == 'bg':
            self._init_bg()
        else:
            logger.warning("Unknown vision.method '%s' → using 'classical'", self.method)
            self.method = 'classical'
            self._init_classical()

    # ---------- Classical ----------
    def _init_classical(self):
        if self.motion_enabled:
            self.mog2 = cv2.createBackgroundSubtractorMOG2(
                history=self.motion_history,
                varThreshold=self.motion_varth,
                detectShadows=self.motion_shadows
            )
        logger.info("Vision[classical]: canny=%s sigma=%.2f open=%d×%d close=%d×%d "
                    "contour_min=%d hough=%s motion=%s flow=%s",
                    self.canny_adaptive, self.canny_sigma,
                    self.open_k, self.open_iter, self.close_k, self.close_iter,
                    self.min_area_px, self.hough_enabled, self.motion_enabled, self.flow_enabled)

    def _pre_gray(self, frame):
        # optional downscale for speed
        if self.input_scale != 1.0:
            frame = cv2.resize(frame, None, fx=self.input_scale, fy=self.input_scale, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.clahe_clip > 0:
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(self.clahe_grid, self.clahe_grid))
            gray = clahe.apply(gray)
        if self.blur_ksize > 0:
            k = max(1, self.blur_ksize | 1)  # ensure odd
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        return frame, gray

    def _canny_edges(self, gray):
        if self.canny_adaptive:
            v = np.median(gray)
            low = int(max(0, (1.0 - self.canny_sigma) * v))
            high = int(min(255, (1.0 + self.canny_sigma) * v))
        else:
            low, high = self.canny_low, self.canny_high
        edges = cv2.Canny(gray, low, high)
        # morphology
        if self.open_k > 0 and self.open_iter > 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (self.open_k, self.open_k))
            edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, k, iterations=self.open_iter)
        if self.close_k > 0 and self.close_iter > 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (self.close_k, self.close_k))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=self.close_iter)
        return edges

    def _contours_from_binary(self, mask, kind='edge', min_area=None):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        min_area = self.min_area_px if min_area is None else min_area
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area: continue
            x, y, w, h = cv2.boundingRect(c)
            if w == 0 or h == 0: continue
            asp = max(w / h, h / w)
            if asp < self.min_aspect: continue
            dets.append({'contour': c, 'bbox': (x, y, x + w, y + h), 'kind': kind, 'score': float(area)})
        return dets

    def _hough_horizontal(self, edges, frame_shape):
        if not self.hough_enabled: return []
        h, w = frame_shape[:2]
        min_len = int(self.hough_min_line_frac * w)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=self.hough_thresh,
                                minLineLength=min_len, maxLineGap=self.hough_max_gap)
        dets = []
        if lines is None: return dets
        for (x1, y1, x2, y2) in lines[:,0,:]:
            ang = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            if ang <= self.hough_theta_tol or abs(ang - 180) <= self.hough_theta_tol:
                # Represent as a thin rectangle around the line so nav can see it as a contour
                t = self.hough_thickness
                x_min, y_min = min(x1, x2), min(y1, y2)
                x_max, y_max = max(x1, x2), max(y1, y2)
                x_min = max(0, x_min); y_min = max(0, y_min - t)
                x_max = min(w-1, x_max); y_max = min(h-1, y_max + t)
                rect = np.array([[[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]]], dtype=np.int32)
                dets.append({'contour': rect, 'bbox': (x_min,y_min,x_max,y_max), 'kind': 'hough', 'score': float(x_max-x_min)})
        return dets

    def _motion_contours(self, frame):
        if not (self.motion_enabled and self.mog2 is not None):
            return []
        fg = self.mog2.apply(frame, learningRate=self.motion_lr)
        if self.motion_open_k > 0 and self.motion_open_iter > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.motion_open_k, self.motion_open_k))
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=self.motion_open_iter)
        return self._contours_from_binary(fg, kind='motion', min_area=self.motion_min_area)

    def _flow_detection(self, gray, frame_shape):
        if not self.flow_enabled:
            self.prev_gray = gray  # keep updated for future
            return []

        h, w = frame_shape[:2]
        # forward ROI similar to nav (duplicated so we don't import nav config)
        x0 = int((1.0 - self.flow_corridor_frac) * 0.5 * w)
        x1 = w - x0
        y0 = int((1.0 - self.flow_band_frac) * h)
        roi = gray[y0:h, x0:x1]

        dets = []
        if self.prev_gray is not None:
            prev_roi = self.prev_gray[y0:h, x0:x1]
            p0 = cv2.goodFeaturesToTrack(prev_roi, maxCorners=self.flow_points,
                                         qualityLevel=self.flow_quality, minDistance=self.flow_min_dist)
            if p0 is not None and len(p0) >= 8:
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_roi, roi, p0, None,
                                                       winSize=(self.flow_winsize, self.flow_winsize),
                                                       maxLevel=self.flow_levels,
                                                       criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 20, 0.03))
                good = (st.reshape(-1) == 1)
                if np.count_nonzero(good) >= 6:
                    flow = p1[good] - p0[good]
                    mags = np.linalg.norm(flow, axis=2).reshape(-1)
                    med_mag = float(np.median(mags)) if mags.size else 0.0
                    if self.debug:
                        logger.debug("FLOW: points=%d good=%d median=%.2fpx", len(p0), np.count_nonzero(good), med_mag)
                    if med_mag >= self.flow_mag_thresh:
                        # Emit a big box over the forward ROI to trigger nav
                        dets.append({'contour': np.array([[[x0,y0],[x1,y0],[x1,h],[x0,h]]], dtype=np.int32),
                                     'bbox': (x0,y0,x1,h), 'kind': 'flow', 'score': med_mag})
        # update prev
        self.prev_gray = gray
        return dets

    def _detect_classical(self, frame):
        # Preprocess
        frame2, gray = self._pre_gray(frame)
        edges = self._canny_edges(gray)
        # A1: edge → contours
        edge_dets = self._contours_from_binary(edges, kind='edge', min_area=self.min_area_px)
        # A2: Hough horizontal (thin rectangles)
        hough_dets = self._hough_horizontal(edges, frame2.shape)
        # A3: motion MOG2
        motion_dets = self._motion_contours(frame2)
        # A4: sparse optical flow in forward ROI
        flow_dets = self._flow_detection(gray, frame2.shape)

        dets = edge_dets + hough_dets + motion_dets + flow_dets
        if self.debug:
            logger.info("CLASSICAL: edge=%d, hough=%d, motion=%d, flow=%d -> total=%d",
                        len(edge_dets), len(hough_dets), len(motion_dets), len(flow_dets), len(dets))
        return dets

    # ---------- BG-only ----------
    def _init_bg(self):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=self.bg_history,
                                                     varThreshold=self.bg_var_threshold,
                                                     detectShadows=self.bg_detect_shadows)
        k = max(1, int(self.bg_kernel))
        self._bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self._bg_open_iter = max(0, int(self.bg_open_iter))
        logger.info("Initialized BG: history=%d varTh=%s kernel=%d open_iter=%d shadows=%s",
                    self.bg_history, self.bg_var_threshold, k, self._bg_open_iter, self.bg_detect_shadows)

    def _detect_bg(self, frame):
        fg = self.bg.apply(frame)
        if self._bg_open_iter > 0:
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._bg_kernel, iterations=self._bg_open_iter)
        return self._contours_from_binary(fg, kind='motion', min_area=self.bg_min_area)

    # ---------- DNN (kept for daytime toggle) ----------
    def _init_dnn(self):
        if not _HAS_ORT or not os.path.isfile(self.model_path):
            logger.warning("DNN unavailable or model missing (%s)", self.model_path)
            return
        providers = ['CUDAExecutionProvider','CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
        try:
            self.sess = ort.InferenceSession(self.model_path, providers=providers)
            inp = self.sess.get_inputs()[0]
            self.input_name = inp.name
            ishape = list(inp.shape)
            if (len(ishape) == 4) and (ishape[-1] == 3 or str(ishape[-1]) == '3'):
                self.is_nchw = False
                H = ishape[1] if isinstance(ishape[1], int) else self.model_input_size[1]
                W = ishape[2] if isinstance(ishape[2], int) else self.model_input_size[0]
            else:
                self.is_nchw = True
                H = ishape[2] if len(ishape)>2 and isinstance(ishape[2], int) else self.model_input_size[1]
                W = ishape[3] if len(ishape)>3 and isinstance(ishape[3], int) else self.model_input_size[0]
            self.model_hw = (int(H), int(W))
            logger.info("Loaded ONNX '%s' layout=%s inputHW=(%d,%d)",
                        os.path.basename(self.model_path), 'NCHW' if self.is_nchw else 'NHWC',
                        self.model_hw[0], self.model_hw[1])
        except Exception as e:
            logger.warning("Failed to load ONNX (%s)", e); self.sess = None

    def _detect_dnn(self, frame):
        if self.sess is None:
            return []
        img = frame
        if self.input_scale != 1.0:
            img = cv2.resize(img, None, fx=self.input_scale, fy=self.input_scale, interpolation=cv2.INTER_LINEAR)
        blob, ratio, (dw, dh) = _letterbox(img, new_shape=(self.model_hw[1], self.model_hw[0]), color=self.pad_color)
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.is_nchw:
            blob = np.transpose(blob, (2,0,1))
        blob = np.expand_dims(blob, 0)
        try:
            out = self.sess.run(None, {self.input_name: blob})
        except Exception as e:
            logger.warning("ONNX run failed: %s", e); return []
        x = out[0]
        if x.ndim == 3 and x.shape[0] == 1: x = x[0]
        if x.ndim == 5:
            b,a,c,h,w = x.shape; x = x.reshape(b,a,-1); x = np.transpose(x,(0,2,1)); x = x[0]
        if x.ndim == 2 and x.shape[0] in (84,85) and x.shape[1] not in (84,85):
            x = x.T
        if x.ndim != 2 or x.shape[1] not in (84,85): return []
        has_obj = (x.shape[1]==85)
        boxes, scores, classes = [], [], []
        oh, ow = frame.shape[:2]; r = ratio; xdw, xdh = dw, dh
        for row in x:
            cx,cy,w,h = row[0],row[1],row[2],row[3]
            if has_obj:
                obj = row[4]; cls_scores = row[5:]; cid = int(np.argmax(cls_scores)); conf = float(obj*cls_scores[cid])
            else:
                cls_scores = row[4:]; cid = int(np.argmax(cls_scores)); conf = float(cls_scores[cid])
            if conf < self.conf_thres: continue
            x1 = (cx - w/2 - xdw)/r; y1 = (cy - h/2 - xdh)/r
            x2 = (cx + w/2 - xdw)/r; y2 = (cy + h/2 - xdh)/r
            x1 = max(0,min(ow-1,x1)); y1 = max(0,min(oh-1,y1)); x2 = max(0,min(ow-1,x2)); y2 = max(0,min(oh-1,y2))
            boxes.append((x1,y1,x2,y2)); scores.append(conf); classes.append(cid)
        keep = _nms_boxes(boxes, scores, self.conf_thres, self.nms_thres)
        return [{'bbox':(int(boxes[i][0]),int(boxes[i][1]),int(boxes[i][2]),int(boxes[i][3])),
                 'confidence':float(scores[i]), 'class_id':int(classes[i]), 'kind':'dnn'} for i in keep]

    # ---------- Public ----------
    def detect(self, frame):
        if self.method == 'classical':
            return self._detect_classical(frame)
        elif self.method == 'dnn':
            return self._detect_dnn(frame)
        else:
            return self._detect_bg(frame)
