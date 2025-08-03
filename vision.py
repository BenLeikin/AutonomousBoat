# vision.py
# Enhanced VisionProcessor with optional deep learning (YOLO) and improved background-subtraction pipeline

import cv2
import numpy as np
import sys

class VisionProcessor:
    """
    VisionProcessor supports two modes of obstacle detection:
      1) background subtraction + contour processing
      2) deep learning via YOLO (OpenCV DNN)

    Methods:
      - detect(frame): returns a list of detections (contours or bboxes)
      - annotate(frame, detections): overlays detection results on frame
    """

    def __init__(
        self,
        method='dnn',                 # 'dnn' for YOLO, 'bg' for background subtraction
        # --- Background-subtraction params ---
        min_area=5000,
        history=500,
        var_threshold=16,
        # --- YOLO params with built-in defaults ---
        yolo_cfg='models/yolo/yolov3-tiny.cfg',
        yolo_weights='models/yolo/yolov3-tiny.weights',
        yolo_names='models/yolo/coco.names',
        input_size=(416, 416),
        conf_threshold=0.5,
        nms_threshold=0.4,
        use_gpu=False
    ):
        self.method = method
        self.min_area = min_area

        # Background subtractor setup
        if method == 'bg':
            self._init_bg(history, var_threshold)
            return

        # YOLO DNN setup
        if method == 'dnn':
            try:
                self.net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
            except cv2.error as e:
                print(f"[Warning] Failed to load YOLO model: {e}. Falling back to background-subtraction.", file=sys.stderr)
                self.method = 'bg'
                self._init_bg(history, var_threshold)
                return

            if use_gpu:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            else:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            # Load class labels
            try:
                with open(yolo_names, 'r') as f:
                    self.classes = [c.strip() for c in f.readlines()]
            except Exception as e:
                print(f"[Warning] Failed to load class names: {e}. DNN detections will show only IDs.", file=sys.stderr)
                self.classes = None

            self.input_width, self.input_height = input_size
            self.conf_threshold = conf_threshold
            self.nms_threshold = nms_threshold

    def _init_bg(self, history, var_threshold):
        """Initialize background subtractor."""
        self.backsub = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def detect(self, frame):
        """
        Dispatch to the chosen detection method.
        Returns:
          - For 'bg': list of cv2 contours (hulls)
          - For 'dnn': list of dicts {'bbox': (x,y,w,h), 'confidence': float, 'class_id': int, 'label': str}
        """
        if self.method == 'bg':
            return self._detect_bg(frame)
        else:
            return self._detect_dnn(frame)

    def _detect_bg(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = self.backsub.apply(blur)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        _, thresh = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        processed = []
        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            hull = cv2.convexHull(approx)
            processed.append(hull)
        return processed

    def _detect_dnn(self, frame):
        blob = cv2.dnn.blobFromImage(
            frame,
            1/255.0,
            (self.input_width, self.input_height),
            swapRB=True,
            crop=False
        )
        self.net.setInput(blob)
        ln = self.net.getUnconnectedOutLayersNames()
        layer_outputs = self.net.forward(ln)

        H, W = frame.shape[:2]
        boxes, confidences, class_ids = [], [], []

        # Parse detections
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence > self.conf_threshold:
                    cx, cy, w, h = (detection[0:4] * [W, H, W, H]).astype('int')
                    x = int(cx - w / 2)
                    y = int(cy - h / 2)
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        detections = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, w, h = boxes[i]
                label = self.classes[class_ids[i]] if self.classes else str(class_ids[i])
                detections.append({
                    'bbox': (x, y, w, h),
                    'confidence': confidences[i],
                    'class_id': class_ids[i],
                    'label': label
                })
        return detections

    def annotate(self, frame, detections, color=(0, 255, 0), thickness=2):
        """
        Draws detections on the frame.
        - For contours: draw the contour outline
        - For DNN: draw bounding boxes and labels
        """
        if self.method == 'bg':
            for cnt in detections:
                cv2.drawContours(frame, [cnt], -1, color, thickness)
        else:
            for det in detections:
                x, y, w, h = det['bbox']
                label = f"{det['label']}:{det['confidence']:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
                cv2.putText(
                    frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )
        return frame

# If this module is run directly, instantiate with YOLO-tiny defaults
if __name__ == '__main__':
    vision = VisionProcessor()
    print("VisionProcessor instantiated with method:", vision.method)
