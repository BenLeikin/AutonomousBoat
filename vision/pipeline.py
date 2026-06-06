"""
Vision pipeline for the autoboat.

Takes a BGR or RGB frame, returns a navigation summary:
  - mask: binary water mask
  - depths: per-column free-water depth (rows above bottom)
  - zones: 5 horizontal zones with median water depth
  - best_zone: index of zone with most open water (0=leftmost, 4=rightmost)
  - center_depth_pct: free water ahead in the center column, as % of frame height
"""
from dataclasses import dataclass
import cv2
import numpy as np


# Calibrated for OV5647 + cloudy evening light, AutoBoat pool.
DEFAULT_THRESHOLDS = {
    "h_lo": 20, "h_hi": 50,
    "s_lo": 10, "s_hi": 150,
    "v_lo": 60, "v_hi": 240,
}

GAP_TOLERANCE = 5
BOTTOM_SKIP_MAX = 30
NUM_ZONES = 5
ROI_TOP_FRAC = 0.5

DEFAULT_METHOD = "texture"   # "texture" (smoothness-based) or "color" (legacy HSV)
TEXTURE_WINDOW = 9           # local window (px) for texture-energy smoothing
VERT_CLOSE = 15              # vertical close (px) to bridge in-column reflection/glare holes
DEPTH_SMOOTH = 15            # cross-column median window to repair reflection-spiked columns


@dataclass
class NavResult:
    mask: np.ndarray
    depths: np.ndarray
    zones: list
    best_zone: int
    center_depth_pct: float
    roi_top: int


def threshold_water(img_bgr, thresholds=None):
    """Apply HSV threshold and morphological cleanup. Returns binary mask."""
    t = thresholds or DEFAULT_THRESHOLDS
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([t["h_lo"], t["s_lo"], t["v_lo"]])
    upper = np.array([t["h_hi"], t["s_hi"], t["v_hi"]])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def texture_water_mask(img_bgr, window=TEXTURE_WINDOW):
    """Binary water mask based on smoothness rather than colour.

    Pool water is a low-texture surface; walls, deck, furniture and plants are
    full of edges (window frames, door lines, slats, foliage). Colour can't tell
    pale water from pale stucco or concrete, but texture can: water is smooth,
    structure is busy. We measure local gradient energy and call the smooth
    regions water, using Otsu so the smooth/busy split adapts to the lighting
    instead of relying on a fixed magic number.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    energy = cv2.blur(cv2.magnitude(gx, gy), (window, window))
    energy8 = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Water = low energy, so threshold inverted: 255 where energy is below Otsu.
    _, mask = cv2.threshold(energy8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def segment_water(img_bgr, method=None, thresholds=None):
    """Binary water mask. method='texture' (default, smoothness) or 'color'
    (legacy HSV box, kept for comparison)."""
    if (method or DEFAULT_METHOD) == "color":
        return threshold_water(img_bgr, thresholds)
    return texture_water_mask(img_bgr)


def _bridge_columns(mask, k=VERT_CLOSE):
    """Vertical morphological close: fills holes shorter than k pixels within a
    column so a reflection or glare band inside the water doesn't read as the
    waterline. A real wall is far taller than k, so it isn't bridged."""
    if k <= 1:
        return mask
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, 1), np.uint8))


def _smooth_depths(depths, k=DEPTH_SMOOTH):
    """Median-filter per-column depths across columns. The waterline is spatially
    coherent, so columns spiked low by a leftover reflection get repaired by their
    neighbours, while the left-to-right trend the zones depend on survives (the
    window is far narrower than a zone)."""
    if k <= 1 or depths.size < k:
        return depths
    if k % 2 == 0:
        k += 1
    pad = k // 2
    padded = np.pad(depths, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, k)
    return np.median(windows, axis=1).astype(depths.dtype)


def _water_depth_per_column(mask, gap_tolerance=GAP_TOLERANCE,
                            bottom_skip_max=BOTTOM_SKIP_MAX):
    """
    For each column, find the highest contiguous water region from the bottom.

    Tolerates up to `bottom_skip_max` non-water rows at the very bottom
    of a column before the first water pixel (handles glare strips).
    Once water is found, tolerates up to `gap_tolerance` consecutive
    non-water rows (handles reflection holes) before declaring an obstacle.

    Vectorized across columns: iterates rows, applies per-row numpy ops
    to all columns simultaneously. Matches the original loop's semantics
    exactly.
    """
    h, w = mask.shape
    if h == 0:
        return np.zeros(w, dtype=np.int32)

    is_water = (mask > 0)[::-1, :]  # row 0 = bottom of frame

    has_seen_water = np.zeros(w, dtype=bool)
    gap = np.zeros(w, dtype=np.int32)
    depth = np.zeros(w, dtype=np.int32)
    stopped = np.zeros(w, dtype=bool)

    for r in range(h):
        is_w = is_water[r, :]
        active = ~stopped

        # Update gap counter: reset on water, increment on non-water
        gap = np.where(is_w, 0, gap + 1)

        # Record water depth where active and this row has water
        water_here = is_w & active
        depth = np.where(water_here, r + 1, depth)
        has_seen_water |= water_here

        # Bottom-skip stop: still no water seen and we've exceeded the budget
        if r >= bottom_skip_max:
            stopped |= active & (~is_w) & (~has_seen_water)

        # Gap-tolerance stop: gap exceeds tolerance after water has been seen
        stopped |= active & has_seen_water & (gap > gap_tolerance)

    return depth


def analyze(frame, thresholds=None, is_rgb=False, method=None):
    """Main entry point. Returns NavResult. `method` selects segmentation:
    'texture' (default) or 'color' (legacy HSV)."""
    if is_rgb:
        img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = frame

    full_h, full_w = img_bgr.shape[:2]
    roi_top = int(full_h * ROI_TOP_FRAC)
    roi = img_bgr[roi_top:, :]

    mask = segment_water(roi, method=method, thresholds=thresholds)
    mask = _bridge_columns(mask)
    depths = _water_depth_per_column(mask)
    depths = _smooth_depths(depths)

    zone_w = full_w // NUM_ZONES
    zones = []
    for z in range(NUM_ZONES):
        slice_ = depths[z * zone_w:(z + 1) * zone_w]
        zones.append(int(np.median(slice_)) if len(slice_) > 0 else 0)
    best_zone = int(np.argmax(zones))

    center_col = full_w // 2
    center_depth = int(depths[center_col])
    center_depth_pct = 100.0 * center_depth / full_h

    return NavResult(
        mask=mask, depths=depths, zones=zones, best_zone=best_zone,
        center_depth_pct=center_depth_pct, roi_top=roi_top,
    )


def annotate(frame, result, is_rgb=False):
    """Draw the navigation result onto a copy of the frame for debugging."""
    if is_rgb:
        vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        vis = frame.copy()

    h, w = vis.shape[:2]

    for col in range(w):
        boundary_row = (h - 1) - result.depths[col]
        if 0 <= boundary_row < h:
            cv2.circle(vis, (col, boundary_row), 1, (0, 255, 255), -1)

    center_col = w // 2
    center_depth_rows = int(result.center_depth_pct / 100.0 * h)
    cv2.line(vis, (center_col, h - 1),
             (center_col, h - 1 - center_depth_rows), (0, 0, 255), 2)

    zone_w = w // NUM_ZONES
    bz = result.best_zone
    bz_depth = result.zones[bz]
    cv2.rectangle(vis,
                  (bz * zone_w, h - 1 - bz_depth),
                  ((bz + 1) * zone_w, h - 1),
                  (0, 255, 0), 2)

    cv2.putText(vis, f"center: {result.center_depth_pct:.0f}%  best: zone {bz}",
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return vis
