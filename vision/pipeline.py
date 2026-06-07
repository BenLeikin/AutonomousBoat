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
import os
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
CONNECT_FROM_BOTTOM = True   # keep only water connected to the frame bottom, dropping
                             # floating false-water patches (tile faces, surface reflections)


# Optional boat/hull mask. When a wide lens sees the boat's own deck and bows in
# the lower frame, those pixels must be excluded or the smooth deck reads as water
# (flooring center depth and disabling blocked-detection). Set AUTOBOAT_BOAT_MASK
# to a grayscale PNG where white (>127) marks the boat. The standard narrow lens
# leaves it unset, so this is a no-op there.
_BOAT_MASK_PATH = os.environ.get("AUTOBOAT_BOAT_MASK")
_boat_mask_cache = {"path": None, "mask": None}


def _load_boat_mask(shape):
    p = _BOAT_MASK_PATH
    if not p:
        return None
    c = _boat_mask_cache
    if c["path"] == p and c["mask"] is not None and c["mask"].shape == shape:
        return c["mask"]
    try:
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        if m.shape != shape:
            m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        c["path"] = p
        c["mask"] = m > 127
        return c["mask"]
    except Exception:
        return None


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


def _keep_bottom_connected(mask):
    """Keep only the water connected to the bottom edge of the ROI.

    The boat floats in the water, so the real water body always touches the
    bottom of the frame. Isolated 'water' blobs that sit above the waterline,
    smooth tile faces between grout lines, sky, or surface reflections that read
    as low-texture, are not connected to that body. Dropping anything not
    reachable from the bottom row stops those false patches from inflating a
    zone's openness. Run this AFTER _bridge_columns so in-column reflection holes
    are already closed and the real water stays one connected region.
    """
    if mask is None or mask.size == 0:
        return mask
    num, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return mask
    bottom = np.unique(labels[-1, :])
    bottom = bottom[bottom != 0]
    if bottom.size == 0:
        return np.zeros_like(mask)
    return np.where(np.isin(labels, bottom), np.uint8(255), np.uint8(0))


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
                            bottom_skip_max=BOTTOM_SKIP_MAX, bottom_offset=None):
    """
    For each column, find the highest contiguous water region from the bottom.

    `bottom_offset` (per-column) is the number of rows at the bottom of each
    column occupied by the boat's own hull; those rows are skipped so the scan
    begins above the boat and depth is measured as water rows above the hull.
    """
    h, w = mask.shape
    if h == 0:
        return np.zeros(w, dtype=np.int32)

    is_water = (mask > 0)[::-1, :]  # row 0 = bottom of frame
    if bottom_offset is None:
        bottom_offset = np.zeros(w, dtype=np.int32)

    has_seen_water = np.zeros(w, dtype=bool)
    gap = np.zeros(w, dtype=np.int32)
    depth = np.zeros(w, dtype=np.int32)
    stopped = np.zeros(w, dtype=bool)

    for r in range(h):
        in_hull = r < bottom_offset             # this row is the boat in this column
        is_w = is_water[r, :] & ~in_hull        # the hull is never water
        active = ~stopped & ~in_hull            # do not scan within the hull

        # Gap counter advances only outside the hull
        gap = np.where(in_hull, gap, np.where(is_w, 0, gap + 1))

        # Depth measured as water rows ABOVE the hull
        water_here = is_w & active
        depth = np.where(water_here, (r + 1) - bottom_offset, depth)
        has_seen_water |= water_here

        # Bottom-skip budget counts from the top of the hull
        past_skip = (r - bottom_offset) >= bottom_skip_max
        stopped |= active & (~is_w) & (~has_seen_water) & past_skip

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

    # Exclude the boat's own hull/deck if a mask is configured (wide lens). Zero
    # it from the water mask and compute, per column, how many bottom rows are
    # hull so the depth scan starts above the boat.
    boat = _load_boat_mask(img_bgr.shape[:2])
    bottom_offset = None
    if boat is not None:
        boat_roi = boat[roi_top:, :]
        mask[boat_roi] = 0
        roi_h = mask.shape[0]
        any_hull = boat_roi.any(axis=0)
        topmost = np.argmax(boat_roi, axis=0)          # first hull row from the top
        bottom_offset = np.where(any_hull, roi_h - topmost, 0).astype(np.int32)

    # connect-from-bottom assumes water touches the frame bottom; with the hull
    # masked there it would drop everything, so it is bypassed when a boat mask
    # is active.
    if CONNECT_FROM_BOTTOM and boat is None:
        mask = _keep_bottom_connected(mask)
    depths = _water_depth_per_column(mask, bottom_offset=bottom_offset)
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
