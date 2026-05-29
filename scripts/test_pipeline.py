"""Test the vision pipeline against all saved pool images."""
import sys
import cv2
from pathlib import Path

sys.path.insert(0, "/home/ben/autoboat")
from vision.pipeline import analyze, annotate

IN_DIR = Path("/home/ben/autoboat/pool_images")
OUT_DIR = Path("/home/ben/autoboat/pool_images/analyzed")
OUT_DIR.mkdir(exist_ok=True, parents=True)

images = sorted(IN_DIR.glob("*.jpg"))
if not images:
    print(f"No images found in {IN_DIR}")
    sys.exit(1)

print(f"{'Image':<40} {'Center':>8} {'Best':>5}  Zones (L-R)")
print("-" * 75)
for path in images:
    img = cv2.imread(str(path))
    if img is None:
        print(f"  skipping {path.name} (could not read)")
        continue
    result = analyze(img, is_rgb=False)
    vis = annotate(img, result, is_rgb=False)
    out = OUT_DIR / f"analyzed_{path.name}"
    cv2.imwrite(str(out), vis)

    print(f"{path.name:<40} {result.center_depth_pct:>6.1f}%  "
          f"{result.best_zone:>5}  {result.zones}")

print(f"\nAnnotated images in {OUT_DIR}/")
