"""
Visualization and metrics utilities.

Includes:
- Drawing detections.
- Road shadow metrics.
- LED matrix rendering.
"""

import cv2
import numpy as np

from .constants import VEHICLE_CLASS_IDS


def draw_detections(frame, detections, class_names):
    """
    Draw YOLO detections on the frame (only vehicle classes).

    Args:
        frame: BGR image.
        detections: N x 6 array [x1,y1,x2,y2,conf,cls] (can be None).
        class_names: List of class name strings from YOLO model.

    Returns:
        out: BGR image with bounding boxes and labels drawn.
    """
    out = frame.copy()
    if detections is None:
        return out

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        if cls not in VEHICLE_CLASS_IDS:
            continue

        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
        cv2.rectangle(out, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
        label = f"{class_names[cls]} {conf:.2f}"
        cv2.putText(
            out,
            label,
            (x1i, max(0, y1i - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return out


def compute_road_shadow_metrics(road_mask, mask):
    """
    Compute metrics for how much of the road is dimmed.

    Args:
        road_mask: (H, W) float32 mask in [0, 1].
        mask: (H, W) float32 beam mask in [0, 1].

    Returns:
        road_pixel_count: Number of pixels inside road_mask.
        shadow_pixel_count: Number of pixels on road where mask < ~1.
        shadow_ratio: shadow_pixel_count / road_pixel_count.
    """
    roi = road_mask > 0.5
    if not np.any(roi):
        return 0, 0, 0.0

    road_pixel_count = int(np.sum(roi))
    shadow_pixels = np.logical_and(roi, mask < 0.999)
    shadow_pixel_count = int(np.sum(shadow_pixels))
    shadow_ratio = shadow_pixel_count / road_pixel_count
    return road_pixel_count, shadow_pixel_count, shadow_ratio


def render_led_matrix(led_vals, cell_size=20, label_text=None):
    """
    Render a grayscale LED matrix visualization (for one lamp or combined).

    Args:
        led_vals: (rows, cols) array in [0, 1].
        cell_size: Pixel size of each cell in the visualization.
        label_text: Optional label (e.g., "L", "C", "R") to draw on top.

    Returns:
        img: BGR image representing the matrix.
    """
    rows, cols = led_vals.shape
    h = rows * cell_size
    w = cols * cell_size
    img = np.zeros((h, w, 3), dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            v = led_vals[r, c]
            brightness = int(255 * v)
            y0, y1 = r * cell_size, (r + 1) * cell_size
            x0, x1 = c * cell_size, (c + 1) * cell_size
            img[y0:y1, x0:x1] = (brightness, brightness, brightness)
            cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (40, 40, 40), 1)

    if label_text is not None:
        cv2.putText(
            img,
            label_text,
            (5, cell_size - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
    return img
