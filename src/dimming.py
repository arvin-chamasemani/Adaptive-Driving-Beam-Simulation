"""
Dimming and beam mask logic.

Includes:
- Distance-aware dimming factor.
- Gaussian / rectangular dimming kernels.
- Global beam mask computation.
- Applying beam mask to images.
"""

import numpy as np

from .constants import (
    DIM_SOFT_MIN,
    DIM_SOFT_MAX_ADD,
    HEIGHT_NORM_DENOM_RATIO,
    HEIGHT_NORM_MIN,
    HEIGHT_NORM_MAX,
    HOTSPOT_HEIGHT_RATIO,
    EPS,
    VEHICLE_CLASS_IDS,
)
import cv2  # used for array operations, no drawing here but kept for consistency


def compute_distance_aware_dim_factor(
    bbox,
    img_height,
    base_dim_factor=DIM_SOFT_MIN,
    max_additional=DIM_SOFT_MAX_ADD,
):
    """
    Compute a dimming factor based on object (vehicle) height in the image.

    Larger bounding boxes (closer vehicles) will get stronger dimming.
    The factor is clamped to [base_dim_factor, base_dim_factor + max_additional].

    Args:
        bbox: (x1, y1, x2, y2) bounding box in pixels.
        img_height: Image height in pixels.
        base_dim_factor: Minimum dimming factor (>=0).
        max_additional: Maximum additional dimming to add based on distance.

    Returns:
        dim_factor: Float scalar in [base_dim_factor, base_dim_factor + max_additional].
    """
    x1, y1, x2, y2 = bbox
    bh = max(1.0, float(y2 - y1))
    denom = max(1.0, HEIGHT_NORM_DENOM_RATIO * img_height)
    height_norm = np.clip(bh / denom, HEIGHT_NORM_MIN, HEIGHT_NORM_MAX)
    dim_factor = base_dim_factor + (1.0 - height_norm) * max_additional
    dim_factor = float(
        np.clip(dim_factor, base_dim_factor, base_dim_factor + max_additional)
    )
    return dim_factor


def add_gaussian_dim(mask, road_mask, bbox, dim_factor):
    """
    Apply a soft Gaussian dimming region near the top of bbox.
    Computed only on a local ROI for efficiency.

    Args:
        mask: Beam brightness mask (H, W) float32 in [0, 1].
        road_mask: Road ROI mask (H, W) float32 in [0, 1].
        bbox: (x1, y1, x2, y2) bounding box in pixels.
        dim_factor: Minimum brightness value in the hotspot region.
    """
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = map(float, bbox)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    # Hotspot center: slightly above the center (towards top of box).
    cx = (x1 + x2) / 2.0
    cy = y1 + HOTSPOT_HEIGHT_RATIO * bh

    sigma_x = 0.5 * bw
    sigma_y = 0.6 * bh

    pad_x = int(3 * sigma_x)
    pad_y = int(3 * sigma_y)

    x0 = max(0, int(cx - pad_x))
    x1r = min(w, int(cx + pad_x))
    y0 = max(0, int(cy - pad_y))
    y1r = min(h, int(cy + pad_y))
    if x0 >= x1r or y0 >= y1r:
        return

    xs = np.arange(x0, x1r)
    ys = np.arange(y0, y1r)
    X, Y = np.meshgrid(xs, ys)

    dx2 = (X - cx) ** 2
    dy2 = (Y - cy) ** 2
    G = np.exp(
        -(
            dx2 / (2.0 * (sigma_x + EPS) ** 2)
            + dy2 / (2.0 * (sigma_y + EPS) ** 2)
        )
    )

    # Brightness is 1 outside hotspot and ~dim_factor at hotspot center.
    local_brightness = dim_factor + (1.0 - dim_factor) * (1.0 - G)

    roi = road_mask[y0:y1r, x0:x1r] > 0.5
    if not np.any(roi):
        return

    sub = mask[y0:y1r, x0:x1r]
    sub_new = np.where(
        roi, np.minimum(sub, local_brightness.astype(np.float32)), sub
    )
    mask[y0:y1r, x0:x1r] = sub_new


def add_rect_dim_fast(mask, road_mask, bbox, dim_factor):
    """
    Apply a fast, rectangular dimming region for the bounding box.

    Args:
        mask: Beam brightness mask (H, W) float32 in [0, 1].
        road_mask: Road ROI mask (H, W) float32 in [0, 1].
        bbox: (x1, y1, x2, y2) bounding box in pixels.
        dim_factor: Target brightness value inside the box.
    """
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    # Clamp to image bounds.
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x1 >= x2 or y1 >= y2:
        return

    roi = road_mask[y1:y2, x1:x2] > 0.5
    if not np.any(roi):
        return

    sub = mask[y1:y2, x1:x2]
    sub[roi] = np.minimum(sub[roi], float(dim_factor))
    mask[y1:y2, x1:x2] = sub


def compute_beam_mask(
    frame_shape,
    detections,
    road_mask,
    base_dim_factor=DIM_SOFT_MIN,
    expand_ratio=0.25,
    fast=False,
):
    """
    Build a global beam brightness mask based on vehicle detections.

    Args:
        frame_shape: Shape of the frame (H, W, C).
        detections: YOLO detections (N x 6: x1,y1,x2,y2,conf,cls) or None.
        road_mask: Road ROI mask (H, W) float32 in [0, 1].
        base_dim_factor: Minimum dimming factor.
        expand_ratio: How much to enlarge bounding boxes before dimming.
        fast: If True use fast rectangular dimming, else Gaussian.

    Returns:
        mask: Float32 (H, W) in [0, 1] where lower values indicate dimmed areas.
    """
    h, w = frame_shape[:2]
    mask = np.ones((h, w), dtype=np.float32)

    if detections is None or len(detections) == 0:
        return mask

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        if cls not in VEHICLE_CLASS_IDS:
            continue

        # Expand bounding box to dim a slightly larger region.
        x1f, y1f, x2f, y2f = map(float, [x1, y1, x2, y2])
        bw = x2f - x1f
        bh = y2f - y1f
        dx = bw * expand_ratio
        dy = bh * expand_ratio
        ex1 = max(0.0, x1f - dx)
        ey1 = max(0.0, y1f - dy)
        ex2 = min(float(w), x2f + dx)
        ey2 = min(float(h), y2f + dy)

        dim_factor = compute_distance_aware_dim_factor(
            (x1f, y1f, x2f, y2f),
            h,
            base_dim_factor=base_dim_factor,
            max_additional=DIM_SOFT_MAX_ADD,
        )

        if fast:
            add_rect_dim_fast(
                mask,
                road_mask,
                (int(ex1), int(ey1), int(ex2), int(ey2)),
                dim_factor,
            )
        else:
            add_gaussian_dim(mask, road_mask, (ex1, ey1, ex2, ey2), dim_factor)

    return mask


def apply_beam_mask(frame, mask):
    """
    Apply a brightness mask to a BGR frame.

    Args:
        frame: Input BGR image.
        mask: Float32 (H, W) mask in [0, 1].

    Returns:
        frame_out: BGR image with brightness scaled by mask.
    """
    frame_f = frame.astype(np.float32)
    frame_out = frame_f * mask[..., None]
    frame_out = np.clip(frame_out, 0, 255).astype(np.uint8)
    return frame_out
