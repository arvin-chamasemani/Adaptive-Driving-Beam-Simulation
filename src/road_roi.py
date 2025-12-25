"""
Road region-of-interest (ROI) utilities.

Includes:
- Trapezoidal road mask (geometric approximation).
- Lane-based road mask via edge + Hough transform.
- Light-zone overlay drawing.
"""

import cv2
import numpy as np

from .constants import (
    ROAD_TOP_Y_RATIO,
    ROAD_TOP_WIDTH_RATIO,
    lower_part_ratio,
    LIGHT_ZONE_MIN_BRIGHTNESS,
)


def compute_trapezoid_road_mask(frame_shape):
    """
    Build a coarse, trapezoidal road mask assuming a forward-facing dashcam.

    Args:
        frame_shape: Shape of the image/frame (H, W, C).

    Returns:
        mask: Float32 mask in [0, 1] with 1 inside the road trapezoid.
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    bottom_y = (h - 1)
    top_y = int(ROAD_TOP_Y_RATIO * h)

    bottom_width = lower_part_ratio * w
    top_width = int(ROAD_TOP_WIDTH_RATIO * w)
    cx = w // 2

    pts = np.array([
        [cx - top_width // 2, top_y],
        [cx + top_width // 2, top_y],
        [bottom_width + bottom_width / 2 - 1, bottom_y],
        [0 - bottom_width / 2, bottom_y]
    ], dtype=np.int32)

    cv2.fillConvexPoly(mask, pts, 1)
    return mask.astype(np.float32)


def detect_lane_based_road_mask(frame):
    """
    Attempt to compute a road mask based on lane marking detection.

    This uses Canny edge detection + Hough lines to infer left/right lane lines
    and builds a quadrilateral road ROI between them.

    Args:
        frame: BGR image.

    Returns:
        mask: Float32 mask in [0, 1] or None if lane-based detection fails.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Restrict edge detection to a lower-half ROI where lanes usually are.
    roi_mask = np.zeros_like(edges)
    y_top = int(0.55 * h)
    roi_vertices = np.array([
        [0, h],
        [w, h],
        [w, y_top],
        [0, y_top]
    ], dtype=np.int32)
    cv2.fillConvexPoly(roi_mask, roi_vertices, 255)
    edges_roi = cv2.bitwise_and(edges, roi_mask)

    lines = cv2.HoughLinesP(
        edges_roi,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=int(0.1 * w),
        maxLineGap=40,
    )

    if lines is None:
        return None

    left_points = []
    right_points = []

    # Classify lines into left/right based on slope sign.
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        if abs(slope) < 0.3:
            continue
        if slope < 0:
            left_points.extend([(x1, y1), (x2, y2)])
        else:
            right_points.extend([(x1, y1), (x2, y2)])

    if len(left_points) < 4 or len(right_points) < 4:
        return None

    left_points = np.array(left_points)
    right_points = np.array(right_points)

    def fit_line(points):
        """Fit y = m x + b to given points using least squares."""
        xs = points[:, 0]
        ys = points[:, 1]
        A = np.vstack([xs, np.ones_like(xs)]).T
        m, b = np.linalg.lstsq(A, ys, rcond=None)[0]
        return m, b

    m_left, b_left = fit_line(left_points)
    m_right, b_right = fit_line(right_points)

    y_bottom = h - 1
    y_top_lane = int(ROAD_TOP_Y_RATIO * h)

    def x_at_y(m, b, y):
        """Compute x coordinate at given y for line y = m x + b."""
        if abs(m) < 1e-6:
            return None
        return int((y - b) / m)

    x_left_bottom = x_at_y(m_left, b_left, y_bottom)
    x_left_top = x_at_y(m_left, b_left, y_top_lane)
    x_right_bottom = x_at_y(m_right, b_right, y_bottom)
    x_right_top = x_at_y(m_right, b_right, y_top_lane)

    xs = [x_left_bottom, x_left_top, x_right_top, x_right_bottom]
    if any(x is None for x in xs):
        return None
    if any(x < -w or x > 2 * w for x in xs):
        return None

    # Clamp polygon into image bounds.
    x_left_bottom = max(0, min(w - 1, x_left_bottom))
    x_left_top = max(0, min(w - 1, x_left_top))
    x_right_bottom = max(0, min(w - 1, x_right_bottom))
    x_right_top = max(0, min(w - 1, x_right_top))

    pts = np.array([
        [x_left_top, y_top_lane],
        [x_right_top, y_top_lane],
        [x_right_bottom, y_bottom],
        [x_left_bottom, y_bottom],
    ], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 1)
    return mask.astype(np.float32)


def compute_road_roi_mask(frame, use_lane=True):
    """
    Compute the road ROI mask for a frame.

    It first tries lane-based detection (if use_lane=True) and falls back to
    a trapezoidal approximation on failure.

    Args:
        frame: Input BGR frame.
        use_lane: If True, attempt lane-based detection first.

    Returns:
        road_mask: Float32 mask in [0, 1].
    """
    if use_lane:
        lane_mask = detect_lane_based_road_mask(frame)
        if lane_mask is not None:
            return lane_mask
    return compute_trapezoid_road_mask(frame.shape)


def draw_light_zone_overlay(frame, beam_mask, color=(255, 255, 0)):
    """
    Draw a semi-transparent overlay where the beam mask is above a brightness
    threshold, visualizing the illuminated "light zone" on the road.

    Args:
        frame: Input BGR frame.
        beam_mask: Float32 mask in [0, 1].
        color: BGR color of the overlay.

    Returns:
        overlayed_frame: BGR frame with overlay.
    """
    overlay = frame.copy()
    alpha = 0.35
    light_region = beam_mask > LIGHT_ZONE_MIN_BRIGHTNESS
    overlay[light_region] = (
        overlay[light_region] * (1 - alpha)
        + np.array(color, dtype=np.float32) * alpha
    )
    return overlay.astype(np.uint8)
