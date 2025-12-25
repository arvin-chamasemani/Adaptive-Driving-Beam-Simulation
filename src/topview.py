"""
Correct bird’s-eye projection of the beam mask onto ground.
"""

import numpy as np
import cv2

from .constants import (
    TOPVIEW_CAM_HEIGHT_M,
    TOPVIEW_CAM_PITCH_DEG,
    TOPVIEW_X_RANGE,
    TOPVIEW_Z_RANGE,
    TOPVIEW_RES_M,
)


def project_beam_to_ground(
    beam_mask,
    fx,
    fy,
    cx,
    cy,
    H_cam=TOPVIEW_CAM_HEIGHT_M,
    pitch_deg=TOPVIEW_CAM_PITCH_DEG,
    x_range=TOPVIEW_X_RANGE,
    z_range=TOPVIEW_Z_RANGE,
    grid_res=TOPVIEW_RES_M,
):
    """
    Project beam_mask onto the ground plane Y = +H_cam.
    Coordinate system:
      X right, Y down, Z forward.
      Camera pitched downward by pitch_deg.
    """
    h, w = beam_mask.shape[:2]

    # 1) Pixel → ray in camera frame
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    x_cam = (us - cx) / fx
    y_cam = (vs - cy) / fy
    z_cam = np.ones_like(x_cam)

    dirs_cam = np.stack((x_cam, y_cam, z_cam), axis=0).reshape(3, -1)

    # 2) Apply downward pitch
    pitch = np.deg2rad(pitch_deg)
    cp = np.cos(pitch)
    sp = np.sin(pitch)

    R = np.array([
        [1,   0,   0],
        [0,   cp, -sp],
        [0,   sp,  cp],
    ], dtype=np.float32)

    dirs = R @ dirs_cam
    dx, dy, dz = dirs

    # 3) Intersection with ground plane Y = +H_cam
    # In this coordinate system, DOWN is POSITIVE → rays hitting ground have dy > 0
    valid = dy > 1e-6

    t = np.zeros_like(dy)
    t[valid] = H_cam / dy[valid]

    X = dx * t
    Z = dz * t

    # Only keep forward hits
    valid &= (Z > 0.0)

    x_min, x_max = x_range
    z_min, z_max = z_range

    Wg = int(np.ceil((x_max - x_min) / grid_res))
    Hg = int(np.ceil((z_max - z_min) / grid_res))

    ground = np.zeros((Hg, Wg), dtype=np.float32)

    cols = ((X - x_min) / grid_res).astype(np.int32)
    rows = ((Z - z_min) / grid_res).astype(np.int32)

    inside = (
        valid &
        (cols >= 0) & (cols < Wg) &
        (rows >= 0) & (rows < Hg)
    )

    if not np.any(inside):
        return ground, (x_min, x_max, z_min, z_max)

    vals = beam_mask.reshape(-1)[inside]
    np.maximum.at(ground, (rows[inside], cols[inside]), vals)

    return ground, (x_min, x_max, z_min, z_max)



def render_top_view(ground, extent, car_length=4.5, car_width=1.9):
    """
    Convert the ground grid into a displayable BGR top-view image.
    """

    x_min, x_max, z_min, z_max = extent
    Hg, Wg = ground.shape

    img = (ground * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Draw car footprint at origin
    x0, x1 = -car_width/2, car_width/2
    z0, z1 = 0.0, car_length

    col0 = int((x0 - x_min) / (x_max - x_min) * Wg)
    col1 = int((x1 - x_min) / (x_max - x_min) * Wg)
    row0 = int((z0 - z_min) / (z_max - z_min) * Hg)
    row1 = int((z1 - z_min) / (z_max - z_min) * Hg)

    col0 = np.clip(col0, 0, Wg - 1)
    col1 = np.clip(col1, 0, Wg - 1)
    row0 = np.clip(row0, 0, Hg - 1)
    row1 = np.clip(row1, 0, Hg - 1)

    cv2.rectangle(img, (col0, row0), (col1, row1), (0,255,0), 2)

    # Flip so near is bottom, far is top
    img = cv2.flip(img, 0)

    cv2.putText(
        img, "Top View (Light Zone)", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2
    )

    return img
