"""
Angular LED modeling and integration.

Includes:
- Mapping pixels to angular coordinates.
- LED angular grid definition.
- Vectorized LED integration using separable kernels.
"""

import numpy as np

from .constants import LED_ROWS, LED_COLS_PER_LAMP, EPS


def compute_angle_grids(h, w, fx=None, fy=None, cx=None, cy=None, fov_deg=90.0):
    """
    Compute per-pixel horizontal (theta) and vertical (phi) angles in radians.
    If fx/fy/cx/cy are None, estimate fx,fy from horizontal FOV (fov_deg).

    Args:
        h, w: Image height / width.
        fx, fy: Focal lengths in pixels (optional).
        cx, cy: Principal point in pixels (optional).
        fov_deg: Approx horizontal FOV in degrees if fx is not provided.

    Returns:
        theta: (h, w) horizontal angle for each pixel.
        phi: (h, w) vertical angle for each pixel.
        theta_x: (w,) 1D horizontal angles per column.
        phi_y: (h,) 1D vertical angles per row.
    """
    if cx is None:
        cx = w / 2.0
    if cy is None:
        cy = h / 2.0

    if fx is None:
        fx = (w / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    if fy is None:
        fy = fx  # assume square pixels

    xs = np.arange(w)
    ys = np.arange(h)
    theta_x = np.arctan2((xs - cx), fx)   # horizontal angle per column
    phi_y = np.arctan2((ys - cy), fy)     # vertical angle per row

    theta = np.tile(theta_x[None, :], (h, 1))
    phi = np.tile(phi_y[:, None], (1, w))
    return theta, phi, theta_x, phi_y


def build_led_angular_grid(
    side,
    rows=LED_ROWS,
    cols=LED_COLS_PER_LAMP,
    theta_span=np.deg2rad(40.0),
    phi_span=np.deg2rad(20.0),
    theta_center_offset=np.deg2rad(10.0),
):
    """
    Build per-LED center angles (theta_leds, phi_leds) for a lamp side.

    Args:
        side: "left" or "right" headlamp.
        rows, cols: LED matrix dimensions.
        theta_span, phi_span: Angular range covered by the LED matrix.
        theta_center_offset: Angular offset to bias left/right lamp.

    Returns:
        theta_centers: (rows, cols) centers in theta.
        phi_centers: (rows, cols) centers in phi.
        sigma_theta: Horizontal spread used in integration kernel.
        sigma_phi: Vertical spread used in integration kernel.
    """
    if side == "left":
        theta_start = -theta_center_offset - theta_span / 2.0
        theta_end = -theta_center_offset + theta_span / 2.0
    else:
        theta_start = theta_center_offset - theta_span / 2.0
        theta_end = theta_center_offset + theta_span / 2.0

    theta_centers_1d = np.linspace(theta_start, theta_end, cols)
    phi_centers_1d = np.linspace(-phi_span / 2.0, phi_span / 2.0, rows)

    theta_centers = np.tile(theta_centers_1d[None, :], (rows, 1))
    phi_centers = np.tile(phi_centers_1d[:, None], (1, cols))

    sigma_theta = (theta_end - theta_start) / max(1.0, cols) * 0.9
    sigma_phi = (
        (phi_centers_1d[1] - phi_centers_1d[0]) * 1.2 if rows > 1 else phi_span * 0.5
    )

    return theta_centers, phi_centers, sigma_theta, sigma_phi


def integrate_leds_from_mask(
    beam_mask,
    road_mask,
    theta_x,
    phi_y,
    theta_centers,
    phi_centers,
    sigma_theta,
    sigma_phi,
):
    """
    Vectorized per-LED integration using separable angular kernels.

    Args:
        beam_mask: (H, W) float32 beam mask in [0, 1].
        road_mask: (H, W) float32 road mask in [0, 1].
        theta_x: (W,) 1D horizontal angles.
        phi_y: (H,) 1D vertical angles.
        theta_centers, phi_centers: (rows, cols) LED center angles.
        sigma_theta, sigma_phi: Spreads for angular kernels.

    Returns:
        led_vals: (rows, cols) per-LED brightness values in [0, 1].
    """
    h, w = beam_mask.shape
    rows, cols = theta_centers.shape

    # For separable kernels we only need 1D center vectors.
    theta_centers_cols = theta_centers[0, :]  # (cols,)
    phi_centers_rows = phi_centers[:, 0]      # (rows,)

    # Horizontal kernel H: (w, cols)
    theta_x_col = theta_x[:, None]           # (w,1)
    theta_diff = theta_x_col - theta_centers_cols[None, :]  # (w, cols)
    H = np.exp(-0.5 * (theta_diff ** 2) / (sigma_theta ** 2 + EPS))

    # Vertical kernel V: (h, rows)
    phi_y_row = phi_y[:, None]               # (h,1)
    phi_diff = phi_y_row - phi_centers_rows[None, :]        # (h, rows)
    V = np.exp(-0.5 * (phi_diff ** 2) / (sigma_phi ** 2 + EPS))

    # Normalize kernels along their integration axes.
    H = H / (H.sum(axis=0, keepdims=True) + EPS)
    V = V / (V.sum(axis=0, keepdims=True) + EPS)

    # Restrict integration to road region.
    B = beam_mask * road_mask  # (h,w)

    # First integrate vertically, then horizontally.
    VT_B = V.T @ B             # (rows, w)
    A = VT_B @ H               # (rows, cols)

    # Integrate road mask for normalization (denominator).
    VT_R = V.T @ road_mask     # (rows, w)
    D = VT_R @ H               # (rows, cols)

    led_vals = A / (D + EPS)
    led_vals = np.clip(led_vals, 0.0, 1.0)

    return led_vals
