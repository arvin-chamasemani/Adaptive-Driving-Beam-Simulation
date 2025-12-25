"""
Global constants and configuration parameters for the ADB demo.
"""

# YOLO class ids to consider as "vehicles" for dimming.
VEHICLE_CLASS_IDS = {0, 1, 2, 3, 4}  # car / motorcycle / bus / truck / bicycle

# Optional downscale factor for processing resolution (1.0 = native resolution).
PROCESS_SCALE = 0.5   # 0.5 = process at half res in each dimension (25% pixels)

# LED matrix configuration (per headlamp).
LED_ROWS = 8
LED_COLS_PER_LAMP = 8

# Display scaling for visualization windows.
DISPLAY_SCALE = 0.48

# Road region-of-interest (ROI) trapezoid parameters (normalized).
ROAD_TOP_Y_RATIO = 0.55
ROAD_TOP_WIDTH_RATIO = 0.4
lower_part_ratio = 1.5

# Dimming & Gaussian hotspot parameters.
HOTSPOT_HEIGHT_RATIO = 0.35
DIM_SOFT_MIN = 0.18
DIM_SOFT_MAX_ADD = 0.55

HEIGHT_NORM_DENOM_RATIO = 0.5
HEIGHT_NORM_MIN = 0.05
HEIGHT_NORM_MAX = 1.0

# Threshold for drawing "light zone" overlay.
LIGHT_ZONE_MIN_BRIGHTNESS = 0.10

# Fast-mode tuning.
FAST_INFER_SIZE = 256
FAST_INFER_SKIP = 2
FAST_ROI_USE_TRAPEZOID = True
FAST_DIM_EXPAND = 0.15

# Temporal smoothing parameters (exponential moving average).
MASK_EMA_ALPHA = 0.45   # smoothing for beam mask
LED_EMA_ALPHA = 0.35    # smoothing for LED matrices

# Detection memory cooldown (seconds to keep last boxes after they disappear).
COOLDOWN_SEC = 2.0

# Top-view projection defaults (approximate, tune for dashcam geometry).
TOPVIEW_CAM_HEIGHT_M = 1   # camera height [m]
TOPVIEW_CAM_PITCH_DEG = 8.0  # camera pitch down [deg]
TOPVIEW_X_RANGE = (-15.0, 15.0)  # lateral [m]
TOPVIEW_Z_RANGE = (0.0, 90.0)    # forward [m]
TOPVIEW_RES_M = 0.25             # grid resolution [m]

# Small epsilon to avoid division by zero.
EPS = 1e-8
