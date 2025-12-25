"""
High-level ADB pipelines for video streams and single images.

These functions glue together:
- Road ROI estimation.
- YOLO inference.
- Detection memory and cooldown.
- Beam mask computation and EMA.
- LED integration and EMA.
- Top-view projection.
- OpenCV visualization windows.
"""

import time

import cv2
import numpy as np

from . import constants as C
from .road_roi import (
    compute_trapezoid_road_mask,
    detect_lane_based_road_mask,
)
from .dimming import compute_beam_mask, apply_beam_mask
from .angular_leds import (
    compute_angle_grids,
    build_led_angular_grid,
    integrate_leds_from_mask,
)
from .topview import project_beam_to_ground, render_top_view
from .visualization import (
    draw_detections,
    compute_road_shadow_metrics,
    render_led_matrix,
)


def _get_intrinsics_for_frame(h, w, args):
    """
    Compute or approximate fx, fy, cx, cy for this frame.

    If fx/fy/cx/cy are missing in args, we estimate fx,fy from FOV and set
    the principal point to the image center.
    """
    if (
        args.fx is not None
        and args.fy is not None
        and args.cx is not None
        and args.cy is not None
    ):
        return args.fx, args.fy, args.cx, args.cy

    # Approx from FOV.
    fx = (w / 2.0) / np.tan(np.deg2rad(args.fov) / 2.0)
    fy = fx
    cx = w / 2.0 if args.cx is None else args.cx
    cy = h / 2.0 if args.cy is None else args.cy
    return fx, fy, cx, cy


def run_video_adb(model, device, source, args):
    """
    Main ADB loop for video / webcam sources.

    Handles:
    - Frame capture and optional downscaling.
    - YOLO inference (with optional fast mode).
    - Road ROI estimation.
    - Detection memory & cooldown.
    - Beam mask computation and EMA smoothing.
    - LED integration and EMA.
    - Top-view projection.
    - Visualization in multiple windows.
    """
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        return

    print("[INFO] Press 'q' to quit.")
    class_names = model.names
    prev_time = time.time()
    road_mask = None
    frame_idx = 0
    lane_update_interval = 20 if args.fast else 10

    ema_mask = None
    ema_left_led = None
    ema_right_led = None
    ema_combined_led = None

    # Detection memory: bbox -> last_seen_time.
    object_memory = {}

    cv2.namedWindow("ADB - Camera Views", cv2.WINDOW_NORMAL)
    cv2.namedWindow("ADB - LED Matrices (L / C / R)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("ADB - Top View (Light Zone)", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream or failed to grab frame.")
            break

        frame_idx += 1

        # Time (for FPS estimation).
        now = time.time()
        dt = now - prev_time
        prev_time = now

        # Downscale for speed if configured.
        if C.PROCESS_SCALE != 1.0:
            h0, w0 = frame.shape[:2]
            new_w = int(w0 * C.PROCESS_SCALE)
            new_h = int(h0 * C.PROCESS_SCALE)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        h, w = frame.shape[:2]

        # In fast mode, skip frames to reduce load.
        if args.fast and (frame_idx % C.FAST_INFER_SKIP != 0):
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Prepare frame for inference.
        if args.fast:
            infer_frame = cv2.resize(
                frame,
                (C.FAST_INFER_SIZE, int(C.FAST_INFER_SIZE * h / w)),
                interpolation=cv2.INTER_AREA,
            )
            infer_size = C.FAST_INFER_SIZE
        else:
            infer_frame = frame
            infer_size = 320

        # Recompute road mask periodically or on size changes.
        if (
            road_mask is None
            or road_mask.shape[:2] != frame.shape[:2]
            or frame_idx % lane_update_interval == 0
        ):
            if args.fast and C.FAST_ROI_USE_TRAPEZOID:
                road_mask = compute_trapezoid_road_mask(frame.shape)
            else:
                rm = detect_lane_based_road_mask(frame)
                road_mask = (
                    rm if rm is not None else compute_trapezoid_road_mask(frame.shape)
                )

        # YOLO inference.
        results = model(infer_frame, size=infer_size)

        if args.fast:
            dets = results.xyxy[0].cpu().numpy() if len(results.xyxy) > 0 else None
            if dets is not None and dets.size > 0:
                scale_x = w / float(infer_frame.shape[1])
                scale_y = h / float(infer_frame.shape[0])
                dets[:, 0] *= scale_x
                dets[:, 2] *= scale_x
                dets[:, 1] *= scale_y
                dets[:, 3] *= scale_y
                detections = dets
            else:
                detections = None
        else:
            detections = (
                results.xyxy[0].cpu().numpy() if len(results.xyxy) > 0 else None
            )

        # ---------- UPDATE OBJECT MEMORY ----------
        now_time = time.time()
        new_memory = {}

        if detections is not None:
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if int(cls) not in C.VEHICLE_CLASS_IDS:
                    continue
                key = (int(x1), int(y1), int(x2), int(y2))
                new_memory[key] = now_time

        # Keep old boxes for a cooldown interval after they disappear.
        for key, last_seen in object_memory.items():
            if now_time - last_seen < C.COOLDOWN_SEC:
                # If a similar key is already in new_memory this keeps newest timestamp.
                new_memory.setdefault(key, last_seen)

        object_memory = new_memory

        # Build detections array from memory to drive the beam mask.
        if len(object_memory) > 0:
            detections_mem = []
            for (x1, y1, x2, y2), last_seen in object_memory.items():
                # conf=1.0, cls=0 (within VEHICLE_CLASS_IDS), we only need bbox for dimming.
                detections_mem.append([x1, y1, x2, y2, 1.0, 0])
            detections_mem = np.array(detections_mem, dtype=np.float32)
        else:
            detections_mem = None
        # -----------------------------------------

        # Use memory-based detections for beam mask (keeps dimmed for a while).
        global_mask = compute_beam_mask(
            frame.shape,
            detections_mem,
            road_mask,
            base_dim_factor=C.DIM_SOFT_MIN,
            expand_ratio=(C.FAST_DIM_EXPAND if args.fast else 0.25),
            fast=args.fast,
        )

        # EMA smoothing of mask to reduce flicker.
        if ema_mask is None:
            ema_mask = global_mask.copy()
        else:
            ema_mask = C.MASK_EMA_ALPHA * global_mask + (1.0 - C.MASK_EMA_ALPHA) * ema_mask

        # Angular grids for LED integration.
        fx, fy, cx, cy = _get_intrinsics_for_frame(h, w, args)
        _, _, theta_x, phi_y = compute_angle_grids(
            h,
            w,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            fov_deg=args.fov,
        )

        theta_span = np.deg2rad(40.0)
        phi_span = np.deg2rad(18.0)
        theta_center_offset = np.deg2rad(12.0)

        left_theta_centers, left_phi_centers, left_sigma_theta, left_sigma_phi = build_led_angular_grid(
            "left",
            rows=C.LED_ROWS,
            cols=C.LED_COLS_PER_LAMP,
            theta_span=theta_span,
            phi_span=phi_span,
            theta_center_offset=theta_center_offset,
        )
        right_theta_centers, right_phi_centers, right_sigma_theta, right_sigma_phi = build_led_angular_grid(
            "right",
            rows=C.LED_ROWS,
            cols=C.LED_COLS_PER_LAMP,
            theta_span=theta_span,
            phi_span=phi_span,
            theta_center_offset=theta_center_offset,
        )

        left_led_vals = integrate_leds_from_mask(
            ema_mask,
            road_mask,
            theta_x,
            phi_y,
            left_theta_centers,
            left_phi_centers,
            left_sigma_theta,
            left_sigma_phi,
        )
        right_led_vals = integrate_leds_from_mask(
            ema_mask,
            road_mask,
            theta_x,
            phi_y,
            right_theta_centers,
            right_phi_centers,
            right_sigma_theta,
            right_sigma_phi,
        )

        combined_led_vals = np.maximum(left_led_vals, right_led_vals)

        # EMA smoothing for LED matrices.
        if ema_left_led is None:
            ema_left_led = left_led_vals.copy()
            ema_right_led = right_led_vals.copy()
            ema_combined_led = combined_led_vals.copy()
        else:
            ema_left_led = C.LED_EMA_ALPHA * left_led_vals + (1.0 - C.LED_EMA_ALPHA) * ema_left_led
            ema_right_led = C.LED_EMA_ALPHA * right_led_vals + (1.0 - C.LED_EMA_ALPHA) * ema_right_led
            ema_combined_led = (
                C.LED_EMA_ALPHA * combined_led_vals
                + (1.0 - C.LED_EMA_ALPHA) * ema_combined_led
            )

        # ---- Top view projection of beam ----
        ground_img, extent = project_beam_to_ground(
            ema_mask,
            fx,
            fy,
            cx,
            cy,
            H_cam=C.TOPVIEW_CAM_HEIGHT_M,
            pitch_deg=C.TOPVIEW_CAM_PITCH_DEG,
            x_range=C.TOPVIEW_X_RANGE,
            z_range=C.TOPVIEW_Z_RANGE,
            grid_res=C.TOPVIEW_RES_M,
        )
        top_view_img = render_top_view(ground_img, extent)

        # ---- Visuals: camera + LEDs ----
        frame_det = draw_detections(
            frame, detections, class_names
        )  # uses current detections only
        frame_det = compute_trapezoid_road_mask(frame.shape)  # placeholder, but keep unchanged?
        frame_det = draw_detections(frame, detections, class_names)
        frame_det = frame_det  # no logic change, existing logic in original remains

        frame_det = draw_detections(frame, detections, class_names)
        # overlay light zone on detection view
        from .road_roi import draw_light_zone_overlay
        frame_det = draw_light_zone_overlay(frame_det, ema_mask)

        frame_adb = apply_beam_mask(frame, ema_mask)

        left_led_img = render_led_matrix(ema_left_led, cell_size=24, label_text="L")
        # combined_led_img = render_led_matrix(
        #     ema_combined_led, cell_size=24, label_text="C"
        # )
        right_led_img = render_led_matrix(ema_right_led, cell_size=24, label_text="R")

        # Compose LED view (L | C | R).
        gap_width = 8
        gap = np.zeros((left_led_img.shape[0], gap_width, 3), dtype=np.uint8)
        gap[:, :] = (20, 20, 20)
        led_img = np.hstack((left_led_img, gap, right_led_img))

        # Metrics for dimming coverage on road.
        road_pix, shadow_pix, shadow_ratio = compute_road_shadow_metrics(
            road_mask, ema_mask
        )

        # Combine camera views: [detections | dimmed output].
        cam_combined = np.hstack((frame_det, frame_adb))

        fps = 1.0 / (dt + 1e-6)

        ch, cw = cam_combined.shape[:2]
        cv2.putText(
            cam_combined,
            f"FPS: {fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            cam_combined,
            f"Road px: {road_pix}  Shadow: {shadow_pix} ({shadow_ratio*100:.1f}%)",
            (10, ch - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            cam_combined,
            "Left: L   Center: overlap   Right: R",
            (10, ch - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Optional display scaling.
        cam_disp = cam_combined
        led_disp = led_img
        top_disp = top_view_img
        if C.DISPLAY_SCALE != 1.0:
            cam_disp = cv2.resize(
                cam_combined,
                (int(cw * C.DISPLAY_SCALE), int(ch * C.DISPLAY_SCALE)),
                interpolation=cv2.INTER_AREA,
            )
            lh, lw = led_img.shape[:2]
            led_disp = cv2.resize(
                led_img,
                (int(lw * C.DISPLAY_SCALE), int(lh * C.DISPLAY_SCALE)),
                interpolation=cv2.INTER_AREA,
            )
            th, tw = top_view_img.shape[:2]
            top_disp = cv2.resize(
                top_view_img,
                (int(tw * C.DISPLAY_SCALE), int(th * C.DISPLAY_SCALE)),
                interpolation=cv2.INTER_AREA,
            )

        # Show all views.
        cv2.imshow("ADB - Camera Views", cam_disp)
        cv2.imshow("ADB - LED Matrices (L / C / R)", led_disp)
        cv2.imshow("ADB - Top View (Light Zone)", top_disp)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_image_adb(model, device, source, args):
    """
    ADB pipeline for a single image input instead of a video stream.

    This runs detection, beam mask, LED integration and top-view projection
    once, then shows the corresponding visualization windows.
    """
    img = cv2.imread(source)
    if img is None:
        print(f"[ERROR] Could not read image: {source}")
        return

    h, w = img.shape[:2]

    # Road ROI.
    if args.fast and C.FAST_ROI_USE_TRAPEZOID:
        road_mask = compute_trapezoid_road_mask(img.shape)
    else:
        rm = detect_lane_based_road_mask(img)
        road_mask = rm if rm is not None else compute_trapezoid_road_mask(img.shape)

    # YOLO inference.
    infer_size = C.FAST_INFER_SIZE if args.fast else 320
    results = model(img, size=infer_size)
    detections = results.xyxy[0].cpu().numpy() if len(results.xyxy) > 0 else None
    class_names = model.names

    # Beam mask from detections.
    global_mask = compute_beam_mask(
        img.shape,
        detections,
        road_mask,
        base_dim_factor=C.DIM_SOFT_MIN,
        expand_ratio=(C.FAST_DIM_EXPAND if args.fast else 0.25),
        fast=args.fast,
    )

    # Single-frame "EMA" (just copy).
    ema_mask = global_mask.copy()

    # Angular grids for LED integration.
    fx, fy, cx, cy = _get_intrinsics_for_frame(h, w, args)
    _, _, theta_x, phi_y = compute_angle_grids(
        h,
        w,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        fov_deg=args.fov,
    )

    theta_span = np.deg2rad(40.0)
    phi_span = np.deg2rad(18.0)
    theta_center_offset = np.deg2rad(12.0)

    left_theta_centers, left_phi_centers, left_sigma_theta, left_sigma_phi = build_led_angular_grid(
        "left",
        rows=C.LED_ROWS,
        cols=C.LED_COLS_PER_LAMP,
        theta_span=theta_span,
        phi_span=phi_span,
        theta_center_offset=theta_center_offset,
    )
    right_theta_centers, right_phi_centers, right_sigma_theta, right_sigma_phi = build_led_angular_grid(
        "right",
        rows=C.LED_ROWS,
        cols=C.LED_COLS_PER_LAMP,
        theta_span=theta_span,
        phi_span=phi_span,
        theta_center_offset=theta_center_offset,
    )

    left_led_vals = integrate_leds_from_mask(
        ema_mask,
        road_mask,
        theta_x,
        phi_y,
        left_theta_centers,
        left_phi_centers,
        left_sigma_theta,
        left_sigma_phi,
    )
    right_led_vals = integrate_leds_from_mask(
        ema_mask,
        road_mask,
        theta_x,
        phi_y,
        right_theta_centers,
        right_phi_centers,
        right_sigma_theta,
        right_sigma_phi,
    )
    combined_led_vals = np.maximum(left_led_vals, right_led_vals)

    # Top-view projection for image.
    ground_img, extent = project_beam_to_ground(
        ema_mask,
        fx,
        fy,
        cx,
        cy,
        H_cam=C.TOPVIEW_CAM_HEIGHT_M,
        pitch_deg=C.TOPVIEW_CAM_PITCH_DEG,
        x_range=C.TOPVIEW_X_RANGE,
        z_range=C.TOPVIEW_Z_RANGE,
        grid_res=C.TOPVIEW_RES_M,
    )
    top_view_img = render_top_view(ground_img, extent)

    # Visualizations.
    from .road_roi import draw_light_zone_overlay

    frame_det = draw_detections(img, detections, class_names)
    frame_det = draw_light_zone_overlay(frame_det, ema_mask)
    frame_adb = apply_beam_mask(img, ema_mask)

    left_led_img = render_led_matrix(left_led_vals, cell_size=24, label_text="L")
    combined_led_img = render_led_matrix(
        combined_led_vals, cell_size=24, label_text="C"
    )
    right_led_img = render_led_matrix(right_led_vals, cell_size=24, label_text="R")

    gap_width = 8
    gap = np.zeros((left_led_img.shape[0], gap_width, 3), dtype=np.uint8)
    gap[:, :] = (20, 20, 20)
    led_img = np.hstack((left_led_img, gap, combined_led_img, gap, right_led_img))

    cam_combined = np.hstack((frame_det, frame_adb))
    ch, cw = cam_combined.shape[:2]
    road_pix, shadow_pix, shadow_ratio = compute_road_shadow_metrics(
        road_mask, ema_mask
    )

    cv2.putText(
        cam_combined,
        f"Road px: {road_pix}  Shadow: {shadow_pix} ({shadow_ratio*100:.1f}%)",
        (10, ch - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        cam_combined,
        "Left: L   Center: overlap   Right: R",
        (10, ch - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cam_disp = cam_combined
    led_disp = led_img
    top_disp = top_view_img
    if C.DISPLAY_SCALE != 1.0:
        cam_disp = cv2.resize(
            cam_combined,
            (int(cw * C.DISPLAY_SCALE), int(ch * C.DISPLAY_SCALE)),
            interpolation=cv2.INTER_AREA,
        )
        lh, lw = led_img.shape[:2]
        led_disp = cv2.resize(
            led_img,
            (int(lw * C.DISPLAY_SCALE), int(lh * C.DISPLAY_SCALE)),
            interpolation=cv2.INTER_AREA,
        )
        th, tw = top_view_img.shape[:2]
        top_disp = cv2.resize(
            top_view_img,
            (int(tw * C.DISPLAY_SCALE), int(th * C.DISPLAY_SCALE)),
            interpolation=cv2.INTER_AREA,
        )

    cv2.namedWindow("ADB - Camera Views (Image)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("ADB - LED Matrices (L / C / R, Image)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("ADB - Top View (Light Zone, Image)", cv2.WINDOW_NORMAL)
    cv2.imshow("ADB - Camera Views (Image)", cam_disp)
    cv2.imshow("ADB - LED Matrices (L / C / R, Image)", led_disp)
    cv2.imshow("ADB - Top View (Light Zone, Image)", top_disp)
    print("[INFO] Press any key in the image windows to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
