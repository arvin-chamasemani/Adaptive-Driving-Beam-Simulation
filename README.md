# Adaptive Driving Beam simulation

This repository contains a **prototype Adaptive Driving Beam (ADB)** pipeline built on top of **YOLO-based vehicle detection** and an **angular LED model**.  
It simulates how a matrix LED headlamp could dim individual LEDs to avoid glaring other road users, while still keeping the road as bright as possible.



## 1. What this demo is about

This demo takes a **camera stream or image** (e.g., a dashcam), detects vehicles in front, and then:

- Builds a **“beam mask”** describing how much to dim the light in each pixel.
- Maps that mask to a grid of **virtual LEDs**, each one representing a narrow angular sector of the headlamp.
- Applies **temporal smoothing** so the light output doesn’t flicker frame to frame.
- Projects the light onto the **road plane in top view (bird’s-eye)** to visualize where the light actually lands.
- Keeps dimming briefly **even after a vehicle disappears** (detection memory / cooldown) to avoid quick light “pops”.

Visually, you get three main views:

1. **Camera views**: original frame with detections + dimmed frame.
2. **LED matrices**: grayscale grids for left, right, and combined headlamp LEDs.
3. **Top view**: bird’s-eye projection of the light distribution on the road.


## 2. Key concepts & design

### 2.1 YOLO-based detection
- Uses a YOLO object detection model based on an open-source  [GitHub](https://github.com/MaryamBoneh/Vehicle-Detection.git) implementation.
- Only a subset of classes are treated as “vehicles” (car, motorcycle, bus, truck, bicycle).
- Detections are converted into **bounding boxes** which drive the dimming logic.

### 2.2 Road region of interest (ROI)
To avoid dimming useless parts of the image, the code builds a **road mask**:

- **Lane-based mask**: uses Canny edges + Hough lines to detect lane markings and infer the drivable area.
- **Fallback trapezoid**: if lane detection fails, it uses a geometric trapezoid in front of the car as a rough road estimate.

All dimming is confined to this road ROI.

### 2.3 Beam mask and dimming
For each detected vehicle:

1. A **dimming factor** is computed based (roughly) on the **height of its box** in the image:
   - Taller boxes → closer vehicles → stronger dimming.
   - Smaller boxes → farther vehicles → weaker dimming.
2. A **local dimming region** is placed around the vehicle’s upper area (hotspot):
   - **Gaussian mode (default)**: soft, 2D Gaussian shape for smooth dimming.
   - **Fast rectangular mode**: simple rectangular region for speed.

Multiple vehicles are combined into a global **beam mask** (float image in `[0, 1]`), where `1` = full brightness and lower values = dimmed.

### 2.4 Temporal smoothing (EMA)
To reduce flicker:

- The beam mask is filtered with an **Exponential Moving Average (EMA)** over time.
- The same idea is applied to **LED matrices**, so LED values evolve smoothly.

This avoids sudden jumps when detections come and go.

### 2.5 Detection memory (cooldown)
Real detectors are noisy and may lose an object for a frame or two. To avoid instant re-brightening:

- The script keeps a **memory** of recent bounding boxes.
- Each box is kept for a few seconds (`COOLDOWN_SEC`) after it disappears.
- During that cooldown, the dimming region is still applied, even if YOLO doesn’t see the object anymore.

### 2.6 Angular LED model
Instead of treating the headlamp as a single blob of light, it’s modeled as **multiple LEDs**, each one covering a small angular sector:

- The image is mapped into **angles**:
  - Horizontal angle θ (left/right).
  - Vertical angle φ (up/down).
- Each LED has:
  - A center direction (θ, φ).
  - A small angular spread (like its “beam width”).

The script uses **separable angular kernels** (one in θ, one in φ) to integrate how much brightness from the beam mask falls into each LED’s sector.  
Result: a **per-LED brightness matrix** for left and right lamps.

### 2.7 Top-view (bird’s-eye) projection
To visualize where light actually hits the road:

1. For each pixel, a 3D **ray** is formed using camera intrinsics (`fx, fy, cx, cy`) and pitch.
2. Each ray is intersected with a **flat ground plane** at a fixed camera height.
3. The intersection points are discretized onto a **2D grid (X lateral, Z forward)**.
4. The light intensity is aggregated into this grid.

Finally, a top-view image is rendered, with an outline of the car footprint, showing the light footprint in front of the vehicle.


## 3. How to run

### 3.0 cloning

use following command
```bash
git clone https://github.com/MaryamBoneh/Vehicle-Detection.git
git clone <this repository>
# Then place this repo inside Vehicle-Detection-main, or adjust paths accordingly.
```

Move the video available at test_video folder to `Vehicle-Detection-main/test_images`

### 3.1 Requirements

Install required dependencies using following code:

```bash
pip install -r requirements.txt
```
Make sure you have a YOLO weights file (default path below) or adjust the --weights argument.

### 3.2 Basic commands
Webcam (default camera, fast mode)
```bash
python adb_demo_topview.py --source 0 --fast
```

- ``` --source 0 ```: use default webcam (or dashcam capture).

- ``` --fast ```: enables a faster pipeline (smaller inference size, rectangular dimming).

**Single image**
```bash
python adb_demo_topview.py \
    --source test_images/imtest13.JPG \
    --image
```

- --image: tells the script that source is a single image file.

### 3.3 Command line arguments

- ``` --weights ``` PATH
  - Path to YOLO ``` .pt ``` weights.
  - Default:``` runs/train/exp1/weights/best.pt ```

- ```---source ``` SRC
  - Video / image source.

    - ``` "0" ``` for webcam.

    - Path to a video or image file.

- ``` --image ```
  - Treat --source as a single image (no video loop).

- ``` --device ``` {cpu|cuda}
  - Force running on CPU or CUDA. Empty = auto.

- ``` --fast ```
  - Fast mode: lower resolution inference,     rectangular dimming, and some tuned parameters for speed.

- ``` --fx --fy --cx --cy ```
  - Optional camera intrinsics (pixels). If not provided:

    - ``` fx, fy ``` are estimated from ```--fov```.

    - ``` cx, cy ``` default to image center.

- ``` --fov ```
  - Approximate horizontal FOV in degrees (used only if fx is not provided).
  - Default: ``` 90.0 ``` degrees.


## 4. Windows & outputs

- During video mode, three OpenCV windows are opened:

1. ADB - Camera Views

- Left: original camera frame with YOLO detections and light zone overlay.

- Right: dimmed frame (beam applied).

- Overlays:

  - FPS

  - Road pixel stats (how much of the road is dimmed and to what ratio)

  - Legend for left/center/right LED regions.

2. ADB - LED Matrices (L / C / R)

- Grayscale grid images for:

  - Left headlamp LED matrix.

  - Combined overlapping center region.

  - Right headlamp LED matrix.

3. ADB - Top View (Light Zone)

- Bird’s-eye view of the light on the road.

- Rectangle representing vehicle footprint.

- Brighter regions = higher light intensity.

Press q in the video windows to exit.
For image mode, press any key to close the windows.

## 5. Notes & limitations

- Parameters like camera height, pitch, FOV, and LED spread are approximate and may need tuning for your specific camera and headlamp.

- The ground is assumed to be a flat plane, which is a simplification.

- YOLO detections and road/lane detection are not perfect; misdetections will influence the dimming pattern.

- The demo focuses on conceptual ADB behavior and visualization rather than production-grade performance.

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for full details.

### Third-Party Notice
This project uses the YOLO model based on the following repository:

- Vehicle-Detection by MaryamBoneh: https://github.com/MaryamBoneh/Vehicle-Detection.git

Please review third-party licensing terms to ensure compliance.
