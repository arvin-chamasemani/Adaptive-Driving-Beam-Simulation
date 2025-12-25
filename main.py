"""
CLI entry point for the ADB demo project.

This file only handles:
- Parsing command-line arguments.
- Creating and running the ADBApplication.

programmer: Arvin Chamasemani
"""

import argparse
from src.adb_app import ADBApplication


def parse_args():
    """
    Parse command-line arguments for the ADB demo.
    """
    parser = argparse.ArgumentParser(
        description=(
            "ADB demo with angular LED modeling, "
            "top-view projection, temporal smoothing, and detection memory"
        )
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="../runs/train/exp12/weights/best.pt",
        help="path to YOLO weights (.pt)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="video/image source (0 for webcam, or path to video/image)",
    )
    parser.add_argument(
        "--image",
        action="store_true",
        help="treat source as a single image",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="'cpu' or 'cuda' (empty = auto)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default="1",
        help="enable faster but less accurate mode",
    )
    parser.add_argument(
        "--fx",
        type=float,
        default=None,
        help="camera focal length fx (pixels)",
    )
    parser.add_argument(
        "--fy",
        type=float,
        default=None,
        help="camera focal length fy (pixels)",
    )
    parser.add_argument(
        "--cx",
        type=float,
        default=None,
        help="camera principal point cx (pixels)",
    )
    parser.add_argument(
        "--cy",
        type=float,
        default=None,
        help="camera principal point cy (pixels)",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=90.0,
        help="approx horizontal FOV (deg) used if fx not provided",
    )
    return parser.parse_args()


def main():
    """
    Create the ADBApplication and run the selected mode.
    """
    args = parse_args()
    app = ADBApplication(args)
    app.run()


if __name__ == "__main__":
    main()
