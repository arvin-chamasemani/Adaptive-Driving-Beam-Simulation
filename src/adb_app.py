"""
High-level ADBApplication class that owns model and arguments,
and chooses between image and video pipelines.
"""

from pathlib import Path
import argparse

from .model_utils import load_model
from .pipeline import run_video_adb, run_image_adb


class ADBApplication:
    """
    High-level OOP façade encapsulating:
      - Model loading.
      - Device configuration.
      - Selection between video and image ADB modes.

    The underlying computation is still done by the pure functional
    helpers in src.pipeline, which keeps all algorithmic logic unchanged.
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize the ADB application with command-line arguments.

        Args:
            args: Parsed argparse Namespace with all CLI arguments.
        """
        self.args = args
        self.model = None
        self.device = None

    def initialize_model(self) -> bool:
        """
        Load the YOLO model and configure it for the chosen device.

        This method wraps load_model() and optional FP16 setup for CUDA.

        Returns:
            True if the model was successfully initialized, False otherwise.
        """
        weights_path = self.args.weights
        if not Path(weights_path).exists():
            print(f"[WARNING] Weights file not found: {weights_path}")
            print("         Adjust --weights or train the model first.")
            return False

        self.model, self.device = load_model(weights_path, device=self.args.device)

        # Optionally switch to FP16 on CUDA for faster inference.
        if self.device == "cuda":
            try:
                self.model.half()
                print("[INFO] Model set to FP16 (half) for faster CUDA inference.")
            except Exception:
                # Some models may not support .half(); fail silently.
                pass

        return True

    def run(self):
        """
        Entry point to execute the ADB pipeline.

        If --image is set, runs the single-image pipeline.
        Otherwise, runs the video/webcam pipeline.
        """
        if not self.initialize_model():
            return

        if self.args.image:
            run_image_adb(self.model, self.device, self.args.source, self.args)
        else:
            run_video_adb(self.model, self.device, self.args.source, self.args)
