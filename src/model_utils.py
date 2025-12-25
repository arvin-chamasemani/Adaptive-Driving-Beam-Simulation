"""
Model loading utilities for YOLO.
"""

from pathlib import Path
import torch


def _find_hub_repo_root() -> Path:
    """
    Walk upwards from this file looking for 'hubconf.py'.

    This allows the project to live inside subfolders (like
    'adaptive driving beam/') while still pointing torch.hub
    to the original YOLO repo root.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        hubconf = parent / "hubconf.py"
        if hubconf.exists():
            return parent

    # Fallback: current working directory (may still fail if no hubconf.py).
    return Path(".")


def load_model(weights_path: str, device: str = ""):
    """
    Load a YOLO model from a local weights file using torch.hub.

    Args:
        weights_path: Path to YOLO .pt weights.
        device: Device string ("cpu", "cuda", or empty for auto).

    Returns:
        model: Loaded YOLO model in eval mode.
        device: Final device string used ("cpu" or "cuda").
    """
    if device == "":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # Find the directory that actually contains hubconf.py (YOLO repo root).
    repo_dir = _find_hub_repo_root()
    print(f"[INFO] Loading model from repo: {repo_dir}")
    print(f"[INFO] Using weights: {weights_path}")
    print(f"[INFO] Device: {device}")

    model = torch.hub.load(
        str(repo_dir),
        "custom",
        path=str(weights_path),
        source="local",
        force_reload=False,
    )
    model.to(device)
    model.conf = 0.25
    model.iou = 0.45
    return model, device
