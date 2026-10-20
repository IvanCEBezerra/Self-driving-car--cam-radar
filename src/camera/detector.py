from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO


def load_detector(model_path=None):
    """Load the detector and select GPU or CPU."""
    if model_path is None:
        project_path = Path(__file__).resolve().parents[2]
        model_path = project_path / "models" / "camera" / "yolo11n.pt"

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))

    return {
        "model": model,
        "device": device,
        "model_path": model_path,
    }


def map_camera_class(original_class):
    """Map supported detector classes to project classes."""
    class_mapping = {
        "car": "vehicle",
        "truck": "vehicle",
        "bus": "vehicle",
        "person": "pedestrian",
        "bicycle": "two_wheeler",
        "motorcycle": "two_wheeler",
    }

    return class_mapping.get(original_class)


def detect_objects(detector, image, confidence=0.25, image_size=960):
    """Detect objects in an RGB image."""
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a NumPy array.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must have three RGB channels.")

    if image.dtype != np.uint8:
        raise ValueError("Image must use uint8 pixels.")

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Image dimensions must be positive.")

    if not 0 < confidence <= 1:
        raise ValueError("Confidence must be greater than zero and at most one.")

    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("Image size must be a positive integer.")

    rgb_image = Image.fromarray(image)

    results = detector["model"].predict(
        source=rgb_image,
        device=detector["device"],
        conf=confidence,
        imgsz=image_size,
        verbose=False,
        save=False,
    )

    result = results[0]
    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        original_class = result.names[class_id]
        project_class = map_camera_class(original_class)

        detection = {
            "detection_id": len(detections),
            "box": box.xyxy[0].cpu().tolist(),
            "class_id": class_id,
            "original_class": original_class,
            "project_class": project_class,
            "confidence": float(box.conf[0].item()),
        }

        detections.append(detection)

    return detections