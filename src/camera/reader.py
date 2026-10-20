from pathlib import Path
import numpy as np
from PIL import Image


def read_camera(file_path):
    """Read an image and return an independent RGB pixel array."""
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"Camera file not found: {file_path}")

    with Image.open(file_path) as image:
        rgb_image = image.convert("RGB")
        pixels = np.array(rgb_image)

    return pixels

def read_camera_sample(dataset, sample_token):
    """Read the front camera image and its acquisition metadata."""
    sample = dataset.get("sample", sample_token)
    camera_token = sample["data"]["CAM_FRONT"]
    camera_data = dataset.get("sample_data", camera_token)

    file_path = Path(dataset.dataroot) / camera_data["filename"]
    image = read_camera(file_path)

    calibration = dataset.get(
        "calibrated_sensor",
        camera_data["calibrated_sensor_token"],
    )

    ego_pose = dataset.get(
        "ego_pose",
        camera_data["ego_pose_token"],
    )

    result = {
        "sample_token": sample_token,
        "camera_token": camera_token,
        "channel": "CAM_FRONT",
        "timestamp": camera_data["timestamp"],
        "file_path": file_path,
        "image": image,
        "width": image.shape[1],
        "height": image.shape[0],
        "calibration": calibration,
        "ego_pose": ego_pose,
    }

    return result

