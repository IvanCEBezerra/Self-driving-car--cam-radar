from pathlib import Path
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.nuscenes import NuScenes


def read_radar(file_path):
    """Read a nuScenes radar file and return its points."""
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"Radar file not found: {file_path}")

    point_cloud = RadarPointCloud.from_file(
        str(file_path),
        invalid_states=list(range(18)),
        dynprop_states=list(range(8)),
        ambig_states=list(range(5)),
    )

    points = point_cloud.points.T.copy()
    return points




def load_dataset(dataset_path):
    """Load the nuScenes mini metadata."""
    dataset_path = Path(dataset_path)

    dataset = NuScenes(
        version="v1.0-mini",
        dataroot=str(dataset_path),
        verbose=False,
    )

    return dataset

def read_radar_sample(dataset, sample_token):
    """Read front radar points and their acquisition metadata."""
    sample = dataset.get("sample", sample_token)
    radar_token = sample["data"]["RADAR_FRONT"]
    radar_data = dataset.get("sample_data", radar_token)

    file_path = Path(dataset.dataroot) / radar_data["filename"]
    points = read_radar(file_path)

    calibration = dataset.get(
        "calibrated_sensor",
        radar_data["calibrated_sensor_token"],
    )

    ego_pose = dataset.get(
        "ego_pose",
        radar_data["ego_pose_token"],
    )

    result = {
        "sample_token": sample_token,
        "radar_token": radar_token,
        "channel": "RADAR_FRONT",
        "timestamp": radar_data["timestamp"],
        "file_path": file_path,
        "points": points,
        "calibration": calibration,
        "ego_pose": ego_pose,
    }

    return result