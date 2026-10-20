import numpy as np
from pyquaternion import Quaternion


def transform_to_parent(points, calibration):
    """Transform positions from a local frame to its parent frame."""
    rotation = Quaternion(calibration["rotation"]).rotation_matrix
    translation = np.array(calibration["translation"])

    transformed = points @ rotation.T + translation

    return transformed


def transform_from_parent(points, calibration):
    """Transform positions from a parent frame to its local frame."""
    rotation = Quaternion(calibration["rotation"]).rotation_matrix
    translation = np.array(calibration["translation"])

    transformed = (points - translation) @ rotation

    return transformed


def radar_to_camera(radar, camera):
    """Transform radar positions into the camera coordinate frame."""
    points = np.asarray(radar["points"])

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("Radar points must have at least three columns.")

    positions = points[:, :3].copy()

    vehicle_points = transform_to_parent(
        positions,
        radar["calibration"],
    )

    world_points = transform_to_parent(
        vehicle_points,
        radar["ego_pose"],
    )

    camera_vehicle_points = transform_from_parent(
        world_points,
        camera["ego_pose"],
    )

    camera_points = transform_from_parent(
        camera_vehicle_points,
        camera["calibration"],
    )

    return camera_points


def project_radar_to_image(radar, camera, min_depth=1.0):
    """Project radar positions and retain points inside the image."""
    if not np.isfinite(min_depth) or min_depth <= 0:
        raise ValueError("Minimum depth must be finite and positive.")

    if radar["sample_token"] != camera["sample_token"]:
        raise ValueError("Radar and camera must belong to the same sample.")

    intrinsics = np.array(
        camera["calibration"]["camera_intrinsic"],
        dtype=float,
    )

    if intrinsics.shape != (3, 3):
        raise ValueError("Camera intrinsics must be a 3x3 matrix.")

    if not np.isfinite(intrinsics).all():
        raise ValueError("Camera intrinsics must be finite.")

    width = camera["width"]
    height = camera["height"]

    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")

    camera_points = radar_to_camera(radar, camera)
    depths = camera_points[:, 2]

    finite_mask = np.isfinite(camera_points).all(axis=1)
    behind_mask = finite_mask & (depths <= 0)
    near_mask = finite_mask & (depths > 0) & (depths <= min_depth)
    projectable_mask = finite_mask & (depths > min_depth)

    pixels = np.full((len(camera_points), 2), np.nan)

    selected_points = camera_points[projectable_mask]
    projected = selected_points @ intrinsics.T

    pixels[projectable_mask] = (
        projected[:, :2] / projected[:, 2:3]
    )

    inside_mask = (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )

    visible_mask = projectable_mask & inside_mask
    outside_mask = projectable_mask & ~inside_mask

    point_indices = np.flatnonzero(visible_mask)

    radar_positions = np.asarray(radar["points"])[:, :3]
    radar_ranges = np.linalg.norm(
        radar_positions[visible_mask],
        axis=1,
    )

    summary = {
        "input_points": len(camera_points),
        "invalid_points": int((~finite_mask).sum()),
        "behind_camera": int(behind_mask.sum()),
        "too_close": int(near_mask.sum()),
        "outside_image": int(outside_mask.sum()),
        "visible_points": int(visible_mask.sum()),
    }

    result = {
        "pixels": pixels[visible_mask],
        "depths": depths[visible_mask],
        "radar_ranges": radar_ranges,
        "point_indices": point_indices,
        "camera_points": camera_points[visible_mask],
        "visible_mask": visible_mask,
        "time_difference_ms": (
            camera["timestamp"] - radar["timestamp"]
        ) / 1000,
        "summary": summary,
    }

    return result