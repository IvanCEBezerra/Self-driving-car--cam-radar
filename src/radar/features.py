import numpy as np


def summarize_values(values):
    """Return the mean and count of finite measurements."""
    valid_values = values[np.isfinite(values)]
    count = len(valid_values)

    if count == 0:
        return np.nan, 0

    mean = float(valid_values.mean())
    return mean, count

def extract_candidate_features(candidate):
    """Extract numerical features without using annotation information."""
    points = candidate["points"]
    quality_flags = candidate["passes_quality_filter"]

    if len(points) == 0:
        raise ValueError("Cannot extract features from an empty candidate.")

    extent_x = float(points[:, 0].max() - points[:, 0].min())
    extent_y = float(points[:, 1].max() - points[:, 1].min())

    mean_rcs, num_rcs_points = summarize_values(points[:, 5])

    velocities = points[:, 8:10]
    finite_velocities = np.isfinite(velocities).all(axis=1)
    velocity_mask = quality_flags & finite_velocities
    selected_velocities = velocities[velocity_mask]

    speeds = np.hypot(
        selected_velocities[:, 0],
        selected_velocities[:, 1],
    )

    mean_speed, num_velocity_points = summarize_values(speeds)

    features = {
        "num_points": len(points),
        "extent_x": extent_x,
        "extent_y": extent_y,
        "mean_rcs": mean_rcs,
        "mean_compensated_speed": mean_speed,
        "num_rcs_points": num_rcs_points,
        "num_velocity_points": num_velocity_points,
        "quality_fraction": float(quality_flags.mean()),
    }

    return features

def build_feature_matrix(candidates):
    """Build one feature row per candidate using a fixed column order."""
    columns = [
        "num_points",
        "extent_x",
        "extent_y",
        "mean_rcs",
        "mean_compensated_speed",
        "num_rcs_points",
        "num_velocity_points",
        "quality_fraction",
    ]

    rows = []

    for candidate in candidates:
        features = extract_candidate_features(candidate)
        row = []

        for column in columns:
            row.append(features[column])

        rows.append(row)

    matrix = np.array(rows, dtype=float)
    matrix = matrix.reshape(len(candidates), len(columns))

    return matrix, columns

