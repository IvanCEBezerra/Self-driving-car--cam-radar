import numpy as np


def create_position_mask(points):
    """Identify points with finite coordinates."""
    coordinates = points[:, :3]
    mask = np.isfinite(coordinates).all(axis=1)
    return mask

def create_quality_mask(points):
    """Identify points passing the selected quality policy."""
    position_mask = create_position_mask(points)
    valid_state = points[:, 14] == 0
    resolved_ambiguity = points[:, 11] == 3

    mask = position_mask & valid_state & resolved_ambiguity
    return mask

def filter_radar_points(points, use_quality_filter=False):
    """Select radar points and preserve their original indices."""
    position_mask = create_position_mask(points)
    quality_mask = create_quality_mask(points)

    mask = position_mask

    if use_quality_filter:
        mask = quality_mask

    selected_points = points[mask]
    original_indices = np.flatnonzero(mask)
    quality_flags = quality_mask[mask]

    result = {
        "points": selected_points,
        "original_indices": original_indices,
        "passes_quality_filter": quality_flags,
        "num_input": len(points),
        "num_output": len(selected_points),
    }

    return result