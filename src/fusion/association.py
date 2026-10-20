import numpy as np


def find_points_in_box(pixels, box):
    """Return projected-point indices inside a detection box."""
    pixels = np.asarray(pixels)
    box = np.asarray(box, dtype=float)

    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("Pixels must have two columns.")

    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("Box must contain four finite coordinates.")

    x_min, y_min, x_max, y_max = box

    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Box dimensions must be positive.")

    inside = (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= x_min)
        & (pixels[:, 0] <= x_max)
        & (pixels[:, 1] >= y_min)
        & (pixels[:, 1] <= y_max)
    )

    return np.flatnonzero(inside)


def summarize_ranges(ranges):
    """Summarize distances without assuming one physical object."""
    ranges = np.asarray(ranges, dtype=float)
    valid_ranges = ranges[np.isfinite(ranges)]

    if len(valid_ranges) == 0:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "maximum": None,
            "spread": None,
        }

    minimum = float(valid_ranges.min())
    maximum = float(valid_ranges.max())

    return {
        "count": len(valid_ranges),
        "minimum": minimum,
        "median": float(np.median(valid_ranges)),
        "maximum": maximum,
        "spread": maximum - minimum,
    }


def associate_points(detections, projection):
    """Associate projected points with boxes and preserve ambiguity."""
    pixels = projection["pixels"]
    radar_indices = projection["point_indices"]
    radar_ranges = projection["radar_ranges"]

    if not (
        len(pixels) == len(radar_indices) == len(radar_ranges)
    ):
        raise ValueError("Projection fields must have matching lengths.")

    detection_ids = [
        detection["detection_id"] for detection in detections
    ]

    if len(set(detection_ids)) != len(detection_ids):
        raise ValueError("Detection IDs must be unique.")

    point_matches = [[] for _ in range(len(pixels))]
    detection_matches = []

    # Record every geometric match before resolving ambiguity.
    for detection in detections:
        indices = find_points_in_box(pixels, detection["box"])
        detection_matches.append(indices)

        for index in indices:
            point_matches[index].append(detection["detection_id"])

    point_records = []

    # Each point gets one status, even when it matches several boxes.
    for index, matched_ids in enumerate(point_matches):
        if len(matched_ids) == 0:
            status = "unmatched"
        elif len(matched_ids) == 1:
            status = "unique_match"
        else:
            status = "ambiguous"

        point_records.append({
            "projection_index": index,
            "radar_index": int(radar_indices[index]),
            "pixel": pixels[index].tolist(),
            "radar_range": float(radar_ranges[index]),
            "detection_ids": matched_ids.copy(),
            "status": status,
        })

    detection_records = []

    # Separate points exclusive to a box from points shared with others.
    for detection, indices in zip(detections, detection_matches):
        unique_indices = [
            int(index)
            for index in indices
            if len(point_matches[index]) == 1
        ]

        ambiguous_indices = [
            int(index)
            for index in indices
            if len(point_matches[index]) > 1
        ]

        if len(indices) == 0:
            status = "no_radar_support"
        elif len(unique_indices) == 0:
            status = "ambiguous_support_only"
        elif len(ambiguous_indices) > 0:
            status = "mixed_support"
        else:
            status = "unique_support"

        unique_ranges = [
            radar_ranges[index] for index in unique_indices
        ]

        detection_records.append({
            "detection_id": detection["detection_id"],
            "box": list(detection["box"]),
            "original_class": detection["original_class"],
            "project_class": detection["project_class"],
            "confidence": detection["confidence"],
            "status": status,
            "num_points": len(indices),
            "num_unique_points": len(unique_indices),
            "num_ambiguous_points": len(ambiguous_indices),
            "radar_indices": [
                int(radar_indices[index]) for index in indices
            ],
            "unique_radar_indices": [
                int(radar_indices[index]) for index in unique_indices
            ],
            "unique_range_summary": summarize_ranges(unique_ranges),
        })

    summary = {
        "projected_points": len(point_records),
        "unmatched_points": sum(
            record["status"] == "unmatched"
            for record in point_records
        ),
        "unique_points": sum(
            record["status"] == "unique_match"
            for record in point_records
        ),
        "ambiguous_points": sum(
            record["status"] == "ambiguous"
            for record in point_records
        ),
        "detections": len(detection_records),
        "detections_without_radar": sum(
            record["status"] == "no_radar_support"
            for record in detection_records
        ),
        "detections_with_unique_points": sum(
            record["num_unique_points"] > 0
            for record in detection_records
        ),
        "detections_with_ambiguous_points": sum(
            record["num_ambiguous_points"] > 0
            for record in detection_records
        ),
    }

    return {
        "points": point_records,
        "detections": detection_records,
        "summary": summary,
    }