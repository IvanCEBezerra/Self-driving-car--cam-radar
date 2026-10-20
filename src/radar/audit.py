import numpy as np
from scipy.spatial import ConvexHull


def get_radar_boxes(dataset, radar_token):
    """Get annotated boxes in the radar coordinate frame."""
    sensor_data = dataset.get_sample_data(radar_token)
    boxes = sensor_data[1]
    return boxes

def find_points_in_box(points, box):
    """Check which points lie inside the box XY footprint."""
    corners = box.corners()
    footprint = corners[:2].T
    footprint = np.unique(footprint, axis=0)
    hull = ConvexHull(footprint)

    mask = np.ones(len(points), dtype=bool)

    for equation in hull.equations:
        distances = (
            equation[0] * points[:, 0]
            + equation[1] * points[:, 1]
            + equation[2]
        )

        mask = mask & (distances <= 0.000001)

    return mask

def audit_candidates(filter_result, candidates, boxes, min_fraction=0.5):
    """Audit candidates using point-to-box masks computed once per acquisition."""
    object_records = []
    candidate_records = []
    objects_by_token = {}
    box_masks = {}
    point_positions = {}

    for position, original_index in enumerate(filter_result["original_indices"]):
        point_positions[int(original_index)] = position

    for box in boxes:
        mask = find_points_in_box(filter_result["points"], box)
        box_masks[box.token] = mask

        record = {
            "box_token": box.token,
            "category": box.name,
            "num_points": int(mask.sum()),
            "accepted_candidate_ids": [],
        }

        object_records.append(record)
        objects_by_token[box.token] = record

    for candidate in candidates:
        matches = []
        positions = []

        for original_index in candidate["original_indices"]:
            position = point_positions[int(original_index)]
            positions.append(position)

        for box in boxes:
            mask = box_masks[box.token][positions]
            num_inside = int(mask.sum())

            if num_inside > 0:
                match = {
                    "box_token": box.token,
                    "num_inside": num_inside,
                    "fraction": num_inside / len(candidate["points"]),
                }

                matches.append(match)

        status = "no_box_support"
        matched_box_token = None

        if len(matches) > 1:
            status = "ambiguous"

        if len(matches) == 1:
            status = "weak_support"

            if matches[0]["fraction"] >= min_fraction:
                status = "accepted"
                matched_box_token = matches[0]["box_token"]

        record = {
            "candidate_id": candidate["candidate_id"],
            "kind": candidate["kind"],
            "matches": matches,
            "status": status,
            "matched_box_token": matched_box_token,
        }

        candidate_records.append(record)

        if status == "accepted":
            object_record = objects_by_token[matched_box_token]
            object_record["accepted_candidate_ids"].append(
                candidate["candidate_id"]
            )

    result = {
        "objects": object_records,
        "candidates": candidate_records,
    }

    return result