def map_category(category):
    """Map a nuScenes category to an MVP class."""
    if category in ["vehicle.bicycle", "vehicle.motorcycle"]:
        return "two_wheeler"

    if category.startswith("vehicle."):
        return "vehicle"

    pedestrian_categories = [
        "human.pedestrian.adult",
        "human.pedestrian.child",
        "human.pedestrian.construction_worker",
        "human.pedestrian.police_officer",
    ]

    if category in pedestrian_categories:
        return "pedestrian"

    if category in ["movable_object.barrier", "movable_object.trafficcone"]:
        return "obstacle"

    return None

def create_candidate_labels(candidates, report):
    """Create provisional labels from accepted audit associations."""
    objects_by_token = {}
    records_by_id = {}

    for record in report["objects"]:
        objects_by_token[record["box_token"]] = record

    for record in report["candidates"]:
        records_by_id[record["candidate_id"]] = record

    labels = []

    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        audit_record = records_by_id[candidate_id]

        label = None
        category = None
        box_token = None
        label_status = audit_record["status"]

        if audit_record["status"] == "accepted":
            box_token = audit_record["matched_box_token"]
            object_record = objects_by_token[box_token]
            category = object_record["category"]
            label = map_category(category)
            label_status = "labeled"

            if label is None:
                label_status = "unsupported_category"

        record = {
            "candidate_id": candidate_id,
            "kind": candidate["kind"],
            "label": label,
            "label_status": label_status,
            "category": category,
            "box_token": box_token,
        }

        labels.append(record)

    return labels

def summarize_labels(labels):
    """Count classes, labeling reasons and labeled candidate kinds."""
    class_counts = {
        "vehicle": 0,
        "pedestrian": 0,
        "two_wheeler": 0,
        "obstacle": 0,
    }

    status_counts = {}
    labeled_clusters = 0
    labeled_isolated = 0

    for record in labels:
        status = record["label_status"]
        previous_count = status_counts.get(status, 0)
        status_counts[status] = previous_count + 1

        if record["label"] is None:
            continue

        class_counts[record["label"]] += 1

        if record["kind"] == "cluster":
            labeled_clusters += 1

        if record["kind"] == "isolated_return":
            labeled_isolated += 1

    result = {
        "class_counts": class_counts,
        "status_counts": status_counts,
        "labeled_clusters": labeled_clusters,
        "labeled_isolated": labeled_isolated,
        "total_candidates": len(labels),
    }

    return result