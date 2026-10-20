import numpy as np
from nuscenes.utils.splits import create_splits_scenes

from src.radar.reader import read_radar_sample
from src.radar.filters import filter_radar_points
from src.radar.clustering import build_candidates
from src.radar.audit import get_radar_boxes, audit_candidates
from src.radar.features import build_feature_matrix
from src.radar.labels import create_candidate_labels


def get_scene_samples(dataset, scene_names, max_samples_per_scene=None):
    """Select evenly spaced samples from the requested scenes."""
    if max_samples_per_scene is not None:
        if max_samples_per_scene < 1:
            raise ValueError("The sample limit must be at least one.")

    selected_tokens = []

    for scene in dataset.scene:
        if scene["name"] not in scene_names:
            continue

        scene_tokens = []
        token = scene["first_sample_token"]

        while token:
            scene_tokens.append(token)
            sample = dataset.get("sample", token)
            token = sample["next"]

        if max_samples_per_scene is None:
            selected_tokens.extend(scene_tokens)
            continue

        count = min(max_samples_per_scene, len(scene_tokens))

        positions = np.linspace(
            0,
            len(scene_tokens) - 1,
            count,
            dtype=int,
        )

        for position in positions:
            selected_tokens.append(scene_tokens[position])

    return selected_tokens

def build_sample_examples(dataset, sample_token):
    """Combine features, provisional labels and source metadata."""
    sample = dataset.get("sample", sample_token)
    scene = dataset.get("scene", sample["scene_token"])

    radar = read_radar_sample(dataset, sample_token)
    filtered = filter_radar_points(radar["points"])
    candidates = build_candidates(filtered)
    boxes = get_radar_boxes(dataset, radar["radar_token"])

    report = audit_candidates(filtered, candidates, boxes)
    labels = create_candidate_labels(candidates, report)
    matrix, columns = build_feature_matrix(candidates)

    examples = []

    for position, label_record in enumerate(labels):
        candidate = candidates[position]
        instance_token = None

        if label_record["box_token"] is not None:
            annotation = dataset.get(
                "sample_annotation",
                label_record["box_token"],
            )

            instance_token = annotation["instance_token"]

        example = {
            "features": matrix[position].copy(),
            "label": label_record["label"],
            "label_status": label_record["label_status"],
            "category": label_record["category"],
            "kind": candidate["kind"],
            "candidate_id": candidate["candidate_id"],
            "scene_name": scene["name"],
            "scene_token": scene["token"],
            "sample_token": sample_token,
            "radar_token": radar["radar_token"],
            "timestamp": radar["timestamp"],
            "annotation_token": label_record["box_token"],
            "instance_token": instance_token,
            "original_indices": candidate["original_indices"].tolist(),
        }

        examples.append(example)

    return examples, columns

def build_radar_dataset(dataset):
    """Build examples from all mini training scenes."""
    splits = create_splits_scenes()
    scene_names = splits["mini_train"]
    sample_tokens = get_scene_samples(dataset, scene_names)

    examples = []
    columns = None

    for sample_number, sample_token in enumerate(sample_tokens):
        sample_examples, sample_columns = build_sample_examples(
            dataset,
            sample_token,
        )

        if columns is None:
            columns = sample_columns

        if sample_columns != columns:
            raise ValueError("Feature column order changed between samples.")

        examples.extend(sample_examples)

        print(
            f"Processed sample {sample_number + 1}/{len(sample_tokens)} "
            f"| candidates collected: {len(examples)}",
            flush=True,
        )

    if columns is None:
        raise ValueError("No development samples were found.")

    return examples, columns

def summarize_dataset(examples, columns):
    """Summarize labeled examples, diversity and missing features."""
    class_names = ["vehicle", "pedestrian", "two_wheeler", "obstacle"]
    class_records = {}
    status_counts = {}

    for class_name in class_names:
        missing_features = {}

        for column in columns:
            missing_features[column] = 0

        class_records[class_name] = {
            "examples": 0,
            "isolated_examples": 0,
            "scenes": set(),
            "instances": set(),
            "categories": {},
            "missing_features": missing_features,
        }

    for example in examples:
        status = example["label_status"]
        status_counts[status] = status_counts.get(status, 0) + 1

        if example["label"] is None:
            continue

        record = class_records[example["label"]]
        record["examples"] += 1
        record["scenes"].add(example["scene_name"])
        record["instances"].add(example["instance_token"])

        if example["kind"] == "isolated_return":
            record["isolated_examples"] += 1

        category = example["category"]
        previous_count = record["categories"].get(category, 0)
        record["categories"][category] = previous_count + 1

        for position, column in enumerate(columns):
            if np.isnan(example["features"][position]):
                record["missing_features"][column] += 1

    for record in class_records.values():
        record["scene_names"] = sorted(record["scenes"])
        record["scenes"] = len(record["scenes"])
        record["instances"] = len(record["instances"])

    result = {
        "total_candidates": len(examples),
        "status_counts": status_counts,
        "classes": class_records,
    }

    return result

