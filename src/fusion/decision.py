from pathlib import Path

import joblib
import numpy as np

from src.radar.features import build_feature_matrix
from src.radar.model import predict_model
from src.fusion.association import summarize_ranges


def load_radar_model(file_path):
    """Load a saved radar model and its feature schema."""
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"Radar model not found: {file_path}")

    saved = joblib.load(file_path)

    required_fields = ["model", "model_name", "feature_columns"]

    for field in required_fields:
        if field not in saved:
            raise ValueError(f"Saved model is missing: {field}")

    return {
        "model": saved["model"],
        "model_name": saved["model_name"],
        "feature_columns": saved["feature_columns"],
        "file_path": file_path,
    }


def predict_radar_candidates(radar_model, candidates):
    """Classify candidates using the original training feature order."""
    features, columns = build_feature_matrix(candidates)

    if columns != radar_model["feature_columns"]:
        raise ValueError("Radar features do not match the training schema.")

    if len(candidates) == 0:
        return []

    candidate_ids = [
        candidate["candidate_id"] for candidate in candidates
    ]

    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Radar candidate IDs must be unique.")

    prediction = predict_model(radar_model["model"], features)
    records = []

    for index, candidate in enumerate(candidates):
        predicted_class = str(prediction["predictions"][index])
        class_scores = {}

        for class_name, score in zip(
            prediction["classes"],
            prediction["scores"][index],
        ):
            class_scores[str(class_name)] = float(score)

        records.append({
            "candidate_id": candidate["candidate_id"],
            "kind": candidate["kind"],
            "radar_class": predicted_class,
            "radar_score": class_scores[predicted_class],
            "radar_class_scores": class_scores,
            "radar_indices": [
                int(value) for value in candidate["original_indices"]
            ],
            "num_points": len(candidate["points"]),
            "model_name": radar_model["model_name"],
        })

    return records


def build_fusion_decisions(radar_predictions, association):
    """Compare sensor predictions while retaining uncertain associations."""
    points_by_radar_index = {}

    for point in association["points"]:
        radar_index = point["radar_index"]

        if radar_index in points_by_radar_index:
            raise ValueError("Projected radar indices must be unique.")

        points_by_radar_index[radar_index] = point

    detections_by_id = {}

    for detection in association["detections"]:
        detection_id = detection["detection_id"]

        if detection_id in detections_by_id:
            raise ValueError("Camera detection IDs must be unique.")

        detections_by_id[detection_id] = detection

    decisions = []
    camera_candidate_ids = {
        detection_id: [] for detection_id in detections_by_id
    }

    for prediction in radar_predictions:
        projected_points = []

        for radar_index in prediction["radar_indices"]:
            point = points_by_radar_index.get(radar_index)

            if point is not None:
                projected_points.append(point)

        matched_points = [
            point for point in projected_points
            if point["detection_ids"]
        ]

        matched_ids = set()

        for point in matched_points:
            matched_ids.update(point["detection_ids"])

        matched_ids = sorted(matched_ids)

        camera_matches = []

        for detection_id in matched_ids:
            if detection_id not in detections_by_id:
                raise ValueError("Point references an unknown detection.")

            detection = detections_by_id[detection_id]

            camera_candidate_ids[detection_id].append(
                prediction["candidate_id"]
            )

            camera_matches.append({
                "detection_id": detection_id,
                "camera_class": detection["project_class"],
                "original_class": detection["original_class"],
                "camera_score": detection["confidence"],
            })

        if len(matched_ids) == 0:
            status = "radar_only"

        elif len(matched_ids) > 1:
            status = "ambiguous_association"

        else:
            camera_class = camera_matches[0]["camera_class"]

            if camera_class is None:
                status = "unsupported_camera_class"
            elif camera_class == prediction["radar_class"]:
                status = "agreement"
            else:
                status = "class_conflict"

        exclusive_ranges = [
            point["radar_range"]
            for point in matched_points
            if point["status"] == "unique_match"
        ]

        matched_fraction = (
            len(matched_points) / prediction["num_points"]
            if prediction["num_points"] else 0.0
        )

        decisions.append({
            "status": status,
            "candidate_id": prediction["candidate_id"],
            "candidate_kind": prediction["kind"],
            "radar_model": prediction["model_name"],
            "radar_class": prediction["radar_class"],
            "radar_score": prediction["radar_score"],
            "radar_class_scores": prediction["radar_class_scores"],
            "camera_matches": camera_matches,
            "num_candidate_points": prediction["num_points"],
            "num_projected_points": len(projected_points),
            "num_matched_points": len(matched_points),
            "matched_point_fraction": matched_fraction,
            "num_ambiguous_points": sum(
                point["status"] == "ambiguous"
                for point in matched_points
            ),
            "exclusive_range_summary": summarize_ranges(
                exclusive_ranges
            ),
        })

    # Preserve camera detections without any associated radar candidate.
    for detection_id, candidate_ids in camera_candidate_ids.items():
        if candidate_ids:
            continue

        detection = detections_by_id[detection_id]

        decisions.append({
            "status": "camera_only",
            "candidate_id": None,
            "radar_class": None,
            "radar_score": None,
            "camera_matches": [{
                "detection_id": detection_id,
                "camera_class": detection["project_class"],
                "original_class": detection["original_class"],
                "camera_score": detection["confidence"],
            }],
        })

    status_counts = {}

    for decision in decisions:
        status = decision["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "decisions": decisions,
        "camera_candidate_ids": camera_candidate_ids,
        "status_counts": status_counts,
    }