import numpy as np
from sklearn.cluster import DBSCAN


def cluster_points(points, eps=1.5, min_samples=2):
    """Group radar points using their XY coordinates."""
    if len(points) == 0:
        return np.array([], dtype=int)

    coordinates = points[:, :2]

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
    )

    labels = model.fit_predict(coordinates)
    return labels

def create_candidate(filter_result, indices, kind):
    """Create a candidate while preserving its source information."""
    candidate = {
        "kind": kind,
        "points": filter_result["points"][indices],
        "original_indices": filter_result["original_indices"][indices],
        "passes_quality_filter": filter_result["passes_quality_filter"][indices],
    }

    return candidate

def build_candidates(filter_result, eps=1.5, min_samples=2):
    """Build cluster candidates and preserve isolated returns."""
    points = filter_result["points"]
    labels = cluster_points(points, eps, min_samples)
    candidates = []

    for label in np.unique(labels):
        if label == -1:
            continue

        indices = np.flatnonzero(labels == label)
        candidate = create_candidate(filter_result, indices, "cluster")
        candidates.append(candidate)

    isolated_indices = np.flatnonzero(labels == -1)

    for index in isolated_indices:
        indices = np.array([index])
        candidate = create_candidate(filter_result, indices, "isolated_return")
        candidates.append(candidate)

    for candidate_id, candidate in enumerate(candidates):
        candidate["candidate_id"] = candidate_id

    return candidates