from nuscenes.utils.splits import create_splits_scenes
from src.radar.reader import read_radar_sample
from src.radar.filters import filter_radar_points
from src.radar.clustering import build_candidates
from src.radar.audit import get_radar_boxes, audit_candidates


def get_development_samples(dataset):
    """Select the first, middle and last samples of each mini training scene."""
    splits = create_splits_scenes()
    scene_names = splits["mini_train"]
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

        positions = [0, len(scene_tokens) // 2, len(scene_tokens) - 1]

        for position in sorted(set(positions)):
            selected_tokens.append(scene_tokens[position])

    return selected_tokens

def summarize_audit(candidates, report):
    """Summarize geometric associations, separating clusters from isolated returns."""
    summary = {
        "supported_boxes": 0,
        "cluster_covered_boxes": 0,
        "fragmented_boxes": 0,
        "clusters": 0,
        "accepted_clusters": 0,
        "ambiguous_clusters": 0,
        "unsupported_clusters": 0,
        "isolated_returns": 0,
    }

    for record in report["candidates"]:
        if record["kind"] == "isolated_return":
            summary["isolated_returns"] += 1
            continue

        summary["clusters"] += 1

        if record["status"] == "accepted":
            summary["accepted_clusters"] += 1

        if record["status"] == "ambiguous":
            summary["ambiguous_clusters"] += 1

        if record["status"] == "no_box_support":
            summary["unsupported_clusters"] += 1

    for record in report["objects"]:
        if record["num_points"] == 0:
            continue

        summary["supported_boxes"] += 1
        accepted_clusters = 0

        for candidate_id in record["accepted_candidate_ids"]:
            if candidates[candidate_id]["kind"] == "cluster":
                accepted_clusters += 1

        if accepted_clusters > 0:
            summary["cluster_covered_boxes"] += 1

        if accepted_clusters > 1:
            summary["fragmented_boxes"] += 1

    return summary


def compare_parameters(dataset):
    """Compare DBSCAN settings using a fixed sample selection and audit policy."""
    sample_tokens = get_development_samples(dataset)
    results = []

    for eps in [0.75, 1.0, 1.5, 2.0]:
        for min_samples in [2, 3]:
            result = {
                "eps": eps,
                "min_samples": min_samples,
                "totals": {},
                "per_sample": [],
            }

            results.append(result)

    for sample_number, sample_token in enumerate(sample_tokens):
        radar = read_radar_sample(dataset, sample_token)
        filtered = filter_radar_points(radar["points"])
        boxes = get_radar_boxes(dataset, radar["radar_token"])

        for result in results:
            candidates = build_candidates(
                filtered,
                eps=result["eps"],
                min_samples=result["min_samples"],
            )

            report = audit_candidates(
                filtered,
                candidates,
                boxes,
                min_fraction=0.5,
            )

            summary = summarize_audit(candidates, report)

            for name, value in summary.items():
                previous_value = result["totals"].get(name, 0)
                result["totals"][name] = previous_value + value

            result["per_sample"].append({
                "sample_token": sample_token,
                "metrics": summary,
            })

        print(
            f"Compared sample {sample_number + 1}/{len(sample_tokens)}",
            flush=True,
        )

    return results