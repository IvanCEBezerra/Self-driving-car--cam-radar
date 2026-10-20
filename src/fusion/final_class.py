from collections import Counter
from math import isfinite


def evaluate_evidence(
    decision,
    camera_threshold=0.25,
    radar_threshold=0.0,
    min_radar_points=1,
    min_match_fraction=0.0,
):
    """Check separate evidence gates for each sensor."""
    for threshold in [
        camera_threshold,
        radar_threshold,
        min_match_fraction,
    ]:
        if not isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Thresholds must be finite and between 0 and 1.")

    if not isinstance(min_radar_points, int) or min_radar_points < 1:
        raise ValueError("Minimum radar points must be a positive integer.")

    matches = decision["camera_matches"]
    ambiguous = len(matches) > 1

    camera_class = None
    camera_score = None

    supported = [m for m in matches if m["camera_class"] is not None]
    if supported:
        selected = max(supported, key=lambda m: m["camera_score"])
        camera_class = selected["camera_class"]
        camera_score = selected["camera_score"]

    radar_class = decision.get("radar_class")
    radar_score = decision.get("radar_score")

    camera_passes = (
        camera_class is not None
        and camera_score is not None
        and isfinite(camera_score)
        and camera_score >= camera_threshold
    )

    radar_passes = (
        radar_class is not None
        and radar_score is not None
        and isfinite(radar_score)
        and radar_score >= radar_threshold
        and decision.get("num_candidate_points", 0) >= min_radar_points
    )

    # A linked candidate needs enough point support inside camera boxes.
    if radar_class is not None and len(matches) == 1:
        radar_passes = (
            radar_passes
            and decision.get("matched_point_fraction", 0)
            >= min_match_fraction
        )

    return {
        "ambiguous": ambiguous,
        "camera_class": camera_class,
        "camera_score": camera_score,
        "camera_passes": camera_passes,
        "radar_class": radar_class,
        "radar_score": radar_score,
        "radar_passes": radar_passes,
    }


def choose_final_class(decision, evidence):
    """Choose a category or retain uncertainty."""
    camera_class = evidence["camera_class"]
    radar_class = evidence["radar_class"]

    camera_passes = evidence["camera_passes"]
    radar_passes = evidence["radar_passes"]

    final_class = "uncertain"
    source = "none"
    reason = "Neither sensor passed its evidence gate."

    if camera_class is not None:
        final_class = camera_class
        source = "camera"
        reason = "Camera priority; highest camera score resolves multiple boxes."
    elif radar_class is not None:
        final_class = radar_class
        source = "radar"
        reason = "Radar fallback because no supported camera category is available."
    else:
        reason = "Neither sensor supplied a supported category."

    result = dict(decision)

    result.update({
        "final_class": final_class,
        "decision_source": source,
        "decision_reason": reason,
        "camera_passes_gate": camera_passes,
        "radar_passes_gate": radar_passes,
        "requires_validation": True,
        "association_ambiguous": evidence["ambiguous"],
        "selection_policy": "camera_priority_radar_fallback",
    })

    return result


def finalize_fusion(
    fusion,
    camera_threshold=0.25,
    radar_threshold=0.0,
    min_radar_points=1,
    min_match_fraction=0.0,
):
    """Add final categories without removing input decisions."""
    decisions = []

    for decision in fusion["decisions"]:
        evidence = evaluate_evidence(
            decision,
            camera_threshold=camera_threshold,
            radar_threshold=radar_threshold,
            min_radar_points=min_radar_points,
            min_match_fraction=min_match_fraction,
        )

        final_decision = choose_final_class(decision, evidence)
        decisions.append(final_decision)

    return {
        "decisions": decisions,
        "final_class_counts": dict(Counter(
            decision["final_class"] for decision in decisions
        )),
        "decision_source_counts": dict(Counter(
            decision["decision_source"] for decision in decisions
        )),
        "settings": {
            "camera_threshold": camera_threshold,
            "radar_threshold": radar_threshold,
            "min_radar_points": min_radar_points,
            "min_match_fraction": min_match_fraction,
        },
    }