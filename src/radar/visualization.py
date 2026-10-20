import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

def create_radar_figure(filter_result):
    """Create an XY plot of the selected radar points."""
    points = filter_result["points"]
    quality_flags = filter_result["passes_quality_filter"]

    passing_points = points[quality_flags]
    other_points = points[~quality_flags]

    figure, axes = plt.subplots(figsize=(9, 6))

    axes.scatter(
        other_points[:, 0],
        other_points[:, 1],
        color="orange",
        s=20,
        label="Other preserved returns",
    )

    axes.scatter(
        passing_points[:, 0],
        passing_points[:, 1],
        color="blue",
        s=20,
        label="Returns passing the quality filter",
    )

    axes.scatter(
        [0],
        [0],
        color="red",
        marker="x",
        s=80,
        label="Radar origin",
    )

    axes.set_xlabel("Radar X (m)")
    axes.set_ylabel("Radar Y (m)")
    axes.set_title("Front radar — XY view")
    axes.set_aspect("equal", adjustable="box")
    axes.grid(True)
    axes.legend()

    figure.tight_layout()
    return figure

def show_radar(filter_result):
    """Create and display the radar figure."""
    figure = create_radar_figure(filter_result)
    plt.show()
    return figure

def create_audit_figure(filter_result, candidates, boxes, report):
    """Plot radar candidates and annotated boxes with XY point support."""
    figure, axes = plt.subplots(figsize=(12, 8))
    object_records = {}

    for record in report["objects"]:
        object_records[record["box_token"]] = record

    for candidate in candidates:
        points = candidate["points"]
        color = "blue"

        if candidate["kind"] == "isolated_return":
            color = "orange"

        axes.scatter(
            points[:, 0],
            points[:, 1],
            color=color,
            s=20,
        )

        if candidate["kind"] == "cluster":
            center = points[:, :2].mean(axis=0)

            axes.annotate(
                f"C{candidate['candidate_id']}",
                (center[0], center[1]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    for box_number, box in enumerate(boxes):
        record = object_records[box.token]

        if record["num_points"] == 0:
            continue

        accepted_count = len(record["accepted_candidate_ids"])
        color = "red"

        if accepted_count == 1:
            color = "green"

        if accepted_count > 1:
            color = "purple"

        footprint = np.unique(box.corners()[:2].T, axis=0)
        hull = ConvexHull(footprint)
        order = np.append(hull.vertices, hull.vertices[0])
        polygon = footprint[order]

        axes.plot(
            polygon[:, 0],
            polygon[:, 1],
            color=color,
            linewidth=1.5,
        )

        center = footprint.mean(axis=0)

        axes.annotate(
            f"B{box_number}",
            (center[0], center[1]),
            color=color,
            fontsize=9,
        )

    axes.scatter(
        [0],
        [0],
        color="black",
        marker="x",
        s=80,
    )

    axes.scatter([], [], color="blue", label="Cluster returns")
    axes.scatter([], [], color="orange", label="Isolated returns")
    axes.plot([], [], color="green", label="One accepted candidate")
    axes.plot([], [], color="purple", label="Multiple accepted candidates")
    axes.plot([], [], color="red", label="Point support without accepted candidate")
    axes.scatter([], [], color="black", marker="x", label="Radar origin")

    axes.set_xlabel("Radar X (m)")
    axes.set_ylabel("Radar Y (m)")
    axes.set_title("Radar audit — boxes with XY point support")
    axes.set_aspect("equal", adjustable="box")
    axes.grid(True)
    axes.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    figure.tight_layout()
    return figure

def show_audit(filter_result, candidates, boxes, report):
    """Create and display the radar audit figure."""
    figure = create_audit_figure(filter_result, candidates, boxes, report)
    plt.show()
    return figure