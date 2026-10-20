from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import Counter
from importlib.metadata import version
import csv
import hashlib
import json
import sys

import joblib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from nuscenes.utils.splits import create_splits_scenes


# Find the project whether this file is in the root, notebooks or src/radar.
PROJECT_PATH = Path(__file__).resolve().parent

while not (PROJECT_PATH / "src" / "radar").is_dir():
    if PROJECT_PATH == PROJECT_PATH.parent:
        raise FileNotFoundError("Could not locate the project's src/radar directory.")

    PROJECT_PATH = PROJECT_PATH.parent

sys.path.insert(0, str(PROJECT_PATH))

from src.radar.reader import load_dataset
from src.radar.dataset import get_scene_samples, build_sample_examples
from src.radar.dataset import summarize_dataset
from src.radar.split import build_model_inputs
from src.radar.validation import cross_validate_model
from src.radar.model import create_model, train_model, predict_model


CLASSES = ["vehicle", "pedestrian", "two_wheeler", "obstacle"]
MINORITY_CLASSES = ["pedestrian", "two_wheeler", "obstacle"]

MODEL_CANDIDATES = [
    "random_forest",
    "balanced_random_forest",
    "moderate_random_forest",
]

N_SPLITS = 4


def json_value(value):
    """Convert NumPy values and paths for JSON reports."""
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    raise TypeError(f"Cannot serialize {type(value)}")


def save_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
            default=json_value,
        )


def save_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    """Fingerprint a file without reading it entirely into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def ratio_percentage(numerator, denominator):
    """Undefined metrics remain unavailable."""
    if denominator == 0:
        return None

    return 100 * numerator / denominator


def ratio_text(numerator, denominator):
    """Counts preserve the exact ratio; the decimal is rounded."""
    if denominator == 0:
        return "N/A"

    return f"{numerator}/{denominator} = {100 * numerator / denominator:.6f}%"


def calculate_metrics(reference, predictions):
    """Calculate candidate classification metrics from exact counts."""
    matrix = confusion_matrix(
        reference,
        predictions,
        labels=CLASSES,
    )

    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    records = []

    for position, class_name in enumerate(CLASSES):
        tp = int(matrix[position, position])
        reference_count = int(matrix[position].sum())
        predicted_count = int(matrix[:, position].sum())

        fp = predicted_count - tp
        fn = reference_count - tp
        tn = total - tp - fp - fn
        f1_denominator = 2 * tp + fp + fn

        records.append({
            "class": class_name,
            "reference_count": reference_count,
            "predicted_count": predicted_count,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision_pct": ratio_percentage(tp, tp + fp),
            "precision_exact": ratio_text(tp, tp + fp),
            "recall_pct": ratio_percentage(tp, tp + fn),
            "recall_exact": ratio_text(tp, tp + fn),
            "f1_pct": ratio_percentage(2 * tp, f1_denominator),
            "f1_exact": ratio_text(2 * tp, f1_denominator),
            "specificity_pct": ratio_percentage(tn, tn + fp),
            "specificity_exact": ratio_text(tn, tn + fp),
            "false_positive_rate_pct": ratio_percentage(fp, fp + tn),
            "false_negative_rate_pct": ratio_percentage(fn, fn + tp),
        })

    # Undefined class metrics contribute zero to these fixed-class averages.
    # Per-class reports still display N/A and preserve the support counts.
    macro_f1 = sum(record["f1_pct"] or 0 for record in records) / len(CLASSES)
    macro_recall = sum(record["recall_pct"] or 0 for record in records) / len(CLASSES)

    minority_recalls = []

    for record in records:
        if record["class"] in MINORITY_CLASSES:
            minority_recalls.append(record["recall_pct"] or 0)

    return {
        "candidate_rows": total,
        "correct_predictions": correct,
        "accuracy_pct": ratio_percentage(correct, total),
        "accuracy_exact": ratio_text(correct, total),
        "macro_f1_pct": macro_f1,
        "macro_recall_pct": macro_recall,
        "minority_macro_recall_pct": sum(minority_recalls) / len(MINORITY_CLASSES),
        "confusion_class_order": CLASSES,
        "confusion_matrix": matrix,
        "classes": records,
    }


def metric_text(title, metrics):
    lines = [
        title,
        "",
        f"Candidate rows: {metrics['candidate_rows']}",
        f"Accuracy: {metrics['accuracy_exact']}",
        f"Macro F1: {metrics['macro_f1_pct']:.6f}%",
        f"Macro recall: {metrics['macro_recall_pct']:.6f}%",
        f"Minority macro recall: {metrics['minority_macro_recall_pct']:.6f}%",
        "",
        "PER-CLASS RESULTS",
    ]

    for record in metrics["classes"]:
        lines.extend([
            "",
            f"Class: {record['class']}",
            f"Reference rows: {record['reference_count']}",
            f"Predicted rows: {record['predicted_count']}",
            f"TP={record['TP']} | FP={record['FP']} | FN={record['FN']} | TN={record['TN']}",
            f"Precision: {record['precision_exact']}",
            f"Recall: {record['recall_exact']}",
            f"F1: {record['f1_exact']}",
            f"Specificity: {record['specificity_exact']}",
        ])

        if record["reference_count"] == 0:
            lines.append("No reference examples: recall cannot be evaluated.")

    lines.extend([
        "",
        "Fractions preserve exact counts; percentages are rounded.",
        "These metrics describe labeled candidates, not physical object detection recall.",
        "Reference labels are provisional geometric associations.",
    ])

    return "\n".join(lines)


def save_metric_bundle(folder, title, metrics):
    """Save numerical reports and charts containing numbers."""
    folder.mkdir(parents=True, exist_ok=True)

    save_json(folder / "metrics.json", metrics)

    text = metric_text(title, metrics)
    (folder / "metrics.txt").write_text(text, encoding="utf-8")

    save_csv(
        folder / "class_metrics.csv",
        list(metrics["classes"][0]),
        metrics["classes"],
    )

    matrix = np.array(metrics["confusion_matrix"])

    with (folder / "confusion_counts.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["reference_class"] + CLASSES)

        for position, class_name in enumerate(CLASSES):
            writer.writerow([class_name] + matrix[position].tolist())

    # Confusion chart with exact counts.
    figure, axes = plt.subplots(figsize=(8, 6))
    image = axes.imshow(matrix, cmap="Blues")

    axes.set_xticks(range(len(CLASSES)), labels=CLASSES)
    axes.set_yticks(range(len(CLASSES)), labels=CLASSES)
    axes.set_xlabel("Predicted class")
    axes.set_ylabel("Reference class")
    axes.set_title(title)

    plt.setp(axes.get_xticklabels(), rotation=25, ha="right")
    threshold = matrix.max() / 2

    for row in range(len(CLASSES)):
        for column in range(len(CLASSES)):
            axes.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )

    figure.colorbar(image, ax=axes)
    figure.tight_layout()
    figure.savefig(folder / "confusion_counts.png", dpi=180)
    plt.close(figure)

    # Metric chart is a numerical table, not a bar chart.
    table_rows = []

    for record in metrics["classes"]:
        table_rows.append([
            record["class"],
            record["reference_count"],
            record["predicted_count"],
            record["precision_exact"],
            record["recall_exact"],
            record["f1_exact"],
        ])

    figure, axes = plt.subplots(figsize=(19, 4))
    axes.axis("off")

    table = axes.table(
        cellText=table_rows,
        colLabels=["Class", "Reference", "Predicted", "Precision", "Recall", "F1"],
        colWidths=[0.10, 0.07, 0.07, 0.25, 0.25, 0.26],
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    axes.set_title(title, pad=20)

    figure.tight_layout()
    figure.savefig(
        folder / "metric_numbers.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)
    print("\n" + text)


def export_examples(path, examples, columns):
    """Preserve all candidates, including unlabeled ones."""
    metadata_columns = [
        "scene_name", "scene_token", "sample_token", "radar_token",
        "timestamp", "candidate_id", "kind", "label", "label_status",
        "category", "annotation_token", "instance_token",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=metadata_columns + ["original_indices"] + columns,
        )

        writer.writeheader()

        for example in examples:
            row = {}

            for name in metadata_columns:
                row[name] = example[name]

            row["original_indices"] = json.dumps(example["original_indices"])

            for position, column in enumerate(columns):
                value = example["features"][position]
                row[column] = "" if np.isnan(value) else float(value)

            writer.writerow(row)


def export_predictions(path, examples, predictions, columns, scores=None, score_classes=None):
    fields = [
        "scene_name", "sample_token", "candidate_id", "kind",
        "instance_token", "annotation_token", "reference",
        "prediction", "correct",
    ] + columns

    if scores is not None:
        for class_name in score_classes:
            fields.append("score_" + str(class_name))

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for position, example in enumerate(examples):
            row = {
                "scene_name": example["scene_name"],
                "sample_token": example["sample_token"],
                "candidate_id": example["candidate_id"],
                "kind": example["kind"],
                "instance_token": example["instance_token"],
                "annotation_token": example["annotation_token"],
                "reference": example["label"],
                "prediction": predictions[position],
                "correct": example["label"] == predictions[position],
            }

            for feature_position, column in enumerate(columns):
                value = example["features"][feature_position]
                row[column] = "" if np.isnan(value) else float(value)

            if scores is not None:
                for class_position, class_name in enumerate(score_classes):
                    row["score_" + str(class_name)] = float(scores[position, class_position])

            writer.writerow(row)


def check_cv(result, columns):
    """Verify fold boundaries and training-only preprocessing."""
    examples = result["examples"]
    features = np.array([example["features"] for example in examples])
    visits = np.zeros(len(examples), dtype=int)

    for prediction in result["predictions"]:
        if prediction is None:
            raise ValueError("An out-of-fold prediction is missing.")

    for fold in result["folds"]:
        training_indices = fold["training_indices"]
        validation_indices = fold["validation_indices"]

        training_scenes = set(fold["training_summary"]["scene_names"])
        validation_scenes = set(fold["validation_summary"]["scene_names"])

        if not training_scenes.isdisjoint(validation_scenes):
            raise ValueError("A scene appears in both fold partitions.")

        if not set(training_indices).isdisjoint(set(validation_indices)):
            raise ValueError("An example appears in both fold partitions.")

        visits[validation_indices] += 1

        expected = np.nanmedian(features[training_indices], axis=0)
        expected[np.isnan(expected)] = 0

        if not np.allclose(expected, fold["imputation_values"]):
            raise ValueError("Fold imputation does not match training data.")

        if len(fold["imputation_values"]) != len(columns):
            raise ValueError("Incorrect feature schema.")

    if not (visits == 1).all():
        raise ValueError("Each example must receive exactly one held-out prediction.")


def save_group_diagnostics(folder, examples, reference, predictions):
    """Evaluate candidate kinds and scenes separately."""
    groups = {}

    for position, example in enumerate(examples):
        keys = [
            ("kind", example["kind"]),
            ("scene", example["scene_name"]),
        ]

        for key in keys:
            if key not in groups:
                groups[key] = []

            groups[key].append(position)

    summary_rows = []

    for key, positions in groups.items():
        group_type, group_name = key
        metrics = calculate_metrics(reference[positions], predictions[positions])
        group_folder = folder / f"{group_type}_{group_name}"

        save_metric_bundle(
            group_folder,
            f"Evaluation — {group_type}: {group_name}",
            metrics,
        )

        summary_rows.append({
            "group_type": group_type,
            "group_name": group_name,
            "candidate_rows": metrics["candidate_rows"],
            "accuracy_pct": metrics["accuracy_pct"],
            "macro_f1_pct": metrics["macro_f1_pct"],
            "minority_macro_recall_pct": metrics["minority_macro_recall_pct"],
        })

    save_csv(
        folder / "group_summary.csv",
        list(summary_rows[0]),
        summary_rows,
    )


def main():
    local_time = datetime.now(timezone(timedelta(hours=-3)))
    run_name = local_time.strftime("%Y%m%d_%H%M%S_%f")
    output_path = PROJECT_PATH / "src" / "radar" / "results" / run_name
    output_path.mkdir(parents=True, exist_ok=False)

    print("Results directory:", output_path)
    print("Selection priority: minority macro recall, then macro F1, then vehicle recall.")

    dataset_path = PROJECT_PATH / "data" / "downloads" / "v1.0-mini"
    dataset = load_dataset(dataset_path)
    splits = create_splits_scenes()

    development_scenes = splits["mini_train"]
    evaluation_scenes = splits["mini_val"]
    all_scenes = development_scenes + evaluation_scenes

    sample_tokens = get_scene_samples(
        dataset,
        all_scenes,
        max_samples_per_scene=None,
    )

    examples = []
    columns = None

    print("Extracting all samples:", len(sample_tokens), flush=True)

    for number, token in enumerate(sample_tokens, start=1):
        sample_examples, sample_columns = build_sample_examples(dataset, token)

        if columns is None:
            columns = sample_columns

        if sample_columns != columns:
            raise ValueError("Feature order changed between samples.")

        examples.extend(sample_examples)

        if number % 20 == 0 or number == len(sample_tokens):
            print(
                f"Processed {number}/{len(sample_tokens)} "
                f"| candidates: {len(examples)}",
                flush=True,
            )

    if columns is None:
        raise ValueError("No dataset samples were processed.")

    development_all = []
    development_labeled = []
    evaluation = []
    all_labeled = []

    for example in examples:
        if example["scene_name"] in development_scenes:
            development_all.append(example)

        if example["label"] is None:
            continue

        all_labeled.append(example)

        if example["scene_name"] in development_scenes:
            development_labeled.append(example)
        else:
            evaluation.append(example)

    development_features, development_labels = build_model_inputs(development_labeled)
    evaluation_features, evaluation_labels = build_model_inputs(evaluation)
    final_features, final_labels = build_model_inputs(all_labeled)

    if set(development_labels) != set(CLASSES):
        raise ValueError("Development data does not contain all four target classes.")

    dataset_csv = output_path / "dataset_candidates.csv"
    export_examples(dataset_csv, examples, columns)

    save_json(
        output_path / "dataset_summary.json",
        summarize_dataset(examples, columns),
    )

    # Select using development scenes only.
    # Reserved-scene results are not used to choose or adjust the model.
    selection_rows = []
    cv_results = {}
    best_name = None
    best_key = None
    reference_cv = None

    for model_name in MODEL_CANDIDATES:
        print("\nDevelopment cross-validation:", model_name, flush=True)

        result = cross_validate_model(
            development_all,
            use_instance_weights=False,
            n_splits=N_SPLITS,
            model_name=model_name,
        )

        check_cv(result, columns)

        if reference_cv is None:
            reference_cv = result
        else:
            if not np.array_equal(reference_cv["reference_labels"], result["reference_labels"]):
                raise ValueError("Model comparisons used different reference labels.")

            for first, second in zip(reference_cv["folds"], result["folds"]):
                if not np.array_equal(first["validation_indices"], second["validation_indices"]):
                    raise ValueError("Model comparisons used different folds.")

                if not np.array_equal(first["training_indices"], second["training_indices"]):
                    raise ValueError("Model comparisons used different training folds.")

        predictions = np.array(result["predictions"].tolist())
        metrics = calculate_metrics(result["reference_labels"], predictions)
        cv_folder = output_path / "development_cv" / model_name

        save_metric_bundle(
            cv_folder,
            f"Development out-of-fold — {model_name}",
            metrics,
        )

        export_predictions(
            cv_folder / "predictions.csv",
            result["examples"],
            predictions,
            columns,
        )

        # Preserve fold assignments, class availability and preprocessing.
        save_json(cv_folder / "folds.json", result["folds"])

        for fold in result["folds"]:
            positions = fold["validation_indices"]
            fold_metrics = calculate_metrics(
                result["reference_labels"][positions],
                predictions[positions],
            )

            save_metric_bundle(
                cv_folder / f"fold_{fold['fold']}",
                f"{model_name} — fold {fold['fold']}",
                fold_metrics,
            )

        vehicle_recall = 0

        for record in metrics["classes"]:
            if record["class"] == "vehicle":
                vehicle_recall = record["recall_pct"] or 0

        selection_key = (
            metrics["minority_macro_recall_pct"],
            metrics["macro_f1_pct"],
            vehicle_recall,
        )

        if best_key is None or selection_key > best_key:
            best_key = selection_key
            best_name = model_name

        selection_rows.append({
            "model": model_name,
            "accuracy_pct": metrics["accuracy_pct"],
            "macro_f1_pct": metrics["macro_f1_pct"],
            "macro_recall_pct": metrics["macro_recall_pct"],
            "minority_macro_recall_pct": metrics["minority_macro_recall_pct"],
            "vehicle_recall_pct": vehicle_recall,
        })

        cv_results[model_name] = metrics

    save_csv(
        output_path / "model_selection.csv",
        list(selection_rows[0]),
        selection_rows,
    )

    print("\nSelected model:", best_name)
    print("Selection used development predictions only.")
    print("Higher minority recall can still come with low precision.")

    # Fit the selected model on eight scenes and evaluate on the reserved two.
    evaluation_model = create_model(best_name, development_labels)
    evaluation_model = train_model(
        evaluation_model,
        development_features,
        development_labels,
    )

    prediction_result = predict_model(evaluation_model, evaluation_features)
    evaluation_predictions = prediction_result["predictions"]
    heldout_metrics = calculate_metrics(evaluation_labels, evaluation_predictions)
    evaluation_folder = output_path / "reserved_evaluation"

    save_metric_bundle(
        evaluation_folder,
        f"Reserved-scene evaluation — {best_name}",
        heldout_metrics,
    )

    export_predictions(
        evaluation_folder / "predictions.csv",
        evaluation,
        evaluation_predictions,
        columns,
        prediction_result["scores"],
        prediction_result["classes"],
    )

    save_group_diagnostics(
        evaluation_folder,
        evaluation,
        evaluation_labels,
        evaluation_predictions,
    )

    # A simple baseline chooses its class from development labels only.
    majority_class = Counter(development_labels).most_common(1)[0][0]
    baseline_predictions = np.full(len(evaluation_labels), majority_class)

    save_metric_bundle(
        output_path / "majority_reference",
        "Reserved evaluation — development majority-class reference",
        calculate_metrics(evaluation_labels, baseline_predictions),
    )

    joblib.dump(
        {
            "model": evaluation_model,
            "model_name": best_name,
            "feature_columns": columns,
            "classes": evaluation_model.classes_.tolist(),
            "training_scenes": development_scenes,
            "evaluation_scenes": evaluation_scenes,
        },
        output_path / "evaluation_model.joblib",
    )

    # Final deployment fit: all labeled candidates from all ten scenes.
    print("\nFitting the final model on all ten scenes...", flush=True)
    final_model = create_model(best_name, final_labels)
    final_model = train_model(final_model, final_features, final_labels)

    dataset_fingerprint = file_hash(dataset_csv)

    joblib.dump(
        {
            "model": final_model,
            "model_name": best_name,
            "feature_columns": columns,
            "classes": final_model.classes_.tolist(),
            "training_scenes": all_scenes,
            "training_examples": len(all_labeled),
            "dataset_sha256": dataset_fingerprint,
            "evaluation_note": (
                "Reserved metrics belong to evaluation_model.joblib. "
                "This final model was refitted on all ten scenes."
            ),
        },
        output_path / "final_model.joblib",
    )

    # Training diagnostics are explicitly in-sample, not a quality estimate.
    final_training_predictions = predict_model(final_model, final_features)["predictions"]

    save_metric_bundle(
        output_path / "final_training_in_sample",
        "Final model — training data only, not independent evaluation",
        calculate_metrics(final_labels, final_training_predictions),
    )

    importance_rows = []
    importances = final_model.named_steps["classifier"].feature_importances_

    for position, column in enumerate(columns):
        importance_rows.append({
            "feature": column,
            "impurity_importance": float(importances[position]),
            "importance_pct": 100 * float(importances[position]),
        })

    importance_rows.sort(
        key=lambda row: row["impurity_importance"],
        reverse=True,
    )

    save_csv(
        output_path / "feature_importance.csv",
        list(importance_rows[0]),
        importance_rows,
    )

    source_hashes = {}

    for path in sorted((PROJECT_PATH / "src" / "radar").glob("*.py")):
        source_hashes[path.name] = file_hash(path)

    package_versions = {}

    for package in [
        "numpy", "scipy", "scikit-learn", "imbalanced-learn",
        "nuscenes-devkit", "joblib", "matplotlib",
    ]:
        package_versions[package] = version(package)

    metadata = {
        "created_at": local_time.isoformat(),
        "selected_model": best_name,
        "selection_priority": [
            "minority macro recall",
            "macro F1",
            "vehicle recall",
        ],
        "selection_results": selection_rows,
        "processed_samples": len(sample_tokens),
        "total_candidates": len(examples),
        "development_labeled_examples": len(development_labeled),
        "reserved_labeled_examples": len(evaluation),
        "final_training_examples": len(all_labeled),
        "unlabeled_candidates": len(examples) - len(all_labeled),
        "development_scenes": development_scenes,
        "reserved_scenes": evaluation_scenes,
        "final_training_scenes": all_scenes,
        "feature_columns": columns,
        "final_classifier_parameters": final_model.named_steps["classifier"].get_params(),
        "final_imputation_values": final_model.named_steps["imputer"].statistics_,
        "dataset_csv_sha256": dataset_fingerprint,
        "source_hashes": source_hashes,
        "package_versions": package_versions,
        "instance_weighting": False,
    }

    save_json(output_path / "training_metadata.json", metadata)

    lines = [
        "FINAL RADAR TRAINING AND EVALUATION",
        "",
        f"Selected model: {best_name}",
        "Selection priority: minority macro recall, then macro F1, then vehicle recall.",
        "Selection used four scene-grouped folds in development scenes.",
        "Reserved evaluation did not participate in model selection.",
        "",
        f"Processed samples: {len(sample_tokens)}",
        f"Total candidates: {len(examples)}",
        f"Development labeled examples: {len(development_labeled)}",
        f"Reserved evaluation examples: {len(evaluation)}",
        f"Final training examples: {len(all_labeled)}",
        "",
        metric_text("RESERVED-SCENE EVALUATION", heldout_metrics),
        "",
        "IMPORTANT DIAGNOSTICS",
    ]

    for record in heldout_metrics["classes"]:
        instances = set()

        for example in evaluation:
            if example["label"] == record["class"]:
                instances.add(example["instance_token"])

        lines.append(
            f"{record['class']}: {record['reference_count']} reference rows "
            f"from {len(instances)} associated annotated instances."
        )

        if record["reference_count"] and len(instances) < 5:
            lines.append("Few physical instances: generalization evidence is limited.")

    # Report the largest confusion using exact counts.
    matrix = heldout_metrics["confusion_matrix"]
    confusions = []

    for row, reference_class in enumerate(CLASSES):
        for column, predicted_class in enumerate(CLASSES):
            if row != column and matrix[row, column] > 0:
                confusions.append((
                    int(matrix[row, column]),
                    reference_class,
                    predicted_class,
                    int(matrix[row].sum()),
                ))

    confusions.sort(reverse=True)

    lines.extend(["", "LARGEST CONFUSIONS"])

    for count, reference_class, predicted_class, support in confusions:
        lines.append(
            f"{reference_class} -> {predicted_class}: "
            f"{ratio_text(count, support)} of reference {reference_class} rows."
        )

    lines.extend([
        "",
        "LIMITATIONS AND MODEL SCOPE",
        "- Choosing by minority recall does not guarantee acceptable precision.",
        "- The selected model is best only under the stated development criterion.",
        "- The final model uses all ten scenes and has no independent evaluation here.",
        "- Reserved metrics belong to evaluation_model.joblib, trained on eight scenes.",
        "- Final training metrics are in-sample and must not be presented as test metrics.",
        "- Candidate rows can repeat the same object over time or within a frame.",
        "- Labels are provisional geometric associations, not confirmed return identities.",
        "- Classification metrics do not measure physical object detection recall.",
        "- Unlabeled candidates were exported but excluded from supervised training.",
        "- Scores are not calibrated confidence that an object exists.",
        "- Impurity feature importance is not causal evidence.",
        "- Reserved mini scenes were examined in the earlier LIVE project.",
        "- This is an internal MVP evaluation, not a new external validation.",
        "",
        "MAIN ARTIFACTS",
        "- final_model.joblib: selected model refitted on all labeled mini data.",
        "- evaluation_model.joblib: model responsible for reserved-scene metrics.",
        "- model_selection.csv: numerical comparison under the selection criterion.",
        "- dataset_candidates.csv: every candidate, features and source metadata.",
        "- dataset_summary.json: availability, categories and missing measurements.",
        "- development_cv/: pooled and per-fold numerical metrics and predictions.",
        "- reserved_evaluation/: exact metrics, predictions and scene/kind diagnostics.",
        "- final_training_in_sample/: explicitly labeled training diagnostics.",
        "- majority_reference/: a simple reference for contextualizing accuracy.",
        "- feature_importance.csv: numerical impurity-based feature importance.",
        "- training_metadata.json: settings, versions, feature order and fingerprints.",
    ])

    report = "\n".join(lines)
    (output_path / "report.txt").write_text(report, encoding="utf-8")

    print("\n" + report)
    print("\nCompleted. All artifacts:", output_path)
    print("Final model:", output_path / "final_model.joblib")


if __name__ == "__main__":
    main()