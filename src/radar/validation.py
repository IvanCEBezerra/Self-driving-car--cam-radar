import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from src.radar.split import build_model_inputs, summarize_partition
from src.radar.model import create_model, train_model, predict_model


def calculate_instance_weights(examples):
    """Give each annotated instance equal total example weight."""
    if len(examples) == 0:
        raise ValueError("Cannot calculate weights for an empty selection.")

    instance_counts = {}

    for example in examples:
        instance_token = example["instance_token"]

        if instance_token is None:
            raise ValueError("Instance weighting requires an annotated instance.")

        previous_count = instance_counts.get(instance_token, 0)
        instance_counts[instance_token] = previous_count + 1

    weights = []

    for example in examples:
        count = instance_counts[example["instance_token"]]
        weights.append(1 / count)

    weights = np.array(weights, dtype=float)
    weights = weights / weights.mean()

    return weights

def create_scene_folds(examples, n_splits=4):
    """Create approximately stratified folds without splitting scenes."""
    features, labels = build_model_inputs(examples)
    scene_tokens = []

    for example in examples:
        scene_tokens.append(example["scene_token"])

    groups = np.array(scene_tokens)

    if n_splits < 2:
        raise ValueError("Cross-validation requires at least two folds.")

    if len(set(scene_tokens)) < n_splits:
        raise ValueError("There are fewer scenes than requested folds.")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    folds = list(splitter.split(features, labels, groups))
    return folds

def cross_validate_model(examples, use_instance_weights=False, n_splits=4, model_name="random_forest"):
    """Generate out-of-fold predictions with preprocessing fitted per fold."""
    labeled_examples = []

    for example in examples:
        if example["label"] is not None:
            labeled_examples.append(example)

    features, labels = build_model_inputs(labeled_examples)
    folds = create_scene_folds(labeled_examples, n_splits)
    predictions = np.full(len(labels), None, dtype=object)
    fold_records = []

    for fold_number, indices in enumerate(folds):
        training_indices, validation_indices = indices
        training_examples = []
        validation_examples = []

        for position in training_indices:
            training_examples.append(labeled_examples[position])

        for position in validation_indices:
            validation_examples.append(labeled_examples[position])

        sample_weights = None

        if use_instance_weights:
            sample_weights = calculate_instance_weights(training_examples)

        model = create_model(model_name, labels[training_indices])

        model = train_model(
            model,
            features[training_indices],
            labels[training_indices],
            sample_weights,
        )

        result = predict_model(
            model,
            features[validation_indices],
        )

        predictions[validation_indices] = result["predictions"]

        record = {
            "fold": fold_number + 1,
            "training_indices": training_indices,
            "validation_indices": validation_indices,
            "training_summary": summarize_partition(training_examples),
            "validation_summary": summarize_partition(validation_examples),
            "training_weights": sample_weights,
            "imputation_values": model.named_steps["imputer"].statistics_.copy(),
            "class_weights": model.named_steps["classifier"].class_weight,
        }

        fold_records.append(record)

        print(
            f"Completed fold {fold_number + 1}/{n_splits} "
            f"| instance weighting: {use_instance_weights}",
            flush=True,
        )

    result = {
        "examples": labeled_examples,
        "reference_labels": labels,
        "predictions": predictions,
        "folds": fold_records,
        "use_instance_weights": use_instance_weights,
        "model_name": model_name,
    }

    return result
