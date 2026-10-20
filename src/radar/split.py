import numpy as np


def split_by_scene(examples, validation_scene_names):
    """Separate labeled examples using complete scenes."""
    training = []
    validation = []

    for example in examples:
        if example["label"] is None:
            continue

        if example["scene_name"] in validation_scene_names:
            validation.append(example)
        else:
            training.append(example)

    if len(training) == 0:
        raise ValueError("No labeled training examples were selected.")

    if len(validation) == 0:
        raise ValueError("No labeled validation examples were selected.")

    return training, validation

def build_model_inputs(examples):
    """Build feature and target arrays from labeled examples."""
    if len(examples) == 0:
        raise ValueError("Cannot build model inputs from an empty selection.")

    feature_rows = []
    targets = []

    for example in examples:
        if example["label"] is None:
            raise ValueError("Model input selection contains an unlabeled example.")

        feature_rows.append(example["features"])
        targets.append(example["label"])

    features = np.array(feature_rows, dtype=float)
    labels = np.array(targets)

    return features, labels

def summarize_partition(examples):
    """Summarize class availability in a training or validation partition."""
    class_names = ["vehicle", "pedestrian", "two_wheeler", "obstacle"]
    class_records = {}
    scene_names = set()

    for class_name in class_names:
        class_records[class_name] = {
            "examples": 0,
            "isolated_examples": 0,
            "scenes": set(),
            "instances": set(),
        }

    for example in examples:
        if example["label"] is None:
            raise ValueError("Partition summary requires labeled examples.")

        scene_names.add(example["scene_name"])
        record = class_records[example["label"]]

        record["examples"] += 1
        record["scenes"].add(example["scene_name"])
        record["instances"].add(example["instance_token"])

        if example["kind"] == "isolated_return":
            record["isolated_examples"] += 1

    for record in class_records.values():
        record["scenes"] = len(record["scenes"])
        record["instances"] = len(record["instances"])

    result = {
        "total_examples": len(examples),
        "scene_names": sorted(scene_names),
        "classes": class_records,
    }

    return result