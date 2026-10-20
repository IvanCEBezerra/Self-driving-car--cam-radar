from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from collections import Counter


def calculate_moderate_class_weights(labels):
    """Use square-root inverse frequencies calculated from training labels."""
    if labels is None or len(labels) == 0:
        raise ValueError("Moderate weights require nonempty training labels.")

    counts = Counter(labels)
    largest_count = max(counts.values())
    weights = {}

    for class_name, count in counts.items():
        weights[class_name] = (largest_count / count) ** 0.5

    return weights


def create_model(model_name="random_forest", training_labels=None):
    """Create an imputation and classification pipeline."""
    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )

    if model_name == "decision_tree":
        classifier = DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        )

    elif model_name in ["random_forest", "moderate_random_forest"]:
        class_weights = "balanced"

        if model_name == "moderate_random_forest":
            class_weights = calculate_moderate_class_weights(training_labels)

        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=2,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1,
        )

    elif model_name == "balanced_random_forest":
        classifier = BalancedRandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=2,
            sampling_strategy="all",
            replacement=True,
            bootstrap=False,
            class_weight=None,
            random_state=42,
            n_jobs=-1,
        )

    else:
        raise ValueError(
            "Unknown model. Use decision_tree, random_forest "
            "balanced_random_forest or moderate_random_forest."
        )

    model = Pipeline([
        ("imputer", imputer),
        ("classifier", classifier),
    ])

    return model

def train_model(model, features, labels, sample_weights=None):
    """Fit the pipeline using training data and optional example weights."""
    if len(features) == 0:
        raise ValueError("Cannot train with an empty feature matrix.")

    if len(features) != len(labels):
        raise ValueError("Feature rows and labels must have the same length.")

    if sample_weights is not None:
        if len(sample_weights) != len(labels):
            raise ValueError("Each training example must have one weight.")

    model.fit(
        features,
        labels,
        classifier__sample_weight=sample_weights,
    )

    return model

def predict_model(model, features):
    """Predict classes and return their model scores."""
    predictions = model.predict(features)
    scores = model.predict_proba(features)

    result = {
        "predictions": predictions,
        "scores": scores,
        "classes": model.classes_,
    }

    return result

