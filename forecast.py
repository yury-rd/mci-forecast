"""Nested PLS/ridge prediction with participant-level cross-validation."""

import numpy as np
from sklearn.cross_decomposition import PLSRegression

REPRESENTATIONS = ("clinical", "static", "dispersion", "combined")
COMPONENTS = (2, 5, 10)
ALPHAS = (0.1, 1.0, 10.0, 100.0)


def synthetic_data():
    """Return the reproducible 48-row example, including constant columns."""
    rng = np.random.Generator(np.random.PCG64(732401))
    static = rng.normal(size=(48, 21))
    noise = rng.normal(size=(48, 21))
    dispersion = 0.3 * static + 0.7 * noise
    static[:, 20] = 2.0
    dispersion[:, 20] = -1.0
    clinical = np.column_stack([rng.normal(size=(48, 2)), np.ones(48)])
    categories = np.array(
        [["A" if i % 2 == 0 else "B", ["x", "y", "z"][i % 3]]
         for i in range(48)], dtype="<U12"
    )
    categories[0, 1] = "only-test"
    target = (0.3 * clinical[:, 0] - 0.2 * clinical[:, 1]
              + 0.7 * static[:, 0] - 0.4 * dispersion[:, 1]
              + 0.25 * static[:, 2] ** 2 + 0.4 * rng.normal(size=48))
    return {
        "static": static,
        "dispersion": dispersion,
        "combined": np.column_stack([static, dispersion]),
        "clinical_numeric": clinical,
        "categories": categories,
        "target": target,
    }


def validate_data(data):
    """Check input arrays and construct the combined 42-coordinate block."""
    required = {"target", "static", "dispersion", "clinical_numeric", "categories"}
    missing = required - data.keys()
    if missing:
        raise ValueError("Missing input arrays: " + ", ".join(sorted(missing)))
    result = dict(data)
    for key in required - {"categories"}:
        result[key] = np.asarray(data[key], dtype=float)
        if not np.isfinite(result[key]).all():
            raise ValueError(f"{key} must contain finite numbers")
    target = result["target"]
    if target.ndim != 1 or len(target) < 2:
        raise ValueError("target must be a vector with at least two participants")
    count = len(target)
    for key in ("static", "dispersion"):
        if result[key].shape != (count, 21):
            raise ValueError(f"{key} must have shape ({count}, 21)")
    clinical = result["clinical_numeric"]
    if clinical.ndim != 2 or len(clinical) != count:
        raise ValueError("clinical_numeric must have one matrix row per participant")
    categories = np.asarray(data["categories"])
    if categories.ndim != 2 or len(categories) != count or categories.dtype.kind != "U":
        raise ValueError("categories must be a Unicode matrix with one row per participant")
    result["categories"] = categories
    if "participant_ids" in data:
        ids = np.asarray(data["participant_ids"])
        if ids.shape != (count,) or len(set(ids.tolist())) != count:
            raise ValueError("participant_ids must be a vector of unique identifiers")
    result["combined"] = np.column_stack([result["static"], result["dispersion"]])
    return result


def validate_splits(splits, count):
    """Require disjoint folds and one held-out prediction per row and repeat."""
    if not isinstance(splits, list) or not splits:
        raise ValueError("The split plan must be a nonempty list")
    all_rows = set(range(count))
    tests_by_seed = {}
    fold_keys = set()

    def row_set(rows):
        if (not isinstance(rows, list) or not rows
                or any(type(row) is not int or not 0 <= row < count for row in rows)
                or len(set(rows)) != len(rows)):
            raise ValueError("Fold rows must be nonempty lists of distinct integer indices")
        return set(rows)

    def partition(fold, expected):
        if not isinstance(fold, dict) or not {"train", "test"} <= fold.keys():
            raise ValueError("Each fold needs train and test indices")
        train, test = row_set(fold["train"]), row_set(fold["test"])
        if train & test or train | test != expected:
            raise ValueError("Training and test rows must partition their parent fold")
        return train

    for split in splits:
        if not isinstance(split, dict) or not {"seed", "fold", "inner"} <= split.keys():
            raise ValueError("Each outer fold needs seed, fold, and inner entries")
        if type(split["seed"]) is not int or type(split["fold"]) is not int:
            raise ValueError("seed and fold must be integers")
        key = (split["seed"], split["fold"])
        if key in fold_keys:
            raise ValueError("Each seed/fold pair must be unique")
        fold_keys.add(key)
        train = partition(split, all_rows)
        tests_by_seed.setdefault(split["seed"], []).extend(split["test"])
        if not isinstance(split["inner"], list) or not split["inner"]:
            raise ValueError("Every outer fold needs inner folds")
        inner_tests = []
        for inner in split["inner"]:
            partition(inner, train)
            inner_tests.extend(inner["test"])
        if sorted(inner_tests) != sorted(train):
            raise ValueError("Inner test folds must cover outer training rows exactly once")
    if any(sorted(rows) != list(range(count)) for rows in tests_by_seed.values()):
        raise ValueError("Outer test folds must cover every participant once per repeat")


def _standardize(train):
    mean = train.mean(axis=0)
    sd = train.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (train - mean) / sd, mean, sd


def _ridge(train, target, test, alpha):
    scaled, mean, sd = _standardize(train)
    target_mean = target.mean()
    coefficients = np.linalg.solve(
        scaled.T @ scaled + alpha * np.eye(scaled.shape[1]),
        scaled.T @ (target - target_mean),
    )
    prediction = ((test - mean) / sd) @ coefficients + target_mean
    return prediction, (mean, sd, coefficients, np.array([target_mean]))


def _covariates(numeric, categories, train, test):
    train_blocks, test_blocks, names = [numeric[train]], [numeric[test]], []
    for column in range(categories.shape[1]):
        for category in sorted(set(categories[train, column])):
            train_blocks.append((categories[train, column] == category).astype(float)[:, None])
            test_blocks.append((categories[test, column] == category).astype(float)[:, None])
            names.append(f"{column}:{category}")
    return np.concatenate(train_blocks, axis=1), np.concatenate(test_blocks, axis=1), names


def _project(features, train, test, target):
    scaled, mean, sd = _standardize(features[train])
    block_weight = np.sqrt(features.shape[1])
    scaled = scaled / block_weight
    held_out = (features[test] - mean) / sd / block_weight
    components = min(10, len(train) - 1, scaled.shape[1])
    singular_values = np.linalg.svd(scaled - scaled.mean(axis=0), compute_uv=False)
    if (not len(singular_values) or singular_values[0] == 0
            or np.count_nonzero(singular_values > singular_values[0] * 1e-10) < components):
        raise ValueError("Training MRI features have insufficient rank for the PLS model")
    model = PLSRegression(n_components=components, scale=False, max_iter=1000, tol=1e-8)
    model.fit(scaled, target[train])
    scores = model.transform(scaled)
    return {
        "train_scores": scores,
        "test_scores": model.transform(held_out),
        "parameters": {
            "MRI_mean": mean,
            "MRI_sd": sd,
            "projection_mean": model._x_mean,
            "projection_matrix": model.x_rotations_.T,
            "projection_variance": scores.var(axis=0),
        },
    }


def _design(train_covariates, test_covariates, projection, components):
    if projection is None:
        return train_covariates, test_covariates
    return (
        np.column_stack([train_covariates, projection["train_scores"][:, :components]]),
        np.column_stack([test_covariates, projection["test_scores"][:, :components]]),
    )


def fit_model(data, split, representation):
    """Select settings on inner folds, then predict the outer test participants."""
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation: {representation}")
    target = data["target"]
    numeric, categories = data["clinical_numeric"], data["categories"]
    features = None if representation == "clinical" else data[representation]
    cache, transforms = [], []
    for inner in split["inner"]:
        train, test = np.array(inner["train"]), np.array(inner["test"])
        train_covariates, test_covariates, names = _covariates(numeric, categories, train, test)
        projection = None if features is None else _project(features, train, test, target)
        cache.append((train, test, train_covariates, test_covariates, projection))
        transforms.append({"categories": names, "MRI": projection})

    candidates = []
    for components in ((0,) if features is None else COMPONENTS):
        for alpha in ALPHAS:
            losses, inner_fits = [], []
            for train, test, a, b, projection in cache:
                train_design, test_design = _design(a, b, projection, components)
                prediction, parameters = _ridge(train_design, target[train], test_design, alpha)
                losses.extend(abs(prediction - target[test]))
                inner_fits.append({"prediction": prediction, "ridge_parameters": parameters})
            candidates.append({"loss": float(np.mean(losses)), "k": components,
                               "alpha": alpha, "inner": inner_fits})

    best = min(candidates, key=lambda candidate: (candidate["loss"], candidate["k"], candidate["alpha"]))
    train, test = np.array(split["train"]), np.array(split["test"])
    a, b, names = _covariates(numeric, categories, train, test)
    projection = None if features is None else _project(features, train, test, target)
    train_design, test_design = _design(a, b, projection, best["k"])
    prediction, parameters = _ridge(train_design, target[train], test_design, best["alpha"])
    return {
        "inner_transforms": transforms,
        "candidates": candidates,
        "selected": {"k": best["k"], "alpha": best["alpha"]},
        "outer_transform": {"categories": names, "MRI": projection},
        "outer_prediction": prediction,
        "outer_ridge_parameters": parameters,
    }


def evaluate(data, splits):
    """Compare all four representations using the same validated split plan."""
    data = validate_data(data)
    count = len(data["target"])
    validate_splits(splits, count)
    seeds = list(dict.fromkeys(split["seed"] for split in splits))
    results = {}
    for representation in REPRESENTATIONS:
        predictions = np.full((len(seeds), count), np.nan)
        choices = []
        for split in splits:
            fitted = fit_model(data, split, representation)
            predictions[seeds.index(split["seed"]), split["test"]] = fitted["outer_prediction"]
            choices.append({"seed": split["seed"], "fold": split["fold"], **fitted["selected"]})
        if not np.isfinite(predictions).all():
            raise ValueError("The model did not produce a finite prediction for every participant")
        errors = np.abs(predictions - data["target"])
        results[representation] = {
            "MAE_by_repeat": errors.mean(axis=1).tolist(),
            "mean_MAE": float(errors.mean()),
            "choices": choices,
        }
    return {"participants": count, "outer_repeats": len(seeds), "results": results}
