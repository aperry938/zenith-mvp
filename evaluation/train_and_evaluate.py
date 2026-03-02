"""
ZENith Ablation Study: Feature Representation × Classifier Evaluation.

Compares four feature representations across three classifiers for
10-class yoga pose classification:

Feature Conditions:
    RAW:          Raw flattened landmarks [x,y,z,vis] × 33 = 132 dims
    BIO:          Biomechanical features = 30 dims
    RAW+BIO:      Concatenated = 162 dims
    BIO+TEMPORAL: Bio features with 5-frame temporal window = 150 dims

Classifiers:
    Random Forest (RF), XGBoost (XGB), MLP (small neural network)

Evaluation:
    - Stratified 5-fold cross-validation
    - Per-fold accuracy, macro F1, weighted F1
    - Per-class precision, recall, F1
    - 95% confidence intervals
    - Confusion matrices
    - Latency benchmarks

Output:
    evaluation/results/ablation_results.json
    evaluation/results/confusion_matrices/
    evaluation/results/per_class_metrics.json
"""

import os
import sys
import json
import time
import warnings
import numpy as np
from collections import defaultdict

# Suppress sklearn warnings
warnings.filterwarnings('ignore')

# Add parent for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from biomechanical_features import (
    extract_biomechanical_features, extract_biomechanical_batch,
    StabilityTracker, FILE_PREFIX_TO_PROFILE, NUM_FEATURES
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Skipping XGB classifier.")


# ── Data Loading ────────────────────────────────────────────────────────────

def load_dataset(keypoints_dir):
    """
    Load all .npy keypoint files and extract features in all representations.

    Returns:
        raw_features: dict mapping filename to (N, 132) array
        bio_features: dict mapping filename to (N, 30) array
        labels: dict mapping filename to pose label string
        quality: dict mapping filename to 'correct' or 'incorrect'
    """
    raw_features = {}
    bio_features = {}
    labels = {}
    quality_labels = {}

    files = sorted(f for f in os.listdir(keypoints_dir) if f.endswith('.npy'))

    for fname in files:
        # Determine pose from filename
        prefix = None
        for known_prefix in FILE_PREFIX_TO_PROFILE:
            if fname.startswith(known_prefix + "_"):
                prefix = known_prefix
                break

        if prefix is None:
            continue

        pose_label = FILE_PREFIX_TO_PROFILE[prefix]
        path = os.path.join(keypoints_dir, fname)
        data = np.load(path)  # (N, 33, 4)

        # RAW features: flatten each frame to 132
        raw = data.reshape(data.shape[0], -1).astype(np.float32)  # (N, 132)

        # BIO features: extract biomechanical features
        tracker = StabilityTracker(buffer_size=15)
        bio = extract_biomechanical_batch(data, tracker).astype(np.float32)  # (N, 30)

        raw_features[fname] = raw
        bio_features[fname] = bio
        labels[fname] = pose_label
        quality_labels[fname] = "correct" if "Correct" in fname else "incorrect"

    return raw_features, bio_features, labels, quality_labels


def prepare_features(raw_features, bio_features, labels, condition, temporal_window=5):
    """
    Prepare feature matrices and label arrays for a given condition.

    Samples frames from the held-pose region (middle 50%) of each video
    to avoid entry/exit transitions.

    Args:
        condition: 'RAW', 'BIO', 'RAW+BIO', or 'BIO+TEMPORAL'
        temporal_window: Number of frames for temporal features

    Returns:
        X: (N_samples, D) feature matrix
        y: (N_samples,) label array (encoded)
        le: LabelEncoder
        video_ids: list of (filename, frame_idx) per sample
    """
    X_list = []
    y_list = []
    video_ids = []

    le = LabelEncoder()
    all_labels = sorted(set(labels.values()))
    le.fit(all_labels)

    for fname in sorted(raw_features.keys()):
        raw = raw_features[fname]
        bio = bio_features[fname]
        n_frames = raw.shape[0]

        # Sample from held-pose region (middle 50%)
        start = int(n_frames * 0.25)
        end = int(n_frames * 0.75)

        # Sample every 5th frame to reduce redundancy
        indices = list(range(start, end, 5))

        pose_label = labels[fname]
        encoded_label = le.transform([pose_label])[0]

        for idx in indices:
            if condition == 'RAW':
                feat = raw[idx]
            elif condition == 'BIO':
                feat = bio[idx]
            elif condition == 'RAW+BIO':
                feat = np.concatenate([raw[idx], bio[idx]])
            elif condition == 'BIO+TEMPORAL':
                # Temporal window: concatenate bio features from surrounding frames
                half_w = temporal_window // 2
                frame_indices = [
                    max(0, min(n_frames - 1, idx + offset))
                    for offset in range(-half_w, half_w + 1)
                ]
                feat = np.concatenate([bio[fi] for fi in frame_indices])
            else:
                raise ValueError(f"Unknown condition: {condition}")

            X_list.append(feat)
            y_list.append(encoded_label)
            video_ids.append((fname, idx))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    return X, y, le, video_ids


# ── Classifiers ─────────────────────────────────────────────────────────────

def get_classifiers():
    """Return dict of classifier name → (constructor, params)."""
    classifiers = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=300, random_state=42,
            early_stopping=True, validation_fraction=0.1
        ),
    }
    if HAS_XGBOOST:
        classifiers["XGBoost"] = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, use_label_encoder=False, eval_metric='mlogloss',
            verbosity=0
        )
    return classifiers


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_condition(X, y, le, clf_name, clf, n_folds=5):
    """
    Run stratified k-fold cross-validation for one condition × classifier.

    Returns dict with accuracy, F1, per-class metrics, confusion matrix.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_metrics = []
    all_y_true = []
    all_y_pred = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train
        clf_copy = type(clf)(**clf.get_params())
        clf_copy.fit(X_train, y_train)

        # Predict
        y_pred = clf_copy.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')

        fold_metrics.append({
            "fold": fold_idx + 1,
            "accuracy": round(acc, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
        })

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

    # Aggregate metrics
    accs = [m["accuracy"] for m in fold_metrics]
    f1s = [m["f1_macro"] for m in fold_metrics]
    f1ws = [m["f1_weighted"] for m in fold_metrics]

    # 95% CI (t-distribution approximation with small n)
    from scipy import stats
    def ci_95(values):
        n = len(values)
        mean = np.mean(values)
        se = np.std(values, ddof=1) / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, n - 1) if n > 1 else 2.0
        return (round(mean - t_crit * se, 4), round(mean + t_crit * se, 4))

    # Per-class metrics
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    prec, rec, f1, sup = precision_recall_fscore_support(
        all_y_true, all_y_pred, average=None, labels=range(len(le.classes_))
    )

    per_class = {}
    for i, cls_name in enumerate(le.classes_):
        per_class[cls_name] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(sup[i]),
        }

    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred, labels=range(len(le.classes_)))

    return {
        "classifier": clf_name,
        "accuracy_mean": round(np.mean(accs), 4),
        "accuracy_std": round(np.std(accs), 4),
        "accuracy_ci95": ci_95(accs),
        "f1_macro_mean": round(np.mean(f1s), 4),
        "f1_macro_std": round(np.std(f1s), 4),
        "f1_macro_ci95": ci_95(f1s),
        "f1_weighted_mean": round(np.mean(f1ws), 4),
        "f1_weighted_ci95": ci_95(f1ws),
        "per_fold": fold_metrics,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": list(le.classes_),
    }


def benchmark_latency(X, clf, n_iters=1000):
    """Measure single-sample inference latency."""
    sample = X[:1]

    # Warmup
    for _ in range(10):
        clf.predict(sample)

    # Benchmark
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        clf.predict(sample)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    return {
        "mean_ms": round(np.mean(times), 3),
        "std_ms": round(np.std(times), 3),
        "p50_ms": round(np.percentile(times, 50), 3),
        "p95_ms": round(np.percentile(times, 95), 3),
        "p99_ms": round(np.percentile(times, 99), 3),
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    keypoints_dir = os.path.join(os.path.dirname(__file__), '..', 'ZENith_Data', 'keypoints')
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'confusion_matrices'), exist_ok=True)

    print("ZENith Ablation Study")
    print("=" * 70)

    # 1. Load dataset
    print("\n[1/4] Loading dataset...")
    raw_features, bio_features, labels, quality_labels = load_dataset(keypoints_dir)
    print(f"  Loaded {len(raw_features)} videos across {len(set(labels.values()))} poses")

    # 2. Define conditions
    conditions = ['RAW', 'BIO', 'RAW+BIO', 'BIO+TEMPORAL']
    classifiers = get_classifiers()

    all_results = {}

    for cond in conditions:
        print(f"\n[2/4] Preparing features: {cond}...")
        X, y, le, video_ids = prepare_features(raw_features, bio_features, labels, cond)
        print(f"  X: {X.shape}, y: {y.shape} ({len(le.classes_)} classes)")

        cond_results = {"condition": cond, "n_samples": len(y), "n_features": X.shape[1]}
        classifier_results = {}

        for clf_name, clf in classifiers.items():
            print(f"  Evaluating {clf_name}...")
            result = evaluate_condition(X, y, le, clf_name, clf)
            print(f"    Accuracy: {result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f} "
                  f"(CI: {result['accuracy_ci95']})")
            print(f"    F1 macro: {result['f1_macro_mean']:.4f} ± {result['f1_macro_std']:.4f}")

            # Latency benchmark (train on full data first)
            clf_full = type(clf)(**clf.get_params())
            clf_full.fit(X, y)
            latency = benchmark_latency(X, clf_full, n_iters=500)
            result["latency"] = latency
            print(f"    Latency: {latency['mean_ms']:.2f}ms (p95: {latency['p95_ms']:.2f}ms)")

            classifier_results[clf_name] = result

            # Save confusion matrix
            cm_path = os.path.join(results_dir, 'confusion_matrices',
                                   f'cm_{cond}_{clf_name.replace(" ", "_")}.json')
            with open(cm_path, 'w') as f:
                json.dump({
                    "condition": cond,
                    "classifier": clf_name,
                    "matrix": result["confusion_matrix"],
                    "class_names": result["class_names"],
                }, f, indent=2)

        cond_results["classifiers"] = classifier_results
        all_results[cond] = cond_results

    # 3. Summary table
    print("\n" + "=" * 70)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 70)
    header = f"{'Condition':15s} {'Classifier':15s} {'Dims':>5s} {'Accuracy':>10s} {'F1 Macro':>10s} {'Latency':>10s}"
    print(header)
    print("-" * 70)

    for cond in conditions:
        cr = all_results[cond]
        for clf_name in classifiers:
            r = cr["classifiers"][clf_name]
            dims = cr["n_features"]
            acc = f"{r['accuracy_mean']:.4f}"
            f1 = f"{r['f1_macro_mean']:.4f}"
            lat = f"{r['latency']['mean_ms']:.2f}ms"
            print(f"{cond:15s} {clf_name:15s} {dims:>5d} {acc:>10s} {f1:>10s} {lat:>10s}")

    # 4. Save all results
    results_path = os.path.join(results_dir, 'ablation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # 5. Per-class metrics summary
    print("\n" + "=" * 70)
    print("BEST CONFIGURATION: Per-Class Metrics")
    print("=" * 70)

    # Find best configuration
    best_acc = 0
    best_key = None
    for cond in conditions:
        for clf_name in classifiers:
            r = all_results[cond]["classifiers"][clf_name]
            if r["accuracy_mean"] > best_acc:
                best_acc = r["accuracy_mean"]
                best_key = (cond, clf_name)

    if best_key:
        cond, clf_name = best_key
        r = all_results[cond]["classifiers"][clf_name]
        print(f"Best: {cond} × {clf_name} (Accuracy: {r['accuracy_mean']:.4f})")
        print(f"\n{'Class':25s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
        print("-" * 60)
        for cls_name, metrics in r["per_class"].items():
            print(f"{cls_name:25s} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
                  f"{metrics['f1']:>10.4f} {metrics['support']:>10d}")

    # Save per-class metrics
    per_class_path = os.path.join(results_dir, 'per_class_metrics.json')
    per_class_all = {}
    for cond in conditions:
        for clf_name in classifiers:
            key = f"{cond}_{clf_name}"
            per_class_all[key] = all_results[cond]["classifiers"][clf_name]["per_class"]
    with open(per_class_path, 'w') as f:
        json.dump(per_class_all, f, indent=2)

    print(f"\nPer-class metrics saved to: {per_class_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
