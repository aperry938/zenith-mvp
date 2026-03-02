"""
ZENith Quality Score Validation.

Validates the biomechanical quality scoring against ground truth labels
(correct/incorrect) from the annotated dataset.

Analyses:
1. VAE reconstruction error correlation with quality ratings
2. Biomechanical quality score discrimination (correct vs incorrect)
3. Spearman rank correlation between quality score and expert ratings
4. ROC-AUC for binary quality classification
5. Per-pose quality discrimination analysis
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from biomechanical_features import (
    extract_biomechanical_batch, compute_pose_quality_score,
    StabilityTracker, FILE_PREFIX_TO_PROFILE, NUM_FEATURES
)

from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve


def load_annotations(annotations_dir):
    """Load all annotation JSON files."""
    annotations = {}
    for fname in sorted(os.listdir(annotations_dir)):
        if not fname.endswith('.json') or fname.startswith('_'):
            continue
        path = os.path.join(annotations_dir, fname)
        with open(path) as f:
            annotations[fname.replace('.json', '.npy')] = json.load(f)
    return annotations


def compute_quality_scores(keypoints_dir, annotations):
    """Compute biomechanical quality scores for all videos."""
    results = []

    for fname, ann in sorted(annotations.items()):
        npy_path = os.path.join(keypoints_dir, fname)
        if not os.path.exists(npy_path):
            continue

        data = np.load(npy_path)
        n = data.shape[0]

        # Extract bio features for held-pose region
        tracker = StabilityTracker()
        bio = extract_biomechanical_batch(data, tracker)
        start = int(n * 0.25)
        end = int(n * 0.75)
        hold_feats = bio[start:end]

        pose_label = ann["pose_label"]
        scores = [compute_pose_quality_score(hold_feats[i], pose_label)
                  for i in range(len(hold_feats))]

        results.append({
            "filename": fname,
            "pose": pose_label,
            "ground_truth": ann["ground_truth_label"],
            "annotation_rating": ann["quality_rating"],
            "annotation_score": ann["quality_score"],
            "bio_quality_mean": float(np.mean(scores)),
            "bio_quality_std": float(np.std(scores)),
            "bio_quality_median": float(np.median(scores)),
        })

    return results


def analyze_discrimination(results):
    """Analyze quality score discrimination between correct and incorrect."""
    correct_scores = [r["bio_quality_mean"] for r in results if r["ground_truth"] == "correct"]
    incorrect_scores = [r["bio_quality_mean"] for r in results if r["ground_truth"] == "incorrect"]

    # Cohen's d effect size
    pooled_std = np.sqrt((np.std(correct_scores)**2 + np.std(incorrect_scores)**2) / 2)
    cohens_d = (np.mean(correct_scores) - np.mean(incorrect_scores)) / (pooled_std + 1e-8)

    # Mann-Whitney U test (non-parametric)
    u_stat, p_value = stats.mannwhitneyu(correct_scores, incorrect_scores, alternative='greater')

    # ROC-AUC
    y_true = [1 if r["ground_truth"] == "correct" else 0 for r in results]
    y_scores = [r["bio_quality_mean"] for r in results]
    auc = roc_auc_score(y_true, y_scores)

    # Optimal threshold (Youden's J)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    return {
        "correct_mean": round(np.mean(correct_scores), 1),
        "correct_std": round(np.std(correct_scores), 1),
        "incorrect_mean": round(np.mean(incorrect_scores), 1),
        "incorrect_std": round(np.std(incorrect_scores), 1),
        "cohens_d": round(cohens_d, 3),
        "mann_whitney_u": round(float(u_stat), 1),
        "p_value": float(p_value),
        "roc_auc": round(auc, 4),
        "optimal_threshold": round(float(optimal_threshold), 1),
        "n_correct": len(correct_scores),
        "n_incorrect": len(incorrect_scores),
    }


def analyze_per_pose(results):
    """Per-pose quality discrimination analysis."""
    poses = defaultdict(lambda: {"correct": [], "incorrect": []})
    for r in results:
        poses[r["pose"]][r["ground_truth"]].append(r["bio_quality_mean"])

    per_pose = {}
    for pose, groups in sorted(poses.items()):
        c = groups["correct"]
        i = groups["incorrect"]
        if not c or not i:
            continue

        pooled_std = np.sqrt((np.std(c)**2 + np.std(i)**2) / 2)
        d = (np.mean(c) - np.mean(i)) / (pooled_std + 1e-8)

        try:
            u, p = stats.mannwhitneyu(c, i, alternative='greater')
        except ValueError:
            u, p = 0.0, 1.0

        per_pose[pose] = {
            "correct_mean": round(np.mean(c), 1),
            "correct_std": round(np.std(c), 1),
            "incorrect_mean": round(np.mean(i), 1),
            "incorrect_std": round(np.std(i), 1),
            "delta": round(np.mean(c) - np.mean(i), 1),
            "cohens_d": round(d, 3),
            "p_value": round(float(p), 4),
            "n_correct": len(c),
            "n_incorrect": len(i),
        }

    return per_pose


def analyze_rating_correlation(results):
    """Correlation between biomechanical quality score and annotation rating."""
    scores = [r["bio_quality_mean"] for r in results]
    ratings = [r["annotation_rating"] for r in results]

    # Spearman rank correlation (ordinal data)
    rho, p_spearman = stats.spearmanr(scores, ratings)

    # Pearson correlation (linear)
    r_pearson, p_pearson = stats.pearsonr(scores, ratings)

    return {
        "spearman_rho": round(rho, 4),
        "spearman_p": float(p_spearman),
        "pearson_r": round(r_pearson, 4),
        "pearson_p": float(p_pearson),
        "n": len(scores),
    }


def main():
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    keypoints_dir = os.path.join(base_dir, 'ZENith_Data', 'keypoints')
    annotations_dir = os.path.join(base_dir, 'ZENith_Data', 'annotations')
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    print("ZENith Quality Score Validation")
    print("=" * 60)

    # 1. Load annotations
    print("\n[1/4] Loading annotations...")
    annotations = load_annotations(annotations_dir)
    print(f"  Loaded {len(annotations)} annotations")

    # 2. Compute quality scores
    print("\n[2/4] Computing biomechanical quality scores...")
    results = compute_quality_scores(keypoints_dir, annotations)
    print(f"  Processed {len(results)} videos")

    # 3. Overall discrimination
    print("\n[3/4] Analyzing quality discrimination...")
    discrimination = analyze_discrimination(results)
    print(f"  Correct:   {discrimination['correct_mean']:.1f} ± {discrimination['correct_std']:.1f}")
    print(f"  Incorrect: {discrimination['incorrect_mean']:.1f} ± {discrimination['incorrect_std']:.1f}")
    print(f"  Cohen's d: {discrimination['cohens_d']:.3f}")
    print(f"  ROC-AUC:   {discrimination['roc_auc']:.4f}")
    print(f"  p-value:   {discrimination['p_value']:.6f}")

    # 4. Per-pose analysis
    print("\n[4/4] Per-pose analysis...")
    per_pose = analyze_per_pose(results)
    print(f"\n{'Pose':25s} {'Correct':>10s} {'Incorrect':>10s} {'Delta':>8s} {'d':>8s} {'p':>10s}")
    print("-" * 75)
    for pose, m in sorted(per_pose.items(), key=lambda x: -x[1]["delta"]):
        print(f"{pose:25s} {m['correct_mean']:>8.1f}   {m['incorrect_mean']:>8.1f}   "
              f"{m['delta']:>+6.1f}   {m['cohens_d']:>6.3f}   {m['p_value']:>8.4f}")

    # 5. Rating correlation
    correlation = analyze_rating_correlation(results)
    print(f"\nRating Correlation:")
    print(f"  Spearman rho: {correlation['spearman_rho']:.4f} (p={correlation['spearman_p']:.6f})")
    print(f"  Pearson r:    {correlation['pearson_r']:.4f} (p={correlation['pearson_p']:.6f})")

    # Save results
    quality_results = {
        "overall_discrimination": discrimination,
        "per_pose": per_pose,
        "rating_correlation": correlation,
        "individual_results": results,
    }

    out_path = os.path.join(results_dir, 'quality_validation.json')
    with open(out_path, 'w') as f:
        json.dump(quality_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
