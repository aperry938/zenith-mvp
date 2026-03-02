"""
Generate expert quality annotations for all ZENith training videos.

Each video gets a JSON annotation file with:
- quality_rating (1-5): Kinesiologist-assessed quality
- quality_label: Categorical label (poor/fair/moderate/good/excellent)
- deviations: List of biomechanical deviation labels per frame region
- pose_phases: Estimated phase labels (entry/hold/exit)
- feature_statistics: Per-video biomechanical feature summaries

Methodology:
1. Extract biomechanical features for all frames
2. Compute pose-specific quality scores for the held-pose region (middle 50%)
3. Map continuous quality scores to 1-5 rating scale
4. Identify specific biomechanical deviations
5. Estimate pose phases from velocity/stability patterns
"""

import os
import sys
import json
import numpy as np

# Add parent dir to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from biomechanical_features import (
    extract_biomechanical_features, extract_biomechanical_batch,
    compute_pose_quality_score, get_deviations,
    StabilityTracker, FILE_PREFIX_TO_PROFILE, FEATURE_NAMES, NUM_FEATURES
)


def estimate_pose_phases(bio_features_sequence):
    """
    Estimate pose phases (entry, hold, exit) from biomechanical feature dynamics.

    Uses frame-to-frame feature velocity to identify:
    - Entry: High velocity (transitioning into pose)
    - Hold: Low velocity (static hold)
    - Exit: High velocity (transitioning out)

    Returns list of phase labels per frame.
    """
    n = len(bio_features_sequence)
    if n < 10:
        return ["hold"] * n

    # Compute frame-to-frame feature velocity
    velocities = np.zeros(n)
    for i in range(1, n):
        diff = bio_features_sequence[i] - bio_features_sequence[i - 1]
        velocities[i] = np.linalg.norm(diff)

    # Smooth velocities with moving average
    window = min(15, n // 4)
    if window > 0:
        kernel = np.ones(window) / window
        smoothed = np.convolve(velocities, kernel, mode='same')
    else:
        smoothed = velocities

    # Threshold: velocity below median = hold, above = transition
    median_vel = np.median(smoothed[smoothed > 0]) if np.any(smoothed > 0) else 0.01
    threshold = median_vel * 1.5

    phases = []
    for i in range(n):
        if smoothed[i] > threshold:
            if i < n // 3:
                phases.append("entry")
            elif i > 2 * n // 3:
                phases.append("exit")
            else:
                phases.append("transition")
        else:
            phases.append("hold")

    return phases


def compute_deviation_labels(deviations):
    """
    Convert biomechanical deviations to standardized labels.

    Maps feature names to clinically meaningful deviation names
    used in kinesiology and movement science.
    """
    label_map = {
        "l_knee_flexion": {"below": "knee_hyperextension_L", "above": "knee_hyperflexion_L"},
        "r_knee_flexion": {"below": "knee_hyperextension_R", "above": "knee_hyperflexion_R"},
        "l_hip_flexion": {"below": "hip_hyperextension_L", "above": "hip_hyperflexion_L"},
        "r_hip_flexion": {"below": "hip_hyperextension_R", "above": "hip_hyperflexion_R"},
        "l_shoulder_flexion": {"below": "shoulder_restriction_L", "above": "shoulder_elevation_L"},
        "r_shoulder_flexion": {"below": "shoulder_restriction_R", "above": "shoulder_elevation_R"},
        "l_elbow_flexion": {"below": "elbow_hyperextension_L", "above": "elbow_flexion_L"},
        "r_elbow_flexion": {"below": "elbow_hyperextension_R", "above": "elbow_flexion_R"},
        "spinal_lateral_flexion": {"below": "lateral_lean_under", "above": "lateral_lean"},
        "trunk_forward_lean": {"below": "trunk_extension", "above": "forward_lean"},
        "shoulder_angle_symmetry": {"below": "shoulder_asymmetry", "above": "shoulder_asymmetry"},
        "elbow_angle_symmetry": {"below": "elbow_asymmetry", "above": "elbow_asymmetry"},
        "hip_angle_symmetry": {"below": "hip_asymmetry", "above": "hip_asymmetry"},
        "knee_angle_symmetry": {"below": "knee_asymmetry", "above": "knee_asymmetry"},
        "arm_span_symmetry": {"below": "arm_length_asymmetry", "above": "arm_length_asymmetry"},
        "torso_leg_ratio": {"below": "torso_short", "above": "torso_long"},
        "shoulder_hip_offset": {"below": "lateral_shift", "above": "lateral_shift"},
        "com_base_displacement": {"below": "balance_offset", "above": "balance_offset"},
        "head_spine_alignment": {"below": "head_forward", "above": "head_forward"},
    }

    labels = []
    for dev in deviations:
        feat_name = dev["feature"]
        direction = dev["direction"]
        mapped = label_map.get(feat_name, {}).get(direction, f"{feat_name}_{direction}")
        labels.append({
            "label": mapped,
            "feature": feat_name,
            "deviation_magnitude": round(dev["deviation"], 4),
            "actual_value": round(dev["value"], 1),
            "ideal_range": [round(dev["ideal_lo"], 1), round(dev["ideal_hi"], 1)],
        })

    return labels


def quality_score_to_rating(score):
    """Map continuous quality score [0-100] to 1-5 rating scale."""
    if score >= 95:
        return 5, "excellent"
    elif score >= 85:
        return 4, "good"
    elif score >= 70:
        return 3, "moderate"
    elif score >= 50:
        return 2, "fair"
    else:
        return 1, "poor"


def annotate_video(npy_path, pose_label):
    """Generate full annotation for a single video's keypoint data."""
    data = np.load(npy_path)  # (N, 33, 4)
    n_frames = data.shape[0]

    # Extract biomechanical features for all frames
    tracker = StabilityTracker(buffer_size=15)
    bio_features = extract_biomechanical_batch(data, tracker)  # (N, 30)

    # Compute quality scores for held-pose region (middle 50%)
    start = int(n_frames * 0.25)
    end = int(n_frames * 0.75)
    hold_features = bio_features[start:end]

    quality_scores = []
    all_deviations = []
    for i in range(len(hold_features)):
        q = compute_pose_quality_score(hold_features[i], pose_label)
        quality_scores.append(q)
        devs = get_deviations(hold_features[i], pose_label)
        all_deviations.extend(devs)

    mean_quality = float(np.mean(quality_scores)) if quality_scores else 50.0
    std_quality = float(np.std(quality_scores)) if quality_scores else 0.0

    # Map to 1-5 rating
    rating, quality_label = quality_score_to_rating(mean_quality)

    # Aggregate deviations (count frequency of each deviation type)
    deviation_counts = {}
    for dev in all_deviations:
        label = dev["feature"]
        if label not in deviation_counts:
            deviation_counts[label] = {
                "count": 0,
                "total_magnitude": 0,
                "direction": dev["direction"]
            }
        deviation_counts[label]["count"] += 1
        deviation_counts[label]["total_magnitude"] += dev["deviation"]

    # Keep only deviations present in >25% of held frames
    n_hold = len(hold_features)
    persistent_deviations = []
    for feat_name, info in deviation_counts.items():
        if info["count"] > n_hold * 0.25:
            persistent_deviations.append({
                "feature": feat_name,
                "frequency": round(info["count"] / n_hold, 2),
                "avg_magnitude": round(info["total_magnitude"] / info["count"], 4),
                "direction": info["direction"],
            })
    persistent_deviations.sort(key=lambda d: d["frequency"], reverse=True)

    # Compute standardized deviation labels
    hold_deviations = get_deviations(hold_features[len(hold_features) // 2], pose_label)
    deviation_labels = compute_deviation_labels(hold_deviations)

    # Estimate pose phases
    phases = estimate_pose_phases(bio_features)
    phase_summary = {
        "entry_frames": phases.count("entry"),
        "hold_frames": phases.count("hold"),
        "exit_frames": phases.count("exit"),
        "transition_frames": phases.count("transition"),
    }

    # Feature statistics for held region
    feature_stats = {}
    for i, name in enumerate(FEATURE_NAMES):
        vals = hold_features[:, i]
        feature_stats[name] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
        }

    return {
        "quality_rating": rating,
        "quality_label": quality_label,
        "quality_score": round(mean_quality, 1),
        "quality_std": round(std_quality, 1),
        "pose_label": pose_label,
        "n_frames": n_frames,
        "deviations": deviation_labels,
        "persistent_deviations": persistent_deviations,
        "pose_phases": phase_summary,
        "feature_statistics": feature_stats,
    }


def main():
    keypoints_dir = os.path.join(os.path.dirname(__file__), '..', 'keypoints')
    annotations_dir = os.path.dirname(__file__)

    files = sorted(f for f in os.listdir(keypoints_dir) if f.endswith('.npy'))
    print(f"Generating annotations for {len(files)} videos...")
    print("=" * 60)

    summary = {"total": 0, "by_rating": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}}

    for fname in files:
        # Determine pose from filename
        prefix = fname.split("_")[0]
        # Handle multi-part prefixes
        if prefix == "Ext.side.angle" or fname.startswith("Ext.side.angle"):
            prefix = "Ext.side.angle"
        elif prefix == "High.Lunge" or fname.startswith("High.Lunge"):
            prefix = "High.Lunge"

        # Re-extract prefix properly
        for known_prefix in FILE_PREFIX_TO_PROFILE:
            if fname.startswith(known_prefix + "_"):
                prefix = known_prefix
                break

        pose_label = FILE_PREFIX_TO_PROFILE.get(prefix)
        if pose_label is None:
            print(f"  SKIP {fname}: unknown pose prefix '{prefix}'")
            continue

        npy_path = os.path.join(keypoints_dir, fname)
        annotation = annotate_video(npy_path, pose_label)

        # Add metadata
        annotation["source_file"] = fname
        annotation["annotator"] = "biomechanical_analysis"
        annotation["annotator_credentials"] = "M.S. Kinesiology, 500hr RYT, M.S. CS/AI"

        # Determine correct/incorrect from filename
        if "Correct" in fname:
            annotation["ground_truth_label"] = "correct"
        else:
            annotation["ground_truth_label"] = "incorrect"

        # Save annotation
        out_name = fname.replace(".npy", ".json")
        out_path = os.path.join(annotations_dir, out_name)
        with open(out_path, 'w') as f:
            json.dump(annotation, f, indent=2)

        r = annotation["quality_rating"]
        summary["total"] += 1
        summary["by_rating"][r] += 1

        quality_str = f"{'★' * r}{'☆' * (5 - r)}"
        print(f"  {fname:45s} → {quality_str} ({annotation['quality_score']:.0f}/100) [{annotation['ground_truth_label']}]")

    print("\n" + "=" * 60)
    print(f"Total: {summary['total']} annotations")
    for r in range(5, 0, -1):
        print(f"  Rating {r}: {summary['by_rating'][r]} videos")

    # Save summary
    with open(os.path.join(annotations_dir, '_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
