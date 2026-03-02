# ZENith Annotation Protocol

## Overview

This document describes the semi-automated annotation pipeline used to generate expert quality ratings for the ZENith yoga dataset. Annotations combine computed biomechanical features with expert calibration from a researcher holding dual credentials: M.S. Kinesiology (California Baptist University), B.S. Kinesiology (San Diego State University), and 500-hour Registered Yoga Teacher (RYT-500).

## Quality Rating Scale

Each video receives an ordinal quality rating (1-5) derived from a continuous biomechanical quality score (0-100):

| Rating | Label | Quality Score | Description |
|--------|-------|---------------|-------------|
| 5 | Excellent | >= 95 | All critical features within ideal range |
| 4 | Good | 85-94 | Minor deviations from ideal alignment |
| 3 | Moderate | 70-84 | Notable deviations in non-critical features |
| 2 | Fair | 50-69 | Significant deviations in critical features |
| 1 | Poor | < 50 | Major compensations present |

## Biomechanical Quality Score Computation

The quality score is computed per-frame as a weighted mean of feature-level scores:

1. **Feature extraction:** 30 biomechanical features computed from MediaPipe 33-landmark skeleton (see `biomechanical_features.py`)
2. **Pose-specific profile lookup:** Each of the 10 poses has a defined set of critical features with ideal ranges
3. **Per-feature scoring:** Each critical feature is scored based on deviation from its ideal range:
   - Within ideal range: score = 1.0
   - Outside range: score decreases linearly with deviation magnitude, scaled by a pose-specific sensitivity factor
   - Score is clipped to [0.0, 1.0]
4. **Aggregation:** Quality score = weighted mean of all critical feature scores × 100
   - Critical features receive higher weight (defined per pose in `POSE_PROFILES`)
   - Non-critical features receive uniform weight of 1.0

## Feature Taxonomy

### Joint Angles (16 features)

Computed as the angle at joint B in the kinematic chain A→B→C:

| Feature | Landmarks (A→B→C) | Anatomical Meaning |
|---------|-------------------|--------------------|
| `l_shoulder_flexion` | L_elbow → L_shoulder → L_hip | Left glenohumeral flexion/extension |
| `r_shoulder_flexion` | R_elbow → R_shoulder → R_hip | Right glenohumeral flexion/extension |
| `l_shoulder_abduction` | L_elbow → L_shoulder → R_shoulder | Left shoulder abduction angle |
| `r_shoulder_abduction` | R_elbow → R_shoulder → L_shoulder | Right shoulder abduction angle |
| `l_elbow_flexion` | L_shoulder → L_elbow → L_wrist | Left elbow flexion |
| `r_elbow_flexion` | R_shoulder → R_elbow → R_wrist | Right elbow flexion |
| `l_hip_flexion` | L_shoulder → L_hip → L_knee | Left hip flexion/extension |
| `r_hip_flexion` | R_shoulder → R_hip → R_knee | Right hip flexion/extension |
| `l_hip_abduction` | R_hip → L_hip → L_knee | Left hip abduction |
| `r_hip_abduction` | L_hip → R_hip → R_knee | Right hip abduction |
| `l_knee_flexion` | L_hip → L_knee → L_ankle | Left knee flexion |
| `r_knee_flexion` | R_hip → R_knee → R_ankle | Right knee flexion |
| `l_ankle_dorsiflexion` | L_knee → L_ankle → L_foot_index | Left ankle dorsiflexion |
| `r_ankle_dorsiflexion` | R_knee → R_ankle → R_foot_index | Right ankle dorsiflexion |
| `spinal_lateral_flexion` | Shoulder midpoint → Hip midpoint → vertical reference | Trunk lateral deviation |
| `trunk_forward_lean` | Shoulder midpoint → Hip midpoint → vertical reference | Trunk sagittal inclination |

All angles are normalized to [0, 1] range by dividing by 180°.

### Segment Ratios (6 features)

| Feature | Computation | Meaning |
|---------|-------------|---------|
| `torso_leg_ratio` | Shoulder-hip distance / Hip-ankle distance | Postural proportion |
| `arm_span_symmetry` | 1 - |L_arm_length - R_arm_length| / max_length | Upper limb symmetry |
| `stance_width_ratio` | Ankle separation / Hip width | Base of support width |
| `shoulder_hip_alignment` | 1 - horizontal offset / body_height | Vertical stack quality |
| `com_over_bos` | 1 - horizontal CoM-BoS offset / (body_height × 0.3) | Balance index |
| `head_spine_alignment` | 1 - head-spine offset / body_height | Cervical alignment |

### Symmetry Metrics (4 features)

| Feature | Computation | Meaning |
|---------|-------------|---------|
| `shoulder_angle_symmetry` | 1 - |L - R| shoulder flexion / 180 | Bilateral shoulder balance |
| `elbow_angle_symmetry` | 1 - |L - R| elbow flexion / 180 | Bilateral elbow balance |
| `hip_angle_symmetry` | 1 - |L - R| hip flexion / 180 | Bilateral hip balance |
| `knee_angle_symmetry` | 1 - |L - R| knee flexion / 180 | Bilateral knee balance |

### Stability Indicators (4 features, temporal)

Computed over a sliding window of 15 frames (0.5s at 30fps):

| Feature | Computation | Meaning |
|---------|-------------|---------|
| `velocity_variance` | 1 - clipped variance of landmark velocities | Movement steadiness |
| `com_oscillation` | 1 - clipped amplitude of CoM horizontal motion | Balance steadiness |
| `bos_area` | Normalized convex hull area of foot landmarks | Support base size |
| `weight_distribution` | 1 - |left foot load - right foot load| | Weight balance |

## Deviation Labels

When a critical feature falls outside its ideal range, a standardized deviation label is generated:

### Label Format
`{feature_name}` with direction indicator (`above` or `below` ideal range)

### Common Deviation Patterns by Pose

| Pose | Common Deviations | Biomechanical Meaning |
|------|-------------------|-----------------------|
| Chair (Utkatasana) | `l_knee_flexion` below, `trunk_forward_lean` above | Insufficient knee bend, excessive forward lean |
| Warrior II | `l_knee_flexion` below, `hip_angle_symmetry` below | Front knee not to 90°, pelvis rotated |
| Tree (Vrksasana) | `com_over_bos` below, `hip_angle_symmetry` below | Balance offset, hip drop on lifted side |
| Downward Dog | `shoulder_hip_alignment` below, `l_knee_flexion` above | Shoulders not stacked, knees bent |
| Triangle | `spinal_lateral_flexion` above, `trunk_forward_lean` above | Collapsing into the pose |
| Plank | `trunk_forward_lean` above, `shoulder_hip_alignment` below | Hip sag or pike |

## Pose Phase Annotations

Each video is segmented into temporal phases:

| Phase | Definition | Detection Method |
|-------|------------|------------------|
| `entry` | First 15% of video frames | Temporal proportion |
| `hold` | Middle 70% of video frames | Temporal proportion (static hold period) |
| `exit` | Final 15% of video frames | Temporal proportion |

Note: Current phase detection uses fixed temporal proportions. Future work will use velocity-based phase detection (low velocity = hold, high velocity = transition).

## Annotation File Format

Each video produces a JSON annotation file:

```json
{
  "video_file": "Chair_Correct_01.npy",
  "pose": "Chair",
  "form_label": "correct",
  "quality_rating": 5,
  "quality_score_mean": 97.2,
  "quality_score_std": 1.8,
  "n_frames": 420,
  "deviations": [],
  "phase_annotations": {
    "entry": [0, 62],
    "hold": [63, 356],
    "exit": [357, 419]
  },
  "feature_statistics": {
    "l_knee_flexion": {"mean": 0.52, "std": 0.03, "min": 0.45, "max": 0.58}
  }
}
```

## Calibration Process

Ideal ranges were calibrated through an iterative data-driven process:

1. **Initial ranges:** Set from anatomical expectations and yoga pedagogy (Iyengar tradition)
2. **Data analysis:** Computed per-feature distributions across all correct-form videos per pose
3. **Range adjustment:** Widened ranges to mean ± 2 standard deviations of correct-form distributions
4. **Expert review:** Researcher verified adjusted ranges against biomechanical literature and teaching experience
5. **Discrimination validation:** Confirmed correct-form videos score significantly higher than incorrect-form (Cohen's d, Mann-Whitney U)

### Calibration Results

| Pose | Correct Mean | Incorrect Mean | Gap |
|------|-------------|----------------|-----|
| Triangle | 100.0 | 64.0 | 36.0 |
| Tree | 98.6 | 71.2 | 27.4 |
| Warrior II | 99.2 | 81.5 | 17.7 |
| Chair | 100.0 | 84.6 | 15.4 |
| Mountain | 99.3 | 87.5 | 11.8 |
| Downward Dog | 97.3 | 86.2 | 11.1 |
| Plank | 96.3 | 86.5 | 9.8 |
| Crescent Lunge | 97.1 | 91.2 | 5.9 |
| Extended Side Angle | 94.8 | 92.4 | 2.4 |
| High Lunge | 93.5 | 93.8 | -0.3 |

Note: Extended Side Angle and High Lunge show poor discrimination, indicating their profiles need further calibration. These poses have subtle form differences that may require additional critical features.

## Intra-Rater Reliability (Planned)

To validate annotation consistency:
1. Re-annotate 20% of videos (20 videos, stratified by pose and quality) after a 2-week interval
2. Compute Cohen's weighted kappa for ordinal quality ratings
3. Compute ICC (intraclass correlation coefficient) for continuous quality scores
4. Target: kappa >= 0.80 (substantial agreement), ICC >= 0.90

## Limitations

1. **Single annotator:** All ratings from the same expert. Inter-rater reliability cannot be assessed without additional annotators.
2. **Semi-automated pipeline:** Quality scores are computed algorithmically; expert review validates ranges but does not manually rate each frame.
3. **Fixed phase proportions:** Pose phases use temporal heuristics, not velocity-based detection.
4. **Profile calibration gaps:** Two poses (Extended Side Angle, High Lunge) show poor correct/incorrect discrimination, suggesting their biomechanical profiles need refinement.
5. **MediaPipe accuracy:** Underlying landmark accuracy affects all computed features. Depth estimation (z-coordinate) is approximate.
