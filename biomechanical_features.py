"""
Biomechanical Feature Engineering for ZENith.

Computes anatomically meaningful features from MediaPipe 33-landmark pose data.
Designed by a dual-credentialed researcher (M.S. Kinesiology + 500hr RYT)
to capture movement quality dimensions that raw landmark coordinates cannot.

Feature Groups:
    1. Joint Angles (16): Bilateral measurements of major articulations
    2. Segment Ratios (6): Proportional and alignment relationships
    3. Symmetry Metrics (4): Left-right bilateral comparison
    4. Stability Indicators (4): Balance and steadiness measures

Total: 30 biomechanical features per frame.

MediaPipe Landmark Reference:
    0: nose, 11/12: L/R shoulder, 13/14: L/R elbow, 15/16: L/R wrist,
    23/24: L/R hip, 25/26: L/R knee, 27/28: L/R ankle,
    29/30: L/R heel, 31/32: L/R foot_index
"""

import numpy as np
from pose_foundations import calculate_angle

# ── MediaPipe landmark indices ──────────────────────────────────────────────
NOSE = 0
L_EYE_INNER = 1
R_EYE_INNER = 4
L_EAR = 7
R_EAR = 8
L_SHOULDER = 11
R_SHOULDER = 12
L_ELBOW = 13
R_ELBOW = 14
L_WRIST = 15
R_WRIST = 16
L_HIP = 23
R_HIP = 24
L_KNEE = 25
R_KNEE = 26
L_ANKLE = 27
R_ANKLE = 28
L_HEEL = 29
R_HEEL = 30
L_FOOT_INDEX = 31
R_FOOT_INDEX = 32

# ── Feature names (ordered) ────────────────────────────────────────────────
FEATURE_NAMES = [
    # Joint Angles (16)
    "l_shoulder_flexion",       # 0
    "r_shoulder_flexion",       # 1
    "l_shoulder_abduction",     # 2
    "r_shoulder_abduction",     # 3
    "l_elbow_flexion",          # 4
    "r_elbow_flexion",          # 5
    "l_hip_flexion",            # 6
    "r_hip_flexion",            # 7
    "l_hip_abduction",          # 8
    "r_hip_abduction",          # 9
    "l_knee_flexion",           # 10
    "r_knee_flexion",           # 11
    "l_ankle_dorsiflexion",     # 12
    "r_ankle_dorsiflexion",     # 13
    "spinal_lateral_flexion",   # 14
    "trunk_forward_lean",       # 15
    # Segment Ratios (6)
    "torso_leg_ratio",          # 16
    "arm_span_symmetry",        # 17
    "stance_width_ratio",       # 18
    "shoulder_hip_offset",      # 19
    "com_base_displacement",    # 20
    "head_spine_alignment",     # 21
    # Symmetry Metrics (4)
    "shoulder_angle_symmetry",  # 22
    "elbow_angle_symmetry",     # 23
    "hip_angle_symmetry",       # 24
    "knee_angle_symmetry",      # 25
    # Stability Indicators (4)
    "landmark_velocity_var",    # 26
    "com_oscillation",          # 27
    "base_of_support_area",     # 28
    "weight_distribution",      # 29
]

NUM_FEATURES = len(FEATURE_NAMES)  # 30


def _pt(landmarks, idx):
    """Extract [x, y, z] for a landmark. Works with both (33,4) array and list-of-landmarks."""
    if hasattr(landmarks, 'landmark'):
        # MediaPipe NormalizedLandmarkList
        lm = landmarks.landmark[idx]
        return np.array([lm.x, lm.y, lm.z])
    else:
        # NumPy array (33, 4) or (33, 3)
        return np.array(landmarks[idx][:3], dtype=np.float64)


def _pt2d(landmarks, idx):
    """Extract [x, y] for a landmark (2D projection)."""
    p = _pt(landmarks, idx)
    return p[:2]


def _dist(a, b):
    """Euclidean distance between two points."""
    return np.linalg.norm(np.array(a) - np.array(b))


def _midpoint(a, b):
    """Midpoint between two points."""
    return (np.array(a) + np.array(b)) / 2.0


def _angle_3pt(landmarks, a_idx, b_idx, c_idx):
    """Angle at vertex b formed by segments ba and bc, in degrees [0, 180]."""
    return calculate_angle(
        _pt2d(landmarks, a_idx),
        _pt2d(landmarks, b_idx),
        _pt2d(landmarks, c_idx)
    )


def _triangle_area(p1, p2, p3):
    """Area of triangle from three 2D points using cross product."""
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    return 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])


# ── Joint Angles (16 features) ─────────────────────────────────────────────

def compute_joint_angles(landmarks):
    """
    Compute 16 joint angles from anatomical landmark positions.

    These capture the kinematic state of major articulations relevant to yoga:
    - Shoulder flexion/abduction: arm position relative to torso
    - Elbow flexion: arm bend
    - Hip flexion/abduction: leg position relative to pelvis
    - Knee flexion: leg bend
    - Ankle dorsiflexion: foot-shin angle
    - Spinal lateral flexion: side-bend of trunk
    - Trunk forward lean: sagittal plane trunk angle

    Returns array of 16 angles in degrees, normalized to [0, 1] by dividing by 180.
    """
    angles = np.zeros(16, dtype=np.float64)

    # Shoulder flexion: angle at shoulder (hip-shoulder-elbow) — sagittal plane arm raise
    angles[0] = _angle_3pt(landmarks, L_HIP, L_SHOULDER, L_ELBOW)       # L shoulder flexion
    angles[1] = _angle_3pt(landmarks, R_HIP, R_SHOULDER, R_ELBOW)       # R shoulder flexion

    # Shoulder abduction: angle at shoulder in frontal plane
    # Approximated as hip-shoulder-wrist (captures full arm abduction)
    angles[2] = _angle_3pt(landmarks, L_HIP, L_SHOULDER, L_WRIST)       # L shoulder abduction
    angles[3] = _angle_3pt(landmarks, R_HIP, R_SHOULDER, R_WRIST)       # R shoulder abduction

    # Elbow flexion: angle at elbow (shoulder-elbow-wrist)
    angles[4] = _angle_3pt(landmarks, L_SHOULDER, L_ELBOW, L_WRIST)     # L elbow
    angles[5] = _angle_3pt(landmarks, R_SHOULDER, R_ELBOW, R_WRIST)     # R elbow

    # Hip flexion: angle at hip (shoulder-hip-knee) — sagittal plane
    angles[6] = _angle_3pt(landmarks, L_SHOULDER, L_HIP, L_KNEE)        # L hip flexion
    angles[7] = _angle_3pt(landmarks, R_SHOULDER, R_HIP, R_KNEE)        # R hip flexion

    # Hip abduction: angle at hip (opposite_hip-hip-knee) — frontal plane
    angles[8] = _angle_3pt(landmarks, R_HIP, L_HIP, L_KNEE)            # L hip abduction
    angles[9] = _angle_3pt(landmarks, L_HIP, R_HIP, R_KNEE)            # R hip abduction

    # Knee flexion: angle at knee (hip-knee-ankle)
    angles[10] = _angle_3pt(landmarks, L_HIP, L_KNEE, L_ANKLE)          # L knee
    angles[11] = _angle_3pt(landmarks, R_HIP, R_KNEE, R_ANKLE)          # R knee

    # Ankle dorsiflexion: angle at ankle (knee-ankle-foot_index)
    angles[12] = _angle_3pt(landmarks, L_KNEE, L_ANKLE, L_FOOT_INDEX)   # L ankle
    angles[13] = _angle_3pt(landmarks, R_KNEE, R_ANKLE, R_FOOT_INDEX)   # R ankle

    # Spinal lateral flexion: angle deviation of shoulder midpoint from vertical
    # through hip midpoint. Measures side-bending of the trunk.
    l_sh = _pt2d(landmarks, L_SHOULDER)
    r_sh = _pt2d(landmarks, R_SHOULDER)
    l_hp = _pt2d(landmarks, L_HIP)
    r_hp = _pt2d(landmarks, R_HIP)
    sh_mid = _midpoint(l_sh, r_sh)
    hp_mid = _midpoint(l_hp, r_hp)
    trunk_vec = sh_mid - hp_mid
    # Angle from vertical (vertical = [0, -1] in image coords where y increases downward)
    vertical = np.array([0.0, -1.0])
    cos_angle = np.dot(trunk_vec, vertical) / (np.linalg.norm(trunk_vec) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angles[14] = np.degrees(np.arccos(cos_angle))  # Spinal lateral flexion

    # Trunk forward lean: sagittal plane angle using z-coordinate
    # Approximated by shoulder-hip-vertical angle in the y-z plane
    sh_mid_3d = _midpoint(_pt(landmarks, L_SHOULDER), _pt(landmarks, R_SHOULDER))
    hp_mid_3d = _midpoint(_pt(landmarks, L_HIP), _pt(landmarks, R_HIP))
    trunk_3d = sh_mid_3d - hp_mid_3d
    # Angle from vertical in sagittal plane (y-z)
    vert_3d = np.array([0.0, -1.0, 0.0])
    cos_lean = np.dot(trunk_3d, vert_3d) / (np.linalg.norm(trunk_3d) + 1e-8)
    cos_lean = np.clip(cos_lean, -1.0, 1.0)
    angles[15] = np.degrees(np.arccos(cos_lean))  # Trunk forward lean

    # Normalize to [0, 1]
    return angles / 180.0


# ── Segment Ratios (6 features) ────────────────────────────────────────────

def compute_segment_ratios(landmarks):
    """
    Compute 6 proportional/alignment features.

    These capture postural alignment relationships that define yoga pose quality:
    - Torso-to-leg ratio: body proportion awareness
    - Arm span symmetry: bilateral arm extension balance
    - Stance width ratio: foot spread relative to hip width
    - Shoulder-hip alignment offset: frontal plane torso alignment
    - CoM-to-base displacement: balance (how centered is mass over support)
    - Head-spine alignment: cervical spine neutral check

    Returns array of 6 values, each normalized to roughly [0, 1].
    """
    ratios = np.zeros(6, dtype=np.float64)

    # Torso length: shoulder midpoint to hip midpoint
    sh_mid = _midpoint(_pt2d(landmarks, L_SHOULDER), _pt2d(landmarks, R_SHOULDER))
    hp_mid = _midpoint(_pt2d(landmarks, L_HIP), _pt2d(landmarks, R_HIP))
    torso_len = _dist(sh_mid, hp_mid)

    # Leg length: hip to ankle (average of both sides)
    l_leg = _dist(_pt2d(landmarks, L_HIP), _pt2d(landmarks, L_ANKLE))
    r_leg = _dist(_pt2d(landmarks, R_HIP), _pt2d(landmarks, R_ANKLE))
    leg_len = (l_leg + r_leg) / 2.0

    # Torso-to-leg ratio (typical ~0.5-0.7; normalized by clamping)
    ratios[0] = np.clip(torso_len / (leg_len + 1e-8), 0.0, 2.0) / 2.0

    # Arm span symmetry: |left_arm_length - right_arm_length| / avg_arm_length
    l_arm = _dist(_pt2d(landmarks, L_SHOULDER), _pt2d(landmarks, L_WRIST))
    r_arm = _dist(_pt2d(landmarks, R_SHOULDER), _pt2d(landmarks, R_WRIST))
    avg_arm = (l_arm + r_arm) / 2.0
    # 0 = perfect symmetry, 1 = maximal asymmetry. Invert so 1 = symmetric.
    ratios[1] = 1.0 - np.clip(abs(l_arm - r_arm) / (avg_arm + 1e-8), 0.0, 1.0)

    # Stance width / hip width
    hip_width = _dist(_pt2d(landmarks, L_HIP), _pt2d(landmarks, R_HIP))
    stance_width = _dist(_pt2d(landmarks, L_ANKLE), _pt2d(landmarks, R_ANKLE))
    ratios[2] = np.clip(stance_width / (hip_width + 1e-8), 0.0, 5.0) / 5.0

    # Shoulder-hip alignment offset (frontal plane)
    # Horizontal offset between shoulder midpoint and hip midpoint
    offset = abs(sh_mid[0] - hp_mid[0])
    # Normalize by hip width
    ratios[3] = 1.0 - np.clip(offset / (hip_width + 1e-8), 0.0, 1.0)

    # Center of Mass (CoM) horizontal displacement from Base of Support (BoS) center
    # Approximate CoM as weighted average of major segments
    # (shoulders 20%, hips 40%, knees 20%, ankles 20%)
    kn_mid = _midpoint(_pt2d(landmarks, L_KNEE), _pt2d(landmarks, R_KNEE))
    an_mid = _midpoint(_pt2d(landmarks, L_ANKLE), _pt2d(landmarks, R_ANKLE))
    com = 0.2 * sh_mid + 0.4 * hp_mid + 0.2 * kn_mid + 0.2 * an_mid

    # BoS center: midpoint of the four foot landmarks
    l_foot_center = _midpoint(_pt2d(landmarks, L_HEEL), _pt2d(landmarks, L_FOOT_INDEX))
    r_foot_center = _midpoint(_pt2d(landmarks, R_HEEL), _pt2d(landmarks, R_FOOT_INDEX))
    bos_center = _midpoint(l_foot_center, r_foot_center)

    # Use HORIZONTAL (x) displacement only — vertical offset is pose-dependent
    com_disp_x = abs(com[0] - bos_center[0])
    # Normalize by body height (shoulder-ankle distance) for pose-invariance
    body_height = _dist(sh_mid, an_mid) + 1e-8
    ratios[4] = 1.0 - np.clip(com_disp_x / (body_height * 0.3), 0.0, 1.0)

    # Head-spine alignment: offset of nose from shoulder midpoint
    nose = _pt2d(landmarks, NOSE)
    head_offset = _dist(nose, sh_mid)
    # Normalize by torso length
    ratios[5] = 1.0 - np.clip(head_offset / (torso_len + 1e-8), 0.0, 1.0)

    return ratios


# ── Symmetry Metrics (4 features) ──────────────────────────────────────────

def compute_symmetry_metrics(joint_angles):
    """
    Compute 4 bilateral symmetry features from pre-computed joint angles.

    Symmetry is critical in yoga for detecting compensatory patterns.
    Each metric is |left_angle - right_angle|, normalized and inverted
    so that 1.0 = perfect symmetry, 0.0 = maximal asymmetry.

    Input: raw joint angles (16 values, already normalized to [0,1])
    """
    symmetry = np.zeros(4, dtype=np.float64)

    # Shoulder symmetry: avg of flexion and abduction symmetry
    sh_flex_diff = abs(joint_angles[0] - joint_angles[1])  # L/R shoulder flexion
    sh_abd_diff = abs(joint_angles[2] - joint_angles[3])   # L/R shoulder abduction
    symmetry[0] = 1.0 - np.clip((sh_flex_diff + sh_abd_diff) / 2.0, 0.0, 1.0)

    # Elbow symmetry
    symmetry[1] = 1.0 - np.clip(abs(joint_angles[4] - joint_angles[5]), 0.0, 1.0)

    # Hip symmetry: avg of flexion and abduction symmetry
    hp_flex_diff = abs(joint_angles[6] - joint_angles[7])
    hp_abd_diff = abs(joint_angles[8] - joint_angles[9])
    symmetry[2] = 1.0 - np.clip((hp_flex_diff + hp_abd_diff) / 2.0, 0.0, 1.0)

    # Knee symmetry
    symmetry[3] = 1.0 - np.clip(abs(joint_angles[10] - joint_angles[11]), 0.0, 1.0)

    return symmetry


# ── Stability Indicators (4 features) ──────────────────────────────────────

class StabilityTracker:
    """
    Tracks temporal stability features across frames.
    Must be called sequentially with each new frame.
    """

    def __init__(self, buffer_size=15):
        self.buffer_size = buffer_size
        self.landmark_history = []  # List of (33, 3) arrays
        self.com_history = []       # List of 2D CoM positions

    def reset(self):
        self.landmark_history.clear()
        self.com_history.clear()

    def update(self, landmarks):
        """Add current frame landmarks to history buffer."""
        # Extract all 33 landmark positions as (33, 3)
        pts = np.array([_pt(landmarks, i) for i in range(33)])
        self.landmark_history.append(pts)
        if len(self.landmark_history) > self.buffer_size:
            self.landmark_history.pop(0)

        # Compute and store CoM
        sh_mid = _midpoint(pts[L_SHOULDER], pts[R_SHOULDER])
        hp_mid = _midpoint(pts[L_HIP], pts[R_HIP])
        kn_mid = _midpoint(pts[L_KNEE], pts[R_KNEE])
        an_mid = _midpoint(pts[L_ANKLE], pts[R_ANKLE])
        com = 0.2 * sh_mid + 0.4 * hp_mid + 0.2 * kn_mid + 0.2 * an_mid
        self.com_history.append(com[:2])
        if len(self.com_history) > self.buffer_size:
            self.com_history.pop(0)

    def compute(self, landmarks):
        """
        Compute 4 stability indicators.

        1. Landmark velocity variance: how much total body movement varies frame-to-frame
        2. CoM oscillation amplitude: sway of center of mass
        3. Base of support area: foot polygon area (wider = more stable)
        4. Weight distribution: how evenly weight is distributed (estimated from foot positions)

        Returns array of 4 values in [0, 1].
        """
        stability = np.zeros(4, dtype=np.float64)

        # 1. Landmark velocity variance
        if len(self.landmark_history) >= 3:
            velocities = []
            for i in range(1, len(self.landmark_history)):
                diff = self.landmark_history[i] - self.landmark_history[i - 1]
                velocities.append(np.linalg.norm(diff, axis=1).mean())
            vel_var = np.var(velocities)
            # Low variance = stable. Normalize: typical variance range [0, 0.01]
            stability[0] = 1.0 - np.clip(vel_var / 0.005, 0.0, 1.0)
        else:
            stability[0] = 0.5  # Neutral until enough history

        # 2. CoM oscillation amplitude
        if len(self.com_history) >= 3:
            com_arr = np.array(self.com_history)
            com_mean = com_arr.mean(axis=0)
            oscillation = np.sqrt(np.mean(np.sum((com_arr - com_mean) ** 2, axis=1)))
            # Low oscillation = stable. Normalize: typical range [0, 0.05]
            stability[1] = 1.0 - np.clip(oscillation / 0.03, 0.0, 1.0)
        else:
            stability[1] = 0.5

        # 3. Base of support area (convex hull of 4 foot landmarks)
        l_heel = _pt2d(landmarks, L_HEEL)
        r_heel = _pt2d(landmarks, R_HEEL)
        l_toe = _pt2d(landmarks, L_FOOT_INDEX)
        r_toe = _pt2d(landmarks, R_FOOT_INDEX)
        # Approximate convex hull area as sum of two triangles
        area = _triangle_area(l_heel, r_heel, l_toe) + _triangle_area(r_heel, l_toe, r_toe)
        # Normalize: typical BoS area [0, 0.1] in normalized coordinates
        stability[2] = np.clip(area / 0.05, 0.0, 1.0)

        # 4. Weight distribution estimate
        # Compare horizontal CoM position to BoS center
        bos_center_x = (l_heel[0] + r_heel[0] + l_toe[0] + r_toe[0]) / 4.0
        bos_width = max(abs(l_heel[0] - r_heel[0]), abs(l_toe[0] - r_toe[0]), 1e-8)

        sh_mid = _midpoint(_pt2d(landmarks, L_SHOULDER), _pt2d(landmarks, R_SHOULDER))
        hp_mid = _midpoint(_pt2d(landmarks, L_HIP), _pt2d(landmarks, R_HIP))
        kn_mid = _midpoint(_pt2d(landmarks, L_KNEE), _pt2d(landmarks, R_KNEE))
        an_mid = _midpoint(_pt2d(landmarks, L_ANKLE), _pt2d(landmarks, R_ANKLE))
        com_x = 0.2 * sh_mid[0] + 0.4 * hp_mid[0] + 0.2 * kn_mid[0] + 0.2 * an_mid[0]

        offset_ratio = abs(com_x - bos_center_x) / bos_width
        # 0 offset = perfectly centered = 1.0
        stability[3] = 1.0 - np.clip(offset_ratio, 0.0, 1.0)

        return stability


# ── Main Extraction Function ────────────────────────────────────────────────

def extract_biomechanical_features(landmarks, stability_tracker=None):
    """
    Extract all 30 biomechanical features from a single frame of landmarks.

    Args:
        landmarks: Either MediaPipe NormalizedLandmarkList or numpy array (33, 4)
        stability_tracker: Optional StabilityTracker for temporal features.
                          If None, stability features default to 0.5 (neutral).

    Returns:
        numpy array of shape (30,), dtype float64, all values in [0, 1]
    """
    # Joint angles (16 features)
    joint_angles = compute_joint_angles(landmarks)

    # Segment ratios (6 features)
    segment_ratios = compute_segment_ratios(landmarks)

    # Symmetry metrics (4 features, derived from joint angles)
    symmetry = compute_symmetry_metrics(joint_angles)

    # Stability indicators (4 features)
    if stability_tracker is not None:
        stability_tracker.update(landmarks)
        stability = stability_tracker.compute(landmarks)
    else:
        stability = np.full(4, 0.5, dtype=np.float64)

    return np.concatenate([joint_angles, segment_ratios, symmetry, stability])


def extract_biomechanical_batch(landmarks_sequence, stability_tracker=None):
    """
    Extract biomechanical features for a sequence of frames.

    Args:
        landmarks_sequence: numpy array of shape (N, 33, 4)
        stability_tracker: Optional StabilityTracker. If None, creates one internally.

    Returns:
        numpy array of shape (N, 30)
    """
    if stability_tracker is None:
        stability_tracker = StabilityTracker()
    else:
        stability_tracker.reset()

    N = landmarks_sequence.shape[0]
    features = np.zeros((N, NUM_FEATURES), dtype=np.float64)

    for i in range(N):
        features[i] = extract_biomechanical_features(
            landmarks_sequence[i], stability_tracker
        )

    return features


# ── Pose-Specific Feature Profiles ──────────────────────────────────────────

# Each profile defines:
#   "critical_features": indices of features most important for this pose
#   "ideal_ranges": {feature_index: (min_normalized, max_normalized)} for key features
#   "description": human-readable biomechanical rationale
#
# Angles are stored normalized (/180), so 90° = 0.5, 180° = 1.0, 45° = 0.25

POSE_PROFILES = {
    # Data-calibrated ideal ranges: derived from correct-form video distributions.
    # Format: feature_idx: (min_normalized, max_normalized)
    # Angles normalized /180: 90°→0.5, 180°→1.0, 45°→0.25
    # Non-angle features are in [0,1] directly.
    "Chair": {
        "critical_features": [0, 1, 6, 7, 10, 11, 14, 15, 22, 25],
        "ideal_ranges": {
            0: (0.94, 1.0),     # L shoulder flexion ~177° (arms overhead)
            1: (0.92, 0.97),    # R shoulder flexion ~170°
            6: (0.91, 0.96),    # L hip flexion ~169° (upright torso, slight hip bend)
            7: (0.92, 0.96),    # R hip flexion ~169°
            10: (0.89, 0.94),   # L knee flexion ~165° (moderate squat)
            11: (0.89, 0.94),   # R knee flexion ~165°
            14: (0.0, 0.02),    # Lateral flexion <4° (upright)
            15: (0.0, 0.15),    # Forward lean <27° (upright torso)
            22: (0.95, 1.0),    # Shoulder symmetry (arms matching)
            25: (0.97, 1.0),    # Knee symmetry (bilateral match)
        },
        "description": "Chair requires bilateral hip/knee flexion with upright torso and arms overhead. "
                       "Key compensations: excessive forward lean, asymmetric knee bend, "
                       "knee valgus (knees collapsing inward), shoulder asymmetry."
    },
    "Crescent Lunge": {
        "critical_features": [6, 7, 10, 11, 14, 15, 20],
        "ideal_ranges": {
            10: (0.72, 0.94),   # Front knee: 130-170° (lunge depth varies)
            11: (0.89, 1.0),    # Back knee: 160-180° (extended)
            14: (0.0, 0.05),    # Spinal lateral: <9° (vertical)
            15: (0.0, 0.17),    # Trunk forward lean: <30°
        },
        "description": "Crescent lunge demands split stance with front knee flexion, "
                       "extended back leg, and vertical torso. Key compensations: "
                       "forward lean, back knee collapse, lateral trunk shift."
    },
    "Downward Dog": {
        "critical_features": [0, 1, 6, 7, 11, 15, 22],
        "ideal_ranges": {
            0: (0.94, 1.0),     # L shoulder flexion ~176° (arms fully extended)
            1: (0.88, 1.0),     # R shoulder flexion ~171°
            6: (0.04, 0.11),    # L hip flexion ~14° (tight inverted V)
            7: (0.02, 0.12),    # R hip flexion ~12°
            11: (0.85, 1.0),    # R knee ~166° (straight leg)
            15: (0.53, 0.58),   # Trunk lean ~101° (inverted orientation)
            22: (0.95, 1.0),    # Shoulder symmetry
        },
        "description": "Downward dog is an inverted V with shoulders flexed overhead, "
                       "hips at acute angle, and straight legs. Key compensations: "
                       "rounded upper back, bent knees, shoulder impingement."
    },
    "Extended Side Angle": {
        "critical_features": [0, 6, 7, 10, 14, 17, 22],
        "ideal_ranges": {
            0: (0.47, 0.54),    # L shoulder flexion ~91° (arm position)
            6: (0.33, 0.41),    # L hip flexion ~67° (side-bent torso)
            7: (0.84, 0.90),    # R hip flexion ~157° (extended side)
            10: (0.97, 1.0),    # L knee ~179° (straight front leg)
            14: (0.29, 0.35),   # Lateral flexion ~57° (side bend)
            17: (0.95, 1.0),    # Arm symmetry (arms in line)
        },
        "description": "Extended side angle requires frontal plane side-bend with "
                       "arm line from hand to hand. Key compensations: torso rotation, "
                       "collapsed knee, shoulder elevation."
    },
    "High Lunge": {
        "critical_features": [6, 7, 10, 11, 14, 15],
        "ideal_ranges": {
            10: (0.72, 0.94),   # Front knee: 130-170°
            11: (0.89, 1.0),    # Back knee: 160-180° (extended)
            14: (0.0, 0.05),    # Spinal lateral: <9° (vertical)
            15: (0.0, 0.17),    # Trunk lean: <30°
        },
        "description": "High lunge is similar to crescent but with back heel lifted. "
                       "Demands balance with front knee flexion and vertical torso. "
                       "Key compensations: forward lean, lateral sway, back knee collapse."
    },
    "Mountain Pose": {
        "critical_features": [6, 7, 10, 11, 14, 15, 19, 22, 25],
        "ideal_ranges": {
            6: (0.96, 1.0),     # L hip ~176° (straight standing)
            7: (0.95, 0.97),    # R hip ~174°
            10: (0.97, 1.0),    # L knee ~178° (straight)
            11: (0.95, 0.98),   # R knee ~175°
            14: (0.0, 0.005),   # Lateral flexion <1° (perfectly vertical)
            15: (0.15, 0.22),   # Forward lean ~34° (normal standing posture offset)
            19: (0.95, 1.0),    # Shoulder-hip alignment
            22: (0.97, 1.0),    # Shoulder symmetry
            25: (0.97, 1.0),    # Knee symmetry
        },
        "description": "Mountain pose (Tadasana) is the baseline standing posture. "
                       "Perfect vertical alignment, bilateral symmetry, centered weight. "
                       "Key compensations: lateral lean, weight shift, head forward."
    },
    "Plank": {
        "critical_features": [6, 7, 10, 11, 15, 17, 22, 25],
        "ideal_ranges": {
            6: (0.78, 0.92),    # L hip ~155° (body line, slight flexion ok)
            7: (0.65, 0.90),    # R hip ~141° (some variance from camera angle)
            10: (0.84, 0.93),   # L knee ~160° (straight)
            11: (0.83, 0.93),   # R knee ~159°
            15: (0.50, 0.55),   # Trunk lean ~95° (horizontal body)
            17: (0.95, 1.0),    # Arm symmetry (both arms supporting)
            22: (0.85, 1.0),    # Shoulder symmetry
            25: (0.95, 1.0),    # Knee symmetry
        },
        "description": "Plank demands a straight line from shoulders through hips to ankles. "
                       "Key compensations: hip sag (excessive extension), pike (hip flexion), "
                       "shoulder protraction, head drop."
    },
    "Tree": {
        "critical_features": [6, 10, 11, 14, 15, 19, 22],
        "ideal_ranges": {
            6: (0.97, 1.0),     # L hip ~179° (standing leg straight)
            10: (0.97, 1.0),    # L knee ~179° (standing leg straight)
            11: (0.15, 0.21),   # R knee ~33° (lifted leg folded)
            14: (0.0, 0.02),    # Lateral flexion <4° (vertical trunk)
            15: (0.0, 0.04),    # Forward lean <7°
            19: (0.85, 1.0),    # Shoulder-hip alignment (vertical)
            22: (0.85, 1.0),    # Shoulder symmetry
        },
        "description": "Tree pose is a single-leg balance with hip external rotation. "
                       "Key compensations: lateral trunk lean, excessive hip hike, "
                       "standing knee hyperextension, excessive sway."
    },
    "Triangle": {
        "critical_features": [0, 6, 7, 10, 11, 14, 17, 22],
        "ideal_ranges": {
            0: (0.47, 0.54),    # L shoulder ~91° (lateral arm position)
            6: (0.33, 0.41),    # L hip ~67° (side bend)
            7: (0.84, 0.90),    # R hip ~157° (extended side)
            10: (0.97, 1.0),    # L knee ~179° (straight front leg)
            11: (0.96, 1.0),    # R knee ~176° (straight back leg)
            14: (0.29, 0.35),   # Lateral flexion ~57° (side bend)
            17: (0.95, 1.0),    # Arm symmetry (vertical arm line)
        },
        "description": "Triangle is a lateral bend with straight legs and vertical arm line. "
                       "Key compensations: front knee bend, torso rotation, "
                       "forward lean instead of lateral bend."
    },
    "Warrior II": {
        "critical_features": [0, 2, 3, 10, 11, 14, 15, 22],
        "ideal_ranges": {
            0: (0.48, 0.65),    # L shoulder flexion ~103° (front arm extended laterally)
            2: (0.48, 0.65),    # L shoulder abduction
            10: (0.81, 0.90),   # L knee ~154° (front knee bent)
            11: (0.95, 1.0),    # R knee ~177° (back leg straight)
            14: (0.0, 0.03),    # Lateral flexion <6° (vertical torso)
            15: (0.0, 0.07),    # Forward lean <13°
            22: (0.35, 0.65),   # Shoulder symmetry (asymmetric by design)
        },
        "description": "Warrior II demands wide stance, bent front knee, horizontal arms, "
                       "and vertical torso. Key compensations: knee past ankle, "
                       "forward lean, dropped back arm, hip alignment."
    },
}

# Map classifier labels to profile keys
LABEL_TO_PROFILE = {
    "Chair": "Chair",
    "Crescent Lunge": "Crescent Lunge",
    "Downward Dog": "Downward Dog",
    "Extended Side Angle": "Extended Side Angle",
    "High Lunge": "High Lunge",
    "Mountain Pose": "Mountain Pose",
    "Plank": "Plank",
    "Tree": "Tree",
    "Triangle": "Triangle",
    "Warrior II": "Warrior II",
}

# Also map dataset file prefixes to profile keys
FILE_PREFIX_TO_PROFILE = {
    "Chair": "Chair",
    "Crescent": "Crescent Lunge",
    "Ddog": "Downward Dog",
    "Ext.side.angle": "Extended Side Angle",
    "High.Lunge": "High Lunge",
    "Tadasana": "Mountain Pose",
    "Plank": "Plank",
    "Tree": "Tree",
    "Triangle": "Triangle",
    "Warrior2": "Warrior II",
}


def compute_pose_quality_score(features, pose_label):
    """
    Compute a biomechanics-based quality score for a given pose.

    Compares the critical features against their ideal ranges.
    Returns a score in [0, 100] where 100 = all critical features within ideal range.

    Args:
        features: (30,) biomechanical feature vector
        pose_label: string matching a key in POSE_PROFILES

    Returns:
        float score [0, 100]
    """
    profile = POSE_PROFILES.get(pose_label)
    if profile is None:
        return 50.0  # Unknown pose → neutral

    ideal = profile["ideal_ranges"]
    if not ideal:
        return 50.0

    scores = []
    for feat_idx, (lo, hi) in ideal.items():
        val = features[feat_idx]
        if lo <= val <= hi:
            scores.append(1.0)
        else:
            # Distance from nearest edge, normalized by range width
            range_width = hi - lo
            if val < lo:
                deviation = (lo - val) / max(range_width, 0.05)
            else:
                deviation = (val - hi) / max(range_width, 0.05)
            # Soft penalty: score drops smoothly with deviation
            scores.append(max(0.0, 1.0 - deviation))

    return 100.0 * (sum(scores) / len(scores))


def get_deviations(features, pose_label):
    """
    Identify which biomechanical features deviate from ideal for a given pose.

    Returns list of dicts: [{"feature": name, "value": float, "ideal": (lo, hi), "deviation": float}]
    Only includes features that are outside their ideal range.
    """
    profile = POSE_PROFILES.get(pose_label)
    if profile is None:
        return []

    deviations = []
    for feat_idx, (lo, hi) in profile["ideal_ranges"].items():
        val = features[feat_idx]
        if val < lo:
            deviations.append({
                "feature": FEATURE_NAMES[feat_idx],
                "feature_idx": feat_idx,
                "value": float(val * 180.0) if feat_idx < 16 else float(val),
                "ideal_lo": float(lo * 180.0) if feat_idx < 16 else float(lo),
                "ideal_hi": float(hi * 180.0) if feat_idx < 16 else float(hi),
                "deviation": float(lo - val),
                "direction": "below"
            })
        elif val > hi:
            deviations.append({
                "feature": FEATURE_NAMES[feat_idx],
                "feature_idx": feat_idx,
                "value": float(val * 180.0) if feat_idx < 16 else float(val),
                "ideal_lo": float(lo * 180.0) if feat_idx < 16 else float(lo),
                "ideal_hi": float(hi * 180.0) if feat_idx < 16 else float(hi),
                "deviation": float(val - hi),
                "direction": "above"
            })

    # Sort by deviation magnitude (worst first)
    deviations.sort(key=lambda d: d["deviation"], reverse=True)
    return deviations


# ── Verification Utility ────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import sys

    data_dir = os.path.join(os.path.dirname(__file__), "ZENith_Data", "keypoints")
    if not os.path.isdir(data_dir):
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    print(f"Biomechanical Feature Extraction — {NUM_FEATURES} features")
    print("=" * 60)

    # Process a few files for verification
    files = sorted(os.listdir(data_dir))[:5]
    for fname in files:
        if not fname.endswith(".npy"):
            continue
        path = os.path.join(data_dir, fname)
        data = np.load(path)  # (N_frames, 33, 4)
        print(f"\n{fname}: {data.shape[0]} frames")

        # Extract features for middle frame
        mid = data.shape[0] // 2
        frame = data[mid]
        feats = extract_biomechanical_features(frame)
        print(f"  Feature vector: shape={feats.shape}, min={feats.min():.3f}, max={feats.max():.3f}")

        # Show a few key angles (denormalized to degrees)
        print(f"  L shoulder flexion: {feats[0]*180:.1f}°")
        print(f"  R shoulder flexion: {feats[1]*180:.1f}°")
        print(f"  L knee flexion:     {feats[10]*180:.1f}°")
        print(f"  R knee flexion:     {feats[11]*180:.1f}°")
        print(f"  Spinal lateral:     {feats[14]*180:.1f}°")
        print(f"  Arm symmetry:       {feats[17]:.3f}")
        print(f"  Knee symmetry:      {feats[25]:.3f}")

        # Batch extraction
        batch_feats = extract_biomechanical_batch(data[:10])
        print(f"  Batch (10 frames): shape={batch_feats.shape}")

        # Pose quality (if we can determine pose from filename)
        prefix = fname.split("_")[0]
        pose_label = FILE_PREFIX_TO_PROFILE.get(prefix)
        if pose_label:
            quality = compute_pose_quality_score(feats, pose_label)
            devs = get_deviations(feats, pose_label)
            print(f"  Quality ({pose_label}): {quality:.1f}/100")
            if devs:
                print(f"  Top deviation: {devs[0]['feature']} = {devs[0]['value']:.1f}° "
                      f"(ideal: {devs[0]['ideal_lo']:.1f}-{devs[0]['ideal_hi']:.1f}°)")
