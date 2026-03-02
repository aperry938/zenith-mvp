# ZENith Yoga Dataset Card

Following the framework from Gebru et al. (2021), "Datasheets for Datasets."

## Motivation

**Purpose:** Training and evaluating webcam-based yoga pose classification and movement quality assessment systems.

**Creators:** Recorded and annotated by a researcher with dual credentials: M.S. Kinesiology (California Baptist University) and 500-hour Registered Yoga Teacher (RYT-500), providing expert ground truth for both biomechanical analysis and yoga pedagogy. B.S. Kinesiology from San Diego State University.

**Funding:** Self-funded research project.

## Composition

**Instances:** 98 video recordings of yoga poses, each with:
- Raw video (.mov)
- Extracted MediaPipe keypoints (.npy, shape: frames × 33 × 4)
- Expert biomechanical annotations (.json)

**Poses (10 classes):**

| Pose | Sanskrit | Correct | Incorrect | Total |
|------|----------|---------|-----------|-------|
| Chair | Utkatasana | 5 | 4 | 9 |
| Crescent Lunge | Anjaneyasana | 5 | 5 | 10 |
| Downward Dog | Adho Mukha Svanasana | 5 | 5 | 10 |
| Extended Side Angle | Utthita Parsvakonasana | 5 | 5 | 10 |
| High Lunge | Ashta Chandrasana | 5 | 5 | 10 |
| Mountain Pose | Tadasana | 5 | 5 | 10 |
| Plank | Kumbhakasana | 5 | 5 | 10 |
| Tree | Vrksasana | 5 | 5 | 10 |
| Triangle | Trikonasana | 5 | 5 | 10 |
| Warrior II | Virabhadrasana II | 5 | 4 | 9 |

**Labels per instance:**
- Pose class (10 classes)
- Form quality: correct / incorrect (binary)
- Expert quality rating: 1-5 ordinal scale
- Biomechanical quality score: 0-100 continuous
- Specific deviation labels (e.g., "knee_hyperflexion_L", "forward_lean")
- Pose phase annotations: entry / hold / exit / transition

**Features per frame (30 biomechanical):**
- 16 joint angles (bilateral shoulder, elbow, hip, knee, ankle, spinal)
- 6 segment ratios (torso-leg, arm symmetry, stance width, alignment, balance)
- 4 symmetry metrics (shoulder, elbow, hip, knee bilateral comparison)
- 4 stability indicators (velocity variance, CoM oscillation, BoS area, weight distribution)

**Statistics:**
- Total frames: ~45,000 across all videos
- Frame rate: 30 fps
- Resolution: 640×480 (webcam)
- Duration per video: 11-19 seconds

## Collection Process

**Recording setup:** Single webcam (MacBook built-in, 640×480), indoor setting with consistent lighting, neutral background.

**Performer:** Single expert practitioner (the researcher), performing each pose with deliberate correct form and deliberate common errors for incorrect examples.

**Correct form criteria:** Based on standard yoga alignment principles (Iyengar tradition) and biomechanical assessment (kinesiology training).

**Incorrect form examples:** Deliberately performed common compensations identified in yoga pedagogy literature and clinical movement assessment:
- Knee valgus/varus
- Forward trunk lean
- Shoulder elevation
- Spinal lateral flexion
- Hip asymmetry
- Insufficient/excessive joint flexion

**Keypoint extraction:** MediaPipe Pose (33 landmarks, 4 values per landmark: x, y, z, visibility). Extracted offline from video files.

## Preprocessing

- Landmarks normalized to [0, 1] range (MediaPipe default)
- z-coordinate relative to hip width (MediaPipe depth estimation)
- No additional normalization or augmentation applied to raw keypoints
- Biomechanical features computed via `biomechanical_features.py` with normalization to [0, 1]

## Annotation Process

**Annotator:** The dataset creator (M.S. Kinesiology, 500hr RYT).

**Annotation method:** Semi-automated biomechanical analysis:
1. Biomechanical features extracted per frame
2. Pose-specific quality profiles define ideal ranges for critical features
3. Quality scores computed from deviation from ideal ranges
4. Expert review and calibration of ideal ranges against actual performance data
5. Deviation labels mapped to standardized biomechanical terminology

**Quality rating scale:**
- 5 (Excellent): Quality score ≥ 95, all critical features within ideal range
- 4 (Good): Quality score 85-94, minor deviations
- 3 (Moderate): Quality score 70-84, notable deviations
- 2 (Fair): Quality score 50-69, significant deviations
- 1 (Poor): Quality score < 50, major compensations

## Uses

**Intended uses:**
- Pose classification model training and evaluation
- Movement quality assessment research
- Ablation studies comparing feature representations
- Real-time yoga coaching system development

**Not suitable for:**
- Clinical movement disorder diagnosis
- Rehabilitation without practitioner supervision
- Cross-population generalization claims (single-performer dataset)

## Distribution

- Dataset is part of the ZENith research project
- Available upon request for research purposes
- License: Research use only

## Limitations

1. **Single performer:** All videos feature the same person, limiting body-type diversity
2. **Single camera:** Fixed webcam angle, no multi-view coverage
3. **Deliberate errors:** Incorrect form is performed deliberately, not naturally occurring errors from novice practitioners
4. **10 poses only:** Limited to standing and floor poses; no inversions, arm balances, or seated poses
5. **Indoor only:** Consistent but non-diverse environment
6. **2D projection:** MediaPipe provides pseudo-3D but fundamentally operates on 2D projections

## Ethical Considerations

- Dataset contains only the researcher's own recordings (no third-party data)
- No personally identifying information beyond body pose
- Not intended for surveillance or non-consensual movement analysis
- Quality ratings reflect biomechanical alignment, not aesthetic judgment
