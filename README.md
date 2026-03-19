# ZENith: Real-Time AI Movement Coach

**Anthony C. Perry** -- [github.com/aperry938](https://github.com/aperry938)

---

## Abstract

ZENith is a real-time biomechanical movement quality assessment system that uses webcam-based pose estimation to provide expert-level yoga coaching feedback. The system integrates MediaPipe pose estimation with 30 expert-designed biomechanical features (joint angles, segment ratios, symmetry metrics, stability indicators), a Variational Autoencoder for continuous quality scoring, and Random Forest classification across 10 yoga poses. A novel dataset of 98 self-recorded video sequences with semi-automated kinesiologist annotations bridges embodied expertise with supervised learning.

The core contribution is **biomechanically-grounded feature engineering**: 30 anatomically meaningful features designed by a researcher with dual credentials (M.S. Kinesiology (CBU), B.S. Kinesiology (SDSU), 500hr RYT) achieve equivalent classification accuracy to 132 raw landmark values with 4.4x fewer dimensions, while enabling interpretable quality feedback.

---

## Demo

A short video demonstrating the live application can be found here:

[![Watch the ZENith Demo Video](https://img.youtube.com/vi/lPnmOBwJfkE/0.jpg)](https://youtu.be/lPnmOBwJfkE)

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                         React Frontend (Vite)                          │
│  ┌──────────┐ ┌─────┐ ┌───────┐ ┌──────────┐ ┌────────┐ ┌────────┐  │
│  │VideoStage│ │ HUD │ │Ghost  │ │Biomech   │ │Generatv│ │Sequence│  │
│  │(webcam)  │ │     │ │Overlay│ │Panel     │ │Coach   │ │Bar     │  │
│  └────┬─────┘ └──▲──┘ └──▲───┘ └───▲──────┘ └───▲────┘ └───▲────┘  │
│       │          │       │         │             │           │        │
│       │         WebSocket (JSON metadata stream)             │        │
└───────┼──────────┼───────┼─────────┼─────────────┼───────────┼────────┘
        │          │       │         │             │           │
   Frame blob  ┌───┴───────┴─────────┴─────────────┴───────────┴──┐
        │      │           FastAPI Gateway (server.py)              │
        └─────►│  PoseHeuristics · PoseSequencer · SessionManager  │
               └───────────────────────┬───────────────────────────┘
                                       │
               ┌───────────────────────▼───────────────────────────┐
               │            ZenithBrain (zenith_brain.py)           │
               │                                                    │
               │  ┌─────────────┐    ┌───────────────────────────┐ │
               │  │  MediaPipe   │    │  Biomechanical Features   │ │
               │  │  33 landmarks│───►│  (30 expert-designed)     │ │
               │  └──────┬──────┘    └────────────┬──────────────┘ │
               │         │                        │                │
               │  ┌──────▼──────┐    ┌────────────▼──────────────┐ │
               │  │ Random Forest│    │  Quality Scoring (profiles │ │
               │  │ (pose label) │    │  + VAE + deviations)       │ │
               │  └─────────────┘    └──────────────────────────┘  │
               │  ┌─────────────┐    ┌──────────────────────────┐  │
               │  │  VAE Ghost   │    │  Bio Flow Score + Stability│ │
               │  │ (ideal form) │    │  (angular velocity + jerk) │ │
               │  └─────────────┘    └──────────────────────────┘  │
               └───────────────────────────────────────────────────┘
                                       │
               ┌───────────────────────▼───────────────────────────┐
               │         Gemini Vision Coach (optional)             │
               │   Context-aware coaching with pose + quality data  │
               └───────────────────────────────────────────────────┘
```

### Processing Pipeline

1. **Webcam capture** → JPEG frames at 30fps via `getUserMedia`
2. **WebSocket transport** → binary frame blobs to FastAPI server
3. **MediaPipe Pose** → 33 3D skeletal landmarks per frame
4. **Dual feature paths** (parallel, for ablation comparison):
   - **RAW:** Flatten 33×4 landmarks → 132-dim vector → Random Forest classifier
   - **BIO:** 30 biomechanical features → pose-specific quality scoring → deviation detection
5. **VAE quality** → reconstruction error against learned correct-form manifold
6. **Bio flow** → angular velocity-based movement quality (Butterworth-filtered jerk)
7. **Heuristic coaching** → rule-based corrections for all 10 poses (knee angle, hip alignment, lateral flexion, trunk lean) with varied coaching text, correction vectors, and server-side debounce
8. **Pose sequencing** → 3 guided flows (Warrior, Balance, Strength) with per-pose hold timers, auto-progression, and oracle-triggered analysis
9. **Gemini Vision** → context-aware coaching using pose label, quality score, and deviations
10. **JSON response** → label, quality, deviations, heuristics, stability, sequence, ghost → frontend

---

## Biomechanical Features (30 dimensions)

The core intellectual contribution: expert-designed features replacing raw landmarks.

| Category | Count | Features |
|----------|-------|----------|
| Joint Angles | 16 | Bilateral shoulder flexion/abduction, elbow flexion, hip flexion/abduction, knee flexion, ankle dorsiflexion, spinal lateral flexion, trunk forward lean |
| Segment Ratios | 6 | Torso-leg ratio, arm span symmetry, stance width, shoulder-hip alignment, CoM over BoS, head-spine alignment |
| Symmetry Metrics | 4 | Bilateral comparison: shoulder, elbow, hip, knee |
| Stability Indicators | 4 | Velocity variance, CoM oscillation, BoS area, weight distribution |

Each of the 10 poses has a **pose-specific quality profile** defining which features are critical and their ideal ranges, calibrated from correct-form video distributions and expert biomechanical knowledge.

See [`ANNOTATION_PROTOCOL.md`](ANNOTATION_PROTOCOL.md) for the full feature taxonomy.

---

## Evaluation Results

### Ablation Study: Feature Representation Comparison

Stratified 5-fold cross-validation, 4,587 frames, 10 pose classes:

| Condition | Dimensions | Random Forest Acc (95% CI) | MLP Acc (95% CI) |
|-----------|-----------|---------------------------|-------------------|
| RAW (landmarks) | 132 | 100.0% [1.00, 1.00] | 99.61% ± 0.21% |
| **BIO (biomechanical)** | **30** | **100.0% [1.00, 1.00]** | **99.74% ± 0.09%** |
| RAW+BIO | 162 | 100.0% [1.00, 1.00] | 99.74% ± 0.09% |
| BIO+TEMPORAL | 150 | 100.0% [1.00, 1.00] | 99.74% ± 0.09% |

**Key finding:** 30 biomechanical features achieve equivalent classification accuracy to 132 raw landmarks (**4.4x dimensionality reduction**) while enabling interpretable quality feedback.

### Quality Score Validation

| Metric | Value |
|--------|-------|
| Spearman rank correlation (ρ) with expert ratings | **0.958** (p < 10⁻⁵³) |
| Pearson correlation | 0.931 |
| ROC-AUC (correct vs. incorrect) | 0.766 |
| Cohen's d (effect size) | 0.611 |
| Mann-Whitney U (p-value) | 1838 (p < 10⁻⁵) |

### VAE Quality Discrimination

Trained on correct-form biomechanical features only:

| Model | Correct MSE | Incorrect MSE | Ratio |
|-------|-------------|---------------|-------|
| Standard Bio-VAE (30→8 latent) | 0.000510 | 0.005021 | 9.84× |
| Conditional VAE (pose-conditioned) | 0.000467 | 0.004822 | 10.33× |

### Real-Time Performance

| Component | Mean Latency | Achievable FPS |
|-----------|-------------|----------------|
| Bio feature extraction | 0.32 ms | 3,119 |
| Quality scoring | 0.001 ms | 692,806 |
| Full bio pipeline | 0.29 ms | 3,493 |
| 30fps frame budget | 33.3 ms | — |

Full pipeline uses **0.9% of the 30fps frame budget**.

### Figures

Generated evaluation figures in `evaluation/results/figures/`:
- `fig_ablation_results.png` — Accuracy comparison across conditions and classifiers
- `fig_confusion_matrix.png` — 10-class confusion matrix (BIO + Random Forest)
- `fig_quality_discrimination.png` — Quality score distributions (correct vs. incorrect)
- `fig_quality_scatter.png` — Quality score vs. expert rating scatter
- `fig_tsne_comparison.png` — t-SNE embedding comparison (RAW vs. BIO)
- `fig_latency.png` — Per-component latency breakdown

---

## Dataset

98 video recordings of 10 yoga poses, each with raw video, extracted MediaPipe keypoints, and expert biomechanical annotations. See [`DATASET_CARD.md`](DATASET_CARD.md) for full documentation following Gebru et al. (2021).

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

---

## Research Questions

1. **How can biomechanical expertise be computationally encoded for real-time feedback?** The system operationalizes kinesiology knowledge through 30 expert-designed biomechanical features and pose-specific quality profiles, testing whether domain expertise transfers effectively into algorithmic form.

2. **Do expert-designed features match raw landmarks for classification while enabling quality assessment?** The ablation study demonstrates equivalent accuracy with 4.4x fewer dimensions, while the biomechanical representation additionally supports interpretable quality scoring.

3. **How should movement quality be modeled beyond binary classification?** The VAE's continuous quality score and pose-specific deviation detection challenge the pass/fail paradigm common in pose estimation research, correlating strongly (ρ = 0.958) with expert ratings.

---

## Project Structure

```
zenith-mvp/
├── server.py                    # FastAPI WebSocket gateway (heuristics, sequencer, coaching)
├── config.py                    # Centralized config from env vars + .env
├── zenith_brain.py              # Core ML pipeline (MediaPipe + RF + VAE + Bio)
├── biomechanical_features.py    # 30 expert-designed features + pose profiles
├── pose_foundations.py          # Heuristic coaching + correction vectors
├── pose_sequencer.py            # Guided yoga sequence state machine
├── vision_client.py             # Gemini Vision API (with mock fallback)
├── session_manager.py           # Session recording + persistence (thread-safe)
├── data_harvester.py            # High-quality frame collection (thread-safe)
├── vae_biomechanical.py         # Bio-VAE and c-VAE training
├── train_pose_classifier.py     # Random Forest training
├── .env.example                 # Configuration template
│
├── ZENith_Data/
│   ├── videos/                  # Raw .mov recordings
│   ├── keypoints/               # Extracted .npy files (frames × 33 × 4)
│   ├── annotations/             # Per-video JSON annotations
│   └── models/                  # Trained model weights
│
├── evaluation/
│   ├── train_and_evaluate.py    # Ablation study (4 conditions × 2 classifiers)
│   ├── quality_validation.py    # Quality score vs. expert rating validation
│   ├── generate_figures.py      # Publication-ready figures
│   ├── benchmark_latency.py     # Latency benchmarks
│   └── results/                 # JSON outputs + 6 publication-ready figures
│
├── zenith-web/                  # React 19 frontend (Vite + TypeScript + Tailwind)
│   └── src/
│       ├── App.tsx              # Root layout with error/connection banners
│       ├── components/
│       │   ├── VideoStage.tsx          # Webcam (with error/loading states)
│       │   ├── HUD.tsx                 # Pose, flow, quality, stability
│       │   ├── BiomechanicalPanel.tsx  # Bio quality + deviations + angles
│       │   ├── GenerativeCoach.tsx     # AI coach (idle/analyzing/result)
│       │   ├── SequenceBar.tsx         # Guided sequence progress
│       │   ├── SessionReport.tsx       # Session summary + timeline
│       │   ├── SessionControls.tsx     # Record, harvest, sequence
│       │   ├── GhostOverlay.tsx        # Skeleton + ghost rendering
│       │   └── ErrorBoundary.tsx       # Crash recovery
│       └── hooks/
│           ├── useZenithConnection.ts  # WebSocket state management
│           └── useZenithVoice.ts       # TTS speech synthesis
│
├── DATASET_CARD.md              # Gebru et al. dataset documentation
├── ANNOTATION_PROTOCOL.md       # Quality rating criteria and feature taxonomy
└── README.md                    # This file
```

---

## Stack and Requirements

| Component | Technology |
|-----------|------------|
| Pose estimation | MediaPipe Pose (33 landmarks) |
| Biomechanical features | NumPy, SciPy (Butterworth filter) |
| Quality scoring | TensorFlow / Keras (VAE, c-VAE) |
| Classification | scikit-learn (Random Forest, MLP) |
| Heuristic coaching | Rule-based corrections for all 10 poses with correction vectors |
| Guided sequences | 3 yoga flows (Warrior, Balance, Strength) |
| Frontend | React 19 / Vite / TypeScript / Tailwind |
| Backend | FastAPI / Uvicorn / WebSocket |
| Configuration | python-dotenv / env vars / structured logging |
| AI Coaching | Google Gemini API (optional, mock fallback) |
| Language | Python 3.10+ |

### Reproduction

```bash
# Clone and setup
git clone https://github.com/aperry938/zenith-mvp
cd zenith-mvp
conda create --name zenith python=3.11 -y
conda activate zenith
pip install -r requirements.txt

# Configure (optional — works without Gemini key using mock coaching)
cp .env.example .env
# Edit .env to add your GEMINI_API_KEY if desired

# Run the live application
python server.py
# In a second terminal:
cd zenith-web && npm install && npm run dev

# Run evaluation suite
cd evaluation
python train_and_evaluate.py    # Ablation study (~2 min)
python quality_validation.py    # Quality score validation (~1 min)
python generate_figures.py      # Publication figures
python benchmark_latency.py     # Latency benchmarks

# Retrain VAE on biomechanical features
python vae_biomechanical.py     # Trains standard + conditional VAE

# Generate dataset annotations
cd ZENith_Data/annotations
python generate_annotations.py
```

Model training notebook: [Google Colab](https://colab.research.google.com/drive/1DSYxlitGTFTivI2nsCgHxrPoP5nJK-5Y)

---

## Future Directions

- **Expanded pose vocabulary:** Train classifier on additional poses (Forward Fold, Cobra, Upward Salute) to enable full Sun Salutation sequences
- **Body-type adaptation:** Personalized biomechanical baselines that account for morphological variation
- **Velocity-based phase detection:** Replace fixed temporal hold timers with angular velocity thresholds for entry/hold/exit segmentation
- **Session analytics dashboard:** Historical session comparison, progress tracking, and weakness identification
- **Multi-person tracking:** Extend to group yoga classes with per-person quality feedback
- **Clinical rehabilitation:** Adapting the pipeline to physical therapy contexts

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
