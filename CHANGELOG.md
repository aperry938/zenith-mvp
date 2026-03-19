# ZENith Changelog

## [3.0.0] - 2026-03-19

### Vision & Roadmap
- Created comprehensive `docs/VISION_AND_ROADMAP.md` organizing all product notes into Already Built / Near-Term / Platform / VR Vision / Business categories

### Streak & Achievement System (Z1)
- New `streak_tracker.py` computing consecutive-day streaks, best streak, total sessions/minutes, personal records
- 9 achievements: First Session, 3/7/30-Day Streak, 10/30 Sessions, Hour of Practice, Perfect Flow, Form Master
- `/api/streaks` endpoint
- Flame badge in header showing current streak
- Achievements tab in session history

### Cross-Session Progress Tracking (Z2)
- New `progress_tracker.py` aggregating pose quality trends across sessions
- Identifies top 3 strengths (quality >80) and top 3 weaknesses (high corrections + low quality)
- Overall quality and flow trend detection (improving/stable/declining)
- `/api/progress` endpoint
- "My Progress" tab in session history with trend arrows and color-coded badges

### 5 Coach Personas (Z3)
- Expanded coaching library with 5 distinct teaching styles:
  - **The Sage** — philosophical, wise ("Find stillness in the tension")
  - **The Scientist** — biomechanical, precise ("Rotate femur 10° externally")
  - **The Warrior** — strict, motivational ("Deeper! Don't hold back.")
  - **The Flow** — sensual, poetic ("Let your spine cascade like water")
  - **The Traditional** — Sanskrit, classical ("Sthira sukham asanam")
- 19 correction keys × 5 personas + persona-specific positive feedback
- PersonaSelector component with WebSocket switching
- Persisted to localStorage, backward-compatible default persona

### Breath Cuing (Z4)
- Inhale/exhale alternation during guided sequences
- Optional Sanskrit counting (Eka, Dvi, Trini, Chatvari, Pancha)
- Breath indicator in HUD: blue ▲ (inhale) / amber ▼ (exhale)

### Adaptive Intensity (Z5)
- 3 levels: Gentle (wider thresholds), Standard (default), Intense (strict)
- Scales heuristic angle thresholds by ±15%
- Intensity selector in SessionControls, WebSocket switching

### Exercise Prescriptions (Z6)
- PRESCRIPTION_MAP covering all 9 tracked poses with targeted corrective exercises
- Maps biomechanical weaknesses to specific drills (based on M.S. Kinesiology principles)
- "Your Prescription" cards in progress view

### Session Export (Z8)
- HTML print export with full session metrics, pose timeline, color coding
- "Export" button on session report using window.print() pattern

---

## [2.6.0] - 2025-03-19

### The Persona (Proposal 005)
- Varied coaching phrases preventing "alarm fatigue"
- 4-10 phrase variations per correction key
- Randomized selection via `PoseHeuristics.get_advice()`
- Decoupled HUD text (short) from spoken text (conversational)

### Session History & Persistence (v2.5)
- Session recording with timeline, metrics, corrections
- JSON persistence (50 max sessions)
- Session history modal with past session stats
- End-of-session report with metrics grid and pose timeline

### Guided Flows (Proposal 004, v2.3-2.4)
- 4 sequences: Warrior, Balance, Strength, Complete
- Auto-advance on 8s hold completion
- Sequence progress bar with next pose preview

### Stability Engine (Proposal 003, v2.3)
- LOCKED/STEADY/MOVING state detection
- Velocity-based stability badges
- 5-second debounce on repeated corrections

### Ghost Overlay & Skeleton (v2.2)
- VAE-reconstructed ideal form silhouette
- Correction vectors as cyan arrows
- Joint angle annotations on skeleton

### Core System (v1.0-2.0)
- MediaPipe Pose (33 3D landmarks)
- 30 biomechanical features (joint angles, ratios, symmetry, stability)
- Random Forest classification (100% accuracy, 10 poses)
- VAE quality scoring (ρ=0.958 with expert ratings)
- Real-time WebSocket streaming (<0.29ms latency)
- Gemini Vision optional coaching
- 10 supported poses with full heuristic profiles
