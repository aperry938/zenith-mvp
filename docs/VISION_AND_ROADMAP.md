# ZENith — Vision & Roadmap

> Comprehensive product vision compiled from brainstorming notes.
> Cross-referenced with Zenith v2.6 (March 2026).

---

## Already Built (v2.6)

| Feature | Implementation | Version |
|---------|---------------|---------|
| Computer vision for yoga | MediaPipe Pose (33 3D landmarks) + 30 biomechanical features | v1.0 |
| AI coaching | Gemini Vision API + heuristic corrections (10 poses) | v2.5 |
| Audio coach | Browser TTS via useZenithVoice hook | v2.2 |
| Data analysis | VAE quality scoring (ρ=0.958), session reports | v1.0 |
| Cueing/sequencing | 4 guided flows with 8s hold timers, auto-advance | v2.3 |
| Ghost silhouette | VAE-reconstructed ideal form overlay (cyan) | v2.2 |
| Fluidity rating | Angular velocity-based flow score (0-100), color-coded | v2.4 |
| Coach phrase variation | Persona feature — randomized phrases prevent repetition | v2.6 |
| Biomechanical analysis | 30 expert-designed features, 4.4× dimensionality reduction | v1.0 |
| Stability detection | LOCKED/STEADY/MOVING badges with debounce | v2.3 |
| Session persistence | JSON file storage, session history UI | v2.5 |

---

## Near-Term Buildable (extends current architecture)

### Streak & Achievement System
- Consecutive-day streaks, best streak, total sessions/time
- Achievement badges: "First Session", "7-Day Streak", "Perfect Flow", "Locked 30s"
- Personal records: longest hold, highest quality, best flow

### Cross-Session Weakness/Strength Tracking
- Per-pose quality trends across sessions
- Most-corrected features → targeted recommendations
- Asymmetry detection (L/R comparison)
- "Your Downward Dog quality declined 12% over 4 sessions"

### Multiple Coach Personas
- **The Sage** — philosophical, wise: "Find stillness in the tension"
- **The Scientist** — biomechanical, precise: "Rotate femur 10° externally"
- **The Warrior** — strict, motivational: "Deeper. Don't settle."
- **The Flow** — sensual, poetic: "Let your spine cascade like water"
- **The Traditional** — Sanskrit, classical: "Eka, dvi, trini... hold"
- User selects persona, persisted across sessions

### Breath Cuing During Sequences
- "Inhale... arms up" / "Exhale... fold forward" at transitions
- Sanskrit counting option: "Eka... dvi... trini... chatvari... pancha"
- Visual breath indicator (inhale/exhale arrows) in HUD

### Adaptive Intensity Settings
- Gentle (wider thresholds, fewer corrections)
- Standard (current behavior)
- Intense (strict thresholds, more corrections)
- Scales angle thresholds by ±15%

### Exercise Prescriptions
- Tight hips → Pigeon prep holds
- Weak core → Plank progressions
- Poor balance → Tree variations
- Based on M.S. Kinesiology rehab/prehab principles

### More Poses (requires training data)
Priority: Forward Fold, Cobra, Child's Pose, Bridge, Low Lunge
Future: Chaturanga, Upward Dog, Pigeon, Camel, Headstand

### Session Export
- Printable HTML or CSV with full metrics, pose timeline, corrections log

---

## Platform Features (requires significant new infrastructure)

### Video Demonstration Generation
- AI-generated video demonstrations for each pose
- Videos showing fluidity/transitions between poses
- Photo references for alignment

### Strength & Conditioning Coach Mode
- Separate mode focused on yoga-adjacent S&C work
- Bodyweight progressions, mobility drills, conditioning circuits

### Language Support
- Multilingual coaching (Spanish, Portuguese, French, German, Japanese)
- Sanskrit terminology throughout
- Language selector in settings

### Class History with Replay
- Relive favorite past sessions
- Class favorites system
- Share sessions

### Dynamic Sequence Builder
- User-created custom flows
- Difficulty levels per sequence
- Transition coaching between poses
- Music/ambient sound during practice

### Adaptive Fitness/Levels
- Beginner → Intermediate → Advanced auto-progression
- Adjust corrections and expectations by level
- "Can show a class following along, but are actually in very different classes"

### Social Layer
- Socializing pre/post class option
- Open to socialize/network with "join me" option
- Friend requests
- In-application call feature (voice and video)
- Send friend request only to people met in virtual session

### Gamification
- Unlockable environments/backdrops
- 5+ teacher personas (base and premium)
- Base and premium tier versions
- Leaderboards, challenges, weekly goals

---

## VR/AR Vision (long-term, requires hardware R&D)

### Core VR Concept
- VR yoga with synthetic and real groups
- Unique backdrop environments (beach, mountain, temple, space)
- Ghost silhouette for cuing in 3D space
- VR visual cues: arrows, angles, adjustment silhouette for perfect posture
- Meditative practices with visual hypnotics
- Integrate more participants via video

### VR Hardware
- Goggles + camera integration
- Wireless 5-camera cross hardware system
  - Multi-function: chandelier mount, retractable cameras for ceiling height adjustment
  - Camera adjusts for common yogi positions (not exactly centered on mat)
- Vertical camera calculation for full body coverage

### Live + Virtual Hybrid
- Live classrooms with cameras for VR join
- Central camera on instructor (live class + virtual attendees)
- Powerful in-class visual for representing virtual attendees
- Sign up for on-demand classes that launch based on virtual "waiting room" limits

### VR Yoga Teacher Training
- One-on-one training sessions
- Various size groups
- AI teaching staff, human teacher, hybrid staff
- Opportunity to virtually teach classes of various sizes

### AI Avatar System
- Program your AI avatar prior to use with others
- Optimal appearance/image
- Create avatar experience/technicality settings
- Goal settings per avatar

### VR-Specific Features
- Video game philosophy: invite friends, solo, multiplayer, online (open/closed group)
- Could be the beginning for AETHERA (app or LLM)

---

## Business / Partnership / Marketing

### Data Strategy
- Create synthetic data to train models
- Partner with yoga platforms for data access
- During video data collection: train AI on corrections, diverse users, unique movement habits
- Prioritize yoga teachers initially for better accuracy
- Free 1-year membership for data contributors at launch
- Obscure training materials for copyright protection

### Studio Partnerships
- Place prototype hardware in yoga studio — free class for participants
- Coupons for free VR session
- Capture movement personalities and nuances for dataset variety

### Business Model
- Base tier: classes, AI coach/tracking
- Premium tier: trainings, group sessions, social features
- For AI community: program your AI avatar prior to use with others

### Marketing Materials
- Print dark and light business cards
- Create professional pitch deck for the project
- Technical execution plan for investors

---

## Technical Notes

### Priorities
1. VR/AR is the ultimate vision
2. Computer vision is the foundation (built)
3. Avatar coach system spans: philosophy, ashtanga, vinyasa, yin, sequencing, biomechanics, anatomy/physiology, rehab
4. Distill expert knowledge into text to feed AI

### Architecture Principles
- Generative AI Coaching + Advanced Biomechanical Analysis + Immersive Social Layer
- Real-time feedback loop (< 1ms latency achieved)
- Expert knowledge embedded in heuristics, not just LLM prompts
- Modular: coaching library decoupled from analysis engine

### Model Training Notes
- Generate professional plan for model training
- Best models for biomechanical framework
- Sign up for on-demand training data capture sessions
- Capture diverse body types, flexibility levels, experience levels

---

*Last updated: March 19, 2026*
*Cross-referenced with Zenith v2.6*
