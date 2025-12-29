# ZENith Core Manifesto: The Living Blueprint

## 1. The Mission Brief
**Identity:** ZENith is "Your Personal Kinesiology-Powered Yoga Coach."
**Core Value:** We bridge the gap between expensive, hardware-gated home gyms (Tonal, Mirror) and generic, low-intelligence video libraries (Alo Moves). We provide true, biomechanically aware coaching using only the user's smartphone.
**North Star:** To create an AI that is not just a rep counter, but an empathetic, ethically-grounded, and scientifically rigorous guide that adapts to the user's unique physiology.

## 2. Core Principles (The Philosophical Constraints)
1.  **Pedagogical Depth over Binary Corrections:** We do not just say "Wrong." We explain *why* based on kinesiology (e.g., "Ground through the heel to protect the knee").
2.  **True Personalization:** The model adapts to the user's flexibility and injury history. We do not enforce a "universal ideal" that causes injury.
3.  **Ethical & Inclusive AI:** We actively fight algorithmic bias. Our models must work for all skin tones, body types, and clothing choices. This is our scientific contribution.
4.  **Fluidity is Mastery:** We value the transition (The Flow/Vinyasa) as much as the static pose. We measure "Jerk" and motion quality, not just final alignment.

## 3. The Workflow (User Journey)
1.  **The "Cold Start":** User onboards. No hardware required.
2.  **The Assessment:** AI analyzes simple movements to baseline flexibility and identify risk factors.
3.  **The Practice:**
    *   **Visual:** 3D Avatar (Generative/GAN-based) demonstrates perfect form *for the user's body type*.
    *   **Feedback:** Real-time multimodal cues (Visual overlays + TTS Verbal corrections).
4.  **The Reflection:** Post-session analytics focusing on "Fluidity Score" and "Mobility Improvements," not just calories burned.

## 4. Key Innovations (Implemented & Planned)
*   **[MVP] Static Pose Classifier:** Random Forest on MediaPipe keypoints.
*   **[MVP] Quality VAE:** Variational Autoencoder for "weirdness" detection in poses.
*   **[BETA] Voice of Flow (TTS Feedback):** Heuristic Text-to-Speech engine (pyttsx3) for real-time form correction.
*   **[BETA] Visual Whispers (AR Cues):** Augmented Reality vector overlays (OpenCV) for precise directional guidance.
*   **[BETA] Stability Engine (Gamified Stillness):** Visual Halo + Audio "Lock-In" reward for holding perfect form.
*   **[BETA] The Fluidity Metric (Flow Score):** Real-time measurement of movement smoothness (Inverse Jerk).
*   **[BETA] The Persona (Dynamic Coaching):** Varied sentence structures and "empathic" feedback.
*   **[BETA] The Record (Session Memory):** Analytics summary (Total Frames, Average Flow, Stability Count) after session.
*   **[BETA] The Sequencer (Automated Flow):** Logic to guide users through structured sequences (e.g., Sun Salutation A).
*   **[BETA] The Unification (UI System):** Centralized Design System class for all visual overlays (HUD, Halo, Flow Bar).
*   **[BETA] The Vault (Persistence):** JSON-based session history and long-term analytics.
*   **[BETA] "Yoga-Diverse" Dataset (Harvesting):** Automatic collection of high-quality frames for training.
*   **[BETA] The Gallery (Session Review):** In-app review of harvested "Best Moments" (Images).
*   **[BETA] The Dynamic Voice (Contextual TTS):** Voice feedback reacting to speed/flow metrics.
*   **[BETA] The Sage (Vision Analysis):** On-demand static frame analysis via Multimodal LLM (Plumbing).
*   **[BETA] The Holo-Deck (Alignment Grid):** Visual grid overlay for spatial reference and self-correction.
*   **[BETA] The Ghost (VAE Reconstruction):** Overlay of the "Idealized" VAE output on the user.
*   **[COMPLETED] The Architect (Core Refactor):** Modularization of the main application logic.
*   **[BETA] The Connection (Gemini API):** Real-time Vision-LLM integration for "The Sage".
*   **[BETA] The Dream (Latent Interpolation):** Generative movement via random walks in VAE latent space.
*   **[BETA] The Identity (Ghost Body):** Filled geometric visualization for the VAE Ghost.
*   **[BETA] The Voice of the Sage (TTS Integration):** Vocalization of Gemini-based coaching.
*   **[BETA] The Mirror (Cinematic UI):** Cinematic color grading and adaptive overlays.
*   **[BETA] The Signal (Async Brain):** Threaded AI processing for high-FPS UI.
*   **[BETA] The Pulse (Smoother Flow):** Robust velocity smoothing for better 'Flow' metrics.
*   **[BETA] The API Gateway (FastAPI Server):** WebSocket-based AI inference server.
*   **[BETA] The Bridge (Client Test):** End-to-end verification of the fullstack loop.
*   **[BETA] The Face (React Initialization):** Modern Vite/React frontend setup.
*   **[BETA] The Retina (React Video Stream):** Bi-directional Webcam -> Server -> JSON stream.
*   **[BETA] The HUD (React UI Overlay):** Modularize UI components and enhance visuals.
*   **[BETA] The Nervous System (State Management):** Robust state handling for connection/metrics.
*   **[BETA] The Aesthetic (Tailwind Integration):** Migrate to Utility-First CSS for rapid styling.
*   **[BETA] The Wisdom (Fullstack Sage):** Re-integrate Gemini Vision for real-time coaching.
*   **[BETA] The Voice (React TTS):** Text-to-Speech via Web Speech API for Sage guidance.
*   **[BETA] The Manuscript (Documentation):** Update docs for React/FastAPI architecture.
*   **[BETA] The Polish (Frontend Optimization):** Error boundaries, robust connection handling, and UI spit-shine.
*   **[BETA] The Mirage (Client-Side Visuals):** Rendering the VAE 'Ghost' and Skeleton overlays in React.
*   **[BETA] The Generator (Placeholder AI Coach):** Visual neural presence of the AI in the interface.
*   **[BETA] The Teacher (Interactive Flow):** Wiring the Pose Sequencer to the React UI for structured classes.
*   **[BETA] The Voice of Flow (Sequencer TTS):** Giving the sequencer a voice for audible guidance.
*   **[PLANNED] The Oracle (Proactive Vision):** Automated, periodic Multimodal LLM feedback during poses.

## 5. Technical Architecture
*   **Input:** Single-camera feed (Smartphone).
*   **Core Engine:** MediaPipe (Keypoints) -> Custom Regressors/Classifiers.
*   **Future Engine:** Multi-camera 'Studio' setup for ground truth -> Distilled to single-camera model.
*   **Social Layer:** Meta Quest 3 Integration (Movement SDK) for avatar-based group classes.

## 6. The Concept Vault (Backlog)
*   **"Bio-Hacking" Integration:** Connect with WHOOP/Oura API. If Recovery is low, AI automatically suggests Restorative Yoga instead of Vinyasa Power.
*   **The "Practice Story":** Daily narrative summaries instead of raw charts.
*   **Haptic Corrections:** Watch buzzes on the wrist that needs adjustment.
*   **Synthetic Group Classes:** VR Environment where remote users practice together as avatars.
