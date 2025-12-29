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
*   **[PLANNED] Visual Whispers (AR Cues):** Augmented Reality vector overlays (OpenCV) for precise directional guidance.
*   **[PLANNED] Generative AI Coach:** Hybrid GAN (Asset) + Diffusion (Motion) model for the avatar.
*   **[PLANNED] Vision-LLM Feedback:** Multimodal LLM to generate natural language coaching cues.
*   **[PLANNED] Fluidity Metric:** Jerk-based calculus to score vinyasa flow.
*   **[PLANNED] "Yoga-Diverse" Dataset:** Proprietary dataset for debiasing pose estimation.

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
