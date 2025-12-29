# ZENITH STRATEGIC PROPOSAL 002: VISUAL WHISPERS (Augmented Reality Cues)

**STATUS:** PENDING ARCHITECT APPROVAL
**FROM:** ANTIGRAVITY LOOP (Cycle 2)

---

## 1. THE FOUNDATION CHECK (Assessment)
*   **Current Reality:** We have implemented "Voice of Flow" (TTS). The app now speaks. This is a huge leap.
*   **The Gap:** While audio is great for broad corrections ("Bend your knee"), it lacks *precision*. A user might not know *which* way to bend. Also, noisy environments render TTS useless.
*   **Manifesto Check:** "Pedagogical Depth" requires multimodal cues. Currently, we have text ("POSE: WARRIOR II") and Audio. We are missing the "Visual Overlay" component described in the "Practice" section ("Visual overlays + TTS Verbal corrections").

## 2. THE "GHOST IN THE MACHINE" (Insight)
*   **The Latent Potential:** We have the vectors! In `pose_foundations.py`, we calculate angles. We know the direction of error.
*   **The Magic:** Imagine the user's knee is caving in.
    *   *Voice:* "Open your knee."
    *   *Visual:* A subtle, glowing **Green Arrow** (CV2 arrow) appears on the screen, drawn from the knee cap pointing outward, showing the *vector of correction*.
*   **The Leap:** This is "Augmented Reality Coaching" without the headset. It creates a "Heads-Up Display" (HUD) for the body.

## 3. THE PROPOSAL (High-Leverage Next Step)

### **What: "Visual Whispers" (Dynamic Vector Overlays)**
Upgrade `PoseHeuristics` to return not just a feedback string, but a **Correction Vector** (Start Point, End Point, Color). Pass this to `app_async_vae.py` to draw dynamic arrows on the video feed.

### **Why: Defending the Choice**
1.  **Multimodal Reinforcement:** Some learners are visual, some auditory. We need both.
2.  **Immediate Feedback:** An arrow is processed faster by the brain than a sentence.
3.  **Low Latency:** Drawing a line in OpenCV costs <1ms. It fits perfectly in our async loop.

### **The Goal: Transcendent Product**
The user feels like Iron Man. The suit (app) is analyzing and highlighting the world (their body) in real-time.

---

### **Implementation Sketch**
1.  **Refactor `pose_foundations.py`:** Update `evaluate` to return a `Correction` object (Text, VectorStart, VectorEnd).
2.  **Update `app_async_vae.py`:** Inside `process_frame`, if a `Correction` exists, use `cv2.arrowedLine` to draw the guidance on the frame.
3.  **Aesthetics:** Use "Sci-Fi" colors (Cyan for correction, Green for perfect).

**RECOMMENDATION:** EXECUTE IMMEDIATELY.
