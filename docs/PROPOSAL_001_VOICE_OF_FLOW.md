# ZENITH STRATEGIC PROPOSAL 001: THE VOICE OF FLOW

**STATUS:** PENDING ARCHITECT APPROVAL
**FROM:** ANTIGRAVITY LOOP (The Architect)

---

## 1. THE FOUNDATION CHECK (Assessment)
*   **Manifesto Alignment:** The `ZENITH_CORE_MANIFESTO.md` demands "Pedagogical Depth" and "Multimodal cues (Visual + TTS)".
*   **Current Reality:** `app_async_vae.py` provides a "Quality Score" (0-100) via VAE. This is a "Black Box" metric. It tells the user *that* they are wrong, but not *how* to fix it. The app is also silent, forcing the user to crane their neck to look at the screen, which ruins yoga form.
*   **Verdict:** Foundation is technically solid (Async/WebRTC/VAE), but **Experientially Hallow**. We are building a monitor, not a coach.

## 2. THE "GHOST IN THE MACHINE" (Insight)
*   **The Latent Potential:** We are already capturing 33 skeletal landmarks 30 times a second. We calculate angles in `app.py` (synchronous) but abandoned them in the superior `app_async_vae.py`.
*   **The Magic:** Imagine the user is in "Warrior II". They are tired. Their front knee drifts inward.
    *   *Current App:* Shows "Quality: 60%" (User is confused).
    *   *Zenith Vision:* A calm voice says: **"Open your right knee toward your pinky toe."**
*   **The Leap:** We don't need a massive LLM for this yet. We can "fake" the intelligence with a robust **Heuristic Feedback Engine** coupled with a non-blocking TTS worker. This brings the "Coach" to life immediately.

## 3. THE PROPOSAL (High-Leverage Next Step)

### **What: The "Voice of Flow" Architecture (Heuristic TTS Engine)**
Merge the geometric calculation logic from `app.py` into `app_async_vae.py` and attach a dedicated, non-blocking Text-to-Speech (TTS) worker thread.

### **Why: Defending the Choice**
1.  **High Impact/Low Cost:** We have the math (angles). We have the async structure. Adding TTS is a lightweight dependency that increases perceived value by 10x.
2.  **Solves "Neck Crane":** Yoga requires looking away from the screen. Audio feedback is the *only* valid interface for flow.
3.  **Differentiation:** This moves us instantly away from "Mirror" (Visual only) and "Alo Moves" (Video only) to "Zenith" (Interactive Audio).

### **The Goal: Transcendent Product**
To enable a "blind" practice where the user can trust the AI to guide them without looking at the device.

---

### **Implementation Sketch**
1.  **Refactor:** Port `calculate_angle` logic to `app_async_vae.py`.
2.  **Logic Layer:** Create `PoseFoundations.py` – a library of geometric truths (e.g., `WarriorII_RightKnee_MinAngle = 90`).
3.  **Audio Engine:** Implement a `TTSWorker` queue that prioritizes messages (don't overlap speech). "Debounce" feedback so it doesn't nag (e.g., waiting 5 seconds before repeating a correction).

**RECOMMENDATION:** EXECUTE IMMEDIATELY.
