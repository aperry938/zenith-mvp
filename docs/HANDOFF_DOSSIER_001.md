# ZENITH HANDOFF DOSSIER 001: VOICE OF FLOW

**FROM:** Antigravity (Lead Architect)
**TO:** Claude (The Evaluator/Builder)
**DATE:** 2025-12-29
**SNAPSHOT:** `Initial Zenith Snapshot: Secured Baseline`

---

## 1. THE MISSION BRIEF
**Objective:** Implement "The Voice of Flow" – a real-time, TTS-based heuristic coaching engine.
**Success Criteria:**
1.  User performs "Warrior II" incorrectly (e.g., knee buckled).
2.  App speaks *specific* correction (e.g., "Check your front knee") within 3 seconds.
3.  App speaks affirmation (e.g., "Better.") when fixed.
4.  Video FPS remains stable (>20 FPS).

## 2. ARCHITECTURAL DEFENSE
**Why Heuristics?** We are prioritizing Speed and Privacy over "General Intelligence." We do not need an LLM to tell us a knee angle is < 90 degrees.
**Why `pyttsx3`?** Locally running TTS avoids network latency. We want the feedback to feel immediate, like a teacher in the room.

## 3. THE ECOSYSTEM MAP
*   **`app_async_vae.py`:** This is the *only* application file you should care about. Ignore `app.py` (legacy).
*   **`pose_foundations.py` (NEW):** You must create this file. It should contain the "Truths" of Yoga (angles, thresholds). Keep it pure Python (no Streamlit dependencies) so we can unit test it.

## 4. THE HONESTY PROTOCOL (Risks)
*   **The "Nagging" Factor:** If the user is struggling, the AI might repeat "Fix your knee" every millisecond. You *must* implement a "Debounce" timer (e.g., silence for 5 seconds after a correction).
*   **Thread Safety:** `pyttsx3` can differ across OS (Mac vs Linux). Ensure the worker thread handles the run loop correctly to avoid "NSInternalInconsistencyException" on Mac.

## 5. AUDIT DIRECTIVES ("Red Team" Mandate)
**Execute these commands verbatim during your build:**
*   **"Global Ecosystem Scan:** Check for loose ends in imports."
*   **"Consistency & Hygiene:** Enforce naming conventions. **Scan for emojis and replace with our style icons/SVGs.** (Standardize this now)."
*   **"Mandatory Remediation:** If the TTS blocks the video feed, you must refactor to `multiprocessing` instead of `threading`."

## 6. CONTEXTUAL LOCK-IN
*   **Constraint:** Do not change the VAE weights or architecture. We are only adding a layer on top.
*   **Style:** Code must be cleaner than when you found it.

---

**"INSTRUCTIONS FOR THE AUDITOR (CLAUDE):"**
*   **The Rating:** Provide a 'Current State Health Rating' (0-100) *after* implementation.
*   **The Return Handoff:** Generate a summary of fixes/changes.

**"POST-HANDOFF PROCEDURE (For Antigravity):"**
*   Note to self: When the user brings Claude's output back, I will verify integrity, perform final GitHub verification (merge/clean), and then execute Phase 0-3 again.
