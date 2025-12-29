# ZENITH HANDOFF DOSSIER 005: THE STABILITY ENGINE

**FROM:** Antigravity (Lead Architect)
**TO:** Claude (The Evaluator/Builder)
**DATE:** 2025-12-29
**SNAPSHOT:** `Snapshot: Cycle 3 Proposal - The Stability Engine`

---

## 1. THE MISSION BRIEF
**Objective:** Implement "The Stability Engine" – Gamified rewards for prolonged stillness.
**Success Criteria:**
1.  **Detection:** Identify when the user is in a "Correct" state (no heuristic errors).
2.  **Duration:** Track how long this state persists.
3.  **Reward:** 
    *   3 Seconds -> Trigger "Locked" state.
    *   Visual: Draw a Halo (Ellipse) that changes color (Blue -> Gold).
    *   Audio: Speak "Locked" (via `pyttsx3`).

## 2. ARCHITECTURAL DEFENSE
**Why Gamification?** We need to incentivize *duration*, not just positioning.
**Why Ellipse?** It naturally frames the human body (which is roughly elliptical) better than a rectangle.

## 3. THE ECOSYSTEM MAP
*   **`app_async_vae.py`:** Add state tracking (`stability_start_time`, `is_locked`). logic.
*   **`pose_foundations.py`:** No changes needed (heuristics are already there).

## 4. THE HONESTY PROTOCOL (Risks)
*   **Flicker:** If the user jitters between Correct/Incorrect, the Halo will flash annoying.
*   **Mitigation:** The timer must reset immediately on error. This is a "Hardcore" mode.

## 5. AUDIT DIRECTIVES ("Red Team" Mandate - CLAUDE MUST FOLLOW ORDER)
**Execute these steps in this exact order:**
1.  **"Global Ecosystem Scan:** Verify `active_correction` logic from previous cycle is robust."
2.  **"Implementation:** Add the Timer/State logic."
3.  **"Visual Layer:** Implement `draw_stability_halo`."
4.  **"Audio Layer:** Add the 'Locked' trigger."
5.  **"Consistency Check:** Ensure the Halo color palette matches the 'Sci-Fi' theme (Cyan/Gold)."

## 6. CONTEXTUAL LOCK-IN
*   **Constraint:** Do not block the main thread.
*   **Constraint:** Maintain the active correction arrows (Visual Whispers) when *not* stable.

---

**"INSTRUCTIONS FOR THE AUDITOR (CLAUDE):"**
*   **The Rating:** Provide a 'Current State Health Rating' (0-100) *after* implementation.
*   **The Return Handoff:** Generate a summary of fixes/changes.

**"POST-HANDOFF PROCEDURE (For Antigravity):"**
*   Note to self: When the user brings Claude's output back, I will verify integrity, perform final GitHub verification (merge/clean), and then execute Phase 0-3 again.
