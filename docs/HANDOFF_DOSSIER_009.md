# ZENITH HANDOFF DOSSIER 009: THE PERSONA

**FROM:** Antigravity (Lead Architect)
**TO:** Claude (The Evaluator/Builder)
**DATE:** 2025-12-29
**SNAPSHOT:** `Snapshot: Cycle 5 Proposal - The Persona`

---

## 1. THE MISSION BRIEF
**Objective:** Implement "The Persona" – Dynamic variety in TTS feedback.
**Success Criteria:**
1.  **Variety:** The app should rarely say the exact same correction sentence twice in a row.
2.  **Structure:** Centralize strings in a `COACHING_LIBRARY`.
3.  **Positive Reinforcement:** Vary the "Locked" stability message.

## 2. ARCHITECTURAL DEFENSE
**Why Randomness?** It cheats the Turing Test. Low effort, high impact on "Soul" rating.

## 3. THE ECOSYSTEM MAP
*   **`pose_foundations.py`:** Add the `COACHING_LIBRARY`. Update `evaluate` to sample from it.
*   **`app_async_vae.py`:** Update the "Locked" trigger to use a variety list.

## 4. THE HONESTY PROTOCOL (Risks)
*   **Confusion:** Don't get too poetic. "Your knee is a blossoming flower" is confusing. Keep it direct: "Fix the knee."
*   **Mitigation:** Keep all variations strictly instructional.

## 5. AUDIT DIRECTIVES ("Red Team" Mandate - CLAUDE ORDER)
1.  **"Global Ecosystem Scan:** Check if text is used for anything other than voice (e.g. HUD)."
    *   *Note:* Using long random strings might break the HUD layout.
    *   *Fix:* Return a tuple `(hud_text, spoken_text)`. Keep HUD short ("KNEE"), make Voice long ("Please watch that front knee alignment").
2.  **"Implementation:** Refactor `PoseHeuristics` to support `(short, long)` return values."
3.  **"App Update:** Update `process_frame` to handle the tuple."

---

**"INSTRUCTIONS FOR THE AUDITOR (CLAUDE):"**
*   **The Rating:** Provide a 'Current State Health Rating' (0-100) *after* implementation.
*   **The Return Handoff:** Generate a summary of fixes/changes.

**"POST-HANDOFF PROCEDURE (For Antigravity):"**
*   **AUTO-MERGE.**
