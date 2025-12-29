# ZENITH HANDOFF DOSSIER 003: VISUAL WHISPERS

**FROM:** Antigravity (Lead Architect)
**TO:** Claude (The Evaluator/Builder)
**DATE:** 2025-12-29
**SNAPSHOT:** `Snapshot: Cycle 2 Assessment & Visual Whispers Proposal`

---

## 1. THE MISSION BRIEF
**Objective:** Implement "Visual Whispers" – AR vector overlays for form correction.
**Success Criteria:**
1.  User performs pose correctly -> No arrows.
2.  User performs pose incorrectly -> Arrow appears on video feed (Cyan color) indicating direction of fix.
3.  FPS remains > 20.

## 2. ARCHITECTURAL DEFENSE
**Why Vectors?** Angles are abstract. Lines are concrete.
**Why OpenCV?** `cv2.arrowedLine` is the most efficient way to draw on a frame. No heavy react-three-fiber or HTML overlay needed yet.

## 3. THE ECOSYSTEM MAP
*   **`pose_foundations.py`:** Update the `evaluate` return signature. It currently returns a string. It must now return a `Correction` object (compatible with old string consumers if possible, or refactor consumers).
*   **`app_async_vae.py`:** Update the HUD drawing logic to render the vector.

## 4. THE HONESTY PROTOCOL (Risks)
*   **Clutter:** If we draw too many arrows, it looks like a cockpit. Limit to 1 arrow at a time (the "primary" error).
*   **Visibility:** Ensure the arrow color (Cyan: `(255, 255, 0)` in BGR) contrasts with the average home background.

## 5. AUDIT DIRECTIVES ("Red Team" Mandate)
*   **"Global Ecosystem Scan:** Ensure `pose_foundations.py` doesn't break `app_async_vae.py` imports."
*   **"Consistency & Hygiene:** Any new drawing code must be encapsulated (e.g., `draw_correction_arrow(img, vector)`), not verified inline."
*   **"Mandatory Remediation:** If the arrow flickers, implement a smoothing buffer."

## 6. CONTEXTUAL LOCK-IN
*   **Constraint:** Maintain the `pyttsx3` voice logic. This is an *addition*, not a replacement.

---

**"INSTRUCTIONS FOR THE AUDITOR (CLAUDE):"**
*   **The Rating:** Provide a 'Current State Health Rating' (0-100) *after* implementation.
*   **The Return Handoff:** Generate a summary of fixes/changes.

**"POST-HANDOFF PROCEDURE (For Antigravity):"**
*   Note to self: When the user brings Claude's output back, I will verify integrity, perform final GitHub verification (merge/clean), and then execute Phase 0-3 again.
