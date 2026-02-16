# ZENITH HANDOFF DOSSIER 007: THE FLUIDITY METRIC

**FROM:** Antigravity (Lead Architect)
**TO:** Claude (The Evaluator/Builder)
**DATE:** 2025-12-29
**SNAPSHOT:** `Snapshot: Cycle 4 Proposal - The Fluidity Metric`

---

## 1. THE MISSION BRIEF
**Objective:** Implement "The Flow Score" – A metric for movement smoothness (Inverse Velocity Variance/Jerk).
**Success Criteria:**
1.  **Calculation:** Differentiate keypoints over time to get Velocity.
2.  **Metric:** `FlowScore = 100 - (JerkFactor * Sensitivity)`.
3.  **Visual:** Render a "Flow Bar" or "Wave" distinct from the Stability Halo.
4.  **Feedback:** If Flow Score drops < 50, provide a visual feedback (e.g., Bar turns Red).

## 2. ARCHITECTURAL DEFENSE
**Why Fluidity?** It bridges the gap between poses.
**Why Inverse Velocity?** True "Jerk" (derivative of acceleration) is noisy with webcams. Maintaining a consistent (or slowly changing) velocity is a good enough proxy for "Grace" in MVP.

## 3. THE ECOSYSTEM MAP
*   **`app_async_vae.py`:** Add `prev_landmarks` state. Add `calculate_flow` function. Render Flow Bar.

## 4. THE HONESTY PROTOCOL (Risks)
*   **Noise:** Webcams jitter. This creates fake "Jerk."
*   **Mitigation:** You MUST apply a smoothing filter (EMA) to the landmarks OR the score before displaying it.

## 5. AUDIT DIRECTIVES ("Red Team" Mandate - CLAUDE ORDER)
**Execute these steps in this exact order:**
1.  **"Global Ecosystem Scan:** Ensure no regression in Stability Engine or Visual Whispers."
2.  **"Implementation:** Add `prev_landmarks` and `flow_history` logic."
3.  **"Math Layer:** Implement `calculate_flow_score` with noise smoothing."
4.  **"Visual Layer:** Draw the Flow Bar (Bottom of screen?)."
5.  **"Consistency Check:** Ensure the Flow Bar style (Neon) matches the Halo."

## 6. CONTEXTUAL LOCK-IN
*   **Constraint:** Do not remove the Halo. The Flow Bar lives alongside it.

---

**"INSTRUCTIONS FOR THE AUDITOR (CLAUDE):"**
*   **The Rating:** Provide a 'Current State Health Rating' (0-100) *after* implementation.
*   **The Return Handoff:** Generate a summary of fixes/changes.

**"POST-HANDOFF PROCEDURE (For Antigravity):"**
*   Note to self: When the user brings Claude's output back, I will verify integrity, perform final GitHub verification (merge/clean), and then execute Phase 0-3 again.
