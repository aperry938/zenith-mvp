"""
Cross-session progress tracking for Zenith.
Analyzes trends in pose quality, corrections, and biomechanical deviations.
Generates exercise prescriptions based on identified weaknesses.
"""
import json
from collections import defaultdict
from typing import Dict, List, Any

SESSIONS_FILE = "zenith_sessions.json"

# Maps weak poses to targeted exercises grounded in exercise science.
# Each entry addresses the primary biomechanical demands of the pose.
PRESCRIPTION_MAP: Dict[str, Dict[str, Any]] = {
    "Warrior II": {
        "weakness": "Hip flexibility and leg strength",
        "exercises": [
            "Low lunge holds (30s each side)",
            "Wall sit (3x30s)",
            "Standing hip circles (10 each direction)",
        ],
    },
    "Tree": {
        "weakness": "Single-leg balance",
        "exercises": [
            "Single-leg stand on uneven surface (30s each)",
            "Heel-to-toe walk (20 steps)",
            "Ankle circles with eyes closed (10 each)",
        ],
    },
    "Plank": {
        "weakness": "Core stability",
        "exercises": [
            "Forearm plank hold (3x20s)",
            "Dead bug (3x10 reps)",
            "Bird dog (3x8 each side)",
        ],
    },
    "Chair": {
        "weakness": "Quad strength and ankle mobility",
        "exercises": [
            "Wall sit with proper form (3x30s)",
            "Goblet squat (3x10)",
            "Calf raises (3x15)",
        ],
    },
    "Downward Dog": {
        "weakness": "Hamstring and shoulder flexibility",
        "exercises": [
            "Standing forward fold with bent knees (30s hold)",
            "Shoulder stretch at wall (30s each arm)",
            "Puppy pose (30s hold)",
        ],
    },
    "Triangle": {
        "weakness": "Lateral flexibility",
        "exercises": [
            "Standing side bend (10 each side)",
            "Gate pose holds (20s each side)",
            "IT band foam rolling (60s each side)",
        ],
    },
    "Extended Side Angle": {
        "weakness": "Lateral chain strength and hip mobility",
        "exercises": [
            "Side plank (3x15s each side)",
            "Cossack squat (3x6 each side)",
            "Standing lateral leg raise (3x10 each side)",
        ],
    },
    "High Lunge": {
        "weakness": "Hip flexor length and single-leg stability",
        "exercises": [
            "Half-kneeling hip flexor stretch (30s each side)",
            "Split squat (3x8 each side)",
            "Single-leg Romanian deadlift (3x6 each side)",
        ],
    },
    "Mountain Pose": {
        "weakness": "Postural alignment and body awareness",
        "exercises": [
            "Wall angel (3x10 reps)",
            "Chin tuck holds (3x10s)",
            "Single-leg balance with eyes closed (3x15s each)",
        ],
    },
}


class ProgressTracker:
    def __init__(self) -> None:
        self.sessions = self._load_sessions()

    def _load_sessions(self) -> List[Dict]:
        try:
            with open(SESSIONS_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def get_progress(self) -> Dict[str, Any]:
        if len(self.sessions) < 2:
            return {"insufficient_data": True, "min_sessions": 2}

        return {
            "pose_trends": self._analyze_pose_trends(),
            "top_strengths": self._identify_strengths(),
            "top_weaknesses": self._identify_weaknesses(),
            "quality_trend": self._quality_trend(),
            "flow_trend": self._flow_trend(),
            "prescriptions": self._generate_prescriptions(),
        }

    # ------------------------------------------------------------------
    # Pose-level analysis
    # ------------------------------------------------------------------

    def _analyze_pose_trends(self) -> List[Dict]:
        """Track quality per pose across sessions."""
        pose_data: Dict[str, List[Dict]] = defaultdict(list)

        for session in self.sessions:
            top_pose = session.get("top_pose", "")
            quality = session.get("peak_quality", 0)
            if top_pose:
                pose_data[top_pose].append({
                    "date": session.get("date", ""),
                    "quality": quality,
                })

        trends = []
        for pose, data_points in pose_data.items():
            if len(data_points) >= 2:
                recent = data_points[-1]["quality"]
                earlier = data_points[0]["quality"]
                change = recent - earlier
                trend = (
                    "improving" if change > 5
                    else "declining" if change < -5
                    else "stable"
                )
                trends.append({
                    "pose": pose,
                    "sessions_practiced": len(data_points),
                    "latest_quality": round(recent, 1),
                    "change": round(change, 1),
                    "trend": trend,
                })

        return sorted(trends, key=lambda x: x["sessions_practiced"], reverse=True)

    def _identify_strengths(self) -> List[str]:
        """Poses with consistently high quality (avg > 80)."""
        pose_qualities: Dict[str, List[float]] = defaultdict(list)

        for session in self.sessions:
            top_pose = session.get("top_pose", "")
            quality = session.get("peak_quality", 0)
            if top_pose:
                pose_qualities[top_pose].append(quality)

            # Also consider per-pose data from the poses dict
            poses = session.get("poses", {})
            corrections = session.get("corrections", 0)
            total_frames = sum(poses.values()) if poses else 0
            for pose, frames in poses.items():
                if pose == top_pose:
                    continue  # Already counted via peak_quality
                # Estimate quality: fewer corrections relative to practice time
                if total_frames > 0 and frames > 0:
                    proportion = frames / total_frames
                    # Poses with many frames but few corrections are strong
                    est_quality = max(0, 100 - (corrections * proportion * 10))
                    pose_qualities[pose].append(est_quality)

        strengths = []
        for pose, qualities in pose_qualities.items():
            avg = sum(qualities) / len(qualities)
            if avg > 80:
                strengths.append((pose, avg))

        strengths.sort(key=lambda x: x[1], reverse=True)
        return [pose for pose, _ in strengths[:3]]

    def _identify_weaknesses(self) -> List[str]:
        """Poses with most corrections or lowest quality (avg < 60)."""
        pose_qualities: Dict[str, List[float]] = defaultdict(list)
        pose_corrections: Dict[str, int] = defaultdict(int)

        for session in self.sessions:
            top_pose = session.get("top_pose", "")
            quality = session.get("peak_quality", 0)
            corrections = session.get("corrections", 0)

            if top_pose:
                pose_qualities[top_pose].append(quality)
                pose_corrections[top_pose] += corrections

            # Weight in poses by frame proportion
            poses = session.get("poses", {})
            total_frames = sum(poses.values()) if poses else 0
            for pose, frames in poses.items():
                if pose == top_pose:
                    continue
                if total_frames > 0 and frames > 0:
                    proportion = frames / total_frames
                    est_quality = max(0, 100 - (corrections * proportion * 10))
                    pose_qualities[pose].append(est_quality)

        # Score: lower average quality or higher correction rate = weaker
        weakness_scores: List[tuple] = []
        for pose, qualities in pose_qualities.items():
            avg = sum(qualities) / len(qualities)
            corr_rate = pose_corrections.get(pose, 0)
            # Combined weakness score: low quality + high corrections
            score = (100 - avg) + corr_rate * 2
            weakness_scores.append((pose, score))

        weakness_scores.sort(key=lambda x: x[1], reverse=True)
        return [pose for pose, _ in weakness_scores[:3]]

    # ------------------------------------------------------------------
    # Overall trends
    # ------------------------------------------------------------------

    def _quality_trend(self) -> str:
        """Overall quality trend across recent sessions."""
        if len(self.sessions) < 3:
            return "insufficient_data"
        recent_3 = [s.get("peak_quality", 0) for s in self.sessions[-3:]]
        earlier_3 = [s.get("peak_quality", 0) for s in self.sessions[:3]]
        avg_recent = sum(recent_3) / len(recent_3)
        avg_earlier = sum(earlier_3) / len(earlier_3)
        if avg_recent > avg_earlier + 5:
            return "improving"
        elif avg_recent < avg_earlier - 5:
            return "declining"
        return "stable"

    def _flow_trend(self) -> str:
        """Flow score trend across recent sessions."""
        if len(self.sessions) < 3:
            return "insufficient_data"
        recent_3 = [s.get("avg_flow", 0) for s in self.sessions[-3:]]
        earlier_3 = [s.get("avg_flow", 0) for s in self.sessions[:3]]
        avg_recent = sum(recent_3) / len(recent_3)
        avg_earlier = sum(earlier_3) / len(earlier_3)
        if avg_recent > avg_earlier + 5:
            return "improving"
        elif avg_recent < avg_earlier - 5:
            return "declining"
        return "stable"

    # ------------------------------------------------------------------
    # Prescriptions
    # ------------------------------------------------------------------

    def _generate_prescriptions(self) -> List[Dict]:
        """Map weak poses to targeted corrective exercises."""
        weaknesses = self._identify_weaknesses()
        prescriptions = []
        for pose in weaknesses[:3]:
            if pose in PRESCRIPTION_MAP:
                prescriptions.append({
                    "pose": pose,
                    "focus": PRESCRIPTION_MAP[pose]["weakness"],
                    "exercises": PRESCRIPTION_MAP[pose]["exercises"],
                })
        return prescriptions
