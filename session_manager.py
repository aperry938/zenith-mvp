import json
import time
import os
import threading
from collections import Counter
from config import setup_logging

logger = setup_logging("zenith.session")
STATS_FILE = "zenith_stats.json"
SESSIONS_FILE = "zenith_sessions.json"
MAX_SESSIONS = 50

class SessionManager:
    """
    Manages session recording and persistence. Thread-safe.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.recording = False

        # Current Session
        self.start_time = time.time()
        self.flow_scores = []
        self.stability_events = 0
        self.stability_seconds = 0.0
        self.poses_detected = Counter()
        self.heuristic_correction_count = 0
        self.peak_flow = 0.0
        self.peak_quality = 0.0
        self.pose_timeline = []

        # Lifetime Stats
        self.lifetime_duration = 0.0
        self.lifetime_sessions = 0

        self.load_history()

    def reset(self):
        with self._lock:
            self.start_time = time.time()
            self.flow_scores = []
            self.stability_events = 0
            self.stability_seconds = 0.0
            self.poses_detected = Counter()
            self.heuristic_correction_count = 0
            self.peak_flow = 0.0
            self.peak_quality = 0.0
            self.pose_timeline = []

    def update(self, pose_label, flow_score, is_stable, fps, bio_quality=None):
        with self._lock:
            if flow_score is not None:
                self.flow_scores.append(flow_score)
                if flow_score > self.peak_flow:
                    self.peak_flow = flow_score
            if bio_quality is not None and bio_quality > self.peak_quality:
                self.peak_quality = bio_quality
            if is_stable and fps > 0:
                self.stability_seconds += (1.0 / fps)
            if pose_label:
                self.poses_detected[pose_label] += 1
                # Sample timeline every ~1s (every 30 frames at 30fps)
                total_frames = sum(self.poses_detected.values())
                if total_frames % 30 == 0:
                    elapsed = time.time() - self.start_time
                    self.pose_timeline.append({
                        "t": round(elapsed, 1),
                        "pose": pose_label,
                        "quality": round(bio_quality, 1) if bio_quality else 0,
                    })

    def log_heuristic_correction(self):
        with self._lock:
            self.heuristic_correction_count += 1

    def log_stability_event(self):
        with self._lock:
            self.stability_events += 1

    def get_current_summary(self):
        with self._lock:
            duration = time.time() - self.start_time
            avg_flow = sum(self.flow_scores) / len(self.flow_scores) if self.flow_scores else 0
            top_pose = self.poses_detected.most_common(1)[0][0] if self.poses_detected else "None"

            return {
                "Duration": f"{int(duration)}s",
                "Avg Flow": f"{int(avg_flow)}",
                "Stability Events": self.stability_events,
                "Zone Time": f"{self.stability_seconds:.1f}s",
                "Top Pose": top_pose,
                "Peak Flow": f"{int(self.peak_flow)}",
                "Peak Quality": f"{int(self.peak_quality)}",
                "Corrections": self.heuristic_correction_count,
                "Pose Timeline": self.pose_timeline[-60:],  # Last 60 samples
            }

    def get_lifetime_summary(self):
        current_duration = time.time() - self.start_time
        total_sec = self.lifetime_duration + current_duration

        hours = int(total_sec // 3600)
        minutes = int((total_sec % 3600) // 60)

        return {
            "Total Time": f"{hours}h {minutes}m",
            "Sessions": self.lifetime_sessions + 1
        }

    def load_history(self):
        if os.path.exists(STATS_FILE):
            try:
                with open(STATS_FILE, 'r') as f:
                    data = json.load(f)
                    self.lifetime_duration = data.get("total_duration", 0.0)
                    self.lifetime_sessions = data.get("total_sessions", 0)
                logger.debug(f"Loaded history: {self.lifetime_sessions} sessions")
            except Exception as e:
                logger.error(f"Error loading stats: {e}")

    def save_session(self):
        with self._lock:
            duration = time.time() - self.start_time

            if duration > 10:
                self.lifetime_duration += duration
                self.lifetime_sessions += 1

                data = {
                    "total_duration": self.lifetime_duration,
                    "total_sessions": self.lifetime_sessions,
                    "last_session_date": time.strftime("%Y-%m-%d %H:%M:%S")
                }

                try:
                    with open(STATS_FILE, 'w') as f:
                        json.dump(data, f, indent=4)
                    self._save_session_record(duration)
                    logger.info(f"Session saved ({int(duration)}s)")
                except Exception as e:
                    logger.error(f"Error saving stats: {e}")

    def _save_session_record(self, duration):
        """Append individual session record to sessions file."""
        avg_flow = sum(self.flow_scores) / len(self.flow_scores) if self.flow_scores else 0
        top_pose = self.poses_detected.most_common(1)[0][0] if self.poses_detected else "None"

        record = {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": int(duration),
            "avg_flow": int(avg_flow),
            "peak_flow": int(self.peak_flow),
            "peak_quality": int(self.peak_quality),
            "corrections": self.heuristic_correction_count,
            "top_pose": top_pose,
            "poses": dict(self.poses_detected),
        }

        sessions = []
        if os.path.exists(SESSIONS_FILE):
            try:
                with open(SESSIONS_FILE, 'r') as f:
                    sessions = json.load(f)
            except Exception:
                sessions = []

        sessions.append(record)
        sessions = sessions[-MAX_SESSIONS:]

        try:
            with open(SESSIONS_FILE, 'w') as f:
                json.dump(sessions, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving session record: {e}")

    @staticmethod
    def load_sessions():
        """Load all saved session records."""
        if os.path.exists(SESSIONS_FILE):
            try:
                with open(SESSIONS_FILE, 'r') as f:
                    return json.load(f)
            except Exception:
                return []
        return []
