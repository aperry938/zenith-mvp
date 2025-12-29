import json
import time
import os
from collections import Counter

STATS_FILE = "zenith_stats.json"

class SessionManager:
    """
    Manages session recording and long-term persistence (The Vault).
    """
    def __init__(self):
        # Current Session
        self.start_time = time.time()
        self.flow_scores = []
        self.stability_events = 0
        self.stability_seconds = 0.0
        self.poses_detected = Counter()
        
        # Lifetime Stats
        self.lifetime_duration = 0.0
        self.lifetime_sessions = 0
        self.lifetime_hacks = 0 # Future use
        
        self.load_history()

    def update(self, pose_label, flow_score, is_stable, fps):
        if flow_score is not None:
            self.flow_scores.append(flow_score)
        if is_stable and fps > 0:
            self.stability_seconds += (1.0 / fps)
        if pose_label:
            self.poses_detected[pose_label] += 1

    def log_stability_event(self):
        self.stability_events += 1

    def get_current_summary(self):
        """Returns stats for the CURRENT session."""
        duration = time.time() - self.start_time
        avg_flow = sum(self.flow_scores) / len(self.flow_scores) if self.flow_scores else 0
        top_pose = self.poses_detected.most_common(1)[0][0] if self.poses_detected else "None"
        
        return {
            "Duration": f"{int(duration)}s",
            "Avg Flow": f"{int(avg_flow)}",
            "Stability Events": self.stability_events,
            "Zone Time": f"{self.stability_seconds:.1f}s",
            "Top Pose": top_pose
        }

    def get_lifetime_summary(self):
        """Returns stats for ALL TIME."""
        # Include current session duration in display
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
            except Exception as e:
                print(f"Error loading stats: {e}")

    def save_session(self):
        """Saves current session data to history."""
        duration = time.time() - self.start_time
        
        # Only save if meaningful (e.g. > 10 seconds)
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
                print("Session saved.")
            except Exception as e:
                print(f"Error saving stats: {e}")
