"""
Streak and achievement tracking for Zenith.
Computes consecutive-day streaks and unlockable achievements.
"""
import json
from datetime import datetime, timedelta

SESSIONS_FILE = "zenith_sessions.json"


class StreakTracker:
    def __init__(self):
        self.sessions = self._load_sessions()

    def _load_sessions(self):
        try:
            with open(SESSIONS_FILE, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def get_stats(self):
        """Returns streak data and achievements."""
        dates = sorted(set(s.get('date', '')[:10] for s in self.sessions if s.get('date')))

        current_streak = self._compute_current_streak(dates)
        best_streak = self._compute_best_streak(dates)
        total_sessions = len(self.sessions)
        total_minutes = sum(s.get('duration', 0) / 60 for s in self.sessions)

        # Personal records
        peak_flow = max((s.get('peak_flow', 0) for s in self.sessions), default=0)
        peak_quality = max((s.get('peak_quality', 0) for s in self.sessions), default=0)

        achievements = self._compute_achievements(
            total_sessions, current_streak, best_streak,
            total_minutes, peak_flow, peak_quality
        )

        return {
            'current_streak': current_streak,
            'best_streak': best_streak,
            'total_sessions': total_sessions,
            'total_minutes': round(total_minutes, 1),
            'peak_flow': round(peak_flow, 1),
            'peak_quality': round(peak_quality, 1),
            'achievements': achievements,
        }

    def _compute_current_streak(self, dates):
        if not dates:
            return 0
        today = datetime.now().strftime('%Y-%m-%d')
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        if dates[-1] != today and dates[-1] != yesterday:
            return 0

        streak = 1
        for i in range(len(dates) - 1, 0, -1):
            d1 = datetime.strptime(dates[i], '%Y-%m-%d')
            d2 = datetime.strptime(dates[i - 1], '%Y-%m-%d')
            if (d1 - d2).days == 1:
                streak += 1
            else:
                break
        return streak

    def _compute_best_streak(self, dates):
        if not dates:
            return 0
        best = 1
        current = 1
        for i in range(1, len(dates)):
            d1 = datetime.strptime(dates[i], '%Y-%m-%d')
            d2 = datetime.strptime(dates[i - 1], '%Y-%m-%d')
            if (d1 - d2).days == 1:
                current += 1
                best = max(best, current)
            else:
                current = 1
        return best

    def _compute_achievements(self, total, current, best, minutes, flow, quality):
        all_achievements = [
            {'id': 'first_session', 'name': 'First Session', 'desc': 'Complete your first practice', 'unlocked': total >= 1},
            {'id': 'streak_3', 'name': '3-Day Streak', 'desc': 'Practice 3 days in a row', 'unlocked': best >= 3},
            {'id': 'streak_7', 'name': 'Week Warrior', 'desc': 'Practice 7 days in a row', 'unlocked': best >= 7},
            {'id': 'streak_30', 'name': 'Monthly Master', 'desc': 'Practice 30 days in a row', 'unlocked': best >= 30},
            {'id': 'sessions_10', 'name': 'Dedicated', 'desc': 'Complete 10 sessions', 'unlocked': total >= 10},
            {'id': 'sessions_30', 'name': 'Committed', 'desc': 'Complete 30 sessions', 'unlocked': total >= 30},
            {'id': 'hour_total', 'name': 'Hour of Practice', 'desc': 'Accumulate 60 minutes total', 'unlocked': minutes >= 60},
            {'id': 'perfect_flow', 'name': 'Perfect Flow', 'desc': 'Achieve 100 flow score', 'unlocked': flow >= 100},
            {'id': 'high_quality', 'name': 'Form Master', 'desc': 'Achieve 95+ quality score', 'unlocked': quality >= 95},
        ]
        return all_achievements
