import time
import random

SEQUENCES = {
    "warrior_flow": {
        "name": "Warrior Flow",
        "description": "Standing strength and lateral flexibility",
        "poses": ["Mountain Pose", "Warrior II", "Triangle", "Extended Side Angle", "Mountain Pose"],
    },
    "balance_flow": {
        "name": "Balance Flow",
        "description": "Single-leg balance and lower body power",
        "poses": ["Mountain Pose", "Tree", "Chair", "High Lunge", "Crescent Lunge", "Mountain Pose"],
    },
    "strength_flow": {
        "name": "Strength Flow",
        "description": "Core and upper body endurance",
        "poses": ["Mountain Pose", "Plank", "Downward Dog", "Mountain Pose"],
    },
}

class PoseSequencer:
    """
    Manages the state of a guided yoga sequence.
    Tracks current pose index, duration held, and transitions.
    """
    HOLD_DURATION = 8.0
    def __init__(self, sequence_key="strength_flow"):
        seq = SEQUENCES.get(sequence_key, SEQUENCES["strength_flow"])
        self.sequence_name = seq["name"]
        self.sequence = list(seq["poses"])
        self.current_index = 0
        self.last_transition_time = time.time()
        self.pose_start_time = None
        self.completed = False
        self._current_announcement = f"Let's begin {self.sequence_name}. Start with {self.sequence[0]}."

        # Oracle State
        self.analysis_triggered_for_current_pose = False

    def get_current_goal(self):
        if self.completed:
            return "Sequence Complete!"
        return self.sequence[self.current_index]

    def get_next_goal(self):
        if self.completed or self.current_index >= len(self.sequence) - 1:
            return "Finish"
        return self.sequence[self.current_index + 1]

    def has_announcement(self):
        return self._current_announcement is not None

    def get_announcement(self):
        msg = self._current_announcement
        self._current_announcement = None
        return msg

    def check_oracle_trigger(self):
        """
        Checks if auto-analysis should trigger.
        Condition: Held correct pose for > 4 seconds, not yet analyzed.
        """
        if self.pose_start_time is None or self.analysis_triggered_for_current_pose:
            return False

        if time.time() - self.pose_start_time > 4.0:
            self.analysis_triggered_for_current_pose = True
            return True
        return False

    def update(self, detected_label, is_stable):
        """Updates sequencer state based on detected pose."""
        if self.completed:
            return "Complete"

        target = self.sequence[self.current_index]

        if detected_label == target:
            if self.pose_start_time is None:
                self.pose_start_time = time.time()
                self.analysis_triggered_for_current_pose = False

            # Advance after HOLD_DURATION seconds of holding
            if time.time() - self.pose_start_time > self.HOLD_DURATION:
                self.advance()
                return "Advance"
        else:
            self.pose_start_time = None

        return "Holding"

    def advance(self):
        self.current_index += 1
        self.pose_start_time = None
        self.analysis_triggered_for_current_pose = False
        self.last_transition_time = time.time()

        if self.current_index >= len(self.sequence):
            self.completed = True
            self._current_announcement = f"{self.sequence_name} complete. Namaste."
        else:
            next_pose = self.sequence[self.current_index]
            praises = ["Great.", "Perfect.", "Smooth.", "Excellent.", "Well done."]
            self._current_announcement = f"{random.choice(praises)} Now, {next_pose}."

    def get_hold_elapsed(self):
        """Returns seconds held on current pose, or 0 if not holding."""
        if self.pose_start_time is None:
            return 0.0
        return time.time() - self.pose_start_time

    def get_progress(self):
        return self.current_index / len(self.sequence)
