import time
import random

class PoseSequencer:
    """
    Manages the state of a specific Yoga Sequence (e.g., Sun Salutation A).
    Tracks current pose index, duration held, and transitions.
    """
    def __init__(self):
        # SUN SALUTATION A (Simplified)
        self.sequence = [
            "Mountain Pose",
            "Upward Salute", # Hands up
            "Forward Fold",
            "Halfway Lift",
            "Plank",
            "Cobra", # or Up Dog
            "Downward Dog",
            # "Halfway Lift", # Return
            # "Forward Fold",
            # "Mountain Pose"
        ]
        self.current_index = 0
        self.last_transition_time = time.time()
        self.pose_start_time = None
        self.completed = False
        self._current_announcement = "Welcome to Zenith. Let's begin with Mountain Pose."
        
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
        Checks if the Oracle (Auto-Analysis) should be triggered.
        Condition: User has held the correct pose for > 5 seconds AND it hasn't been analyzed yet.
        """
        if self.pose_start_time is None or self.analysis_triggered_for_current_pose:
            return False
            
        time_held = time.time() - self.pose_start_time
        if time_held > 5.0:
            self.analysis_triggered_for_current_pose = True # Prevent spam
            return True
            
        return False

    def update(self, detected_label, is_stable):
        """
        Updates the sequencer state based on the detected pose.
        """
        if self.completed:
            return "Complete"

        target_pose = self.sequence[self.current_index]

        # Simple Check: If we detect the target pose
        # (In reality, we might map "Tree" or "Warrior" labels to these, 
        # but our classifier currently outputs a limited set. 
        # Let's assume we update the classifier or map closely.)
        
        # MAPPING: Our classifier has: 
        # ["Chair","Downward Dog","Extended Side Angle","High Lunge","Mountain Pose","Plank","Tree","Triangle","Warrior II"]
        # It's missing Forward Fold, Cobra, etc.
        # This is the "Ghost in the Machine" - we need a better classifier!
        # For MVP, we will simulate the sequence logic using the available poses
        # and just wait for *any* stable pose to advance if no direct match exists,
        # OR just strictly check the ones we have.
        
        # Let's simplify the sequence to what we CAN detect for Cycle 7 Proof of Concept:
        # Mountain -> Plank -> Downward Dog -> Mountain
        
        effective_sequence = ["Mountain Pose", "Plank", "Downward Dog", "Mountain Pose"]
        
        # Override the init sequence for this cycle
        if len(self.sequence) != len(effective_sequence):
            self.sequence = effective_sequence
        
        target = self.sequence[self.current_index]
        
        # Transition Logic
        if detected_label == target:
            if self.pose_start_time is None:
                self.pose_start_time = time.time()
                self.analysis_triggered_for_current_pose = False # Reset for new attempt at this pose
            
            # If held for 3 seconds (Stable) -> Advance
            # Wait, if we advance at 3 seconds, we'll never reach 5 seconds for the Oracle!
            # Let's bump the advance time to 8 seconds to allow for analysis, 
            # or make Oracle trigger quicker (e.g. 2s) and Advance at 5s.
            # Let's do: Oracle at 4s, Advance at 8s. giving time to hear advice.
            
            if time.time() - self.pose_start_time > 8.0:
                 self.advance()
                 return "Advance"
        else:
            self.pose_start_time = None # Reset if they break form
            
        return "Holding"

    def advance(self):
        self.current_index += 1
        self.pose_start_time = None
        self.analysis_triggered_for_current_pose = False # Reset for next pose
        self.last_transition_time = time.time()
        
        if self.current_index >= len(self.sequence):
            self.completed = True
            self._current_announcement = "Sequence Complete. Namaste."
        else:
            # Generate Transition Message
            next_pose = self.sequence[self.current_index]
            praises = ["Great job.", "Perfect.", "Smooth.", "Excellent."]
            praise = random.choice(praises)
            self._current_announcement = f"{praise} Now transition to {next_pose}."

    def get_progress(self):
        return (self.current_index / len(self.sequence))
