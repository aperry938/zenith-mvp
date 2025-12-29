import time

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

    def get_current_goal(self):
        if self.completed:
            return "Sequence Complete!"
        return self.sequence[self.current_index]

    def get_next_goal(self):
        if self.completed or self.current_index >= len(self.sequence) - 1:
            return "Finish"
        return self.sequence[self.current_index + 1]

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
            
            # If held for 3 seconds (Stable) -> Advance
            if time.time() - self.pose_start_time > 3.0:
                 self.advance()
                 return "Advance"
        else:
            self.pose_start_time = None # Reset if they break form
            
        return "Holding"

    def advance(self):
        self.current_index += 1
        self.pose_start_time = None
        self.last_transition_time = time.time()
        if self.current_index >= len(self.sequence):
            self.completed = True

    def get_progress(self):
        return (self.current_index / len(self.sequence))
