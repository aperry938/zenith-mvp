import numpy as np

# --- GEOMETRY UTILS ---
def calculate_angle(a, b, c):
    """
    Calculates the interior angle at point b (in degrees) given three points [x, y].
    a--b--c
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# --- HEURISTIC DEFINITIONS ---
class PoseHeuristics:
    @staticmethod
    def evaluate(pose_name, landmarks):
        """
        Input: 
            pose_name (str): "Warrior II", "Tree", etc.
            landmarks (dict): {mp_pose.PoseLandmark: [x, y]}
        Output:
            feedback (str or None): The correction string, or None if good.
        """
        if pose_name == "Warrior II":
            return PoseHeuristics.check_warrior_ii(landmarks)
        elif pose_name == "Tree":
            return PoseHeuristics.check_tree(landmarks)
        elif pose_name == "Plank":
            return PoseHeuristics.check_plank(landmarks)
        
        return None

    @staticmethod
    def check_warrior_ii(landmarks):
        # MediaPipe Landmarks: 
        # 11/12 (Shoulders), 23/24 (Hips), 25/26 (Knees), 27/28 (Ankles)
        # We need to detect WHICH leg is forward. 
        # Heuristic: The foot with the larger X spread from the center might be forward? 
        # Simpler: Check BOTH knees. If one is bent (~90) and one is straight (~180), focus on the bent one.
        
        # Get coordinates (normalized 0-1)
        # We assume the caller passes a dict accessible by index or enum
        # But to be generic, let's assume standard MediaPipe indices if passed as list, or dict.
        # Let's demand a dictionary keyed by integer indices for simplicity here.
        
        # 23: LEFT_HIP, 25: LEFT_KNEE, 27: LEFT_ANKLE
        l_knee_ang = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
        # 24: RIGHT_HIP, 26: RIGHT_KNEE, 28: RIGHT_ANKLE
        r_knee_ang = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
        
        # Determine front leg (the one prominently bent)
        # Threshold: "Bent" < 135 deg. "Straight" > 150 deg.
        
        front_knee_angle = None
        if l_knee_ang < 135:
            front_knee_angle = l_knee_ang
        elif r_knee_ang < 135:
            front_knee_angle = r_knee_ang
            
        if front_knee_angle:
            if front_knee_angle > 110:
                return "Deepen your lunge. Bend your front knee to ninety degrees."
            elif front_knee_angle < 75:
                return "Ease up. Your knee is bent too far."
        else:
             return "Bend your front knee."

        # Arms Check (11-13-15 and 12-14-16)
        l_arm_ang = calculate_angle(landmarks[11], landmarks[13], landmarks[15])
        r_arm_ang = calculate_angle(landmarks[12], landmarks[14], landmarks[16])
        
        if l_arm_ang < 150 or r_arm_ang < 150:
            return "Extend your arms fully."

        return None # "Good Form"

    @staticmethod
    def check_tree(landmarks):
        # Tree Pose: Foot should not be on knee.
        # 25/26 are knees. 29/30 are heels.
        
        # Simple check: Is a foot vertical position close to a knee vertical position?
        # Note: y increases downwards in image coordinates usually, but MP provides normalized.
        
        # Left foot (29) vs Right knee (26)
        # Right foot (30) vs Left knee (25)
        
        # Check alignment drift
        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]
        
        # If shoulders are not level? (y diff)
        y_diff = abs(l_shoulder[1] - r_shoulder[1])
        if y_diff > 0.05:
            return "Keep your shoulders level."
            
        return None

    @staticmethod
    def check_plank(landmarks):
        # Plank: Shoulders, Hips, Ankles should be collinear.
        # Check angle at Hips. Should be ~180.
        
        l_hip_ang = calculate_angle(landmarks[11], landmarks[23], landmarks[27])
        r_hip_ang = calculate_angle(landmarks[12], landmarks[24], landmarks[28])
        
        # Average hip angle
        avg_hip = (l_hip_ang + r_hip_ang) / 2
        
        if avg_hip < 160:
            return "Lower your hips to a straight line."
        # If angle is > 180 (hyperextension/sagging), calculate_angle handles 0-180 usually. 
        # Need to be careful with angle definition.
        
        return None
