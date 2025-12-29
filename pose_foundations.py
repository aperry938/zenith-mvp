import numpy as np

# --- GEOMETRY UTILS ---
def calculate_angle(a, b, c):
    """
    Calculates the interior angle at point b (in degrees).
    """
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def calculate_vector(start_point, direction_angle, magnitude=0.15):
    """
    Calculates an end point for a vector starting at 'start_point' with a given angle.
    Used for drawing correction arrows.
    start_point: [x, y] normalized
    direction_angle: radians (or we can just use relative offsets)
    """
    # Simple relative vector for now.
    # We want to point "Outward" perpendicular to the leg, ideally.
    # For now, let's just return fixed offsets based on heuristic.
    pass

# --- HEURISTIC DEFINITIONS ---
class PoseHeuristics:
    @staticmethod
    def evaluate(pose_name, landmarks):
        """
        Input: landmarks (dict): {mp_pose.PoseLandmark: [x, y]}
        Output: dict {'text': str, 'vector': tuple((x1,y1), (x2,y2)), 'color': tuple} OR None
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
        # 23: LEFT_HIP, 25: LEFT_KNEE, 27: LEFT_ANKLE
        l_knee_ang = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
        # 24: RIGHT_HIP, 26: RIGHT_KNEE, 28: RIGHT_ANKLE
        r_knee_ang = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
        
        # Determine bent knee
        front_knee_idx = 25 if l_knee_ang < 135 else 26
        angle = l_knee_ang if front_knee_idx == 25 else r_knee_ang
        knee_pt = landmarks[front_knee_idx]
        
        correction = None
        
        if angle > 110:
            correction = {
                'text': "Deepen your lunge.",
                'vector': (tuple(knee_pt), (knee_pt[0], knee_pt[1] + 0.15)), # Point Down
                'color': (0, 255, 255) # Yellow/Cyan
            }
        elif angle < 75:
             correction = {
                'text': "Ease up.",
                'vector': (tuple(knee_pt), (knee_pt[0], knee_pt[1] - 0.1)), # Point Up
                'color': (0, 255, 255)
            }
        
        # Knee Cave-in Heuristic (Valgus Collapse)
        # If knee x is too far "in" relative to ankle x?
        # Requires 3D really, but 2D heuristic: 
        # If front leg is Right (26) and facing right...
        # Let's simple check arms for now as secondary
        
        l_arm_ang = calculate_angle(landmarks[11], landmarks[13], landmarks[15])
        if not correction and l_arm_ang < 150:
             correction = {
                'text': "Extend left arm.",
                'vector': (tuple(landmarks[13]), (landmarks[13][0]-0.1, landmarks[13][1])), # Point Left
                'color': (0, 255, 255)
             }

        return correction

    @staticmethod
    def check_tree(landmarks):
        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]
        if abs(l_shoulder[1] - r_shoulder[1]) > 0.05:
            # Point to lower shoulder to move it up
            lower = 11 if l_shoulder[1] > r_shoulder[1] else 12
            pt = landmarks[lower]
            return {
                'text': "Level shoulders.",
                'vector': (tuple(pt), (pt[0], pt[1] - 0.1)), # Up
                'color': (0, 255, 255)
            }
        return None

    @staticmethod
    def check_plank(landmarks):
         l_hip_ang = calculate_angle(landmarks[11], landmarks[23], landmarks[27])
         r_hip_ang = calculate_angle(landmarks[12], landmarks[24], landmarks[28])
         avg = (l_hip_ang + r_hip_ang) / 2
         if avg < 160:
             # Hips up
             pt = landmarks[23]
             return {
                 'text': "Lower hips.",
                 'vector': (tuple(pt), (pt[0], pt[1] + 0.1)), # Down
                 'color': (0, 255, 255)
             }
         return None
