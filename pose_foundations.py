import numpy as np
import random

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
    Calculates an end point for a vector.
    """
    pass

# --- COACHING LIBRARY ---
# Tuples: (Short HUD Text, Long Spoken Text)
COACHING_LIBRARY = {
    "WARRIOR_DEEPEN": [
        ("DEEPEN", "Deepen your lunge. Sink into it."),
        ("BEND KNEE", "Bend your front knee to ninety degrees."),
        ("LOWER HIPS", "Challenge yourself. Lower your hips."),
        ("SINK", "Find your edge. Deepen the pose.")
    ],
    "WARRIOR_EASE": [
        ("EASE UP", "Ease up slightly. Protect your knee."),
        ("BACK OFF", "Listen to your body. Back off a bit."),
        ("TOO DEEP", "You are too deep. Come up a little.")
    ],
    "ARM_EXTEND": [
        ("EXTEND ARM", "Extend your back arm fully."),
        ("REACH BACK", "Reach back with your left arm."),
        ("OPEN CHEST", "Open your chest. Reach through your fingertips.")
    ],
    "SHOULDERS_LEVEL": [
        ("LEVEL SHLDR", "Level your shoulders."),
        ("RELAX SHLDR", "Relax your shoulders down nicely."),
        ("DROP SHLDR", "Drop your shoulders away from your ears.")
    ],
    "HIPS_LOWER": [
        ("LOWER HIPS", "Lower your hips to form a straight line."),
        ("FLAT BACK", "Flatten your back. Engage core."),
        ("CORE ENGAGE", "Squeeze your belly button to your spine.")
    ]
}

class PoseHeuristics:
    @staticmethod
    def evaluate(pose_name, landmarks):
        """
        Input: landmarks (dict): {mp_pose.PoseLandmark: [x, y]}
        Output: dict {
            'text': (hud_str, spoken_str), 
            'vector': tuple((x1,y1), (x2,y2)), 
            'color': tuple
        } OR None
        """
        if pose_name == "Warrior II":
            return PoseHeuristics.check_warrior_ii(landmarks)
        elif pose_name == "Tree":
            return PoseHeuristics.check_tree(landmarks)
        elif pose_name == "Plank":
            return PoseHeuristics.check_plank(landmarks)
        return None

    @staticmethod
    def get_advice(key, default_short, default_long):
        """Helper to get random advice from library."""
        if key in COACHING_LIBRARY:
            return random.choice(COACHING_LIBRARY[key])
        return (default_short, default_long)

    @staticmethod
    def check_warrior_ii(landmarks):
        l_knee_ang = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
        r_knee_ang = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
        front_knee_idx = 25 if l_knee_ang < 135 else 26
        angle = l_knee_ang if front_knee_idx == 25 else r_knee_ang
        knee_pt = landmarks[front_knee_idx]
        
        correction = None
        
        if angle > 110:
            advice = PoseHeuristics.get_advice("WARRIOR_DEEPEN", "DEEPEN", "Deepen your lunge.")
            correction = {
                'text': advice,
                'vector': (tuple(knee_pt), (knee_pt[0], knee_pt[1] + 0.15)), 
                'color': (0, 255, 255)
            }
        elif angle < 75:
             advice = PoseHeuristics.get_advice("WARRIOR_EASE", "EASE UP", "Ease up.")
             correction = {
                'text': advice,
                'vector': (tuple(knee_pt), (knee_pt[0], knee_pt[1] - 0.1)),
                'color': (0, 255, 255)
            }
        
        l_arm_ang = calculate_angle(landmarks[11], landmarks[13], landmarks[15])
        if not correction and l_arm_ang < 150:
             advice = PoseHeuristics.get_advice("ARM_EXTEND", "EXTEND", "Extend left arm.")
             correction = {
                'text': advice,
                'vector': (tuple(landmarks[13]), (landmarks[13][0]-0.1, landmarks[13][1])),
                'color': (0, 255, 255)
             }

        return correction

    @staticmethod
    def check_tree(landmarks):
        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]
        if abs(l_shoulder[1] - r_shoulder[1]) > 0.05:
            lower = 11 if l_shoulder[1] > r_shoulder[1] else 12
            pt = landmarks[lower]
            advice = PoseHeuristics.get_advice("SHOULDERS_LEVEL", "LEVEL SHLDR", "Level shoulders.")
            return {
                'text': advice,
                'vector': (tuple(pt), (pt[0], pt[1] - 0.1)),
                'color': (0, 255, 255)
            }
        return None

    @staticmethod
    def check_plank(landmarks):
         l_hip_ang = calculate_angle(landmarks[11], landmarks[23], landmarks[27])
         r_hip_ang = calculate_angle(landmarks[12], landmarks[24], landmarks[28])
         avg = (l_hip_ang + r_hip_ang) / 2
         if avg < 160:
             pt = landmarks[23]
             advice = PoseHeuristics.get_advice("HIPS_LOWER", "LOWER HIPS", "Lower hips.")
             return {
                 'text': advice,
                 'vector': (tuple(pt), (pt[0], pt[1] + 0.1)),
                 'color': (0, 255, 255)
             }
         return None
