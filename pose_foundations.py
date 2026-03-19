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
    Calculates an end point for a correction vector arrow.
    start_point: (x, y) in normalized coords
    direction_angle: degrees (0=right, 90=down in screen coords)
    magnitude: length in normalized coords
    Returns: (end_x, end_y)
    """
    rad = np.radians(direction_angle)
    end_x = start_point[0] + magnitude * np.cos(rad)
    end_y = start_point[1] + magnitude * np.sin(rad)
    return (end_x, end_y)

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
    ],
    # Chair
    "CHAIR_DEEPEN": [
        ("DEEPER", "Sit deeper. Like you are sitting in a chair."),
        ("BEND KNEES", "Bend your knees more. Find that burn."),
        ("SIT BACK", "Sit back and down. Weight in your heels."),
    ],
    "CHAIR_EASE": [
        ("EASE UP", "Ease up slightly. Protect your knees."),
        ("NOT SO DEEP", "Come up a little. Keep it sustainable."),
    ],
    "CHAIR_UPRIGHT": [
        ("UPRIGHT", "Lift your chest. Stay upright."),
        ("CHEST UP", "Chest up. Don't lean forward."),
        ("TALL SPINE", "Lengthen your spine upward."),
    ],
    # Downward Dog
    "DDOG_PIKE": [
        ("HIPS UP", "Press your hips up and back."),
        ("PIKE MORE", "Push your hips higher toward the ceiling."),
        ("LIFT HIPS", "Lift through your sit bones."),
    ],
    "DDOG_LEGS": [
        ("STRAIGHTEN", "Straighten your legs. Press heels down."),
        ("EXTEND LEGS", "Work toward straight legs."),
        ("HEELS DOWN", "Press your heels toward the floor."),
    ],
    # Extended Side Angle
    "ESA_BEND": [
        ("SIDE BEND", "Deepen your side bend. Reach further."),
        ("REACH OVER", "Lengthen through your top arm."),
        ("OPEN SIDE", "Open your side body. Create space."),
    ],
    # Crescent Lunge
    "CLUNGE_DEEPEN": [
        ("DEEPEN", "Sink deeper into your crescent lunge."),
        ("BEND MORE", "Bend your front knee toward ninety degrees."),
        ("LOWER HIPS", "Drop your hips lower."),
    ],
    "CLUNGE_UPRIGHT": [
        ("UPRIGHT", "Lift your torso over your hips."),
        ("CHEST UP", "Draw your chest upward."),
        ("TALL SPINE", "Lengthen through the crown of your head."),
    ],
    # High Lunge
    "HLUNGE_DEEPEN": [
        ("DEEPEN", "Bend your front knee deeper."),
        ("SINK DOWN", "Sink your hips lower."),
        ("LUNGE DEEP", "Find depth in your lunge."),
    ],
    "HLUNGE_UPRIGHT": [
        ("UPRIGHT", "Stack your torso over your hips."),
        ("LIFT CHEST", "Lift your chest. Don't lean forward."),
        ("TALL TORSO", "Lengthen your spine upward."),
    ],
    # Mountain Pose
    "MOUNTAIN_LEAN": [
        ("STAND TALL", "Stand tall. Find your center."),
        ("CENTER", "Center your weight evenly."),
        ("VERTICAL", "Stack your spine vertically."),
    ],
    "MOUNTAIN_ALIGN": [
        ("ALIGN", "Stack shoulders over hips."),
        ("CENTER BODY", "Bring your body to center."),
        ("EVEN WEIGHT", "Distribute weight evenly through both feet."),
    ],
    # Triangle
    "TRI_LEG": [
        ("STRAIGHTEN", "Straighten your front leg."),
        ("EXTEND LEG", "Press through your front knee."),
        ("LONG LEG", "Lengthen your front leg fully."),
    ],
    "TRI_BEND": [
        ("SIDE BEND", "Deepen the lateral bend."),
        ("REACH DOWN", "Reach your lower hand toward the floor."),
        ("OPEN SIDE", "Create length in your side body."),
    ],
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
        checks = {
            "Warrior II": PoseHeuristics.check_warrior_ii,
            "Tree": PoseHeuristics.check_tree,
            "Plank": PoseHeuristics.check_plank,
            "Chair": PoseHeuristics.check_chair,
            "Crescent Lunge": PoseHeuristics.check_crescent_lunge,
            "Downward Dog": PoseHeuristics.check_downward_dog,
            "Extended Side Angle": PoseHeuristics.check_extended_side_angle,
            "High Lunge": PoseHeuristics.check_high_lunge,
            "Mountain Pose": PoseHeuristics.check_mountain,
            "Triangle": PoseHeuristics.check_triangle,
        }
        check_fn = checks.get(pose_name)
        if check_fn:
            return check_fn(landmarks)
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

    @staticmethod
    def check_crescent_lunge(landmarks):
        # Front knee — ideal 130-170° (POSE_PROFILES: 0.72-0.94)
        l_knee = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
        r_knee = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
        front_idx = 25 if l_knee < r_knee else 26
        front_angle = l_knee if front_idx == 25 else r_knee
        knee_pt = landmarks[front_idx]

        if front_angle > 170:
            advice = PoseHeuristics.get_advice("CLUNGE_DEEPEN", "DEEPEN", "Sink deeper into your lunge.")
            return {
                'text': advice,
                'vector': (tuple(knee_pt), (knee_pt[0], knee_pt[1] + 0.12)),
                'color': (0, 255, 255)
            }

        # Trunk lean — ideal <30° (POSE_PROFILES: 0.0-0.17)
        mid_shoulder = [(landmarks[11][0] + landmarks[12][0]) / 2,
                        (landmarks[11][1] + landmarks[12][1]) / 2]
        mid_hip = [(landmarks[23][0] + landmarks[24][0]) / 2,
                   (landmarks[23][1] + landmarks[24][1]) / 2]
        dx = abs(mid_shoulder[0] - mid_hip[0])
        if dx > 0.1:
            advice = PoseHeuristics.get_advice("CLUNGE_UPRIGHT", "UPRIGHT", "Lift chest upright.")
            return {
                'text': advice,
                'vector': (tuple(mid_shoulder), (mid_shoulder[0], mid_shoulder[1] - 0.12)),
                'color': (0, 255, 255)
            }
        return None

    @staticmethod
    def check_chair(landmarks):
        # Knee flexion — ideal ~165° (POSE_PROFILES: 0.89-0.94 normalized)
        l_knee = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
        r_knee = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
        avg_knee = (l_knee + r_knee) / 2
        knee_pt = landmarks[25]

        if avg_knee > 170:
            advice = PoseHeuristics.get_advice("CHAIR_DEEPEN", "DEEPER", "Sit deeper.")
            return {
                'text': advice,
                'vector': (tuple(knee_pt), (knee_pt[0], knee_pt[1] + 0.12)),
                'color': (0, 255, 255)
            }
        if avg_knee < 130:
            advice = PoseHeuristics.get_advice("CHAIR_EASE", "EASE UP", "Come up a bit.")
            return {
                'text': advice,
                'vector': (tuple(knee_pt), (knee_pt[0], knee_pt[1] - 0.1)),
                'color': (0, 255, 255)
            }

        # Forward lean — trunk angle (shoulder-hip vertical)
        mid_shoulder = [(landmarks[11][0] + landmarks[12][0]) / 2,
                        (landmarks[11][1] + landmarks[12][1]) / 2]
        mid_hip = [(landmarks[23][0] + landmarks[24][0]) / 2,
                   (landmarks[23][1] + landmarks[24][1]) / 2]
        dx = abs(mid_shoulder[0] - mid_hip[0])
        if dx > 0.08:
            advice = PoseHeuristics.get_advice("CHAIR_UPRIGHT", "UPRIGHT", "Lift your chest.")
            return {
                'text': advice,
                'vector': (tuple(mid_shoulder), (mid_shoulder[0], mid_shoulder[1] - 0.12)),
                'color': (0, 255, 255)
            }
        return None

    @staticmethod
    def check_downward_dog(landmarks):
        # Hip angle — ideal very acute (~14°, POSE_PROFILES: 0.04-0.11)
        l_hip = calculate_angle(landmarks[11], landmarks[23], landmarks[25])
        r_hip = calculate_angle(landmarks[12], landmarks[24], landmarks[26])
        avg_hip = (l_hip + r_hip) / 2
        hip_pt = landmarks[23]

        if avg_hip > 45:
            advice = PoseHeuristics.get_advice("DDOG_PIKE", "HIPS UP", "Push hips higher.")
            return {
                'text': advice,
                'vector': (tuple(hip_pt), (hip_pt[0], hip_pt[1] - 0.15)),
                'color': (0, 255, 255)
            }

        # Knee straightness — ideal ~166° (POSE_PROFILES: 0.85-1.0)
        l_knee = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
        r_knee = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
        avg_knee = (l_knee + r_knee) / 2
        if avg_knee < 150:
            knee_pt = landmarks[25]
            advice = PoseHeuristics.get_advice("DDOG_LEGS", "STRAIGHTEN", "Straighten legs.")
            return {
                'text': advice,
                'vector': (tuple(knee_pt), (knee_pt[0], knee_pt[1] + 0.1)),
                'color': (0, 255, 255)
            }
        return None

    @staticmethod
    def check_extended_side_angle(landmarks):
        # Lateral flexion — ideal ~57° (POSE_PROFILES: 0.29-0.35)
        # Approximate via shoulder-hip angle offset
        l_shoulder = landmarks[11]
        l_hip = landmarks[23]
        dy = abs(l_shoulder[1] - l_hip[1])
        dx = abs(l_shoulder[0] - l_hip[0])
        if dy > 0 and dx / dy < 0.4:
            advice = PoseHeuristics.get_advice("ESA_BEND", "SIDE BEND", "Deepen side bend.")
            mid_torso = [(l_shoulder[0] + l_hip[0]) / 2,
                         (l_shoulder[1] + l_hip[1]) / 2]
            return {
                'text': advice,
                'vector': (tuple(mid_torso), (mid_torso[0] - 0.12, mid_torso[1] + 0.06)),
                'color': (0, 255, 255)
            }
        return None

    @staticmethod
    def check_high_lunge(landmarks):
        # Front knee — ideal 130-170° (POSE_PROFILES: 0.72-0.94)
        l_knee = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
        r_knee = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
        front_idx = 25 if l_knee < r_knee else 26
        front_angle = l_knee if front_idx == 25 else r_knee
        knee_pt = landmarks[front_idx]

        if front_angle > 170:
            advice = PoseHeuristics.get_advice("HLUNGE_DEEPEN", "DEEPEN", "Bend front knee deeper.")
            return {
                'text': advice,
                'vector': (tuple(knee_pt), (knee_pt[0], knee_pt[1] + 0.12)),
                'color': (0, 255, 255)
            }

        # Trunk lean — ideal <30° (POSE_PROFILES: 0.0-0.17)
        mid_shoulder = [(landmarks[11][0] + landmarks[12][0]) / 2,
                        (landmarks[11][1] + landmarks[12][1]) / 2]
        mid_hip = [(landmarks[23][0] + landmarks[24][0]) / 2,
                   (landmarks[23][1] + landmarks[24][1]) / 2]
        dx = abs(mid_shoulder[0] - mid_hip[0])
        if dx > 0.1:
            advice = PoseHeuristics.get_advice("HLUNGE_UPRIGHT", "UPRIGHT", "Lift chest upright.")
            return {
                'text': advice,
                'vector': (tuple(mid_shoulder), (mid_shoulder[0], mid_shoulder[1] - 0.12)),
                'color': (0, 255, 255)
            }
        return None

    @staticmethod
    def check_mountain(landmarks):
        # Lateral lean — ideal <1° (POSE_PROFILES: 0.0-0.005)
        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]
        l_hip = landmarks[23]
        r_hip = landmarks[24]
        shoulder_mid_y = (l_shoulder[1] + r_shoulder[1]) / 2
        hip_mid_y = (l_hip[1] + r_hip[1]) / 2
        shoulder_mid_x = (l_shoulder[0] + r_shoulder[0]) / 2
        hip_mid_x = (l_hip[0] + r_hip[0]) / 2

        lateral_offset = abs(shoulder_mid_x - hip_mid_x)
        if lateral_offset > 0.04:
            pt = [shoulder_mid_x, shoulder_mid_y]
            advice = PoseHeuristics.get_advice("MOUNTAIN_LEAN", "STAND TALL", "Stand tall.")
            return {
                'text': advice,
                'vector': (tuple(pt), (hip_mid_x, pt[1])),
                'color': (0, 255, 255)
            }

        # Shoulder-hip alignment — ideal 0.95-1.0
        shoulder_diff = abs(l_shoulder[1] - r_shoulder[1])
        if shoulder_diff > 0.04:
            lower = 11 if l_shoulder[1] > r_shoulder[1] else 12
            pt = landmarks[lower]
            advice = PoseHeuristics.get_advice("MOUNTAIN_ALIGN", "ALIGN", "Align shoulders.")
            return {
                'text': advice,
                'vector': (tuple(pt), (pt[0], pt[1] - 0.08)),
                'color': (0, 255, 255)
            }
        return None

    @staticmethod
    def check_triangle(landmarks):
        # Front knee straightness — ideal ~179° (POSE_PROFILES: 0.97-1.0)
        l_knee = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
        r_knee = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
        # Front leg is the one with more bend
        front_idx = 25 if l_knee < r_knee else 26
        front_angle = l_knee if front_idx == 25 else r_knee
        if front_angle < 165:
            knee_pt = landmarks[front_idx]
            advice = PoseHeuristics.get_advice("TRI_LEG", "STRAIGHTEN", "Straighten front leg.")
            return {
                'text': advice,
                'vector': (tuple(knee_pt), (knee_pt[0], knee_pt[1] + 0.1)),
                'color': (0, 255, 255)
            }

        # Lateral flexion — check torso side bend
        l_shoulder = landmarks[11]
        l_hip = landmarks[23]
        dy = abs(l_shoulder[1] - l_hip[1])
        dx = abs(l_shoulder[0] - l_hip[0])
        if dy > 0 and dx / dy < 0.4:
            mid = [(l_shoulder[0] + l_hip[0]) / 2, (l_shoulder[1] + l_hip[1]) / 2]
            advice = PoseHeuristics.get_advice("TRI_BEND", "SIDE BEND", "Deepen side bend.")
            return {
                'text': advice,
                'vector': (tuple(mid), (mid[0] - 0.1, mid[1] + 0.06)),
                'color': (0, 255, 255)
            }
        return None
