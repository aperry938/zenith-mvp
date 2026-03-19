import numpy as np
import random
from typing import Optional

# --- VALID PERSONAS ---
VALID_PERSONAS = ("default", "sage", "scientist", "warrior", "flow", "traditional")

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
    # --- POSITIVE FEEDBACK ---
    "FORM_EXCELLENT": [
        ("PERFECT", "Beautiful form. Hold it right there."),
        ("NAILED IT", "That's it. Strong and steady."),
        ("SOLID", "Solid alignment. Keep breathing."),
        ("BEAUTIFUL", "Beautiful. You're in the zone."),
        ("GREAT FORM", "Great form. Stay with it."),
        ("ON POINT", "Right on point. Keep holding."),
    ],
}

# --- PERSONA-SPECIFIC COACHING ---
# Each persona provides alternative phrasings for the same correction keys.
# If a key is missing for a persona, falls back to COACHING_LIBRARY (default).
PERSONA_COACHING = {
    "sage": {
        "WARRIOR_DEEPEN": [
            ("FIND DEPTH", "In the depth of this pose, you find strength. Surrender to gravity."),
            ("SINK IN", "The warrior finds power not by fighting, but by rooting. Deepen."),
            ("PATIENCE", "Patience. Let yourself descend one breath at a time."),
        ],
        "WARRIOR_EASE": [
            ("LISTEN", "Wisdom is knowing when to pull back. Honor your edge."),
            ("SOFTEN", "Soften. The strongest trees bend with the wind."),
            ("YIELD", "Yield slightly. True strength lives in restraint."),
        ],
        "ARM_EXTEND": [
            ("REACH OUT", "Extend outward as if touching the horizon with your intention."),
            ("EXPAND", "Let your arms be an expression of your inner reach."),
            ("OPEN WIDE", "The arms are the wings of your awareness. Spread them."),
        ],
        "SHOULDERS_LEVEL": [
            ("BALANCE", "Find balance in the shoulders. Let them rest like still water."),
            ("RELEASE", "Release the weight you carry in your shoulders."),
            ("SETTLE", "Let the shoulders settle. They hold no burden here."),
        ],
        "HIPS_LOWER": [
            ("ALIGN", "Bring the hips into alignment. Let the body form one line."),
            ("INTEGRATE", "Integrate your core. The center holds everything together."),
            ("STILL LINE", "Become a still, unbroken line from head to heel."),
        ],
        "CHAIR_DEEPEN": [
            ("SIT DEEPER", "Sit into the invisible throne beneath you. Trust it will hold."),
            ("ROOT DOWN", "Root down through your heels. The earth will meet you."),
            ("DESCEND", "Descend with awareness. Each inch reveals new ground."),
        ],
        "CHAIR_EASE": [
            ("ENOUGH", "There is wisdom in enough. Rise slightly."),
            ("RESPECT", "Respect the body's message. Ease up."),
        ],
        "CHAIR_UPRIGHT": [
            ("RISE WITHIN", "While the legs descend, let the spine rise within."),
            ("TALL HEART", "Lift from the heart center. Stand tall inside."),
            ("DIGNIFY", "Dignify the pose. Crown reaches skyward."),
        ],
        "DDOG_PIKE": [
            ("INVERTED V", "Become the mountain, inverted. Press hips toward the sky."),
            ("ASCEND", "Let the hips ascend. The body forms a temple arch."),
            ("LIFT", "Lift through the sit bones. Seek the ceiling."),
        ],
        "DDOG_LEGS": [
            ("GROUND", "Ground through the heels. The earth is your anchor."),
            ("LENGTHEN", "Lengthen through the backs of the legs. Find your edge."),
            ("REACH DOWN", "Let the heels reach for the floor. Gravity is your teacher."),
        ],
        "ESA_BEND": [
            ("OPEN", "Open your side body like a door letting in light."),
            ("LATERAL", "The lateral bend reveals the spaces between the ribs."),
            ("UNFOLD", "Unfold through the side. Every breath creates more room."),
        ],
        "CLUNGE_DEEPEN": [
            ("SURRENDER", "Surrender into the lunge. Depth is found in letting go."),
            ("MELT", "Melt downward. The hips are heavy with intention."),
            ("TRUST", "Trust your legs. Deepen with confidence."),
        ],
        "CLUNGE_UPRIGHT": [
            ("RISE UP", "While the legs root down, let the torso rise like smoke."),
            ("VERTICAL", "Find the vertical axis within. Spine ascends."),
            ("LIFT HEART", "Lift from the heart. The upper body is light."),
        ],
        "HLUNGE_DEEPEN": [
            ("EXPLORE", "Explore the depth. Your legs know the way."),
            ("BELOW", "What you seek lies below. Bend deeper."),
            ("COMMIT", "Commit to the lunge. Halfway serves no one."),
        ],
        "HLUNGE_UPRIGHT": [
            ("TALL SPINE", "The spine is your antenna to the sky. Stand tall within."),
            ("CENTER", "Center the torso over the hips. Poise is balance."),
            ("UPWARD", "Energy flows upward through the crown."),
        ],
        "MOUNTAIN_LEAN": [
            ("STILLNESS", "Find stillness. The mountain does not lean."),
            ("CENTER", "Return to center. You are the axis of your world."),
            ("VERTICAL", "Become the vertical. Rooted and reaching."),
        ],
        "MOUNTAIN_ALIGN": [
            ("SYMMETRY", "Seek symmetry. Shoulders mirror hips in quiet balance."),
            ("HARMONY", "Bring the upper and lower body into harmony."),
            ("STACK", "Stack bone upon bone. The body is an elegant tower."),
        ],
        "TRI_LEG": [
            ("STRAIGHTEN", "The front leg is a pillar. Let it be straight and strong."),
            ("EXTEND", "Extend fully. The leg is your foundation in this pose."),
            ("UNBEND", "Release the bend. A straight leg grounds the triangle."),
        ],
        "TRI_BEND": [
            ("OPEN SIDE", "Open the side body. The triangle reveals hidden space."),
            ("LATERAL", "Deepen the lateral reach. The body becomes geometry."),
            ("LEAN", "Lean further. Trust the shape to hold you."),
        ],
    },
    "scientist": {
        "WARRIOR_DEEPEN": [
            ("90 DEGREES", "Your knee angle is above 90 degrees. Flex to load the quadriceps evenly."),
            ("ENGAGE QUADS", "Increase knee flexion. Target 85 to 95 degrees for optimal quad activation."),
            ("LOWER CG", "Lower your center of gravity. More knee flexion recruits the glutes."),
        ],
        "WARRIOR_EASE": [
            ("PROTECT ACL", "Knee flexion below 75 degrees increases anterior shear force. Ease up."),
            ("REDUCE LOAD", "Reduce the eccentric load on the patellar tendon. Come up slightly."),
            ("JOINT SAFE", "Current angle places excessive stress on the knee joint. Back off."),
        ],
        "ARM_EXTEND": [
            ("FULL EXT", "Elbow angle is less than 150 degrees. Extend to full range of motion."),
            ("ARM STRAIGHT", "Straighten the arm. Full extension maximizes deltoid and trapezius engagement."),
            ("REACH", "Increase arm extension. The scapula should be stabilized and retracted."),
        ],
        "SHOULDERS_LEVEL": [
            ("LEVEL", "Shoulder asymmetry detected. Depress the elevated side by 2 to 3 centimeters."),
            ("SYMMETRY", "Trapezius imbalance observed. Level the acromion processes."),
            ("CORRECT", "Lateral shoulder tilt exceeds threshold. Equalize both sides."),
        ],
        "HIPS_LOWER": [
            ("STRAIGHT LINE", "Hip flexion breaks the plank line. Engage the transversus abdominis."),
            ("CORE ACTIVE", "Posterior pelvic tilt detected. Activate the anterior core muscles."),
            ("NEUTRAL SPINE", "Bring the pelvis to neutral. Spine should form a straight line."),
        ],
        "CHAIR_DEEPEN": [
            ("FLEX MORE", "Average knee flexion is above 170 degrees. Target 140 to 160 for Utkatasana."),
            ("LOAD GLUTES", "Deeper knee bend increases gluteal activation. Sit lower."),
            ("ENGAGE", "Quadriceps not fully engaged. Increase knee flexion by 10 to 20 degrees."),
        ],
        "CHAIR_EASE": [
            ("REDUCE ANGLE", "Knee flexion below 130 degrees. Risk of patellar tendon overload."),
            ("BACK OFF", "Current depth exceeds safe range. Reduce flexion angle."),
        ],
        "CHAIR_UPRIGHT": [
            ("TORSO ANGLE", "Forward lean exceeds 8 centimeters. Shift torso over the pelvis."),
            ("VERTICAL", "Trunk should be near vertical. Correct the anterior lean."),
            ("SPINE STACK", "Stack the thoracic spine over the lumbar. Reduce forward displacement."),
        ],
        "DDOG_PIKE": [
            ("HIP ANGLE", "Hip angle exceeds 45 degrees. Target an acute angle for proper inversion."),
            ("HIPS HIGH", "Push the ischial tuberosities toward the ceiling. Increase hip flexion."),
            ("PIKE", "Insufficient hip pike. The torso should approach the thighs."),
        ],
        "DDOG_LEGS": [
            ("KNEE EXT", "Knee extension below 150 degrees. Straighten toward 170 for full hamstring engagement."),
            ("HAMSTRINGS", "Lengthen the hamstrings. Press heels toward the ground."),
            ("STRAIGHT", "Knee angle too acute. Extend to near full range of motion."),
        ],
        "ESA_BEND": [
            ("LATERAL FLEX", "Lateral flexion insufficient. Increase the angle between ribs and pelvis."),
            ("OBLIQUES", "Engage the obliques to deepen the side bend."),
            ("SIDE RATIO", "Torso-to-hip lateral ratio below 0.4. Increase the lateral reach."),
        ],
        "CLUNGE_DEEPEN": [
            ("KNEE FLEX", "Front knee angle exceeds 170 degrees. Target 130 to 160 for Anjaneyasana."),
            ("DROP HIPS", "Lower the pelvis. Increase hip flexor stretch and quad engagement."),
            ("DEEPEN", "Insufficient knee flexion. Bend 10 to 20 degrees more."),
        ],
        "CLUNGE_UPRIGHT": [
            ("TRUNK ALIGN", "Forward trunk displacement exceeds 10 centimeters. Correct to vertical."),
            ("UPRIGHT", "The torso should be stacked directly over the pelvis. Reduce lean."),
            ("VERTICAL", "Excessive anterior trunk tilt. Engage the erector spinae."),
        ],
        "HLUNGE_DEEPEN": [
            ("FLEX", "Front knee angle above 170 degrees. Increase flexion for deeper engagement."),
            ("LOWER", "Lower the center of mass. Bend the front knee toward 130 to 160 degrees."),
            ("TARGET", "Knee angle above optimal range. Increase flexion by 15 to 25 degrees."),
        ],
        "HLUNGE_UPRIGHT": [
            ("STACK", "Stack the thoracic spine directly over the lumbar region. Reduce lean."),
            ("UPRIGHT", "Trunk displacement above threshold. Correct to near-vertical alignment."),
            ("ALIGN", "Anterior trunk lean detected. Engage spinal extensors."),
        ],
        "MOUNTAIN_LEAN": [
            ("LATERAL", "Lateral offset exceeds 4 centimeters. Center of mass should be over the base."),
            ("CENTER CG", "Shift center of gravity to midline. Even pressure through both feet."),
            ("PLUMB LINE", "The plumb line from ear to ankle should be vertical. Correct the lean."),
        ],
        "MOUNTAIN_ALIGN": [
            ("SHOULDERS", "Shoulder height differential above 4 centimeters. Level the acromions."),
            ("BILATERAL", "Bilateral asymmetry detected. Equalize shoulder height."),
            ("CORRECT", "Uneven scapular elevation. Depress the higher side."),
        ],
        "TRI_LEG": [
            ("EXTEND LEG", "Front knee angle below 165 degrees. Target near-full extension for Trikonasana."),
            ("STRAIGHTEN", "Insufficient knee extension. Straighten to engage the vastus medialis."),
            ("KNEE FULL", "The front knee should be at 170 to 180 degrees. Extend fully."),
        ],
        "TRI_BEND": [
            ("LATERAL", "Lateral flexion ratio below 0.4. Increase the torso side bend."),
            ("OBLIQUES", "Engage the lateral chain. Deepen the side bend from the waist."),
            ("REACH", "Increase lateral reach. The hand should approach the shin or floor."),
        ],
    },
    "warrior": {
        "WARRIOR_DEEPEN": [
            ("DEEPER!", "Deeper! You've got more in you. Don't hold back."),
            ("DIG IN", "Push it. Your legs can take more than your mind thinks."),
            ("GET LOW", "Get lower! Own this stance. Show it who's boss."),
        ],
        "WARRIOR_EASE": [
            ("BACK OFF", "Back off a notch. Smart fighters know when to retreat."),
            ("PROTECT", "Protect that knee. You need it for the long game."),
            ("EASE UP", "Ease up. There's no honor in a blown-out knee."),
        ],
        "ARM_EXTEND": [
            ("REACH!", "Reach! Like you're punching through a wall."),
            ("EXTEND", "Full extension. No half measures. Commit."),
            ("ARMS OUT", "Arms out. Strong. Purposeful. Like a weapon."),
        ],
        "SHOULDERS_LEVEL": [
            ("LEVEL UP", "Level those shoulders. Warriors don't slump."),
            ("SQUARE UP", "Square up. Drop the tension. Stay dangerous."),
            ("SHOULDERS!", "Shoulders down and even. Iron discipline."),
        ],
        "HIPS_LOWER": [
            ("FLAT BACK", "Lock that back flat. Not one inch of sag."),
            ("CORE!", "Engage that core like your life depends on it."),
            ("HOLD IT", "Hold that line. Straight as steel from head to heel."),
        ],
        "CHAIR_DEEPEN": [
            ("LOWER!", "Lower! You're barely sitting. Commit to the burn."),
            ("BURN IT", "Feel that burn? Good. Go deeper."),
            ("SIT DOWN", "Sit down like you mean it. Full commitment."),
        ],
        "CHAIR_EASE": [
            ("SMART", "Be smart. Come up before your knees give out."),
            ("BACK UP", "Back it up a bit. Live to fight another set."),
        ],
        "CHAIR_UPRIGHT": [
            ("CHEST UP!", "Chest up! Don't crumble forward."),
            ("STAND TALL", "Stand tall in the fire. Head high."),
            ("DON'T FOLD", "Don't fold. Spine up. Chest proud."),
        ],
        "DDOG_PIKE": [
            ("HIPS UP!", "Hips UP! Drive them to the ceiling."),
            ("PIKE!", "Pike harder. Make that shape look lethal."),
            ("PUSH!", "Push those hips. Higher. More. Don't stop."),
        ],
        "DDOG_LEGS": [
            ("STRAIGHT!", "Straighten those legs. No shortcuts."),
            ("EXTEND!", "Full extension. Drive the heels down."),
            ("LOCK 'EM", "Lock those legs out. Show them what discipline looks like."),
        ],
        "ESA_BEND": [
            ("REACH!", "Reach further! Your side body can take it."),
            ("PUSH IT", "Push deeper into that side bend. Feel the stretch."),
            ("MORE!", "More side bend. Don't play it safe."),
        ],
        "CLUNGE_DEEPEN": [
            ("DEEPER!", "Deeper into that lunge. No mercy on yourself."),
            ("DROP!", "Drop those hips. Every inch counts."),
            ("COMMIT!", "Commit to the depth. Halfway is nowhere."),
        ],
        "CLUNGE_UPRIGHT": [
            ("UPRIGHT!", "Stand tall in that lunge. Chest up."),
            ("DON'T LEAN", "Don't lean. Stay over your center."),
            ("SPINE UP", "Spine straight up. Warrior posture."),
        ],
        "HLUNGE_DEEPEN": [
            ("BEND!", "Bend that knee deeper. Feel the power."),
            ("GET LOW!", "Get low! Your legs are stronger than you think."),
            ("MORE!", "More depth. Push your limits."),
        ],
        "HLUNGE_UPRIGHT": [
            ("STAND TALL", "Stand tall. Don't collapse forward."),
            ("UPRIGHT!", "Upright! Own that lunge from the crown."),
            ("CHEST!", "Chest up. You're a tower, not a tent."),
        ],
        "MOUNTAIN_LEAN": [
            ("CENTER!", "Center yourself. A warrior stands balanced."),
            ("STAND!", "Stand like a monument. Straight and solid."),
            ("LOCK IN", "Lock in that center line. No wavering."),
        ],
        "MOUNTAIN_ALIGN": [
            ("ALIGN!", "Align those shoulders. Clean and sharp."),
            ("EVEN OUT", "Even it out. Symmetry is strength."),
            ("SQUARE!", "Square those shoulders. Perfect symmetry."),
        ],
        "TRI_LEG": [
            ("STRAIGHT!", "Straighten that leg. Full power."),
            ("LOCK IT", "Lock that front leg. Solid as iron."),
            ("EXTEND!", "Extend! No bend. Full commitment."),
        ],
        "TRI_BEND": [
            ("REACH DOWN", "Reach down further. Push the limit."),
            ("DEEPER!", "Deeper side bend. You've got it in you."),
            ("GO FOR IT", "Go for it. The floor is your target."),
        ],
    },
    "flow": {
        "WARRIOR_DEEPEN": [
            ("MELT DOWN", "Melt deeper into the earth. Feel the pull of gravity like warm honey."),
            ("POUR IN", "Pour yourself into the lunge. Every breath draws you closer to the ground."),
            ("DRIP DOWN", "Let your hips drip downward, slow and sweet."),
        ],
        "WARRIOR_EASE": [
            ("FLOAT UP", "Float upward. Gently. Like a bubble rising."),
            ("EASE", "Ease into a softer place. The body knows its limit."),
            ("LIGHTEN", "Lighten. Let the pose breathe with you."),
        ],
        "ARM_EXTEND": [
            ("UNFURL", "Unfurl your arms like silk ribbons caught in a breeze."),
            ("REACH", "Reach outward, fingertip to fingertip, painting the air."),
            ("EXPAND", "Expand like light through a prism. Arms wide, heart open."),
        ],
        "SHOULDERS_LEVEL": [
            ("SOFTEN", "Soften the shoulders. Let them melt like snow on a warm stone."),
            ("RELEASE", "Release the tension. Shoulders are soft, water flowing downhill."),
            ("EASE DOWN", "Let the shoulders ease down, heavy and warm."),
        ],
        "HIPS_LOWER": [
            ("SMOOTH LINE", "Smooth the line of your body. A river from crown to sole."),
            ("RIPPLE", "Let the hips ripple into alignment. No sharp edges."),
            ("LONG BODY", "Lengthen. Your body is a silken thread, taut and beautiful."),
        ],
        "CHAIR_DEEPEN": [
            ("SINK", "Sink lower, like settling into a warm bath."),
            ("HEAVY", "Let your hips grow heavy. Gravity is your partner."),
            ("DESCEND", "Descend slowly. Savor every inch of the journey down."),
        ],
        "CHAIR_EASE": [
            ("RISE", "Rise gently. Like steam curling upward."),
            ("FLOAT", "Float up slightly. Protect the beautiful machinery."),
        ],
        "CHAIR_UPRIGHT": [
            ("LIFT", "Lift through the chest. A flower opening to the sun."),
            ("UNFOLD", "Unfold the torso upward. Light and tall."),
            ("BLOOM", "Bloom upward from the core. The spine rises naturally."),
        ],
        "DDOG_PIKE": [
            ("RISE UP", "Let the hips rise like a wave cresting."),
            ("ARCH", "Arch upward. Your body draws a beautiful peak."),
            ("FLOAT HIPS", "Float the hips skyward. Light. Effortless."),
        ],
        "DDOG_LEGS": [
            ("LENGTHEN", "Lengthen through the legs. Feel them stretch like warm taffy."),
            ("SOFTEN", "Soften the knees, then slowly straighten. Feel each degree."),
            ("REACH", "Reach the heels toward the earth. A gentle, persistent pull."),
        ],
        "ESA_BEND": [
            ("POUR", "Pour to the side. Let the torso cascade like water over rock."),
            ("OPEN", "Open the side body. A door swinging wide to let the breeze in."),
            ("TILT", "Tilt with grace. The body is a compass needle finding its angle."),
        ],
        "CLUNGE_DEEPEN": [
            ("MELT", "Melt into the lunge. Every exhale takes you lower."),
            ("FLOW DOWN", "Flow downward. The hips are heavy, warm, surrendering."),
            ("SINK IN", "Sink in. Let the pose wrap around you like a warm embrace."),
        ],
        "CLUNGE_UPRIGHT": [
            ("RISE", "Rise through the crown. The spine is a ribbon of light."),
            ("LIFT", "Lift gently. The upper body floats while the lower body roots."),
            ("UNCOIL", "Uncoil the spine upward. Vertebra by vertebra."),
        ],
        "HLUNGE_DEEPEN": [
            ("DEEPER", "Deeper. Each breath melts you further into the shape."),
            ("SETTLE", "Settle into the lunge like sinking into soft earth."),
            ("DRIP", "Let the hips drip lower with each exhale."),
        ],
        "HLUNGE_UPRIGHT": [
            ("ASCEND", "Ascend through the torso. Light and effortless above."),
            ("RISE", "Rise from the waist. The spine curls upward like smoke."),
            ("LIFT", "Lift the heart. The upper body is weightless."),
        ],
        "MOUNTAIN_LEAN": [
            ("STILLNESS", "Find stillness. You are a candle flame in a room with no wind."),
            ("CENTER", "Center yourself. A single point of perfect calm."),
            ("BALANCE", "Balance like a raindrop on the tip of a leaf."),
        ],
        "MOUNTAIN_ALIGN": [
            ("EVEN", "Even the shoulders. Symmetry is the shape of serenity."),
            ("MIRROR", "Let one side mirror the other. Perfect, quiet reflection."),
            ("HARMONIZE", "Harmonize the shoulders. Left and right in gentle agreement."),
        ],
        "TRI_LEG": [
            ("LENGTHEN", "Lengthen the front leg. A smooth, unbroken line."),
            ("EXTEND", "Extend fully. The leg is a brushstroke on a canvas."),
            ("STRAIGHTEN", "Straighten with grace. No force, just intention."),
        ],
        "TRI_BEND": [
            ("CASCADE", "Cascade to the side. The body bends like a willow."),
            ("LEAN", "Lean into the beauty of the angle. Let it find you."),
            ("SIDE POUR", "Pour to the side. Effortless. Gravity does the work."),
        ],
    },
    "traditional": {
        "WARRIOR_DEEPEN": [
            ("VIRABHADRA", "Virabhadrasana. Deepen. Eka... dvi... trini."),
            ("STHIRA", "Sthira sukham asanam. Find steadiness. Go deeper."),
            ("DEEPEN", "Deepen the asana. The warrior's seat must be firm."),
        ],
        "WARRIOR_EASE": [
            ("AHIMSA", "Ahimsa. Non-violence to yourself. Ease from the depth."),
            ("SATYA", "Honor your truth. Pull back with awareness."),
            ("GENTLE", "Gently release some depth. Protect the body."),
        ],
        "ARM_EXTEND": [
            ("HASTA", "Hasta extension. Arms like rays from the heart center."),
            ("PRANA", "Channel prana through the fingertips. Extend fully."),
            ("REACH", "Reach to the cardinal points. Arms are your compass."),
        ],
        "SHOULDERS_LEVEL": [
            ("SAMA", "Sama. Evenness. Level the shoulders with equanimity."),
            ("BALANCE", "Balance the shoulder girdle. Krama by krama."),
            ("LEVEL", "Level the shoulders. Samasthiti in the upper body."),
        ],
        "HIPS_LOWER": [
            ("BANDHA", "Engage uddiyana bandha. Core draws in and up."),
            ("DANDA", "Dandasana alignment. One straight line, head to feet."),
            ("CORE", "Draw the navel toward the spine. Activate the core."),
        ],
        "CHAIR_DEEPEN": [
            ("UTKATASANA", "Utkatasana. The fierce pose. Sit deeper into your power."),
            ("AGNI", "Stoke the agni. Bend the knees. Build the internal fire."),
            ("DEEPEN", "Go deeper. Tapas. The heat of disciplined effort."),
        ],
        "CHAIR_EASE": [
            ("AHIMSA", "Ahimsa. Kindness to the knees. Rise slightly."),
            ("EASE", "Ease from the depth. Honor your prakriti."),
        ],
        "CHAIR_UPRIGHT": [
            ("URDHVA", "Urdhva. Upward energy through the spine."),
            ("DRISHTI", "Drishti forward. Lift the chest. Spine ascends."),
            ("TALL", "Elongate the spine. Crown reaches toward Brahman."),
        ],
        "DDOG_PIKE": [
            ("ADHO MUKHA", "Adho Mukha Svanasana. Press the hips upward and back."),
            ("INVERTED", "The inverted V. Sit bones rise to the sky."),
            ("HIPS UP", "Lift through the sit bones. The body forms a mountain."),
        ],
        "DDOG_LEGS": [
            ("PADA", "Pada bandha. Ground through the feet. Straighten the legs."),
            ("HEELS", "Press the heels toward prithvi. The earth element below."),
            ("EXTEND", "Extend the legs. Work toward full Adho Mukha."),
        ],
        "ESA_BEND": [
            ("PARSVA", "Utthita Parsvakonasana. Extend the lateral line."),
            ("SIDE BODY", "Open the parsva. The side body lengthens."),
            ("DEEPEN", "Deepen the konasana. Reach through the upper hasta."),
        ],
        "CLUNGE_DEEPEN": [
            ("ANJANEYA", "Anjaneyasana. Surrender the hips earthward."),
            ("DEEPEN", "Deepen the asana. Let gravity guide the pelvis."),
            ("LOW", "Lower into the crescent. The front knee bends deeply."),
        ],
        "CLUNGE_UPRIGHT": [
            ("URDHVA", "Urdhva hastasana alignment. Torso rises over the hips."),
            ("SPINE", "Lengthen the merudanda. The spinal column rises."),
            ("LIFT", "Lift from the muladhara. Root to rise."),
        ],
        "HLUNGE_DEEPEN": [
            ("DEEPEN", "Deepen the lunge. Eka pada alignment."),
            ("BEND", "Bend into the front leg. Steady and controlled."),
            ("LOWER", "Lower. The body descends with awareness."),
        ],
        "HLUNGE_UPRIGHT": [
            ("TADASANA", "Tadasana alignment in the upper body. Stack and lengthen."),
            ("VERTICAL", "Vertical spine. The axis of the body is sacred."),
            ("RISE", "Rise through the sushumna. Central channel ascends."),
        ],
        "MOUNTAIN_LEAN": [
            ("TADASANA", "Tadasana. The mountain is perfectly vertical."),
            ("SAMASTHITI", "Samasthiti. Equal standing. Find your center."),
            ("GROUNDED", "Root through all four corners of the feet. Stand firm."),
        ],
        "MOUNTAIN_ALIGN": [
            ("SAMA", "Sama. Evenness. Align the shoulders over the hips."),
            ("ALIGN", "Stack the body along the plumb line. Shoulder to hip."),
            ("BALANCE", "Find bilateral balance. The body is a sacred temple."),
        ],
        "TRI_LEG": [
            ("TRIKONASANA", "Trikonasana. The triangle demands a straight front leg."),
            ("EXTEND", "Extend fully. The leg is one side of the sacred geometry."),
            ("STRAIGHTEN", "Straighten the front leg. Honor the form of the triangle."),
        ],
        "TRI_BEND": [
            ("PARSVA", "Deepen the parsva bend. The triangle opens the side body."),
            ("LATERAL", "Lateral extension. Reach toward the lower hand."),
            ("KONASANA", "Trikonasana. Deepen the angle. The body becomes sacred shape."),
        ],
    },
}

# --- PERSONA-SPECIFIC POSITIVE FEEDBACK ---
PERSONA_POSITIVE = {
    "sage": [
        ("HARMONY", "Harmony. Your body speaks wisdom."),
        ("PRESENCE", "Beautiful presence. You are here completely."),
        ("STILLNESS", "Perfect stillness. The pose has found you."),
        ("WISDOM", "Your practice reflects deep wisdom."),
        ("PEACE", "Peace in every line of your body."),
        ("CENTERED", "Centered. The world revolves around your stillness."),
    ],
    "scientist": [
        ("OPTIMAL", "Optimal alignment achieved. All angles within target range."),
        ("TEXTBOOK", "Textbook form. Biomechanically sound."),
        ("PERFECT", "Perfect joint angles. Maintaining ideal range."),
        ("PRECISE", "Precise execution. All metrics in the green zone."),
        ("ALIGNED", "Alignment verified. Excellent bilateral symmetry."),
        ("MEASURED", "Measured and exact. Your form is a reference model."),
    ],
    "warrior": [
        ("YES!", "Yes! That's it. Crushing it."),
        ("UNSTOPPABLE", "Unstoppable. That's world-class form."),
        ("BEAST MODE", "Beast mode. Absolutely locked in."),
        ("DOMINANT", "Dominant. You own this pose."),
        ("FIERCE!", "Fierce. Not a single weakness showing."),
        ("POWERFUL", "Powerful. That's what strength looks like."),
    ],
    "flow": [
        ("LIKE WATER", "Like water. Effortless beauty in every line."),
        ("POETRY", "You're poetry in motion. Breathtaking."),
        ("BEAUTIFUL", "Beautiful. A painting come to life."),
        ("GRACEFUL", "Graceful. Every line sings."),
        ("SERENE", "Serene. You've found the stillness between breaths."),
        ("EXQUISITE", "Exquisite. The pose flows through you like music."),
    ],
    "traditional": [
        ("BAHUT ACHHA", "Bahut achha. Very good. Hold the asana."),
        ("STHIRA", "Sthira. Steady and firm. Perfect."),
        ("NAMASTE", "Namaste. The light in your practice shines."),
        ("OM SHANTI", "Om shanti. Peace in the pose."),
        ("SADHU", "Sadhu. Well done. The form is correct."),
        ("SUNDARA", "Sundara. Beautiful asana. Maintain."),
    ],
}

class PoseHeuristics:
    @staticmethod
    def evaluate(pose_name, landmarks, intensity=2, persona='default'):
        """
        Input:
          landmarks (dict): {mp_pose.PoseLandmark: [x, y]}
          intensity (int): 1=gentle (wider thresholds), 2=standard, 3=intense (tighter)
          persona (str): 'default', 'sage', 'scientist', 'warrior', 'flow', 'traditional'
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
            return check_fn(landmarks, intensity, persona)
        return None

    @staticmethod
    def _scale_upper(value, intensity):
        """Scale an upper-bound threshold. Gentle=wider (more forgiving), intense=tighter."""
        if intensity == 1:
            return value * 1.15
        elif intensity == 3:
            return value * 0.85
        return value

    @staticmethod
    def _scale_lower(value, intensity):
        """Scale a lower-bound threshold. Gentle=wider (more forgiving), intense=tighter."""
        if intensity == 1:
            return value * 0.85
        elif intensity == 3:
            return value * 1.15
        return value

    @staticmethod
    def get_advice(key, default_short, default_long, persona='default'):
        """Helper to get random advice from library, with persona support.
        Falls back to default COACHING_LIBRARY if persona doesn't have the key."""
        if persona != 'default' and persona in PERSONA_COACHING:
            persona_lib = PERSONA_COACHING[persona]
            if key in persona_lib:
                return random.choice(persona_lib[key])
        # Default fallback
        if key in COACHING_LIBRARY:
            return random.choice(COACHING_LIBRARY[key])
        return (default_short, default_long)

    @staticmethod
    def _positive(persona='default'):
        """Return positive feedback when form is good, with persona support."""
        if persona != 'default' and persona in PERSONA_POSITIVE:
            advice = random.choice(PERSONA_POSITIVE[persona])
        else:
            advice = PoseHeuristics.get_advice("FORM_EXCELLENT", "GREAT", "Looking good.")
        return {
            'text': advice,
            'vector': None,
            'color': None,
            'positive': True,
        }

    @staticmethod
    def check_warrior_ii(landmarks, intensity=2, persona='default'):
        s_upper = PoseHeuristics._scale_upper
        s_lower = PoseHeuristics._scale_lower
        l_knee_ang = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
        r_knee_ang = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
        front_knee_idx = 25 if l_knee_ang < 135 else 26
        angle = l_knee_ang if front_knee_idx == 25 else r_knee_ang
        knee_pt = landmarks[front_knee_idx]

        correction = None

        if angle > s_upper(110, intensity):
            advice = PoseHeuristics.get_advice("WARRIOR_DEEPEN", "DEEPEN", "Deepen your lunge.", persona)
            correction = {
                'text': advice,
                'vector': (tuple(knee_pt), (knee_pt[0], knee_pt[1] + 0.15)),
                'color': (0, 255, 255)
            }
        elif angle < s_lower(75, intensity):
             advice = PoseHeuristics.get_advice("WARRIOR_EASE", "EASE UP", "Ease up.", persona)
             correction = {
                'text': advice,
                'vector': (tuple(knee_pt), (knee_pt[0], knee_pt[1] - 0.1)),
                'color': (0, 255, 255)
            }

        l_arm_ang = calculate_angle(landmarks[11], landmarks[13], landmarks[15])
        if not correction and l_arm_ang < s_lower(150, intensity):
             advice = PoseHeuristics.get_advice("ARM_EXTEND", "EXTEND", "Extend left arm.", persona)
             correction = {
                'text': advice,
                'vector': (tuple(landmarks[13]), (landmarks[13][0]-0.1, landmarks[13][1])),
                'color': (0, 255, 255)
             }

        return correction if correction else PoseHeuristics._positive(persona)

    @staticmethod
    def check_tree(landmarks, intensity=2, persona='default'):
        s_upper = PoseHeuristics._scale_upper
        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]
        if abs(l_shoulder[1] - r_shoulder[1]) > s_upper(0.05, intensity):
            lower = 11 if l_shoulder[1] > r_shoulder[1] else 12
            pt = landmarks[lower]
            advice = PoseHeuristics.get_advice("SHOULDERS_LEVEL", "LEVEL SHLDR", "Level shoulders.", persona)
            return {
                'text': advice,
                'vector': (tuple(pt), (pt[0], pt[1] - 0.1)),
                'color': (0, 255, 255)
            }
        return PoseHeuristics._positive(persona)

    @staticmethod
    def check_plank(landmarks, intensity=2, persona='default'):
         s_lower = PoseHeuristics._scale_lower
         l_hip_ang = calculate_angle(landmarks[11], landmarks[23], landmarks[27])
         r_hip_ang = calculate_angle(landmarks[12], landmarks[24], landmarks[28])
         avg = (l_hip_ang + r_hip_ang) / 2
         if avg < s_lower(160, intensity):
             pt = landmarks[23]
             advice = PoseHeuristics.get_advice("HIPS_LOWER", "LOWER HIPS", "Lower hips.", persona)
             return {
                 'text': advice,
                 'vector': (tuple(pt), (pt[0], pt[1] + 0.1)),
                 'color': (0, 255, 255)
             }
         return PoseHeuristics._positive(persona)

    @staticmethod
    def check_crescent_lunge(landmarks, intensity=2, persona='default'):
        s_upper = PoseHeuristics._scale_upper
        # Front knee — ideal 130-170° (POSE_PROFILES: 0.72-0.94)
        l_knee = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
        r_knee = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
        front_idx = 25 if l_knee < r_knee else 26
        front_angle = l_knee if front_idx == 25 else r_knee
        knee_pt = landmarks[front_idx]

        if front_angle > s_upper(170, intensity):
            advice = PoseHeuristics.get_advice("CLUNGE_DEEPEN", "DEEPEN", "Sink deeper into your lunge.", persona)
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
        if dx > s_upper(0.1, intensity):
            advice = PoseHeuristics.get_advice("CLUNGE_UPRIGHT", "UPRIGHT", "Lift chest upright.", persona)
            return {
                'text': advice,
                'vector': (tuple(mid_shoulder), (mid_shoulder[0], mid_shoulder[1] - 0.12)),
                'color': (0, 255, 255)
            }
        return PoseHeuristics._positive(persona)

    @staticmethod
    def check_chair(landmarks, intensity=2, persona='default'):
        s_upper = PoseHeuristics._scale_upper
        s_lower = PoseHeuristics._scale_lower
        # Knee flexion — ideal ~165° (POSE_PROFILES: 0.89-0.94 normalized)
        l_knee = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
        r_knee = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
        avg_knee = (l_knee + r_knee) / 2
        knee_pt = landmarks[25]

        if avg_knee > s_upper(170, intensity):
            advice = PoseHeuristics.get_advice("CHAIR_DEEPEN", "DEEPER", "Sit deeper.", persona)
            return {
                'text': advice,
                'vector': (tuple(knee_pt), (knee_pt[0], knee_pt[1] + 0.12)),
                'color': (0, 255, 255)
            }
        if avg_knee < s_lower(130, intensity):
            advice = PoseHeuristics.get_advice("CHAIR_EASE", "EASE UP", "Come up a bit.", persona)
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
        if dx > s_upper(0.08, intensity):
            advice = PoseHeuristics.get_advice("CHAIR_UPRIGHT", "UPRIGHT", "Lift your chest.", persona)
            return {
                'text': advice,
                'vector': (tuple(mid_shoulder), (mid_shoulder[0], mid_shoulder[1] - 0.12)),
                'color': (0, 255, 255)
            }
        return PoseHeuristics._positive(persona)

    @staticmethod
    def check_downward_dog(landmarks, intensity=2, persona='default'):
        s_upper = PoseHeuristics._scale_upper
        s_lower = PoseHeuristics._scale_lower
        # Hip angle — ideal very acute (~14°, POSE_PROFILES: 0.04-0.11)
        l_hip = calculate_angle(landmarks[11], landmarks[23], landmarks[25])
        r_hip = calculate_angle(landmarks[12], landmarks[24], landmarks[26])
        avg_hip = (l_hip + r_hip) / 2
        hip_pt = landmarks[23]

        if avg_hip > s_upper(45, intensity):
            advice = PoseHeuristics.get_advice("DDOG_PIKE", "HIPS UP", "Push hips higher.", persona)
            return {
                'text': advice,
                'vector': (tuple(hip_pt), (hip_pt[0], hip_pt[1] - 0.15)),
                'color': (0, 255, 255)
            }

        # Knee straightness — ideal ~166° (POSE_PROFILES: 0.85-1.0)
        l_knee = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
        r_knee = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
        avg_knee = (l_knee + r_knee) / 2
        if avg_knee < s_lower(150, intensity):
            knee_pt = landmarks[25]
            advice = PoseHeuristics.get_advice("DDOG_LEGS", "STRAIGHTEN", "Straighten legs.", persona)
            return {
                'text': advice,
                'vector': (tuple(knee_pt), (knee_pt[0], knee_pt[1] + 0.1)),
                'color': (0, 255, 255)
            }
        return PoseHeuristics._positive(persona)

    @staticmethod
    def check_extended_side_angle(landmarks, intensity=2, persona='default'):
        s_upper = PoseHeuristics._scale_upper
        # Lateral flexion — ideal ~57° (POSE_PROFILES: 0.29-0.35)
        # Approximate via shoulder-hip angle offset
        l_shoulder = landmarks[11]
        l_hip = landmarks[23]
        dy = abs(l_shoulder[1] - l_hip[1])
        dx = abs(l_shoulder[0] - l_hip[0])
        if dy > 0 and dx / dy < s_upper(0.4, intensity):
            advice = PoseHeuristics.get_advice("ESA_BEND", "SIDE BEND", "Deepen side bend.", persona)
            mid_torso = [(l_shoulder[0] + l_hip[0]) / 2,
                         (l_shoulder[1] + l_hip[1]) / 2]
            return {
                'text': advice,
                'vector': (tuple(mid_torso), (mid_torso[0] - 0.12, mid_torso[1] + 0.06)),
                'color': (0, 255, 255)
            }
        return PoseHeuristics._positive(persona)

    @staticmethod
    def check_high_lunge(landmarks, intensity=2, persona='default'):
        s_upper = PoseHeuristics._scale_upper
        # Front knee — ideal 130-170° (POSE_PROFILES: 0.72-0.94)
        l_knee = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
        r_knee = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
        front_idx = 25 if l_knee < r_knee else 26
        front_angle = l_knee if front_idx == 25 else r_knee
        knee_pt = landmarks[front_idx]

        if front_angle > s_upper(170, intensity):
            advice = PoseHeuristics.get_advice("HLUNGE_DEEPEN", "DEEPEN", "Bend front knee deeper.", persona)
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
        if dx > s_upper(0.1, intensity):
            advice = PoseHeuristics.get_advice("HLUNGE_UPRIGHT", "UPRIGHT", "Lift chest upright.", persona)
            return {
                'text': advice,
                'vector': (tuple(mid_shoulder), (mid_shoulder[0], mid_shoulder[1] - 0.12)),
                'color': (0, 255, 255)
            }
        return PoseHeuristics._positive(persona)

    @staticmethod
    def check_mountain(landmarks, intensity=2, persona='default'):
        s_upper = PoseHeuristics._scale_upper
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
        if lateral_offset > s_upper(0.04, intensity):
            pt = [shoulder_mid_x, shoulder_mid_y]
            advice = PoseHeuristics.get_advice("MOUNTAIN_LEAN", "STAND TALL", "Stand tall.", persona)
            return {
                'text': advice,
                'vector': (tuple(pt), (hip_mid_x, pt[1])),
                'color': (0, 255, 255)
            }

        # Shoulder-hip alignment — ideal 0.95-1.0
        shoulder_diff = abs(l_shoulder[1] - r_shoulder[1])
        if shoulder_diff > s_upper(0.04, intensity):
            lower = 11 if l_shoulder[1] > r_shoulder[1] else 12
            pt = landmarks[lower]
            advice = PoseHeuristics.get_advice("MOUNTAIN_ALIGN", "ALIGN", "Align shoulders.", persona)
            return {
                'text': advice,
                'vector': (tuple(pt), (pt[0], pt[1] - 0.08)),
                'color': (0, 255, 255)
            }
        return PoseHeuristics._positive(persona)

    @staticmethod
    def check_triangle(landmarks, intensity=2, persona='default'):
        s_lower = PoseHeuristics._scale_lower
        s_upper = PoseHeuristics._scale_upper
        # Front knee straightness — ideal ~179° (POSE_PROFILES: 0.97-1.0)
        l_knee = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
        r_knee = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
        # Front leg is the one with more bend
        front_idx = 25 if l_knee < r_knee else 26
        front_angle = l_knee if front_idx == 25 else r_knee
        if front_angle < s_lower(165, intensity):
            knee_pt = landmarks[front_idx]
            advice = PoseHeuristics.get_advice("TRI_LEG", "STRAIGHTEN", "Straighten front leg.", persona)
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
        if dy > 0 and dx / dy < s_upper(0.4, intensity):
            mid = [(l_shoulder[0] + l_hip[0]) / 2, (l_shoulder[1] + l_hip[1]) / 2]
            advice = PoseHeuristics.get_advice("TRI_BEND", "SIDE BEND", "Deepen side bend.", persona)
            return {
                'text': advice,
                'vector': (tuple(mid), (mid[0] - 0.1, mid[1] + 0.06)),
                'color': (0, 255, 255)
            }
        return PoseHeuristics._positive(persona)
