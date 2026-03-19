// MediaPipe Pose Connections
export const POSE_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10],
    [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
    [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20], [11, 23],
    [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28], [27, 29],
    [28, 30], [29, 31], [30, 32], [27, 31], [28, 32]
];

export interface Landmark {
    x: number;
    y: number;
    z: number;
    visibility: number;
}

export function drawSkeleton(
    ctx: CanvasRenderingContext2D,
    landmarks: Landmark[],
    color: string = '#00ff99',
    width: number = 640,
    height: number = 480
) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.lineCap = 'round';

    // Draw Lines
    for (const [start, end] of POSE_CONNECTIONS) {
        const l1 = landmarks[start];
        const l2 = landmarks[end];

        if (l1 && l2 && l1.visibility > 0.5 && l2.visibility > 0.5) {
            ctx.beginPath();
            ctx.moveTo(l1.x * width, l1.y * height);
            ctx.lineTo(l2.x * width, l2.y * height);
            ctx.stroke();
        }
    }

    // Draw Dots (larger, with glow)
    for (const lm of landmarks) {
        if (lm.visibility > 0.5) {
            // Glow
            ctx.fillStyle = color + '40';
            ctx.beginPath();
            ctx.arc(lm.x * width, lm.y * height, 6, 0, 2 * Math.PI);
            ctx.fill();
            // Core dot
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(lm.x * width, lm.y * height, 3, 0, 2 * Math.PI);
            ctx.fill();
        }
    }
}

// Key joint angle definitions: [pointA, joint, pointB, label]
const ANGLE_JOINTS: [number, number, number, string][] = [
    [11, 13, 15, 'L Elbow'],   // L shoulder -> L elbow -> L wrist
    [12, 14, 16, 'R Elbow'],   // R shoulder -> R elbow -> R wrist
    [13, 11, 23, 'L Shoulder'], // L elbow -> L shoulder -> L hip
    [14, 12, 24, 'R Shoulder'], // R elbow -> R shoulder -> R hip
    [11, 23, 25, 'L Hip'],     // L shoulder -> L hip -> L knee
    [12, 24, 26, 'R Hip'],     // R shoulder -> R hip -> R knee
    [23, 25, 27, 'L Knee'],    // L hip -> L knee -> L ankle
    [24, 26, 28, 'R Knee'],    // R hip -> R knee -> R ankle
];

function computeAngle(a: Landmark, b: Landmark, c: Landmark): number {
    const ba = { x: a.x - b.x, y: a.y - b.y };
    const bc = { x: c.x - b.x, y: c.y - b.y };
    const dot = ba.x * bc.x + ba.y * bc.y;
    const magBA = Math.sqrt(ba.x * ba.x + ba.y * ba.y);
    const magBC = Math.sqrt(bc.x * bc.x + bc.y * bc.y);
    if (magBA === 0 || magBC === 0) return 0;
    const cosAngle = Math.max(-1, Math.min(1, dot / (magBA * magBC)));
    return Math.round((Math.acos(cosAngle) * 180) / Math.PI);
}

export function drawAngles(
    ctx: CanvasRenderingContext2D,
    landmarks: Landmark[],
    width: number = 640,
    height: number = 480
) {
    ctx.font = 'bold 10px monospace';
    ctx.textAlign = 'center';

    for (const [aIdx, bIdx, cIdx, _label] of ANGLE_JOINTS) {
        const a = landmarks[aIdx];
        const b = landmarks[bIdx];
        const c = landmarks[cIdx];

        if (!a || !b || !c) continue;
        if (a.visibility < 0.5 || b.visibility < 0.5 || c.visibility < 0.5) continue;

        const angle = computeAngle(a, b, c);
        const x = b.x * width;
        const y = b.y * height;

        // Draw angle arc
        const arcRadius = 15;
        const startAngle = Math.atan2(a.y - b.y, a.x - b.x);
        const endAngle = Math.atan2(c.y - b.y, c.x - b.x);
        ctx.strokeStyle = '#ffffff60';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(x, y, arcRadius, startAngle, endAngle);
        ctx.stroke();

        // Draw angle label
        const offsetX = ((a.x + c.x) / 2 - b.x) * width * 0.4;
        const offsetY = ((a.y + c.y) / 2 - b.y) * height * 0.4;
        const labelX = x + offsetX;
        const labelY = y + offsetY;

        // Background
        ctx.fillStyle = '#000000aa';
        const textWidth = ctx.measureText(`${angle}`).width;
        ctx.fillRect(labelX - textWidth / 2 - 3, labelY - 7, textWidth + 6, 14);

        // Text
        ctx.fillStyle = '#ffffff';
        ctx.fillText(`${angle}°`, labelX, labelY + 3);
    }
}

export function flatToLandmarks(flat: number[]): Landmark[] {
    const lms: Landmark[] = [];
    // 33 landmarks * 4 (x,y,z,v) = 132
    for (let i = 0; i < flat.length; i += 4) {
        lms.push({
            x: flat[i],
            y: flat[i + 1],
            z: flat[i + 2],
            visibility: flat[i + 3]
        });
    }
    return lms;
}
