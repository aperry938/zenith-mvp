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
    ctx.lineWidth = 2;

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

    // Draw Dots
    ctx.fillStyle = color;
    for (const lm of landmarks) {
        if (lm.visibility > 0.5) {
            ctx.beginPath();
            ctx.arc(lm.x * width, lm.y * height, 3, 0, 2 * Math.PI);
            ctx.fill();
        }
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
