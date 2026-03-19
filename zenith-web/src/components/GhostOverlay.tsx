import React, { useRef, useEffect } from 'react';
import { drawSkeleton, drawAngles, flatToLandmarks, type Landmark } from '../utils/drawing';

interface CorrectionVector {
    start: number[];
    end: number[];
}

interface GhostOverlayProps {
    landmarks: Landmark[] | null;
    ghostFlat: number[] | null;
    correctionVector?: CorrectionVector | null;
    correctionColor?: number[] | null;
}

function drawArrow(
    ctx: CanvasRenderingContext2D,
    start: number[],
    end: number[],
    color: number[],
    w: number,
    h: number
) {
    const sx = start[0] * w;
    const sy = start[1] * h;
    const ex = end[0] * w;
    const ey = end[1] * h;

    const cssColor = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;

    ctx.save();
    ctx.strokeStyle = cssColor;
    ctx.fillStyle = cssColor;
    ctx.lineWidth = 3;
    ctx.shadowColor = cssColor;
    ctx.shadowBlur = 12;

    // Shaft
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(ex, ey);
    ctx.stroke();

    // Arrowhead
    const angle = Math.atan2(ey - sy, ex - sx);
    const headLen = 12;
    ctx.beginPath();
    ctx.moveTo(ex, ey);
    ctx.lineTo(ex - headLen * Math.cos(angle - Math.PI / 6), ey - headLen * Math.sin(angle - Math.PI / 6));
    ctx.moveTo(ex, ey);
    ctx.lineTo(ex - headLen * Math.cos(angle + Math.PI / 6), ey - headLen * Math.sin(angle + Math.PI / 6));
    ctx.stroke();

    ctx.restore();
}

export const GhostOverlay: React.FC<GhostOverlayProps> = ({
    landmarks,
    ghostFlat,
    correctionVector,
    correctionColor,
}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw Ghost (Cyan)
        if (ghostFlat && ghostFlat.length > 0) {
            const ghostLms = flatToLandmarks(ghostFlat);
            drawSkeleton(ctx, ghostLms, 'cyan', 640, 480);
        }

        // Draw User skeleton + angle annotations
        if (landmarks && landmarks.length > 0) {
            drawSkeleton(ctx, landmarks, '#00ff99', 640, 480);
            drawAngles(ctx, landmarks, 640, 480);
        }

        // Draw Correction Vector Arrow
        if (correctionVector && correctionColor) {
            drawArrow(ctx, correctionVector.start, correctionVector.end, correctionColor, 640, 480);
        }

    }, [landmarks, ghostFlat, correctionVector, correctionColor]);

    return (
        <canvas
            ref={canvasRef}
            width={640}
            height={480}
            className="absolute inset-0 w-[640px] h-[480px] pointer-events-none z-10"
        />
    );
};
