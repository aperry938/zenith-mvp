import React, { useRef, useEffect } from 'react';
import { drawSkeleton, drawAngles, flatToLandmarks, type Landmark } from '../utils/drawing';

interface GhostOverlayProps {
    landmarks: Landmark[] | null;
    ghostFlat: number[] | null;
}

export const GhostOverlay: React.FC<GhostOverlayProps> = ({ landmarks, ghostFlat }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Clear
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

    }, [landmarks, ghostFlat]);

    return (
        <canvas
            ref={canvasRef}
            width={640}
            height={480}
            className="absolute inset-0 w-[640px] h-[480px] pointer-events-none z-10"
        />
    );
};
