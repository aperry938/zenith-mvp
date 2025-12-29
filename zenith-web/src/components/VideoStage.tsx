import React, { useRef, useEffect } from 'react';

interface VideoStageProps {
    onFrame: (blob: Blob) => void;
}

export const VideoStage: React.FC<VideoStageProps> = ({ onFrame }) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const startCamera = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480 }
                });
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                    videoRef.current.play();
                }
            } catch (err) {
                console.error("Camera Error:", err);
            }
        };
        startCamera();
    }, []);

    useEffect(() => {
        const interval = setInterval(() => {
            if (videoRef.current && canvasRef.current) {
                const ctx = canvasRef.current.getContext('2d');
                if (ctx) {
                    ctx.drawImage(videoRef.current, 0, 0, 640, 480);
                    canvasRef.current.toBlob((blob) => {
                        if (blob) onFrame(blob);
                    }, 'image/jpeg', 0.8);
                }
            }
        }, 33);

        return () => clearInterval(interval);
    }, [onFrame]);

    return (
        <>
            <video ref={videoRef} className="hidden" playsInline muted />
            <canvas
                ref={canvasRef}
                width={640}
                height={480}
                className="w-[640px] h-[480px] border-4 border-zinc-800 rounded shadow-2xl scale-x-[-1]"
            />
        </>
    );
};
