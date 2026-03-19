import React, { useRef, useEffect, useState } from 'react';

interface VideoStageProps {
    onFrame: (blob: Blob) => void;
}

export const VideoStage: React.FC<VideoStageProps> = ({ onFrame }) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [cameraError, setCameraError] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        const startCamera = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480 }
                });
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                    videoRef.current.play();
                    setIsLoading(false);
                }
            } catch (err) {
                console.error("Camera Error:", err);
                setIsLoading(false);
                if (err instanceof DOMException) {
                    if (err.name === 'NotAllowedError') {
                        setCameraError("Camera access denied. Please allow camera permissions and reload.");
                    } else if (err.name === 'NotFoundError') {
                        setCameraError("No camera detected. Please connect a camera and reload.");
                    } else {
                        setCameraError(`Camera error: ${err.message}`);
                    }
                } else {
                    setCameraError("Failed to initialize camera.");
                }
            }
        };
        startCamera();
    }, []);

    useEffect(() => {
        const interval = setInterval(() => {
            if (videoRef.current && canvasRef.current && !cameraError) {
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
    }, [onFrame, cameraError]);

    if (cameraError) {
        return (
            <div className="w-[640px] h-[480px] border-4 border-red-500/30 rounded shadow-2xl bg-zenith-panel flex flex-col items-center justify-center gap-4 px-8">
                <svg className="w-12 h-12 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M12 18.75H4.5a2.25 2.25 0 01-2.25-2.25V7.5A2.25 2.25 0 014.5 5.25h7.5" />
                    <line x1="2" y1="2" x2="22" y2="22" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" />
                </svg>
                <p className="text-red-400 text-sm text-center font-mono">{cameraError}</p>
            </div>
        );
    }

    if (isLoading) {
        return (
            <div className="w-[640px] h-[480px] border-4 border-zinc-800 rounded shadow-2xl bg-zenith-panel flex flex-col items-center justify-center gap-3">
                <div className="w-8 h-8 border-2 border-zinc-600 border-t-zenith-neonBlue rounded-full animate-spin" />
                <p className="text-zinc-500 text-xs font-mono tracking-wider">Initializing camera...</p>
            </div>
        );
    }

    return (
        <>
            <video ref={videoRef} className="hidden" playsInline muted />
            <canvas
                ref={canvasRef}
                width={640}
                height={480}
                className="w-[640px] h-[480px] border-4 border-zinc-800 rounded shadow-2xl"
            />
        </>
    );
};
