import React, { useEffect, useState } from 'react';
import { useZenithVoice } from '../hooks/useZenithVoice';
import { InfoTooltip } from './InfoTooltip';

interface ZenithMetrics {
    label: string | null;
    confidence?: number;
    form_assessment?: string;
    flow: number;
    velocity: number;
    q: number;
}

interface HeuristicCorrection {
    hud: string;
    spoken: string;
    speak: boolean;
}

interface HUDProps {
    metrics: ZenithMetrics | null;
    advice: string | null;
    heuristicCorrection: HeuristicCorrection | null;
    onRequestAnalysis: () => void;
}

export const HUD: React.FC<HUDProps> = ({ metrics, advice, heuristicCorrection, onRequestAnalysis }) => {
    const { speak } = useZenithVoice();
    const [visibleAdvice, setVisibleAdvice] = useState<string | null>(null);
    const [visibleHeuristic, setVisibleHeuristic] = useState<string | null>(null);

    // Trigger speech and auto-dismiss when advice updates
    useEffect(() => {
        if (advice) {
            speak(advice);
            setVisibleAdvice(advice);
            const timer = setTimeout(() => setVisibleAdvice(null), 10000);
            return () => clearTimeout(timer);
        }
    }, [advice, speak]);

    // Heuristic coaching corrections
    useEffect(() => {
        if (heuristicCorrection) {
            setVisibleHeuristic(heuristicCorrection.hud);
            if (heuristicCorrection.speak) {
                speak(heuristicCorrection.spoken);
            }
        } else {
            setVisibleHeuristic(null);
        }
    }, [heuristicCorrection, speak]);

    return (
        <>
            <div className="absolute top-5 right-5 w-52 flex flex-col gap-4 pointer-events-none">
                <div className="relative z-20 bg-zenith-panel/85 border border-zinc-800 p-4 rounded-lg backdrop-blur-sm pointer-events-auto transition-all hover:bg-zenith-panel/95">
                    <h3 className="m-0 text-xs text-zinc-400 tracking-widest mb-1 flex items-center">
                        POSE
                        <InfoTooltip text="Random Forest classifier identifies which of 10 yoga poses you're performing. Shows 'No Pose' when confidence is below 60%." />
                    </h3>
                    <div className="text-3xl font-bold font-mono text-zenith-neonBlue drop-shadow-[0_0_10px_rgba(0,204,255,0.3)]">
                        {metrics?.label || <span className="text-zinc-600 text-lg">Step into frame</span>}
                    </div>
                    {metrics?.confidence != null && metrics.label && (
                        <div className="text-xs text-zinc-400 font-mono mt-0.5">
                            {(metrics.confidence * 100).toFixed(0)}% confidence
                        </div>
                    )}
                    {metrics?.form_assessment && (
                        <div className={`text-xs font-bold mt-1 ${metrics.form_assessment === 'Correct' ? 'text-green-400' : 'text-red-400'}`}>
                            {metrics.form_assessment} Form
                        </div>
                    )}
                </div>

                <div className="relative bg-zenith-panel/85 border border-zinc-800 p-4 rounded-lg backdrop-blur-sm pointer-events-auto z-20" role="status" aria-label="Flow score">
                    <h3 className="m-0 text-xs text-zinc-400 tracking-widest mb-1 flex items-center">
                        FLOW
                        <InfoTooltip text="Movement smoothness (0-100). Computed from joint angular velocity jerk. High scores indicate steady, controlled movement. Low scores indicate jerky transitions." />
                    </h3>
                    <div className="text-3xl font-bold font-mono text-zenith-neonPurple drop-shadow-[0_0_10px_rgba(214,51,255,0.3)]">
                        {metrics?.flow?.toFixed(0) || "--"}
                    </div>
                    <div className="mt-2 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                        <div
                            className={`h-full rounded-full transition-all duration-300 ${
                                (metrics?.flow ?? 0) >= 80 ? 'bg-purple-400' :
                                (metrics?.flow ?? 0) >= 50 ? 'bg-yellow-500' :
                                (metrics?.flow ?? 0) >= 30 ? 'bg-orange-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${Math.min(100, metrics?.flow ?? 0)}%` }}
                        />
                    </div>
                </div>

                <div className="relative z-20 bg-zenith-panel/85 border border-zinc-800 p-4 rounded-lg backdrop-blur-sm pointer-events-auto">
                    <h3 className="m-0 text-xs text-zinc-400 tracking-widest mb-1 flex items-center">
                        QUALITY
                        <InfoTooltip text="VAE reconstruction score (0-100). A Variational Autoencoder trained only on correct-form poses measures how closely your form matches the learned 'good form' manifold." />
                    </h3>
                    <div className="text-3xl font-bold font-mono text-zenith-neonGreen drop-shadow-[0_0_10px_rgba(0,255,153,0.3)]">
                        {metrics?.q?.toFixed(0) || "--"}
                    </div>
                </div>

                {/* Stability Indicator */}
                {metrics && (
                    <div className="relative z-20 bg-zenith-panel/85 border border-zinc-800 px-4 py-2 rounded-lg backdrop-blur-sm pointer-events-auto">
                        {(() => {
                            const vel = metrics.velocity ?? 1;
                            if (vel < 0.05) return (
                                <div className="flex items-center gap-2">
                                    <div className="w-2 h-2 rounded-full bg-green-400 shadow-[0_0_6px_rgba(74,222,128,0.6)]" />
                                    <span className="text-green-400 text-xs font-bold font-mono tracking-widest">LOCKED</span>
                                </div>
                            );
                            if (vel < 0.15) return (
                                <div className="flex items-center gap-2">
                                    <div className="w-2 h-2 rounded-full bg-yellow-500" />
                                    <span className="text-yellow-500 text-xs font-mono tracking-widest">STEADY</span>
                                </div>
                            );
                            return (
                                <div className="flex items-center gap-2">
                                    <div className="w-2 h-2 rounded-full bg-zinc-600" />
                                    <span className="text-zinc-500 text-xs font-mono tracking-widest">MOVING</span>
                                </div>
                            );
                        })()}
                    </div>
                )}

                {/* Heuristic Correction Badge */}
                {visibleHeuristic && (
                    <div className="relative z-20 bg-zenith-neonBlue/10 border border-zenith-neonBlue/50 p-3 rounded-lg backdrop-blur-sm shadow-[0_0_15px_rgba(0,204,255,0.15)]">
                        <div className="flex items-center justify-between">
                            <span className="text-zenith-neonBlue text-sm font-bold font-mono tracking-wider">
                                {visibleHeuristic}
                            </span>
                            <button
                                onClick={() => setVisibleHeuristic(null)}
                                className="text-zinc-500 hover:text-white text-xs ml-2 pointer-events-auto cursor-pointer"
                            >
                                ✕
                            </button>
                        </div>
                    </div>
                )}

                <button
                    onClick={onRequestAnalysis}
                    aria-label="Request AI form analysis"
                    className="relative z-20 w-full px-3 py-2 rounded border border-zinc-700 bg-zenith-panel/85 backdrop-blur-sm text-xs font-bold tracking-widest uppercase text-zinc-400 hover:text-zenith-neonPurple hover:border-zenith-neonPurple/50 transition-all pointer-events-auto cursor-pointer"
                >
                    Analyze Form
                </button>
            </div>

            {visibleAdvice && (
                <div className="absolute bottom-12 left-1/2 -translate-x-1/2 w-full max-w-2xl px-6 pointer-events-auto">
                    <div className="bg-black/90 border-l-4 border-zenith-neonPurple p-6 rounded-r-lg shadow-2xl backdrop-blur-md">
                        <div className="flex items-start justify-between">
                            <div>
                                <h4 className="text-zenith-neonPurple text-xs font-bold tracking-widest mb-2 uppercase">Coach Feedback</h4>
                                <p className="text-gray-200 text-lg leading-relaxed font-light">{visibleAdvice}</p>
                            </div>
                            <button
                                onClick={() => setVisibleAdvice(null)}
                                className="text-zinc-500 hover:text-white text-sm ml-4 flex-shrink-0 cursor-pointer"
                            >
                                ✕
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
};
