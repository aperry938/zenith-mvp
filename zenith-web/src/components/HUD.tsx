import React from 'react';

interface ZenithMetrics {
    label: string;
    flow: number;
    velocity: number;
    q: number;
}

interface HUDProps {
    metrics: ZenithMetrics | null;
    advice: string | null;
    onRequestAnalysis: () => void;
}

export const HUD: React.FC<HUDProps> = ({ metrics, advice, onRequestAnalysis }) => {
    return (
        <>
            <div className="absolute top-5 right-5 w-52 flex flex-col gap-4 pointer-events-none">
                <div className="bg-zenith-panel/85 border border-zinc-800 p-4 rounded-lg backdrop-blur-sm pointer-events-auto transition-all hover:bg-zenith-panel/95">
                    <h3 className="m-0 text-xs text-zinc-400 tracking-widest mb-1">POSE</h3>
                    <div className="text-3xl font-bold font-mono text-zenith-neonBlue drop-shadow-[0_0_10px_rgba(0,204,255,0.3)]">
                        {metrics?.label || "--"}
                    </div>
                </div>

                <div className="bg-zenith-panel/85 border border-zinc-800 p-4 rounded-lg backdrop-blur-sm pointer-events-auto">
                    <h3 className="m-0 text-xs text-zinc-400 tracking-widest mb-1">FLOW</h3>
                    <div className="text-3xl font-bold font-mono text-zenith-neonPurple drop-shadow-[0_0_10px_rgba(214,51,255,0.3)]">
                        {metrics?.flow?.toFixed(0) || "--"}
                    </div>
                </div>

                <div className="bg-zenith-panel/85 border border-zinc-800 p-4 rounded-lg backdrop-blur-sm pointer-events-auto">
                    <h3 className="m-0 text-xs text-zinc-400 tracking-widest mb-1">QUALITY</h3>
                    <div className="text-3xl font-bold font-mono text-zenith-neonGreen drop-shadow-[0_0_10px_rgba(0,255,153,0.3)]">
                        {metrics?.q?.toFixed(0) || "--"}
                    </div>
                </div>

                <button
                    onClick={onRequestAnalysis}
                    className="pointer-events-auto mt-4 bg-zinc-900 border border-zinc-700 text-zinc-300 hover:text-white hover:border-zenith-neonBlue hover:shadow-[0_0_15px_rgba(0,204,255,0.2)] px-4 py-3 rounded uppercase font-bold tracking-widest text-xs transition-all active:scale-95 flex items-center justify-center gap-2"
                >
                    <span>✦ Ask Sage</span>
                </button>
            </div>

            {advice && (
                <div className="absolute bottom-12 left-1/2 -translate-x-1/2 w-full max-w-2xl px-6 pointer-events-none">
                    <div className="bg-black/90 border-l-4 border-zenith-neonPurple p-6 rounded-r-lg shadow-2xl backdrop-blur-md animate-in slide-in-from-bottom-4 fade-in duration-500">
                        <h4 className="text-zenith-neonPurple text-xs font-bold tracking-widest mb-2 uppercase">Sage Guidance</h4>
                        <p className="text-gray-200 text-lg leading-relaxed font-light">{advice}</p>
                    </div>
                </div>
            )}
        </>
    );
};
