import React from 'react';

interface ZenithMetrics {
    label: string;
    flow: number;
    velocity: number;
    q: number;
}

interface HUDProps {
    metrics: ZenithMetrics | null;
}

export const HUD: React.FC<HUDProps> = ({ metrics }) => {
    return (
        <div className="absolute top-5 right-5 w-52 flex flex-col gap-4">
            <div className="bg-zenith-panel/85 border border-zinc-800 p-4 rounded-lg backdrop-blur-sm">
                <h3 className="m-0 text-xs text-zinc-400 tracking-widest mb-1">POSE</h3>
                <div className="text-3xl font-bold font-mono text-zenith-neonBlue drop-shadow-[0_0_10px_rgba(0,204,255,0.3)]">
                    {metrics?.label || "--"}
                </div>
            </div>
            <div className="bg-zenith-panel/85 border border-zinc-800 p-4 rounded-lg backdrop-blur-sm">
                <h3 className="m-0 text-xs text-zinc-400 tracking-widest mb-1">FLOW</h3>
                <div className="text-3xl font-bold font-mono text-zenith-neonPurple drop-shadow-[0_0_10px_rgba(214,51,255,0.3)]">
                    {metrics?.flow?.toFixed(0) || "--"}
                </div>
            </div>
            <div className="bg-zenith-panel/85 border border-zinc-800 p-4 rounded-lg backdrop-blur-sm">
                <h3 className="m-0 text-xs text-zinc-400 tracking-widest mb-1">QUALITY</h3>
                <div className="text-3xl font-bold font-mono text-zenith-neonGreen drop-shadow-[0_0_10px_rgba(0,255,153,0.3)]">
                    {metrics?.q?.toFixed(0) || "--"}
                </div>
            </div>
        </div>
    );
};
