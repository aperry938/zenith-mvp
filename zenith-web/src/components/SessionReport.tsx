import React from 'react';

interface SessionStats {
    Duration: string;
    "Avg Flow": string;
    "Stability Events": number;
    "Zone Time": string;
    "Top Pose": string;
}

interface SessionReportProps {
    stats: SessionStats | null;
    onClose: () => void;
}

export const SessionReport: React.FC<SessionReportProps> = ({ stats, onClose }) => {
    if (!stats) return null;

    return (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md animate-fadeIn">
            <div className="bg-zenith-panel/95 border border-zinc-700 p-8 rounded-2xl shadow-[0_0_50px_rgba(0,0,0,0.8)] w-[90%] max-w-lg flex flex-col items-center gap-6">

                {/* Header */}
                <div className="flex flex-col items-center">
                    <h2 className="text-3xl font-bold uppercase tracking-widest text-white mb-2">Namaste</h2>
                    <p className="text-zinc-400 text-sm italic">"The light in me honors the light in you."</p>
                </div>

                <div className="w-full h-px bg-gradient-to-r from-transparent via-zinc-600 to-transparent" />

                {/* Metrics Grid */}
                <div className="grid grid-cols-2 gap-4 w-full">
                    <div className="bg-zinc-800/50 p-4 rounded-lg flex flex-col items-center border border-zinc-700">
                        <span className="text-xs text-zinc-500 uppercase tracking-wider mb-1">Duration</span>
                        <span className="text-2xl font-mono text-zenith-cyan">{stats.Duration}</span>
                    </div>
                    <div className="bg-zinc-800/50 p-4 rounded-lg flex flex-col items-center border border-zinc-700">
                        <span className="text-xs text-zinc-500 uppercase tracking-wider mb-1">Avg Flow</span>
                        <span className="text-2xl font-mono text-purple-400">{stats["Avg Flow"]}</span>
                    </div>
                    <div className="bg-zinc-800/50 p-4 rounded-lg flex flex-col items-center border border-zinc-700">
                        <span className="text-xs text-zinc-500 uppercase tracking-wider mb-1">Zone Time</span>
                        <span className="text-2xl font-mono text-green-400">{stats["Zone Time"]}</span>
                    </div>
                    <div className="bg-zinc-800/50 p-4 rounded-lg flex flex-col items-center border border-zinc-700">
                        <span className="text-xs text-zinc-500 uppercase tracking-wider mb-1">Top Pose</span>
                        <span className="text-lg font-bold text-white text-center break-words w-full truncate px-2">{stats["Top Pose"]}</span>
                    </div>
                </div>

                {/* Action */}
                <button
                    onClick={onClose}
                    className="mt-4 px-8 py-3 bg-white text-black font-bold uppercase tracking-widest rounded-full hover:bg-zenith-cyan hover:shadow-[0_0_20px_rgba(0,255,255,0.4)] transition-all"
                >
                    Return to Reality
                </button>
            </div>
        </div>
    );
};
