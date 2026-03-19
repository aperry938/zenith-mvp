import React from 'react';

interface SessionStats {
    Duration: string;
    "Avg Flow": string;
    "Stability Events": number;
    "Zone Time": string;
    "Top Pose": string;
    "Peak Flow"?: string;
    "Peak Quality"?: string;
    Corrections?: number;
    "Pose Timeline"?: Array<{ t: number; pose: string; quality: number }>;
}

interface SessionReportProps {
    stats: SessionStats | null;
    onClose: () => void;
}

const POSE_COLORS: Record<string, string> = {
    "Mountain Pose": "bg-green-500",
    "Warrior II": "bg-red-500",
    "Tree": "bg-emerald-400",
    "Downward Dog": "bg-blue-500",
    "Plank": "bg-orange-500",
    "Chair": "bg-yellow-500",
    "Triangle": "bg-purple-500",
    "High Lunge": "bg-pink-500",
    "Extended Side Angle": "bg-cyan-500",
};

export const SessionReport: React.FC<SessionReportProps> = ({ stats, onClose }) => {
    if (!stats) return null;

    const timeline = stats["Pose Timeline"] || [];

    return (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md animate-fadeIn">
            <div className="bg-zenith-panel/95 border border-zinc-700 p-8 rounded-2xl shadow-[0_0_50px_rgba(0,0,0,0.8)] w-[90%] max-w-lg flex flex-col items-center gap-6">

                {/* Header */}
                <div className="flex flex-col items-center">
                    <h2 className="text-3xl font-bold uppercase tracking-widest text-white mb-2">Session Complete</h2>
                    <p className="text-zinc-400 text-sm">Movement quality analysis summary</p>
                </div>

                <div className="w-full h-px bg-gradient-to-r from-transparent via-zinc-600 to-transparent" />

                {/* Primary Metrics Grid */}
                <div className="grid grid-cols-2 gap-4 w-full">
                    <MetricCard label="Duration" value={stats.Duration} color="text-zenith-neonBlue" />
                    <MetricCard label="Avg Flow" value={stats["Avg Flow"]} color="text-purple-400" />
                    <MetricCard label="Zone Time" value={stats["Zone Time"]} color="text-green-400" />
                    <MetricCard label="Top Pose" value={stats["Top Pose"]} color="text-white" small />
                </div>

                {/* Secondary Metrics */}
                {(stats["Peak Flow"] || stats["Peak Quality"] || stats.Corrections !== undefined) && (
                    <div className="grid grid-cols-3 gap-3 w-full">
                        {stats["Peak Flow"] && (
                            <MetricCard label="Peak Flow" value={stats["Peak Flow"]} color="text-purple-300" compact />
                        )}
                        {stats["Peak Quality"] && (
                            <MetricCard label="Peak Quality" value={stats["Peak Quality"]} color="text-green-300" compact />
                        )}
                        {stats.Corrections !== undefined && (
                            <MetricCard label="Corrections" value={String(stats.Corrections)} color="text-zenith-neonBlue" compact />
                        )}
                    </div>
                )}

                {/* Pose Timeline */}
                {timeline.length > 0 && (
                    <div className="w-full">
                        <h4 className="text-xs text-zinc-500 uppercase tracking-wider mb-2 text-center">Pose Timeline</h4>
                        <div className="flex h-3 rounded-full overflow-hidden bg-zinc-800 w-full">
                            {timeline.map((entry, i) => (
                                <div
                                    key={i}
                                    className={`h-full ${POSE_COLORS[entry.pose] || 'bg-zinc-600'}`}
                                    style={{ flex: 1 }}
                                    title={`${entry.pose} (${entry.t}s)`}
                                />
                            ))}
                        </div>
                        {/* Legend */}
                        <div className="flex flex-wrap gap-2 mt-2 justify-center">
                            {[...new Set(timeline.map(e => e.pose))].map(pose => (
                                <div key={pose} className="flex items-center gap-1">
                                    <div className={`w-2 h-2 rounded-full ${POSE_COLORS[pose] || 'bg-zinc-600'}`} />
                                    <span className="text-[10px] text-zinc-500">{pose}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Action */}
                <button
                    onClick={onClose}
                    className="mt-4 px-8 py-3 bg-white text-black font-bold uppercase tracking-widest rounded-full hover:bg-zenith-neonBlue hover:shadow-[0_0_20px_rgba(0,255,255,0.4)] transition-all cursor-pointer"
                >
                    Close Report
                </button>
            </div>
        </div>
    );
};

function MetricCard({ label, value, color, small, compact }: {
    label: string;
    value: string;
    color: string;
    small?: boolean;
    compact?: boolean;
}) {
    return (
        <div className={`bg-zinc-800/50 ${compact ? 'p-3' : 'p-4'} rounded-lg flex flex-col items-center border border-zinc-700`}>
            <span className={`text-zinc-500 uppercase tracking-wider mb-1 ${compact ? 'text-[9px]' : 'text-xs'}`}>{label}</span>
            <span className={`font-mono ${color} ${
                compact ? 'text-lg' : small ? 'text-lg font-bold text-center break-words w-full truncate px-2' : 'text-2xl'
            }`}>
                {value}
            </span>
        </div>
    );
}
