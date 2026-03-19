import React, { useEffect, useState } from 'react';

interface SessionRecord {
    date: string;
    duration: number;
    avg_flow: number;
    peak_flow: number;
    peak_quality: number;
    corrections: number;
    top_pose: string;
}

interface SessionHistoryProps {
    onClose: () => void;
}

export const SessionHistory: React.FC<SessionHistoryProps> = ({ onClose }) => {
    const [sessions, setSessions] = useState<SessionRecord[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch('http://localhost:8000/api/sessions')
            .then(res => res.json())
            .then(data => {
                setSessions(Array.isArray(data) ? data.reverse() : []);
                setLoading(false);
            })
            .catch(() => {
                setSessions([]);
                setLoading(false);
            });
    }, []);

    return (
        <div className="absolute inset-0 z-[55] flex items-center justify-center bg-black/80 backdrop-blur-md animate-fadeIn">
            <div className="bg-zenith-panel/95 border border-zinc-700 p-6 rounded-2xl shadow-[0_0_50px_rgba(0,0,0,0.8)] w-[90%] max-w-lg max-h-[80vh] flex flex-col">
                {/* Header */}
                <div className="flex justify-between items-center mb-4">
                    <h2 className="text-xl font-bold uppercase tracking-widest text-white">Session History</h2>
                    <button
                        onClick={onClose}
                        aria-label="Close history"
                        className="text-zinc-500 hover:text-white text-sm cursor-pointer"
                    >
                        ✕
                    </button>
                </div>

                <div className="w-full h-px bg-gradient-to-r from-transparent via-zinc-600 to-transparent mb-4" />

                {/* Content */}
                <div className="flex-1 overflow-y-auto space-y-3 pr-1">
                    {loading && (
                        <div className="text-center text-zinc-500 text-sm py-8 font-mono">Loading...</div>
                    )}

                    {!loading && sessions.length === 0 && (
                        <div className="text-center py-8">
                            <p className="text-zinc-500 text-sm">No sessions recorded yet.</p>
                            <p className="text-zinc-600 text-xs mt-1">Record a session and end it to see history here.</p>
                        </div>
                    )}

                    {sessions.map((s, i) => (
                        <div key={i} className="bg-zinc-800/50 border border-zinc-700 rounded-lg p-3">
                            <div className="flex justify-between items-start mb-2">
                                <span className="text-xs text-zinc-400 font-mono">{s.date}</span>
                                <span className="text-xs text-zenith-neonBlue font-mono">{formatDuration(s.duration)}</span>
                            </div>
                            <div className="grid grid-cols-4 gap-2 text-center">
                                <Stat label="Flow" value={String(s.avg_flow)} color="text-purple-400" />
                                <Stat label="Peak Q" value={String(s.peak_quality)} color="text-green-400" />
                                <Stat label="Fixes" value={String(s.corrections)} color="text-zenith-neonBlue" />
                                <Stat label="Top" value={s.top_pose.split(' ')[0]} color="text-white" />
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

function Stat({ label, value, color }: { label: string; value: string; color: string }) {
    return (
        <div>
            <div className="text-[9px] text-zinc-500 uppercase tracking-wider">{label}</div>
            <div className={`text-sm font-mono ${color}`}>{value}</div>
        </div>
    );
}

function formatDuration(seconds: number): string {
    if (seconds < 60) return `${seconds}s`;
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}m ${s}s`;
}
