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

interface PoseTrend {
    pose: string;
    sessions_practiced: number;
    latest_quality: number;
    change: number;
    trend: 'improving' | 'declining' | 'stable';
}

interface Prescription {
    pose: string;
    focus: string;
    exercises: string[];
}

interface ProgressData {
    insufficient_data?: boolean;
    min_sessions?: number;
    pose_trends?: PoseTrend[];
    top_strengths?: string[];
    top_weaknesses?: string[];
    quality_trend?: string;
    flow_trend?: string;
    prescriptions?: Prescription[];
}

interface Achievement {
    id: string;
    name: string;
    desc: string;
    unlocked: boolean;
}

interface StreakData {
    current_streak: number;
    best_streak: number;
    total_sessions: number;
    total_minutes: number;
    peak_flow: number;
    peak_quality: number;
    achievements: Achievement[];
}

interface SessionHistoryProps {
    onClose: () => void;
}

type Tab = 'history' | 'progress' | 'achievements';

export const SessionHistory: React.FC<SessionHistoryProps> = ({ onClose }) => {
    const [tab, setTab] = useState<Tab>('history');
    const [sessions, setSessions] = useState<SessionRecord[]>([]);
    const [progress, setProgress] = useState<ProgressData | null>(null);
    const [streakData, setStreakData] = useState<StreakData | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchSessions = fetch('http://localhost:8000/api/sessions')
            .then(res => res.json())
            .then(data => setSessions(Array.isArray(data) ? data.reverse() : []))
            .catch(() => setSessions([]));

        const fetchProgress = fetch('http://localhost:8000/api/progress')
            .then(res => res.json())
            .then(data => setProgress(data))
            .catch(() => setProgress(null));

        const fetchStreaks = fetch('http://localhost:8000/api/streaks')
            .then(res => res.json())
            .then(data => setStreakData(data))
            .catch(() => setStreakData(null));

        Promise.all([fetchSessions, fetchProgress, fetchStreaks]).finally(() => setLoading(false));
    }, []);

    return (
        <div className="absolute inset-0 z-[55] flex items-center justify-center bg-black/80 backdrop-blur-md animate-fadeIn">
            <div className="bg-zenith-panel/95 border border-zinc-700 p-6 rounded-2xl shadow-[0_0_50px_rgba(0,0,0,0.8)] w-[90%] max-w-lg max-h-[80vh] flex flex-col">
                {/* Header */}
                <div className="flex justify-between items-center mb-4">
                    <h2 className="text-xl font-bold uppercase tracking-widest text-white">
                        {tab === 'history' ? 'Session History' : 'My Progress'}
                    </h2>
                    <button
                        onClick={onClose}
                        aria-label="Close history"
                        className="text-zinc-500 hover:text-white text-sm cursor-pointer"
                    >
                        ✕
                    </button>
                </div>

                {/* Tabs */}
                <div className="flex gap-1 mb-4">
                    <TabButton active={tab === 'history'} onClick={() => setTab('history')}>History</TabButton>
                    <TabButton active={tab === 'progress'} onClick={() => setTab('progress')}>Progress</TabButton>
                    <TabButton active={tab === 'achievements'} onClick={() => setTab('achievements')}>Achievements</TabButton>
                </div>

                <div className="w-full h-px bg-gradient-to-r from-transparent via-zinc-600 to-transparent mb-4" />

                {/* Content */}
                <div className="flex-1 overflow-y-auto space-y-3 pr-1">
                    {loading && (
                        <div className="text-center text-zinc-500 text-sm py-8 font-mono">Loading...</div>
                    )}

                    {!loading && tab === 'history' && <HistoryTab sessions={sessions} />}
                    {!loading && tab === 'progress' && <ProgressTab progress={progress} />}
                    {!loading && tab === 'achievements' && <AchievementsTab streakData={streakData} />}
                </div>
            </div>
        </div>
    );
};

/* ------------------------------------------------------------------ */
/*  Tabs                                                              */
/* ------------------------------------------------------------------ */

function TabButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
    return (
        <button
            onClick={onClick}
            className={`px-3 py-1.5 rounded text-xs font-mono tracking-wider transition-colors cursor-pointer ${
                active
                    ? 'bg-zinc-700 text-white'
                    : 'text-zinc-500 hover:text-white hover:bg-zinc-800'
            }`}
        >
            {children}
        </button>
    );
}

/* ------------------------------------------------------------------ */
/*  History Tab                                                       */
/* ------------------------------------------------------------------ */

function HistoryTab({ sessions }: { sessions: SessionRecord[] }) {
    if (sessions.length === 0) {
        return (
            <div className="text-center py-8">
                <p className="text-zinc-500 text-sm">No sessions recorded yet.</p>
                <p className="text-zinc-600 text-xs mt-1">Record a session and end it to see history here.</p>
            </div>
        );
    }

    return (
        <>
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
        </>
    );
}

/* ------------------------------------------------------------------ */
/*  Progress Tab                                                      */
/* ------------------------------------------------------------------ */

function ProgressTab({ progress }: { progress: ProgressData | null }) {
    if (!progress) {
        return (
            <div className="text-center py-8">
                <p className="text-zinc-500 text-sm">Unable to load progress data.</p>
            </div>
        );
    }

    if (progress.insufficient_data) {
        return (
            <div className="text-center py-8">
                <p className="text-zinc-500 text-sm">Need at least {progress.min_sessions} sessions to show progress.</p>
                <p className="text-zinc-600 text-xs mt-1">Keep practicing and check back.</p>
            </div>
        );
    }

    return (
        <div className="space-y-4">
            {/* Overall Trends */}
            <div className="bg-zinc-800/50 border border-zinc-700 rounded-lg p-3">
                <div className="text-[10px] text-zinc-500 uppercase tracking-widest mb-2">Overall Trends</div>
                <div className="grid grid-cols-2 gap-3">
                    <TrendStat label="Quality" trend={progress.quality_trend} />
                    <TrendStat label="Flow" trend={progress.flow_trend} />
                </div>
            </div>

            {/* Strengths */}
            {progress.top_strengths && progress.top_strengths.length > 0 && (
                <div className="bg-zinc-800/50 border border-zinc-700 rounded-lg p-3">
                    <div className="text-[10px] text-zinc-500 uppercase tracking-widest mb-2">Strengths</div>
                    <div className="flex flex-wrap gap-1.5">
                        {progress.top_strengths.map(pose => (
                            <span key={pose} className="px-2 py-0.5 rounded text-xs font-mono bg-green-900/40 text-green-400 border border-green-800/50">
                                {pose}
                            </span>
                        ))}
                    </div>
                </div>
            )}

            {/* Weaknesses */}
            {progress.top_weaknesses && progress.top_weaknesses.length > 0 && (
                <div className="bg-zinc-800/50 border border-zinc-700 rounded-lg p-3">
                    <div className="text-[10px] text-zinc-500 uppercase tracking-widest mb-2">Needs Work</div>
                    <div className="flex flex-wrap gap-1.5">
                        {progress.top_weaknesses.map(pose => (
                            <span key={pose} className="px-2 py-0.5 rounded text-xs font-mono bg-amber-900/40 text-amber-400 border border-amber-800/50">
                                {pose}
                            </span>
                        ))}
                    </div>
                </div>
            )}

            {/* Pose Trends */}
            {progress.pose_trends && progress.pose_trends.length > 0 && (
                <div className="bg-zinc-800/50 border border-zinc-700 rounded-lg p-3">
                    <div className="text-[10px] text-zinc-500 uppercase tracking-widest mb-2">Pose Trends</div>
                    <div className="space-y-1.5">
                        {progress.pose_trends.map(pt => (
                            <div key={pt.pose} className="flex items-center justify-between text-xs font-mono">
                                <span className="text-zinc-300">{pt.pose}</span>
                                <div className="flex items-center gap-2">
                                    <span className="text-zinc-500">{pt.sessions_practiced}x</span>
                                    <span className="text-zinc-400">Q:{pt.latest_quality}</span>
                                    <TrendArrow trend={pt.trend} change={pt.change} />
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Exercise Prescriptions */}
            {progress.prescriptions && progress.prescriptions.length > 0 && (
                <div className="space-y-2">
                    <div className="text-[10px] text-zinc-500 uppercase tracking-widest px-1">Your Prescription</div>
                    {progress.prescriptions.map(rx => (
                        <div key={rx.pose} className="bg-zinc-800/50 border border-zinc-700 rounded-lg p-3">
                            <div className="flex items-start justify-between mb-1.5">
                                <span className="text-sm font-mono text-white">{rx.pose}</span>
                                <span className="text-[10px] text-amber-400/80 font-mono">{rx.focus}</span>
                            </div>
                            <ul className="space-y-1">
                                {rx.exercises.map((ex, i) => (
                                    <li key={i} className="text-xs text-zinc-400 font-mono pl-2 border-l border-zinc-700">
                                        {ex}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

/* ------------------------------------------------------------------ */
/*  Achievements Tab                                                  */
/* ------------------------------------------------------------------ */

function AchievementsTab({ streakData }: { streakData: StreakData | null }) {
    if (!streakData) {
        return (
            <div className="text-center py-8">
                <p className="text-zinc-500 text-sm">Unable to load achievement data.</p>
            </div>
        );
    }

    const unlocked = streakData.achievements.filter(a => a.unlocked);
    const locked = streakData.achievements.filter(a => !a.unlocked);

    return (
        <div className="space-y-4">
            {/* Streak Stats */}
            <div className="bg-zinc-800/50 border border-zinc-700 rounded-lg p-3">
                <div className="text-[10px] text-zinc-500 uppercase tracking-widest mb-2">Streak Stats</div>
                <div className="grid grid-cols-3 gap-3 text-center">
                    <Stat label="Current" value={String(streakData.current_streak)} color="text-orange-400" />
                    <Stat label="Best" value={String(streakData.best_streak)} color="text-yellow-400" />
                    <Stat label="Sessions" value={String(streakData.total_sessions)} color="text-zenith-neonBlue" />
                </div>
            </div>

            {/* Personal Records */}
            <div className="bg-zinc-800/50 border border-zinc-700 rounded-lg p-3">
                <div className="text-[10px] text-zinc-500 uppercase tracking-widest mb-2">Personal Records</div>
                <div className="grid grid-cols-3 gap-3 text-center">
                    <Stat label="Minutes" value={String(streakData.total_minutes)} color="text-zinc-300" />
                    <Stat label="Peak Flow" value={String(streakData.peak_flow)} color="text-purple-400" />
                    <Stat label="Peak Q" value={String(streakData.peak_quality)} color="text-green-400" />
                </div>
            </div>

            {/* Unlocked */}
            {unlocked.length > 0 && (
                <div>
                    <div className="text-[10px] text-zinc-500 uppercase tracking-widest px-1 mb-2">Unlocked ({unlocked.length})</div>
                    <div className="space-y-1.5">
                        {unlocked.map(a => (
                            <div key={a.id} className="bg-green-900/20 border border-green-800/40 rounded-lg px-3 py-2 flex items-center gap-3">
                                <span className="text-green-400 text-lg">&#x2713;</span>
                                <div>
                                    <div className="text-xs font-bold text-green-400">{a.name}</div>
                                    <div className="text-[10px] text-zinc-500">{a.desc}</div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Locked */}
            {locked.length > 0 && (
                <div>
                    <div className="text-[10px] text-zinc-500 uppercase tracking-widest px-1 mb-2">Locked ({locked.length})</div>
                    <div className="space-y-1.5">
                        {locked.map(a => (
                            <div key={a.id} className="bg-zinc-800/30 border border-zinc-700/50 rounded-lg px-3 py-2 flex items-center gap-3 opacity-50">
                                <span className="text-zinc-600 text-lg">&#x25CB;</span>
                                <div>
                                    <div className="text-xs font-bold text-zinc-500">{a.name}</div>
                                    <div className="text-[10px] text-zinc-600">{a.desc}</div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

/* ------------------------------------------------------------------ */
/*  Shared components                                                 */
/* ------------------------------------------------------------------ */

function TrendStat({ label, trend }: { label: string; trend?: string }) {
    const display = trendDisplay(trend);
    return (
        <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-400 font-mono">{label}</span>
            <span className={`text-xs font-mono ${display.color}`}>{display.text}</span>
        </div>
    );
}

function TrendArrow({ trend, change }: { trend: string; change: number }) {
    const display = trendDisplay(trend);
    const sign = change > 0 ? '+' : '';
    return (
        <span className={`text-xs font-mono ${display.color}`} title={`${sign}${change}`}>
            {display.arrow}
        </span>
    );
}

function trendDisplay(trend?: string): { text: string; color: string; arrow: string } {
    switch (trend) {
        case 'improving':
            return { text: 'Improving', color: 'text-green-400', arrow: '\u2191' };
        case 'declining':
            return { text: 'Declining', color: 'text-red-400', arrow: '\u2193' };
        case 'stable':
            return { text: 'Stable', color: 'text-zinc-400', arrow: '\u2192' };
        default:
            return { text: '--', color: 'text-zinc-600', arrow: '--' };
    }
}

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
