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

const POSE_HEX_COLORS: Record<string, string> = {
    "Mountain Pose": "#22c55e",
    "Warrior II": "#ef4444",
    "Tree": "#34d399",
    "Downward Dog": "#3b82f6",
    "Plank": "#f97316",
    "Chair": "#eab308",
    "Triangle": "#a855f7",
    "High Lunge": "#ec4899",
    "Extended Side Angle": "#06b6d4",
};

function exportSession(stats: SessionStats) {
    const timeline = stats["Pose Timeline"] || [];
    const uniquePoses = [...new Set(timeline.map(e => e.pose))];
    const date = new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });

    const timelineBarHtml = timeline.length > 0 ? `
        <div style="margin: 24px 0;">
            <h3 style="font-size: 11px; text-transform: uppercase; letter-spacing: 2px; color: #888; margin-bottom: 8px;">Pose Timeline</h3>
            <div style="display: flex; height: 16px; border-radius: 8px; overflow: hidden; background: #e5e5e5;">
                ${timeline.map(entry => `<div style="flex: 1; background: ${POSE_HEX_COLORS[entry.pose] || '#999'};" title="${entry.pose} (${entry.t}s)"></div>`).join('')}
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 12px; margin-top: 8px; justify-content: center;">
                ${uniquePoses.map(pose => `<div style="display: flex; align-items: center; gap: 4px;">
                    <div style="width: 8px; height: 8px; border-radius: 50%; background: ${POSE_HEX_COLORS[pose] || '#999'};"></div>
                    <span style="font-size: 11px; color: #666;">${pose}</span>
                </div>`).join('')}
            </div>
        </div>
    ` : '';

    const html = `<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>ZENith Session Report - ${date}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; max-width: 600px; margin: 40px auto; padding: 20px; color: #111; }
        h1 { font-size: 24px; text-transform: uppercase; letter-spacing: 4px; text-align: center; margin-bottom: 4px; }
        .subtitle { font-size: 13px; color: #888; text-align: center; margin-bottom: 24px; }
        .divider { height: 1px; background: linear-gradient(to right, transparent, #ccc, transparent); margin: 20px 0; }
        .grid { display: grid; gap: 12px; }
        .grid-2 { grid-template-columns: 1fr 1fr; }
        .grid-3 { grid-template-columns: 1fr 1fr 1fr; }
        .metric { background: #f8f8f8; border: 1px solid #e5e5e5; border-radius: 8px; padding: 16px; text-align: center; }
        .metric .label { font-size: 10px; text-transform: uppercase; letter-spacing: 2px; color: #888; margin-bottom: 4px; }
        .metric .value { font-size: 22px; font-weight: 700; font-family: 'SF Mono', 'Menlo', monospace; }
        .metric-sm .value { font-size: 16px; }
        .footer { margin-top: 32px; text-align: center; font-size: 11px; color: #aaa; }
        @media print { body { margin: 0; } }
    </style>
</head>
<body>
    <h1>Session Report</h1>
    <p class="subtitle">${date}</p>
    <div class="divider"></div>

    <div class="grid grid-2" style="margin-bottom: 12px;">
        <div class="metric"><div class="label">Duration</div><div class="value">${stats.Duration}</div></div>
        <div class="metric"><div class="label">Avg Flow</div><div class="value">${stats["Avg Flow"]}</div></div>
        <div class="metric"><div class="label">Zone Time</div><div class="value">${stats["Zone Time"]}</div></div>
        <div class="metric"><div class="label">Top Pose</div><div class="value" style="font-size: 16px;">${stats["Top Pose"]}</div></div>
    </div>

    <div class="grid grid-3">
        ${stats["Peak Flow"] ? `<div class="metric metric-sm"><div class="label">Peak Flow</div><div class="value">${stats["Peak Flow"]}</div></div>` : ''}
        ${stats["Peak Quality"] ? `<div class="metric metric-sm"><div class="label">Peak Quality</div><div class="value">${stats["Peak Quality"]}</div></div>` : ''}
        ${stats.Corrections !== undefined ? `<div class="metric metric-sm"><div class="label">Corrections</div><div class="value">${stats.Corrections}</div></div>` : ''}
    </div>

    ${timelineBarHtml}

    <div class="footer">ZENith - Real-Time Biomechanical Movement Analysis</div>
</body>
</html>`;

    const printWindow = window.open('', '_blank');
    if (printWindow) {
        printWindow.document.write(html);
        printWindow.document.close();
        printWindow.onload = () => printWindow.print();
    }
}

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

                {/* Actions */}
                <div className="flex gap-3 mt-4">
                    <button
                        onClick={() => exportSession(stats)}
                        className="px-6 py-3 bg-zinc-800 text-zinc-300 font-bold uppercase tracking-widest rounded-full border border-zinc-700 hover:bg-zinc-700 hover:text-white transition-all cursor-pointer text-sm"
                    >
                        Export
                    </button>
                    <button
                        onClick={onClose}
                        className="px-8 py-3 bg-white text-black font-bold uppercase tracking-widest rounded-full hover:bg-zenith-neonBlue hover:shadow-[0_0_20px_rgba(0,255,255,0.4)] transition-all cursor-pointer"
                    >
                        Close Report
                    </button>
                </div>
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
