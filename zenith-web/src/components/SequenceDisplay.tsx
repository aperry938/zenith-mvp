import React from 'react';

interface SequenceState {
    target_pose: string;
    next_pose: string;
    status: string;
    progress: number;
}

interface SequenceDisplayProps {
    state: SequenceState | null;
}

export const SequenceDisplay: React.FC<SequenceDisplayProps> = ({ state }) => {
    if (!state) return null;

    return (
        <div className="absolute top-20 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 pointer-events-none z-30">
            {/* Progress Bar */}
            <div className="w-64 h-1 bg-zinc-800 rounded-full overflow-hidden">
                <div
                    className="h-full bg-zenith-cyan transition-all duration-1000 ease-out"
                    style={{ width: `${state.progress * 100}%` }}
                />
            </div>

            {/* Current Objective */}
            <div className="flex flex-col items-center">
                <h2 className="text-xs font-bold tracking-[0.2em] text-zinc-400 uppercase mb-1">Current Asana</h2>
                <div className={`text-3xl font-black text-white uppercase tracking-tight drop-shadow-[0_0_15px_rgba(255,255,255,0.3)] transition-all duration-300 ${state.status === 'Advance' ? 'scale-110 text-zenith-neonGreen' : ''}`}>
                    {state.target_pose}
                </div>
            </div>

            {/* Up Next */}
            <div className="flex items-center gap-2 mt-2 opacity-60">
                <span className="text-[10px] uppercase tracking-widest text-zinc-500">Next:</span>
                <span className="text-xs font-bold text-zinc-300 uppercase">{state.next_pose}</span>
            </div>
        </div>
    );
};
