import React, { useEffect } from 'react';
import { useZenithVoice } from '../hooks/useZenithVoice';

interface SequenceBarProps {
    sequence: {
        name: string;
        current_goal: string;
        next_goal: string;
        progress: number;
        completed: boolean;
        announcement?: string;
    };
    onStop: () => void;
}

export const SequenceBar: React.FC<SequenceBarProps> = ({ sequence, onStop }) => {
    const { speak } = useZenithVoice();

    useEffect(() => {
        if (sequence.announcement) {
            speak(sequence.announcement);
        }
    }, [sequence.announcement, speak]);

    return (
        <div className="absolute top-20 left-1/2 -translate-x-1/2 z-30 pointer-events-auto">
            <div className="bg-black/90 border border-zenith-neonBlue/30 rounded-lg px-6 py-3 backdrop-blur-md shadow-[0_0_20px_rgba(0,204,255,0.1)] flex items-center gap-4 min-w-[400px]">
                {/* Sequence Name */}
                <div className="text-[9px] text-zinc-500 uppercase tracking-widest font-mono min-w-[60px]">
                    {sequence.name}
                </div>

                {/* Current Goal */}
                <div className="flex flex-col items-center min-w-[120px]">
                    <span className="text-[9px] text-zinc-500 uppercase tracking-widest">Current</span>
                    <span className="text-zenith-neonBlue font-bold font-mono text-sm">
                        {sequence.current_goal}
                    </span>
                </div>

                {/* Progress Bar */}
                <div className="flex-1 flex flex-col gap-1">
                    <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                        <div
                            className="h-full bg-gradient-to-r from-zenith-neonBlue to-zenith-neonPurple rounded-full transition-all duration-500"
                            style={{ width: `${Math.min(100, sequence.progress * 100)}%` }}
                        />
                    </div>
                </div>

                {/* Next Goal */}
                <div className="flex flex-col items-center min-w-[90px]">
                    <span className="text-[9px] text-zinc-500 uppercase tracking-widest">Next</span>
                    <span className="text-zinc-400 font-mono text-xs">
                        {sequence.next_goal}
                    </span>
                </div>

                {/* Stop Button */}
                <button
                    onClick={onStop}
                    className="text-zinc-600 hover:text-red-400 text-xs transition-colors cursor-pointer"
                    aria-label="Stop sequence"
                >
                    ✕
                </button>
            </div>

            {/* Announcement Flash */}
            {sequence.announcement && (
                <div className="mt-2 text-center text-zenith-neonGreen text-sm font-mono animate-pulse">
                    {sequence.announcement}
                </div>
            )}
        </div>
    );
};
