import React from 'react';
import { InfoTooltip } from './InfoTooltip';

interface SessionControlsProps {
    isRecording: boolean;
    isHarvesting: boolean;
    isSequencing: boolean;
    onToggleRecord: () => void;
    onToggleHarvest: () => void;
    onEndSession: () => void;
    onStartSequence: () => void;
}

export const SessionControls: React.FC<SessionControlsProps> = ({
    isRecording,
    isHarvesting,
    isSequencing,
    onToggleRecord,
    onToggleHarvest,
    onEndSession,
    onStartSequence,
}) => {
    return (
        <div className="absolute bottom-5 left-5 flex flex-col gap-3 pointer-events-auto z-40">
            {/* Record Button */}
            <button
                onClick={onToggleRecord}
                className={`flex items-center gap-3 px-4 py-3 rounded border backdrop-blur-sm transition-all text-xs font-bold tracking-widest uppercase
            ${isRecording
                        ? 'bg-red-500/20 border-red-500 text-red-500 shadow-[0_0_15px_rgba(239,68,68,0.3)]'
                        : 'bg-zenith-panel/85 border-zinc-800 text-zinc-400 hover:text-white hover:border-zinc-600'
                    }`}
            >
                <div className={`w-3 h-3 rounded-full ${isRecording ? 'bg-red-500 animate-pulse' : 'bg-current'}`} />
                <span>{isRecording ? 'Recording' : 'Record Session'}</span>
                {!isRecording && (
                    <InfoTooltip text="Start recording a practice session. Tracks pose detection, flow scores, and stability metrics. End the session to generate a summary report." />
                )}
            </button>

            {/* Harvest Button */}
            <button
                onClick={onToggleHarvest}
                className={`flex items-center gap-3 px-4 py-3 rounded border backdrop-blur-sm transition-all text-xs font-bold tracking-widest uppercase
            ${isHarvesting
                        ? 'bg-orange-500/20 border-orange-500 text-orange-500 shadow-[0_0_15px_rgba(249,115,22,0.3)]'
                        : 'bg-zenith-panel/85 border-zinc-800 text-zinc-400 hover:text-white hover:border-zinc-600'
                    }`}
            >
                <div className={`w-3 h-3 rounded-sm ${isHarvesting ? 'bg-orange-500 animate-pulse' : 'bg-current'}`} />
                <span>{isHarvesting ? 'Harvesting' : 'Harvest Data'}</span>
                {!isHarvesting && (
                    <InfoTooltip text="Save labeled frames for model training. Captures video frames with detected pose labels and quality scores to expand the training dataset." />
                )}
            </button>

            {/* Sequence Button */}
            {!isSequencing && (
                <button
                    onClick={onStartSequence}
                    className="flex items-center gap-3 px-4 py-3 rounded border backdrop-blur-sm transition-all text-xs font-bold tracking-widest uppercase bg-zenith-panel/85 border-zinc-800 text-zinc-400 hover:text-zenith-neonBlue hover:border-zenith-neonBlue/50"
                >
                    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h7" />
                    </svg>
                    <span>Start Sequence</span>
                    <InfoTooltip text="Begin a guided Sun Salutation sequence. Hold each pose for 8 seconds to advance. The coach will analyze your form automatically." />
                </button>
            )}

            {/* End Session Button — only visible when recording */}
            {isRecording && (
                <button
                    onClick={onEndSession}
                    className="mt-2 flex items-center justify-center gap-2 px-4 py-2 rounded bg-red-600/20 border border-red-600/50 text-red-400 hover:bg-red-600 hover:text-white hover:border-red-500 transition-all text-xs font-bold uppercase tracking-widest"
                >
                    <span>End Session</span>
                </button>
            )}
        </div>
    );
};
