import React from 'react';

interface SessionControlsProps {
    isRecording: boolean;
    isHarvesting: boolean;
    onToggleRecord: () => void;
    onToggleHarvest: () => void;
}

export const SessionControls: React.FC<SessionControlsProps> = ({
    isRecording,
    isHarvesting,
    onToggleRecord,
    onToggleHarvest
}) => {
    return (
        <div className="absolute top-5 left-5 flex flex-col gap-3 pointer-events-auto">
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
            </button>
        </div>
    );
};
