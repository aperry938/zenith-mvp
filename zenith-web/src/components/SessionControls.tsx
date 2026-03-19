import React from 'react';
import { InfoTooltip } from './InfoTooltip';
import { PersonaSelector } from './PersonaSelector';

const INTENSITY_LABELS = ['Gentle', 'Standard', 'Intense'] as const;

interface SessionControlsProps {
    isRecording: boolean;
    isHarvesting: boolean;
    isSequencing: boolean;
    onToggleRecord: () => void;
    onToggleHarvest: () => void;
    onEndSession: () => void;
    onStartSequence: (key: string) => void;
    intensity: number;
    onSetIntensity: (level: number) => void;
    persona: string;
    onSetPersona: (persona: string) => void;
}

export const SessionControls: React.FC<SessionControlsProps> = ({
    isRecording,
    isHarvesting,
    isSequencing,
    onToggleRecord,
    onToggleHarvest,
    onEndSession,
    onStartSequence,
    intensity,
    onSetIntensity,
    persona,
    onSetPersona,
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

            {/* Intensity Selector */}
            <div className="flex flex-col gap-1.5">
                <span className="text-[9px] text-zinc-500 uppercase tracking-widest px-1">Intensity</span>
                <div className="flex gap-1">
                    {INTENSITY_LABELS.map((label, i) => {
                        const level = i + 1;
                        const isActive = intensity === level;
                        return (
                            <button
                                key={label}
                                onClick={() => onSetIntensity(level)}
                                className={`flex-1 px-2 py-1.5 rounded text-[10px] font-bold tracking-wider uppercase transition-all border cursor-pointer ${
                                    isActive
                                        ? level === 1
                                            ? 'bg-green-500/20 border-green-500/50 text-green-400'
                                            : level === 2
                                            ? 'bg-zenith-neonBlue/20 border-zenith-neonBlue/50 text-zenith-neonBlue'
                                            : 'bg-red-500/20 border-red-500/50 text-red-400'
                                        : 'bg-zenith-panel/85 border-zinc-800 text-zinc-500 hover:text-zinc-300'
                                }`}
                            >
                                {label}
                            </button>
                        );
                    })}
                </div>
            </div>

            {/* Coach Persona */}
            <PersonaSelector persona={persona} onSetPersona={onSetPersona} />

            {/* Sequence Buttons */}
            {!isSequencing && (
                <div className="flex flex-col gap-1.5">
                    <span className="text-[9px] text-zinc-500 uppercase tracking-widest px-1">Guided Flows</span>
                    {([
                        { key: "warrior_flow", label: "Warrior", desc: "Standing strength and lateral flexibility" },
                        { key: "balance_flow", label: "Balance", desc: "Single-leg balance and lower body power" },
                        { key: "strength_flow", label: "Strength", desc: "Core and upper body endurance" },
                        { key: "complete_flow", label: "Complete", desc: "All 10 poses in one comprehensive practice" },
                    ] as const).map(({ key, label, desc }) => (
                        <button
                            key={key}
                            onClick={() => onStartSequence(key)}
                            aria-label={`Start ${label} flow sequence`}
                            className="flex items-center gap-2 px-3 py-2 rounded border backdrop-blur-sm transition-all text-xs font-bold tracking-wider uppercase bg-zenith-panel/85 border-zinc-800 text-zinc-400 hover:text-zenith-neonBlue hover:border-zenith-neonBlue/50"
                        >
                            <span>{label}</span>
                            <InfoTooltip text={`${desc}. Hold each pose for 8 seconds to advance.`} />
                        </button>
                    ))}
                </div>
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
