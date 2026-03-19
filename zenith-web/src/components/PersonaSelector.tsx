import React from 'react';

interface PersonaSelectorProps {
    persona: string;
    onSetPersona: (persona: string) => void;
}

const PERSONAS = [
    { key: 'default', label: 'Default', desc: 'Clear, neutral coaching' },
    { key: 'sage', label: 'Sage', desc: 'Philosophical, mindful' },
    { key: 'scientist', label: 'Scientist', desc: 'Biomechanical, precise' },
    { key: 'warrior', label: 'Warrior', desc: 'Strict, motivational' },
    { key: 'flow', label: 'Flow', desc: 'Poetic, sensory' },
    { key: 'traditional', label: 'Traditional', desc: 'Sanskrit, classical' },
] as const;

export const PersonaSelector: React.FC<PersonaSelectorProps> = ({ persona, onSetPersona }) => {
    return (
        <div className="flex flex-col gap-1.5">
            <span className="text-[9px] text-zinc-500 uppercase tracking-widest px-1">Coach Persona</span>
            <div className="flex flex-wrap gap-1.5">
                {PERSONAS.map(({ key, label, desc }) => {
                    const isActive = persona === key;
                    return (
                        <button
                            key={key}
                            onClick={() => onSetPersona(key)}
                            title={desc}
                            className={`px-2.5 py-1.5 rounded border backdrop-blur-sm transition-all text-[10px] font-bold tracking-wider uppercase
                                ${isActive
                                    ? 'bg-zenith-neonPurple/20 border-zenith-neonPurple text-zenith-neonPurple shadow-[0_0_10px_rgba(214,51,255,0.2)]'
                                    : 'bg-zenith-panel/85 border-zinc-800 text-zinc-500 hover:text-zinc-300 hover:border-zinc-600'
                                }`}
                        >
                            {label}
                        </button>
                    );
                })}
            </div>
        </div>
    );
};
