import React, { useState, useEffect } from 'react';

interface GenerativeCoachProps {
    advice: string | null;
    adviceSource: string | null;
    isAnalyzing: boolean;
    onRequestAnalysis: () => void;
}

export const GenerativeCoach: React.FC<GenerativeCoachProps> = ({
    advice,
    adviceSource,
    isAnalyzing,
    onRequestAnalysis,
}) => {
    const [visibleAdvice, setVisibleAdvice] = useState<string | null>(null);

    useEffect(() => {
        if (advice) {
            setVisibleAdvice(advice);
            const timer = setTimeout(() => setVisibleAdvice(null), 15000);
            return () => clearTimeout(timer);
        }
    }, [advice]);

    return (
        <div className="absolute bottom-5 right-5 flex flex-col items-end gap-3 z-20">
            {/* Advice Speech Bubble */}
            {visibleAdvice && !isAnalyzing && (
                <div className="max-w-xs bg-black/90 border border-zenith-neonBlue/30 rounded-lg p-4 backdrop-blur-md shadow-[0_0_20px_rgba(0,204,255,0.1)] pointer-events-auto">
                    {adviceSource && (
                        <span className={`inline-block text-[9px] font-mono tracking-widest uppercase px-1.5 py-0.5 rounded mb-2 ${
                            adviceSource === 'gemini'
                                ? 'bg-zenith-neonBlue/10 text-zenith-neonBlue/70 border border-zenith-neonBlue/20'
                                : 'bg-zinc-800 text-zinc-500 border border-zinc-700'
                        }`}>
                            {adviceSource === 'gemini' ? 'Gemini Vision' : 'Demo Coach'}
                        </span>
                    )}
                    <div className="flex items-start justify-between gap-2">
                        <p className="text-gray-200 text-sm leading-relaxed">{visibleAdvice}</p>
                        <button
                            onClick={() => setVisibleAdvice(null)}
                            className="text-zinc-500 hover:text-white text-xs flex-shrink-0 cursor-pointer"
                        >
                            ✕
                        </button>
                    </div>
                </div>
            )}

            {/* Coach Avatar Button */}
            <button
                onClick={onRequestAnalysis}
                disabled={isAnalyzing}
                aria-label="Request AI coach analysis"
                className="w-20 h-20 md:w-24 md:h-24 rounded-full border-2 border-zenith-neonBlue/30 bg-black/50 backdrop-blur-md shadow-[0_0_20px_rgba(0,204,255,0.1)] flex flex-col items-center justify-center gap-1 pointer-events-auto cursor-pointer transition-all hover:border-zenith-neonBlue/60 hover:shadow-[0_0_30px_rgba(0,204,255,0.2)] disabled:opacity-60 disabled:cursor-wait overflow-hidden relative"
            >
                {/* Scanline effect */}
                <div className="absolute inset-0 bg-[linear-gradient(transparent_50%,rgba(0,204,255,0.03)_50%)] bg-[length:100%_4px] animate-scanline pointer-events-none" />

                {isAnalyzing ? (
                    <>
                        <div className="w-8 h-8 border-2 border-zenith-neonBlue/40 border-t-zenith-neonBlue rounded-full animate-spin" />
                        <span className="text-[8px] font-mono text-zenith-neonBlue/80 tracking-widest uppercase">
                            Analyzing
                        </span>
                    </>
                ) : (
                    <>
                        <div className="w-8 h-8 rounded-full bg-zenith-neonBlue/10 border border-zenith-neonBlue/30 flex items-center justify-center">
                            <svg className="w-4 h-4 text-zenith-neonBlue/80" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                            </svg>
                        </div>
                        <span className="text-[8px] font-mono text-zenith-neonBlue/80 tracking-widest uppercase">
                            Analyze
                        </span>
                    </>
                )}
            </button>
        </div>
    );
};
