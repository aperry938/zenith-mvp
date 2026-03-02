import React from 'react';

export const GenerativeCoach: React.FC = () => {
    return (
        <div className="absolute bottom-5 right-5 w-32 h-32 md:w-48 md:h-48 pointer-events-none z-20 overflow-hidden rounded-full border-2 border-zenith-cyan/30 bg-black/50 backdrop-blur-md shadow-[0_0_20px_rgba(0,255,255,0.1)]">
            <div className="absolute inset-0 bg-[linear-gradient(transparent_50%,rgba(0,255,255,0.05)_50%)] bg-[length:100%_4px] animate-scanline pointer-events-none"></div>

            <div className="w-full h-full flex items-center justify-center relative">
                <div className="absolute inset-0 bg-gradient-to-tr from-zenith-cyan/10 to-purple-500/10 animate-pulse"></div>
                <div className="w-24 h-24 border border-zenith-cyan/40 rounded-full animate-[spin_10s_linear_infinite] opacity-60 flex items-center justify-center">
                    <div className="w-16 h-16 border border-purple-400/40 rounded-full animate-[spin_5s_linear_infinite_reverse]">
                        <div className="w-8 h-8 bg-zenith-cyan/20 rounded-full animate-ping"></div>
                    </div>
                </div>

                <div className="absolute bottom-2 text-[8px] font-mono text-zenith-cyan/80 tracking-widest uppercase">
                    Gemini Vision
                </div>
            </div>
        </div>
    );
};
