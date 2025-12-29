import React from 'react';

// Using a placeholder for now, usually we'd import the asset
// But since I'm generating it, I'll assume it's placed in /public or similar.
// For the React code, I'll assume the image is at /assets/neural_avatar_placeholder.png if I can move it there.
// Or I can just style it nicely with CSS if the image isn't available yet in the build pipeline.

export const GenerativeCoach: React.FC = () => {
    return (
        <div className="absolute bottom-5 right-5 w-32 h-32 md:w-48 md:h-48 pointer-events-none z-20 overflow-hidden rounded-full border-2 border-zenith-cyan/30 bg-black/50 backdrop-blur-md shadow-[0_0_20px_rgba(0,255,255,0.1)]">
            {/* Scanline Effect */}
            <div className="absolute inset-0 bg-[linear-gradient(transparent_50%,rgba(0,255,255,0.05)_50%)] bg-[length:100%_4px] animate-scanline pointer-events-none"></div>

            {/* Placeholder Content (Neural Pulse) */}
            <div className="w-full h-full flex items-center justify-center relative">
                <div className="absolute inset-0 bg-gradient-to-tr from-zenith-cyan/10 to-purple-500/10 animate-pulse"></div>
                {/* 
                In a real scenario, this would be the <img src="/neural_avatar.png" /> 
                For now we create a CSS generative pattern 
            */}
                <div className="w-24 h-24 border border-zenith-cyan/40 rounded-full animate-[spin_10s_linear_infinite] opacity-60 flex items-center justify-center">
                    <div className="w-16 h-16 border border-purple-400/40 rounded-full animate-[spin_5s_linear_infinite_reverse]">
                        <div className="w-8 h-8 bg-zenith-cyan/20 rounded-full animate-ping"></div>
                    </div>
                </div>

                <div className="absolute bottom-2 text-[8px] font-mono text-zenith-cyan/80 tracking-widest uppercase">
                    AI COACH
                </div>
            </div>
        </div>
    );
};
