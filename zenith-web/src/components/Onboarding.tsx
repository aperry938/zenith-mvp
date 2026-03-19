import React, { useState, useEffect } from 'react';

const STORAGE_KEY = 'zenith_onboarded';

interface OnboardingProps {
    forceShow?: boolean;
    onDismiss?: () => void;
}

const steps = [
    {
        icon: (
            <svg className="w-10 h-10 text-zenith-neonBlue" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M12 18.75H4.5a2.25 2.25 0 01-2.25-2.25V7.5A2.25 2.25 0 014.5 5.25h7.5" />
            </svg>
        ),
        title: "Allow Camera",
        description: "Grant camera access when prompted. Your video stays local and is never recorded or sent externally.",
    },
    {
        icon: (
            <svg className="w-10 h-10 text-zenith-neonPurple" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0" />
            </svg>
        ),
        title: "Step Into Frame",
        description: "Stand 6-8 feet from camera. Full body visible. The system detects 10 yoga poses in real time.",
    },
    {
        icon: (
            <svg className="w-10 h-10 text-zenith-neonGreen" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
            </svg>
        ),
        title: "Get Coaching",
        description: "Watch for real-time corrections, quality scores, and spoken feedback. Try a guided flow sequence.",
    },
];

export const Onboarding: React.FC<OnboardingProps> = ({ forceShow, onDismiss }) => {
    const [visible, setVisible] = useState(false);

    useEffect(() => {
        if (forceShow) {
            setVisible(true);
        } else if (!localStorage.getItem(STORAGE_KEY)) {
            setVisible(true);
        }
    }, [forceShow]);

    const dismiss = () => {
        localStorage.setItem(STORAGE_KEY, '1');
        setVisible(false);
        onDismiss?.();
    };

    if (!visible) return null;

    return (
        <div className="absolute inset-0 z-[60] flex items-center justify-center bg-black/85 backdrop-blur-md animate-fadeIn">
            <div className="bg-zenith-panel/95 border border-zinc-700 p-8 rounded-2xl shadow-[0_0_50px_rgba(0,0,0,0.8)] w-[90%] max-w-md flex flex-col items-center gap-6">
                {/* Header */}
                <div className="flex flex-col items-center">
                    <h2 className="text-2xl font-bold uppercase tracking-widest text-white mb-1">ZENith</h2>
                    <p className="text-zinc-400 text-sm">AI-Powered Yoga Coach</p>
                </div>

                <div className="w-full h-px bg-gradient-to-r from-transparent via-zinc-600 to-transparent" />

                {/* Steps */}
                <div className="flex flex-col gap-5 w-full">
                    {steps.map((step, i) => (
                        <div key={i} className="flex items-start gap-4">
                            <div className="flex-shrink-0 w-12 h-12 rounded-lg bg-zinc-800/50 border border-zinc-700 flex items-center justify-center">
                                {step.icon}
                            </div>
                            <div>
                                <h3 className="text-sm font-bold text-white tracking-wider">{step.title}</h3>
                                <p className="text-xs text-zinc-400 leading-relaxed mt-0.5">{step.description}</p>
                            </div>
                        </div>
                    ))}
                </div>

                {/* Action */}
                <button
                    onClick={dismiss}
                    className="mt-2 px-8 py-3 bg-white text-black font-bold uppercase tracking-widest rounded-full hover:bg-zenith-neonBlue hover:shadow-[0_0_20px_rgba(0,255,255,0.4)] transition-all cursor-pointer"
                >
                    Begin Practice
                </button>
            </div>
        </div>
    );
};
