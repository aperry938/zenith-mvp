import React, { useState, useRef, useEffect } from 'react';

interface InfoTooltipProps {
    text: string;
}

export const InfoTooltip: React.FC<InfoTooltipProps> = ({ text }) => {
    const [show, setShow] = useState(false);
    const [position, setPosition] = useState<'below' | 'above'>('below');
    const [align, setAlign] = useState<'center' | 'right' | 'left'>('center');
    const buttonRef = useRef<HTMLButtonElement>(null);

    useEffect(() => {
        if (show && buttonRef.current) {
            const rect = buttonRef.current.getBoundingClientRect();
            const tooltipWidth = 224; // w-56 = 14rem = 224px
            const tooltipHeight = 80; // approximate

            // Vertical: if tooltip would overflow bottom, show above
            if (rect.bottom + tooltipHeight + 24 > window.innerHeight) {
                setPosition('above');
            } else {
                setPosition('below');
            }

            // Horizontal: check overflow
            const centerX = rect.left + rect.width / 2;
            if (centerX + tooltipWidth / 2 > window.innerWidth - 8) {
                setAlign('right');
            } else if (centerX - tooltipWidth / 2 < 8) {
                setAlign('left');
            } else {
                setAlign('center');
            }
        }
    }, [show]);

    const tooltipPositionClass = position === 'above' ? 'bottom-6' : 'top-6';
    const tooltipAlignClass =
        align === 'right' ? 'right-0' :
        align === 'left' ? 'left-0' :
        'left-1/2 -translate-x-1/2';
    const arrowAlignClass =
        align === 'right' ? 'right-2' :
        align === 'left' ? 'left-2' :
        'left-1/2 -translate-x-1/2';
    const arrowPositionClass = position === 'above'
        ? '-bottom-1.5 rotate-[225deg]'
        : '-top-1.5 rotate-45';

    return (
        <span className="relative inline-block ml-1.5">
            <button
                ref={buttonRef}
                onClick={() => setShow(!show)}
                className="w-4 h-4 rounded-full bg-zinc-700 text-zinc-400 text-[10px] font-bold leading-none flex items-center justify-center hover:bg-zinc-600 hover:text-zinc-200 transition-colors cursor-pointer"
                aria-label="Info"
            >
                ?
            </button>
            {show && (
                <>
                    <div
                        className="fixed inset-0 z-50"
                        onClick={() => setShow(false)}
                    />
                    <div className={`absolute z-[55] ${tooltipPositionClass} ${tooltipAlignClass} w-56 bg-zinc-900 border border-zinc-700 rounded-lg p-3 shadow-xl text-xs text-zinc-300 leading-relaxed`}>
                        {text}
                        <div className={`absolute ${arrowPositionClass} ${arrowAlignClass} w-3 h-3 bg-zinc-900 border-l border-t border-zinc-700`} />
                    </div>
                </>
            )}
        </span>
    );
};
