import { useCallback, useEffect, useState } from 'react';

export const useZenithVoice = () => {
    const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
    const [ready, setReady] = useState(false);

    useEffect(() => {
        const loadVoices = () => {
            const vs = window.speechSynthesis.getVoices();
            setVoices(vs);
            if (vs.length > 0) setReady(true);
        };

        loadVoices();
        window.speechSynthesis.onvoiceschanged = loadVoices;
    }, []);

    const speak = useCallback((text: string) => {
        if (!text) return;

        // Cancel current speech to prevent queue buildup
        window.speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);

        // Select the best available voice (prefer natural-sounding voices)
        const voice = voices.find(v =>
            v.name.includes('Samantha') // macOS natural voice
        ) || voices.find(v =>
            v.name.includes('Karen') || v.name.includes('Daniel')
        ) || voices.find(v =>
            v.name.includes('Google US English')
        ) || voices.find(v =>
            v.lang.startsWith('en') && v.localService
        ) || voices[0];

        if (voice) utterance.voice = voice;

        utterance.rate = 0.95;
        utterance.pitch = 1.0;

        window.speechSynthesis.speak(utterance);
    }, [voices]);

    return { speak, ready };
};
