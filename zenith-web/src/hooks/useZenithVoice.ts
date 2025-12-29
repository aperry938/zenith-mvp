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

        // Select a voice (Prefer Female/Google/Microsoft/Samantha)
        const voice = voices.find(v =>
            v.name.includes('Google US English') ||
            v.name.includes('Samantha') ||
            v.name.includes('Microsoft Zira')
        ) || voices[0];

        if (voice) utterance.voice = voice;

        // Tweak parameters for a calm "Yoga Instructor" vibe
        utterance.rate = 0.9;
        utterance.pitch = 1.0;

        window.speechSynthesis.speak(utterance);
    }, [voices]);

    return { speak, ready };
};
