import { useState, useEffect, useRef, useCallback } from 'react';

interface ZenithMetrics {
    label: string;
    flow: number;
    velocity: number;
    q: number;
    advice?: string;
}

export const useZenithConnection = () => {
    const [isConnected, setIsConnected] = useState(false);
    const [metrics, setMetrics] = useState<ZenithMetrics | null>(null);
    const [advice, setAdvice] = useState<string | null>(null);
    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimeoutRef = useRef<number | undefined>(undefined);

    const connect = useCallback(() => {
        wsRef.current = new WebSocket("ws://localhost:8000/ws/stream");

        wsRef.current.onopen = () => {
            console.log("Zenith Connected");
            setIsConnected(true);
        };

        wsRef.current.onclose = () => {
            console.log("Zenith Disconnected");
            setIsConnected(false);
            reconnectTimeoutRef.current = setTimeout(connect, 3000);
        };

        wsRef.current.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'advice') {
                    setAdvice(data.text);
                } else {
                    setMetrics(data);
                }
            } catch (e) {
                console.error("Parse Error", e);
            }
        };
    }, []);

    useEffect(() => {
        connect();
        return () => {
            wsRef.current?.close();
            clearTimeout(reconnectTimeoutRef.current);
        };
    }, [connect]);

    const sendFrame = useCallback((blob: Blob) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(blob);
        }
    }, []);

    const requestAnalysis = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ action: "analyze" }));
        }
    }, []);

    return { isConnected, metrics, advice, sendFrame, requestAnalysis };
};
