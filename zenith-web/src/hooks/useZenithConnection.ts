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
    const [isConnecting, setIsConnecting] = useState(false);
    const [metrics, setMetrics] = useState<ZenithMetrics | null>(null);
    const [advice, setAdvice] = useState<string | null>(null);

    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimeoutRef = useRef<number | undefined>(undefined);
    const isMounted = useRef(false);

    const connect = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN || wsRef.current?.readyState === WebSocket.CONNECTING) {
            return;
        }

        setIsConnecting(true);
        console.log("Zenith: Attempting Connection...");

        wsRef.current = new WebSocket("ws://localhost:8000/ws/stream");

        wsRef.current.onopen = () => {
            console.log("Zenith: Connected");
            setIsConnected(true);
            setIsConnecting(false);
        };

        wsRef.current.onclose = () => {
            console.log("Zenith: Disconnected");
            setIsConnected(false);
            setIsConnecting(true); // Technically waiting to connect
            // Simple reconnect logic
            reconnectTimeoutRef.current = setTimeout(connect, 3000);
        };

        wsRef.current.onerror = (err) => {
            console.error("Zenith: WS Error", err);
            wsRef.current?.close();
        }

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
        isMounted.current = true;
        connect();

        return () => {
            isMounted.current = false;
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

    return { isConnected, isConnecting, metrics, advice, sendFrame, requestAnalysis };
};
