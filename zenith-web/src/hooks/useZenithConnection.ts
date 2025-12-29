import { useState, useEffect, useRef, useCallback } from 'react';
import type { Landmark } from '../utils/drawing';

interface ZenithMetrics {
    label: string;
    flow: number;
    velocity: number;
    q: number;
    advice?: string;
    landmarks?: Landmark[];
    ghost?: number[];
}

export const useZenithConnection = () => {
    const [isConnected, setIsConnected] = useState(false);
    const [isConnecting, setIsConnecting] = useState(false);
    const [metrics, setMetrics] = useState<ZenithMetrics | null>(null);
    const [advice, setAdvice] = useState<string | null>(null);
    const [landmarks, setLandmarks] = useState<Landmark[] | null>(null);
    const [ghost, setGhost] = useState<number[] | null>(null);

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
                    if (data.landmarks) setLandmarks(data.landmarks);
                    if (data.ghost) setGhost(data.ghost);
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

    return { isConnected, isConnecting, metrics, advice, landmarks, ghost, sendFrame, requestAnalysis };
};
