import { useState, useEffect, useRef, useCallback } from 'react';
import type { Landmark } from '../utils/drawing';

interface SessionStats {
    Duration: string;
    "Avg Flow": string;
    "Stability Events": number;
    "Zone Time": string;
    "Top Pose": string;
}

interface BioDeviation {
    feature: string;
    feature_idx: number;
    value: number;
    ideal_lo: number;
    ideal_hi: number;
    deviation: number;
    direction: 'above' | 'below';
}

interface HeuristicCorrection {
    hud: string;
    spoken: string;
    speak: boolean;
    positive?: boolean;
    vector?: { start: number[]; end: number[] };
    color?: number[];
}

interface ZenithMetrics {
    label: string | null;
    confidence?: number;
    form_assessment?: string;
    flow: number;
    velocity: number;
    q: number;
    advice?: string;
    landmarks?: Landmark[];
    ghost?: number[];
    is_recording?: boolean;
    is_harvesting?: boolean;
    bio_features?: number[];
    bio_quality?: number;
    bio_deviations?: BioDeviation[];
    heuristic?: HeuristicCorrection;
}

export const useZenithConnection = () => {
    const [isConnected, setIsConnected] = useState(false);
    const [isConnecting, setIsConnecting] = useState(false);
    const [metrics, setMetrics] = useState<ZenithMetrics | null>(null);
    const [advice, setAdvice] = useState<string | null>(null);
    const [adviceSource, setAdviceSource] = useState<string | null>(null);
    const [landmarks, setLandmarks] = useState<Landmark[] | null>(null);
    const [ghost, setGhost] = useState<number[] | null>(null);

    // Persistence State
    const [isRecording, setIsRecording] = useState(false);
    const [isHarvesting, setIsHarvesting] = useState(false);
    const [heuristicCorrection, setHeuristicCorrection] = useState<HeuristicCorrection | null>(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [connectionError, setConnectionError] = useState<string | null>(null);
    const [sequence, setSequence] = useState<{
        name: string;
        current_goal: string;
        next_goal: string;
        progress: number;
        completed: boolean;
        announcement?: string;
        hold_seconds?: number;
        hold_target?: number;
    } | null>(null);

    // Session Report
    const [sessionReport, setSessionReport] = useState<SessionStats | null>(null);

    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimeoutRef = useRef<number | undefined>(undefined);
    const reconnectDelayRef = useRef<number>(1000);
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
            setConnectionError(null);
            reconnectDelayRef.current = 1000; // Reset backoff on success
        };

        wsRef.current.onclose = () => {
            console.log("Zenith: Disconnected");
            setIsConnected(false);
            setIsConnecting(false);
            setConnectionError("Connection lost. Reconnecting...");
            // Auto-reconnect with exponential backoff
            if (isMounted.current) {
                const delay = reconnectDelayRef.current;
                console.log(`Zenith: Reconnecting in ${delay}ms...`);
                reconnectTimeoutRef.current = window.setTimeout(() => {
                    if (isMounted.current) connect();
                }, delay);
                reconnectDelayRef.current = Math.min(delay * 2, 10000); // Max 10s
            }
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
                    setAdviceSource(data.source || null);
                    setIsAnalyzing(false);
                } else if (data.type === 'analysis_started') {
                    setIsAnalyzing(true);
                } else if (data.type === 'session_report') {
                    // Received End of Session Report
                    setSessionReport(data.stats);
                } else {
                    // Only update metrics when brain actually produced results
                    if (data.has_result) {
                        setMetrics(data);
                        // Heuristic corrections come with metric frames
                        if (data.heuristic) {
                            setHeuristicCorrection(data.heuristic);
                        } else {
                            setHeuristicCorrection(null);
                        }
                    }
                    if (data.landmarks) setLandmarks(data.landmarks);
                    if (data.ghost) setGhost(data.ghost);
                    if (data.sequence) setSequence(data.sequence);
                    else setSequence(null);

                    // Sync Server State
                    if (data.is_recording !== undefined) setIsRecording(data.is_recording);
                    if (data.is_harvesting !== undefined) setIsHarvesting(data.is_harvesting);
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

    const toggleRecording = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ action: "toggle_record" }));
        }
    }, []);

    const toggleHarvesting = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ action: "toggle_harvest" }));
        }
    }, []);

    const endSession = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ action: "end_session" }));
        }
    }, []);

    const startSequence = useCallback((sequenceKey: string = "strength_flow") => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ action: "start_sequence", sequence: sequenceKey }));
        }
    }, []);

    const stopSequence = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ action: "stop_sequence" }));
            setSequence(null);
        }
    }, []);

    const clearSessionReport = useCallback(() => {
        setSessionReport(null);
    }, []);

    return {
        isConnected,
        isConnecting,
        metrics,
        advice,
        adviceSource,
        landmarks,
        ghost,
        isRecording,
        isHarvesting,
        heuristicCorrection,
        isAnalyzing,
        connectionError,
        sequence,
        sessionReport,
        clearSessionReport,
        sendFrame,
        requestAnalysis,
        toggleRecording,
        toggleHarvesting,
        endSession,
        startSequence,
        stopSequence
    };
};
