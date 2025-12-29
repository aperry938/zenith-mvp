import { useState, useEffect, useRef } from 'react';
import './App.css';
import { HUD } from './components/HUD';
import { VideoStage } from './components/VideoStage';

interface ZenithMetrics {
  label: string;
  flow: number;
  velocity: number;
  q: number;
}

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [metrics, setMetrics] = useState<ZenithMetrics | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    wsRef.current = new WebSocket("ws://localhost:8000/ws/stream");

    wsRef.current.onopen = () => setIsConnected(true);
    wsRef.current.onclose = () => setIsConnected(false);
    wsRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setMetrics(data);
      } catch (e) {
        console.error("Parse Error", e);
      }
    };

    return () => {
      wsRef.current?.close();
    };
  }, []);

  const handleFrame = (blob: Blob) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(blob);
    }
  };

  return (
    <div className="app-container">
      <header className="zenith-header">
        <div className="logo">ZENith <span className="version">v1.2 (HUD)</span></div>
        <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
          {isConnected ? 'LIVE' : 'OFFLINE'}
        </div>
      </header>

      <main className="main-stage">
        <VideoStage onFrame={handleFrame} />
        <HUD metrics={metrics} />
      </main>

      <footer className="zenith-footer">
        <p>Cycle 28: The HUD</p>
      </footer>
    </div>
  );
}

export default App;
