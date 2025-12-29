import { useState, useEffect, useRef } from 'react'
import './App.css'

interface ZenithMetrics {
  label: string;
  flow: number;
  velocity: number;
  q: number;
}

function App() {
  const [isConnected, setIsConnected] = useState(false)
  const [metrics, setMetrics] = useState<ZenithMetrics | null>(null)

  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wsRef = useRef<WebSocket | null>(null)

  // 1. Setup Webcam
  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 }
        })
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          videoRef.current.play()
        }
      } catch (err) {
        console.error("Camera Error:", err)
      }
    }
    startCamera()
  }, [])

  // 2. Setup WebSocket & Loop
  useEffect(() => {
    wsRef.current = new WebSocket("ws://localhost:8000/ws/stream")

    wsRef.current.onopen = () => setIsConnected(true)
    wsRef.current.onclose = () => setIsConnected(false)
    wsRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        setMetrics(data)
      } catch (e) {
        console.error("Parse Error", e)
      }
    }

    // Transmission Loop (30 FPS)
    const interval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN && videoRef.current && canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d')
        if (ctx) {
          // Draw video to canvas
          ctx.drawImage(videoRef.current, 0, 0, 640, 480)
          // Convert to Blob and Send
          canvasRef.current.toBlob((blob) => {
            if (blob) wsRef.current?.send(blob)
          }, 'image/jpeg', 0.8)
        }
      }
    }, 33)

    return () => {
      clearInterval(interval)
      wsRef.current?.close()
    }
  }, [])

  return (
    <div className="app-container">
      <header className="zenith-header">
        <div className="logo">ZENith <span className="version">v1.1 (Retina)</span></div>
        <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
          {isConnected ? 'LIVE' : 'OFFLINE'}
        </div>
      </header>

      <main className="main-stage">
        {/* Hidden Elements for Processing */}
        <video ref={videoRef} style={{ display: 'none' }} playsInline muted />
        <canvas ref={canvasRef} width={640} height={480} className="live-canvas" />

        <div className="hud-panel">
          <div className="hud-item">
            <h3>POSE</h3>
            <div className="hud-value neon-blue">{metrics?.label || "--"}</div>
          </div>
          <div className="hud-item">
            <h3>FLOW</h3>
            <div className="hud-value neon-purple">{metrics?.flow?.toFixed(0) || "--"}</div>
          </div>
          <div className="hud-item">
            <h3>QUALITY</h3>
            <div className="hud-value neon-green">{metrics?.q?.toFixed(0) || "--"}</div>
          </div>
        </div>
      </main>

      <footer className="zenith-footer">
        <p>Cycle 27: The Retina</p>
      </footer>
    </div>
  )
}

export default App
