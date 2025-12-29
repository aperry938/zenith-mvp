import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [isConnected, setIsConnected] = useState(false)

  useEffect(() => {
    // Simulate connection check
    const timer = setTimeout(() => setIsConnected(true), 1000)
    return () => clearTimeout(timer)
  }, [])

  return (
    <div className="app-container">
      <header className="zenith-header">
        <div className="logo">ZENith <span className="version">v1.0 (Face)</span></div>
        <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
          {isConnected ? 'LIVE' : 'OFFLINE'}
        </div>
      </header>

      <main className="main-stage">
        <div className="video-placeholder">
          <p>Waiting for Stream...</p>
        </div>

        <div className="hud-panel">
          <h3>Correction</h3>
          <p className="hud-text">--</p>
        </div>
      </main>

      <footer className="zenith-footer">
        <p>Cycle 26: The Face</p>
      </footer>
    </div>
  )
}

export default App
