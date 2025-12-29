import './App.css';
import { HUD } from './components/HUD';
import { VideoStage } from './components/VideoStage';
import { useZenithConnection } from './hooks/useZenithConnection';

function App() {
  const { isConnected, metrics, sendFrame } = useZenithConnection();

  return (
    <div className="app-container">
      <header className="zenith-header">
        <div className="logo">ZENith <span className="version">v1.3 (Nervous System)</span></div>
        <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
          {isConnected ? 'LIVE' : 'OFFLINE'}
        </div>
      </header>

      <main className="main-stage">
        <VideoStage onFrame={sendFrame} />
        <HUD metrics={metrics} />
      </main>

      <footer className="zenith-footer">
        <p>Cycle 29: The Nervous System</p>
      </footer>
    </div>
  );
}

export default App;
