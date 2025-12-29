import { VideoStage } from './components/VideoStage';
import { HUD } from './components/HUD';
import { GhostOverlay } from './components/GhostOverlay';
import { SessionControls } from './components/SessionControls';
import { GenerativeCoach } from './components/GenerativeCoach';
import { SequenceDisplay } from './components/SequenceDisplay';
import { useZenithConnection } from './hooks/useZenithConnection';
import { ErrorBoundary } from './components/ErrorBoundary';

function ZenithApp() {
  const {
    isConnected,
    isConnecting,
    metrics,
    advice,
    landmarks,
    ghost,
    isRecording,
    isHarvesting,
    sequenceState,
    sendFrame,
    requestAnalysis,
    toggleRecording,
    toggleHarvesting
  } = useZenithConnection();

  return (
    <div className="w-screen h-screen flex flex-col bg-zenith-bg text-gray-100 overflow-hidden">
      {/* Header */}
      <header className="h-16 border-b border-zinc-800 flex justify-between items-center px-8 bg-zenith-panel z-10 relative">
        <div className="font-bold text-2xl tracking-widest text-white uppercase">
          ZENith <span className="text-sm text-zinc-500 font-normal normal-case ml-2">v2.0 (Teacher)</span>
        </div>

        <div className="flex items-center gap-3">
          {isConnecting && !isConnected && (
            <span className="flex h-3 w-3 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-yellow-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-yellow-500"></span>
            </span>
          )}
          <div className={`px-3 py-1 rounded text-xs font-bold tracking-wider transition-colors duration-500 ${isConnected
              ? 'bg-zenith-neonGreen text-black shadow-[0_0_10px_rgba(0,255,153,0.3)]'
              : 'bg-zinc-800 text-zinc-500'
            }`}>
            {isConnected ? 'LIVE' : isConnecting ? 'CONNECTING...' : 'OFFLINE'}
          </div>
        </div>
      </header>

      {/* Main Stage */}
      <main className="flex-1 flex justify-center items-center relative bg-[radial-gradient(circle_at_center,#111_0%,#000_100%)]">
        {/* Layer 0: Video */}
        <VideoStage onFrame={sendFrame} />

        {/* Layer 1: The Mirage (Ghost/Skeleton) */}
        <GhostOverlay landmarks={landmarks} ghostFlat={ghost} />

        {/* Layer 2: Sequence Guidance (Teacher) */}
        <SequenceDisplay state={sequenceState} />

        {/* Layer 3: UI Overlays */}
        <HUD metrics={metrics} advice={advice} onRequestAnalysis={requestAnalysis} />

        <SessionControls
          isRecording={isRecording}
          isHarvesting={isHarvesting}
          onToggleRecord={toggleRecording}
          onToggleHarvest={toggleHarvesting}
        />

        {/* Layer 4: The Generator (AI Coach Avatar) */}
        <GenerativeCoach />
      </main>

      {/* Footer */}
      <footer className="h-10 border-t border-zinc-800 flex justify-center items-center text-xs text-zinc-600 bg-zenith-panel z-10 relative">
        <p>Cycle 38: The Teacher</p>
      </footer>
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <ZenithApp />
    </ErrorBoundary>
  )
}
