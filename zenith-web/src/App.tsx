import { VideoStage } from './components/VideoStage';
import { HUD } from './components/HUD';
import { GhostOverlay } from './components/GhostOverlay';
import { SessionControls } from './components/SessionControls';
import { GenerativeCoach } from './components/GenerativeCoach';
import { SessionReport } from './components/SessionReport';
import { BiomechanicalPanel } from './components/BiomechanicalPanel';
import { SequenceBar } from './components/SequenceBar';
import { useZenithConnection } from './hooks/useZenithConnection';
import { ErrorBoundary } from './components/ErrorBoundary';

function ZenithApp() {
  const {
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
    stopSequence,
  } = useZenithConnection();

  return (
    <div className="w-screen h-screen flex flex-col bg-zenith-bg text-gray-100 overflow-hidden">
      {/* Header */}
      <header className="h-16 border-b border-zinc-800 flex justify-between items-center px-8 bg-zenith-panel z-40 relative">
        <div className="font-bold text-2xl tracking-widest text-white uppercase">
          ZENith <span className="text-sm text-zinc-500 font-normal normal-case ml-2">v2.4</span>
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

      {/* Connection Error Banner */}
      {connectionError && !isConnected && (
        <div className="absolute top-16 left-0 right-0 z-50 flex justify-center pointer-events-none">
          <div className="bg-yellow-500/10 border border-yellow-500/30 text-yellow-400 text-xs font-mono tracking-wider px-4 py-2 rounded-b-lg backdrop-blur-sm">
            {connectionError}
          </div>
        </div>
      )}

      {/* Main Stage */}
      <main className="flex-1 flex justify-center items-center relative bg-[radial-gradient(circle_at_center,#111_0%,#000_100%)]">
        {/* Layer 0+1: Video + Skeleton in shared coordinate space */}
        <div className="relative" style={{ transform: 'scaleX(-1)' }}>
          <VideoStage onFrame={sendFrame} />
          <GhostOverlay
            landmarks={landmarks}
            ghostFlat={ghost}
            correctionVector={heuristicCorrection?.vector}
            correctionColor={heuristicCorrection?.color}
          />
        </div>

        {/* Layer 2: UI Overlays */}
        <BiomechanicalPanel
          bioQuality={metrics?.bio_quality}
          bioDeviations={metrics?.bio_deviations}
          bioFeatures={metrics?.bio_features}
          poseLabel={metrics?.label}
        />
        <HUD metrics={metrics} advice={advice} heuristicCorrection={heuristicCorrection} onRequestAnalysis={requestAnalysis} />

        <SessionControls
          isRecording={isRecording}
          isHarvesting={isHarvesting}
          isSequencing={!!sequence}
          onToggleRecord={toggleRecording}
          onToggleHarvest={toggleHarvesting}
          onEndSession={endSession}
          onStartSequence={startSequence}
        />

        {/* Pose Sequence Bar */}
        {sequence && <SequenceBar sequence={sequence} onStop={stopSequence} />}

        {/* AI Coach Avatar */}
        <GenerativeCoach advice={advice} adviceSource={adviceSource} isAnalyzing={isAnalyzing} onRequestAnalysis={requestAnalysis} />

        {/* Session Report Overlay */}
        <SessionReport stats={sessionReport} onClose={clearSessionReport} />
      </main>

      {/* Footer */}
      <footer className="h-10 border-t border-zinc-800 flex justify-center items-center text-xs text-zinc-600 bg-zenith-panel z-40 relative">
        <p>ZENith v2.4 — Real-Time Biomechanical Movement Analysis</p>
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
