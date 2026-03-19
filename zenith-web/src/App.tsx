import { useState, useEffect } from 'react';
import { VideoStage } from './components/VideoStage';
import { HUD } from './components/HUD';
import { GhostOverlay } from './components/GhostOverlay';
import { SessionControls } from './components/SessionControls';
import { GenerativeCoach } from './components/GenerativeCoach';
import { SessionReport } from './components/SessionReport';
import { BiomechanicalPanel } from './components/BiomechanicalPanel';
import { SequenceBar } from './components/SequenceBar';
import { Onboarding } from './components/Onboarding';
import { SessionHistory } from './components/SessionHistory';
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
    intensity,
    clearSessionReport,
    sendFrame,
    requestAnalysis,
    toggleRecording,
    toggleHarvesting,
    endSession,
    startSequence,
    stopSequence,
    setIntensity,
    persona,
    setPersona,
  } = useZenithConnection();

  const [showOnboarding, setShowOnboarding] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [currentStreak, setCurrentStreak] = useState(0);

  useEffect(() => {
    fetch('http://localhost:8000/api/streaks')
      .then(res => res.json())
      .then(data => setCurrentStreak(data.current_streak ?? 0))
      .catch(() => { /* streak fetch failed silently */ });
  }, []);

  return (
    <div className="w-screen h-screen flex flex-col bg-zenith-bg text-gray-100 overflow-hidden">
      {/* Header */}
      <header className="h-16 border-b border-zinc-800 flex justify-between items-center px-8 bg-zenith-panel z-40 relative">
        <div className="font-bold text-2xl tracking-widest text-white uppercase">
          ZENith <span className="text-sm text-zinc-500 font-normal normal-case ml-2">v2.6</span>
        </div>

        <div className="flex items-center gap-3">
          {currentStreak > 0 && (
            <div className="flex items-center gap-1 px-2.5 py-1 rounded bg-orange-500/10 border border-orange-500/30" title={`${currentStreak}-day streak`}>
              <span className="text-orange-400 text-sm">&#x1F525;</span>
              <span className="text-orange-400 text-xs font-mono font-bold">{currentStreak}</span>
            </div>
          )}
          <button
            onClick={() => setShowHistory(h => !h)}
            aria-label="View session history"
            className="px-2.5 py-1 rounded text-xs font-mono tracking-wider text-zinc-500 hover:text-white hover:bg-zinc-800 transition-colors cursor-pointer"
          >
            History
          </button>
          <button
            onClick={() => setShowOnboarding(true)}
            aria-label="Show help guide"
            className="w-7 h-7 rounded-full border border-zinc-700 text-zinc-500 hover:text-white hover:border-zinc-500 transition-colors text-xs font-bold cursor-pointer"
          >
            ?
          </button>
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
        <HUD metrics={metrics} advice={advice} heuristicCorrection={heuristicCorrection} onRequestAnalysis={requestAnalysis} breathCue={sequence?.breath_cue} />

        <SessionControls
          isRecording={isRecording}
          isHarvesting={isHarvesting}
          isSequencing={!!sequence}
          onToggleRecord={toggleRecording}
          onToggleHarvest={toggleHarvesting}
          onEndSession={endSession}
          onStartSequence={startSequence}
          intensity={intensity}
          onSetIntensity={setIntensity}
          persona={persona}
          onSetPersona={setPersona}
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
        <p>ZENith v2.6 — Real-Time Biomechanical Movement Analysis</p>
      </footer>

      {/* Overlays */}
      <Onboarding forceShow={showOnboarding} onDismiss={() => setShowOnboarding(false)} />
      {showHistory && <SessionHistory onClose={() => setShowHistory(false)} />}
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
