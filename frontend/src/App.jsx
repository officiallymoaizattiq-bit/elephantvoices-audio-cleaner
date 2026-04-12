import { useCallback, useEffect, useRef, useState } from 'react'
import axios from 'axios'
import WaveSurfer from 'wavesurfer.js'
import {
  AudioLines,
  CheckCircle2,
  ChevronDown,
  Clock,
  Download,
  Edit3,
  FileAudio,
  Loader2,
  Pause,
  Play,
  RotateCcw,
  Save,
  Sparkles,
  Upload,
  Users,
  Waves,
  XCircle,
  ZoomIn,
  ZoomOut,
} from 'lucide-react'

const API = ''

const CALLER_COLORS = ['#4DA3FF', '#22C55E', '#F59E0B', '#F43F5E', '#9BA7B4']

/* ── helpers ─────────────────────────────────────────────────────────── */
function formatTime(s) {
  if (s == null || isNaN(s)) return '0:00'
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${sec.toString().padStart(2, '0')}`
}

/* ══════════════════════════════════════════════════════════════════════
   UPLOAD VIEW
   ══════════════════════════════════════════════════════════════════════ */
function UploadView({ dragActive, onDrag, onDrop, onPick, fileInputRef }) {
  return (
    <div className="flex items-center justify-center min-h-[70vh]">
      <div className="w-full max-w-xl animate-fade-in-up">
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2.5 mb-4">
            <AudioLines className="h-7 w-7 text-primary" strokeWidth={2.5} />
            <span className="text-2xl font-semibold text-text-primary">ElephantVoices</span>
          </div>
          <p className="text-sm text-text-secondary">Bioacoustic analysis for elephant vocalizations</p>
        </div>
        <div
          onDragEnter={onDrag} onDragLeave={onDrag} onDragOver={onDrag} onDrop={onDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`cursor-pointer rounded-2xl border-2 border-dashed p-16 sm:p-20 text-center transition-all duration-300 ease-out
            ${dragActive
              ? 'border-primary bg-primary/5 shadow-[0_0_40px_-12px_rgba(77,163,255,0.15)]'
              : 'border-border hover:border-border-hover hover:shadow-[0_2px_20px_-6px_rgba(0,0,0,0.4)]'}`}
        >
          <input ref={fileInputRef} type="file" accept=".wav" onChange={onPick} className="hidden" />
          <div className="inline-flex mb-6">
            <div className={`flex h-16 w-16 items-center justify-center rounded-2xl transition-all duration-200
              ${dragActive ? 'bg-primary/10' : 'bg-card border border-border'} animate-float`}>
              <Upload className={`h-7 w-7 transition-colors duration-200 ${dragActive ? 'text-primary' : 'text-text-muted'}`} />
            </div>
          </div>
          <p className="text-lg font-semibold text-text-primary mb-2">Drop a field recording</p>
          <p className="text-sm text-text-secondary mb-5">or click to browse your files</p>
          <div className="inline-flex items-center gap-2 rounded-lg bg-card border border-border px-4 py-1.5">
            <FileAudio className="h-3.5 w-3.5 text-text-muted" />
            <span className="text-xs font-mono text-text-muted">.wav files supported</span>
          </div>
        </div>
      </div>
    </div>
  )
}

/* ══════════════════════════════════════════════════════════════════════
   PROCESSING VIEW
   ══════════════════════════════════════════════════════════════════════ */
const PROCESSING_STEPS = [
  'Loading audio data...',
  'Running noise reduction...',
  'Detecting individual callers...',
  'Extracting harmonic segments...',
]

function ProcessingView({ originalName, processingStep }) {
  return (
    <div className="flex items-center justify-center min-h-[70vh]">
      <div className="w-full max-w-md text-center animate-fade-in-scale">
        <div className="relative inline-flex mb-8">
          <div className="absolute inset-0 rounded-full border-2 border-border" />
          <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-primary animate-spin" />
          <div className="relative flex h-20 w-20 items-center justify-center">
            <Waves className="h-8 w-8 text-primary" />
          </div>
        </div>
        <p className="text-xl font-semibold text-text-primary mb-2">Processing recording</p>
        <p className="text-sm font-mono text-text-muted mb-8 truncate max-w-sm mx-auto">{originalName}</p>
        <div className="max-w-xs mx-auto space-y-3">
          {PROCESSING_STEPS.map((step, i) => (
            <div key={i} className={`flex items-center gap-3 transition-all duration-500
              ${i <= processingStep ? 'opacity-100' : 'opacity-30'}`}>
              <div className={`flex h-6 w-6 items-center justify-center rounded-full transition-all duration-300 ${
                i < processingStep ? 'bg-accent-green/20' : i === processingStep ? 'bg-primary/20' : 'bg-card'}`}>
                {i < processingStep
                  ? <CheckCircle2 className="h-3.5 w-3.5 text-accent-green" />
                  : i === processingStep
                    ? <Loader2 className="h-3.5 w-3.5 text-primary animate-spin" />
                    : <div className="h-1.5 w-1.5 rounded-full bg-text-muted" />}
              </div>
              <span className={`text-xs font-mono text-left ${
                i === processingStep ? 'text-primary' : i < processingStep ? 'text-text-secondary' : 'text-text-muted'
              }`}>{step}</span>
            </div>
          ))}
        </div>
        <div className="mt-8 mx-auto max-w-xs h-1 rounded-full bg-card overflow-hidden">
          <div className="h-full w-1/4 rounded-full bg-primary" style={{ animation: 'progressIndeterminate 1.5s ease-in-out infinite' }} />
        </div>
      </div>
    </div>
  )
}

/* ══════════════════════════════════════════════════════════════════════
   AUDIO HERO — WaveSurfer waveform with play/pause, zoom, time
   ══════════════════════════════════════════════════════════════════════ */
function AudioHero({ audioUrl, clips, callers, onSeekToClip, wsRef }) {
  const containerRef = useRef(null)
  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [zoom, setZoom] = useState(1)

  useEffect(() => {
    if (!containerRef.current || !audioUrl) return

    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: '#1E2A3A',
      progressColor: '#4DA3FF',
      cursorColor: '#4DA3FF',
      cursorWidth: 2,
      height: 120,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      normalize: true,
      backend: 'WebAudio',
    })

    ws.load(audioUrl)

    ws.on('ready', () => {
      setDuration(ws.getDuration())
      wsRef.current = ws
    })
    ws.on('audioprocess', () => setCurrentTime(ws.getCurrentTime()))
    ws.on('seeking', () => setCurrentTime(ws.getCurrentTime()))
    ws.on('play', () => setPlaying(true))
    ws.on('pause', () => setPlaying(false))
    ws.on('finish', () => setPlaying(false))

    return () => { ws.destroy(); wsRef.current = null }
  }, [audioUrl])

  const togglePlay = () => { wsRef.current?.playPause() }

  const handleZoom = (dir) => {
    const next = dir > 0 ? Math.min(zoom * 1.5, 10) : Math.max(zoom / 1.5, 1)
    setZoom(next)
    wsRef.current?.zoom(next * 50)
  }

  /* Build all clips flat for overlay markers */
  const allClips = []
  ;(callers || []).forEach((caller, ci) => {
    ;(caller.clips || []).forEach(clip => {
      allClips.push({ ...clip, callerIndex: ci })
    })
  })

  return (
    <div className="bg-card rounded-2xl border border-border overflow-hidden animate-fade-in-scale shadow-[0_4px_24px_-4px_rgba(0,0,0,0.5)]">
      {/* Waveform */}
      <div className="relative px-6 pt-6 pb-3">
        <div ref={containerRef} className="w-full rounded-xl overflow-hidden" />

        {/* Timeline overlay markers — shows where calls are on the waveform */}
        {duration > 0 && (
          <div className="absolute bottom-2 left-6 right-6 h-1 pointer-events-none">
            {allClips.map((clip, i) => {
              const left = ((clip.start_time || 0) / duration) * 100
              const width = Math.max(0.5, (((clip.end_time || 0) - (clip.start_time || 0)) / duration) * 100)
              return (
                <div key={i} className="absolute h-1 rounded-full opacity-60"
                  style={{ left: `${left}%`, width: `${width}%`, backgroundColor: CALLER_COLORS[clip.callerIndex % CALLER_COLORS.length] }}
                />
              )
            })}
          </div>
        )}
      </div>

      {/* Controls bar */}
      <div className="flex items-center justify-between px-6 py-4 border-t border-border/50">
        <div className="flex items-center gap-3">
          <button onClick={togglePlay}
            className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary text-white
              hover:bg-primary-hover hover:shadow-[0_0_16px_-2px_rgba(77,163,255,0.4)]
              active:scale-95 transition-all duration-200">
            {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4 ml-0.5" />}
          </button>
          <div className="text-sm font-mono text-text-secondary">
            <span className="text-text-primary">{formatTime(currentTime)}</span>
            <span className="text-text-muted mx-1">/</span>
            <span>{formatTime(duration)}</span>
          </div>
        </div>
        <div className="flex items-center gap-1">
          <button onClick={() => handleZoom(-1)}
            className="p-2 rounded-lg text-text-muted hover:text-text-primary hover:bg-card-hover transition-all duration-200">
            <ZoomOut className="h-4 w-4" />
          </button>
          <span className="text-xs font-mono text-text-muted w-10 text-center">{zoom.toFixed(1)}x</span>
          <button onClick={() => handleZoom(1)}
            className="p-2 rounded-lg text-text-muted hover:text-text-primary hover:bg-card-hover transition-all duration-200">
            <ZoomIn className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  )
}

/* ══════════════════════════════════════════════════════════════════════
   INSIGHTS PANEL — summary stats from the cleaning result
   ══════════════════════════════════════════════════════════════════════ */
function InsightsPanel({ result }) {
  const totalCalls = result.clip_count || 0
  const callerCount = result.detected_callers || 0

  /* Compute dominant call type from all clips */
  const typeCounts = {}
  ;(result.callers || []).forEach(c => (c.clips || []).forEach(clip => {
    const t = clip.clip_type || 'unknown'
    typeCounts[t] = (typeCounts[t] || 0) + 1
  }))
  const dominantType = Object.entries(typeCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || 'N/A'

  const insights = [
    { label: 'Total Calls', value: totalCalls, color: 'text-text-primary' },
    { label: 'Callers Detected', value: callerCount, color: 'text-primary' },
    { label: 'Dominant Type', value: dominantType.charAt(0).toUpperCase() + dominantType.slice(1), color: 'text-accent-green' },
    { label: 'Recording', value: result.parent_wav?.replace('.wav', '') || 'N/A', color: 'text-text-secondary', mono: true, small: true },
  ]

  return (
    <div className="bg-card rounded-2xl border border-border p-6 animate-fade-in shadow-[0_2px_12px_-4px_rgba(0,0,0,0.3)]">
      <div className="flex items-center gap-2 mb-5">
        <Sparkles className="h-4 w-4 text-primary" />
        <h2 className="text-sm font-semibold text-text-primary tracking-wide">Overview</h2>
      </div>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
        {insights.map((item, i) => (
          <div key={i} className="group">
            <p className="text-[11px] uppercase tracking-wider text-text-muted mb-1.5 font-medium">{item.label}</p>
            <p className={`${item.small ? 'text-sm truncate' : 'text-2xl'} ${item.mono ? 'font-mono' : ''} font-semibold ${item.color}
              transition-colors duration-200`}>
              {item.value}
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ══════════════════════════════════════════════════════════════════════
   TIMELINE — horizontal call segments, synced with waveform
   ══════════════════════════════════════════════════════════════════════ */
function Timeline({ callers, duration, selectedClipId, onSelectClip, wsRef }) {
  if (!duration || duration <= 0) return null

  const handleClick = (clip) => {
    onSelectClip(clip)
    if (wsRef.current && clip.start_time != null) {
      wsRef.current.seekTo(clip.start_time / duration)
    }
  }

  return (
    <div className="bg-card rounded-2xl border border-border p-6 animate-fade-in shadow-[0_2px_12px_-4px_rgba(0,0,0,0.3)]">
      <div className="flex items-center justify-between mb-5">
        <h2 className="text-sm font-semibold text-text-primary tracking-wide">Timeline</h2>
        <span className="text-xs font-mono text-text-muted">{formatTime(duration)}</span>
      </div>

      {(callers || []).map((caller, ci) => {
        const color = CALLER_COLORS[ci % CALLER_COLORS.length]
        return (
          <div key={caller.elephant_id} className="mb-3 last:mb-0">
            <div className="flex items-center gap-2 mb-1.5">
              <div className="h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
              <span className="text-xs text-text-muted">{caller.elephant_id.replace('_', ' ')}</span>
            </div>
            <div className="relative h-9 bg-base rounded-lg overflow-hidden">
              {/* Time markers */}
              {[0.25, 0.5, 0.75].map(frac => (
                <div key={frac} className="absolute top-0 bottom-0 w-px bg-border" style={{ left: `${frac * 100}%` }} />
              ))}

              {/* Call segments */}
              {(caller.clips || []).map((clip, i) => {
                const left = ((clip.start_time || 0) / duration) * 100
                const width = Math.max(0.8, (((clip.end_time || 0) - (clip.start_time || 0)) / duration) * 100)
                const isSelected = selectedClipId === clip.clip_id
                return (
                  <div key={i} onClick={() => handleClick(clip)}
                    className={`absolute top-1.5 bottom-1.5 rounded-md cursor-pointer transition-all duration-200 ease-out
                      ${isSelected
                        ? 'opacity-100 ring-2 ring-white/40 shadow-[0_0_8px_-1px_rgba(255,255,255,0.15)] scale-y-110'
                        : 'opacity-50 hover:opacity-85 hover:scale-y-105'}`}
                    style={{ left: `${left}%`, width: `${width}%`, backgroundColor: color, transformOrigin: 'center' }}
                    title={`${clip.start_time?.toFixed(1)}s – ${clip.end_time?.toFixed(1)}s`}
                  />
                )
              })}
            </div>
          </div>
        )
      })}

      {/* Time labels */}
      <div className="flex justify-between mt-2">
        <span className="text-[10px] font-mono text-text-muted">0:00</span>
        <span className="text-[10px] font-mono text-text-muted">{formatTime(duration * 0.5)}</span>
        <span className="text-[10px] font-mono text-text-muted">{formatTime(duration)}</span>
      </div>
    </div>
  )
}

/* ══════════════════════════════════════════════════════════════════════
   CALL DETAILS — progressive disclosure panel for selected call
   ══════════════════════════════════════════════════════════════════════ */
function CallDetails({ clip, callerIndex, sessionKey, onClose }) {
  const [analysis, setAnalysis] = useState(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [features, setFeatures] = useState(null)
  const [analyzeError, setAnalyzeError] = useState(false)
  const color = CALLER_COLORS[callerIndex % CALLER_COLORS.length]
  const duration = ((clip.end_time || 0) - (clip.start_time || 0)).toFixed(1)

  const runAnalysis = async () => {
    if (!clip.clip_id) return
    setAnalyzing(true)
    setAnalyzeError(false)
    try {
      const res = await axios.get(`${API}/api/analyze_clip/${clip.clip_id}`)
      setAnalysis(res.data.analysis)
      setFeatures(res.data.features)
    } catch {
      setAnalyzeError(true)
    } finally {
      setAnalyzing(false)
    }
  }

  return (
    <div className="bg-card rounded-2xl border border-border overflow-hidden animate-slide-open shadow-[0_4px_24px_-4px_rgba(0,0,0,0.5)]">
      {/* Header — left accent bar */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-border/40"
        style={{ borderLeft: `3px solid ${color}` }}>
        <div className="flex items-center gap-3">
          <h2 className="text-sm font-semibold text-text-primary tracking-wide">Selected Call</h2>
        </div>
        <button onClick={onClose}
          className="p-1.5 rounded-lg text-text-muted hover:text-text-primary hover:bg-card-hover active:scale-95 transition-all duration-200">
          <XCircle className="h-4 w-4" />
        </button>
      </div>

      <div className="p-6">
        {/* Metadata row — compact horizontal pills */}
        <div className="flex flex-wrap items-center gap-3 mb-6">
          {[
            { label: 'Start', value: `${clip.start_time?.toFixed(1)}s` },
            { label: 'End', value: `${clip.end_time?.toFixed(1)}s` },
            { label: 'Duration', value: `${duration}s` },
            { label: 'Type', value: (clip.clip_type || 'unknown').charAt(0).toUpperCase() + (clip.clip_type || 'unknown').slice(1) },
          ].map((m, i) => (
            <div key={i} className="flex items-center gap-2 rounded-lg bg-base border border-border/60 px-3 py-2">
              <span className="text-[10px] uppercase tracking-wider text-text-muted font-medium">{m.label}</span>
              <span className="text-sm font-mono text-text-primary">{m.value}</span>
            </div>
          ))}
        </div>

        {/* Audio player */}
        {clip.clip_id != null && (
          <div className="mb-5">
            <audio key={`audio-${clip.clip_id}-${sessionKey}`} controls preload="none"
              src={`${API}/api/clip/${clip.clip_id}?session=${sessionKey}`} />
          </div>
        )}

        {/* Analyze button */}
        {!analysis && !analyzing && !analyzeError && (
          <button onClick={runAnalysis}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-primary px-4 py-3 text-sm font-medium text-white
              hover:bg-primary-hover hover:shadow-[0_0_20px_-4px_rgba(77,163,255,0.35)]
              active:scale-[0.98] transition-all duration-200">
            <Sparkles className="h-4 w-4" /> Analyze Vocalization
          </button>
        )}
        {analyzing && (
          <div className="flex items-center justify-center gap-3 rounded-xl bg-primary/5 border border-primary/10 px-4 py-3">
            <Loader2 className="h-4 w-4 text-primary animate-spin" />
            <span className="text-xs font-mono text-primary">Extracting acoustic features...</span>
          </div>
        )}
        {analyzeError && (
          <div className="flex items-center gap-3 rounded-xl border border-accent-rose/20 bg-accent-rose/5 px-4 py-3">
            <XCircle className="h-4 w-4 text-accent-rose" />
            <span className="text-xs font-mono text-accent-rose/80">Analysis failed</span>
            <button onClick={runAnalysis} className="ml-auto text-xs font-mono text-text-muted hover:text-primary transition-all duration-200">retry</button>
          </div>
        )}

        {/* Analysis results */}
        {analysis && (
          <div className="mt-6 grid grid-cols-1 lg:grid-cols-5 gap-4 animate-fade-in-scale">
            {/* AI Insight (60%) */}
            <div className="lg:col-span-3 bg-card-elevated rounded-xl p-6 border border-border/30">
              <div className="flex items-center gap-2 mb-4">
                <Sparkles className="h-4 w-4 text-primary" />
                <h3 className="text-sm font-semibold text-text-primary tracking-wide">AI Analysis</h3>
              </div>
              <p className="text-[13px] leading-[1.7] text-text-secondary">{analysis}</p>
            </div>

            {/* Signal Breakdown (40%) */}
            {features?.is_valid && (
              <div className="lg:col-span-2 bg-card-elevated rounded-xl p-6 border border-border/30">
                <h3 className="text-sm font-semibold text-text-primary mb-5 tracking-wide">Signal Breakdown</h3>
                <div className="space-y-4">
                  {features.fundamental_hz > 0 && <Metric v={`${features.fundamental_hz} Hz`} l="Fundamental (F0)" />}
                  {features.harmonic_count > 0 && <Metric v={features.harmonic_count} l="Harmonic Count" />}
                  {features.rms_db != null && <Metric v={`${features.rms_db} dB`} l="RMS Energy" />}
                  {features.spectral_centroid_hz > 0 && <Metric v={`${features.spectral_centroid_hz} Hz`} l="Spectral Centroid" />}
                  {features.onset_profile && <Metric v={features.onset_profile} l="Temporal Envelope" />}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function Metric({ v, l }) {
  return (
    <div className="flex items-baseline justify-between py-1 border-b border-border/20 last:border-0">
      <span className="text-[11px] text-text-muted tracking-wide">{l}</span>
      <span className="text-sm font-mono text-primary font-medium">{v}</span>
    </div>
  )
}

/* ══════════════════════════════════════════════════════════════════════
   CALLER HEADER with nickname editing
   ══════════════════════════════════════════════════════════════════════ */
function CallerBar({ caller, callerIndex, nicknames, editingId, editValue, setEditValue, setEditingId, saveNickname, saveConfirm }) {
  const color = CALLER_COLORS[callerIndex % CALLER_COLORS.length]
  const name = nicknames[caller.elephant_id]
  const isEditing = editingId === caller.elephant_id
  const clipCount = caller.clips?.length || 0

  return (
    <div className="flex items-center gap-3 bg-card rounded-xl border border-border px-5 py-3.5 mb-1.5 group animate-fade-in
      hover:border-border-hover hover:bg-card-hover/30 transition-all duration-200">
      <div className="h-3 w-3 rounded-full shrink-0" style={{ backgroundColor: color }} />

      {isEditing ? (
        <div className="flex flex-1 items-center gap-2">
          <input autoFocus value={editValue} onChange={e => setEditValue(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') saveNickname(caller.elephant_id); if (e.key === 'Escape') setEditingId(null) }}
            onBlur={() => saveNickname(caller.elephant_id)}
            className="flex-1 rounded-lg border border-border bg-base px-3 py-1.5 text-sm text-text-primary outline-none focus:border-primary/40 transition-all duration-200"
            placeholder="Enter name..." />
          <button onClick={() => saveNickname(caller.elephant_id)} className="p-2 rounded-lg hover:bg-card-hover transition-all duration-200">
            <Save className="h-4 w-4 text-accent-green" />
          </button>
        </div>
      ) : (
        <div className="flex flex-1 items-center gap-2 min-w-0">
          <span className="text-sm font-semibold text-text-primary truncate">{name || caller.elephant_id.replace('_', ' ')}</span>
          {!name && <span className="text-xs text-text-muted italic">unnamed</span>}
          <button onClick={() => { setEditingId(caller.elephant_id); setEditValue(nicknames[caller.elephant_id] || '') }}
            className="p-1 rounded-lg hover:bg-card-hover transition-all duration-200 opacity-0 group-hover:opacity-100">
            <Edit3 className="h-3 w-3 text-text-muted" />
          </button>
        </div>
      )}

      <span className="text-xs font-mono text-text-muted shrink-0">{clipCount} call{clipCount !== 1 ? 's' : ''}</span>

      {saveConfirm === caller.elephant_id && (
        <span className="text-xs text-accent-green font-mono animate-fade-in"><CheckCircle2 className="h-3 w-3 inline mr-1" />Saved</span>
      )}
    </div>
  )
}

/* ══════════════════════════════════════════════════════════════════════
   RESULTS VIEW — the main layout
   ══════════════════════════════════════════════════════════════════════ */
function ResultsView({
  result, originalName, originalUrl, sessionKey,
  nicknames, editingId, editValue, setEditValue, setEditingId,
  saveNickname, saveConfirm, reset,
}) {
  const [selectedClip, setSelectedClip] = useState(null)
  const [selectedCallerIdx, setSelectedCallerIdx] = useState(0)
  const wsRef = useRef(null)
  const [wsDuration, setWsDuration] = useState(0)

  /* Compute duration from first clip's parent or from WaveSurfer */
  useEffect(() => {
    const checkDuration = () => {
      if (wsRef.current) {
        const d = wsRef.current.getDuration()
        if (d > 0) { setWsDuration(d); return }
      }
      requestAnimationFrame(checkDuration)
    }
    checkDuration()
  }, [result])

  /* Find which caller a clip belongs to */
  const handleSelectClip = (clip) => {
    if (selectedClip?.clip_id === clip.clip_id) {
      setSelectedClip(null)
      return
    }
    setSelectedClip(clip)
    ;(result.callers || []).forEach((c, ci) => {
      if ((c.clips || []).some(cl => cl.clip_id === clip.clip_id)) setSelectedCallerIdx(ci)
    })
  }

  return (
    <div className="animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2.5">
          <AudioLines className="h-5 w-5 text-primary" strokeWidth={2.5} />
          <span className="text-lg font-semibold text-primary">ElephantVoices</span>
        </div>
        <div className="flex items-center gap-3">
          {result.download_url && (
            <a href={`${API}${result.download_url}`} download
              className="inline-flex items-center gap-2 rounded-xl border border-border px-4 py-2 text-sm text-text-secondary hover:border-border-hover hover:text-text-primary transition-all duration-200">
              <Download className="h-4 w-4" /> Download
            </a>
          )}
          <button onClick={reset}
            className="inline-flex items-center gap-2 rounded-xl border border-border px-4 py-2 text-sm text-text-secondary hover:border-border-hover hover:text-text-primary transition-all duration-200">
            <RotateCcw className="h-4 w-4" /> New
          </button>
        </div>
      </div>

      {/* Stack: Hero → Insights → Callers → Timeline → Selected Call */}
      <div className="space-y-6">
        {/* 1. Audio Hero (waveform is the primary focus) */}
        <AudioHero audioUrl={originalUrl} clips={null} callers={result.callers} wsRef={wsRef} />

        {/* 2. Insights Panel */}
        <InsightsPanel result={result} />

        {/* 3. Caller bars (compact, with nickname editing) */}
        <div>
          {(result.callers || []).map((caller, ci) => (
            <CallerBar key={caller.elephant_id}
              caller={caller} callerIndex={ci} nicknames={nicknames}
              editingId={editingId} editValue={editValue} setEditValue={setEditValue}
              setEditingId={setEditingId} saveNickname={saveNickname} saveConfirm={saveConfirm} />
          ))}
        </div>

        {/* 4. Timeline (synced with waveform) */}
        <Timeline callers={result.callers} duration={wsDuration}
          selectedClipId={selectedClip?.clip_id} onSelectClip={handleSelectClip} wsRef={wsRef} />

        {/* 5. Selected call details (progressive disclosure) */}
        {selectedClip && (
          <CallDetails clip={selectedClip} callerIndex={selectedCallerIdx}
            sessionKey={sessionKey} onClose={() => setSelectedClip(null)} />
        )}
      </div>
    </div>
  )
}

/* ══════════════════════════════════════════════════════════════════════
   MAIN APP — state management (UNCHANGED logic)
   ══════════════════════════════════════════════════════════════════════ */
export default function App() {
  const [dragActive, setDragActive] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState(null)
  const [originalUrl, setOriginalUrl] = useState(null)
  const [originalName, setOriginalName] = useState('')
  const [result, setResult] = useState(null)
  const [nicknames, setNicknames] = useState({})
  const [editingId, setEditingId] = useState(null)
  const [editValue, setEditValue] = useState('')
  const [saveConfirm, setSaveConfirm] = useState(null)
  const [sessionKey, setSessionKey] = useState(0)
  const [processingStep, setProcessingStep] = useState(0)
  const fileInputRef = useRef(null)

  const reset = useCallback(() => {
    if (originalUrl) URL.revokeObjectURL(originalUrl)
    setOriginalUrl(null); setOriginalName(''); setResult(null); setError(null)
    setIsProcessing(false); setNicknames({}); setEditingId(null)
    setSaveConfirm(null); setSessionKey(k => k + 1); setProcessingStep(0)
  }, [originalUrl])

  const handleFile = useCallback(async (file) => {
    if (!file) return
    if (!file.name.toLowerCase().endsWith('.wav')) { setError('Only .wav files are supported.'); return }
    setError(null)
    if (originalUrl) URL.revokeObjectURL(originalUrl)
    setResult(null); setOriginalUrl(URL.createObjectURL(file)); setOriginalName(file.name)
    setIsProcessing(true); setSaveConfirm(null); setSessionKey(k => k + 1); setProcessingStep(0)
    try {
      const fd = new FormData(); fd.append('file', file)
      const res = await axios.post(`${API}/api/clean`, fd)
      setResult(res.data)
      const n = {}; for (const e of res.data.elephants || []) n[e.elephant_id] = e.nickname || ''
      setNicknames(n)
    } catch (err) { setError(`Processing failed: ${err?.response?.data?.detail || err.message}`) }
    finally { setIsProcessing(false) }
  }, [originalUrl])

  const onDrop = useCallback(e => { e.preventDefault(); e.stopPropagation(); setDragActive(false); handleFile(e.dataTransfer?.files?.[0]) }, [handleFile])
  const onDrag = useCallback(e => { e.preventDefault(); e.stopPropagation(); setDragActive(e.type === 'dragenter' || e.type === 'dragover') }, [])
  const onPick = useCallback(e => handleFile(e.target.files?.[0]), [handleFile])

  const saveNickname = async (id) => {
    try {
      await axios.put(`${API}/api/rename_elephant`, { elephant_id: id, nickname: editValue.trim() })
      setNicknames(p => ({ ...p, [id]: editValue.trim() })); setEditingId(null)
      setSaveConfirm(id); setTimeout(() => setSaveConfirm(null), 2500)
    } catch (err) { setError(`Save failed: ${err?.response?.data?.detail || err.message}`) }
  }

  useEffect(() => {
    const handler = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return
      if (e.key === 'r' && result) {
        const first = result.callers?.[0]
        if (first) { setEditingId(first.elephant_id); setEditValue(nicknames[first.elephant_id] || '') }
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [result, nicknames])

  useEffect(() => {
    if (!isProcessing) return
    let i = 0
    const interval = setInterval(() => { i = Math.min(i + 1, PROCESSING_STEPS.length - 1); setProcessingStep(i) }, 2200)
    return () => clearInterval(interval)
  }, [isProcessing])

  const showLanding = !originalUrl && !isProcessing
  const showResults = result && !isProcessing

  return (
    <div className="min-h-screen bg-base">
      <div className="mx-auto max-w-5xl px-6 py-8">
        {error && (
          <div className="mb-6 flex items-start gap-3 rounded-xl bg-accent-rose/5 border border-accent-rose/20 p-4 text-sm text-accent-rose animate-fade-in">
            <XCircle className="h-5 w-5 shrink-0 mt-0.5" />
            <div className="flex-1">{error}</div>
            <button onClick={() => setError(null)} className="p-1 rounded-lg hover:bg-accent-rose/10 transition-all duration-200">
              <XCircle className="h-4 w-4 text-accent-rose/50 hover:text-accent-rose" />
            </button>
          </div>
        )}
        {showLanding && <UploadView dragActive={dragActive} onDrag={onDrag} onDrop={onDrop} onPick={onPick} fileInputRef={fileInputRef} />}
        {isProcessing && <ProcessingView originalName={originalName} processingStep={processingStep} />}
        {showResults && (
          <ResultsView result={result} originalName={originalName} originalUrl={originalUrl}
            sessionKey={sessionKey} nicknames={nicknames} editingId={editingId} editValue={editValue}
            setEditValue={setEditValue} setEditingId={setEditingId} saveNickname={saveNickname}
            saveConfirm={saveConfirm} reset={reset} />
        )}
      </div>
    </div>
  )
}
