import React, { useState, useRef, useEffect } from 'react'
import VideoPlayer from './VideoPlayer.jsx'
import { useAuth } from '../contexts/AuthContext.jsx'
import { apiA2aQuery } from '../api/client.js'

const MODES = [
  { id: 'direct', label: 'Direct', desc: 'intent → search → rerank → LLM' },
  { id: 'agent',  label: 'Agent',  desc: 'smolagents CodeAgent' },
  { id: 'tool',   label: 'Tool',   desc: 'smolagents ToolCallingAgent' },
]

export default function ChatPanel() {
  const { token } = useAuth()
  const [history, setHistory] = useState([])
  const [input, setInput] = useState('')
  const [mode, setMode] = useState('direct')
  const [topK, setTopK] = useState(5)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [lastQ, setLastQ] = useState('')
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [history, loading])

  const handleAsk = async () => {
    const q = input.trim()
    if (!q || loading) return
    setLastQ(q)
    setInput('')
    setError('')
    setLoading(true)
    try {
      const data = await apiA2aQuery(token, { question: q, topK, mode })
      setHistory(h => [...h, {
        question: q,
        answer: data.answer || data.response || JSON.stringify(data),
        sources: data.sources || [],
        chunks_used: data.chunks_used,
      }])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col flex-1 min-h-0">
      {/* Header */}
      <div className="page-header">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="page-title">Chat</h1>
            <p className="page-desc">RAG-powered Q&amp;A with source citations.</p>
          </div>
          <div className="flex items-center gap-3">
            {/* Mode selector */}
            <div className="flex gap-1 p-0.5 rounded-xl"
              style={{ background: 'var(--bg-raised)', border: '1px solid var(--border-md)' }}>
              {MODES.map(m => (
                <ModeTab key={m.id} m={m} active={mode === m.id} onClick={() => setMode(m.id)} />
              ))}
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Top K</span>
              <input type="number" min={1} max={50} value={topK}
                onChange={e => setTopK(Number(e.target.value))}
                className="w-14 input-field text-xs py-1.5" />
            </div>
            {history.length > 0 && (
              <button onClick={() => setHistory([])}
                className="flex items-center gap-1 text-xs transition-colors"
                style={{ color: 'var(--text-muted)' }}
                onMouseEnter={e => e.currentTarget.style.color = 'var(--text-secondary)'}
                onMouseLeave={e => e.currentTarget.style.color = 'var(--text-muted)'}>
                <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3 h-3">
                  <path strokeLinecap="round" d="M2 2l10 10M12 2L2 12"/>
                </svg>
                Clear
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 min-h-0 overflow-y-auto px-6 py-5 space-y-6">
        {history.length === 0 && !loading && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <div className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-5 relative"
                style={{ background: 'var(--accent-dim)', border: '1px solid var(--accent-glow)' }}>
                <div className="absolute inset-0 rounded-2xl blur-md opacity-30"
                  style={{ background: 'var(--accent)' }} />
                <svg viewBox="0 0 24 24" fill="none" stroke="var(--accent-text)" strokeWidth="1.5" className="w-8 h-8 relative">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
                </svg>
              </div>
              <p className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>Ask anything about your knowledge base</p>
              <p className="text-xs mt-1.5" style={{ color: 'var(--text-muted)' }}>
                Mode: <span style={{ color: 'var(--text-secondary)' }}>{mode}</span> · Top K: {topK}
              </p>
            </div>
          </div>
        )}

        {history.map((item, i) => <QAItem key={i} item={item} />)}

        {loading && (
          <div className="space-y-3">
            <div className="flex justify-end">
              <div className="rounded-2xl rounded-br-sm px-4 py-3 text-sm text-white max-w-[80%]"
                style={{ background: 'linear-gradient(135deg,var(--accent),var(--accent-2))' }}>
                {lastQ}
              </div>
            </div>
            <div className="flex gap-3">
              <RaptorAvatar />
              <div className="rounded-2xl rounded-tl-sm px-4 py-3.5 card" style={{ minWidth: 120 }}>
                <div className="flex items-center gap-2">
                  <span className="dot-pulse"><span/><span/><span/></span>
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Thinking… (15–60s)</span>
                </div>
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {error && <div className="error-box text-xs mx-6 mb-3 flex-shrink-0">{error}</div>}

      {/* Input */}
      <div className="px-6 pb-5 flex gap-2 flex-shrink-0">
        <textarea
          className="input-field flex-1 resize-none min-h-[44px] max-h-36"
          placeholder="Ask a question… (Enter to send, Shift+Enter for newline)"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleAsk() } }}
          rows={2}
          disabled={loading}
        />
        <button onClick={handleAsk} disabled={loading || !input.trim()} className="btn-primary px-4 self-end">
          <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
            <path d="M3.105 2.289a.75.75 0 00-.826.95l1.414 4.925A1.5 1.5 0 005.135 9.25h6.115a.75.75 0 010 1.5H5.135a1.5 1.5 0 00-1.442 1.086l-1.414 4.926a.75.75 0 00.826.95 28.896 28.896 0 0015.293-7.154.75.75 0 000-1.115A28.897 28.897 0 003.105 2.289z"/>
          </svg>
        </button>
      </div>
    </div>
  )
}

function ModeTab({ m, active, onClick }) {
  const [hov, setHov] = useState(false)
  return (
    <button onClick={onClick} title={m.desc}
      onMouseEnter={() => setHov(true)} onMouseLeave={() => setHov(false)}
      className="px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-150"
      style={active ? {
        background: 'linear-gradient(135deg,var(--accent),var(--accent-2))',
        color: '#fff',
        boxShadow: '0 2px 8px var(--accent-glow)',
      } : hov ? {
        background: 'var(--bg-overlay)',
        color: 'var(--text-primary)',
      } : { color: 'var(--text-muted)' }}>
      {m.label}
    </button>
  )
}

function RaptorAvatar() {
  return (
    <div className="w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0 mt-1 text-white text-xs font-bold relative"
      style={{ background: 'linear-gradient(135deg,var(--accent),#4f46e5)' }}>
      <div className="absolute inset-0 rounded-xl blur-md opacity-30"
        style={{ background: 'linear-gradient(135deg,var(--accent),#4f46e5)' }} />
      <span className="relative">R</span>
    </div>
  )
}

function QAItem({ item }) {
  const [showSources, setShowSources] = useState(false)
  return (
    <div className="space-y-3">
      {/* Question */}
      <div className="flex justify-end">
        <div className="rounded-2xl rounded-br-sm px-4 py-3 text-sm text-white max-w-[80%] whitespace-pre-wrap break-words"
          style={{ background: 'linear-gradient(135deg,var(--accent),var(--accent-2))', boxShadow: '0 4px 16px var(--accent-glow)' }}>
          {item.question}
        </div>
      </div>

      {/* Answer */}
      <div className="flex gap-3 max-w-[90%]">
        <RaptorAvatar />
        <div className="flex-1 card rounded-2xl rounded-tl-sm">
          <p className="text-sm leading-relaxed whitespace-pre-wrap break-words" style={{ color: 'var(--text-primary)' }}>
            {item.answer}
          </p>
          <div className="flex items-center gap-3 mt-3 pt-3" style={{ borderTop: '1px solid var(--border)' }}>
            {item.chunks_used != null && (
              <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>{item.chunks_used} chunks</span>
            )}
            {item.sources?.length > 0 && (
              <button onClick={() => setShowSources(v => !v)}
                className="flex items-center gap-1 text-[11px] transition-colors"
                style={{ color: 'var(--accent-text)' }}>
                <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.4" className="w-3 h-3">
                  <path strokeLinecap="round" d="M4 7h6M7 4v6"/>
                </svg>
                {item.sources.length} source{item.sources.length !== 1 ? 's' : ''}
                <span>{showSources ? '▾' : '▸'}</span>
              </button>
            )}
          </div>
          {showSources && item.sources?.length > 0 && (
            <div className="mt-3 space-y-2">
              {item.sources.map((src, i) => <SourceCard key={i} src={src} rank={i+1} />)}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function SourceCard({ src, rank }) {
  const [showPlayer, setShowPlayer] = useState(false)
  const [videoError, setVideoError] = useState(false)
  const videoRef = useRef(null)
  const meta = src.metadata || {}
  const filename = meta.filename || src.asset_path?.split('/').pop() || `Source ${rank}`
  const isVideo = meta.type === 'videos' || /\.(mp4|mov|avi|mkv|webm|ts|m2ts|mts)$/i.test(filename)
  const hasUrl = !!(src.storage_uri || src.asset_url)
  const videoUrl = src.storage_uri || src.asset_url
  const timeStr = t => t != null ? `${Math.floor(t/60)}:${String(Math.floor(t%60)).padStart(2,'0')}` : null
  const t0 = timeStr(src.start_time), t1 = timeStr(src.end_time)

  const handlePlay = () => {
    setShowPlayer(true)
    setVideoError(false)
    setTimeout(() => {
      if (videoRef.current && src.start_time != null) {
        videoRef.current.currentTime = src.start_time
        videoRef.current.play().catch(()=>{})
      }
    }, 100)
  }

  return (
    <div className="rounded-xl p-3" style={{ background: 'var(--bg-raised)', border: '1px solid var(--border)' }}>
      <div className="flex items-start justify-between gap-2 mb-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="badge text-[10px]">#{rank}</span>
          <span className="text-[11px] font-medium truncate" style={{ color: 'var(--text-primary)' }}>{filename}</span>
          {t0 && <span className="text-[10px] font-mono flex-shrink-0" style={{ color: 'var(--accent-text)' }}>{t0}{t1?` → ${t1}`:''}</span>}
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className="text-[11px] font-mono" style={{ color: 'var(--text-secondary)' }}>{Number(src.score).toFixed(4)}</span>
          {isVideo && hasUrl && (
            <button onClick={handlePlay}
              className={`text-[11px] px-2 py-0.5 rounded-lg transition-all ${showPlayer?'badge-accent':'badge'}`}>▶</button>
          )}
        </div>
      </div>
      {showPlayer && videoUrl && (
        <div className="mb-2">
          {!videoError ? (
            <div className="rounded-xl overflow-hidden bg-black">
              <VideoPlayer ref={videoRef} src={videoUrl} assetPath={src.asset_path} className="w-full max-h-48"
                onError={() => setVideoError(true)}/>
            </div>
          ) : (
            <div className="rounded-xl px-3 py-2 flex items-center justify-between gap-2"
              style={{ background:'rgba(234,179,8,0.07)', border:'1px solid rgba(234,179,8,0.22)' }}>
              <p className="text-[11px]" style={{ color:'#fde047' }}>此格式瀏覽器無法播放，請下載後用播放器開啟。</p>
              <a href={videoUrl} target="_blank" rel="noreferrer"
                className="badge text-[10px] flex-shrink-0">↓ 下載</a>
            </div>
          )}
        </div>
      )}
      {src.content && <p className="text-[11px] whitespace-pre-wrap leading-relaxed" style={{ color: 'var(--text-secondary)' }}>{src.content}</p>}
    </div>
  )
}
