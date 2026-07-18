import React, { useState, useRef, useEffect } from 'react'
import VideoPlayer from './VideoPlayer.jsx'
import { useAuth } from '../contexts/AuthContext.jsx'
import { apiVideoSearch, apiGetVideoUrl } from '../api/client.js'

export default function VideoSearchPanel() {
  const { token } = useAuth()
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(10)
  const [scoreThreshold, setScoreThreshold] = useState(0.52)
  const [candidateMultiplier, setCandidateMultiplier] = useState(5)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState('')

  const handleSearch = async () => {
    if (!query.trim()) return
    setLoading(true); setResults(null); setError('')
    try {
      const data = await apiVideoSearch(token, { query, top_k:topK, score_threshold:scoreThreshold, candidate_multiplier:candidateMultiplier })
      setResults(data)
    } catch (err) { setError(err.message) }
    finally { setLoading(false) }
  }

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <div className="page-header">
        <h1 className="page-title">Video Search</h1>
        <p className="page-desc">4-way RRF (BM25 + Vector + GraphRAG + TKG) with cross-encoder reranking.</p>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-5">
        <div className="card mb-5 space-y-4">
          <div>
            <label className="block text-xs font-medium mb-1.5" style={{ color: 'var(--text-secondary)' }}>Query</label>
            <div className="flex gap-2">
              <input className="input-field flex-1"
                placeholder="Describe what you're looking for in the videos…"
                value={query} onChange={e=>setQuery(e.target.value)}
                onKeyDown={e=>e.key==='Enter'&&handleSearch()}/>
              <button onClick={handleSearch} disabled={loading||!query.trim()} className="btn-primary px-5">
                {loading ? <Spinner/> : (
                  <span className="flex items-center gap-1.5">
                    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" className="w-3.5 h-3.5">
                      <circle cx="7" cy="7" r="4.5"/><path strokeLinecap="round" d="M10.5 10.5L14 14"/>
                    </svg>
                    Search
                  </span>
                )}
              </button>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-xs font-medium mb-1.5" style={{ color: 'var(--text-secondary)' }}>Top K Videos</label>
              <input className="input-field" type="number" min={1} max={50} value={topK} onChange={e=>setTopK(Number(e.target.value))}/>
            </div>
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>Score Threshold</label>
                <span className="text-xs font-mono font-semibold" style={{ color: 'var(--accent-text)' }}>{scoreThreshold.toFixed(2)}</span>
              </div>
              <input className="w-full h-1.5 rounded-full appearance-none cursor-pointer mt-3"
                type="range" min={0} max={1} step={0.01} value={scoreThreshold}
                onChange={e=>setScoreThreshold(Number(e.target.value))}
                style={{ accentColor: 'var(--accent)' }}/>
            </div>
            <div>
              <label className="block text-xs font-medium mb-1.5" style={{ color: 'var(--text-secondary)' }}>Candidate Multiplier</label>
              <input className="input-field" type="number" min={1} max={20} value={candidateMultiplier} onChange={e=>setCandidateMultiplier(Number(e.target.value))}/>
            </div>
          </div>
        </div>

        {error && <div className="error-box mb-4">{error}</div>}

        {results && (
          <div>
            <div className="flex items-center gap-3 mb-5">
              <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                {results.total} video{results.total!==1?'s':''} found
              </span>
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>for "{results.query}"</span>
              <div className="flex-1 divider"/>
            </div>

            {results.results?.length===0 ? (
              <div className="text-center py-16">
                <div className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4"
                  style={{ background: 'var(--bg-surface)', border: '1px solid var(--border)' }}>
                  <svg viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1.5" className="w-8 h-8">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z"/>
                  </svg>
                </div>
                <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No videos matched the search criteria.</p>
              </div>
            ) : (
              <div className="space-y-4">
                {results.results?.map((video,i)=><VideoCard key={video.video_id} video={video} rank={i+1} token={token}/>)}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function VideoCard({ video, rank, token }) {
  const [expanded, setExpanded] = useState(false)
  const [showPlayer, setShowPlayer] = useState(false)
  const [videoError, setVideoError] = useState(false)
  const [playUrl, setPlayUrl] = useState(null)
  const [playLoading, setPlayLoading] = useState(false)
  const videoRef = useRef(null)
  const score = Number(video.score)
  const scoreColor = score>0.7 ? '#34d399' : score>0.5 ? '#fbbf24' : '#94a3b8'

  const fetchPlayUrl = async () => {
    try {
      const url = await apiGetVideoUrl(token, video.asset_path, video.video_id)
      return url || video.asset_url
    } catch {
      return video.asset_url
    }
  }

  const handleTogglePlayer = async () => {
    if (showPlayer) { setShowPlayer(false); return }
    setPlayLoading(true)
    setVideoError(false)
    const url = await fetchPlayUrl()
    setPlayUrl(url)
    setShowPlayer(true)
    setPlayLoading(false)
  }

  const seekTo = startTime => {
    if (!playUrl && !video.asset_url) return
    setShowPlayer(true)
    setVideoError(false)
    setTimeout(()=>{
      if (videoRef.current && startTime!=null) { videoRef.current.currentTime=startTime; videoRef.current.play().catch(()=>{}) }
    },100)
  }

  useEffect(()=>{
    if (showPlayer && videoRef.current && video.segments?.[0]?.start_time!=null)
      videoRef.current.currentTime=video.segments[0].start_time
  },[showPlayer])

  return (
    <div className="card relative overflow-hidden">
      {/* Top accent */}
      <div className="absolute top-0 left-0 right-0 h-[2px]"
        style={{ background: `linear-gradient(to right, var(--accent), transparent 60%)` }}/>

      <div className="flex items-start justify-between mb-3">
        <div className="flex items-start gap-3">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center text-xs font-semibold flex-shrink-0"
            style={{ background:'var(--bg-raised)', border:'1px solid var(--border-md)', color:'var(--text-secondary)' }}>
            #{rank}
          </div>
          <div>
            <p className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>{video.filename || video.asset_path?.split('/').pop() || 'Untitled Video'}</p>
            <div className="flex items-center gap-3 mt-0.5" style={{ color: 'var(--text-muted)' }}>
              <span className="text-[11px] font-mono">{video.video_id?.slice(0,16)}…</span>
              {video.upload_time && <span className="text-[11px]">{new Date(video.upload_time).toLocaleDateString('zh-TW')}</span>}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2.5 flex-shrink-0">
          {/* Score with bar */}
          <div className="flex items-center gap-2">
            <div className="w-16 h-2 rounded-full overflow-hidden" style={{ background: 'var(--bg-overlay)' }}>
              <div className="h-full rounded-full" style={{ width:`${Math.min(100,score*100)}%`, background:scoreColor }}/>
            </div>
            <span className="text-sm font-mono font-bold" style={{ color:scoreColor }}>{score.toFixed(4)}</span>
          </div>
          {video.asset_url && (
            <button onClick={handleTogglePlayer} disabled={playLoading}
              className="text-xs px-2.5 py-1 rounded-lg transition-all duration-150 flex items-center gap-1"
              style={showPlayer ? {
                background:'var(--accent-dim)', border:'1px solid var(--accent-glow)', color:'var(--accent-text)',
              } : {
                background:'var(--bg-raised)', border:'1px solid var(--border-md)', color:'var(--text-secondary)',
              }}>
              {playLoading
                ? <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z"/></svg>
                : showPlayer?'⏹ Hide':'▶ Play'}
            </button>
          )}
          {(playUrl || video.asset_url) && (
            <a href={playUrl || video.asset_url} target="_blank" rel="noreferrer" title="Download"
              className="text-xs px-2.5 py-1 rounded-lg transition-all duration-150 flex items-center gap-1"
              style={{ background:'var(--bg-raised)', border:'1px solid var(--border-md)', color:'var(--text-secondary)' }}>
              <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3 h-3">
                <path strokeLinecap="round" d="M7 2v7m0 0l-2.5-2.5M7 9l2.5-2.5M2 11.5h10"/>
              </svg>
            </a>
          )}
        </div>
      </div>

      {showPlayer && (playUrl || video.asset_url) && (
        <div className="mb-3">
          {!videoError ? (
            <div className="rounded-xl overflow-hidden bg-black">
              <VideoPlayer ref={videoRef} src={playUrl || video.asset_url} assetPath={video.asset_path} className="w-full max-h-80"
                onError={() => setVideoError(true)}/>
            </div>
          ) : (
            <div className="rounded-xl px-4 py-3 flex items-center justify-between gap-3"
              style={{ background:'rgba(234,179,8,0.07)', border:'1px solid rgba(234,179,8,0.22)' }}>
              <p className="text-[11px]" style={{ color:'#fde047' }}>此格式瀏覽器無法播放，請下載後用播放器開啟。</p>
              <a href={playUrl || video.asset_url} target="_blank" rel="noreferrer"
                className="btn-secondary text-xs flex-shrink-0">↓ 下載</a>
            </div>
          )}
        </div>
      )}
      {!showPlayer && !video.asset_url && (
        <p className="text-[11px] mb-3" style={{ color: 'var(--text-muted)' }}>No playback URL available</p>
      )}

      <div className="flex items-center justify-between">
        <div className="flex gap-1.5 flex-wrap">
          {getSourceTags(video.segments).map(src=><span key={src} className="badge text-[10px]">{src}</span>)}
        </div>
        <button onClick={()=>setExpanded(v=>!v)}
          className="text-[11px] flex items-center gap-1 transition-colors"
          style={{ color: 'var(--accent-text)' }}>
          {video.segments?.length} segment{video.segments?.length!==1?'s':''}
          <span>{expanded?'▾':'▸'}</span>
        </button>
      </div>

      {expanded && (
        <div className="mt-3 pt-3 space-y-1.5" style={{ borderTop:'1px solid var(--border)' }}>
          {video.segments?.map((seg,j)=><SegmentRow key={j} seg={seg} onPlay={video.asset_url?()=>seekTo(seg.start_time):null}/>)}
        </div>
      )}
    </div>
  )
}

function parseSegText(raw) {
  if (!raw?.trim()) return []
  const s = raw.trim()
  if (/^\w+:\{/.test(s)) {
    return s.split(' / ').reduce((acc, part) => {
      const m = part.match(/^(\w+):\{([\s\S]*)\}$/)
      if (m && m[2].trim()) acc.push({ label: m[1], content: m[2].trim() })
      return acc
    }, [])
  }
  return [{ label: null, content: s }]
}

const SEG_LABEL_CLASS = { contextual:'badge-accent', asr:'badge-green', ocr:'badge-yellow', lvlm:'badge-purple' }

function SegmentRow({ seg, onPlay }) {
  const timeStr = t => t!=null ? `${Math.floor(t/60)}:${String(Math.floor(t%60)).padStart(2,'0')}` : '?'
  const parts = parseSegText(seg.text)
  const segScore = Number(seg.score)
  const scoreColor = segScore>0.7?'#34d399':segScore>0.5?'#fbbf24':'#94a3b8'
  return (
    <div className="rounded-xl px-3 py-2.5" style={{ background:'var(--bg-raised)', border:'1px solid var(--border)' }}>
      <div className="flex items-center justify-between mb-2">
        <button onClick={onPlay} disabled={!onPlay}
          className="text-xs font-mono flex items-center gap-1.5 transition-colors"
          style={{ color: onPlay?'var(--accent-text)':'var(--text-muted)', cursor: onPlay?'pointer':'default' }}>
          {onPlay && <svg viewBox="0 0 10 10" fill="currentColor" className="w-2.5 h-2.5"><path d="M2 1.5l6 3.5-6 3.5z"/></svg>}
          {timeStr(seg.start_time)} → {timeStr(seg.end_time)}
        </button>
        <div className="flex items-center gap-1.5">
          {seg.sources?.map(s=><span key={s} className="badge text-[10px]">{s}</span>)}
          <span className="text-xs font-mono font-bold" style={{ color: scoreColor }}>{segScore.toFixed(4)}</span>
        </div>
      </div>
      {parts.length > 0 && (
        <div className="space-y-2">
          {parts.map((p, i) => (
            <div key={i}>
              {p.label && (
                <span className={`badge text-[10px] uppercase tracking-widest font-bold mb-1 ${SEG_LABEL_CLASS[p.label] || 'badge-accent'}`}>
                  {p.label}
                </span>
              )}
              <p className="text-xs leading-relaxed" style={{ color:'var(--text-primary)' }}>{p.content}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function getSourceTags(segments) {
  if (!segments) return []
  const seen = new Set()
  segments.forEach(s=>s.sources?.forEach(src=>seen.add(src)))
  return [...seen]
}

function Spinner() {
  return (
    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z"/>
    </svg>
  )
}
