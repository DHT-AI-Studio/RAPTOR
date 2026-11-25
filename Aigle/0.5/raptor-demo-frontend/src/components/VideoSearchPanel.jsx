import React, { useState, useRef, useEffect } from 'react'
import { createPortal } from 'react-dom'
import VideoPlayer from './VideoPlayer.jsx'
import { useAuth } from '../contexts/AuthContext.jsx'
import { apiVideoSearch, apiGetVideoUrl, apiGetThumbnailUrl } from '../api/client.js'
import { FaceThumb } from './VideoIdentityDetail.jsx'

export default function VideoSearchPanel() {
  const { token } = useAuth()
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(10)
  const [scoreThreshold, setScoreThreshold] = useState(0.52)
  const [candidateMultiplier, setCandidateMultiplier] = useState(5)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState(null) // a result video, or null = grid view

  const handleSearch = async () => {
    if (!query.trim()) return
    setLoading(true); setResults(null); setError(''); setSelected(null)
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
        <p className="page-desc">4-way RRF (BM25 + Vector + GraphRAG + TKG) with cross-encoder reranking, plus People name/alias matches shown per video.</p>
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

        {results && selected && (
          <VideoResultDetail video={selected} rank={results.results.indexOf(selected)+1} token={token}
            onBack={() => setSelected(null)}/>
        )}

        {results && !selected && (
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
              <div className="grid gap-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 260px))' }}>
                {results.results?.map((video,i)=>
                  <VideoGridCard key={video.video_id} video={video} rank={i+1} token={token} onClick={() => setSelected(video)}/>)}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

// A video with zero content-retriever hits only got into the results list
// via a name/alias match (see video_search's own docstring) -- its `score`
// is face-detection confidence, not a content-relevance score, and isn't
// comparable to the other cards' scores even though it shares the field.
const isPeopleOnlyMatch = video => !video.segments?.length && video.people?.length > 0

// Grid view: thumbnail-led, same pattern as the Identity page's video
// library (VideoThumbCard) -- one consistent "this is how Raptor shows a
// list of videos" language app-wide. Click opens the detail view below.
function VideoGridCard({ video, rank, token, onClick }) {
  const [thumbUrl, setThumbUrl] = useState(null)
  const [failed, setFailed] = useState(false)
  const objectUrlRef = useRef(null)
  const score = Number(video.score)
  const peopleOnly = isPeopleOnlyMatch(video)
  const scoreColor = peopleOnly ? '#d8b4fe' : score>0.7 ? '#34d399' : score>0.5 ? '#fbbf24' : '#94a3b8'

  useEffect(() => {
    if (!video.asset_path) { setFailed(true); return }
    let cancelled = false
    apiGetThumbnailUrl(token, video.asset_path, video.video_id, 1)
      .then(url => {
        if (cancelled) { URL.revokeObjectURL(url); return }
        objectUrlRef.current = url
        setThumbUrl(url)
      })
      .catch(() => { if (!cancelled) setFailed(true) })
    return () => {
      cancelled = true
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current)
    }
  }, [token, video.asset_path, video.video_id])

  return (
    <button onClick={onClick}
      className="text-left group rounded-2xl overflow-hidden transition-all duration-200 hover:-translate-y-1"
      style={{ background: 'var(--glass-bg)', border: '1px solid var(--glass-border)', boxShadow: 'var(--shadow-card)' }}>
      <div className="relative overflow-hidden aspect-video flex items-center justify-center" style={{ background: 'var(--bg-raised)' }}>
        {thumbUrl ? (
          <img src={thumbUrl} alt="" className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105" />
        ) : failed ? (
          <span className="text-3xl opacity-60">🎬</span>
        ) : (
          <Spinner/>
        )}
        <span className="absolute top-2 left-2 text-[10px] font-semibold px-1.5 py-0.5 rounded-md" style={{ background:'rgba(0,0,0,0.6)', color:'#fff' }}>#{rank}</span>
        <span className="absolute top-2 right-2 text-[11px] font-mono font-bold px-2 py-0.5 rounded-lg flex items-center gap-1"
          style={{ background:'rgba(0,0,0,0.6)', color:scoreColor }}
          title={peopleOnly ? 'Found by face match only -- not a content-relevance score' : 'Content-relevance score'}>
          {peopleOnly && <span aria-hidden="true">👤</span>}
          {score.toFixed(2)}
        </span>
        <div className="absolute inset-0 transition-opacity duration-200 opacity-0 group-hover:opacity-100 flex items-center justify-center" style={{ background:'rgba(0,0,0,0.2)' }}>
          <div className="w-10 h-10 rounded-full flex items-center justify-center" style={{ background:'rgba(255,255,255,0.18)', backdropFilter:'blur(4px)' }}>
            <svg viewBox="0 0 16 16" fill="white" className="w-4 h-4 ml-0.5"><path d="M4 2.5l9 5.5-9 5.5z"/></svg>
          </div>
        </div>
      </div>
      <div className="px-3 py-2.5">
        <p className="text-xs font-medium truncate" style={{ color: 'var(--text-primary)' }}>{video.filename || video.asset_path?.split('/').pop() || 'Untitled Video'}</p>
        <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>
          {video.upload_time ? new Date(video.upload_time).toLocaleDateString('zh-TW') : video.video_id?.slice(0,16)+'…'}
        </p>
      </div>
    </button>
  )
}

// Detail view: back button + sticky player on the left, score/sources/
// segments/people on the right -- the same split the Identity page's own
// video detail screen uses.
function VideoResultDetail({ video, rank, token, onBack }) {
  const [playUrl, setPlayUrl] = useState(null)
  const [videoError, setVideoError] = useState(false)
  const videoRef = useRef(null)
  const score = Number(video.score)
  const peopleOnly = isPeopleOnlyMatch(video)
  const scoreColor = peopleOnly ? '#d8b4fe' : score>0.7 ? '#34d399' : score>0.5 ? '#fbbf24' : '#94a3b8'

  useEffect(() => {
    let cancelled = false
    apiGetVideoUrl(token, video.asset_path, video.video_id)
      .then(url => { if (!cancelled) setPlayUrl(url || video.asset_url) })
      .catch(() => { if (!cancelled) setPlayUrl(video.asset_url) })
    return () => { cancelled = true }
  }, [token, video.asset_path, video.video_id])

  const seekTo = startTime => {
    setVideoError(false)
    if (videoRef.current && startTime != null) { videoRef.current.currentTime = startTime; videoRef.current.play().catch(() => {}) }
  }

  return (
    <>
      <div className="flex items-center gap-3 mb-4 min-w-0">
        <button onClick={onBack} aria-label="Back to results"
          className="flex items-center gap-1.5 pl-2.5 pr-3.5 h-9 rounded-xl transition-colors flex-shrink-0 text-xs font-semibold focus-visible:outline focus-visible:outline-2"
          style={{ color: 'var(--text-primary)', background: 'var(--bg-raised)', border: '1px solid var(--border-md)', outlineColor: 'var(--accent)' }}
          onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-overlay)'}
          onMouseLeave={e => e.currentTarget.style.background = 'var(--bg-raised)'}>
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" className="w-3.5 h-3.5 flex-shrink-0">
            <path strokeLinecap="round" strokeLinejoin="round" d="M10 3L5 8l5 5"/>
          </svg>
          Back
        </button>
        <div className="min-w-0">
          <p className="text-sm font-medium truncate" style={{ color: 'var(--text-primary)' }}>{video.filename || video.asset_path?.split('/').pop() || 'Untitled Video'}</p>
          <p className="text-[11px] font-mono" style={{ color: 'var(--text-muted)' }}>
            {video.video_id?.slice(0,16)}…{video.upload_time && ` · ${new Date(video.upload_time).toLocaleDateString('zh-TW')}`}
          </p>
        </div>
      </div>

      <div className="flex gap-6 items-start flex-wrap xl:flex-nowrap">
        <div className="flex-shrink-0 self-start sticky top-0" style={{ width: 420, maxWidth: '100%' }}>
          <div className="rounded-2xl overflow-hidden" style={{ border: '1px solid var(--border)', boxShadow: 'var(--shadow-card)' }}>
            {!playUrl ? (
              <div className="aspect-video flex items-center justify-center" style={{ background: 'var(--bg-raised)' }}><Spinner/></div>
            ) : !videoError ? (
              <VideoPlayer ref={videoRef} src={playUrl} assetPath={video.asset_path} className="w-full block" onError={() => setVideoError(true)}/>
            ) : (
              <div className="p-3 rounded-xl flex items-center justify-between gap-3" style={{ background:'rgba(234,179,8,0.07)', border:'1px solid rgba(234,179,8,0.22)' }}>
                <p className="text-[11px]" style={{ color:'#fde047' }}>此格式瀏覽器無法播放，請下載後用播放器開啟。</p>
                <a href={playUrl} target="_blank" rel="noreferrer" className="btn-secondary text-xs flex-shrink-0">↓ 下載</a>
              </div>
            )}
          </div>
        </div>

        <div className="flex-1 min-w-0" style={{ minWidth: 380 }}>
          <div className="card">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2.5">
                <span className="text-xs font-semibold px-2 py-1 rounded-lg" style={{ background: 'var(--bg-raised)', border:'1px solid var(--border-md)', color: 'var(--text-secondary)' }}>#{rank}</span>
                <div className="w-16 h-2 rounded-full overflow-hidden" style={{ background: 'var(--bg-overlay)' }}>
                  <div className="h-full rounded-full" style={{ width:`${Math.min(100,score*100)}%`, background:scoreColor }}/>
                </div>
                <span className="text-sm font-mono font-bold" style={{ color:scoreColor }}>{score.toFixed(4)}</span>
                {peopleOnly && (
                  <span className="badge badge-purple text-[10px]" title="No content-retriever hit this video -- found only by matching a person's name/alias; the number above is face-match confidence, not a relevance score">
                    👤 person match only
                  </span>
                )}
              </div>
              {playUrl && (
                <a href={playUrl} target="_blank" rel="noreferrer" title="Download"
                  className="text-xs px-2.5 py-1 rounded-lg transition-all duration-150 flex items-center gap-1"
                  style={{ background:'var(--bg-raised)', border:'1px solid var(--border-md)', color:'var(--text-secondary)' }}>
                  <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3 h-3">
                    <path strokeLinecap="round" d="M7 2v7m0 0l-2.5-2.5M7 9l2.5-2.5M2 11.5h10"/>
                  </svg>
                </a>
              )}
            </div>

            <div className="flex gap-1.5 flex-wrap">
              {getSourceTags(video.segments).map(src=><span key={src} className="badge text-[10px]">{src}</span>)}
            </div>

            <PeopleRow people={video.people} onPlay={playUrl ? seekTo : null}/>

            {video.segments?.length > 0 && (
              <div className="mt-3 pt-3 space-y-1.5" style={{ borderTop:'1px solid var(--border)' }}>
                <p className="text-[11px] font-semibold uppercase tracking-wide mb-1.5" style={{ color:'var(--text-muted)' }}>
                  {video.segments.length} segment{video.segments.length!==1?'s':''}
                </p>
                {video.segments.map((seg,j)=><SegmentRow key={j} seg={seg} defaultOpen={j===0 || video.segments.length<=3} onPlay={playUrl?()=>seekTo(seg.start_time):null}/>)}
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  )
}

const timeStr = t => t!=null ? `${Math.floor(t/60)}:${String(Math.floor(t%60)).padStart(2,'0')}` : '?'

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

function SegmentRow({ seg, onPlay, defaultOpen }) {
  const [open, setOpen] = useState(defaultOpen)
  const parts = parseSegText(seg.text)
  const segScore = Number(seg.score)
  const scoreColor = segScore>0.7?'#34d399':segScore>0.5?'#fbbf24':'#94a3b8'
  return (
    <div className="rounded-xl px-3 py-2.5" style={{ background:'var(--bg-raised)', border:'1px solid var(--border)' }}>
      <div className="flex items-center justify-between" style={{ marginBottom: open ? 8 : 0 }}>
        <button onClick={onPlay} disabled={!onPlay}
          className="text-xs font-mono flex items-center gap-1.5 transition-colors"
          style={{ color: onPlay?'var(--accent-text)':'var(--text-muted)', cursor: onPlay?'pointer':'default' }}>
          {onPlay && <svg viewBox="0 0 10 10" fill="currentColor" className="w-2.5 h-2.5"><path d="M2 1.5l6 3.5-6 3.5z"/></svg>}
          {timeStr(seg.start_time)} → {timeStr(seg.end_time)}
        </button>
        <div className="flex items-center gap-1.5">
          {seg.sources?.map(s=><span key={s} className="badge text-[10px]">{s}</span>)}
          <span className="text-xs font-mono font-bold" style={{ color: scoreColor }}>{segScore.toFixed(4)}</span>
          {parts.length > 0 && (
            <button onClick={() => setOpen(o => !o)} aria-label={open ? 'Collapse segment details' : 'Expand segment details'} aria-expanded={open}
              className="w-6 h-6 flex items-center justify-center rounded-md transition-colors flex-shrink-0 focus-visible:outline focus-visible:outline-2"
              style={{ color: 'var(--text-secondary)', outlineColor: 'var(--accent)' }}>
              <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" className="w-3 h-3 transition-transform"
                style={{ transform: open ? 'rotate(180deg)' : 'none' }}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 6l4 4 4-4"/>
              </svg>
            </button>
          )}
        </div>
      </div>
      {open && parts.length > 0 && (
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

// Many PersonHits can share one identity_id (one chip per appearance made
// the row unreadable for anyone with several hits in one video) -- group
// into one chip per identity, showing count instead, expandable to see
// each individual appearance.
function groupPeopleByIdentity(people) {
  const map = new Map()
  for (const p of people || []) {
    if (!map.has(p.identity_id)) map.set(p.identity_id, [])
    map.get(p.identity_id).push(p)
  }
  return [...map.values()]
}

function PeopleRow({ people, onPlay }) {
  const [openGroup, setOpenGroup] = useState(null) // array of hits for one identity, or null
  const groups = groupPeopleByIdentity(people)
  if (!groups.length) return null
  return (
    <>
      <div className="flex items-center gap-2 flex-wrap mt-3 pt-3" style={{ borderTop:'1px solid var(--border)' }}>
        <span className="text-[11px] font-semibold uppercase tracking-wide" style={{ color:'var(--text-muted)' }}>People</span>
        {groups.map(hits => {
          const best = hits.reduce((a, b) => (b.confidence || 0) > (a.confidence || 0) ? b : a, hits[0])
          return (
            <button key={best.identity_id} onClick={() => onPlay && setOpenGroup(hits)} disabled={!onPlay}
              className="flex items-center gap-1.5 pl-1 pr-2.5 py-1 rounded-full transition-all duration-150 max-w-[220px] focus-visible:outline focus-visible:outline-2"
              style={{ background: 'var(--bg-raised)', border: '1px solid var(--border-md)', cursor: onPlay ? 'pointer' : 'default', outlineColor: 'var(--accent)' }}
              title={`${best.identity_name} · ${hits.length} appearance${hits.length !== 1 ? 's' : ''}`}>
              <span className="w-6 h-6 rounded-full overflow-hidden flex-shrink-0 inline-flex" style={{ background:'var(--bg-overlay)' }}>
                <FaceThumb url={best.keyframe_url} />
              </span>
              <span className="text-xs font-medium truncate min-w-0" style={{ color:'var(--text-primary)' }}>{best.identity_name}</span>
              {hits.length > 1
                ? <span className="text-[10px] font-mono flex-shrink-0 whitespace-nowrap" style={{ color:'var(--text-muted)' }}>×{hits.length}</span>
                : best.confidence != null && <span className="text-[10px] font-mono flex-shrink-0 whitespace-nowrap" style={{ color:'var(--text-muted)' }}>{Number(best.confidence).toFixed(2)}</span>}
            </button>
          )
        })}
      </div>

      {openGroup && (
        <PersonHitsModal hits={openGroup} onClose={() => setOpenGroup(null)}
          onSeek={t => { onPlay?.(t); setOpenGroup(null) }} />
      )}
    </>
  )
}

function PersonHitsModal({ hits, onClose, onSeek }) {
  // Portal to <body>: a `.card` ancestor's backdrop-filter becomes the
  // containing block for position:fixed descendants (CSS spec behavior),
  // which traps a plain fixed modal inside the card it was opened from
  // instead of the viewport -- this is what rendered the popup inline,
  // undimmed, clipped by the card below instead of centered over the page.
  return createPortal((
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" style={{ background: 'rgba(0,0,0,0.55)' }}
      onClick={onClose}>
      <div className="card w-full max-w-2xl" style={{ background: 'var(--bg-surface)' }} onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <p className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
            {hits[0].identity_name} · {hits.length} appearance{hits.length !== 1 ? 's' : ''}
          </p>
          <button onClick={onClose} aria-label="Close" className="text-sm w-7 h-7 flex items-center justify-center rounded-lg transition-colors focus-visible:outline focus-visible:outline-2"
            style={{ color: 'var(--text-muted)', outlineColor: 'var(--accent)' }}>✕</button>
        </div>

        {/* Horizontal, wrapping grid -- same pattern as the Identity page's trace popup */}
        <div className="flex flex-wrap gap-3 max-h-[60vh] overflow-y-auto pr-1">
          {hits.map((p, i) => (
            <button key={p.trace_id || i} onClick={() => onSeek(p.start_time)}
              aria-label={`Jump to ${timeStr(p.start_time)}`}
              // origin-bottom: a bottom-row tile's hover:scale grows upward
              // instead of downward, so it never pokes past this list's own
              // bottom edge. Growing centered (the default) pushed the last
              // row's scaled box a few px past that edge on hover, which
              // flipped overflow-y-auto's scrollbar on, ate ~15px of width,
              // and reflowed the whole wrap by a column.
              className="flex flex-col items-center gap-1.5 origin-bottom transition-transform hover:scale-105 rounded-xl focus-visible:outline focus-visible:outline-2"
              style={{ outlineColor: 'var(--accent)' }}>
              <div className="w-24 h-24 rounded-xl overflow-hidden p-1.5" style={{ background: 'var(--bg-raised)', border: '1px solid var(--border-md)' }}>
                <div className="w-full h-full rounded-lg overflow-hidden">
                  <FaceThumb url={p.keyframe_url} />
                </div>
              </div>
              <span className="text-[11px] font-mono" style={{ color: 'var(--text-secondary)' }}>
                {timeStr(p.start_time)} → {timeStr(p.end_time)}
              </span>
              {p.confidence != null && (
                <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>{Number(p.confidence).toFixed(2)}</span>
              )}
            </button>
          ))}
        </div>
      </div>
    </div>
  ), document.body)
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
