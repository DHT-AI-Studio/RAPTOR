import React, { useState, useEffect, useCallback, useRef } from 'react'
import VideoPlayer from './VideoPlayer.jsx'
import { useAuth } from '../contexts/AuthContext.jsx'
import { useTheme } from '../contexts/ThemeContext.jsx'
import { apiGetUserCommits, apiGetVideoUrl, apiGetKeyframeUrls, apiArchiveAsset, apiDeleteAsset } from '../api/client.js'

const VIDEO_EXTS = new Set(['mp4', 'mov', 'webm', 'mkv', 'ts', 'm2ts', 'mts'])

function fileIcon(name) {
  const ext=(name||'').split('.').pop().toLowerCase()
  const m={pdf:'📄',docx:'📝',doc:'📝',txt:'📄',csv:'📊',xlsx:'📊',xls:'📊',jpg:'🖼️',jpeg:'🖼️',png:'🖼️',mp4:'🎬',mov:'🎬',mp3:'🎵',wav:'🎵',m4a:'🎵'}
  return m[ext]||'📎'
}

function fmt(dateStr) {
  if (!dateStr) return '—'
  return new Date(dateStr).toLocaleString('zh-TW',{dateStyle:'short',timeStyle:'short'})
}

export default function HistoryPanel() {
  const { token } = useAuth()
  const [commits, setCommits]   = useState([])
  const [total, setTotal]       = useState(0)
  const [totalPages, setTotalPages] = useState(1)
  const [page, setPage]         = useState(1)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState('')
  const [keyword, setKeyword]   = useState('')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate]   = useState('')
  const [inputKw, setInputKw]   = useState('')

  const fetchCommits = useCallback(async (p=1) => {
    setLoading(true); setError('')
    try {
      const data = await apiGetUserCommits(token,{keyword,startDate,endDate,page:p})
      setCommits(data.commits||[])
      setTotal(data.total_count??0)
      setTotalPages(data.total_pages??1)
      setPage(p)
    } catch(err) { setError(err.message) }
    finally { setLoading(false) }
  },[token,keyword,startDate,endDate])

  useEffect(()=>{fetchCommits(1)},[fetchCommits])

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <div className="page-header">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="page-title">Upload History</h1>
            <p className="page-desc">All files you have uploaded.</p>
          </div>
          <button onClick={()=>fetchCommits(page)} disabled={loading}
            className="btn-secondary text-xs gap-1.5">
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"
              className={`w-3.5 h-3.5 ${loading?'animate-spin':''}`}>
              <path strokeLinecap="round" d="M13.5 8A5.5 5.5 0 112.5 8a5.5 5.5 0 0111 0z"/>
              <path strokeLinecap="round" d="M13.5 3.5V8H9"/>
            </svg>
            Refresh
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-5">
        {/* Filters */}
        <div className="card mb-5">
          <div className="flex gap-3 flex-wrap items-end">
            <div className="flex-1 min-w-40">
              <label className="block text-xs font-medium mb-1.5" style={{ color:'var(--text-secondary)' }}>Keyword</label>
              <input className="input-field" placeholder="Search by filename…" value={inputKw}
                onChange={e=>setInputKw(e.target.value)}
                onKeyDown={e=>e.key==='Enter'&&setKeyword(inputKw)}/>
            </div>
            <div>
              <label className="block text-xs font-medium mb-1.5" style={{ color:'var(--text-secondary)' }}>Start Date</label>
              <input className="input-field" type="date" value={startDate} onChange={e=>setStartDate(e.target.value)}/>
            </div>
            <div>
              <label className="block text-xs font-medium mb-1.5" style={{ color:'var(--text-secondary)' }}>End Date</label>
              <input className="input-field" type="date" value={endDate} onChange={e=>setEndDate(e.target.value)}/>
            </div>
            <button onClick={()=>setKeyword(inputKw)} className="btn-primary">Search</button>
            <button onClick={()=>{setInputKw('');setKeyword('');setStartDate('');setEndDate('')}} className="btn-secondary text-xs">Clear</button>
          </div>
        </div>

        {error && <div className="error-box mb-4">{error}</div>}

        {loading && commits.length===0 ? (
          <div className="flex flex-col items-center justify-center py-20">
            <svg className="animate-spin h-8 w-8 mb-4" viewBox="0 0 24 24" fill="none" style={{ color:'var(--accent)' }}>
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z"/>
            </svg>
            <p className="text-sm" style={{ color:'var(--text-muted)' }}>Loading…</p>
          </div>
        ) : commits.length===0 ? (
          <div className="text-center py-20">
            <div className="w-14 h-14 rounded-2xl flex items-center justify-center mx-auto mb-4"
              style={{ background:'var(--bg-surface)', border:'1px solid var(--border)' }}>
              <svg viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1.5" className="w-7 h-7">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 13.5l3 3m0 0l3-3m-3 3v-6m1.06-4.19l-2.12-2.12a1.5 1.5 0 00-1.061-.44H4.5A2.25 2.25 0 002.25 6v12a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18V9a2.25 2.25 0 00-2.25-2.25h-5.379a1.5 1.5 0 01-1.06-.44z"/>
              </svg>
            </div>
            <p className="text-sm" style={{ color:'var(--text-muted)' }}>No upload history found.</p>
          </div>
        ) : (
          <>
            <div className="flex items-center gap-3 mb-4">
              <p className="text-xs" style={{ color:'var(--text-muted)' }}>{total} records · Page {page} / {totalPages}</p>
              <div className="flex-1 divider"/>
            </div>

            <div className="space-y-3">
              {commits.map(c=><CommitCard key={c.version_id} c={c} token={token} onRefresh={()=>fetchCommits(page)}/>)}
            </div>

            {totalPages>1 && (
              <div className="flex items-center justify-center gap-2 mt-6">
                <button onClick={()=>fetchCommits(1)} disabled={page<=1||loading} className="btn-secondary text-xs px-2">
                  <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3 h-3"><path strokeLinecap="round" d="M10 2L5 7l5 5M6 2L1 7l5 5"/></svg>
                </button>
                <button onClick={()=>fetchCommits(page-1)} disabled={page<=1||loading} className="btn-secondary text-xs px-3 gap-1">
                  <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3 h-3"><path strokeLinecap="round" d="M9 2L4 7l5 5"/></svg>Prev
                </button>
                <div className="flex items-center gap-1.5">
                  <input
                    type="text" inputMode="numeric" pattern="[0-9]*"
                    defaultValue={page} key={page}
                    className="input-field text-xs text-center py-1"
                    style={{ width: String(totalPages).length <= 2 ? 52 : 64 }}
                    onKeyDown={e => {
                      if (e.key === 'Enter') {
                        const v = parseInt(e.target.value)
                        if (v >= 1 && v <= totalPages) fetchCommits(v)
                      }
                    }}
                    onBlur={e => {
                      const v = parseInt(e.target.value)
                      if (v >= 1 && v <= totalPages && v !== page) fetchCommits(v)
                    }}
                  />
                  <span className="text-xs" style={{ color:'var(--text-muted)' }}>/ {totalPages}</span>
                </div>
                <button onClick={()=>fetchCommits(page+1)} disabled={page>=totalPages||loading} className="btn-secondary text-xs px-3 gap-1">
                  Next<svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3 h-3"><path strokeLinecap="round" d="M5 2l5 5-5 5"/></svg>
                </button>
                <button onClick={()=>fetchCommits(totalPages)} disabled={page>=totalPages||loading} className="btn-secondary text-xs px-2">
                  <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3 h-3"><path strokeLinecap="round" d="M4 2l5 5-5 5M8 2l5 5-5 5"/></svg>
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

const IMG_EXTS = /\.(jpg|jpeg|png|gif|webp)$/i

function CommitCard({ c, token, onRefresh }) {
  const { themeKey } = useTheme()
  const isLight = themeKey === 'light'
  const [expanded, setExpanded] = useState(false)
  const [showPlayer, setShowPlayer] = useState(false)
  const [playerUrl, setPlayerUrl] = useState(null)
  const [playerLoading, setPlayerLoading] = useState(false)
  const [playerError, setPlayerError] = useState('')
  const [videoPlayError, setVideoPlayError] = useState(false)
  const [actionLoading, setActionLoading] = useState(false)
  const [actionError, setActionError] = useState('')
  const [dlLoading, setDlLoading] = useState(false)
  const [showKeyframes, setShowKeyframes] = useState(false)
  const [keyframes, setKeyframes] = useState(null)
  const [keyframeLoading, setKeyframeLoading] = useState(false)
  const [keyframeError, setKeyframeError] = useState('')
  const videoRef = useRef(null)
  const assocCount = c.associated_filenames?.length??0
  const isVideo = VIDEO_EXTS.has((c.primary_filename||'').split('.').pop().toLowerCase())
  const hasKeyframeFiles = c.associated_filenames?.some(([name]) => IMG_EXTS.test(name)) ?? false

  const handleArchive = async () => {
    if (!window.confirm(`Archive "${c.primary_filename}"?`)) return
    setActionLoading(true); setActionError('')
    try {
      await apiArchiveAsset(token, c.asset_path, c.version_id)
      onRefresh()
    } catch(e) { setActionError(e.message) }
    finally { setActionLoading(false) }
  }

  const handleDelete = async () => {
    if (!window.confirm(`Delete "${c.primary_filename}"? This cannot be undone.`)) return
    setActionLoading(true); setActionError('')
    try {
      await apiDeleteAsset(token, c.asset_path, c.version_id)
      onRefresh()
    } catch(e) { setActionError(e.message) }
    finally { setActionLoading(false) }
  }

  const handlePlay = async () => {
    if (showPlayer) { setShowPlayer(false); return }
    setPlayerLoading(true); setPlayerError(''); setVideoPlayError(false)
    try {
      const url = await apiGetVideoUrl(token, c.asset_path, c.version_id)
      if (!url) throw new Error('No URL returned')
      setPlayerUrl(url)
      setShowPlayer(true)
    } catch(e) { setPlayerError(e.message) }
    finally { setPlayerLoading(false) }
  }

  const handleDownload = async () => {
    setDlLoading(true)
    try {
      const url = await apiGetVideoUrl(token, c.asset_path, c.version_id)
      if (!url) throw new Error('No URL returned')
      window.open(url, '_blank')
    } catch(e) { alert(e.message) }
    finally { setDlLoading(false) }
  }

  const handleKeyframes = async () => {
    if (showKeyframes) { setShowKeyframes(false); return }
    setShowKeyframes(true)
    if (keyframes !== null) return
    setKeyframeLoading(true); setKeyframeError('')
    try {
      const frames = await apiGetKeyframeUrls(token, c.asset_path, c.version_id)
      setKeyframes(frames)
    } catch(e) { setKeyframeError(e.message) }
    finally { setKeyframeLoading(false) }
  }

  const statusStyle = isLight ? {
    active:   { bg:'rgba(5,150,105,0.10)',   border:'rgba(5,150,105,0.30)',   color:'#065f46' },
    archived: { bg:'rgba(180,83,9,0.08)',    border:'rgba(180,83,9,0.28)',    color:'#92400e' },
    deleted:  { bg:'rgba(220,38,38,0.08)',   border:'rgba(220,38,38,0.25)',   color:'#b91c1c' },
  }[c.status] || { bg:'var(--bg-raised)', border:'var(--border-md)', color:'var(--text-muted)' }
  : {
    active:   { bg:'rgba(16,185,129,0.08)',  border:'rgba(16,185,129,0.25)',  color:'#6ee7b7' },
    archived: { bg:'rgba(234,179,8,0.08)',   border:'rgba(234,179,8,0.22)',   color:'#fde047' },
    deleted:  { bg:'rgba(239,68,68,0.08)',   border:'rgba(239,68,68,0.22)',   color:'#fca5a5' },
  }[c.status] || { bg:'var(--bg-raised)', border:'var(--border-md)', color:'var(--text-muted)' }

  return (
    <div className="card transition-all duration-150">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-3 min-w-0 flex-1">
          <span className="text-2xl flex-shrink-0 mt-0.5">{fileIcon(c.primary_filename)}</span>
          <div className="min-w-0">
            <p className="text-sm font-medium truncate" style={{ color:'var(--text-primary)' }}>{c.primary_filename}</p>
            <p className="text-[11px] font-mono mt-0.5 truncate" style={{ color:'var(--text-muted)' }}>{c.version_id}</p>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className="text-[11px] px-2 py-0.5 rounded-lg font-medium"
            style={{ background:statusStyle.bg, border:`1px solid ${statusStyle.border}`, color:statusStyle.color }}>
            {c.status}
          </span>
          {isVideo && (
            <button onClick={handlePlay} disabled={playerLoading}
              className="text-xs px-2.5 py-1 rounded-lg transition-all duration-150 flex items-center gap-1"
              style={showPlayer ? {
                background:'var(--accent-dim)', border:'1px solid var(--accent-glow)', color:'var(--accent-text)',
              } : {
                background:'var(--bg-raised)', border:'1px solid var(--border-md)', color:'var(--text-secondary)',
              }}>
              {playerLoading
                ? <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z"/></svg>
                : showPlayer ? '⏹' : '▶'}
              {playerLoading ? 'Loading…' : showPlayer ? 'Hide' : 'Play'}
            </button>
          )}
          <button onClick={handleDownload} disabled={dlLoading}
            title="Download"
            className="text-xs px-2.5 py-1 rounded-lg transition-all duration-150 flex items-center gap-1"
            style={{ background:'var(--bg-raised)', border:'1px solid var(--border-md)', color:'var(--text-secondary)' }}>
            {dlLoading
              ? <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z"/></svg>
              : <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5">
                  <path strokeLinecap="round" d="M7 2v7m0 0l-2.5-2.5M7 9l2.5-2.5M2 11.5h10"/>
                </svg>
            }
          </button>
          {false && isVideo && hasKeyframeFiles && (
            <button onClick={handleKeyframes} disabled={keyframeLoading}
              className="text-xs px-2.5 py-1 rounded-lg transition-all duration-150 flex items-center gap-1"
              style={showKeyframes ? {
                background:'var(--accent-dim)', border:'1px solid var(--accent-glow)', color:'var(--accent-text)',
              } : {
                background:'var(--bg-raised)', border:'1px solid var(--border-md)', color:'var(--text-secondary)',
              }}>
              {keyframeLoading
                ? <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z"/></svg>
                : '🖼'}
              {keyframeLoading ? 'Loading…' : showKeyframes ? 'Hide' : 'Frames'}
            </button>
          )}
          {c.status === 'active' && (
            <button onClick={handleArchive} disabled={actionLoading}
              className="text-[11px] px-2 py-0.5 rounded-lg transition-all"
              style={isLight
                ? { background:'rgba(180,83,9,0.08)', border:'1px solid rgba(180,83,9,0.28)', color:'#92400e' }
                : { background:'rgba(234,179,8,0.08)', border:'1px solid rgba(234,179,8,0.22)', color:'#fde047' }}>
              Archive
            </button>
          )}
          {c.status === 'archived' && (
            <button onClick={handleDelete} disabled={actionLoading}
              className="text-[11px] px-2 py-0.5 rounded-lg transition-all"
              style={isLight
                ? { background:'rgba(220,38,38,0.08)', border:'1px solid rgba(220,38,38,0.25)', color:'#b91c1c' }
                : { background:'rgba(239,68,68,0.08)', border:'1px solid rgba(239,68,68,0.22)', color:'#fca5a5' }}>
              Delete
            </button>
          )}
          {false && assocCount>0 && (
            <button onClick={()=>setExpanded(v=>!v)}
              className="text-[11px] flex items-center gap-1 transition-colors"
              style={{ color:'var(--text-muted)' }}
              onMouseEnter={e=>e.currentTarget.style.color='var(--text-secondary)'}
              onMouseLeave={e=>e.currentTarget.style.color='var(--text-muted)'}>
              +{assocCount} file{assocCount>1?'s':''}{' '}{expanded?'▾':'▸'}
            </button>
          )}
        </div>
      </div>

      {playerError && <div className="error-box mt-2 text-[11px]">{playerError}</div>}
      {actionError && <div className="error-box mt-2 text-[11px]">{actionError}</div>}

      {showPlayer && playerUrl && (
        <div className="mt-3">
          {!videoPlayError ? (
            <div className="rounded-xl overflow-hidden bg-black">
              <VideoPlayer ref={videoRef} src={playerUrl} assetPath={c.asset_path} className="w-full max-h-72"
                onError={()=>setVideoPlayError(true)}/>
            </div>
          ) : (
            <div className="rounded-xl px-4 py-3 flex items-center justify-between gap-3"
              style={isLight
                ? { background:'rgba(180,83,9,0.07)', border:'1px solid rgba(180,83,9,0.25)' }
                : { background:'rgba(234,179,8,0.07)', border:'1px solid rgba(234,179,8,0.22)' }}>
              <p className="text-[11px]" style={{ color: isLight ? '#92400e' : '#fde047' }}>
                此格式（{c.asset_path.split('/')[1]||'未知'}）瀏覽器無法直接播放，請下載後用播放器開啟。
              </p>
              <a href={playerUrl} target="_blank" rel="noreferrer"
                className="btn-secondary text-xs flex-shrink-0" style={{ whiteSpace:'nowrap' }}>
                ↓ 下載
              </a>
            </div>
          )}
        </div>
      )}

      {false && showKeyframes && (
        <div className="mt-3">
          {keyframeLoading && (
            <div className="flex items-center gap-2 text-xs py-2" style={{ color:'var(--text-muted)' }}>
              <svg className="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z"/>
              </svg>
              Loading keyframes…
            </div>
          )}
          {keyframeError && <div className="error-box text-[11px] mt-1">{keyframeError}</div>}
          {keyframes !== null && !keyframeLoading && (
            keyframes.length === 0 ? (
              <p className="text-[11px]" style={{ color:'var(--text-muted)' }}>No keyframes found.</p>
            ) : (
              <div className="overflow-x-auto rounded-xl" style={{ background:'var(--bg-raised)', border:'1px solid var(--border)' }}>
                <div className="flex gap-2 p-2" style={{ width:'max-content' }}>
                  {keyframes.map((kf, i) => (
                    <a key={i} href={kf.url} target="_blank" rel="noreferrer" title={kf.filename}
                      className="flex-shrink-0 rounded-lg overflow-hidden transition-opacity hover:opacity-80"
                      style={{ border:'1px solid var(--border-md)' }}>
                      <img src={kf.url} alt={kf.filename}
                        className="object-cover block"
                        style={{ width:140, height:80 }}/>
                    </a>
                  ))}
                </div>
                <p className="text-[10px] px-2 pb-1.5" style={{ color:'var(--text-muted)' }}>
                  {keyframes.length} frame{keyframes.length !== 1 ? 's' : ''} · click to open full size
                </p>
              </div>
            )
          )}
        </div>
      )}

      <div className="flex flex-wrap gap-1.5 mt-3">
        <Chip label="Uploaded" value={fmt(c.upload_date)}/>
        {c.archive_date&&<Chip label="Archive" value={fmt(c.archive_date)}/>}
        {c.destroy_date&&<Chip label="Delete" value={fmt(c.destroy_date)} warn/>}
        <Chip label="path" value={c.asset_path} mono/>
      </div>

      {false && expanded && assocCount>0 && (
        <div className="mt-3 pt-3 space-y-1.5" style={{ borderTop:'1px solid var(--border)' }}>
          {c.associated_filenames.map(([name,vid],i)=>(
            <div key={i} className="flex items-center gap-2 text-[11px]" style={{ color:'var(--text-secondary)' }}>
              <span>{fileIcon(name)}</span>
              <span className="truncate flex-1">{name}</span>
              <span className="font-mono text-[10px]" style={{ color:'var(--text-muted)' }}>{String(vid).slice(0,12)}…</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function Chip({ label, value, mono=false, warn=false }) {
  return (
    <span className="text-[11px] px-2 py-0.5 rounded-lg"
      style={warn?{ background:'rgba(239,68,68,0.08)', color:'#fca5a5' }:{ background:'var(--bg-raised)', border:'1px solid var(--border)', color:'var(--text-secondary)' }}>
      <span style={{ color:'var(--text-muted)' }}>{label}: </span>
      <span style={mono?{fontFamily:'monospace'}:{}}>{value}</span>
    </span>
  )
}
