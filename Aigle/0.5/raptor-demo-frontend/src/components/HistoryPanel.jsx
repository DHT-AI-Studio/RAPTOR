import React, { useState, useEffect, useCallback, useRef } from 'react'
import VideoPlayer from './VideoPlayer.jsx'
import { IdentityGroupingPanel, hasAnyGroups } from './VideoIdentityDetail.jsx'
import { useAuth } from '../contexts/AuthContext.jsx'
import { useTheme } from '../contexts/ThemeContext.jsx'
import {
  apiGetUserCommits, apiGetVideoUrl, apiArchiveAsset, apiDeleteAsset,
  apiGetThumbnailUrl, apiPendingPersons,
} from '../api/client.js'

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
  const [openFile, setOpenFile] = useState(null) // commit record, or null

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

  if (openFile) {
    return <div className="flex-1 flex flex-col min-h-0">
      <FileDetail commit={openFile} token={token} onBack={() => setOpenFile(null)} onRefresh={() => fetchCommits(page)}/>
    </div>
  }

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

            <div className="grid gap-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 260px))' }}>
              {commits.map(c=><FileCard key={c.version_id} c={c} token={token} onOpenFile={setOpenFile} onRefresh={()=>fetchCommits(page)}/>)}
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

// Pure display -- the "open detail"/"manage identity" click targets live
// as sibling buttons in FileCard, not here (a <button> can't contain
// another focusable control).
function VideoThumb({ url, failed }) {
  return (
    <div className="absolute inset-0 w-full h-full">
      {url
        ? <img src={url} alt="" className="w-full h-full object-cover"/>
        : failed
          ? <span className="w-full h-full flex items-center justify-center text-4xl opacity-70">🎬</span>
          : <span className="w-full h-full flex items-center justify-center">
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24" fill="none" style={{ color:'var(--text-muted)' }}>
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z"/>
              </svg>
            </span>}
    </div>
  )
}

// A solid dark scrim + bright text, independent of the app's own light/
// dark theme -- this sits on top of an arbitrary video frame, not the app
// background, so the theme-tinted translucent fill used elsewhere (too
// low-contrast here) would disappear into a bright frame.
function statusOverlayColor(status) {
  return { active: '#4ade80', archived: '#fbbf24', deleted: '#f87171' }[status] || '#cbd5e1'
}

// A grid card is thumbnail/icon + filename/date (click either to open
// FileDetail, which also carries every action) plus its own quick-action
// row for Download/Archive/Delete -- no need to open the detail screen
// just to archive something. Play stays detail-only: an inline player
// that expanded the card in place read as broken/janky.
function FileCard({ c, token, onOpenFile, onRefresh }) {
  const { themeKey } = useTheme()
  const isLight = themeKey === 'light'
  const [thumbUrl, setThumbUrl] = useState(null)
  const [thumbFailed, setThumbFailed] = useState(false)
  const [hasIdentity, setHasIdentity] = useState(false)
  const [actionLoading, setActionLoading] = useState(false)
  const [actionError, setActionError] = useState('')
  const [dlLoading, setDlLoading] = useState(false)
  const thumbObjectUrlRef = useRef(null)
  const isVideo = VIDEO_EXTS.has((c.primary_filename||'').split('.').pop().toLowerCase())

  // The badge only shows once we know this video actually has face data --
  // most videos won't, and there's no bulk "which of these have faces"
  // endpoint, so this checks lazily per card instead of a new API. Skipped
  // for non-active videos: nothing to bind on a failed/processing/archived/
  // unregistered asset.
  useEffect(() => {
    if (!isVideo || c.status !== 'active') return
    let cancelled = false
    apiGetThumbnailUrl(token, c.asset_path, c.version_id, 1)
      .then(url => {
        if (cancelled) { URL.revokeObjectURL(url); return }
        thumbObjectUrlRef.current = url
        setThumbUrl(url)
      })
      .catch(() => { if (!cancelled) setThumbFailed(true) })
    apiPendingPersons(token, c.version_id)
      .then(pp => { if (!cancelled) setHasIdentity(hasAnyGroups(pp)) })
      .catch(() => {})
    return () => {
      cancelled = true
      if (thumbObjectUrlRef.current) URL.revokeObjectURL(thumbObjectUrlRef.current)
    }
  }, [token, isVideo, c.status, c.asset_path, c.version_id])

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

  const handleDownload = async () => {
    setDlLoading(true)
    try {
      const url = await apiGetVideoUrl(token, c.asset_path, c.version_id)
      if (!url) throw new Error('No URL returned')
      window.open(url, '_blank')
    } catch(e) { alert(e.message) }
    finally { setDlLoading(false) }
  }

  return (
    <div className="rounded-2xl overflow-hidden flex flex-col transition-all duration-200 hover:-translate-y-1"
      style={{ background: 'var(--glass-bg)', border: '1px solid var(--glass-border)', boxShadow: 'var(--shadow-card)' }}>

      <div className="relative aspect-video flex items-center justify-center flex-shrink-0" style={{ background: 'var(--bg-raised)' }}>
        {isVideo && c.status === 'active'
          ? <VideoThumb url={thumbUrl} failed={thumbFailed}/>
          : <span className="text-5xl opacity-70">{fileIcon(c.primary_filename)}</span>}

        <button onClick={() => onOpenFile(c)} aria-label={`Open ${c.primary_filename}`} className="absolute inset-0 w-full h-full"/>

        {hasIdentity && (
          <span title="Has identified faces" aria-hidden="true" className="absolute bottom-1.5 right-1.5 w-5 h-5 rounded-md flex items-center justify-center pointer-events-none">
            <span className="absolute inset-0 rounded-md" style={{ background: 'rgba(16,185,129,0.9)' }}/>
            <svg viewBox="0 0 20 20" fill="none" stroke="white" strokeWidth="1.8" className="relative w-3 h-3">
              <circle cx="10" cy="7" r="3"/><path strokeLinecap="round" d="M4 16.5c0-3 2.7-5 6-5s6 2 6 5"/>
            </svg>
          </span>
        )}

        <span className="absolute top-2 right-2 text-[10px] px-2 py-0.5 rounded-lg font-semibold pointer-events-none"
          style={{ background: 'rgba(0,0,0,0.6)', color: statusOverlayColor(c.status) }}>
          {c.status}
        </span>
      </div>

      <div className="p-3 flex flex-col min-w-0">
        {/* Its own button (not the thumbnail's absolute overlay, which only
            covers the media area above) so clicking the filename/date also
            opens the detail view, not just the thumbnail. */}
        <button onClick={() => onOpenFile(c)} aria-label={`Open ${c.primary_filename}`} className="text-left w-full">
          <p className="text-xs font-medium truncate" title={c.asset_path} style={{ color: 'var(--text-primary)' }}>{c.primary_filename}</p>
          <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>{fmt(c.upload_date)}</p>
        </button>

        {actionError && <div className="error-box mt-2 text-[10px]">{actionError}</div>}

        <div className="flex items-center gap-1.5 mt-2.5 flex-wrap">
          <button onClick={handleDownload} disabled={dlLoading} title="Download"
            className="text-[11px] px-2.5 py-1 rounded-lg transition-all duration-150 flex items-center gap-1"
            style={{ background:'var(--bg-raised)', border:'1px solid var(--border-md)', color:'var(--text-secondary)' }}>
            {dlLoading
              ? <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z"/></svg>
              : <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5">
                  <path strokeLinecap="round" d="M7 2v7m0 0l-2.5-2.5M7 9l2.5-2.5M2 11.5h10"/>
                </svg>}
          </button>
          <div className="flex-1"/>
          {c.status === 'active' && (
            <button onClick={handleArchive} disabled={actionLoading} title="Archive"
              className="text-[11px] px-2.5 py-1 rounded-lg transition-all duration-150 flex items-center gap-1"
              style={{ background:'var(--bg-raised)', border:'1px solid var(--border-md)', color: isLight ? '#92400e' : '#fde047' }}>
              <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5">
                <rect x="2" y="3" width="12" height="3" rx="0.5"/><path d="M3 6.5v6a1 1 0 001 1h8a1 1 0 001-1v-6"/><path strokeLinecap="round" d="M6.5 9h3"/>
              </svg>
              Archive
            </button>
          )}
          {c.status === 'archived' && (
            <button onClick={handleDelete} disabled={actionLoading} title="Delete"
              className="text-[11px] px-2.5 py-1 rounded-lg transition-all duration-150 flex items-center gap-1"
              style={{ background:'var(--bg-raised)', border:'1px solid var(--border-md)', color: isLight ? '#b91c1c' : '#fca5a5' }}>
              <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 4.5h10M6 4.5V3a1 1 0 011-1h2a1 1 0 011 1v1.5m-6.5 0l.5 8a1 1 0 001 1h5a1 1 0 001-1l.5-8"/>
              </svg>
              Delete
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

// Opened from a grid card: back button + player (video files) or a big
// icon (others), every file action, and -- when this video has face
// groups -- the Identity Grouping Results panel embedded right here
// (rather than a separate screen to navigate to).
function FileDetail({ commit, token, onBack, onRefresh }) {
  const { themeKey } = useTheme()
  const isLight = themeKey === 'light'
  const [playerUrl, setPlayerUrl] = useState(null)
  const [playerError, setPlayerError] = useState('')
  const [videoPlayError, setVideoPlayError] = useState(false)
  const [actionLoading, setActionLoading] = useState(false)
  const [actionError, setActionError] = useState('')
  const [dlLoading, setDlLoading] = useState(false)
  const [hasIdentity, setHasIdentity] = useState(false)
  const videoRef = useRef(null)
  const isVideo = VIDEO_EXTS.has((commit.primary_filename||'').split('.').pop().toLowerCase())

  useEffect(() => {
    if (!isVideo) return
    let cancelled = false
    apiGetVideoUrl(token, commit.asset_path, commit.version_id)
      .then(url => { if (!cancelled) setPlayerUrl(url) })
      .catch(e => { if (!cancelled) setPlayerError(e.message) })
    return () => { cancelled = true }
  }, [token, isVideo, commit.asset_path, commit.version_id])

  // Same lazy check FileCard does for its badge -- the grid card's "has
  // identified faces" indicator is the only other place this is known, and
  // it doesn't carry over to this screen on its own.
  useEffect(() => {
    if (!isVideo || commit.status !== 'active') return
    let cancelled = false
    apiPendingPersons(token, commit.version_id)
      .then(pp => { if (!cancelled) setHasIdentity(hasAnyGroups(pp)) })
      .catch(() => {})
    return () => { cancelled = true }
  }, [token, isVideo, commit.status, commit.version_id])

  const seekTo = sec => {
    if (videoRef.current) { videoRef.current.currentTime = sec; videoRef.current.play?.() }
  }

  const handleDownload = async () => {
    setDlLoading(true)
    try {
      const url = playerUrl || await apiGetVideoUrl(token, commit.asset_path, commit.version_id)
      if (!url) throw new Error('No URL returned')
      window.open(url, '_blank')
    } catch(e) { alert(e.message) }
    finally { setDlLoading(false) }
  }

  const handleArchive = async () => {
    if (!window.confirm(`Archive "${commit.primary_filename}"?`)) return
    setActionLoading(true); setActionError('')
    try {
      await apiArchiveAsset(token, commit.asset_path, commit.version_id)
      onRefresh(); onBack()
    } catch(e) { setActionError(e.message); setActionLoading(false) }
  }

  const handleDelete = async () => {
    if (!window.confirm(`Delete "${commit.primary_filename}"? This cannot be undone.`)) return
    setActionLoading(true); setActionError('')
    try {
      await apiDeleteAsset(token, commit.asset_path, commit.version_id)
      onRefresh(); onBack()
    } catch(e) { setActionError(e.message); setActionLoading(false) }
  }

  return (
    <>
      <div className="page-header">
        <div className="flex items-center gap-3 min-w-0">
          <button onClick={onBack} aria-label="Back to Upload History"
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
            <h1 className="page-title truncate">{commit.primary_filename}</h1>
            <p className="page-desc font-mono truncate">{commit.version_id}</p>
          </div>
        </div>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto px-6 py-5">
        {renderFileDetailBody({
          commit, isVideo, isLight, playerUrl, playerError, videoPlayError, videoRef, setVideoPlayError,
          hasIdentity, seekTo, dlLoading, handleDownload, actionLoading, handleArchive, handleDelete, actionError,
        })}
      </div>
    </>
  )
}

// The status/download/archive-delete row + metadata chips are identical
// either way -- only their position (beside the video vs. above it) and
// the Identity panel's presence change, so they're built once here.
function renderFileDetailBody({
  commit, isVideo, isLight, playerUrl, playerError, videoPlayError, videoRef, setVideoPlayError,
  hasIdentity, seekTo, dlLoading, handleDownload, actionLoading, handleArchive, handleDelete, actionError,
}) {
  const player = (
    <div className="rounded-2xl overflow-hidden" style={{ border: '1px solid var(--border)', boxShadow: 'var(--shadow-card)' }}>
      {!isVideo ? (
        <div className="aspect-video flex items-center justify-center" style={{ background: 'var(--bg-raised)' }}>
          <span className="text-6xl opacity-70">{fileIcon(commit.primary_filename)}</span>
        </div>
      ) : playerError ? (
        <div className="error-box m-3 text-[11px]">{playerError}</div>
      ) : !playerUrl ? (
        <div className="aspect-video flex items-center justify-center" style={{ background: 'var(--bg-raised)' }}>
          <svg className="animate-spin h-6 w-6" viewBox="0 0 24 24" fill="none" style={{ color:'var(--text-muted)' }}>
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z"/>
          </svg>
        </div>
      ) : !videoPlayError ? (
        <VideoPlayer ref={videoRef} src={playerUrl} assetPath={commit.asset_path} className="w-full block" onError={() => setVideoPlayError(true)}/>
      ) : (
        // Keep the aspect-video sizing every other player state uses -- a
        // bare message box here shrinks to its own text height, leaving the
        // whole rest of this column's reserved space as dead empty area
        // next to the (often much taller) right column.
        <div className="aspect-video flex flex-col items-center justify-center gap-3 text-center px-8" style={{ background: 'var(--bg-raised)' }}>
          <svg viewBox="0 0 24 24" fill="none" stroke={isLight ? '#92400e' : '#fde047'} strokeWidth="1.5" className="w-8 h-8 opacity-80">
            <rect x="3" y="4.5" width="18" height="15" rx="2"/><path strokeLinecap="round" strokeLinejoin="round" d="M9.5 9l5 3-5 3V9z"/>
          </svg>
          <p className="text-[11px] max-w-xs" style={{ color: isLight ? '#92400e' : '#fde047' }}>This format can't play directly in the browser.</p>
          <a href={playerUrl} target="_blank" rel="noreferrer" className="btn-secondary text-xs gap-1.5">↓ Download and open in a video player</a>
        </div>
      )}
    </div>
  )

  // Tinted pill buttons (bg/border derived from the same hex as the icon,
  // ~12%/30% alpha via 8-digit hex) read as distinct actions at a glance --
  // a flat bg-raised square with only an icon doesn't, and forced every
  // action behind a tooltip-only label (flagged by icon-button accessibility
  // guidance as a hard anti-pattern, not just a style nit).
  const amber = '#f59e0b'
  const red = '#ef4444'
  const btnBase = "inline-flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg transition-all disabled:opacity-50"

  const actionsRow = (
    <div className="flex items-center gap-2 flex-wrap">
      <span className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-lg font-semibold" title="Status"
        style={{ background: `${statusOverlayColor(commit.status)}1f`, border: `1px solid ${statusOverlayColor(commit.status)}4d`, color: statusOverlayColor(commit.status) }}>
        <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: statusOverlayColor(commit.status) }}/>
        {commit.status}
      </span>
      <div className="flex-1"/>
      {isVideo && playerUrl && (
        <a href={playerUrl} target="_blank" rel="noreferrer" title="Download this file" className={btnBase}
          style={{ background:'var(--bg-raised)', border:'1px solid var(--border-md)', color:'var(--text-secondary)' }}>
          <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5">
            <path strokeLinecap="round" d="M7 2v7m0 0l-2.5-2.5M7 9l2.5-2.5M2 11.5h10"/>
          </svg>
          Download
        </a>
      )}
      {!isVideo && (
        <button onClick={handleDownload} disabled={dlLoading} title="Download this file" className={btnBase}
          style={{ background:'var(--bg-raised)', border:'1px solid var(--border-md)', color:'var(--text-secondary)' }}>
          {dlLoading
            ? <svg className="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z"/></svg>
            : <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5">
                <path strokeLinecap="round" d="M7 2v7m0 0l-2.5-2.5M7 9l2.5-2.5M2 11.5h10"/>
              </svg>}
          Download
        </button>
      )}
      {commit.status === 'active' && (
        <button onClick={handleArchive} disabled={actionLoading} title="Move this file to archive" className={btnBase}
          style={{ background:`${amber}1f`, border:`1px solid ${amber}4d`, color: isLight ? '#92400e' : '#fde047' }}>
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5">
            <rect x="2" y="3" width="12" height="3" rx="0.5"/><path d="M3 6.5v6a1 1 0 001 1h8a1 1 0 001-1v-6"/><path strokeLinecap="round" d="M6.5 9h3"/>
          </svg>
          Archive
        </button>
      )}
      {commit.status === 'archived' && (
        <button onClick={handleDelete} disabled={actionLoading} title="Permanently delete this file" className={btnBase}
          style={{ background:`${red}1f`, border:`1px solid ${red}4d`, color: isLight ? '#b91c1c' : '#fca5a5' }}>
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5">
            <path strokeLinecap="round" strokeLinejoin="round" d="M3 4.5h10M6 4.5V3a1 1 0 011-1h2a1 1 0 011 1v1.5m-6.5 0l.5 8a1 1 0 001 1h5a1 1 0 001-1l.5-8"/>
          </svg>
          Delete
        </button>
      )}
    </div>
  )

  // Short fixed-length facts (dates) stay as compact pills -- they never
  // need truncation. Path/version are unpredictable-length identifiers, not
  // quick-glance facts, so they get their own full-width row instead of
  // being squeezed into a 200px pill (that's what was clipping them).
  const metadataRow = (
    <div className="flex flex-col gap-1.5">
      <div className="flex flex-wrap gap-1.5">
        <Chip label="Uploaded" value={fmt(commit.upload_date)}/>
        {commit.archive_date && <Chip label="Archived" value={fmt(commit.archive_date)}/>}
        {commit.destroy_date && <Chip label="Deletes" value={fmt(commit.destroy_date)} warn/>}
      </div>
      <IdentifierRow label="Path" value={commit.asset_path}/>
      <IdentifierRow label="Version" value={commit.version_id}/>
    </div>
  )

  // With identity data to show: video + a right column (actions, metadata,
  // the actual Identity Grouping Results panel) side by side. Without one,
  // that right column would be mostly empty space next to a tall player --
  // instead the actions and metadata sit centered above the video, full
  // width, and that's it.
  if (hasIdentity) {
    // Wider cap than the no-identity column (below): the right column's
    // content -- actions, metadata, a full face-tile grid -- has real room
    // to use, and 920 left a lot of dead gutter on anything wider than a
    // laptop screen while that grid still only filled 3 columns.
    return (
      <div className="flex gap-5 items-start" style={{ maxWidth: 1320, margin: '0 auto' }}>
        {/* The right column (actions, metadata, the full identity grid) is
            routinely taller than the player, which has its own fixed
            intrinsic height -- without sticky, that reads as two unrelated
            boxes with dead space past the video's bottom edge. Pinning the
            player while the taller column scrolls past it is the
            established pattern for this (e.g. video sites' watch pages). */}
        <div className="flex-shrink-0 self-start sticky top-0" style={{ width: 420 }}>{player}</div>
        <div className="flex-1 min-w-0 flex flex-col gap-3 self-stretch">
          <InfoPanel actionsRow={actionsRow} metadataRow={metadataRow} actionError={actionError}/>
          <IdentityGroupingPanel commit={commit} onSeek={seekTo}/>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-3" style={{ maxWidth: 420, margin: '0 auto' }}>
      <InfoPanel actionsRow={actionsRow} metadataRow={metadataRow} actionError={actionError}/>
      {player}
    </div>
  )
}

// A raised, bordered card around actions+metadata -- without it this block
// read as loose text floating on the page background, not a distinct unit.
function InfoPanel({ actionsRow, metadataRow, actionError }) {
  return (
    <div className="flex flex-col gap-3 p-3.5 rounded-2xl" style={{ background: 'var(--bg-overlay)', border: '1px solid var(--border-lg)' }}>
      {actionsRow}
      {actionError && <div className="error-box text-[11px]">{actionError}</div>}
      {metadataRow}
    </div>
  )
}

// Full-width row for an unpredictable-length identifier (path/hash) -- flex
// truncation gives it all the room the row has instead of a fixed cap, and
// a copy button covers the value a sighted user can't read in full without
// relying on a hover-only title tooltip.
function IdentifierRow({ label, value }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try { await navigator.clipboard.writeText(value) } catch { return }
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }
  return (
    <div className="flex items-center gap-2 text-[11px] px-2.5 py-1.5 rounded-lg" style={{ background: 'var(--bg-raised)', border: '1px solid var(--border-md)' }}>
      <span className="flex-shrink-0 w-14" style={{ color: 'var(--text-muted)' }}>{label}</span>
      <span className="flex-1 min-w-0 truncate" style={{ fontFamily: 'monospace', color: 'var(--text-primary)', fontWeight: 500 }} title={value}>{value}</span>
      <button onClick={copy} title={copied ? 'Copied' : `Copy ${label.toLowerCase()}`} aria-label={`Copy ${label.toLowerCase()}`}
        className="flex-shrink-0 w-6 h-6 flex items-center justify-center rounded-md transition-colors"
        style={{ color: copied ? 'var(--accent-text)' : 'var(--text-muted)' }}>
        {copied
          ? <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" className="w-3.5 h-3.5"><path strokeLinecap="round" strokeLinejoin="round" d="M3 8.5l3 3 7-7"/></svg>
          : <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5"><rect x="5.5" y="5.5" width="8" height="8" rx="1.3"/><path d="M3 10.5V3.8a1 1 0 011-1h6.7"/></svg>}
      </button>
    </div>
  )
}

function Chip({ label, value, warn=false }) {
  return (
    <span className="inline-flex items-baseline gap-1 text-[11px] px-2 py-1 rounded-lg"
      style={{ border: '1px solid', borderColor: warn ? 'rgba(239,68,68,0.35)' : 'var(--border-md)', background: warn ? 'rgba(239,68,68,0.08)' : 'var(--bg-raised)' }}>
      <span className="flex-shrink-0" style={{ color:'var(--text-muted)' }}>{label}:</span>
      <span style={{ color: warn ? '#fca5a5' : 'var(--text-primary)', fontWeight: 500 }}>{value}</span>
    </span>
  )
}
