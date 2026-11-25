import React, { useCallback, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useAuth } from '../contexts/AuthContext.jsx'
import {
  apiPendingPersons, apiListIdentities, apiBindGroup, apiCreatePendingPerson, apiUnbindTrace,
} from '../api/client.js'

// Shared basics neither portaled modal had: Esc closes, and focus returns
// to whatever had it before the modal opened -- a portal escapes the DOM
// tree the browser would otherwise track this through for free.
function useModalA11y(onClose) {
  const previouslyFocused = useRef(null)
  useEffect(() => {
    previouslyFocused.current = document.activeElement
    const onKey = e => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('keydown', onKey)
      previouslyFocused.current?.focus?.()
    }
  }, [onClose])
}

function Spinner({ className = 'h-4 w-4' }) {
  return (
    <svg className={`animate-spin ${className}`} viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z" />
    </svg>
  )
}

// Face-image presigned URLs come back with Content-Disposition: attachment
// (fine for a download link, but a browser refuses to render an
// "attachment" response inline) and Content-Type: application/octet-stream
// -- a plain <img src={url}> renders nothing for either reason. A JS
// fetch()+blob() ignores both; this is the same workaround Upload
// History's video thumbnails already use for the same backend behavior.
export function FaceThumb({ url }) {
  const [objUrl, setObjUrl] = useState(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    setObjUrl(null); setFailed(false)
    if (!url) return
    let cancelled = false
    let created = null
    fetch(url)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.blob() })
      .then(blob => {
        if (cancelled) return
        created = URL.createObjectURL(blob)
        setObjUrl(created)
      })
      .catch(() => { if (!cancelled) setFailed(true) })
    return () => {
      cancelled = true
      if (created) URL.revokeObjectURL(created)
    }
  }, [url])

  if (objUrl) return <img src={objUrl} alt="" className="w-full h-full object-cover" />
  if (!url || failed) {
    return <span className="w-full h-full flex items-center justify-center text-xl" style={{ color: 'var(--text-muted)' }}>?</span>
  }
  return <span className="w-full h-full flex items-center justify-center" style={{ color: 'var(--text-muted)' }}><Spinner className="h-4 w-4" /></span>
}

function RefreshButton({ onClick, loading }) {
  return (
    <button onClick={onClick} disabled={loading} className="btn-secondary text-xs gap-1.5 flex-shrink-0">
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"
        className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`}>
        <path strokeLinecap="round" d="M13.5 8A5.5 5.5 0 112.5 8a5.5 5.5 0 0111 0z" />
        <path strokeLinecap="round" d="M13.5 3.5V8H9" />
      </svg>
      Refresh
    </button>
  )
}

// ── Grouping results, shared by both the pending-persons presence check
// (Upload History) and this file's own detail screen ─────────────────────

// Flatten pending-persons' three buckets into one row per tile: an
// unassigned group is its own row, but every group an identity already
// holds in this video is merged into ONE row for that identity -- they're
// the same person now, so showing them as separate tiles after binding
// would contradict the "已綁定" status right next to them.
export function buildRows(pp) {
  const rows = []
  for (const g of pp.unassigned_groups || []) {
    rows.push({ key: `g:${g.group_id}`, groups: [g], traces: g.traces, representative: g.representative, identity: null })
  }
  for (const id of [...(pp.pending_identities || []), ...(pp.active_identities || [])]) {
    if (!id.groups?.length) continue
    rows.push({
      key: `i:${id.identity_id}`,
      groups: id.groups,
      traces: id.groups.flatMap(g => g.traces),
      representative: id.groups.map(g => g.representative).find(Boolean) || null,
      identity: id,
    })
  }
  rows.sort((a, b) => {
    const t = r => Math.min(...(r.traces.length ? r.traces.map(tr => tr.start_sec) : [0]))
    return t(a) - t(b)
  })
  return rows
}

// Whether a pending-persons response has any face group at all (bound or
// not) -- used by Upload History to decide whether a video's thumbnail
// should open this detail view.
export function hasAnyGroups(pp) {
  return (pp.unassigned_groups?.length || 0) > 0
    || (pp.pending_identities?.some(id => id.groups?.length) ?? false)
    || (pp.active_identities?.some(id => id.groups?.length) ?? false)
}

// ── Identity Grouping Results panel: the actual identity-binding UI,
// embedded directly in Upload History's File Detail screen (it owns the
// video player and back button; this panel just needs an onSeek callback
// to jump that player when a trace thumbnail is clicked). ────────────────

export function IdentityGroupingPanel({ commit, onSeek }) {
  const { token } = useAuth()
  const [rows, setRows] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState(() => new Set()) // row keys
  const [bindRows, setBindRows] = useState(null) // rows being bound, or null
  const [traceModal, setTraceModal] = useState(null) // { title, traces } for the popup, or null
  const [batchBusy, setBatchBusy] = useState(false)

  const refresh = useCallback(async () => {
    setLoading(true); setError('')
    try {
      const pp = await apiPendingPersons(token, commit.version_id)
      setRows(buildRows(pp))
      setSelected(new Set())
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }, [token, commit.version_id])

  useEffect(() => { refresh() }, [refresh])

  const toggleSelected = key => {
    setSelected(prev => {
      const next = new Set(prev)
      next.has(key) ? next.delete(key) : next.add(key)
      return next
    })
  }

  const selectedRows = rows.filter(r => selected.has(r.key))
  const canUnbind = selectedRows.length > 0 && selectedRows.every(r => r.identity)

  const handleBatchUnbind = async () => {
    setBatchBusy(true); setError('')
    try {
      for (const row of selectedRows) {
        for (const t of row.traces) {
          await apiUnbindTrace(token, row.identity.identity_id, commit.version_id, t.trace_id)
        }
      }
      await refresh()
    } catch (e) { setError(e.message) }
    finally { setBatchBusy(false) }
  }

  const boundCount = rows.filter(r => r.identity).length
  const allSelected = rows.length > 0 && selected.size === rows.length

  return (
    <div className="card">
      <div className="flex items-center justify-between gap-3 mb-3">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>Identity Grouping Results</h2>
          <span className="text-xs cursor-help" style={{ color: 'var(--text-muted)' }}
            title="Each group is a cluster of faces the pipeline believes is one person. Click a thumbnail to see every trace in the group.">ⓘ</span>
          {rows.length > 0 && (
            <span className="text-[11px] px-2 py-0.5 rounded-full" style={{ background: 'var(--bg-raised)', color: 'var(--text-muted)' }}>
              {boundCount}/{rows.length} bound
            </span>
          )}
        </div>
        <RefreshButton onClick={refresh} loading={loading} />
      </div>

      {error && <div className="error-box mb-3 text-[11px]">{error}</div>}

      {/* Batch Actions bar -- one Bind/Unbind pair for however many groups are checked,
          instead of a Bind+Unbind button pair repeated on every row. */}
      {rows.length > 0 && (
        <div className="flex items-center gap-3 mb-3 px-3 py-2 rounded-xl"
          style={{ background: 'var(--bg-overlay)', border: '1px solid var(--border)' }}>
          <label className="flex items-center gap-2 cursor-pointer flex-shrink-0">
            <input type="checkbox" checked={allSelected}
              onChange={() => setSelected(allSelected ? new Set() : new Set(rows.map(r => r.key)))} />
            <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
              {selected.size > 0 ? `${selected.size} selected` : 'Select all'}
            </span>
          </label>
          <div className="flex-1" />
          <button onClick={() => setBindRows(selectedRows)} disabled={selected.size === 0 || batchBusy}
            className="btn-primary text-[11px] px-3 py-1.5">Bind</button>
          <button onClick={handleBatchUnbind} disabled={!canUnbind || batchBusy}
            className="btn-secondary text-[11px] px-3 py-1.5">
            {batchBusy ? <Spinner className="h-3 w-3" /> : 'Unbind'}
          </button>
        </div>
      )}

      {loading && rows.length === 0 ? (
        <div className="flex flex-col items-center py-10">
          <Spinner className="h-6 w-6 mb-3" />
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Loading groups…</p>
        </div>
      ) : rows.length === 0 ? (
        <p className="text-xs py-10 text-center" style={{ color: 'var(--text-muted)' }}>No faces detected in this video.</p>
      ) : (
        <div className="grid gap-3" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(116px, 140px))' }}>
          {rows.map((row, i) => (
            <GroupTile key={row.key} index={i + 1} row={row}
              checked={selected.has(row.key)}
              onToggle={() => toggleSelected(row.key)}
              onExpand={() => setTraceModal({
                title: row.identity ? (row.identity.name || '(unnamed)') : `Group ${i + 1}`,
                traces: row.traces,
              })} />
          ))}
        </div>
      )}

      {bindRows && (
        <BindModal rows={bindRows} versionId={commit.version_id} token={token}
          onClose={() => setBindRows(null)}
          onBound={async () => { setBindRows(null); await refresh() }} />
      )}

      {traceModal && (
        <TraceThumbsModal title={traceModal.title} traces={traceModal.traces}
          onClose={() => setTraceModal(null)}
          onSeek={sec => { onSeek?.(sec); setTraceModal(null) }} />
      )}
    </div>
  )
}

// Clicking the thumbnail opens the trace popup (TraceThumbsModal); the
// checkbox in its corner is a separate hit target for batch select. A row
// whose identity holds more than one group in this video (bound together
// via a batch Bind) renders as a single tile -- see buildRows.
function GroupTile({ index, row, checked, onToggle, onExpand }) {
  const { groups, traces, identity, representative } = row
  const bound = !!identity
  const img = representative?.image_url
  const merged = groups.length > 1
  const label = bound ? (identity.name || '(unnamed)') : `Group ${index}`
  return (
    <div className="flex flex-col gap-1.5">
      {/* A <button> can't contain another focusable control (the checkbox), so the
          expand-trigger and the checkbox are two sibling buttons, not nested ones. */}
      <div className="relative aspect-square rounded-xl overflow-hidden"
        style={{
          background: 'var(--bg-overlay)',
          boxShadow: bound ? '0 0 0 2px var(--bg-raised), 0 0 0 3.5px rgba(16,185,129,0.55)' : '0 0 0 1px var(--border-md)',
        }}>
        <button onClick={onExpand} aria-label={`${label}, view traces`} disabled={traces.length === 0}
          className="absolute inset-0 w-full h-full transition-transform hover:scale-105 focus-visible:outline focus-visible:outline-2"
          style={{ outlineColor: 'var(--accent)' }}>
          <FaceThumb url={img} />
        </button>

        <button type="button" onClick={onToggle} role="checkbox" aria-checked={checked}
          aria-label={checked ? `Deselect ${label}` : `Select ${label}`}
          className="absolute top-1 left-1 w-6 h-6 rounded-md flex items-center justify-center transition-colors focus-visible:outline focus-visible:outline-2"
          style={{ background: checked ? 'var(--accent)' : 'rgba(0,0,0,0.45)', border: '1px solid rgba(255,255,255,0.3)', outlineColor: 'var(--accent)' }}>
          {checked && <svg viewBox="0 0 12 12" fill="none" stroke="white" strokeWidth="2" className="w-3 h-3"><path strokeLinecap="round" strokeLinejoin="round" d="M2.5 6l2.5 2.5 4.5-5" /></svg>}
        </button>

        <span className="absolute bottom-1.5 right-1.5 text-[10px] font-mono px-1 rounded pointer-events-none"
          style={{ background: 'rgba(0,0,0,0.5)', color: 'white' }}>{traces.length}</span>
      </div>

      <div className="px-0.5">
        <p className="text-[11px] font-medium truncate" style={{ color: 'var(--text-primary)' }}>
          {bound ? (identity.name || '(unnamed)') : `Group ${index}`}
        </p>
        {bound
          ? <p className="text-[10px] truncate" style={{ color: 'var(--text-muted)' }}>{merged ? `${groups.length} groups merged` : 'Bound'}</p>
          : <span className="badge text-[9px] mt-0.5 inline-block">Unbound</span>}
      </div>
    </div>
  )
}

// ── Trace thumbnails popup ───────────────────────────────────────────────────

const timeStr = t => t != null ? `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}` : '?'

function TraceThumbsModal({ title, traces, onClose, onSeek }) {
  useModalA11y(onClose)
  // Portal to <body>: a `.card` ancestor's backdrop-filter makes it the
  // containing block for position:fixed descendants (CSS spec behavior,
  // not a bug in this component), which traps a plain fixed-position modal
  // inside whatever card it was opened from instead of the viewport.
  return createPortal((
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" style={{ background: 'rgba(0,0,0,0.55)' }}
      onClick={onClose}>
      <div className="card w-full max-w-2xl" role="dialog" aria-modal="true" aria-label={`${title} appearances`}
        style={{ background: 'var(--bg-surface)' }} onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <p className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
            {title} · {traces.length} appearance{traces.length !== 1 ? 's' : ''}
          </p>
          <button onClick={onClose} aria-label="Close" autoFocus className="text-sm w-7 h-7 flex items-center justify-center rounded-lg transition-colors focus-visible:outline focus-visible:outline-2"
            style={{ color: 'var(--text-muted)', outlineColor: 'var(--accent)' }}>✕</button>
        </div>

        {/* Horizontal, wrapping grid -- never a vertical list, so many traces stay compact */}
        <div className="flex flex-wrap gap-3 max-h-[60vh] overflow-y-auto pr-1">
          {traces.map(t => (
            <button key={t.trace_id} onClick={() => onSeek(t.start_sec)}
              aria-label={`Jump to ${timeStr(t.start_sec)}`}
              // origin-bottom -- see PersonHitsModal's identical fix in
              // VideoSearchPanel.jsx for why a centered hover-scale on the
              // last row flips this list's scrollbar and reflows the wrap.
              className="flex flex-col items-center gap-1.5 origin-bottom transition-transform hover:scale-105 rounded-xl focus-visible:outline focus-visible:outline-2"
              style={{ outlineColor: 'var(--accent)' }}>
              <div className="w-28 h-28 rounded-xl overflow-hidden p-1.5" style={{ background: 'var(--bg-raised)', border: '1px solid var(--border-md)' }}>
                <div className="w-full h-full rounded-lg overflow-hidden">
                  <FaceThumb url={t.representative?.image_url} />
                </div>
              </div>
              <span className="text-[11px] font-mono" style={{ color: 'var(--text-secondary)' }}>
                {timeStr(t.start_sec)} → {timeStr(t.end_sec)}
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  ), document.body)
}

// ── Bind modal (batch: one or more selected groups) ─────────────────────────

function BindModal({ rows, versionId, token, onClose, onBound }) {
  useModalA11y(onClose)
  const [identities, setIdentities] = useState([])
  const [identitiesLoading, setIdentitiesLoading] = useState(true)
  const [mode, setMode] = useState('existing') // 'existing' | 'new'
  const [identityId, setIdentityId] = useState('')
  const [identityFilter, setIdentityFilter] = useState('')
  const [newName, setNewName] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    // No status filter: an identity created from the Bind flow below starts
    // out `pending` (only PATCH .../identity_status=active confirms it), so
    // filtering to status=active here would hide every identity this same
    // modal just created, making them look like they don't exist.
    apiListIdentities(token, { limit: 200 })
      .then(data => {
        const list = (data.items || data.identities || []).filter(id => id.identity_status !== 'merged' && id.identity_status !== 'deleted')
        setIdentities(list)
        // No auto-pick: binding the wrong person is a real data-integrity
        // risk, so "existing" starts with nothing selected and Confirm
        // stays disabled until the user explicitly chooses one.
        if (!list.length) setMode('new')
      })
      .catch(e => setError(e.message))
      .finally(() => setIdentitiesLoading(false))
  }, [token])

  const filteredIdentities = identityFilter.trim()
    ? identities.filter(id => {
        const q = identityFilter.trim().toLowerCase()
        return (id.name || '').toLowerCase().includes(q) || id.identity_id.toLowerCase().includes(q)
      })
    : identities

  // A selected row may already be several merged groups (see buildRows) --
  // flatten before binding, since bind/group works one group_id at a time.
  const allGroups = rows.flatMap(r => r.groups)

  const handleConfirm = async () => {
    setSubmitting(true); setError('')
    try {
      if (mode === 'existing') {
        if (!identityId) throw new Error('Pick an identity')
        for (const g of allGroups) {
          await apiBindGroup(token, identityId, versionId, g.group_id)
        }
      } else {
        if (!newName.trim()) throw new Error('Name is required')
        const [first, ...rest] = allGroups
        const created = await apiCreatePendingPerson(token, versionId, { name: newName.trim(), group_id: first.group_id })
        const newId = created.identity.identity_id
        for (const g of rest) {
          await apiBindGroup(token, newId, versionId, g.group_id)
        }
      }
      await onBound()
    } catch (e) { setError(e.message) }
    finally { setSubmitting(false) }
  }

  // Portal to <body>: see TraceThumbsModal above for why.
  return createPortal((
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" style={{ background: 'rgba(0,0,0,0.55)' }}
      onClick={onClose}>
      <div className="card w-full max-w-md" role="dialog" aria-modal="true" aria-label="Bind identity"
        style={{ background: 'var(--bg-surface)' }} onClick={e => e.stopPropagation()}>
        <div className="flex items-start justify-between mb-4">
          <p className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>Bind Identity</p>
          <button onClick={onClose} aria-label="Close" autoFocus className="text-sm w-7 h-7 flex items-center justify-center rounded-lg transition-colors focus-visible:outline focus-visible:outline-2"
            style={{ color: 'var(--text-muted)', outlineColor: 'var(--accent)' }}>✕</button>
        </div>

        <div className="flex flex-wrap gap-2 mb-2">
          {rows.map(row => (
            <div key={row.key} className="flex items-center gap-2 pl-1 pr-2.5 py-1 rounded-full"
              style={{ background: 'var(--bg-raised)', border: '1px solid var(--border-md)' }}>
              <div className="w-7 h-7 rounded-full overflow-hidden flex-shrink-0" style={{ background: 'var(--bg-overlay)' }}>
                <FaceThumb url={row.representative?.image_url} />
              </div>
              <span className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>{row.traces.length} person{row.traces.length !== 1 ? 's' : ''}</span>
            </div>
          ))}
        </div>
        <p className="text-[11px] mb-4" style={{ color: 'var(--text-muted)' }}>Selected Groups ({allGroups.length})</p>

        {error && <div className="error-box mb-3 text-[11px]">{error}</div>}

        <label className="flex items-center gap-2 mb-2 cursor-pointer">
          <input type="radio" checked={mode === 'existing'} onChange={() => setMode('existing')} disabled={identities.length === 0} />
          <span className="text-xs" style={{ color: 'var(--text-primary)' }}>Bind to existing identity</span>
        </label>
        {mode === 'existing' && (
          identitiesLoading ? (
            <div className="ml-6 mb-3 flex items-center gap-2 text-xs" style={{ color: 'var(--text-muted)' }}>
              <Spinner className="h-3.5 w-3.5" /> Loading…
            </div>
          ) : (
            <div className="ml-6 mb-3" style={{ width: 'calc(100% - 1.5rem)' }}>
              {identities.length > 5 && (
                <input className="input-field mb-1.5 text-xs" placeholder="Search identities…"
                  value={identityFilter} onChange={e => setIdentityFilter(e.target.value)}/>
              )}
              <select className="input-field" style={{ width: '100%' }}
                value={identityId} onChange={e => setIdentityId(e.target.value)}>
                <option value="" disabled>
                  {identities.length === 0 ? 'No identities yet' : 'Select an identity…'}
                </option>
                {filteredIdentities.map(id => (
                  <option key={id.identity_id} value={id.identity_id}>
                    {id.name || '(unnamed)'}{id.identity_status !== 'active' ? ` · ${id.identity_status}` : ''} ({id.identity_id.slice(0, 8)})
                  </option>
                ))}
              </select>
            </div>
          )
        )}

        <label className="flex items-center gap-2 mb-2 cursor-pointer">
          <input type="radio" checked={mode === 'new'} onChange={() => setMode('new')} />
          <span className="text-xs" style={{ color: 'var(--text-primary)' }}>Create new identity</span>
        </label>
        {mode === 'new' && (
          <input className="input-field mb-3 ml-6" style={{ width: 'calc(100% - 1.5rem)' }}
            placeholder="Name" value={newName} onChange={e => setNewName(e.target.value)} />
        )}

        <div className="flex justify-end gap-2 mt-4">
          <button onClick={onClose} className="btn-secondary text-xs">Cancel</button>
          <button onClick={handleConfirm} disabled={submitting || (mode === 'existing' && !identityId) || (mode === 'new' && !newName.trim())} className="btn-primary text-xs">
            {submitting ? <span className="flex items-center gap-1.5"><Spinner className="h-3 w-3" />Confirming…</span> : 'Confirm'}
          </button>
        </div>
      </div>
    </div>
  ), document.body)
}
