import React, { useState, useRef, useCallback } from 'react'
import { useAuth } from '../contexts/AuthContext.jsx'
import { apiUploadAnalysis, apiGetProgress } from '../api/client.js'

const PROCESSING_MODES = ['default','ocr','transcribe','summarize']
const ACCEPT_TYPES = '.pdf,.docx,.doc,.txt,.csv,.xlsx,.xls,.jpg,.jpeg,.png,.mp4,.mp3,.wav,.m4a'
const STEPS = ['queued','analyzing','complete']
const STEP_LABELS = { queued:'Queued', analyzing:'Analyzing', complete:'Complete', failed:'Failed' }

function getMType(filename) {
  const ext = (filename||'').split('.').pop().toLowerCase()
  if (['mp4','mov','avi','mkv','webm'].includes(ext)) return 'video'
  if (['mp3','wav','m4a','aac','flac'].includes(ext)) return 'audio'
  if (['jpg','jpeg','png','gif','bmp','webp'].includes(ext)) return 'image'
  return 'document'
}

function fileIcon(name) {
  const ext=(name||'').split('.').pop().toLowerCase()
  const m={pdf:'📄',docx:'📝',doc:'📝',txt:'📄',csv:'📊',xlsx:'📊',xls:'📊',jpg:'🖼️',jpeg:'🖼️',png:'🖼️',mp4:'🎬',mov:'🎬',mp3:'🎵',wav:'🎵',m4a:'🎵'}
  return m[ext]||'📎'
}

function fmtSize(bytes) {
  return bytes > 1048576 ? `${(bytes/1048576).toFixed(1)} MB` : `${(bytes/1024).toFixed(1)} KB`
}

let _id = 0
function makeEntry(file) {
  return { id: ++_id, file, status: 'pending', result: null, error: '' }
}

export default function UploadPanel() {
  const { token } = useAuth()
  const [queue, setQueue] = useState([])
  const [mode, setMode] = useState('default')
  const [archiveTtl, setArchiveTtl] = useState('')
  const [destroyTtl, setDestroyTtl] = useState('')
  const [uploading, setUploading] = useState(false)
  const [dragging, setDragging] = useState(false)
  const fileInputRef = useRef(null)

  const addFiles = useCallback(files => {
    const entries = Array.from(files).map(makeEntry)
    setQueue(q => [...q, ...entries])
  }, [])

  const removeEntry = id => setQueue(q => q.filter(e => e.id !== id))
  const clearAll = () => setQueue([])

  const handleDrop = e => {
    e.preventDefault(); setDragging(false)
    if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files)
  }

  const updateEntry = (id, patch) => setQueue(q => q.map(e => e.id === id ? { ...e, ...patch } : e))

  const handleUpload = async () => {
    const pending = queue.filter(e => e.status === 'pending')
    if (!pending.length) return
    setUploading(true)
    for (const entry of pending) {
      updateEntry(entry.id, { status: 'uploading' })
      try {
        const data = await apiUploadAnalysis(token, entry.file, {
          processingMode: mode,
          archiveTtl: archiveTtl || undefined,
          destroyTtl: destroyTtl || undefined,
        })
        updateEntry(entry.id, { status: 'done', result: { ...data, _filename: entry.file.name } })
      } catch (err) {
        updateEntry(entry.id, { status: 'error', error: err.message })
      }
    }
    setUploading(false)
  }

  const pendingCount = queue.filter(e => e.status === 'pending').length

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <div className="page-header">
        <h1 className="page-title">File Upload</h1>
        <p className="page-desc">Upload files for analysis and AI processing.</p>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-5">
        {/* Drop zone */}
        <div
          className="rounded-2xl p-8 text-center cursor-pointer mb-4 transition-all duration-200"
          style={dragging ? {
            background: 'var(--accent-dim)',
            border: '2px dashed var(--accent)',
            boxShadow: '0 0 0 4px var(--accent-dim)',
          } : {
            background: 'var(--bg-surface)',
            border: '2px dashed var(--border-md)',
          }}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onMouseEnter={e => { if (!dragging) e.currentTarget.style.borderColor = 'var(--border-lg)' }}
          onMouseLeave={e => { if (!dragging) e.currentTarget.style.borderColor = 'var(--border-md)' }}
        >
          <input ref={fileInputRef} type="file" className="hidden" accept={ACCEPT_TYPES} multiple
            onChange={e => { if (e.target.files.length) addFiles(e.target.files); e.target.value = '' }} />
          <div className="w-12 h-12 rounded-2xl flex items-center justify-center mx-auto mb-3 relative"
            style={{ background: 'var(--accent-dim)', border: '1px solid var(--accent-glow)' }}>
            <svg viewBox="0 0 24 24" fill="none" stroke="var(--accent-text)" strokeWidth="1.5" className="w-6 h-6">
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"/>
            </svg>
          </div>
          <p className="text-sm font-medium mb-1" style={{ color: 'var(--text-primary)' }}>Drop files here or click to browse</p>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>PDF, DOCX, CSV, JPG, MP4, MP3 and more · Multiple files supported</p>
        </div>

        {/* File queue */}
        {queue.length > 0 && (
          <div className="card mb-4">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-semibold" style={{ color: 'var(--text-primary)' }}>
                {queue.length} file{queue.length !== 1 ? 's' : ''} selected
              </span>
              <button onClick={clearAll} className="text-xs transition-colors"
                style={{ color: 'var(--text-muted)' }}
                onMouseEnter={e => e.currentTarget.style.color = '#f87171'}
                onMouseLeave={e => e.currentTarget.style.color = 'var(--text-muted)'}>
                Clear all
              </button>
            </div>
            <div className="space-y-2">
              {queue.map(entry => (
                <FileQueueRow key={entry.id} entry={entry}
                  onRemove={() => removeEntry(entry.id)} />
              ))}
            </div>
          </div>
        )}

        {/* Options */}
        <div className="card mb-5 space-y-4">
          <h3 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>Processing Options</h3>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-xs font-medium mb-1.5" style={{ color: 'var(--text-secondary)' }}>Processing Mode</label>
              <select className="input-field" value={mode} onChange={e => setMode(e.target.value)}>
                {PROCESSING_MODES.map(m => <option key={m} value={m}>{m}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium mb-1.5" style={{ color: 'var(--text-secondary)' }}>
                Archive TTL <span className="font-normal text-[11px]" style={{ color: 'var(--text-muted)' }}>(days or date)</span>
              </label>
              <input className="input-field" placeholder="e.g. 30 or 2025-01-01" value={archiveTtl} onChange={e => setArchiveTtl(e.target.value)}/>
            </div>
            <div>
              <label className="block text-xs font-medium mb-1.5" style={{ color: 'var(--text-secondary)' }}>Destroy TTL</label>
              <input className="input-field" placeholder="e.g. 365 or 2026-01-01" value={destroyTtl} onChange={e => setDestroyTtl(e.target.value)}/>
            </div>
          </div>

          <button onClick={handleUpload} disabled={!pendingCount || uploading} className="btn-primary w-full py-2.5">
            {uploading ? (
              <span className="flex items-center justify-center gap-2"><Spinner /> Uploading…</span>
            ) : (
              <span className="flex items-center justify-center gap-2">
                <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" className="w-4 h-4">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M10 13V4m0 0L7 7m3-3l3 3"/>
                  <path strokeLinecap="round" d="M3 14.5v.5a3 3 0 003 3h8a3 3 0 003-3v-.5"/>
                </svg>
                Upload & Process{pendingCount > 0 ? ` (${pendingCount})` : ''}
              </span>
            )}
          </button>
        </div>

        {/* Results — shown below options after upload */}
        {queue.some(e => e.status === 'done' || e.status === 'error') && (
          <div className="space-y-3">
            <h3 className="text-xs font-semibold px-1" style={{ color: 'var(--text-primary)' }}>Upload Results</h3>
            {queue.filter(e => e.status === 'done' || e.status === 'error').map(entry => (
              <div key={entry.id} className="card">
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-base flex-shrink-0">{fileIcon(entry.file.name)}</span>
                  <p className="text-xs font-medium truncate flex-1 min-w-0" style={{ color: 'var(--text-primary)' }}>
                    {entry.file.name}
                  </p>
                </div>
                {entry.status === 'error' && (
                  <p className="text-xs" style={{ color: '#f87171' }}>{entry.error}</p>
                )}
                {entry.status === 'done' && entry.result && (
                  <UploadResultView result={entry.result} token={token} />
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function FileQueueRow({ entry, onRemove }) {
  const { file, status, result } = entry
  const statusColor = {
    pending:  'var(--text-muted)',
    uploading:'var(--accent-text)',
    done:     '#6ee7b7',
    error:    '#f87171',
  }[status]

  const statusLabel = {
    pending: 'Pending',
    uploading: 'Uploading…',
    done: result?.processing_result?.status === 'skipped' ? 'Duplicate' : 'Queued',
    error: 'Error',
  }[status]

  return (
    <div className="flex items-center gap-3 px-3 py-2.5 rounded-xl"
      style={{ background: 'var(--bg-raised)', border: '1px solid var(--border)' }}>
      <span className="text-xl flex-shrink-0">{fileIcon(file.name)}</span>
      <div className="flex-1 min-w-0">
        <p className="text-xs font-medium truncate" style={{ color: 'var(--text-primary)' }}>{file.name}</p>
        <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>{fmtSize(file.size)} · {getMType(file.name)}</p>
      </div>
      <span className="text-[11px] font-medium flex-shrink-0" style={{ color: statusColor }}>{statusLabel}</span>
      {status === 'uploading' && <Spinner />}
      {status === 'pending' && (
        <button onClick={onRemove} className="flex-shrink-0 transition-colors"
          style={{ color: 'var(--text-muted)' }}
          onMouseEnter={e => e.currentTarget.style.color = '#f87171'}
          onMouseLeave={e => e.currentTarget.style.color = 'var(--text-muted)'}>
          <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5">
            <path strokeLinecap="round" d="M2 2l10 10M12 2L2 12"/>
          </svg>
        </button>
      )}
    </div>
  )
}

function UploadResultView({ result, token }) {
  const upload = result?.upload_result || {}
  const proc   = result?.processing_result || {}
  const filename = result?._filename || upload.primary_filename || ''
  const isSkipped = proc.status === 'skipped'
  const correlationId = proc.correlation_id || proc.kafka_result?.correlation_id
  const mType = getMType(filename)

  return (
    <div className="space-y-2 pl-1">
      {isSkipped && (
        <p className="text-xs px-3" style={{ color: '#fde047' }}>
          {proc.message || 'This file was previously uploaded and processed. Reusing existing index.'}
        </p>
      )}
      {!isSkipped && correlationId && (
        <ProcessingTracker token={token} correlationId={correlationId} mType={mType} />
      )}
      {!isSkipped && !correlationId && (
        <p className="text-[11px] px-3" style={{ color: 'var(--text-muted)' }}>
          No correlation_id returned. Cannot track progress.
        </p>
      )}
    </div>
  )
}

function ProcessingTracker({ token, correlationId, mType }) {
  const [status, setStatus] = useState(null)
  const [step, setStep] = useState('queued')
  const [done, setDone] = useState(false)
  const [pollError, setPollError] = useState('')
  const intervalRef = useRef(null)

  const poll = async () => {
    try {
      const res = await apiGetProgress(token, mType, correlationId)
      const val = res?.value ?? {}
      setStatus(val)
      const s = (typeof val==='string'?val:val?.step||val?.status||'').toLowerCase()
      if (s) setStep(s)
      if (s.includes('complete')||s.includes('done')||s.includes('failed')||s.includes('error')) {
        setDone(true); clearInterval(intervalRef.current)
      }
    } catch (err) {
      if (err.message.includes('404')||err.message.includes('not found')) setStep('queued')
      else { setPollError(err.message); clearInterval(intervalRef.current) }
    }
  }

  React.useEffect(() => {
    poll()
    intervalRef.current = setInterval(poll, 5000)
    return () => clearInterval(intervalRef.current)
  }, [correlationId])

  const isFailed   = step.includes('fail') || step.includes('error')
  const isComplete = step.includes('complete') || step.includes('done')
  const currentIdx = isFailed ? 0 : (!step||step==='queued') ? 0 : (isComplete ? 2 : 1)

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xs font-semibold" style={{ color: 'var(--text-primary)' }}>Processing Progress</h3>
        <div className="flex items-center gap-2">
          {!done && <span className="flex items-center gap-1 text-[11px]" style={{ color: 'var(--text-muted)' }}><Spinner/> polling…</span>}
          {isComplete && <span className="badge badge-green text-[10px]">Complete</span>}
          {isFailed && <span className="badge badge-red text-[10px]">Failed</span>}
        </div>
      </div>

      {!isFailed && (
        <div className="flex items-center mb-4">
          {STEPS.map((s, i) => {
            const isActive = i === currentIdx && !isComplete
            const isDone_  = i < currentIdx || isComplete
            const isLast   = i === STEPS.length - 1
            return (
              <React.Fragment key={s}>
                <div className="flex flex-col items-center flex-shrink-0">
                  <div className="w-8 h-8 rounded-full flex items-center justify-center font-semibold text-xs transition-all duration-300"
                    style={isDone_ ? {
                      background:'rgba(16,185,129,0.12)', border:'1.5px solid rgba(16,185,129,0.35)', color:'#6ee7b7',
                    } : isActive ? {
                      background:'var(--accent-dim)', border:'2px solid var(--accent)', color:'var(--accent-text)', boxShadow:'0 0 12px var(--accent-glow)',
                    } : {
                      background:'var(--bg-raised)', border:'1px solid var(--border-md)', color:'var(--text-muted)',
                    }}>
                    {isDone_ ? (
                      <svg viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="2.2" className="w-3 h-3">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M2 6l3 3 5-5"/>
                      </svg>
                    ) : i+1}
                  </div>
                  <span className="text-[10px] mt-1.5 font-medium whitespace-nowrap"
                    style={isDone_ ? {color:'#6ee7b7'} : isActive ? {color:'var(--accent-text)'} : {color:'var(--text-muted)'}}>
                    {STEP_LABELS[s]||s}
                  </span>
                </div>
                {!isLast && (
                  <div className="flex-1 h-px mx-2 mb-4 transition-colors duration-500"
                    style={{ background: i<currentIdx||isComplete ? 'rgba(16,185,129,0.4)' : 'var(--border-md)' }}/>
                )}
              </React.Fragment>
            )
          })}
        </div>
      )}

      {isFailed && <div className="error-box text-xs mb-3">Processing failed</div>}
      {pollError && <div className="text-xs mb-2" style={{ color: '#f87171' }}>Polling error: {pollError}</div>}

      <div className="flex flex-wrap gap-1.5">
        <Chip label="type" value={mType}/>
        <Chip label="id" value={correlationId.slice(0,16)+'…'}/>
      </div>

      {status && typeof status==='object' && Object.keys(status).length>0 && (
        <details className="mt-3">
          <summary className="text-[11px] cursor-pointer transition-colors"
            style={{ color: 'var(--text-muted)' }}
            onMouseEnter={e=>e.currentTarget.style.color='var(--text-secondary)'}
            onMouseLeave={e=>e.currentTarget.style.color='var(--text-muted)'}>
            Details ▸
          </summary>
          <pre className="text-[11px] mt-2 overflow-auto max-h-48 whitespace-pre-wrap" style={{ color: 'var(--text-secondary)' }}>
            {JSON.stringify(status,null,2)}
          </pre>
        </details>
      )}
    </div>
  )
}

function Chip({ label, value }) {
  return (
    <span className="badge text-[11px]">
      <span style={{ color: 'var(--text-muted)' }}>{label}:</span>{' '}{value}
    </span>
  )
}

function Spinner() {
  return (
    <svg className="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z"/>
    </svg>
  )
}
