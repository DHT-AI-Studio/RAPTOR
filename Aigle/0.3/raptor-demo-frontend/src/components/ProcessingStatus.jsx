import React, { useState, useEffect } from 'react'
import { useAuth } from '../contexts/AuthContext.jsx'
import { apiGetAllProgress, apiGetProgress } from '../api/client.js'

const STATUS_COLORS = {
  pending:    'bg-yellow-900/40 text-yellow-300 border-yellow-700',
  processing: 'bg-blue-900/40 text-blue-300 border-blue-700',
  completed:  'bg-green-900/40 text-green-300 border-green-700',
  done:       'bg-green-900/40 text-green-300 border-green-700',
  failed:     'bg-red-900/40 text-red-300 border-red-700',
  error:      'bg-red-900/40 text-red-300 border-red-700',
}

const MEDIA_ICONS = { video: '🎬', audio: '🎵', document: '📄', image: '🖼️' }

export default function ProcessingStatus() {
  const { token } = useAuth()
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [filterType, setFilterType] = useState('all')
  const [selectedKey, setSelectedKey] = useState(null)
  const [detailData, setDetailData] = useState(null)

  const fetchAll = async () => {
    setLoading(true); setError('')
    try {
      const res = await apiGetAllProgress(token)
      setData(res)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchAll()
  }, [])

  useEffect(() => {
    if (!autoRefresh) return
    const id = setInterval(fetchAll, 5000)
    return () => clearInterval(id)
  }, [autoRefresh, token])

  const fetchDetail = async (mType, key) => {
    try {
      const res = await apiGetProgress(token, mType, key)
      setDetailData(res)
      setSelectedKey(key)
    } catch (err) {
      setDetailData({ error: err.message })
      setSelectedKey(key)
    }
  }

  const entries = data?.data ? Object.entries(data.data) : []
  const filtered = filterType === 'all'
    ? entries
    : entries.filter(([key]) => key.startsWith(filterType))

  // group by media type
  const grouped = {}
  filtered.forEach(([key, val]) => {
    const mtype = key.split('_orchestrator:')[0] || 'unknown'
    if (!grouped[mtype]) grouped[mtype] = []
    grouped[mtype].push([key, val])
  })

  return (
    <div>
      <div className="flex items-center justify-between mb-5">
        <div>
          <h1 className="text-xl font-bold text-white">Processing Status</h1>
          <p className="text-slate-400 text-sm mt-0.5">
            {data ? `${data.count} active task${data.count !== 1 ? 's' : ''}` : 'Track file processing progress'}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-xs text-slate-400 cursor-pointer">
            <input type="checkbox" className="rounded" checked={autoRefresh}
              onChange={e => setAutoRefresh(e.target.checked)} />
            Auto-refresh (5s)
          </label>
          <button onClick={fetchAll} disabled={loading} className="btn-secondary text-xs">
            {loading ? '⟳ Refreshing…' : '↺ Refresh'}
          </button>
        </div>
      </div>

      {/* Filter tabs */}
      <div className="flex gap-2 mb-5">
        {['all', 'video', 'audio', 'document', 'image'].map(t => (
          <button key={t} onClick={() => setFilterType(t)}
            className={`tab-btn ${filterType === t ? 'active' : ''}`}>
            {MEDIA_ICONS[t] || '📋'} {t}
          </button>
        ))}
      </div>

      {error && (
        <div className="bg-red-900/40 border border-red-700 rounded-lg px-4 py-3 text-red-300 text-sm mb-4">
          {error}
        </div>
      )}

      {data && entries.length === 0 && (
        <div className="text-center py-16 text-slate-500">
          <div className="text-4xl mb-3">📭</div>
          <p>No active processing tasks found.</p>
          <p className="text-sm mt-1">Upload a file to start processing.</p>
        </div>
      )}

      {/* Grouped tasks */}
      {Object.entries(grouped).map(([mtype, items]) => (
        <div key={mtype} className="mb-6">
          <div className="flex items-center gap-2 mb-3">
            <span>{MEDIA_ICONS[mtype] || '📎'}</span>
            <h3 className="text-sm font-semibold text-slate-300 capitalize">{mtype}</h3>
            <span className="text-xs text-slate-500">({items.length})</span>
          </div>

          <div className="space-y-2">
            {items.map(([key, val]) => {
              const correlationId = key.split('_orchestrator:')[1] || key
              const mType = key.split('_orchestrator:')[0]
              const status = val?.status || val?.processing_status || 'unknown'
              const filename = val?.filename || val?.primary_filename || val?.file_name || ''
              const progress = val?.progress != null ? val.progress : null
              const statusCls = STATUS_COLORS[status.toLowerCase()] || 'bg-slate-700/50 text-slate-400 border-slate-600'
              const isSelected = selectedKey === correlationId

              return (
                <div key={key}
                  className={`card cursor-pointer transition-all ${isSelected ? 'border-blue-600' : 'hover:border-slate-600'}`}
                  onClick={() => isSelected ? setSelectedKey(null) : fetchDetail(mType, correlationId)}>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3 min-w-0">
                      <span className={`text-xs px-2 py-0.5 rounded border ${statusCls} flex-shrink-0`}>
                        {status}
                      </span>
                      <div className="min-w-0">
                        {filename && <p className="text-sm text-slate-200 truncate">{filename}</p>}
                        <p className="text-xs text-slate-500 font-mono truncate">{correlationId}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-3 flex-shrink-0 ml-2">
                      {progress != null && (
                        <div className="flex items-center gap-2">
                          <div className="w-24 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                            <div className="h-full bg-blue-500 rounded-full transition-all"
                              style={{ width: `${Math.min(100, Number(progress) * 100)}%` }} />
                          </div>
                          <span className="text-xs text-slate-400">{Math.round(Number(progress) * 100)}%</span>
                        </div>
                      )}
                      <span className="text-xs text-slate-500">{isSelected ? '▾' : '▸'}</span>
                    </div>
                  </div>

                  {isSelected && detailData && (
                    <div className="mt-3 pt-3 border-t border-slate-700" onClick={e => e.stopPropagation()}>
                      <pre className="text-xs text-slate-300 overflow-auto max-h-64 whitespace-pre-wrap">
                        {JSON.stringify(detailData?.value ?? detailData, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      ))}
    </div>
  )
}
