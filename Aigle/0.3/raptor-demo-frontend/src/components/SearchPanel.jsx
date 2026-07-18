import React, { useState, useRef } from 'react'
import VideoPlayer from './VideoPlayer.jsx'
import GraphView from './GraphView.jsx'
import { useAuth } from '../contexts/AuthContext.jsx'
import {
  apiHybridSearch, apiBm25Search, apiVectorSearch,
  apiTkgSearch, apiGraphRagSearch,
} from '../api/client.js'

const SEARCH_TYPES = [
  { id: 'hybrid',   label: 'Hybrid',   desc: 'Vector + BM25 + Rerank' },
  { id: 'bm25',     label: 'BM25',     desc: 'Keyword search' },
  { id: 'vector',   label: 'Vector',   desc: 'Semantic search' },
  { id: 'tkg',      label: 'TKG',      desc: 'Temporal Knowledge Graph' },
  { id: 'graphrag', label: 'GraphRAG', desc: 'Graph-based RAG' },
]

const APIFN = {
  hybrid: apiHybridSearch, bm25: apiBm25Search, vector: apiVectorSearch,
  tkg: apiTkgSearch, graphrag: apiGraphRagSearch,
}

export default function SearchPanel() {
  const { token } = useAuth()
  const [type, setType] = useState('hybrid')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [results, setResults] = useState(null)
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(10)
  const [payloadSchema, setPayloadSchema] = useState('contextual')
  const [embeddingType, setEmbeddingType] = useState('')
  const [filterSource, setFilterSource] = useState('')
  const [filterType, setFilterType] = useState('')
  const [tkg, setTkg] = useState({ time_start:'', time_end:'', max_depth:2, limit:50, score_threshold:0.3 })
  const [grag, setGrag] = useState({ max_depth:2, limit:50, score_threshold:0.5, strategy:'hybrid' })

  const handleSearch = async () => {
    if (!query.trim()) return
    setError(''); setResults(null); setLoading(true)
    try {
      let payload = {}
      if (['hybrid','bm25','vector'].includes(type)) {
        payload = { query, top_k: topK, payload_schema: payloadSchema }
        if (embeddingType) payload.embedding_type = embeddingType
        if (filterSource) payload.source = filterSource
        if (filterType) payload.type = filterType
      } else if (type === 'tkg') {
        payload = { query, max_depth: tkg.max_depth, limit: tkg.limit, score_threshold: tkg.score_threshold }
        if (tkg.time_start) payload.time_start = tkg.time_start
        if (tkg.time_end) payload.time_end = tkg.time_end
      } else if (type === 'graphrag') {
        payload = { query, ...grag }
      }
      const data = await APIFN[type](token, payload)
      setResults(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex-1 flex flex-col min-h-0">
      {/* Page header */}
      <div className="page-header">
        <div className="flex items-end justify-between">
          <div>
            <h1 className="page-title">Search</h1>
            <p className="page-desc">Query your knowledge base using multiple retrieval strategies.</p>
          </div>
          {/* Search type pill group */}
          <div className="flex gap-1 p-1 rounded-xl"
            style={{ background: 'var(--bg-raised)', border: '1px solid var(--border-md)' }}>
            {SEARCH_TYPES.map(t => (
              <TypeTab key={t.id} t={t} active={type === t.id}
                onClick={() => { setType(t.id); setResults(null); setError('') }} />
            ))}
          </div>
        </div>
      </div>

      {/* Scrollable body */}
      <div className="flex-1 overflow-y-auto px-6 py-5">

        {/* Search input card */}
        <div className="card mb-5 space-y-4">
          <div>
            <label className="block text-xs font-medium mb-1.5" style={{ color: 'var(--text-secondary)' }}>Query</label>
            <div className="flex gap-2">
              <input className="input-field flex-1" placeholder="Enter search query…"
                value={query} onChange={e => setQuery(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleSearch()} />
              <button onClick={handleSearch} disabled={loading || !query.trim()} className="btn-primary px-5">
                {loading ? <Spinner /> : (
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

          {['hybrid','bm25','vector'].includes(type) && (
            <div className="grid grid-cols-3 gap-4 pt-1">
              <Field label="Top K">
                <input className="input-field" type="number" min={1} max={100} value={topK}
                  onChange={e => setTopK(Number(e.target.value))} />
              </Field>
              <Field label="Payload Schema">
                <select className="input-field" value={payloadSchema} onChange={e => setPayloadSchema(e.target.value)}>
                  <option value="contextual">contextual</option>
                  <option value="temporal">temporal</option>
                </select>
              </Field>
              <Field label="Embedding Type">
                <select className="input-field" value={embeddingType} onChange={e => setEmbeddingType(e.target.value)}>
                  <option value="">auto (不指定)</option>
                  <option value="text">text — 細粒度段落</option>
                  <option value="summary">summary — 文件摘要</option>
                </select>
              </Field>
              <Field label="Source filter" hint="pdf, mp4…">
                <input className="input-field" placeholder="Optional" value={filterSource}
                  onChange={e => setFilterSource(e.target.value)} />
              </Field>
              <Field label="Type filter" hint="videos, documents…">
                <input className="input-field" placeholder="Optional" value={filterType}
                  onChange={e => setFilterType(e.target.value)} />
              </Field>
            </div>
          )}

          {type === 'tkg' && (
            <div className="grid grid-cols-2 gap-4 pt-1">
              <Field label="Time Start" hint="ISO 8601"><input className="input-field" placeholder="2024-01-01T00:00:00" value={tkg.time_start} onChange={e => setTkg(t=>({...t,time_start:e.target.value}))}/></Field>
              <Field label="Time End"><input className="input-field" placeholder="2024-12-31T23:59:59" value={tkg.time_end} onChange={e => setTkg(t=>({...t,time_end:e.target.value}))}/></Field>
              <Field label="Max Depth (1–4)"><input className="input-field" type="number" min={1} max={4} value={tkg.max_depth} onChange={e => setTkg(t=>({...t,max_depth:Number(e.target.value)}))}/></Field>
              <Field label="Limit (1–200)"><input className="input-field" type="number" min={1} max={200} value={tkg.limit} onChange={e => setTkg(t=>({...t,limit:Number(e.target.value)}))}/></Field>
              <Field label="Score Threshold"><input className="input-field" type="number" min={0} max={10} step={0.1} value={tkg.score_threshold} onChange={e => setTkg(t=>({...t,score_threshold:Number(e.target.value)}))}/></Field>
            </div>
          )}

          {type === 'graphrag' && (
            <div className="grid grid-cols-2 gap-4 pt-1">
              <Field label="Strategy"><select className="input-field" value={grag.strategy} onChange={e=>setGrag(g=>({...g,strategy:e.target.value}))}><option value="hybrid">hybrid</option><option value="literal">literal</option><option value="semantic">semantic</option></select></Field>
              <Field label="Max Depth"><input className="input-field" type="number" min={1} max={4} value={grag.max_depth} onChange={e=>setGrag(g=>({...g,max_depth:Number(e.target.value)}))}/></Field>
              <Field label="Limit"><input className="input-field" type="number" min={1} max={200} value={grag.limit} onChange={e=>setGrag(g=>({...g,limit:Number(e.target.value)}))}/></Field>
              <Field label="Score Threshold"><input className="input-field" type="number" min={0} max={10} step={0.1} value={grag.score_threshold} onChange={e=>setGrag(g=>({...g,score_threshold:Number(e.target.value)}))}/></Field>
            </div>
          )}
        </div>

        {error && <div className="error-box mb-4">{error}</div>}
        {results && <SearchResults type={type} data={results} />}
      </div>
    </div>
  )
}

function TypeTab({ t, active, onClick }) {
  const [hov, setHov] = useState(false)
  return (
    <button key={t.id} onClick={onClick} title={t.desc}
      onMouseEnter={() => setHov(true)} onMouseLeave={() => setHov(false)}
      className="px-3 py-1.5 rounded-lg text-[12px] font-medium transition-all duration-150"
      style={active ? {
        background: 'linear-gradient(135deg,var(--accent),var(--accent-2))',
        color: '#fff',
        boxShadow: '0 2px 8px var(--accent-glow)',
      } : hov ? {
        background: 'var(--bg-overlay)',
        color: 'var(--text-primary)',
      } : { color: 'var(--text-muted)' }}>
      {t.label}
    </button>
  )
}

function Field({ label, hint, children }) {
  return (
    <div>
      <label className="block text-xs font-medium mb-1.5" style={{ color: 'var(--text-secondary)' }}>
        {label}{hint && <span className="ml-1 font-normal text-[11px]" style={{ color: 'var(--text-muted)' }}>({hint})</span>}
      </label>
      {children}
    </div>
  )
}

function SearchResults({ type, data }) {
  if (['hybrid','bm25','vector'].includes(type)) {
    const items = data?.results || []
    return (
      <div>
        <div className="flex items-center gap-3 mb-4">
          <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>{items.length} results</span>
          <div className="flex-1 divider" />
        </div>
        {items.length === 0 ? (
          <EmptyState text="No results found." />
        ) : items.map((item, i) => <ResultCard key={i} item={item} rank={i + 1} />)}
      </div>
    )
  }
  if (type === 'graphrag') return <GraphRagView data={data} />
  if (type === 'tkg')      return <TkgView data={data} />
  return (
    <div className="card">
      <pre className="text-xs overflow-auto max-h-96 whitespace-pre-wrap" style={{ color: 'var(--text-secondary)' }}>
        {JSON.stringify(data, null, 2)}
      </pre>
    </div>
  )
}

const fmtSec = t => t != null ? `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}` : null
const LABEL_CLASS = { contextual: 'badge-accent', asr: 'badge-green', ocr: 'badge-yellow', lvlm: 'badge-purple' }

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

const DownloadBtn = ({ url }) => url ? (
  <a href={url} target="_blank" rel="noreferrer" title="Download"
    className="text-xs px-2.5 py-1 rounded-lg transition-all duration-150 flex items-center gap-1"
    style={{ background: 'var(--bg-raised)', border: '1px solid var(--border-md)', color: 'var(--text-secondary)' }}>
    <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3 h-3">
      <path strokeLinecap="round" d="M7 2v7m0 0l-2.5-2.5M7 9l2.5-2.5M2 11.5h10"/>
    </svg>
  </a>
) : null

function ResultCard({ item, rank }) {
  const [open, setOpen] = useState(false)
  const [showPlayer, setShowPlayer] = useState(false)
  const [videoError, setVideoError] = useState(false)
  const videoRef = useRef(null)
  const payload = item.payload || {}
  const isVideo = payload.type === 'videos' || /\.(mp4|mov|avi|mkv|webm|ts|m2ts|mts)$/i.test(payload.filename || '')
  const videoUrl = payload.asset_url
  const score = Number(item.score)
  const scoreColor = score > 0.7 ? '#34d399' : score > 0.4 ? '#fbbf24' : '#94a3b8'
  const title = payload.filename || (item.id || '').slice(0, 40)
  const start = fmtSec(payload.start_time)
  const end = fmtSec(payload.end_time)

  const handlePlay = () => {
    const next = !showPlayer
    setShowPlayer(next)
    setVideoError(false)
    if (next) {
      setTimeout(() => {
        if (videoRef.current && payload.start_time != null) {
          videoRef.current.currentTime = payload.start_time
          videoRef.current.play().catch(() => {})
        }
      }, 100)
    }
  }

  return (
    <div className="card mb-3 relative overflow-hidden">
      <div className="absolute top-0 left-0 right-0 h-[2px]"
        style={{ background: 'linear-gradient(to right, var(--accent), transparent 60%)' }} />

      <div className="flex items-start justify-between">
        <div className="flex items-start gap-3">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center text-xs font-semibold flex-shrink-0"
            style={{ background: 'var(--bg-raised)', border: '1px solid var(--border-md)', color: 'var(--text-secondary)' }}>
            #{rank}
          </div>
          <div className="min-w-0">
            <p className="text-sm font-medium truncate" style={{ color: 'var(--text-primary)' }}>{title}</p>
            <div className="flex items-center gap-2 mt-0.5 flex-wrap">
              <span className="text-[11px] font-mono" style={{ color: 'var(--accent-text)' }}>{(item.id || '').slice(0, 16)}…</span>
              {payload.upload_time && <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>{new Date(payload.upload_time).toLocaleDateString('zh-TW')}</span>}
              {start && end && <span className="text-[11px] font-mono" style={{ color: 'var(--text-muted)' }}>{start} → {end}</span>}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <div className="flex items-center gap-2">
            <div className="w-16 h-2 rounded-full overflow-hidden" style={{ background: 'var(--bg-overlay)' }}>
              <div className="h-full rounded-full" style={{ width: `${Math.min(100, score * 100)}%`, background: scoreColor }} />
            </div>
            <span className="text-sm font-mono font-bold" style={{ color: scoreColor }}>{score.toFixed(4)}</span>
          </div>
          {isVideo && videoUrl && (
            <button onClick={handlePlay}
              className="text-xs px-2.5 py-1 rounded-lg transition-all duration-150"
              style={showPlayer ? {
                background: 'var(--accent-dim)', border: '1px solid var(--accent-glow)', color: 'var(--accent-text)',
              } : {
                background: 'var(--bg-raised)', border: '1px solid var(--border-md)', color: 'var(--text-secondary)',
              }}>
              {showPlayer ? '⏹ Hide' : '▶ Play'}
            </button>
          )}
          <DownloadBtn url={videoUrl} />
        </div>
      </div>

      {showPlayer && videoUrl && (
        <div className="mt-3">
          {!videoError ? (
            <div className="rounded-xl overflow-hidden bg-black">
              <VideoPlayer ref={videoRef} src={videoUrl} assetPath={payload.asset_path} className="w-full max-h-72"
                onError={() => setVideoError(true)} />
            </div>
          ) : (
            <div className="rounded-xl px-4 py-3 flex items-center justify-between gap-3"
              style={{ background: 'rgba(234,179,8,0.07)', border: '1px solid rgba(234,179,8,0.22)' }}>
              <p className="text-[11px]" style={{ color: '#fde047' }}>此格式瀏覽器無法播放，請下載後用播放器開啟。</p>
              <a href={videoUrl} target="_blank" rel="noreferrer" className="btn-secondary text-xs flex-shrink-0">↓ 下載</a>
            </div>
          )}
        </div>
      )}

      <div className="flex items-center justify-between mt-3">
        <div className="flex gap-1.5 flex-wrap">
          {payload.embedding_type && (
            <span className={payload.embedding_type === 'summary' ? 'badge badge-purple' : 'badge badge-accent'}>
              {payload.embedding_type}
            </span>
          )}
          {payload.type && <span className="badge text-[10px]">{payload.type}</span>}
          {payload.source && <span className="badge text-[10px]">{payload.source}</span>}
          {payload.chunk_index != null && <span className="badge text-[10px]">chunk {payload.chunk_index}</span>}
        </div>
        <button onClick={() => setOpen(v => !v)}
          className="text-[11px] flex items-center gap-1 transition-colors"
          style={{ color: 'var(--accent-text)' }}>
          detail <span>{open ? '▾' : '▸'}</span>
        </button>
      </div>

      {open && (
        <div className="mt-3 pt-3" style={{ borderTop: '1px solid var(--border)' }}>
          <PayloadDisplay payload={payload} />
        </div>
      )}
    </div>
  )
}

function PayloadDisplay({ payload }) {
  const textFields = ['summary', 'text', 'contextual_text', 'asr_text', 'ocr_text']
  const metaFields = ['type', 'source', 'chunk_index', 'start_time', 'end_time', 'upload_time', 'version_id']
  const shownText = textFields.filter(f => payload[f] != null)
  const shownMeta = metaFields.filter(f => payload[f] != null)

  if (shownText.length === 0 && shownMeta.length === 0)
    return <pre className="text-xs overflow-auto max-h-40 whitespace-pre-wrap" style={{ color: 'var(--text-secondary)' }}>{JSON.stringify(payload, null, 2)}</pre>

  return (
    <div className="space-y-2">
      {shownText.map(f => {
        const parts = parseSegText(String(payload[f]))
        return (
          <div key={f} className="rounded-xl p-3" style={{ background: 'var(--bg-raised)', border: '1px solid var(--border)' }}>
            <span className="inline-block text-[10px] font-bold uppercase tracking-widest px-1.5 py-0.5 rounded-md mb-2"
              style={{ background: 'var(--bg-overlay)', color: 'var(--accent-text)' }}>{f}</span>
            {parts.length === 1 && !parts[0].label ? (
              <p className="text-sm leading-relaxed whitespace-pre-wrap break-words" style={{ color: 'var(--text-primary)' }}>{parts[0].content}</p>
            ) : (
              <div className="space-y-2">
                {parts.map((p, i) => (
                  <div key={i}>
                    {p.label && (
                      <span className={`badge text-[10px] uppercase tracking-widest font-bold mb-1 ${LABEL_CLASS[p.label] || 'badge-accent'}`}>
                        {p.label}
                      </span>
                    )}
                    <p className="text-sm leading-relaxed break-words" style={{ color: 'var(--text-primary)' }}>{p.content}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )
      })}
      {shownMeta.length > 0 && (
        <div className="flex flex-wrap gap-1.5 pt-1">
          {shownMeta.map(f => (
            <span key={f} className="badge text-[11px]">
              <span style={{ color: 'var(--text-secondary)' }}>{f}:</span>{' '}{String(payload[f])}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Shared graph-view helpers ─────────────────────────────────────────────────

function GraphSection({ title, children }) {
  return (
    <div>
      <p className="text-xs font-bold uppercase tracking-widest mb-2 pb-1"
        style={{ color: 'var(--text-secondary)', borderBottom: '1px solid var(--border)' }}>
        {title}
      </p>
      {children}
    </div>
  )
}

function StatRow({ stats }) {
  return (
    <div className="flex gap-2 flex-wrap">
      {stats.filter(s => s.value != null && s.value !== 0).map(({ label, value }) => (
        <div key={label} className="rounded-xl px-3 py-1.5 flex items-center gap-1.5"
          style={{ background: 'var(--bg-raised)', border: '1px solid var(--border-md)' }}>
          <span className="text-sm font-bold" style={{ color: 'var(--accent-text)' }}>{value}</span>
          <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>{label}</span>
        </div>
      ))}
    </div>
  )
}

function GraphRagView({ data }) {
  const matchedEntities = data.matched_entities || []
  const nodes = data.nodes || []
  const rels = data.relationships || []

  return (
    <div className="card space-y-5">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>GraphRAG Results</h3>
        <div className="flex gap-1.5">
          {data.strategy && <span className="badge text-[10px]">{data.strategy}</span>}
        </div>
      </div>

      <StatRow stats={[
        { label: 'entities', value: matchedEntities.length },
        { label: 'nodes', value: nodes.length },
        { label: 'relationships', value: rels.length },
      ]} />

      {matchedEntities.length > 0 && (
        <GraphSection title="Matched Entities">
          <div className="flex flex-wrap gap-1.5">
            {matchedEntities.map((e, i) => (
              <span key={i} className="text-xs px-2.5 py-1 rounded-lg font-medium"
                style={{ background: 'var(--bg-overlay)', border: '1px solid var(--border-md)', color: 'var(--text-primary)' }}>
                {e.name || e.id}
                {e.type && <span className="ml-1.5 text-[10px]" style={{ color: 'var(--accent-text)' }}>[{e.type}]</span>}
              </span>
            ))}
          </div>
        </GraphSection>
      )}

      {(nodes.length > 0 || rels.length > 0) && (
        <GraphSection title="Graph Visualization">
          <GraphView nodes={nodes} edges={rels} />
        </GraphSection>
      )}

    </div>
  )
}

function TkgView({ data }) {
  const nodes = data.subgraph_nodes || []
  const edges = data.subgraph_edges || []
  const moments = data.moment_ids || []
  const facts = data.temporal_facts || []

  return (
    <div className="card space-y-5">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>TKG Results</h3>
        {data.entity_id && <span className="badge badge-accent text-[10px]">{data.entity_id}</span>}
      </div>

      <StatRow stats={[
        { label: 'nodes', value: nodes.length },
        { label: 'edges', value: edges.length },
        { label: 'moments', value: moments.length },
        { label: 'facts', value: facts.length },
      ]} />

      {(nodes.length > 0 || edges.length > 0) && (
        <GraphSection title="Graph Visualization">
          <GraphView nodes={nodes} edges={edges} />
        </GraphSection>
      )}

      {facts.length > 0 && (
        <GraphSection title="Temporal Facts">
          <div className="space-y-1">
            {facts.slice(0, 15).map((f, i) => (
              <div key={i} className="text-xs px-3 py-1.5 rounded-lg"
                style={{ background: 'var(--bg-raised)', color: 'var(--text-secondary)' }}>
                {typeof f === 'object' ? JSON.stringify(f) : String(f)}
              </div>
            ))}
            {facts.length > 15 && (
              <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>+{facts.length - 15} more</p>
            )}
          </div>
        </GraphSection>
      )}
    </div>
  )
}

function EmptyState({ text }) {
  return (
    <div className="text-center py-16">
      <div className="w-14 h-14 rounded-2xl flex items-center justify-center mx-auto mb-4"
        style={{ background: 'var(--bg-surface)', border: '1px solid var(--border)' }}>
        <svg viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1.5" className="w-6 h-6">
          <path strokeLinecap="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 15.803 7.5 7.5 0 0015.803 15.803z"/>
        </svg>
      </div>
      <p className="text-sm" style={{ color: 'var(--text-muted)' }}>{text}</p>
    </div>
  )
}

function Spinner() {
  return (
    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z"/>
    </svg>
  )
}
