import { useEffect, useRef, useState } from 'react'
import { Graph, NodeEvent } from '@antv/g6'

const PALETTE = [
  '#6366f1','#3b82f6','#10b981','#f59e0b','#ef4444',
  '#8b5cf6','#ec4899','#14b8a6','#f97316','#84cc16',
]

function hashColor(str) {
  let h = 0
  for (const c of (str || '')) h = (h * 31 + c.charCodeAt(0)) & 0xffff
  return PALETTE[h % PALETTE.length]
}

function truncate(s, n) {
  if (!s) return ''
  return String(s).length > n ? String(s).slice(0, n) + '…' : String(s)
}

export default function GraphView({ nodes = [], edges = [], height = 480 }) {
  const containerRef = useRef(null)
  const graphRef = useRef(null)
  const selectedSetterRef = useRef(null)

  const [maxNodes, setMaxNodes] = useState(80)
  const [maxEdges, setMaxEdges] = useState(120)
  const [pendingNodes, setPendingNodes] = useState(80)
  const [pendingEdges, setPendingEdges] = useState(120)
  const [selectedNode, setSelectedNode] = useState(null)

  // keep ref in sync so G6 event handler doesn't go stale
  selectedSetterRef.current = setSelectedNode

  useEffect(() => {
    if (!containerRef.current || (!nodes.length && !edges.length)) return

    if (graphRef.current) {
      graphRef.current.destroy()
      graphRef.current = null
    }

    // Build node map from explicit list (capped)
    // Index by BOTH n.id AND n.properties.id to handle Source node ID mismatch
    const nodeMap = new Map()
    nodes.slice(0, maxNodes).forEach(n => {
      const id = String(n.id)
      const entry = {
        id,
        data: {
          label: truncate(n.properties?.name || n.name || n.properties?.title || n.title || id, 16),
          nodeType: n.labels?.[0] || 'Node',
          raw: n,
        },
      }
      nodeMap.set(id, entry)
      const altId = n.properties?.id
      if (altId != null && String(altId) !== id) nodeMap.set(String(altId), entry)
    })

    // Build edges (capped), auto-add placeholder nodes for unknown endpoints
    const g6Edges = edges.slice(0, maxEdges).map((e, i) => ({
      id: e.id != null ? String(e.id) : `e${i}`,
      source: String(e.start_id ?? e.start ?? e.source),
      target: String(e.end_id ?? e.end ?? e.target),
      data: { label: e.type || '' },
    }))

    g6Edges.forEach(e => {
      const mkPlaceholder = id => ({ id, data: { label: truncate(id, 16), nodeType: 'Node', raw: null } })
      if (!nodeMap.has(e.source)) nodeMap.set(e.source, mkPlaceholder(e.source))
      if (!nodeMap.has(e.target)) nodeMap.set(e.target, mkPlaceholder(e.target))
    })

    const g6Nodes = [...nodeMap.values()]

    const graph = new Graph({
      container: containerRef.current,
      width: containerRef.current.offsetWidth || 700,
      height,
      data: { nodes: g6Nodes, edges: g6Edges },
      node: {
        style: {
          size: 52,
          fill: d => hashColor(d.data?.nodeType),
          stroke: 'rgba(255,255,255,0.25)',
          strokeWidth: 2,
          cursor: 'pointer',
          labelText: d => d.data?.label,
          labelFill: '#ffffff',
          labelFontSize: 12,
          labelFontWeight: 700,
          labelPlacement: 'center',
          labelWordWrap: false,
        },
        state: {
          selected: {
            stroke: '#ffffff',
            strokeWidth: 3,
            shadowColor: '#ffffff',
            shadowBlur: 15,
          },
        },
      },
      edge: {
        style: {
          stroke: '#7c8fa6',
          strokeWidth: 2,
          endArrow: true,
          endArrowType: 'vee',
          endArrowSize: 10,
          labelText: d => d.data?.label,
          labelFontSize: 10,
          labelFill: '#cbd5e1',
          labelBackground: true,
          labelBackgroundFill: '#1e293b',
          labelBackgroundOpacity: 0.9,
          labelBackgroundRadius: 3,
          labelPadding: [2, 5],
        },
      },
      layout: {
        type: 'd3-force',
        preventOverlap: true,
        nodeSize: 56,
        linkDistance: 120,
        nodeStrength: -300,
        edgeStrength: 1,
        collideStrength: 1,
        iterations: 300,
      },
      behaviors: ['drag-canvas', 'drag-element', 'zoom-canvas'],
      autoFit: 'view',
      padding: 40,
    })

    let activeId = null

    graph.on(NodeEvent.CLICK, e => {
      const id = e.target?.id
      if (!id) return
      if (activeId && activeId !== id) {
        graph.setElementState(activeId, false)   // clear states on prev node
      }
      if (activeId === id) {
        graph.setElementState(id, false)
        activeId = null
        selectedSetterRef.current(null)
      } else {
        graph.setElementState(id, 'selected')
        activeId = id
        const d = graph.getNodeData(id)
        selectedSetterRef.current(d?.data?.raw || { id })
      }
    })

    graph.on('canvas:click', () => {
      if (activeId) { graph.setElementState(activeId, false); activeId = null }
      selectedSetterRef.current(null)
    })

    graph.render().catch(() => {})
    graphRef.current = graph

    return () => {
      if (graphRef.current) { graphRef.current.destroy(); graphRef.current = null }
    }
  }, [nodes, edges, maxNodes, maxEdges, height])

  const nodeTypes = [...new Set(nodes.map(n => n.labels?.[0] || 'Node'))]
  const truncatedNodes = nodes.length > maxNodes
  const truncatedEdges = edges.length > maxEdges

  if (!nodes.length && !edges.length) return null

  return (
    <div>
      {/* Controls */}
      <div className="flex items-center gap-4 mb-2 flex-wrap">
        <div className="flex items-center gap-2">
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>顯示節點</span>
          <input type="number" min={10} max={300} value={pendingNodes}
            onChange={e => setPendingNodes(Number(e.target.value))}
            className="input-field text-xs text-center py-0.5"
            style={{ width: 76 }}/>
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>邊</span>
          <input type="number" min={10} max={1000} value={pendingEdges}
            onChange={e => setPendingEdges(Number(e.target.value))}
            className="input-field text-xs text-center py-0.5"
            style={{ width: 76 }}/>
          <button
            onClick={() => { setMaxNodes(pendingNodes); setMaxEdges(pendingEdges) }}
            className="btn-secondary text-xs px-3 py-1">
            Apply
          </button>
        </div>
        <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
          {nodes.length} 節點 · {edges.length} 邊 (共)
        </span>
      </div>

      <div style={{ position: 'relative' }}>
        {/* Toolbar */}
        <div style={{ position:'absolute', top:10, right:10, zIndex:10, display:'flex', gap:6 }}>
          <button onClick={() => graphRef.current?.fitView()}
            style={{ fontSize:11, padding:'3px 8px', borderRadius:6, cursor:'pointer',
              background:'rgba(30,41,59,0.9)', border:'1px solid rgba(255,255,255,0.15)', color:'#94a3b8' }}>
            Fit
          </button>
          <button onClick={() => graphRef.current?.zoomTo(1)}
            style={{ fontSize:11, padding:'3px 8px', borderRadius:6, cursor:'pointer',
              background:'rgba(30,41,59,0.9)', border:'1px solid rgba(255,255,255,0.15)', color:'#94a3b8' }}>
            1:1
          </button>
        </div>
        {/* Legend */}
        {nodeTypes.length > 0 && (
          <div style={{ position:'absolute', bottom:10, left:10, zIndex:10, display:'flex', gap:6, flexWrap:'wrap', maxWidth:'50%' }}>
            {nodeTypes.map(t => (
              <span key={t} style={{ fontSize:10, padding:'2px 7px', borderRadius:10,
                background: hashColor(t)+'33', border:`1px solid ${hashColor(t)}66`,
                color: hashColor(t), fontWeight:700 }}>{t}</span>
            ))}
          </div>
        )}
        {/* Canvas */}
        <div ref={containerRef} style={{ width:'100%', height, background:'#0f172a', borderRadius:12,
          overflow:'hidden', border:'1px solid rgba(255,255,255,0.07)' }}/>
        {/* Detail panel — overlays top-right of canvas */}
        {selectedNode && (
          <div style={{ position:'absolute', top:44, right:10, zIndex:20, width:260,
            background:'rgba(15,23,42,0.97)', border:'1px solid rgba(255,255,255,0.15)',
            borderRadius:12, padding:'14px 16px', overflowY:'auto', maxHeight: height - 60, fontSize:12 }}>
            <div className="flex items-center justify-between mb-2">
              <div className="flex gap-1 flex-wrap">
                {(Array.isArray(selectedNode.labels) ? selectedNode.labels : [selectedNode.labels]).filter(Boolean).map((l,i) => (
                  <span key={i} style={{ fontSize:10, padding:'2px 7px', borderRadius:10,
                    background: hashColor(l)+'33', border:`1px solid ${hashColor(l)}66`,
                    color: hashColor(l), fontWeight:700 }}>{l}</span>
                ))}
              </div>
              <button onClick={() => setSelectedNode(null)}
                style={{ color:'#64748b', fontSize:14, lineHeight:1, cursor:'pointer', background:'none', border:'none' }}>✕</button>
            </div>
            <div style={{ borderTop:'1px solid rgba(255,255,255,0.07)', paddingTop:8 }}>
              <NodeProperties node={selectedNode} />
            </div>
          </div>
        )}
      </div>

      <p style={{ fontSize:10, color:'#475569', marginTop:4, display:'flex', justifyContent:'space-between' }}>
        <span>
          {(truncatedNodes || truncatedEdges) && (
            <span style={{ color:'#f59e0b' }}>
              顯示 {Math.min(nodes.length, maxNodes)} / {nodes.length} 節點、{Math.min(edges.length, maxEdges)} / {edges.length} 邊
            </span>
          )}
        </span>
        <span>拖曳節點 · 滾輪縮放 · 點選節點查看詳情</span>
      </p>
    </div>
  )
}

const SKIP_PROPS = new Set(['id','branch_id'])

function NodeProperties({ node }) {
  if (!node) return null
  // Merge top-level semantic fields + properties, deduplicate
  const merged = {}
  const topLevel = ['name','type','description','title','media_type','asset_path',
    'version_id','asr_text','contextual_text','lvlm_description','start_sec','end_sec',
    'entity','relation','value','time_start','time_end','confidence']
  topLevel.forEach(k => { if (node[k] != null && node[k] !== '') merged[k] = node[k] })
  Object.entries(node.properties || {}).forEach(([k, v]) => {
    if (!SKIP_PROPS.has(k) && v != null && v !== '' && merged[k] == null) merged[k] = v
  })
  const entries = Object.entries(merged)
  if (entries.length === 0) {
    return <p style={{ color:'#475569', fontSize:12 }}>No properties</p>
  }
  return (
    <div style={{ display:'flex', flexDirection:'column', gap:8 }}>
      {entries.map(([k, v]) => (
        <div key={k}>
          <p style={{ color:'#64748b', fontSize:11, marginBottom:2, fontWeight:600 }}>{k}</p>
          <p style={{ color:'#e2e8f0', fontSize:12, lineHeight:1.5, wordBreak:'break-word' }}>
            {String(v).length > 300 ? String(v).slice(0, 300) + '…' : String(v)}
          </p>
        </div>
      ))}
    </div>
  )
}
