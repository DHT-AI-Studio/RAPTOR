// Used only if the gateway's own /health can't be reached. No value is
// hardcoded here -- a stale guess (e.g. a version the gateway has since
// dropped its legacy alias for) is worse than failing loudly. Set
// VITE_FALLBACK_API_VERSION to give this a value for a deployment that
// expects to need it (e.g. the gateway's /health route itself unreachable
// through a restrictive proxy).
const FALLBACK_API_VERSION = import.meta.env.VITE_FALLBACK_API_VERSION || null

function getBase() {
  return localStorage.getItem('raptor_api_base') || ''
}

// Cached per base URL (not a module-level constant): the gateway's current
// API version comes from its own /health, not a hardcoded guess, so pointing
// the demo at a different deployment (a different RAPTOR_VERSION) just works.
// Caches only a successful lookup -- a failed one retries next call instead
// of being stuck on a stale answer for the rest of the session.
let versionCache = null // { base, version }

export async function resolveApiVersion() {
  const base = getBase()
  if (versionCache?.base === base) return versionCache.version
  try {
    const res = await fetch(`${base}/health`)
    const version = res.ok ? (await res.json()).version : null
    if (version) {
      versionCache = { base, version }
      return version
    }
  } catch {}
  if (FALLBACK_API_VERSION) return FALLBACK_API_VERSION
  throw new Error(
    'Could not reach the gateway\'s /health to determine its API version. ' +
    'Set VITE_FALLBACK_API_VERSION to configure one explicitly.'
  )
}

function authHeaders(token) {
  return {
    Authorization: `Bearer ${token}`,
    'Content-Type': 'application/json',
  }
}

async function handleResponse(res) {
  if (res.status === 401) {
    window.dispatchEvent(new CustomEvent('raptor:unauthorized'))
    throw new Error('Session expired')
  }
  if (!res.ok) {
    let msg = `HTTP ${res.status}`
    try { const d = await res.json(); msg = d.detail || JSON.stringify(d) } catch {}
    throw new Error(msg)
  }
  return res.json()
}

// ── Auth ──────────────────────────────────────────────────────────────────────

export async function apiLogin({ username, password, realmName = 'dhtsolution', clientId = 'raptor' }) {
  const body = new URLSearchParams({ username, password, realm_name: realmName })
  const res = await fetch(
    `${getBase()}/api/${await resolveApiVersion()}/sso/login?client_id=${encodeURIComponent(clientId)}`,
    { method: 'POST', body, headers: { 'Content-Type': 'application/x-www-form-urlencoded' } }
  )
  return handleResponse(res)
}

export async function apiLogout(token) {
  const res = await fetch(`${getBase()}/api/${await resolveApiVersion()}/sso/logout`, {
    method: 'POST',
    headers: authHeaders(token),
  })
  if (!res.ok) { const d = await res.json().catch(() => ({})); throw new Error(d.detail || `HTTP ${res.status}`) }
  return res.json()
}

// ── Search ────────────────────────────────────────────────────────────────────

export async function apiHybridSearch(token, payload) {
  const res = await fetch(`${getBase()}/api/${await resolveApiVersion()}/search/hybrid`, {
    method: 'POST', headers: authHeaders(token), body: JSON.stringify(payload),
  })
  return handleResponse(res)
}

export async function apiBm25Search(token, payload) {
  const res = await fetch(`${getBase()}/api/${await resolveApiVersion()}/search/bm25`, {
    method: 'POST', headers: authHeaders(token), body: JSON.stringify(payload),
  })
  return handleResponse(res)
}

export async function apiVectorSearch(token, payload) {
  const res = await fetch(`${getBase()}/api/${await resolveApiVersion()}/search/vector`, {
    method: 'POST', headers: authHeaders(token), body: JSON.stringify(payload),
  })
  return handleResponse(res)
}

export async function apiTkgSearch(token, payload) {
  const res = await fetch(`${getBase()}/api/${await resolveApiVersion()}/search/tkg`, {
    method: 'POST', headers: authHeaders(token), body: JSON.stringify(payload),
  })
  return handleResponse(res)
}

export async function apiGraphRagSearch(token, payload) {
  const res = await fetch(`${getBase()}/api/${await resolveApiVersion()}/search/graphrag`, {
    method: 'POST', headers: authHeaders(token), body: JSON.stringify(payload),
  })
  return handleResponse(res)
}

// ── Video Search ──────────────────────────────────────────────────────────────

export async function apiVideoSearch(token, payload) {
  const res = await fetch(`${getBase()}/api/${await resolveApiVersion()}/search/video_search`, {
    method: 'POST', headers: authHeaders(token), body: JSON.stringify(payload),
  })
  return handleResponse(res)
}

// ── RAG Query (a2a) ───────────────────────────────────────────────────────────

export async function apiA2aQuery(token, { question, topK = 5, mode = 'direct' }) {
  const res = await fetch(`${getBase()}/api/${await resolveApiVersion()}/a2a/query`, {
    method: 'POST',
    headers: authHeaders(token),
    body: JSON.stringify({ question, top_k: topK, mode }),
  })
  return handleResponse(res)
}

// ── Asset ─────────────────────────────────────────────────────────────────────

export async function apiUploadAnalysis(token, file, { processingMode = 'default', archiveTtl, destroyTtl } = {}) {
  const form = new FormData()
  form.append('primary_file', file)
  if (processingMode) form.append('processing_mode', processingMode)
  if (archiveTtl) form.append('archive_date_or_ttl', archiveTtl)
  if (destroyTtl) form.append('destroy_date_or_ttl', destroyTtl)

  const res = await fetch(`${getBase()}/api/${await resolveApiVersion()}/asset/fileupload_analysis`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
    body: form,
  })
  return handleResponse(res)
}

// ── Processing Progress ───────────────────────────────────────────────────────

export async function apiGetProgress(token, mType, key) {
  const res = await fetch(
    `${getBase()}/api/${await resolveApiVersion()}/processing/cache/${encodeURIComponent(mType)}/${encodeURIComponent(key)}`,
    { headers: authHeaders(token) }
  )
  return handleResponse(res)
}

// ── Asset presigned URL (for video playback in history) ──────────────────────

export async function apiGetVideoUrl(token, assetPath, versionId) {
  const encodedPath = assetPath.split('/').map(encodeURIComponent).join('/')
  const res = await fetch(
    `${getBase()}/api/${await resolveApiVersion()}/asset/filedownload/${encodedPath}/${encodeURIComponent(versionId)}`,
    { headers: authHeaders(token) }
  )
  const data = await handleResponse(res)
  return data.primary_file?.url ?? null
}

export async function apiGetKeyframeUrls(token, assetPath, versionId) {
  const encodedPath = assetPath.split('/').map(encodeURIComponent).join('/')
  const res = await fetch(
    `${getBase()}/api/${await resolveApiVersion()}/asset/filedownload/${encodedPath}/${encodeURIComponent(versionId)}`,
    { headers: authHeaders(token) }
  )
  const data = await handleResponse(res)
  return Object.entries(data)
    .filter(([k]) => k.startsWith('associated_file_'))
    .map(([, f]) => f)
    .filter(f => /\.(jpg|jpeg|png|gif|webp)$/i.test(f?.filename ?? ''))
    .map(f => ({ filename: f.filename, url: f.url }))
}

// ── Asset commits (upload history) ───────────────────────────────────────────

export async function apiGetUserCommits(token, { keyword, startDate, endDate, page = 1, pageSize = 10, status } = {}) {
  const params = new URLSearchParams({ page, page_size: pageSize })
  if (keyword)   params.set('keyword',    keyword)
  if (startDate) params.set('start_date', startDate)
  if (endDate)   params.set('end_date',   endDate)
  // Repeatable query param -- omitted means all 6 statuses (uploaded/processing/
  // active/failed/unregistered/archived), not just active.
  if (status)    status.forEach(s => params.append('status', s))
  const res = await fetch(`${getBase()}/api/${await resolveApiVersion()}/asset/users/commits?${params}`, {
    headers: authHeaders(token),
  })
  return handleResponse(res)
}

function encodePath(p) {
  return p.split('/').map(encodeURIComponent).join('/')
}

export async function apiArchiveAsset(token, assetPath, versionId) {
  const res = await fetch(
    `${getBase()}/api/${await resolveApiVersion()}/asset/filearchive/${encodePath(assetPath)}/${encodeURIComponent(versionId)}`,
    { method: 'POST', headers: authHeaders(token) }
  )
  return handleResponse(res)
}

export async function apiDeleteAsset(token, assetPath, versionId) {
  const res = await fetch(
    `${getBase()}/api/${await resolveApiVersion()}/asset/delfile/${encodePath(assetPath)}/${encodeURIComponent(versionId)}`,
    { method: 'POST', headers: authHeaders(token) }
  )
  return handleResponse(res)
}

// ── Asset thumbnail (for the Identity video grid) ─────────────────────────────

export async function apiGetThumbnailUrl(token, assetPath, versionId, second = 1) {
  const res = await fetch(
    `${getBase()}/api/${await resolveApiVersion()}/asset/thumbnail/${encodePath(assetPath)}/${encodeURIComponent(versionId)}?second=${second}`,
    { headers: { Authorization: `Bearer ${token}` } }
  )
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return URL.createObjectURL(await res.blob())
}

// ── Identity ──────────────────────────────────────────────────────────────────

export async function apiListIdentities(token, { limit = 100, offset = 0, status } = {}) {
  const params = new URLSearchParams({ limit, offset })
  if (status) params.set('status', status)
  const res = await fetch(`${getBase()}/api/${await resolveApiVersion()}/identity/identities?${params}`, {
    headers: authHeaders(token),
  })
  return handleResponse(res)
}

export async function apiPendingPersons(token, versionId) {
  const res = await fetch(
    `${getBase()}/api/${await resolveApiVersion()}/identity/file/${encodeURIComponent(versionId)}/pending-persons`,
    { headers: authHeaders(token) }
  )
  return handleResponse(res)
}

export async function apiCreatePendingPerson(token, versionId, body) {
  const res = await fetch(
    `${getBase()}/api/${await resolveApiVersion()}/identity/file/${encodeURIComponent(versionId)}/pending-person`,
    { method: 'POST', headers: authHeaders(token), body: JSON.stringify(body) }
  )
  return handleResponse(res)
}

export async function apiBindGroup(token, identityId, versionId, groupId) {
  const res = await fetch(
    `${getBase()}/api/${await resolveApiVersion()}/identity/${encodeURIComponent(identityId)}/bind/group`,
    { method: 'POST', headers: authHeaders(token), body: JSON.stringify({ version_id: versionId, group_id: groupId }) }
  )
  return handleResponse(res)
}

export async function apiUnbindTrace(token, identityId, versionId, traceId) {
  const res = await fetch(
    `${getBase()}/api/${await resolveApiVersion()}/identity/${encodeURIComponent(identityId)}/unbind`,
    { method: 'POST', headers: authHeaders(token), body: JSON.stringify({ version_id: versionId, trace_id: traceId }) }
  )
  return handleResponse(res)
}

// ── GDPR Consent (GD-5) ────────────────────────────────────────────────────────

export async function apiGetConsent(token) {
  const res = await fetch(`${getBase()}/api/${await resolveApiVersion()}/gdpr/consent`, {
    headers: authHeaders(token),
  })
  return handleResponse(res)
}

// Withdrawing some purposes also deletes that purpose's already-stored data
// (the response's erasure_job_id is set when that cascade started) -- the
// caller is responsible for warning the user first, this just makes the call.
export async function apiSetConsent(token, purpose, granted) {
  const res = await fetch(
    `${getBase()}/api/${await resolveApiVersion()}/gdpr/consent/${encodeURIComponent(purpose)}`,
    { method: 'PUT', headers: authHeaders(token), body: JSON.stringify({ granted }) }
  )
  return handleResponse(res)
}
