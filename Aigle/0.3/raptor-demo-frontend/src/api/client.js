const API_VERSION = '0.3'

function getBase() {
  return localStorage.getItem('raptor_api_base') || ''
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
    `${getBase()}/api/${API_VERSION}/sso/login?client_id=${encodeURIComponent(clientId)}`,
    { method: 'POST', body, headers: { 'Content-Type': 'application/x-www-form-urlencoded' } }
  )
  return handleResponse(res)
}

export async function apiLogout(token) {
  const res = await fetch(`${getBase()}/api/${API_VERSION}/sso/logout`, {
    method: 'POST',
    headers: authHeaders(token),
  })
  if (!res.ok) { const d = await res.json().catch(() => ({})); throw new Error(d.detail || `HTTP ${res.status}`) }
  return res.json()
}

// ── Search ────────────────────────────────────────────────────────────────────

export async function apiHybridSearch(token, payload) {
  const res = await fetch(`${getBase()}/api/${API_VERSION}/search/hybrid`, {
    method: 'POST', headers: authHeaders(token), body: JSON.stringify(payload),
  })
  return handleResponse(res)
}

export async function apiBm25Search(token, payload) {
  const res = await fetch(`${getBase()}/api/${API_VERSION}/search/bm25`, {
    method: 'POST', headers: authHeaders(token), body: JSON.stringify(payload),
  })
  return handleResponse(res)
}

export async function apiVectorSearch(token, payload) {
  const res = await fetch(`${getBase()}/api/${API_VERSION}/search/vector`, {
    method: 'POST', headers: authHeaders(token), body: JSON.stringify(payload),
  })
  return handleResponse(res)
}

export async function apiTkgSearch(token, payload) {
  const res = await fetch(`${getBase()}/api/${API_VERSION}/search/tkg`, {
    method: 'POST', headers: authHeaders(token), body: JSON.stringify(payload),
  })
  return handleResponse(res)
}

export async function apiGraphRagSearch(token, payload) {
  const res = await fetch(`${getBase()}/api/${API_VERSION}/search/graphrag`, {
    method: 'POST', headers: authHeaders(token), body: JSON.stringify(payload),
  })
  return handleResponse(res)
}

// ── Video Search ──────────────────────────────────────────────────────────────

export async function apiVideoSearch(token, payload) {
  const res = await fetch(`${getBase()}/api/${API_VERSION}/search/video_search`, {
    method: 'POST', headers: authHeaders(token), body: JSON.stringify(payload),
  })
  return handleResponse(res)
}

// ── RAG Query (a2a) ───────────────────────────────────────────────────────────

export async function apiA2aQuery(token, { question, topK = 5, mode = 'direct' }) {
  const res = await fetch(`${getBase()}/api/${API_VERSION}/a2a/query`, {
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

  const res = await fetch(`${getBase()}/api/${API_VERSION}/asset/fileupload_analysis`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
    body: form,
  })
  return handleResponse(res)
}

// ── Processing Progress ───────────────────────────────────────────────────────

export async function apiGetProgress(token, mType, key) {
  const res = await fetch(
    `${getBase()}/api/${API_VERSION}/processing/cache/${encodeURIComponent(mType)}/${encodeURIComponent(key)}`,
    { headers: authHeaders(token) }
  )
  return handleResponse(res)
}

// ── Asset presigned URL (for video playback in history) ──────────────────────

export async function apiGetVideoUrl(token, assetPath, versionId) {
  const encodedPath = assetPath.split('/').map(encodeURIComponent).join('/')
  const res = await fetch(
    `${getBase()}/api/${API_VERSION}/asset/filedownload/${encodedPath}/${encodeURIComponent(versionId)}`,
    { headers: authHeaders(token) }
  )
  const data = await handleResponse(res)
  return data.primary_file?.url ?? null
}

export async function apiGetKeyframeUrls(token, assetPath, versionId) {
  const encodedPath = assetPath.split('/').map(encodeURIComponent).join('/')
  const res = await fetch(
    `${getBase()}/api/${API_VERSION}/asset/filedownload/${encodedPath}/${encodeURIComponent(versionId)}`,
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

export async function apiGetUserCommits(token, { keyword, startDate, endDate, page = 1, pageSize = 10 } = {}) {
  const params = new URLSearchParams({ page, page_size: pageSize })
  if (keyword)   params.set('keyword',    keyword)
  if (startDate) params.set('start_date', startDate)
  if (endDate)   params.set('end_date',   endDate)
  const res = await fetch(`${getBase()}/api/${API_VERSION}/asset/users/commits?${params}`, {
    headers: authHeaders(token),
  })
  return handleResponse(res)
}

function encodePath(p) {
  return p.split('/').map(encodeURIComponent).join('/')
}

export async function apiArchiveAsset(token, assetPath, versionId) {
  const res = await fetch(
    `${getBase()}/api/${API_VERSION}/asset/filearchive/${encodePath(assetPath)}/${encodeURIComponent(versionId)}`,
    { method: 'POST', headers: authHeaders(token) }
  )
  return handleResponse(res)
}

export async function apiDeleteAsset(token, assetPath, versionId) {
  const res = await fetch(
    `${getBase()}/api/${API_VERSION}/asset/delfile/${encodePath(assetPath)}/${encodeURIComponent(versionId)}`,
    { method: 'POST', headers: authHeaders(token) }
  )
  return handleResponse(res)
}
