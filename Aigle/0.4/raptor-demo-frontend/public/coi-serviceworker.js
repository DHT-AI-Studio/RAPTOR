// coi-serviceworker — enables SharedArrayBuffer on non-HTTPS origins
// by injecting COOP/COEP headers via Service Worker interception.
self.addEventListener('install', () => self.skipWaiting())
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()))

async function handleFetch(request) {
  if (request.cache === 'only-if-cached' && request.mode !== 'same-origin') return
  let r
  try { r = await fetch(request) } catch (e) { throw e }
  if (r.status === 0) return r
  const h = new Headers(r.headers)
  h.set('Cross-Origin-Opener-Policy', 'same-origin')
  h.set('Cross-Origin-Embedder-Policy', 'credentialless')
  return new Response(r.body, { status: r.status, statusText: r.statusText, headers: h })
}

self.addEventListener('fetch', e => e.respondWith(handleFetch(e.request)))
