# Proxy test: nginx / Envoy in front of grpc-gateway

Confirms grpc-gateway is reachable through a real reverse proxy with HTTP/2
forwarding enabled — both nginx and Envoy need explicit config for this
(gRPC never works over plain HTTP/1.1 proxying).

## What's here

- `nginx.conf` — `listen ... http2` + `grpc_pass` (nginx's native gRPC proxy
  mode, not `proxy_pass`), with read/send timeouts raised for the
  long-lived streaming RPCs (DownloadAsset/UploadAsset/WatchJob).
- `envoy.yaml` — an HTTP/2 listener + cluster, with
  `http2_protocol_options` set on *both* the listener and the upstream
  cluster (Envoy defaults to HTTP/1.1 for the backend connection otherwise,
  which silently breaks every call), and a raised route timeout for the
  same streaming-RPC reason.
- `../docker-compose.proxy-test.yml` — brings both proxies up alongside
  grpc-gateway on the `raptor` network, nginx on `:8826`, Envoy on `:8827`
  (admin UI on `:9901`).

## Running it

```bash
cd deployment/modules/36-grpc-gateway
docker compose -f docker-compose.yml -f docker-compose.proxy-test.yml up -d --build

# via nginx
grpcurl -plaintext -H "authorization: Bearer x" -d '{"message":"hi"}' \
  localhost:8826 raptor.v1.RaptorService/Ping

# via Envoy
grpcurl -plaintext -H "authorization: Bearer x" -d '{"message":"hi"}' \
  localhost:8827 raptor.v1.RaptorService/Ping

docker compose -f grpc-gateway/docker-compose.yml -f grpc-gateway/docker-compose.proxy-test.yml down
```

## Verified (2026-09-30, local docker)

Ran the above for real (not just written and assumed correct) — built the
image, brought up grpc-gateway + both proxies on the `raptor` network, and
called through each with `grpcurl` (container-to-container, no host
networking tricks):

| Check | via nginx (:8826) | via Envoy (:8827) |
| --- | --- | --- |
| `Ping` (authenticated) | ✅ correct `message`/`request_id` echoed back | ✅ same |
| `grpc.health.v1.Health/Check` | ✅ `SERVING` | ✅ `SERVING` |
| `Ping` (no credentials) | ✅ `UNAUTHENTICATED`, correct message | ✅ same |

Status codes and error messages — not just successful responses — pass
through both proxies intact, confirming trailer/metadata forwarding works,
not only the HTTP/2 framing.

## Verified again with streaming and real auth (2026-10-20, local docker)

The first run (above) only covered unary calls with the old stub auth. This one ran this
branch's gateway image, redis, nginx 1.27 and Envoy 1.31 on one docker network, with the
unmodified configs from this directory, a real Keycloak token and the real asset service,
and sent the same calls directly and through each proxy:

| Check | direct | nginx | Envoy |
| --- | --- | --- | --- |
| `Health/Check`, authenticated `Ping` | ok | ok | ok |
| no credentials / invalid token | `UNAUTHENTICATED` | same | same |
| missing asset (`NOT_FOUND`), server-streaming error (`WatchJob` unknown job) | ok | same | same |
| `DownloadAsset` 9.9 MB, sha256 compared | match | match | match |
| `UploadAsset` 150 MB, downloaded back and compared | identical | identical | identical |
| 20 cancelled downloads | counted `CANCELLED`, `active_streams` 0 | same | same |

**Found and fixed by this run:** `UploadAsset` over 1 MiB through nginx failed with HTTP 413 ("client
intended to send too large chunked body"): nginx counts the messages of a client-streaming call toward
`client_max_body_size`, default 1 MiB. `nginx.conf` now sets `client_max_body_size 0;`.

Not covered: TLS through the proxies, and HTTP/2 behaviour under many concurrent streams.

## TLS terminated at the proxy (nginx-tls.conf, envoy-tls.yaml)

The examples above are plaintext. In front of the gateway you normally want TLS at the proxy, with the gateway
reachable only on the private network. `nginx-tls.conf` (`:8828`) and `envoy-tls.yaml` (`:8829`) do that: the
client speaks TLS with HTTP/2 (ALPN `h2`) to the proxy, and the proxy speaks plaintext HTTP/2 to `grpc-gateway:8825`.
Everything else (timeouts, `client_max_body_size 0`) is the same as in the plaintext examples.

```bash
cd deployment/modules/36-grpc-gateway
bash proxy-test/make_certs.sh            # throwaway CA and server certificate, in proxy-test/certs/ (git-ignored)
docker compose -f docker-compose.yml -f docker-compose.proxy-test.yml up -d --build

grpcurl -cacert proxy-test/certs/ca.pem -authority localhost -H "authorization: Bearer <jwt>" \
  -d '{"message":"hi"}' localhost:8828 raptor.v1.RaptorService/Ping     # via nginx
grpcurl -cacert proxy-test/certs/ca.pem -authority localhost -H "authorization: Bearer <jwt>" \
  -d '{"message":"hi"}' localhost:8829 raptor.v1.RaptorService/Ping     # via Envoy
```

**Verified (2026-10-27, local docker):** the gateway image built from this repo, redis, a fake asset service,
nginx 1.27 and Envoy 1.31 on one docker network, a real Keycloak token, and a Python gRPC client over TLS.

| Check | via nginx (:8828) | via Envoy (:8829) |
| --- | --- | --- |
| `Ping` with a real token; `Health/Check` | ok | ok |
| no credentials | `UNAUTHENTICATED` (from the gateway) | same |
| plaintext client to the TLS port | refused | refused |
| client that trusts a different root | refused | refused |
| server name not in the certificate | refused | refused |
| 150 MiB `UploadAsset` | ok (0.8 s) | ok (0.7 s) |
| 200 MiB `DownloadAsset` | ok (0.9 s) | ok (0.9 s) |
| 10 cancelled downloads | ok | ok |
| server-streaming error (`WatchJob`, unknown job) | `NOT_FOUND` | same |

Not covered: re-encrypting to a gateway that itself has TLS on (`grpc_pass grpcs://...`, or an Envoy upstream
`transport_socket`), mTLS at the proxy, certificates from a real CA, and certificate rotation.

## Production notes

- Both configs listen `plaintext` (no TLS) here, matching how this test
  runs. In a real deployment, terminate TLS at the proxy (or pass through
  to grpc-gateway's own TLS listener — see `GRPC_GATEWAY_GRPC_SECURE` in
  `config.py`) and use `grpc_pass grpcs://...` (nginx) or
  `transport_socket` with a TLS context (Envoy) accordingly.
- The raised timeouts here (3600s) are deliberately generous for a proxy
  that has to accommodate long file transfers; tune down if that's not a
  real requirement for a given deployment.
