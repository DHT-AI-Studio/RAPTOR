#!/usr/bin/env bash
# Throwaway certificates for the TLS proxy examples (nginx-tls.conf, envoy-tls.yaml): a CA and a
# server certificate valid for localhost, 127.0.0.1 and the proxies' service names. For trying
# things out only: use real certificates anywhere else. Output: proxy-test/certs/ (git-ignored).
set -euo pipefail
out="$(cd "$(dirname "$0")" && pwd)/certs"
mkdir -p "$out"
cd "$out"
openssl genrsa -out ca.key 2048 2>/dev/null
openssl req -x509 -new -key ca.key -days 7 -subj "/CN=raptor-proxy-test-ca" -out ca.pem
openssl genrsa -out server.key 2048 2>/dev/null
openssl req -new -key server.key -subj "/CN=localhost" -out server.csr
cat > san.cnf <<'CNF'
subjectAltName=DNS:localhost,DNS:nginx-tls-proxy-test,DNS:envoy-tls-proxy-test,IP:127.0.0.1
extendedKeyUsage=serverAuth
CNF
openssl x509 -req -in server.csr -CA ca.pem -CAkey ca.key -CAcreateserial -days 7 -extfile san.cnf -out server.pem 2>/dev/null
rm -f server.csr san.cnf ca.srl
chmod 644 ca.pem server.pem server.key   # the proxies' containers read them as another user
echo "wrote $out/ca.pem server.pem server.key"
