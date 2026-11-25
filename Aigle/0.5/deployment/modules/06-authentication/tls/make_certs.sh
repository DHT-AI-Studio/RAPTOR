#!/usr/bin/env bash
# Throwaway certificates for the TLS entry (docker-compose.tls.yml): a CA and a server certificate for the
# names and addresses clients use to reach Keycloak and the gateway. For rehearsals only -- a deployment
# that other machines use should get a certificate from a real CA (see README.md, "TLS for Keycloak and
# the gateway"). Output: tls/certs/ (git-ignored).
#   bash tls/make_certs.sh <name-or-ip> [<name-or-ip> ...]      e.g. bash tls/make_certs.sh 192.168.1.20 raptor.lan
# Clients must trust certs/ca.pem: Claude Code through NODE_EXTRA_CA_CERTS, browsers through their
# certificate store.
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: bash tls/make_certs.sh <name-or-ip> [<name-or-ip> ...]" >&2; exit 1; }
out="$(cd "$(dirname "$0")" && pwd)/certs"
mkdir -p "$out"
cd "$out"
san="DNS:localhost,IP:127.0.0.1"
for n in "$@"; do
  if [[ "$n" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then san="$san,IP:$n"; else san="$san,DNS:$n"; fi
done
openssl genrsa -out ca.key 2048 2>/dev/null
openssl req -x509 -new -key ca.key -days 30 -subj "/CN=raptor-tls-rehearsal-ca" -out ca.pem
openssl genrsa -out server.key 2048 2>/dev/null
openssl req -new -key server.key -subj "/CN=$1" -out server.csr
printf 'subjectAltName=%s\nextendedKeyUsage=serverAuth\n' "$san" > san.cnf
openssl x509 -req -in server.csr -CA ca.pem -CAkey ca.key -CAcreateserial -days 30 -extfile san.cnf -out server.pem 2>/dev/null
rm -f server.csr san.cnf ca.srl
chmod 644 ca.pem server.pem server.key   # nginx's container reads them as another user
echo "wrote $out/ca.pem server.pem server.key for $san"
