# Raw JSON-RPC calls to Raptor's MCP server with curl — no LLM, no SDK.
# Requires: curl, python3 (for JSON parsing/pretty-printing only)
# Run:
#   ./examples/curl_mcp.sh --jwt <token>
#   KEYCLOAK_USERNAME=... KEYCLOAK_PASSWORD=... ./examples/curl_mcp.sh   # logs in itself
set -euo pipefail

MCP_SERVER_URL="${MCP_SERVER_URL:-http://localhost:8027/mcp}"
GATEWAY_BASE_URL="${GATEWAY_BASE_URL:-http://raptor_open_0_3_api.dhtsolution.com:8012}"
REALM_NAME="${REALM_NAME:-dhtsolution}"
CLIENT_ID="${CLIENT_ID:-raptor}"

JWT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --jwt) JWT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$JWT" ]]; then
  echo "== Authenticate with Keycloak, get a JWT ==" >&2
  JWT=$(curl -sf -X POST "${GATEWAY_BASE_URL}/api/0.4/sso/login?client_id=${CLIENT_ID}" \
    --data-urlencode "username=${KEYCLOAK_USERNAME}" \
    --data-urlencode "password=${KEYCLOAK_PASSWORD}" \
    --data-urlencode "realm_name=${REALM_NAME}" \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['access_token'])")
  echo "JWT acquired (${#JWT} chars)" >&2
fi

echo >&2
echo "== 1. initialize — MCP handshake, capture Mcp-Session-Id ==" >&2
INIT_HEADERS=$(mktemp)
curl -sf -D "$INIT_HEADERS" -o /dev/null -X POST "$MCP_SERVER_URL" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "Authorization: Bearer $JWT" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl-mcp-example","version":"1.0"}}}'
SESSION_ID=$(grep -i "^mcp-session-id:" "$INIT_HEADERS" | cut -d' ' -f2 | tr -d '\r')
rm -f "$INIT_HEADERS"
echo "Session: $SESSION_ID" >&2

echo "== (required) notifications/initialized — completes the handshake; tools/list and tools/call are rejected before this ==" >&2
curl -sf -X POST "$MCP_SERVER_URL" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "Authorization: Bearer $JWT" \
  -H "Mcp-Session-Id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized"}' > /dev/null

echo >&2
echo "== 2. tools/list — discover all registered Raptor tools ==" >&2
curl -sf -X POST "$MCP_SERVER_URL" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "Authorization: Bearer $JWT" \
  -H "Mcp-Session-Id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  | grep '^data: ' | tail -1 | sed 's/^data: //' \
  | python3 -c "
import json, sys
envelope = json.load(sys.stdin)
names = [t['name'] for t in envelope['result']['tools']]
print(f'{len(names)} tools:')
for n in names:
    print(f'  - {n}')
"

echo >&2
echo "== 3. tools/call — raptor_search(query='video', top_k=3) ==" >&2
curl -sf -X POST "$MCP_SERVER_URL" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "Authorization: Bearer $JWT" \
  -H "Mcp-Session-Id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"raptor_search","arguments":{"query":"video","top_k":3}}}' \
  | grep '^data: ' | tail -1 | sed 's/^data: //' \
  | python3 -c "
import json, sys
envelope = json.load(sys.stdin)
results = json.loads(envelope['result']['content'][0]['text'])
print(f'{len(results)} hit(s):')
for r in results:
    print(f\"  - [{r.get('score', 0):.3f}] {r.get('asset_path', '')}\")
"
