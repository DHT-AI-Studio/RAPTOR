# MCP OAuth Support (Modules 06, 13, 27)

## Problem

Module 27's MCP server had no working login path for standard MCP clients (e.g. Claude Code):

- Module 06 had no public OAuth client with PKCE — only a confidential client with Keycloak's factory-default redirect URIs, unusable by a client that can't hold a secret.
- Module 13's MCP proxy exposed no discovery metadata, so a compliant client had no way to learn how to authenticate.
- Module 27's `agent_token` (`client_credentials`) registration was open to any bearer token, and collapsed every registrant of a given `client_id` into one shared service-account identity, with no way to tell users apart.

## Fix

- **Module 06** — added a new public Keycloak client, `raptor-mcp` (PKCE required, fixed-port loopback redirect URI). Additive only; the existing `raptor` client is unchanged.
- **Module 13** — serves OAuth Protected Resource Metadata ([RFC 9728](https://www.rfc-editor.org/rfc/rfc9728)) at `/.well-known/oauth-protected-resource`, and adds a `WWW-Authenticate` header on 401s from the MCP proxy so a compliant client can auto-discover the flow. No `/authorize`, `/callback`, or token-exchange endpoint was added — Keycloak remains the sole Authorization Server; the gateway stays a Resource Server.
- **Module 27** — `agent_token` registration is now default-deny via `MCP_AGENT_ALLOWED_CLIENT_IDS`. Ownership of a `client_id` is claimed by whoever registers it first; a different person attempting to reuse the same `client_id` gets `409 Conflict` instead of silently sharing that identity.

Redirect URIs use a fixed callback port rather than a wildcard port — Keycloak's wildcard matching is trailing-path only (`http://host/*`), it does not match an arbitrary port.

## How to use

### Human login via a standard MCP client (e.g. Claude Code)

```bash
claude mcp add --transport http raptor https://<gateway-host>:8013/api/<version>/mcp --client-id raptor-mcp --callback-port 53682
# then, inside claude: /mcp -> raptor -> Authenticate
```

This opens a browser and authenticates directly against Keycloak; the gateway never sees a password. It then completes `authorization_code` + PKCE. The resulting identity is the user's own Keycloak account, not a shared or service one.

The browser must run on the machine that runs `claude`, because the callback is `http://localhost:53682/callback`. Over SSH, forward the port from the machine with the browser: `ssh -L 53682:localhost:53682 <user>@<host running claude>`. If the deployment's certificate is not from a public CA, start `claude` with `NODE_EXTRA_CA_CERTS=/path/to/ca.pem`.

Deployment requirements:

- `GATEWAY_KEYCLOAK_PUBLIC_URL` (module 13) and `KEYCLOAK_PUBLIC_URL` (module 06, drives `KC_HOSTNAME`; defaults to `HOST_IP`) must resolve to the same host as seen by the client — otherwise the discovery metadata and the token's `iss` claim disagree and login fails.
- **HTTPS for clients on other machines.** The token endpoint must be reachable over HTTPS unless client and Keycloak share the same host (loopback is exempt): most OAuth clients refuse to send credentials to a plaintext, non-loopback token endpoint (GitHub #163). `06-authentication/docker-compose.yml`'s opt-in `tls` Compose profile (`COMPOSE_PROFILES=tls`) puts nginx in front of Keycloak (`:8443`) and the gateway (`:8013`) to terminate TLS. Its README section "TLS for Keycloak and the gateway" covers turning it on and the certificates. With it:
  - `KEYCLOAK_PUBLIC_URL` and `GATEWAY_PUBLIC_URL` are the https addresses (`GATEWAY_KEYCLOAK_PUBLIC_URL` just copies `KEYCLOAK_PUBLIC_URL`, nothing to set there separately). When a root `deployment/modules/.env` exists, set them there: the gateway reads it after its own `.env`, so its values win.
  - `GATEWAY_PUBLIC_URL` is needed because the gateway sees plain http from the proxy. Without it, the discovery metadata would name the internal http address and clients would reject it.
  - This was rehearsed from a second machine on 2026-10-29 and reached "Got new credentials".
- On an already-running realm, the `raptor-mcp` client must also be created via the Keycloak Admin API — `--import-realm` only seeds a realm on first creation, it does not update a live one.

### Autonomous agent via `agent_token`

```
POST /auth/register  {"client_id": ..., "client_secret": ..., "scope": ...}
```

Requires `client_id` to be listed in `MCP_AGENT_ALLOWED_CLIENT_IDS` (empty by default, i.e. disabled). The first caller to register a given `client_id` owns it going forward.

- **Permission checks use the resource server.** Module 06's `/auth/permission` used to send the token's `azp` as the UMA audience. Tokens from `raptor-mcp` carry `azp=raptor-mcp`, a public client with no authorization services, so Keycloak failed the request and every MCP call after a successful login was refused, on plain HTTP too. It now asks the resource server `KEYCLOAK_CLIENT_ID` (`raptor`), where the API's UMA resources live.

## Open items

- Single shared `raptor-mcp` client_id vs. one per installation — currently shared, which is normal for public OAuth clients since PKCE + the user's own login is the actual security boundary, not client_id secrecy.
- The discovery metadata's resource identifier currently points at the gateway (`/api/<version>/mcp`); confirm this matches how module 27 is reached in every deployment (some may expose it directly).

## References

- [RFC 8252 — OAuth 2.0 for Native Apps](https://www.rfc-editor.org/rfc/rfc8252)
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://www.rfc-editor.org/rfc/rfc9728)
- [MCP Authorization specification](https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization)
