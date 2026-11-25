# Module 06: Authentication

Keycloak identity provider + `keycloak-api` admin/auth helper API + a
weekly password-rotation sidecar for designated accounts.

## Services

| Container | Port | Purpose |
|-----------|------|---------|
| `raptor-keycloak-db` | — | Postgres backing store for Keycloak |
| `raptor-keycloak` | 8080 | Keycloak server (realm `dhtsolution`) |
| `raptor-keycloak-init` | — | One-shot: creates the permanent admin account, then exits |
| `raptor-keycloak-api` | 8800 | FastAPI wrapper around Keycloak's admin/auth APIs |
| `raptor-password-rotator` | — | Weekly password rotation for accounts in `rotate_accounts.txt` |

## Permanent Admin Bootstrap (`keycloak-init`)

Keycloak's own `KC_BOOTSTRAP_ADMIN_USERNAME`/`PASSWORD` only exist until
the first admin login, then Keycloak disables them. `keycloak-init` runs
once after `keycloak` reports healthy, uses that bootstrap account to log
in, and creates a **permanent** master-realm admin
(`PERMANENT_MASTER_ADMIN_USER`/`PERMANENT_ADMIN_PASSWORD`) via
`create-permanent-admin.sh`. `keycloak-api` and `password-rotator` both
wait on this container's `service_completed_successfully` before starting,
since they depend on that admin account existing.

### Required `.env` values

```env
KEYCLOAK_ADMIN_USERNAME=
KEYCLOAK_ADMIN_PASSWORD=
PERMANENT_MASTER_ADMIN_USER=
PERMANENT_ADMIN_PASSWORD=
```

## Password Rotation (`password-rotator`)

Every account listed in `rotate_accounts.txt` gets a new strong random
password on a fixed weekly schedule; the new password is pushed to
Keycloak via the admin API and emailed to the account's contact address.

**No state is persisted.** The next scheduled time (e.g. "next Monday
09:00") is recomputed from the current clock on every loop iteration,
so restarting this container — for example to pick up code changes —
never re-triggers a rotation early or twice. It just resumes waiting
for the same weekly slot.

### Required `.env` values

```env
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
SMTP_FROM=

ROTATE_REALM=dhtsolution
ROTATE_ACCOUNTS=              # fallback only — prefer rotate_accounts.txt (see below)
ROTATE_BCC_EMAIL=             # optional — also BCCs this address on every rotation email
ROTATE_WEEKDAY=0              # 0=Monday ... 6=Sunday
ROTATE_HOUR=9                 # 24h, local TZ (TIMEZONE env var)
ROTATE_MINUTE=0
```

### Account list — `rotate_accounts.txt`

Bind-mounted into the container (`./rotate_accounts.txt` next to this
`docker-compose.yml`) and re-read fresh on every run — **edit the file,
no rebuild or restart needed**, the change is picked up automatically
at the next scheduled rotation.

```
# one "username:email" per line, # for comments
alice:alice@dhtsolution.group
bob:bob@dhtsolution.group
```

The `email` is just where the new password gets sent — it doesn't have
to match the email registered on the Keycloak account.

### Manual one-off run

Useful right after adding a brand-new account, instead of waiting for
the next scheduled weekday:

```bash
# rotate everyone in the list right now
docker exec -e ROTATE_RUN_ONCE=1 raptor-password-rotator \
    python -m app.scripts.rotate_passwords

# rotate just one account, leave everyone else untouched
docker exec -e ROTATE_RUN_ONCE=1 -e ROTATE_ONLY=alice raptor-password-rotator \
    python -m app.scripts.rotate_passwords
```

### Applying a code change

`password-rotator` shares the `keycloak-api` image, which bakes `./app`
in at build time (no volume mount) — editing files on disk inside the
running container is lost on the next rebuild/restart. After changing
`app/scripts/rotate_passwords.py` or anything under `app/`:

```bash
docker compose -f 06-authentication/docker-compose.yml build keycloak-api
docker compose -f 06-authentication/docker-compose.yml up -d
```

`rotate_accounts.txt` is the one exception — it's bind-mounted, so
editing it needs neither a rebuild nor a restart.

## GDPR Consent

Each user's consent per processing purpose lives on their Keycloak account as
user attributes, written only by Module 33 (privacy-compliance-service):

| Attribute | Default for a new user |
|-----------|------------------------|
| `gdpr_consent_memory` | `true` |
| `gdpr_consent_analytics` | `false` |
| `gdpr_consent_benchmark` | `true` |
| `gdpr_consent_voice_recording` | `false` |
| `gdpr_consent_face_detection` | `false` (face detection, seed photos and People Search are denied until granted) |
| `gdpr_consent_voice_clone` | `false` (Module 35 `/voice/clone` is denied until granted) |
| `gdpr_consent_version` | `GD_CONSENT_POLICY_VERSION` (Module 33, default `1.0`) |

Users change them through Module 13's `GET/PUT /api/{v}/gdpr/consent[/{purpose}]`,
never directly: Module 33 also records every change in `gdpr_consent_history`
and deletes a purpose's data when its consent is withdrawn.

### 1. Realm setting: Unmanaged attributes (required)

Keycloak 24+ **silently discards** user attributes that its user profile
doesn't declare. Without this setting none of the attributes above is ever
saved (Module 33 detects this and fails the request with a 500 that points
here).

**Applied automatically** by `app/sync_resources.py` (same startup sync that
already keeps the resource URIs below in sync) -- `keycloak-api`'s own
lifespan runs it, `--apply`, every time the container starts, so restarting
it is enough to fix an existing deployment too, not just a brand-new one.
This setting lives in its own REST sub-resource (`.../users/profile`), not
the realm-import JSON `--import-realm` consumes once at creation -- which is
exactly why it needed its own sync step instead of just being added there.
Still best-effort like the rest of that sync (logged, never fails startup):
the manual steps below are for checking it directly or applying it by hand
if the sync itself can't reach Keycloak. In the admin console:

*Realm settings → General → Unmanaged attributes → **Only administrators can write***

"Only administrators can write" (`ADMIN_EDIT`) — not "Enabled" — so users
can't edit their own consent attributes from the account console, which would
bypass the consent history. The same change from the command line:

```bash
docker exec raptor-keycloak /opt/keycloak/bin/kcadm.sh config credentials \
    --server http://localhost:8080 --realm master \
    --user "$PERMANENT_MASTER_ADMIN_USER" --password "$PERMANENT_ADMIN_PASSWORD"
docker exec raptor-keycloak /opt/keycloak/bin/kcadm.sh get users/profile -r dhtsolution > up.json
# set "unmanagedAttributePolicy": "ADMIN_EDIT" at the top level of up.json, then:
docker exec -i raptor-keycloak /opt/keycloak/bin/kcadm.sh update users/profile -r dhtsolution -f - < up.json
```

### 2. Registration webhook

`keycloak-api` calls Module 33's `POST /internal/gdpr/new-user` (with
`X-Internal-Token: $GD_INTERNAL_TOKEN`) right after a user is fully created by
`POST /SSO/register` or `POST /admin/keycloak/user` — see `app/core/gdpr.py`.
Module 33 then sets the defaults above. The call is best-effort: if it fails,
registration still succeeds, and Module 33 writes the same defaults the first
time that user's consent is read (without `gdpr_consent_version`, so they are
still asked to review the policy).

Users created any other way (e.g. by hand in the admin console) get their
defaults that same first-read way. If Keycloak's own registration page is
ever enabled, add a Keycloak event listener that calls the same endpoint.

```env
GD_INTERNAL_TOKEN=            # same value as the root .env
PRIVACY_COMPLIANCE_URL=http://raptor-privacy-compliance-service:8033
```

### 3. Required action `REVIEW_CONSENT` (setup — not yet built)

When the consent notice changes, `GD_CONSENT_POLICY_VERSION` is bumped and
every user whose `gdpr_consent_version` is older must see the notice at their
next login. Keycloak does this with a **required action** — a step it forces
at login, like *Update Password* — but `REVIEW_CONSENT` is not built in; it is
a custom required-action provider (Java SPI) that has to be built and
installed. Until it is, logins are unaffected and users are simply not shown
the notice.

**What the provider does**

- `RequiredActionFactory.getId()` returns `REVIEW_CONSENT`.
- Reads the current policy version from its provider config (the factory's
  `init(Config.Scope)`), e.g. a `policy-version` option — keep it equal to
  Module 33's `GD_CONSENT_POLICY_VERSION`, and bump both together.
- `evaluateTriggers()` runs at every login: if the user's
  `gdpr_consent_version` attribute is missing or older than the configured
  version (compare numerically, `1.10` > `1.9`), add `REVIEW_CONSENT` to the
  user's required actions.
- `requiredActionChallenge()` renders the consent notice (a FreeMarker
  template) with a link to the consent settings.
- `processAction()` on submit sets `gdpr_consent_version` to the configured
  version and calls `context.success()`.
- It must **not** write any `gdpr_consent_<purpose>` attribute itself:
  purpose changes go through Module 13 → Module 33, which records the
  history and runs the withdrawal deletions.

**Install and enable**

1. Build the provider JAR and mount it into the `keycloak` service's
   `/opt/keycloak/providers/` (add a volume in `docker-compose.yml`), then
   restart Keycloak so it picks the provider up.
2. Pass the `policy-version` provider option (see Keycloak's *Configuring
   providers* guide for the option/env-var syntax of the running version).
3. Admin console: *Authentication → Required actions* → find
   **REVIEW_CONSENT** → switch **Enabled** on. Leave *Set as default action*
   off — `evaluateTriggers()` decides who needs it.
4. Verify: set a test user's `gdpr_consent_version` to `0.9`, log in, and
   check the notice appears and the attribute reads the current version
   afterwards.

## TLS for Keycloak and the gateway

An MCP client on another machine (Claude Code's `/mcp` → Authenticate) cannot complete OAuth against a
plain-HTTP Keycloak: OAuth clients refuse to send the authorization code to a non-loopback `http://`
token endpoint (GitHub #163). This module's own `docker-compose.yml` has an opt-in Compose **profile**,
`tls`, that puts an nginx (`raptor-tls-proxy`) in front of Keycloak and the API gateway to terminate TLS,
and tells Keycloak that clients reach it over https -- entirely within the one file, nothing to stack
with a second `-f`.

| Public address | Goes to |
|---|---|
| `https://<host>:8443` (`PORT_KEYCLOAK_TLS`) | Keycloak (`raptor-keycloak:8080`). `/admin/` and `/realms/master/` answer 403 here: the admin console stays on Keycloak's own port, which `KEYCLOAK_BIND_ADDR` below limits to the host's loopback. |
| `https://<host>:8013` (`PORT_GATEWAY_TLS`) | Module 13 (`raptor-api-gateway:8012`), unbuffered (SSE/MCP streams), no body limit, WebSocket upgrade passed through |

### Turning it on

1. Certificates in `tls/certs/` (or `TLS_CERT_DIR`): `server.pem` (full chain) and `server.key`, for
   the name or IP clients use. See "Certificates" below.
2. Pick the env file. If `deployment/modules/.env` (the root `.env`) exists, use it for everything
   below -- it is the only file `build.py` reads, and the gateway reads it *after* `13-api-services/.env`,
   so its values win over the gateway's own. Without a root `.env`, use this module's `.env` for steps
   3-5 and Module 13's `.env` for step 6.
3. `KEYCLOAK_TLS_ENABLED=true` (root `.env.example`'s own 06 section) -- `build.py` translates this
   into the "tls" Compose profile itself (`_apply_keycloak_tls_mode()`), combined with any other
   module's own profile already active there (e.g. 02's "cluster") rather than replacing it. Deploying
   this module standalone, directly via `docker compose` with no `build.py` in the loop to do that
   translation: set the raw `COMPOSE_PROFILES=tls` instead (this module's own `.env.example`).
   Either way, also `KEYCLOAK_BIND_ADDR=127.0.0.1`, `KC_PROXY_HEADERS=xforwarded`,
   `KC_HOSTNAME_BACKCHANNEL_DYNAMIC=true` -- all commented out by default in both `.env.example` files.
4. `KEYCLOAK_PUBLIC_URL=https://<host>:8443`, the full URL. It becomes Keycloak's own `KC_HOSTNAME`
   (fixing the token issuer) *and* the gateway's `GATEWAY_KEYCLOAK_PUBLIC_URL` -- the gateway just
   copies this one value verbatim, it doesn't rebuild its own URL from it, so there's nothing else to
   set or get wrong here.
5. `GATEWAY_PUBLIC_URL=https://<host>:8013`, then restart the gateway. This one has no equivalent to
   copy from -- fill it in directly, in the same env file as the rest of this section. Check what the
   gateway got with `docker exec raptor-api-gateway-<version> env | grep GATEWAY_.*PUBLIC_URL`.
6. Redeploy Module 06 the normal way -- `./deploy.sh -m 06` (or plain `docker compose up -d` in this
   directory) -- nothing TLS-specific about this step: step 3 is what brings `raptor-tls-proxy` up and
   switches Keycloak's own settings, the same as any other env change, through the same command used
   for anything else. Keycloak restarts, so plan it like any Keycloak restart; sessions survive in the
   database. Unlike a separate TLS compose file, there's nothing a later, unrelated
   `./deploy.sh`/`build.py` redeploy of Module 06 could leave out -- the "tls" profile is already
   inside `docker-compose.yml` and `KEYCLOAK_TLS_ENABLED=true` keeps re-activating it every run.
7. Check that `curl https://<host>:8443/realms/dhtsolution/.well-known/openid-configuration` reports
   the https issuer. From another machine, `claude mcp add --transport http raptor
   https://<host>:8013/api/<version>/mcp --client-id raptor-mcp --callback-port 53682`, then `/mcp` →
   Authenticate, must end with "Got new credentials".

Services on the private network keep calling `http://raptor-keycloak:8080`;
`KC_HOSTNAME_BACKCHANNEL_DYNAMIC` gives them that address back for token and JWKS endpoints. The tokens
they see still carry the https issuer, which the gateway accepts (it fetches JWKS internally).
`KC_PROXY_HEADERS` is only set in this file: without the proxy, Keycloak's port is reached directly
and the headers would come from clients.

The file also moves Keycloak's own port (`PORT_KEYCLOAK`, 8080) to the host's loopback, so other machines
reach Keycloak only through the proxy: on every interface they could bypass https and the `/admin/`
block and send the `X-Forwarded-*` headers Keycloak trusts here. The admin console is
`http://127.0.0.1:8080/admin/` on the host, or through `ssh -L 8080:127.0.0.1:8080 <host>`. This needs
Docker Compose v2.24.4 or later (`ports: !override`). Anything on another machine that called
`http://<host>:8080` must move to `https://<host>:8443`.

The gateway's plain port (`PORT_API_GATEWAY`, 8012) belongs to Module 13's compose file and stays open,
because existing clients still use it. Once they have moved to `https://<host>:8013`, block 8012 for
other machines in the host firewall.

### Certificates

- **A deployment that other machines use**: a certificate from a real CA for a DNS name, e.g. from
  Let's Encrypt or the organisation's CA, so clients need no extra trust. Renew it before it expires.
  nginx reads the files at start, so replace them and run `docker restart raptor-tls-proxy-<version>`.
- **Rehearsals**: `bash tls/make_certs.sh <ip-or-name> ...` makes a throwaway CA and a 30-day server
  certificate in `tls/certs/` (git-ignored). Clients must trust `tls/certs/ca.pem`: Claude Code through
  `NODE_EXTRA_CA_CERTS=/path/ca.pem`, browsers by importing it or accepting the warning.
- The callback of the MCP login is `http://localhost:53682/callback` on the machine running `claude`,
  so the browser must run there too. Over SSH, forward the port from the machine with the browser:
  `ssh -L 53682:localhost:53682 <user>@<machine running claude>`.

Rehearsed on tmp34 on 2026-10-29. Keycloak 26.6.2 and the gateway sat behind this nginx at a LAN
address with a throwaway CA. Claude Code 2.1.312 on a second machine completed `/mcp` → Authenticate
("Got new credentials"), and the gateway accepted the token (JWT and UMA permission).

## Configuration

- **Realm**: `dhtsolution` (imported from `realm-import/dhtsolution-realm.json`)
- **Network**: `raptor` (external)
- **SMTP**: shared `_send()` helper in `app/routers/email_system.py` (STARTTLS, port 587)
