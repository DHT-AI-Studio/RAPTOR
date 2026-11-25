"""
The TLS entry (CF-3a) restricts Keycloak's own port to the host's loopback only, once
KEYCLOAK_BIND_ADDR is set -- otherwise (the plain-HTTP default) it's published on every interface, as
before. This used to be a separate docker-compose.tls.yml overriding the base file's `ports:` with
`!override`; it's now one `ports:` entry in docker-compose.yml itself, driven by that one variable, so
there's no second file's binding to drift out of sync with the base file's.

The merged result was checked with `docker compose config`; CI has no docker, so this reads the file.
PyYAML is not a dependency of this module, hence the plain text parsing.

Run from the 06-authentication/ directory:
    python -m pytest tests/unit/test_tls_compose_ports.py -v
"""
import os
import re

MODULE = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def _service(path, name):
    """The lines of one service of a compose file, by its two-space indentation."""
    with open(os.path.join(MODULE, path)) as f:
        lines = f.read().splitlines()
    start = lines.index(f"  {name}:")
    end = next((i for i in range(start + 1, len(lines)) if re.match(r"  \S", lines[i])), len(lines))
    return lines[start:end]


def _ports(block):
    """(key line, published entries) of a service's ports: list."""
    i = next(i for i, line in enumerate(block) if re.match(r"    ports:", line))
    entries = []
    for line in block[i + 1:]:
        m = re.match(r'\s+- "([^"]+)"', line)
        if not m:
            break
        entries.append(m.group(1))
    return block[i].strip(), entries


def test_keycloak_port_is_driven_by_keycloak_bind_addr_not_a_second_file():
    # ${VAR:+VAR:}PORT -- empty KEYCLOAK_BIND_ADDR (default) resolves to just "PORT:8080" (every
    # interface, the plain-HTTP behaviour); a non-empty one prefixes "host:PORT:8080" (loopback once
    # the TLS entry sets it to 127.0.0.1). See test_keycloak_tls_mode.py for the resolved values.
    key, entries = _ports(_service("docker-compose.yml", "keycloak"))
    assert key == "ports:"
    assert entries == ["${KEYCLOAK_BIND_ADDR:+${KEYCLOAK_BIND_ADDR}:}${PORT_KEYCLOAK:-8080}:8080"]


def test_no_separate_tls_compose_file_exists():
    # Regression guard: the TLS entry is the "tls" Compose profile inside docker-compose.yml itself
    # (tls-proxy service + KEYCLOAK_BIND_ADDR/KC_PROXY_HEADERS/KC_HOSTNAME_BACKCHANNEL_DYNAMIC above),
    # not a second file stacked with `-f` -- nothing should ever re-add docker-compose.tls.yml.
    assert not os.path.exists(os.path.join(MODULE, "docker-compose.tls.yml"))


def test_tls_proxy_service_is_profile_gated():
    block = _service("docker-compose.yml", "tls-proxy")
    i = next(i for i, line in enumerate(block) if line.strip() == "profiles: [\"tls\"]")
    assert block[i].strip() == 'profiles: ["tls"]'


def test_keycloak_proxy_settings_default_off():
    block = _service("docker-compose.yml", "keycloak")
    joined = "\n".join(block)
    # Bare (no "=value") on purpose -- confirmed live against the real image, an *empty*
    # KC_PROXY_HEADERS is an invalid enum value Keycloak refuses to start with, unlike a truly
    # absent one; this is Compose's one construct that omits the key rather than passing it blank.
    assert re.search(r"^\s+- KC_PROXY_HEADERS\s*$", joined, re.MULTILINE)
    assert "- KC_HOSTNAME_BACKCHANNEL_DYNAMIC=${KC_HOSTNAME_BACKCHANNEL_DYNAMIC:-false}" in joined
