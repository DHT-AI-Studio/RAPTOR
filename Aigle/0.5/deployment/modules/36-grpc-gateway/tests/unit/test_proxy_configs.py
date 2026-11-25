"""The reverse-proxy examples in proxy-test/ must not break the RPCs they front.

Found by running the nginx example for real: UploadAsset over 1 MiB came back as HTTP 413.
nginx counts the messages of a client-streaming gRPC call toward `client_max_body_size`
(default 1 MiB), so the example needs it switched off; the gateway enforces its own limits.
"""
from __future__ import annotations

import re
from pathlib import Path

_PROXY_TEST = Path(__file__).parent.parent.parent / "proxy-test"


def _nginx() -> str:
    return (_PROXY_TEST / "nginx.conf").read_text()


def test_nginx_does_not_cap_the_size_of_a_client_streaming_upload():
    conf = _nginx()
    assert re.search(r"^\s*client_max_body_size\s+0\s*;", conf, re.MULTILINE), (
        "nginx's default 1 MiB client_max_body_size cuts UploadAsset off with HTTP 413"
    )


def test_nginx_proxies_grpc_over_http2():
    conf = _nginx()
    assert re.search(r"^\s*listen\s+\d+\s+http2\s*;", conf, re.MULTILINE)
    assert re.search(r"^\s*grpc_pass\s+grpc://", conf, re.MULTILINE)


def test_nginx_timeouts_allow_long_streams():
    conf = _nginx()
    for directive in ("grpc_read_timeout", "grpc_send_timeout", "client_body_timeout"):
        match = re.search(rf"^\s*{directive}\s+(\d+)s\s*;", conf, re.MULTILINE)
        assert match and int(match.group(1)) >= 3600, directive


def test_envoy_speaks_http2_on_the_listener_and_to_the_upstream():
    conf = (_PROXY_TEST / "envoy.yaml").read_text()
    # Envoy talks HTTP/1.1 to the backend unless told otherwise, which silently breaks every call.
    assert len(re.findall(r"http2_protocol_options|explicit_http_config", conf)) >= 2


# ── the TLS-terminating variants ────────────────────────────────────────────

def test_nginx_tls_terminates_tls_with_http2_and_keeps_the_grpc_settings():
    conf = (_PROXY_TEST / "nginx-tls.conf").read_text()
    assert re.search(r"^\s*listen\s+\d+\s+ssl\s+http2\s*;", conf, re.MULTILINE), "gRPC over TLS needs `ssl http2`"
    assert re.search(r"^\s*ssl_certificate\s+\S+;", conf, re.MULTILINE)
    assert re.search(r"^\s*ssl_certificate_key\s+\S+;", conf, re.MULTILINE)
    assert re.search(r"^\s*grpc_pass\s+grpc://", conf, re.MULTILINE)
    assert re.search(r"^\s*client_max_body_size\s+0\s*;", conf, re.MULTILINE), "uploads over 1 MiB would fail with 413"
    for directive in ("grpc_read_timeout", "grpc_send_timeout", "client_body_timeout"):
        match = re.search(rf"^\s*{directive}\s+(\d+)s\s*;", conf, re.MULTILINE)
        assert match and int(match.group(1)) >= 3600, directive


def test_envoy_tls_terminates_tls_and_negotiates_h2():
    conf = (_PROXY_TEST / "envoy-tls.yaml").read_text()
    assert "DownstreamTlsContext" in conf
    assert re.search(r'alpn_protocols:\s*\["h2"\]', conf), "without h2 in ALPN a gRPC client cannot use the listener"
    assert "certificate_chain" in conf
    assert "private_key" in conf
    assert len(re.findall(r"http2_protocol_options|explicit_http_config", conf)) >= 2
    assert re.search(r"timeout:\s*3600s", conf)


def test_the_tls_proxies_use_their_own_ports_and_the_compose_file_wires_them():
    compose = (Path(__file__).parent.parent.parent / "docker-compose.proxy-test.yml").read_text()
    for service, port in (("nginx-tls-proxy-test", "8828"), ("envoy-tls-proxy-test", "8829")):
        assert service in compose
        assert f'"{port}:{port}"' in compose
    assert "./proxy-test/certs:/certs:ro" in compose
    assert (_PROXY_TEST / "make_certs.sh").exists()


def test_the_throwaway_certificates_are_never_committed():
    gitignore = (Path(__file__).parent.parent.parent / ".gitignore").read_text()
    assert "proxy-test/certs/" in gitignore
