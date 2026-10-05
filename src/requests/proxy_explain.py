"""
requests.proxy_explain
~~~~~~~~~~~~~~~~~~~~~~

Explain which proxy would be selected for a request *without* sending it.

This module mirrors the proxy resolution performed by
:meth:`requests.Session.request` (and the adapter's
:func:`~requests.utils.select_proxy`) so that the reported selection is
consistent with the one the real send path would make.  It is intentionally
side-effect free:

* no network request is sent, and
* the process environment is never mutated (the ``no_proxy`` matching is a
  pure function of the environment mapping it is given).

The main entry point is :func:`explain_proxy`.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse, urlunparse

from .compat import getproxies_environment
from .utils import (
    address_in_network,
    is_ipv4_address,
    is_valid_cidr,
)

try:  # ``proxy_bypass_registry`` only exists on Windows in requests.utils.
    from .utils import proxy_bypass_registry
except ImportError:  # pragma: no cover - non-Windows
    proxy_bypass_registry = None  # type: ignore[assignment]


__all__ = ("ProxyExplanation", "explain_proxy")

# Layer labels used in :class:`ProxyExplanation`.
LAYER_REQUEST = "request"
LAYER_SESSION = "session"
LAYER_ENVIRONMENT = "environment"


def _redact_proxy(proxy: str | None) -> tuple[str | None, bool]:
    """Remove credentials from a proxy URL.

    Returns a tuple of ``(redacted_url, has_auth)`` where ``has_auth`` is
    ``True`` when the original URL embedded a username and/or password.
    """
    if proxy is None:
        return None, False
    parsed = urlparse(proxy)
    has_auth = parsed.username is not None or parsed.password is not None
    if not has_auth:
        return proxy, False
    host = parsed.hostname or ""
    if parsed.port is not None:
        host += f":{parsed.port}"
    netloc = f"***@{host}"
    redacted = urlunparse(
        (
            parsed.scheme,
            netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )
    return redacted, True


def _proxy_bypass_environment(host: str, no_proxy: str) -> bool:
    """Pure replica of :func:`urllib.request.proxy_bypass_environment`.

    The standard library function's second parameter changed shape across
    Python versions (a ``no_proxy`` string on older releases, a ``proxies``
    mapping on newer ones).  Keeping our own small replica lets the explainer
    behave identically on every supported version without touching the
    environment.
    """
    if no_proxy == "*":
        return True
    host = host.lower()
    # ``host`` is a hostname (no port) in our usage, but strip a port if
    # one is present to mirror the standard library function.
    hostonly = host.rsplit(":", 1)[0] if ":" in host else host
    for name in no_proxy.split(","):
        name = name.strip()
        if not name:
            continue
        name = name.lstrip(".").lower()
        if hostonly == name or host == name:
            return True
        name = "." + name
        if hostonly.endswith(name) or host.endswith(name):
            return True
    return False


def _should_bypass_proxies(
    url: str, no_proxy: str | None, env: Mapping[str, str]
) -> tuple[bool, str | None]:
    """Pure re-implementation of :func:`requests.utils.should_bypass_proxies`.

    Unlike the original, this never temporarily sets ``os.environ``; the
    ``no_proxy`` value is read from *env* and passed straight to the
    platform-specific bypass check.

    Returns ``(bypassed, effective_no_proxy)``.
    """

    def get_proxy(key: str) -> str | None:
        return env.get(key) or env.get(key.upper())

    no_proxy_arg = no_proxy
    if no_proxy is None:
        no_proxy = get_proxy("no_proxy")
    parsed = urlparse(url)
    hostname = parsed.hostname

    if hostname is None:
        # URLs don't always have hostnames, e.g. file:/// urls.
        return True, no_proxy

    if no_proxy:
        no_proxy_hosts = (host for host in no_proxy.replace(" ", "").split(",") if host)

        if is_ipv4_address(hostname):
            for proxy_ip in no_proxy_hosts:
                if is_valid_cidr(proxy_ip):
                    if address_in_network(hostname, proxy_ip):
                        return True, no_proxy
                elif hostname == proxy_ip:
                    return True, no_proxy
        else:
            host_with_port = hostname
            if parsed.port:
                host_with_port += f":{parsed.port}"

            for host in no_proxy_hosts:
                host = host.lstrip(".")
                if hostname == host or host_with_port == host:
                    return True, no_proxy
                host = "." + host
                if hostname.endswith(host) or host_with_port.endswith(host):
                    return True, no_proxy

    # Platform-specific fallback, computed without mutating the environment.
    effective_no_proxy = (
        no_proxy_arg
        if no_proxy_arg is not None
        else (env.get("no_proxy") or env.get("NO_PROXY"))
    )
    try:
        if sys.platform == "win32":
            # Mirror requests.utils.proxy_bypass: environment proxies take
            # precedence, otherwise fall back to the Windows registry.
            if getproxies_environment():
                bypass = _proxy_bypass_environment(hostname, effective_no_proxy or "")
            else:
                bypass = (
                    proxy_bypass_registry(hostname)
                    if proxy_bypass_registry is not None
                    else False
                )
        else:
            bypass = _proxy_bypass_environment(hostname, effective_no_proxy or "")
    except (TypeError, socket.gaierror):
        bypass = False

    return bool(bypass), no_proxy


def _getproxies_from_env(env: Mapping[str, str]) -> dict[str, str]:
    """Pure replica of :func:`urllib.request.getproxies_environment`.

    Used when the caller supplies an explicit environment mapping so that
    mock environments (and tests) behave exactly like the real one.
    """
    proxies: dict[str, str] = {}
    environment: list[tuple[str, str, str]] = []
    for name in env:
        if len(name) > 5 and name[-6] == "_" and name[-5:].lower() == "proxy":
            value = env[name]
            proxy_name = name[:-6].lower()
            environment.append((name, value, proxy_name))
            if value:
                proxies[proxy_name] = value
    # CVE-2016-1000110: drop CGI HTTP_PROXY when REQUEST_METHOD is set.
    if "REQUEST_METHOD" in env:
        proxies.pop("http", None)
    for name, value, proxy_name in environment:
        if name[-6:] == "_proxy":
            if value:
                proxies[proxy_name] = value
            else:
                proxies.pop(proxy_name, None)
    return proxies


def _redact_layer(layer: Mapping[str, str | None]) -> dict[str, str | None]:
    """Return a copy of *layer* with every proxy URL redacted."""
    redacted: dict[str, str | None] = {}
    for key, value in layer.items():
        redacted[key], _ = _redact_proxy(value)
    return redacted


def _matched_proxy_key(url: str, proxies: Mapping[str, str | None]) -> str | None:
    """Return the key that :func:`select_proxy` would pick."""
    urlparts = urlparse(url)
    if urlparts.hostname is None:
        scheme = urlparts.scheme
        if scheme in proxies:
            return scheme
        if "all" in proxies:
            return "all"
        return None
    proxy_keys = [
        urlparts.scheme + "://" + urlparts.hostname,
        urlparts.scheme,
        "all://" + urlparts.hostname,
        "all",
    ]
    for proxy_key in proxy_keys:
        if proxy_key in proxies:
            return proxy_key
    return None


@dataclass(frozen=True)
class ProxyExplanation:
    """The result of :func:`explain_proxy`.

    Attributes:
        url: The URL the explanation was produced for.
        proxy: The proxy URL that would be used (credentials redacted), or
            ``None`` for a direct connection.
        proxy_source: Which layer provided the proxy -- ``"request"``,
            ``"session"`` or ``"environment"``.  ``None`` when connecting
            directly.
        proxy_auth: ``True`` if the selected proxy URL embeds credentials.
        bypassed: ``True`` if ``no_proxy`` excludes the target host.
        no_proxy: The effective ``no_proxy`` value used for the bypass check,
            or ``None`` when none was configured.
        no_proxy_source: Which layer provided ``no_proxy`` -- ``"request"``,
            ``"session"`` or ``"environment"``.
        trust_env: Whether environment proxy configuration was trusted.
        proxies: The effective merged proxy mapping (credentials redacted).
        layers: The per-layer proxy mappings (credentials redacted).
        request_proxies_missing: ``True`` when no request-level proxies were
            supplied (``None``), as opposed to an explicit empty mapping.
        session_proxies_missing: ``True`` when no session-level proxies were
            supplied.
    """

    url: str
    proxy: str | None
    proxy_source: str | None
    proxy_auth: bool
    bypassed: bool
    no_proxy: str | None
    no_proxy_source: str | None
    trust_env: bool
    proxies: Mapping[str, str | None]
    layers: Mapping[str, Mapping[str, str | None]]
    request_proxies_missing: bool
    session_proxies_missing: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the explanation."""
        return {
            "url": self.url,
            "proxy": self.proxy,
            "proxy_source": self.proxy_source,
            "proxy_auth": self.proxy_auth,
            "bypassed": self.bypassed,
            "no_proxy": self.no_proxy,
            "no_proxy_source": self.no_proxy_source,
            "trust_env": self.trust_env,
            "proxies": dict(self.proxies),
            "layers": {layer: dict(values) for layer, values in self.layers.items()},
            "request_proxies_missing": self.request_proxies_missing,
            "session_proxies_missing": self.session_proxies_missing,
        }

    def __str__(self) -> str:
        lines = [f"Proxy selection for {self.url}"]
        if self.proxy is None:
            lines.append("  Final proxy: (direct connection)")
        else:
            lines.append(f"  Final proxy: {self.proxy} (from {self.proxy_source})")
        lines.append(
            f"  Proxy auth: {'configured (credentials hidden)' if self.proxy_auth else 'not configured'}"
        )
        lines.append(f"  Bypassed by no_proxy: {'yes' if self.bypassed else 'no'}")
        if self.no_proxy is None:
            lines.append("  no_proxy: not set")
        else:
            lines.append(f"  no_proxy: {self.no_proxy} (from {self.no_proxy_source})")
        lines.append(f"  trust_env: {'yes' if self.trust_env else 'no'}")

        lines.append("  Effective proxies:")
        if not self.proxies:
            lines.append("    (none)")
        else:
            for key, value in self.proxies.items():
                lines.append(f"    {key:<8} {value}")

        lines.append("  Configured layers:")
        for layer in (LAYER_REQUEST, LAYER_SESSION, LAYER_ENVIRONMENT):
            values = self.layers.get(layer, {})
            if values:
                rendered = ", ".join(f"{k}={v}" for k, v in values.items())
                lines.append(f"    {layer:<12} {rendered}")
            else:
                lines.append(f"    {layer:<12} (none)")
        return "\n".join(lines)


def explain_proxy(
    url: str,
    proxies: Mapping[str, str | None] | None = None,
    session: Any | None = None,
    *,
    session_proxies: Mapping[str, str | None] | None = None,
    trust_env: bool | None = None,
    env: Mapping[str, str] | None = None,
) -> ProxyExplanation:
    """Explain which proxy would be used for *url* without sending a request.

    This mirrors the proxy resolution of :meth:`requests.Session.request`:
    request-level proxies take precedence over environment proxies, which in
    turn fill keys not provided by the session.  ``no_proxy`` excludes the
    target host before any environment proxy is applied.

    :param url: The URL to explain proxy selection for.
    :param proxies: Request-level proxies (the ``proxies`` argument of
        :meth:`~requests.Session.request`).  ``None`` means the layer was not
        supplied; an explicit empty mapping is treated as a separate, empty
        layer.
    :param session: An optional :class:`requests.Session` whose ``proxies``
        and ``trust_env`` are used.
    :param session_proxies: Explicit session-level proxies.  Overrides
        ``session.proxies`` when both are given.
    :param trust_env: Whether to trust environment proxy configuration.
        Overrides ``session.trust_env`` when given.
    :param env: An optional environment mapping to use instead of
        ``os.environ``.  Useful for deterministic tests and mock
        environments.

    :rtype: ProxyExplanation
    """
    if session is not None:
        if session_proxies is None:
            session_proxies = getattr(session, "proxies", None)
        if trust_env is None:
            trust_env = getattr(session, "trust_env", None)
    if trust_env is None:
        trust_env = True

    if env is None:
        environment: Mapping[str, str] = os.environ
        env_proxies_raw = getproxies_environment()
    else:
        environment = env
        env_proxies_raw = _getproxies_from_env(env)

    request_missing = proxies is None
    session_missing = session_proxies is None

    req: dict[str, str | None] = dict(proxies) if proxies is not None else {}
    sess: dict[str, str | None] = (
        dict(session_proxies) if session_proxies is not None else {}
    )

    # no_proxy from the request layer (mirrors merge_environment_settings).
    no_proxy_req = req.get("no_proxy")
    bypassed, no_proxy_effective = _should_bypass_proxies(
        url, no_proxy_req, environment
    )

    env_layer: dict[str, str] = {}
    if trust_env and not bypassed:
        env_layer = dict(env_proxies_raw)

    # Build the effective merged mapping while tracking the origin of each
    # key.  The order mirrors merge_environment_settings + merge_setting:
    # request-explicit keys win, environment keys fill the request's missing
    # keys, and session keys fill whatever is left.  A key explicitly set to
    # None on the request layer blocks both environment and session.
    req_keys = set(req.keys())
    req_none_keys = {key for key, value in req.items() if value is None}

    merged: dict[str, str | None] = {}
    provenance: dict[str, str] = {}

    for key, value in req.items():
        if value is not None:
            merged[key] = value
            provenance[key] = LAYER_REQUEST

    for key, value in env_layer.items():
        if key in req_keys or key in req_none_keys:
            continue
        if key not in merged:
            merged[key] = value
            provenance[key] = LAYER_ENVIRONMENT

    for key, value in sess.items():
        if value is None:
            continue
        if key in req_none_keys:
            continue
        if key not in merged:
            merged[key] = value
            provenance[key] = LAYER_SESSION

    selected_key = _matched_proxy_key(url, merged)
    selected = merged[selected_key] if selected_key is not None else None
    source = provenance.get(selected_key) if selected_key is not None else None

    redacted, has_auth = _redact_proxy(selected)

    # Determine where no_proxy came from.
    no_proxy_source: str | None = None
    if no_proxy_effective is not None:
        if no_proxy_req is not None:
            no_proxy_source = LAYER_REQUEST
        elif environment.get("no_proxy") or environment.get("NO_PROXY"):
            no_proxy_source = LAYER_ENVIRONMENT

    layers = {
        LAYER_REQUEST: _redact_layer(req),
        LAYER_SESSION: _redact_layer(sess),
        LAYER_ENVIRONMENT: _redact_layer(env_layer),
    }

    return ProxyExplanation(
        url=url,
        proxy=redacted,
        proxy_source=source,
        proxy_auth=has_auth,
        bypassed=bypassed,
        no_proxy=no_proxy_effective,
        no_proxy_source=no_proxy_source,
        trust_env=trust_env,
        proxies=_redact_layer(merged),
        layers=layers,
        request_proxies_missing=request_missing,
        session_proxies_missing=session_missing,
    )


def main() -> None:
    """Command-line entry point: ``python -m requests.proxy_explain <url>``."""
    parser = argparse.ArgumentParser(
        prog="python -m requests.proxy_explain",
        description=(
            "Explain which proxy would be used for a URL without sending a "
            "request or modifying the environment."
        ),
    )
    parser.add_argument("url", help="The URL to explain proxy selection for.")
    parser.add_argument(
        "--proxies",
        metavar="JSON",
        help=(
            "Request-level proxies as a JSON object, e.g. "
            '\'{"http": "http://proxy:8080"}\'.'
        ),
    )
    parser.add_argument(
        "--session-proxies",
        metavar="JSON",
        help="Session-level proxies as a JSON object.",
    )
    parser.add_argument(
        "--no-trust-env",
        action="store_true",
        help="Do not trust environment proxy configuration.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit the explanation as JSON.",
    )
    args = parser.parse_args()

    proxies = json.loads(args.proxies) if args.proxies else None
    session_proxies = json.loads(args.session_proxies) if args.session_proxies else None

    explanation = explain_proxy(
        args.url,
        proxies=proxies,
        session_proxies=session_proxies,
        trust_env=not args.no_trust_env,
    )

    if args.as_json:
        print(json.dumps(explanation.to_dict(), indent=2, sort_keys=True))
    else:
        print(str(explanation))


if __name__ == "__main__":
    main()
