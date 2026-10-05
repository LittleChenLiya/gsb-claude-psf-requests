"""
requests.explain
~~~~~~~~~~~~~~~~

An explicit proxy-selection explainer.

This module answers the question "which proxy would Requests use for this
URL, and why?" *without* sending any request and *without* modifying the
process environment, not even temporarily.

The resolution mirrors the real send path exactly:

* ``Session.request`` / ``Session.merge_environment_settings`` --
  environment proxies fill in keys missing from the per-request
  ``proxies`` mapping, then the result is merged over
  ``Session.proxies`` (``merge_setting`` semantics, where a key
  explicitly set to ``None`` removes the lower-layer value).
* ``Session.rebuild_proxies`` / ``resolve_proxies`` --
  ``no_proxy``/``NO_PROXY`` bypass is evaluated and the environment
  proxy for the URL scheme is filled in when still missing.
* ``select_proxy`` -- the final proxy URL is picked by key precedence:
  ``scheme://host``, ``scheme``, ``all://host``, ``all``.

Because the explainer is pure, an alternative environment mapping can be
supplied via the ``env`` parameter (useful for tests and for asking
"what if" questions); ``os.environ`` is then left completely untouched.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .compat import (
    Mapping,
    proxy_bypass,
    proxy_bypass_environment,
    urlparse,
    urlsplit,
)
from .utils import address_in_network, getproxies, is_ipv4_address, is_valid_cidr

if TYPE_CHECKING:
    from .sessions import Session

__all__ = [
    "ProxyExplanation",
    "ProxyLayer",
    "explain_proxies",
    "explain_proxy_selection",
]


def _redact_proxy_url(url: str) -> str:
    """Return ``url`` with any embedded credentials replaced by ``***``.

    The explainer only ever reports *that* proxy authentication is
    configured, never the configured values.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if parts.username is None and parts.password is None:
        return url
    host = parts.hostname or ""
    if ":" in host and not host.startswith("["):  # IPv6 literal
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError:
        port = None
    netloc = f"***:***@{host}"
    if port is not None:
        netloc += f":{port}"
    redacted = f"{parts.scheme}://{netloc}" if parts.scheme else f"//{netloc}"
    if parts.path:
        redacted += parts.path
    if parts.query:
        redacted += f"?{parts.query}"
    return redacted


def _has_proxy_auth(url: str) -> bool:
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    return parts.username is not None or parts.password is not None


def _getproxies_from_env(env: Mapping[str, str]) -> dict[str, str]:
    """Pure re-implementation of ``urllib.request.getproxies_environment``
    against an explicit mapping instead of ``os.environ``."""
    proxies = {}
    for name, value in env.items():
        name = name.lower()
        if value and name[-6:] == "_proxy":
            proxies[name[:-6]] = value
    return proxies


def _get_env_no_proxy(env: Mapping[str, str]) -> str | None:
    """Return the effective ``no_proxy`` value from a mapping, prioritizing
    the lowercase variant like ``should_bypass_proxies``."""
    return env.get("no_proxy") or env.get("NO_PROXY")


def _match_no_proxy(url: str, no_proxy: str | None) -> str | None:
    """Pure replica of the explicit ``no_proxy`` list matching performed by
    ``requests.utils.should_bypass_proxies``.

    Returns the matching ``no_proxy`` entry, or ``None`` when there is no
    match. Never touches ``os.environ``.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname
    if hostname is None:
        # URLs don't always have hostnames, e.g. file:/// urls; the send
        # path treats those as bypassed.
        return "<no-hostname>"

    if not no_proxy:
        return None

    no_proxy_hosts = (host for host in no_proxy.replace(" ", "").split(",") if host)

    if is_ipv4_address(hostname):
        for proxy_ip in no_proxy_hosts:
            if is_valid_cidr(proxy_ip):
                if address_in_network(hostname, proxy_ip):
                    return proxy_ip
            elif hostname == proxy_ip:
                return proxy_ip
    else:
        host_with_port = hostname
        if parsed.port:
            host_with_port += f":{parsed.port}"

        for host in no_proxy_hosts:
            host = host.lstrip(".")
            if hostname == host or host_with_port == host:
                return host
            host = "." + host
            if hostname.endswith(host) or host_with_port.endswith(host):
                return host.lstrip(".")
    return None


def _system_bypass(hostname: str, env: Mapping[str, str], real_env: bool) -> bool:
    """Evaluate the OS-level proxy bypass fallback without mutating the
    process environment.

    The send path wraps ``proxy_bypass`` so that the process ``no_proxy``
    is masked while it runs; we reproduce that by passing an explicit
    proxies mapping (without the ``no`` key) to ``proxy_bypass_environment``
    and by only consulting OS-level settings (registry / system
    configuration) when explaining against the real environment.
    """
    env_proxies = _getproxies_from_env(env)
    if env_proxies:
        # Environment proxies are set: the send path delegates to
        # proxy_bypass_environment with no_proxy masked out, which always
        # yields False. Reproduce that deterministically.
        proxies = {k: v for k, v in env_proxies.items() if k != "no"}
        try:
            return bool(proxy_bypass_environment(hostname, proxies))
        except Exception:  # never fail an explanation
            return False
    if real_env:
        # No environment proxies: the send path consults OS-level settings
        # (e.g. the Windows registry or macOS system configuration). Those
        # reads do not modify the environment.
        try:
            return bool(proxy_bypass(hostname))
        except Exception:  # never fail an explanation
            return False
    # A mock environment was supplied: stay deterministic and do not leak
    # host OS proxy settings into the explanation.
    return False


def _bypass_check(
    url: str,
    no_proxy: str | None,
    env: Mapping[str, str],
    real_env: bool,
) -> tuple[bool, str | None, str | None]:
    """Return ``(bypassed, matched_entry, no_proxy_source)`` for ``url``.

    ``no_proxy`` is the explicit (request/session layer) value, which takes
    precedence over the environment's, mirroring
    ``requests.utils.should_bypass_proxies``.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname
    if hostname is None:
        return True, "<no-hostname>", None

    effective = no_proxy
    source = "request/session"
    if effective is None:
        effective = _get_env_no_proxy(env)
        source = "environment"

    match = _match_no_proxy(url, effective)
    if match is not None:
        return True, match, source if effective else None

    if _system_bypass(hostname, env, real_env):
        return True, "<system>", "system"
    return False, None, None


@dataclass
class ProxyLayer:
    """Description of one proxy configuration layer."""

    name: str  # "request", "session" or "environment"
    provided: bool  # False when the layer was simply not configured (missing)
    entries: dict[str, str] = field(default_factory=dict)  # redacted view
    cleared: tuple[str, ...] = ()  # keys explicitly set to None (override)

    def describe(self) -> str:
        if not self.provided:
            return f"{self.name}: not provided"
        parts = [f"{key}={value}" for key, value in sorted(self.entries.items())]
        parts += [f"{key}=None (clears lower layers)" for key in sorted(self.cleared)]
        if not parts:
            return f"{self.name}: explicitly empty"
        return f"{self.name}: " + ", ".join(parts)


@dataclass
class ProxyExplanation:
    """The explained result of proxy selection for a URL.

    This is a pure description: producing it sent no request and did not
    modify the process environment.
    """

    url: str
    trust_env: bool
    layers: tuple[ProxyLayer, ...]
    #: Redacted proxy URL that would be used, or ``None`` for a direct
    #: connection. Credentials are never exposed; ``***:***@`` is shown
    #: instead and ``auth_configured`` flags their presence.
    final_proxy: str | None
    #: Which layer the final proxy came from: "request", "session",
    #: "environment" or ``None`` for a direct connection.
    source: str | None
    #: The proxies-mapping key that matched (e.g. ``"https"``, ``"all"``,
    #: ``"https://example.org"``).
    matched_key: str | None
    #: True when the URL is reached directly (no proxy selected).
    direct: bool
    #: True when a no_proxy / system bypass rule matched the URL.
    bypassed: bool
    #: The no_proxy entry that matched, if any.
    bypass_match: str | None
    #: Where the effective no_proxy setting came from.
    no_proxy_source: str | None
    #: True when the selected proxy URL carries credentials (values hidden).
    auth_configured: bool

    def explain(self) -> str:
        """Render a human-readable explanation of the selection."""
        lines = [
            f"Proxy selection for {self.url!r} (explanation only; no request was sent)",
            f"  trust_env: {self.trust_env}",
        ]
        for layer in self.layers:
            lines.append(f"  {layer.describe()}")
        if self.bypassed:
            match = self.bypass_match
            source = f" from {self.no_proxy_source}" if self.no_proxy_source else ""
            lines.append(f"  no_proxy: entry {match!r}{source} matched this URL")
        if self.direct:
            reason = (
                " (bypassed by no_proxy)" if self.bypassed else " (no proxy configured)"
            )
            lines.append(f"Result: direct connection{reason}")
        else:
            auth = (
                "; proxy authentication configured (credentials hidden)"
                if self.auth_configured
                else ""
            )
            lines.append(
                f"Result: proxy {self.final_proxy} "
                f"(from {self.source}, key {self.matched_key!r}{auth})"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.explain()


def explain_proxy_selection(
    url: str,
    request_proxies: Mapping[str, str | None] | None = None,
    session_proxies: Mapping[str, str | None] | None = None,
    trust_env: bool = True,
    env: Mapping[str, str] | None = None,
) -> ProxyExplanation:
    """Explain which proxy would be selected for ``url`` and why.

    Follows the exact resolution order of the ``Session`` send path, but
    performs no I/O and never modifies ``os.environ``.

    :param url: The URL a request would be sent to.
    :param request_proxies: The per-request ``proxies`` mapping, or ``None``
        when the request would not set one. ``None`` (missing) is reported
        distinctly from an explicitly empty mapping.
    :param session_proxies: The session-level proxies (``Session.proxies``).
    :param trust_env: Mirrors ``Session.trust_env``.
    :param env: Environment mapping to explain against. Defaults to the real
        ``os.environ`` (read-only). Pass an explicit mapping to explain a
        hypothetical/mock environment deterministically.
    :rtype: ProxyExplanation
    """
    real_env = env is None
    environ = os.environ if env is None else env

    # Environment proxies, exactly as get_environ_proxies() would see them
    # (including OS-level proxy settings when explaining the real env).
    if real_env:
        env_proxies = getproxies()
    else:
        env_proxies = _getproxies_from_env(environ)

    request_provided = request_proxies is not None
    req = dict(request_proxies) if request_proxies is not None else {}
    sess = dict(session_proxies) if session_proxies is not None else {}

    # -- Step 1: merge_environment_settings (per-request layer) -----------
    # Environment proxies fill keys missing from the request mapping, gated
    # on the bypass check with the *request-level* no_proxy.
    effective = {k: (v, "request") for k, v in req.items()}
    bypassed = False
    bypass_match = None
    no_proxy_source = None
    if trust_env:
        req_no_proxy = req.get("no_proxy") if request_provided else None
        bypassed, bypass_match, no_proxy_source = _bypass_check(
            url, req_no_proxy, environ, real_env
        )
        if not bypassed:
            for k, v in env_proxies.items():
                effective.setdefault(k, (v, "environment"))

    # -- Step 2: merge_setting(request, session) --------------------------
    merged = {k: (v, "session") for k, v in sess.items()}
    merged.update(effective)
    cleared = tuple(sorted(k for k, (v, _s) in merged.items() if v is None))
    for key in cleared:
        del merged[key]

    # -- Step 3: resolve_proxies (rebuild_proxies on send) ----------------
    # The merged no_proxy (request over session, else environment) gates
    # filling in the environment proxy for this URL's scheme.
    if trust_env:
        merged_no_proxy, merged_no_proxy_layer = merged.get("no_proxy", (None, None))
        bypassed2, match2, source2 = _bypass_check(
            url, merged_no_proxy, environ, real_env
        )
        if bypassed2:
            bypassed, bypass_match = True, match2
            no_proxy_source = (
                merged_no_proxy_layer if merged_no_proxy is not None else source2
            )
        if not bypassed2:
            scheme = urlparse(url).scheme
            env_proxy = env_proxies.get(scheme, env_proxies.get("all"))
            if env_proxy:
                merged.setdefault(scheme, (env_proxy, "environment"))

    # -- Step 4: select_proxy ---------------------------------------------
    urlparts = urlparse(url)
    if urlparts.hostname is None:
        key_order = [urlparts.scheme, "all"]
    else:
        key_order = [
            f"{urlparts.scheme}://{urlparts.hostname}",
            urlparts.scheme,
            f"all://{urlparts.hostname}",
            "all",
        ]
    final_value = None
    final_source = None
    matched_key = None
    for key in key_order:
        if key in merged and merged[key][0] is not None:
            final_value, final_source = merged[key]
            matched_key = key
            break

    direct = final_value is None
    auth_configured = bool(final_value) and _has_proxy_auth(final_value)

    def _layer(name, provided, mapping):
        entries = {
            k: _redact_proxy_url(v) if isinstance(v, str) else repr(v)
            for k, v in mapping.items()
            if v is not None
        }
        layer_cleared = tuple(sorted(k for k, v in mapping.items() if v is None))
        return ProxyLayer(
            name=name, provided=provided, entries=entries, cleared=layer_cleared
        )

    env_entries = {k: v for k, v in env_proxies.items() if k != "no"}
    if trust_env:
        env_no_proxy = _get_env_no_proxy(environ)
        if env_no_proxy:
            env_entries["no_proxy"] = env_no_proxy
    layers = (
        _layer("request", request_provided, req),
        _layer("session", session_proxies is not None and bool(sess), sess),
        ProxyLayer(
            name="environment",
            provided=trust_env and bool(env_entries),
            entries=(
                {k: _redact_proxy_url(v) for k, v in env_entries.items()}
                if trust_env
                else {}
            ),
        ),
    )

    return ProxyExplanation(
        url=url,
        trust_env=trust_env,
        layers=layers,
        final_proxy=None if direct else _redact_proxy_url(final_value),
        source=None if direct else final_source,
        matched_key=matched_key,
        direct=direct,
        bypassed=bypassed,
        bypass_match=bypass_match,
        no_proxy_source=no_proxy_source,
        auth_configured=auth_configured,
    )


def explain_proxies(
    url: str,
    proxies: Mapping[str, str | None] | None = None,
    *,
    session: Session | None = None,
    trust_env: bool = True,
    env: Mapping[str, str] | None = None,
) -> ProxyExplanation:
    """Explain proxy selection for ``url`` without sending a request.

    :param url: The URL a request would be sent to.
    :param proxies: Optional per-request proxies mapping.
    :param session: Optional ``requests.Session`` whose ``proxies`` and
        ``trust_env`` should be taken into account.
    :param trust_env: Whether environment configuration is trusted (ignored
        when ``session`` is given; the session's own value is used).
    :param env: Optional environment mapping to explain against instead of
        the real ``os.environ`` (which is only ever read, never modified).
    :rtype: ProxyExplanation

    Usage::

        >>> import requests
        >>> s = requests.Session()
        >>> s.proxies["http"] = "http://10.10.1.10:3128"
        >>> print(requests.explain_proxies("http://example.org/", session=s))
    """
    session_proxies = None
    if session is not None:
        session_proxies = session.proxies
        trust_env = session.trust_env
    return explain_proxy_selection(
        url,
        request_proxies=proxies,
        session_proxies=session_proxies,
        trust_env=trust_env,
        env=env,
    )


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: explain proxy selection for a URL.

    Usage: ``python -m requests.explain <url>``
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m requests.explain",
        description=(
            "Explain which proxy Requests would use for a URL, and why. "
            "No request is sent and the process environment is never modified."
        ),
    )
    parser.add_argument("url", help="the URL a request would be sent to")
    args = parser.parse_args(argv)
    print(explain_proxies(args.url).explain())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
