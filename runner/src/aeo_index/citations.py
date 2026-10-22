"""Reading the sources an answer cited: the domain of a citation, whether a
domain is a brand's own, and which citations are web URLs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final
from urllib.parse import urlsplit

_CITATION_SCHEMES: Final[frozenset[str]] = frozenset({"", "http", "https"})
_URL_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})


def domain_from_url(url: str | None) -> str | None:
    """The bare lowercase host of a URL or `host/path` string, with one
    leading `www.` stripped; None when there is no host with a dot."""
    if not url or not url.strip():
        return None
    s = url.strip()
    if "//" not in s:
        s = "//" + s
    try:
        host = urlsplit(s).hostname
    except ValueError:
        return None
    if not host:
        return None
    host = host.lower().rstrip(".")
    if host.startswith("www."):
        host = host[len("www.") :]
    if not host or "." not in host:
        return None
    return host


def domain_matches(candidate: str, target: str) -> bool:
    """`candidate` is `target` or a subdomain of it."""
    return candidate == target or candidate.endswith("." + target)


def citation_domains(citations: Iterable[object]) -> list[str]:
    """The domains one answer cited, de-duplicated, first seen first; a
    non-web scheme, a string with no host, or one with whitespace in its
    host is skipped."""
    domains: list[str] = []
    for entry in citations:
        if not isinstance(entry, str):
            continue
        try:
            scheme = urlsplit(entry).scheme.lower()
        except ValueError:
            continue
        if scheme not in _CITATION_SCHEMES:
            continue
        domain = domain_from_url(entry)
        if domain and not any(ch.isspace() for ch in domain) and domain not in domains:
            domains.append(domain)
    return domains


def citation_urls(citations: Iterable[object]) -> list[str]:
    """The http(s) URLs one answer cited, de-duplicated, first seen first."""
    urls: list[str] = []
    for entry in citations:
        if not isinstance(entry, str):
            continue
        url = entry.strip()
        try:
            parts = urlsplit(url)
        except ValueError:
            continue
        if parts.scheme.lower() not in _URL_SCHEMES or not parts.netloc or url in urls:
            continue
        urls.append(url)
    return urls
