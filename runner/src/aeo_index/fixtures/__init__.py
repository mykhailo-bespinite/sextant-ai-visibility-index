"""Recorded vendor bodies, shipped as package data.

Each file in `vendors/` is one HTTP reply a vendor API gives, in the
vendor's documented shape, with no real key or account id in it:
`status`, `headers`, `body` (or `raw` for a body that is not JSON) and an
`expected` block saying what a provider must make of it (the text,
citations, search count and queries, model version, tokens and cost, or
the error type). The package's own tests serve them through
`httpx.MockTransport`; another implementation of the same vendors can read
the same files and must reach the same `expected` values.
"""

from __future__ import annotations

import json
from importlib import resources
from importlib.resources.abc import Traversable
from typing import Any

import httpx


def _vendors() -> Traversable:
    return resources.files("aeo_index") / "fixtures" / "vendors"


def vendor_fixture_names() -> list[str]:
    """Every recorded body's file name, sorted."""
    return sorted(entry.name for entry in _vendors().iterdir() if entry.name.endswith(".json"))


def load_vendor_fixture(name: str) -> dict[str, Any]:
    """A fresh copy of one recorded body (callers may change it)."""
    if name not in vendor_fixture_names():
        raise KeyError(f"no recorded vendor body named {name!r}")
    loaded = json.loads((_vendors() / name).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{name}: a recorded vendor body must be a JSON object")
    return loaded


def fixture_response(fixture: dict[str, Any]) -> httpx.Response:
    """The httpx response a recorded body describes."""
    headers = dict(fixture.get("headers") or {})
    raw = fixture.get("raw")
    if raw is not None:
        return httpx.Response(int(fixture["status"]), headers=headers, text=str(raw))
    return httpx.Response(int(fixture["status"]), headers=headers, json=fixture["body"])
