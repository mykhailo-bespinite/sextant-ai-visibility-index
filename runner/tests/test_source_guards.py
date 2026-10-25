"""Guards on what the public repository may contain.

The leak guard reads every text file of `runner/` that git tracks or would
track (tests, examples, recorded vendor bodies, pyproject, README) plus the
runner workflow and the repository's own README, which documents the
runner. Public checks: generic shapes (IPv4 and e-mail addresses,
organisation ids) everywhere, and key shapes and unknown hostnames outside
the tests. Named internal terms are read from the file the environment
variable AEO_INDEX_LEAK_TERMS names, which lives outside this repository;
without it that test is skipped with its reason. This file is exempt from
the generic checks, because it holds their samples.
"""

import asyncio
import os
import re
import socket
import subprocess
import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest
from pytest_socket import SocketBlockedError

RUNNER = Path(__file__).parents[1]
REPO = RUNNER.parent
SRC = RUNNER / "src" / "aeo_index"
WORKFLOW = REPO / ".github" / "workflows" / "runner.yml"
ROOT_README = REPO / "README.md"
THIS_FILE = Path(__file__).resolve()

# Hosts the package may call.
ALLOWED_HOSTS = {
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "api.x.ai",
    "api.perplexity.ai",
    "api.dataforseo.com",
    "api.github.com",
    "raw.githubusercontent.com",
}
# Domains the source may name without calling them: the tracked brand, the
# example category, and the Google hosts the providers parse (Gemini's
# grounding redirect host, the results page's own links and its unresolved
# google.com/goto redirects, which are dropped from citations).
NAMED_DOMAINS = {
    "sextantaeo.com",
    "acmecrm.com",
    "zedcrm.io",
    "example.com",
    "vertexaisearch.cloud.google.com",
    "google.com",
    "www.google.com",
}

# Hosts the documentation names: the repository, and the examples the judge's
# docstring uses for host bounding.
# zenodo.org and doi.org: the DOI badge and cite line in the README and CITATION.cff.
DOC_HOSTS = {
    "github.com",
    "type.googleapis.com",
    "acme.co",
    "acme.com",
    "notacme.co",
    "zenodo.org",
    "doi.org",
}

# The public repository slug, removed before the private terms are read, so
# the owner's name in the repository URL is not a finding.
ALLOWED_LITERALS = ("mykhailo-bespinite/sextant-ai-visibility-index",)

# (name, pattern): generic shapes that must never reach the public tree. The
# named terms (internal hosts, products, paths) are not listed here, because
# the list would publish them; they come from the file AEO_INDEX_LEAK_TERMS
# names (one regex per line, "(?i)" for case-insensitive, "#" for a comment).
GENERIC_PATTERNS: tuple[tuple[str, str], ...] = (
    ("an IPv4 address", r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
    ("an e-mail address", r"[\w.+-]+@[\w-]+\.[\w.]+"),
    ("an organisation id", r"\borg-[A-Za-z0-9]{8,}"),
)
GENERIC_SAMPLES = ("host 10.0.0.12", "write to someone@example.org", '"organization": "org-AbCdEf123456"')

# Key shapes, checked outside the tests (the redaction tests hold fake ones on purpose).
KEY_SHAPE = re.compile(r"(?<![A-Za-z0-9])(?:sk|xai|pplx)-[A-Za-z0-9_-]{16,}|AIza[0-9A-Za-z_-]{30,}")
# A hostname under a top-level domain a URL here could use.
HOST = re.compile(r"\b(?:[a-z0-9-]+\.)+(?:com|ai|io|net|org|dev|app|co|cloud)\b")
LEAK_TERMS_ENV = "AEO_INDEX_LEAK_TERMS"

QUOTED_DOMAIN = re.compile(r"[\"']([a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|ai|io))[\"']")
SCHEME_HOST = re.compile(r"https?://([A-Za-z0-9.-]+)")


def public_files() -> list[Path]:
    """Every file of runner/ that git tracks or would track, plus the workflow
    and the root README."""
    try:
        listed = subprocess.run(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "--",
                "runner",
                str(WORKFLOW),
                str(ROOT_README),
            ],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("needs a git checkout of the repository")
    files = {(REPO / name).resolve() for name in listed}
    return sorted(p for p in files if p.is_file() and p != THIS_FILE)


def package_files(root: Path = SRC) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts]


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def generic_leaks(text: str) -> list[str]:
    return [name for name, pattern in GENERIC_PATTERNS if re.search(pattern, text)]


def private_terms() -> list[tuple[int, re.Pattern[str]]] | None:
    """The terms in the file AEO_INDEX_LEAK_TERMS names, each with its 1-based
    line number in that file, or None when the variable is not set."""
    path = os.environ.get(LEAK_TERMS_ENV, "").strip()
    if not path:
        return None
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [
        (number, re.compile(line.strip()))
        for number, line in enumerate(lines, start=1)
        if line.strip() and not line.lstrip().startswith("#")
    ]


def term_leaks(text: str, terms: list[tuple[int, re.Pattern[str]]]) -> list[tuple[int, int]]:
    """(term's line in the private list, 1-based line of `text` it matches) for
    each match. Neither the pattern nor the matched text is returned, so a
    finding never prints a private term in a public log."""
    for literal in ALLOWED_LITERALS:
        text = text.replace(literal, "")
    hits: list[tuple[int, int]] = []
    for number, term in terms:
        hits.extend((number, text.count("\n", 0, m.start()) + 1) for m in term.finditer(text))
    return hits


def private_term_report(
    files: list[Path], terms: list[tuple[int, re.Pattern[str]]], read: Callable[[Path], str | None] = read_text
) -> list[str]:
    return [
        f"the term on line {number} of the private list matches {path.relative_to(REPO)}:{line}"
        for path in files
        for number, line in term_leaks(read(path) or "", terms)
    ]


def is_test_file(path: Path) -> bool:
    return (RUNNER / "tests") in path.parents


def allowed_host(host: str) -> bool:
    if host in ALLOWED_HOSTS | NAMED_DOMAINS | DOC_HOSTS:
        return True
    return host.split(".")[-2:] in (["example", "com"], ["example", "net"], ["example", "org"])


def hosts_in(files: list[Path]) -> set[str]:
    found: set[str] = set()
    for path in files:
        text = read_text(path) or ""
        found.update(SCHEME_HOST.findall(text))
        found.update(QUOTED_DOMAIN.findall(text))
    return found


def test_only_allowed_hosts_appear_in_the_package():
    assert hosts_in(package_files()) <= ALLOWED_HOSTS | NAMED_DOMAINS


def test_unquoted_scheme_less_hosts_are_caught(tmp_path):
    sample = tmp_path / "x.py"
    sample.write_text('BASE = "internal.example-corp.io"\nURL = "https://api.evil.ai/v1"\n', encoding="utf-8")
    assert hosts_in([sample]) == {"internal.example-corp.io", "api.evil.ai"}


def test_no_generic_leak_in_the_public_tree():
    scanned = public_files()
    assert any(p.name == "pyproject.toml" for p in scanned) and any(p.name == "runner.yml" for p in scanned)
    assert ROOT_README.resolve() in scanned and (RUNNER / "README.md").resolve() in scanned
    found = {str(p.relative_to(REPO)): hits for p in scanned if (hits := generic_leaks(read_text(p) or ""))}
    assert found == {}


def test_no_key_shape_or_unknown_host_outside_the_tests():
    keys: dict[str, list[str]] = {}
    hosts: dict[str, list[str]] = {}
    for path in public_files():
        if is_test_file(path):
            continue
        text = read_text(path) or ""
        if found := KEY_SHAPE.findall(text):
            keys[str(path.relative_to(REPO))] = [f"{k[:6]}..." for k in found]
        if unknown := sorted({h for h in HOST.findall(text) if not allowed_host(h)}):
            hosts[str(path.relative_to(REPO))] = unknown
    assert keys == {} and hosts == {}


@pytest.mark.parametrize(("name", "sample"), list(zip([n for n, _ in GENERIC_PATTERNS], GENERIC_SAMPLES, strict=True)))
def test_each_generic_pattern_is_caught_in_a_file(tmp_path, name, sample):
    fixture = tmp_path / "vendor_body.json"
    fixture.write_text(f'{{"text": "{sample}"}}\n', encoding="utf-8")
    assert name in generic_leaks(read_text(fixture) or "")


def test_key_shapes_and_unknown_hosts_are_caught():
    assert KEY_SHAPE.search("key sk-proj-AAAAAAAAAAAAAAAAAAAA") and KEY_SHAPE.search("AIza" + "B" * 35)
    assert not KEY_SHAPE.search("a task-list-of-many-words")
    assert [h for h in HOST.findall("see https://intranet.corp.dev/x and api.openai.com") if not allowed_host(h)] == [
        "intranet.corp.dev"
    ]


def test_no_private_term_in_the_public_tree():
    terms = private_terms()
    if terms is None:
        pytest.skip(f"{LEAK_TERMS_ENV} is not set: the private term list lives outside this repository")
    assert terms, f"{LEAK_TERMS_ENV} names a file with no term in it"
    # This file too: it holds no named term. The failure names line numbers
    # only (pytrace off, so no local is printed): the CI log is public.
    report = private_term_report([*public_files(), THIS_FILE], terms)
    if report:
        pytest.fail("\n".join(report), pytrace=False)


def test_the_private_term_file_is_read_and_applied(tmp_path, monkeypatch):
    terms_file = tmp_path / "terms.txt"
    terms_file.write_text("# a comment\n\n(?i)secret-project\nCodeName\n", encoding="utf-8")
    monkeypatch.setenv(LEAK_TERMS_ENV, str(terms_file))
    terms = private_terms()
    assert [(n, t.pattern) for n, t in terms or []] == [(3, "(?i)secret-project"), (4, "CodeName")]
    assert term_leaks("one\nthe Secret-Project and codename", terms or []) == [(3, 2)]
    assert (
        term_leaks("https://github.com/mykhailo-bespinite/sextant-ai-visibility-index", [(1, re.compile("visibility"))])
        == []
    )
    monkeypatch.delenv(LEAK_TERMS_ENV)
    assert private_terms() is None


def test_runtime_dependency_is_httpx_only_and_licence_is_mit():
    project = tomllib.loads((RUNNER / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["name"] == "aeo-index"
    assert project["dependencies"] == ["httpx>=0.27,<1"]
    assert project["license"] == "MIT"
    assert project["requires-python"] == ">=3.12"
    assert project["scripts"] == {"aeo-index": "aeo_index.cli:main"}
    assert "MIT License" in (RUNNER / "LICENSE").read_text(encoding="utf-8")
    assert "2026 Shearwater Labs" in (RUNNER / "LICENSE").read_text(encoding="utf-8")


def test_the_root_licence_is_mit_and_the_data_licence_is_its_own_file():
    # GitHub reads the root LICENSE as the repository's licence: the code's MIT.
    assert (REPO / "LICENSE").read_text(encoding="utf-8") == (RUNNER / "LICENSE").read_text(encoding="utf-8")
    data = (REPO / "LICENSE-DATA").read_text(encoding="utf-8")
    assert "CC BY 4.0" in data and "data/" in data and "creativecommons.org/licenses/by/4.0" in data
    media = (
        "The files in media/ are (c) Shearwater Labs, all rights reserved; "
        "they may be reproduced unaltered when citing or linking the index."
    )
    assert media in data
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "`media/` are © Shearwater Labs, all rights reserved" in readme


def test_package_is_typed():
    assert (SRC / "py.typed").is_file()


def test_env_example_lists_every_variable_with_no_value():
    lines = [
        ln for ln in (RUNNER / ".env.example").read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")
    ]
    assert lines == [
        "OPENAI_API_KEY=",
        "ANTHROPIC_API_KEY=",
        "GEMINI_API_KEY=",
        "GOOGLE_API_KEY=",
        "XAI_API_KEY=",
        "GROK_API_KEY=",
        "PERPLEXITY_API_KEY=",
        "DATAFORSEO_LOGIN=",
        "DATAFORSEO_PASSWORD=",
    ]


def test_version_is_the_release():
    import aeo_index

    assert aeo_index.__version__ == "0.1.0"


# pytest-socket warns each time it blocks a socket; the block is the point here.
@pytest.mark.filterwarnings("ignore:A test tried to use socket.socket")
def test_network_sockets_are_disabled_in_tests():
    with pytest.raises(SocketBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(SocketBlockedError):
        socket.socket(socket.AF_INET6, socket.SOCK_STREAM)


@pytest.mark.filterwarnings("ignore:A test tried to use socket.socket")
async def test_async_tcp_connect_is_blocked():
    with pytest.raises(SocketBlockedError):
        await asyncio.open_connection("127.0.0.1", 9)


async def test_asyncio_runs_with_sockets_disabled():
    await asyncio.sleep(0)


def test_a_private_term_finding_names_only_line_numbers(tmp_path, monkeypatch):
    terms_file = tmp_path / "terms.txt"
    terms_file.write_text("# private\nnothing-here\n(?i)zorblax-[0-9]+\n", encoding="utf-8")
    monkeypatch.setenv(LEAK_TERMS_ENV, str(terms_file))
    planted = REPO / "runner" / "examples" / "planted.txt"
    report = private_term_report([planted], private_terms() or [], read=lambda p: "line one\nthe ZORBLAX-42 project\n")
    assert report == ["the term on line 3 of the private list matches runner/examples/planted.txt:2"]
    assert "zorblax" not in "\n".join(report).lower() and "42" not in "\n".join(report)
