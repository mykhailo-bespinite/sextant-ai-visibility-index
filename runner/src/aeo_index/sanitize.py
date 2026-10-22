"""Text that came from a file or a vendor, made safe to print or to write
as Markdown: no terminal escape sequences, no zero-width or direction
marks, no line breaks inside a table cell, no stray pipes."""

from __future__ import annotations

import unicodedata


def strip_controls(text: str) -> str:
    """`text` without control or format characters (Unicode categories
    starting with C: terminal escapes, zero-width and direction marks),
    newlines kept. Every string that came from a file or a vendor passes
    this before it is printed or written as Markdown."""
    return "".join(ch for ch in text if ch == "\n" or not unicodedata.category(ch).startswith("C"))


def plain(text: object) -> str:
    """`text` on one line, its whitespace collapsed and its control and
    format characters removed."""
    return strip_controls(" ".join(str(text).split()))


def md_cell(text: object) -> str:
    """`text` safe inside a Markdown table cell."""
    return plain(text).replace("|", "\\|")
