"""
Query image stems vs CholecT50 ``categories`` instrument/target strings.

Filenames often use underscores (``cystic_artery.png``) while JSON labels use
hyphens (``cystic-artery``). Normalizes both sides for comparison.
"""

from __future__ import annotations


def canonical_label_key(s: str | None) -> str:
    if s is None:
        return ""
    t = str(s).strip().lower()
    for ch in ("_", " ", "/", "\\"):
        t = t.replace(ch, "-")
    while "--" in t:
        t = t.replace("--", "-")
    return t.strip("-")


def query_matches_frame_labels(query_stem: str, frame_labels: set[str]) -> bool:
    """True if ``query_stem`` matches any GT label under :func:`canonical_label_key`."""
    k = canonical_label_key(query_stem)
    if not k:
        return False
    return any(canonical_label_key(lab) == k for lab in frame_labels)
