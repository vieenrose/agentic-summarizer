"""A2 — decision-line highlighter (deterministic).

Marks commitment/decision-bearing transcript lines in the CHUNK rendering so the
proposer's attention lands on content-dense lines of noisy ASR input (the measured
coverage collapse). Zero model tokens.

Phase 0.1 measures whether the marker helps or hurts: the proposer was trained on
unmarked renderings, so the G1 screen must be re-checked with --highlight on.
"""
from __future__ import annotations

import re

_COMMIT_EN = re.compile(
    r"\b(agree[sd]?|approv(e|ed|es)|decide[sd]?|reject(ed|s)?|will|shall|assign(ed|s)?|"
    r"commit(ment|ted|s)?|deadline|due (on|by)|confirm(ed|s)?|plan(ned|s)?)\b",
    re.I,
)

_COMMIT_ZH = re.compile(
    r"(通過|同意|決定|否決|拒絕|指派|負責|確認|期限|承諾|定案|批准|駁回|決議|"
    r"會在|將在|會由|將由|由.{0,4}負責|截止|"
    r"就(搬|採|用|定|決定|這麼|這樣|好)|那就|目前先|先不|先否)"
)

#: Marker prepended to a flagged line. Kept to one character so it perturbs the
#: rendering as little as possible; the [m:ss] text stays byte-intact for anchors.
MARKER = "» "


def is_commit_line(text: str, lang: str) -> bool:
    return bool((_COMMIT_ZH if lang == "zh-TW" else _COMMIT_EN).search(text))


def highlight_line(line: str, lang: str) -> str:
    """Render one chunk line, marking it if it bears commitment language.

    The line is '<marker>[m:ss] ...' — the timestamp and text are unchanged, so the
    anchor-copy rule still applies verbatim.
    """
    if is_commit_line(line, lang):
        return MARKER + line
    return line


def highlight_chunk(text: str, lang: str) -> str:
    return "".join(highlight_line(line, lang) for line in text.splitlines(keepends=True))
