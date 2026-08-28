"""Convert a Legislative Yuan (立法院) OP-MSF committee transcript (.odt) to format v2.

    python tools/ly_odt_to_v2.py ly_21.odt --out data/ly_pilot/op-msf-21.txt

Why this material: it is a REAL multi-speaker meeting, natively zh-TW, and CLEAN
(stenographic, not ASR). That combination is what makes it a controlled probe. The
Phase 3 pilot (`runs/phase3-pilot/RESULT.md`) found the agent NOPing 8/8 on zh-TW
podcasts, but domain, ASR noise and translationese were all confounded there. This
slice removes two of the three:

    podcasts     : out-of-domain + ASR noise  + native zh
    LY committee : IN-domain     + clean text + native zh
    MeetingBank  : in-domain     + clean text + TRANSLATED (what it was trained on)

So a normal curation run here points at ASR noise or podcast-domain as the cause; a
NOP collapse here points at the model having learned translated-council register
specifically, which is the reading that undermines Phase 4's justification.

**Continuation paragraphs are attached to the preceding speaker.** The gazette format
gives the speaker once and then continues over several paragraphs; treating those as
separate speakerless lines would emit `UNK:` utterances and misattribute the content.
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

TEXT_NS = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"

#: `主席（林委員昶佐）：…` or `鄭婷宇：…`. Fullwidth colon is the gazette's delimiter;
#: the speaker is bounded to keep a colon inside prose from being read as a delimiter
#: (the same hazard `transcript.MAX_SPEAKER_LEN` guards against).
SPEAKER_RE = re.compile(r"^(?P<speaker>[^：]{1,20})：(?P<text>.*)$")

#: Front matter: title, 時間, 地點, 主席, 出席, 列席, 紀錄 — metadata, not utterances.
HEADER_RE = re.compile(r"^(時\s*間|地\s*點|主\s*席|出\s*席|列\s*席|紀\s*錄|報告事項|討論事項)\s")


def paragraphs(odt_path: Path) -> list[str]:
    with zipfile.ZipFile(odt_path) as z:
        root = ET.fromstring(z.read("content.xml"))
    out = []
    for node in root.iter():
        if node.tag in (TEXT_NS + "p", TEXT_NS + "h"):
            s = "".join(node.itertext()).strip()
            if s:
                out.append(s)
    return out


def to_v2(paras: list[str]) -> tuple[str, dict]:
    lines: list[tuple[str, str]] = []
    skipped_header = 0
    for i, para in enumerate(paras):
        # The first paragraph is the meeting title; header fields follow.
        if i == 0 or HEADER_RE.match(para):
            skipped_header += 1
            continue
        m = SPEAKER_RE.match(para)
        if m and m.group("text").strip():
            lines.append((m.group("speaker").strip(), m.group("text").strip()))
        elif m and not m.group("text").strip():
            # Speaker with the utterance starting on the next paragraph.
            lines.append((m.group("speaker").strip(), ""))
        elif lines:
            # Continuation of the current speaker's turn.
            speaker, text = lines[-1]
            lines[-1] = (speaker, (text + " " + para).strip())
        else:
            skipped_header += 1  # stray front matter before any speaker

    lines = [(s, t) for s, t in lines if t]
    body = "\n".join(f"{s}: {t}" for s, t in lines) + "\n"
    stats = {
        "utterances": len(lines),
        "speakers": sorted({s for s, _ in lines}),
        "skipped_header_paras": skipped_header,
    }
    return body, stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("odt", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    body, stats = to_v2(paragraphs(args.odt))
    if stats["utterances"] < 5:
        print(f"[ly->v2] REFUSED {args.odt}: only {stats['utterances']} utterances", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body, encoding="utf-8")
    print(
        f"[ly->v2] {args.odt.name} -> {args.out}  utterances={stats['utterances']} "
        f"speakers={len(stats['speakers'])} skipped_header={stats['skipped_header_paras']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
