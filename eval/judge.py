#!/usr/bin/env python3
"""Judged evaluation (CLAUDE.md §7.1, §7.2) against a three-family judge panel.

**Judge panel — a spec amendment, measured not inherited.** §7 mandates gemma-4 judges. On
a planted-inversion probe `google/gemma-3n-E4B-it` answered SUPPORTED to all three cases in
4 tokens, including an inversion and an unrelated claim — it would certify "0% inversions"
on inverted notes. It is disqualified. The panel is three *distinct* families, none of them
the student's or the teacher's (both Gemma):

| role | model | family |
|---|---|---|
| FAITH-claim / FAITH-anchor / INVERT | `openai/gpt-oss-20b` | OpenAI |
| COVER / SYNTH (1M ctx: full-context mode) | `deepseek-ai/DeepSeek-V4-Flash-0731` | DeepSeek |
| second opinion | `Prism-ML/Ternary-Bonsai-27B` (free) | Qwen |

**Budget.** Spend is capped and accounted per call. A judged eval is a loop over bullets ×
meetings × modes × systems, so an unguarded bug is an unbounded bill; `JudgeBudgetExceeded`
stops the run instead.

**Empty content is an error, not a verdict.** All three judges route through a reasoning
channel and return an empty `content` when the token budget runs out mid-thought. Scoring
that as "missing" would quietly depress whichever system was unlucky.

    TOGETHER_API_KEY=... python eval/judge.py --notes notes.txt --transcript t.txt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voxsum.index import TranscriptIndex  # noqa: E402
from voxsum.state import BULLET_SECTIONS  # noqa: E402
from voxsum.transcript import clock_to_sec, parse_transcript, sec_to_clock  # noqa: E402

ENDPOINT = "https://api.together.xyz/v1/chat/completions"

# OpenCode Go: a second provider, OpenAI-compatible, reached with `opencode-go/<model>`.
# Flat-rate subscription rather than per-token, so `PRICES` records 0.0 and the USD budget
# guard does not constrain it — the real limits are the plan's $12/5h, $30/week, $60/month.
OPENCODE_GO_ENDPOINT = "https://opencode.ai/zen/go/v1/chat/completions"
OPENCODE_GO_PREFIX = "opencode-go/"

# USD per 1M tokens, from the Together model list. Used for accounting only — verify
# against the live list before quoting a cost.
PRICES = {
    "openai/gpt-oss-20b": (0.05, 0.20),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "deepseek-ai/DeepSeek-V4-Flash-0731": (0.14, 0.28),
    "Prism-ML/Ternary-Bonsai-27B": (0.0, 0.0),
    # OpenCode Go — subscription, not per-token. Verified 3/3 on the planted-inversion probe.
    "opencode-go/deepseek-v4-pro": (0.0, 0.0),
    "opencode-go/glm-5.2": (0.0, 0.0),
    "opencode-go/kimi-k3": (0.0, 0.0),
}

PANEL = {
    "faith": "openai/gpt-oss-20b",
    "cover": "deepseek-ai/DeepSeek-V4-Flash-0731",
    "second": "Prism-ML/Ternary-Bonsai-27B",
}

# Disqualified by the planted-inversion probe — kept as a named constant so nobody
# reintroduces it from the spec by mistake.
DISQUALIFIED = {"google/gemma-3n-E4B-it", "google/gemma-4-E4B-it", "google/gemma-4-E2B-it"}

_VERDICT = re.compile(r"\b(SUPPORTED|CONTRADICTED|UNSUPPORTED)\b", re.I)
_SCORE = re.compile(r"^\s*(?P<key>COVER|SYNTH)\s*[:=]\s*(?P<value>[1-5])", re.I | re.M)


class JudgeBudgetExceeded(RuntimeError):
    """The run would exceed the configured USD cap."""


@dataclass
class Spend:
    """Per-model token and dollar accounting."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    by_model: dict[str, float] = field(default_factory=dict)

    def add(self, model: str, in_tok: int, out_tok: int) -> None:
        cin, cout = PRICES.get(model, (0.0, 0.0))
        cost = in_tok / 1e6 * cin + out_tok / 1e6 * cout
        self.calls += 1
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        self.usd += cost
        self.by_model[model] = self.by_model.get(model, 0.0) + cost

    def report(self) -> str:
        lines = [
            f"spend: ${self.usd:.4f} over {self.calls} calls "
            f"({self.input_tokens:,} in / {self.output_tokens:,} out)"
        ]
        lines += [f"  {m:<44} ${c:.4f}" for m, c in sorted(self.by_model.items())]
        return "\n".join(lines)


@dataclass
class TogetherJudge:
    """Together.ai chat client hardened for the three traps this project has hit."""

    api_key: str
    #: OpenCode Go key, for `opencode-go/*` models. Defaults to the environment.
    opencode_go_key: str = field(
        default_factory=lambda: os.environ.get("OPENCODE_GO_API_KEY", "")
    )
    spend: Spend = field(default_factory=Spend)
    budget_usd: float = 1.00
    # High by default: all three judges reason before answering, and Bonsai burns ~1000
    # tokens for a one-word verdict.
    max_tokens: int = 3000
    timeout: float = 300.0
    retries: int = 1

    def __post_init__(self) -> None:
        if not self.api_key:
            raise SystemExit("TOGETHER_API_KEY is not set")

    def __call__(self, model: str, system: str, user: str) -> str:
        if model in DISQUALIFIED:
            raise ValueError(
                f"{model} failed the planted-inversion probe (answers SUPPORTED to "
                "everything) and must not be used as a judge"
            )
        if self.spend.usd >= self.budget_usd:
            raise JudgeBudgetExceeded(
                f"spent ${self.spend.usd:.4f} of ${self.budget_usd:.2f}; stopping"
            )

        # An empty `content` has two causes: reasoning overran max_tokens, or the endpoint
        # simply returned nothing (observed at 13 completion tokens). Retry with more room
        # before giving up — a judged run is hundreds of calls and must not die on one.
        attempts = self.retries + 1
        last = ""
        for attempt in range(attempts):
            budget = self.max_tokens * (attempt + 1)
            payload = self._post(model, system, user, budget)
            usage = payload.get("usage", {})
            self.spend.add(
                model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            )
            content = (payload["choices"][0]["message"].get("content") or "").strip()
            if content:
                return content
            last = (
                f"{model} returned {usage.get('completion_tokens', 0)} tokens with empty "
                f"content at max_tokens={budget}"
            )
        raise RuntimeError(f"{last} after {attempts} attempts")

    def _post(self, model: str, system: str, user: str, max_tokens: int) -> dict:
        body = json.dumps(
            {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
        ).encode()
        endpoint, key = ENDPOINT, self.api_key
        if model.startswith(OPENCODE_GO_PREFIX):
            endpoint = OPENCODE_GO_ENDPOINT
            key = self.opencode_go_key
            if not key:
                raise SystemExit(
                    f"{model} needs OPENCODE_GO_API_KEY (or opencode_go_key) to be set"
                )
            body = body.replace(
                json.dumps(model).encode(),
                json.dumps(model[len(OPENCODE_GO_PREFIX) :]).encode(),
                1,
            )

        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                # Together 403s the default python-urllib User-Agent.
                "User-Agent": "voxsum-eval/0.1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"together {exc.code}: {exc.read().decode()[:300]}") from exc


# --- prompts -------------------------------------------------------------------

_FAITH_SYS = (
    "You verify one bullet from a set of meeting notes against transcript evidence.\n"
    "SUPPORTED   - the evidence states the claim.\n"
    "CONTRADICTED - the evidence states the OPPOSITE of the claim (e.g. the notes say a "
    "plan was rejected but the evidence shows it was approved).\n"
    "UNSUPPORTED - the evidence neither states nor contradicts the claim.\n"
    "A bullet that is a noun phrase (a topic, an open question, or a described action) "
    "asserts no decision or outcome; call it CONTRADICTED only if the evidence states the "
    "opposite of something the bullet itself asserts, not merely because the evidence "
    "discusses a different framing of the subject.\n"
    "Pay particular attention to reversals over time: when the evidence shows a decision "
    "changed, the claim must match the LATEST state, not the earliest.\n"
    "Answer with exactly one word."
)

_COVER_SYS = (
    "You score a set of meeting notes against the meeting itself, on two axes.\n"
    "COVER (1-5): how much of the meeting's important content (decisions, actions, "
    "commitments, key topics) the notes capture. 5 = essentially complete.\n"
    "SYNTH (1-5): meeting-level insight — does it convey the arc of the discussion, how "
    "decisions evolved, and the bottom line, rather than disconnected local fragments? "
    "5 = strong global insight, 1 = a pile of unrelated details.\n"
    "Reply with exactly two lines:\nCOVER: <1-5>\nSYNTH: <1-5>"
)


def faith_prompt(bullet: str, evidence: list) -> str:
    body = "\n".join(e.render() for e in evidence) or "(no evidence retrieved)"
    return f"EVIDENCE:\n{body}\n\nBULLET: {bullet}\n\nSUPPORTED, CONTRADICTED or UNSUPPORTED?"


def cover_prompt(notes: str, transcript: str) -> str:
    return f"MEETING TRANSCRIPT:\n{transcript}\n\nNOTES:\n{notes}\n"


# --- scoring -------------------------------------------------------------------

@dataclass
class BulletVerdict:
    section: str
    bullet: str
    anchor: int | None
    mode: str
    verdicts: dict[str, str] = field(default_factory=dict)

    @property
    def majority(self) -> str:
        """Majority verdict across families; ties fall to the most severe.

        Severity order matters for a 0%-inversion product requirement: on a 1-1-1 split we
        report CONTRADICTED rather than average it away.
        """
        if not self.verdicts:
            return "MISSING"
        counts: dict[str, int] = {}
        for v in self.verdicts.values():
            counts[v] = counts.get(v, 0) + 1
        best = max(counts.values())
        tied = [v for v, c in counts.items() if c == best]
        for severe in ("CONTRADICTED", "UNSUPPORTED", "SUPPORTED"):
            if severe in tied:
                return severe
        return tied[0]


@dataclass
class MeetingScore:
    meeting_id: str
    system: str
    faith_claim: float | None = None
    faith_anchor: float | None = None
    cover: int | None = None
    synth: int | None = None
    inverted: bool = False
    unsupported: int = 0
    bullets: list[BulletVerdict] = field(default_factory=list)
    judges: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        def fmt(v: float | None) -> str:
            return "n/a" if v is None else f"{v:.2f}"

        return (
            f"{self.meeting_id} [{self.system}] "
            f"FAITH-claim {fmt(self.faith_claim)} FAITH-anchor {fmt(self.faith_anchor)} "
            f"COVER {self.cover or 'n/a'} SYNTH {self.synth or 'n/a'} "
            f"INVERT {'YES' if self.inverted else 'NO'} UNSUPPORTED {self.unsupported}"
        )


def parse_bullets(notes: str) -> list[tuple[str, str, int | None]]:
    """(section, bullet, anchor) from rendered NOTES v2. TITLE carries no anchor."""
    out: list[tuple[str, str, int | None]] = []
    section = None
    for line in notes.splitlines():
        line = line.rstrip()
        head = line.rstrip(":")
        if head in BULLET_SECTIONS:
            section = head
            continue
        if not line.startswith("- ") or section is None:
            continue
        text = line[2:].strip()
        anchor = None
        m = re.search(r"\[([0-9:]+)\]\s*$", text)
        if m:
            try:
                anchor = clock_to_sec(m.group(1))
                text = text[: m.start()].strip()
            except ValueError:
                anchor = None
        out.append((section, text, anchor))
    return out


def _score_from(text: str) -> tuple[int | None, int | None]:
    """Last-match parsing per key (§7.2) — a judge that restates then decides ends right."""
    found: dict[str, int] = {}
    for m in _SCORE.finditer(text):
        found[m.group("key").upper()] = int(m.group("value"))
    return found.get("COVER"), found.get("SYNTH")


def judge_meeting(
    notes: str,
    utterances: list,
    client: TogetherJudge,
    *,
    meeting_id: str,
    system_name: str,
    panel: dict[str, str] | None = None,
    second_opinion: bool = True,
    full_context: bool = True,
) -> MeetingScore:
    """Score one system's notes for one meeting on every §7.1 metric."""
    models = {**PANEL, **(panel or {})}
    index = TranscriptIndex(utterances)
    score = MeetingScore(meeting_id, system_name, judges=dict(models))
    bullets = parse_bullets(notes)

    for mode in ("claim", "anchor"):
        supported = 0
        scored = 0
        for section, bullet, anchor in bullets:
            evidence = index.evidence_for(bullet, anchor, mode=mode)
            prompt = faith_prompt(
                bullet + (f" [{sec_to_clock(anchor)}]" if anchor is not None else ""), evidence
            )
            record = BulletVerdict(section, bullet, anchor, mode)

            voters = [models["faith"]] + ([models["second"]] if second_opinion else [])
            for model in voters:
                raw = client(model, _FAITH_SYS, prompt)
                hits = _VERDICT.findall(raw)
                if hits:
                    record.verdicts[model] = hits[-1].upper()

            verdict = record.majority
            scored += 1
            if verdict == "SUPPORTED":
                supported += 1
            elif verdict == "CONTRADICTED" and section in ("DECISIONS", "ACTIONS", "SUMMARY"):
                # INVERT is about decisions, approvals, outcomes and commitments (§7.1).
                score.inverted = True
            elif verdict == "UNSUPPORTED" and mode == "claim":
                score.unsupported += 1
            score.bullets.append(record)

        # 1-5 scale: 5 = every bullet supported (§7.1).
        rate = supported / scored if scored else None
        value = None if rate is None else 1 + 4 * rate
        if mode == "claim":
            score.faith_claim = value
        else:
            score.faith_anchor = value

    # COVER/SYNTH. Full-context mode is the §7.2 cross-validation reference and is now the
    # default, because the primary judge's 1M window fits an entire 80k-token meeting —
    # the per-part agenda approximation the spec was forced into is no longer necessary.
    transcript = (
        "".join(u.render() + "\n" for u in utterances)
        if full_context
        else "\n".join(u.render() for u in utterances[:: max(len(utterances) // 20, 1)])
    )
    raw = client(models["cover"], _COVER_SYS, cover_prompt(notes, transcript))
    cover, synth = _score_from(raw)
    if cover is None or synth is None:
        raw = client(models["cover"], _COVER_SYS, cover_prompt(notes, transcript))
        cover, synth = _score_from(raw)
    score.cover, score.synth = cover, synth
    return score


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--notes", type=Path, required=True, help="rendered NOTES v2")
    p.add_argument("--transcript", type=Path, required=True, help="transcript v1")
    p.add_argument("--system", default="cursor", help="system label for the report")
    p.add_argument("--meeting-id", default=None)
    p.add_argument("--budget-usd", type=float, default=0.25)
    p.add_argument("--no-second-opinion", action="store_true")
    p.add_argument(
        "--agenda-mode", action="store_true", help="per-part agenda instead of full context"
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    client = TogetherJudge(
        api_key=os.environ.get("TOGETHER_API_KEY", ""), budget_usd=args.budget_usd
    )
    utterances = parse_transcript(args.transcript.read_text(encoding="utf-8"))
    try:
        score = judge_meeting(
            args.notes.read_text(encoding="utf-8"),
            utterances,
            client,
            meeting_id=args.meeting_id or args.transcript.stem,
            system_name=args.system,
            second_opinion=not args.no_second_opinion,
            full_context=not args.agenda_mode,
        )
    except JudgeBudgetExceeded as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        print(client.spend.report(), file=sys.stderr)
        return 2

    print(score.summary())
    print(client.spend.report())
    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "meeting_id": score.meeting_id,
                    "system": score.system,
                    "faith_claim": score.faith_claim,
                    "faith_anchor": score.faith_anchor,
                    "cover": score.cover,
                    "synth": score.synth,
                    "inverted": score.inverted,
                    "unsupported": score.unsupported,
                    "judges": score.judges,
                    "spend_usd": client.spend.usd,
                    "bullets": [
                        {
                            "section": b.section,
                            "bullet": b.bullet,
                            "anchor": b.anchor,
                            "mode": b.mode,
                            "verdicts": b.verdicts,
                            "majority": b.majority,
                        }
                        for b in score.bullets
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
