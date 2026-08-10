"""Streaming cursor and chunker (CLAUDE.md §4, §5.3).

The transcript is processed as a stream: contiguous ~2048-token chunks with a 2-line
overlap so a decision straddling a boundary is visible whole at least once.

Two things this must get right, both from the spec's own caveats:

* **A single line can exceed a chunk.** VCSum zh lines run to ~2.6k chars, so a line is
  split across chunks rather than assumed to fit — with the *same* start timestamp on each
  piece, because an anchor must resolve to a real transcript line.
* **Token counting is pluggable.** The default is a cheap heuristic; trace generation and
  training pass the student's real tokenizer, since the per-step budget is normative
  (PLAN.md §2c) and a heuristic must never be what decides whether a step is on budget.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from .transcript import Utterance, format_line

__all__ = ["Chunk", "CHUNK_TOKENS", "OVERLAP_LINES", "heuristic_token_len", "iter_chunks"]

CHUNK_TOKENS = 2048
OVERLAP_LINES = 2


def heuristic_token_len(text: str) -> int:
    """Cheap token estimate: ~4 chars/token for latin text, ~1 token per CJK char.

    Deliberately conservative (over-estimates rather than under) so a heuristic-built
    chunk never silently exceeds the real budget. Replace with the real tokenizer for
    anything normative.
    """
    cjk = sum(1 for c in text if "　" <= c <= "鿿" or "＀" <= c <= "￯")
    return cjk + (len(text) - cjk + 3) // 4


@dataclass(frozen=True, slots=True)
class Chunk:
    """A contiguous window of transcript lines handed to the model in one step."""

    index: int
    utterances: tuple[Utterance, ...]

    @property
    def start(self) -> int:
        return self.utterances[0].start if self.utterances else 0

    @property
    def end(self) -> int:
        return self.utterances[-1].start if self.utterances else 0

    def has_line(self, anchor_sec: int) -> bool:
        """True if `anchor_sec` is the start of a line in this chunk (CLAUDE.md §6.1)."""
        return any(u.start == anchor_sec for u in self.utterances)

    def is_content_rich(self) -> bool:
        """Heuristic for the NOP-collapse guard: enough substance to be worth an op.

        Short back-channel exchanges ("mm-hm", "right", "okay") are not content-rich, and
        answering NOP on them is correct behaviour rather than collapse.
        """
        body = " ".join(u.text for u in self.utterances)
        return heuristic_token_len(body) >= 120

    def render(self) -> str:
        return "".join(u.render() + "\n" for u in self.utterances)

    def __len__(self) -> int:
        return len(self.utterances)


def _split_long(u: Utterance, budget: int, token_len: Callable[[str], int]) -> list[Utterance]:
    """Split one over-long utterance into pieces that each fit `budget`.

    Every piece keeps the original start timestamp: the anchor must resolve to a real
    line, and the line's start is the only timestamp v1 records.
    """
    overhead = token_len(format_line(u.start, u.speaker, "")) + 1
    room = max(budget - overhead, 1)
    if token_len(u.text) <= room:
        return [u]

    pieces: list[Utterance] = []
    words = u.text.split(" ")
    # CJK text has no spaces; fall back to character-wise accumulation.
    units = words if len(words) > 1 else list(u.text)
    joiner = " " if len(words) > 1 else ""
    current: list[str] = []
    for unit in units:
        candidate = joiner.join([*current, unit])
        if current and token_len(candidate) > room:
            pieces.append(Utterance(u.start, u.speaker, joiner.join(current)))
            current = [unit]
        else:
            current.append(unit)
    if current:
        pieces.append(Utterance(u.start, u.speaker, joiner.join(current)))
    return pieces


def iter_chunks(
    utterances: list[Utterance],
    *,
    budget: int = CHUNK_TOKENS,
    overlap: int = OVERLAP_LINES,
    token_len: Callable[[str], int] = heuristic_token_len,
) -> Iterator[Chunk]:
    """Yield contiguous chunks of <= `budget` tokens with `overlap` lines of carry-over."""
    if not utterances:
        return

    # Pre-split anything that cannot fit on its own, so the packer never stalls.
    lines: list[Utterance] = []
    for u in utterances:
        lines.extend(_split_long(u, budget, token_len))

    index, i = 0, 0
    while i < len(lines):
        current: list[Utterance] = []
        used = 0
        while i < len(lines):
            cost = token_len(lines[i].render()) + 1
            if current and used + cost > budget:
                room = budget - used
                # A long monologue line that will not fit leaves the chunk part-empty, and
                # that waste is not free: it inflates the number of steps, and every step
                # pays SYS + STATE again. On long-turn transcripts (VCSum zh runs to ~2.6k
                # chars per line) it cost ~27% of the chunk and pushed GT4 over its +25%
                # bound. So split the line here and carry the remainder rather than
                # yielding a three-quarters-full chunk.
                pieces = _split_long(lines[i], room, token_len) if room > 64 else []
                if len(pieces) > 1:
                    lines[i : i + 1] = pieces
                    continue
                break
            current.append(lines[i])
            used += cost
            i += 1
        yield Chunk(index, tuple(current))
        index += 1
        if i >= len(lines):
            break
        # Rewind for the overlap, but never so far that the chunk fails to advance.
        i = max(i - overlap, i - len(current) + 1)
