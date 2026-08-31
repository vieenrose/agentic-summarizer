"""arcsum — live agentic zh-TW meeting summarizer.

Runs the project's REAL harness, imported directly (`arcsum/` is vendored here verbatim
and has zero runtime dependencies): the chunker, the op parser, the two-slot memory, the
deterministic guards, and the final SYNTHESIZE call. What the panels show is the actual
mechanism, not a re-enactment of it.

The step loop is driven here rather than through `arcsum.agent.run_agent` for one
reason: streaming. `run_agent` returns a finished `Trace`, which would make the demo a
progress bar over a black box. The ops applied, the caps enforced, and the memory
rendered are all the same code paths `run_agent` uses.
"""

from __future__ import annotations

import os
import time

import gradio as gr
import spaces
from examples import EXAMPLES
from huggingface_hub import hf_hub_download
from model_backend import ArcsumModel

from arcsum.chunker import CHUNK_TOKENS, iter_chunks
from arcsum.guards import apply_ops
from arcsum.memory import ARC_TOKENS, POINT_TOKENS, POINTS_CAP, Memory
from arcsum.ops import parse_ops
from arcsum.prompts import (
    build_step_prompt,
    build_synth_prompt,
    step_system_prompt,
    synth_system_prompt,
)
from arcsum.prose import finalize
from arcsum.tokens import heuristic_token_len
from arcsum.transcript import Utterance, parse_transcript

MODEL_REPO = os.environ.get("ARCSUM_MODEL_REPO", "Luigi/minicpm5-1b-arcsum")
#: Q8_0 — the EVALUATED artifact, and now the only one published.
#: Q4_K_M used to be served here for speed. It was then MEASURED against Q8_0 on the same
#: 40 held-out meetings and is materially worse: the agent's margin over the map-reduce
#: baseline more than halves on ROUGE-1 (+0.077 -> +0.034, wins 29/40 -> 22/40) and its
#: summaries run ~30% shorter (226 vs 320 chars), i.e. it simply records less. A demo
#: running that quant under the card's Q8 numbers would misrepresent the system, so the
#: file was withdrawn from the model repo.
#: Cost: 1.15 GB instead of 688 MB, and slower on Space CPU. `ARCSUM_MODEL_FILE` can
#: still point elsewhere — but measure any replacement before trusting it.
MODEL_FILE = os.environ.get("ARCSUM_MODEL_FILE", "MiniCPM5-1B.Q8_0.gguf")

MAX_TOKENS_STEP = 512
MAX_TOKENS_SYNTH = 1000
#: Prose calls only — see `ArcsumModel`.
SYNTH_REPEAT_PENALTY = 1.1

#: Chat-template turn markers. The backend stops on these, but a leaked marker must
#: never reach the UI either -- measured, a synthesis once emitted its summary, then the
#: literal "assistant", then a second near-duplicate summary. Cutting here as well means
#: one missed stop cannot put two summaries in the panel.
_TURN_MARKERS = ("<|im_end|>", "<|im_start|>", "\nassistant\n", "\nuser\n")

_model: ArcsumModel | None = None


def _cut_at_turn_end(text: str) -> str:
    for marker in _TURN_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.strip()


@spaces.GPU(duration=1)
def _zerogpu_registration_noop() -> None:
    """Never called. ZeroGPU Spaces hard-fail at startup unless the `spaces` package
    detects at least one @spaces.GPU function; all real inference here stays plain CPU
    code (`n_gpu_layers=0` in ArcsumModel) so it draws zero GPU quota."""
    return None


def get_model() -> ArcsumModel:
    global _model
    if _model is None:
        path = hf_hub_download(MODEL_REPO, MODEL_FILE)
        _model = ArcsumModel(path)
    return _model


# --- rendering -------------------------------------------------------------------------


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _panel(title: str, status: str, body_html: str) -> str:
    badge = f"<span style='opacity:.6;font-size:.85em'>{_esc(status)}</span>" if status else ""
    return (
        "<div style='border:1px solid #d0d7de;border-radius:8px;padding:10px;"
        "height:460px;overflow:auto;background:#fff'>"
        f"<div style='display:flex;justify-content:space-between;align-items:baseline;"
        f"margin-bottom:8px'><b>{_esc(title)}</b>{badge}</div>{body_html}</div>"
    )


def render_transcript_html(utterances: list[Utterance], first: int, last: int) -> str:
    if not utterances:
        return _panel("Transcript", "", "<i style='opacity:.6'>No transcript loaded.</i>")
    rows = []
    for i, u in enumerate(utterances):
        active = first <= i <= last
        style = (
            "background:#fff3cd;border-left:3px solid #f0a000;padding:2px 6px;margin:1px 0"
            if active
            else "padding:2px 6px;margin:1px 0;opacity:.55"
        )
        rows.append(
            f"<div style='{style}'><b style='opacity:.7'>{_esc(u.speaker)}:</b> "
            f"{_esc(u.text)}</div>"
        )
    # En dash is correct typography for a numeric range, and this string is UI copy, not
    # an identifier -- RUF001's ambiguity concern does not apply.
    label = f"chunk lines {first + 1}\u2013{last + 1}" if first >= 0 else ""
    return _panel("Transcript", label, "".join(rows))


def render_ops_html(status: str, raw: str, live: bool) -> str:
    if not raw:
        body = "<i style='opacity:.6'>Waiting…</i>"
    else:
        lines = []
        for line in raw.splitlines():
            s = line.strip()
            if not s:
                continue
            color = (
                "#0969da"
                if s.startswith("ARC")
                else "#1a7f37"
                if s.startswith("ADD")
                else "#cf222e"
                if s.startswith("DROP")
                else "#57606a"
            )
            lines.append(
                f"<div style='font-family:ui-monospace,monospace;font-size:.9em;"
                f"color:{color};padding:1px 0'>{_esc(s)}</div>"
            )
        body = "".join(lines) + ("<span style='opacity:.5'>▌</span>" if live else "")
    return _panel("Model output (edit lines)", status, body)


def render_memory_html(mem: Memory, n_chunks: int, step: int) -> str:
    arc_tok = heuristic_token_len(mem.arc) if mem.arc else 0
    arc = (
        f"<div style='background:#ddf4ff;border-left:3px solid #0969da;padding:6px 8px;"
        f"margin-bottom:8px'><b>ARC</b> <span style='opacity:.6;font-size:.85em'>"
        f"{arc_tok}/{ARC_TOKENS} tok</span><br>{_esc(mem.arc)}</div>"
        if mem.arc
        else "<div style='opacity:.5;margin-bottom:8px'><b>ARC</b> — empty</div>"
    )
    pts = (
        "".join(
            f"<div style='padding:2px 6px;border-bottom:1px solid #eee;font-size:.92em'>"
            f"<span style='opacity:.45'>{i + 1}.</span> {_esc(p.text)} "
            f"<span style='opacity:.45;font-size:.85em'>"
            f"({heuristic_token_len(p.text)}/{POINT_TOKENS})</span></div>"
            for i, p in enumerate(mem.points)
        )
        or "<div style='opacity:.5;padding:4px'>no points yet</div>"
    )
    head = (
        f"<b>POINTS</b> <span style='opacity:.6;font-size:.85em'>"
        f"{len(mem.points)}/{POINTS_CAP}</span>"
    )
    status = f"step {step}/{n_chunks}" if n_chunks else ""
    return _panel("External memory", status, arc + head + pts)


def render_prose_html(text: str, live: bool) -> str:
    if not text:
        body = "<i style='opacity:.6'>Produced after the last chunk, from the memory alone.</i>"
    else:
        body = (
            f"<div style='line-height:1.75;font-size:1.02em'>{_esc(text)}"
            + ("<span style='opacity:.5'>▌</span>" if live else "")
            + f"</div><div style='margin-top:10px;opacity:.55;font-size:.85em'>"
            f"{len(text)} characters</div>"
        )
    return _panel("Final summary (SYNTHESIZE)", "", body)


def _progress(pct: float, label: str) -> str:
    return (
        f"<div style='margin:6px 0'><div style='background:#eaeef2;border-radius:4px;height:8px'>"
        f"<div style='background:#0969da;height:8px;border-radius:4px;"
        f"width:{pct:.1f}%'></div></div>"
        f"<div style='opacity:.7;font-size:.88em;margin-top:4px'>{_esc(label)}</div></div>"
    )


# --- the run loop ----------------------------------------------------------------------

_BUSY = (gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True))
_IDLE = (gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False))


def run_demo(custom_transcript: str, example_name: str):
    text = (custom_transcript or "").strip() or EXAMPLES.get(example_name, "")
    empty = (
        render_transcript_html([], -1, -1),
        render_ops_html("", "", False),
        render_memory_html(Memory(token_len=heuristic_token_len), 0, 0),
        render_prose_html("", False),
    )
    if not text.strip():
        yield (
            *empty,
            _progress(0, "Pick an example or paste a transcript, then press Run."),
            *_IDLE,
        )
        return

    try:
        utterances = parse_transcript(text)
    except ValueError as exc:
        yield (*empty, _progress(0, f"Transcript format error: {exc}"), *_IDLE)
        return

    chunks = list(iter_chunks(utterances, budget=CHUNK_TOKENS, token_len=heuristic_token_len))
    mem = Memory(token_len=heuristic_token_len)
    sys_step = step_system_prompt()

    yield (
        render_transcript_html(utterances, -1, -1),
        render_ops_html("", "", False),
        render_memory_html(mem, len(chunks), 0),
        render_prose_html("", False),
        _progress(0, "Loading the model…" if _model is None else "Starting…"),
        *_BUSY,
    )
    model = get_model()

    consecutive_nops = 0
    idx = 0
    for ci, chunk in enumerate(chunks):
        first = idx
        last = idx + len(chunk.utterances) - 1
        idx = last + 1
        t_html = render_transcript_html(utterances, first, last)
        m_html = render_memory_html(mem, len(chunks), ci)
        status = f"step {ci + 1}/{len(chunks)}"
        prog = _progress(ci / len(chunks) * 90, f"{status} — reading…")

        yield (
            t_html,
            render_ops_html(status, "", True),
            m_html,
            render_prose_html("", False),
            prog,
            *_BUSY,
        )

        user = build_step_prompt(mem, chunk)
        t0 = time.time()
        raw = ""
        for partial in model.stream(sys_step, user, max_tokens=MAX_TOKENS_STEP):
            raw = partial
            yield (
                t_html,
                render_ops_html(status, _cut_at_turn_end(raw), True),
                m_html,
                render_prose_html("", False),
                prog,
                *_BUSY,
            )
        raw = _cut_at_turn_end(raw).strip()
        elapsed = time.time() - t0

        # The real harness: parse, then apply deterministically with every guard and cap.
        outcome = apply_ops(mem, parse_ops(raw), chunk, consecutive_nops=consecutive_nops)
        consecutive_nops = consecutive_nops + 1 if outcome.nop_collapse else 0

        done = f"{status} ({elapsed:.1f}s)"
        yield (
            t_html,
            render_ops_html(done, raw, False),
            render_memory_html(mem, len(chunks), ci + 1),
            render_prose_html("", False),
            _progress((ci + 1) / len(chunks) * 90, done),
            *_BUSY,
        )

    # SYNTHESIZE: the memory alone, no transcript. This is the step map-reduce has no
    # equivalent of -- everything the summary says has already survived curation.
    final_t = render_transcript_html(utterances, -1, -1)
    final_m = render_memory_html(mem, len(chunks), len(chunks))
    yield (
        final_t,
        render_ops_html("reading done", "", False),
        final_m,
        render_prose_html("", True),
        _progress(92, "SYNTHESIZE — writing the summary…"),
        *_BUSY,
    )

    prose_raw = ""
    for partial in model.stream(
        synth_system_prompt(),
        build_synth_prompt(mem),
        max_tokens=MAX_TOKENS_SYNTH,
        repeat_penalty=SYNTH_REPEAT_PENALTY,
    ):
        prose_raw = partial
        yield (
            final_t,
            render_ops_html("reading done", "", False),
            final_m,
            render_prose_html(_cut_at_turn_end(prose_raw), True),
            _progress(96, "SYNTHESIZE — writing the summary…"),
            *_BUSY,
        )

    prose = finalize(_cut_at_turn_end(prose_raw), token_len=heuristic_token_len)
    yield (
        final_t,
        render_ops_html("reading done", "", False),
        final_m,
        render_prose_html(prose.text, False),
        _progress(100, "done"),
        *_IDLE,
    )


# --- UI --------------------------------------------------------------------------------

with gr.Blocks(title="arcsum — live agentic zh-TW meeting summarizer") as demo:
    gr.HTML(
        "<div style='padding:14px 4px 6px'>"
        "<h1 style='margin:0'>arcsum — live agentic meeting summarizer</h1>"
        "<p style='margin:6px 0 0;opacity:.75'>A 1B model reads a zh-TW transcript in "
        "~2,500-token chunks, curating a two-slot external memory (<b>ARC</b> + "
        "<b>POINTS</b>) with <code>ADD</code> / <code>DROP</code> / <code>ARC</code> / "
        "<code>NOP</code> edit lines. No conversation history crosses steps — the memory "
        "on the right is the <i>only</i> thing carried forward. A final SYNTHESIZE call "
        "turns it into prose.</p></div>"
    )
    with gr.Row():
        example_dd = gr.Dropdown(
            choices=list(EXAMPLES.keys()),
            value=(list(EXAMPLES.keys()) or [None])[0],
            label="transcript",
        )
        run_btn = gr.Button("▶ Run", variant="primary")
        stop_btn = gr.Button("■ Stop", interactive=False)

    progress_html = gr.HTML(_progress(0, "Pick a transcript and press Run."))

    with gr.Accordion(
        "Paste your own zh-TW transcript (format: `speaker: text`, one per line)", open=False
    ):
        transcript_box = gr.Textbox(
            lines=8, show_label=False, placeholder="S1: 我們開始今天的會議。\nS2: 好的。"
        )

    with gr.Row():
        transcript_html = gr.HTML(render_transcript_html([], -1, -1))
        ops_html = gr.HTML(render_ops_html("", "", False))
        memory_html = gr.HTML(render_memory_html(Memory(token_len=heuristic_token_len), 0, 0))
    prose_html = gr.HTML(render_prose_html("", False))

    gr.Markdown(
        f"""
### What you are watching

The **middle panel** is the model's raw output — it emits *only* edit lines, never prose,
until the very end. The **right panel** is the harness's memory after applying them
deterministically (with the token caps enforced, and malformed or unsafe ops refused).

The design bet is what the memory buys: **a later chunk can overturn an earlier
conclusion.** Map-reduce structurally cannot do that — each window is summarised
independently, so a decision reversed at minute 90 never reaches the summary of minute 10.

### Honest status

This checkpoint clears **6 of 7** ship gates on **40 held-out meetings** it was never
trained or tuned on. It beats a fair map-reduce baseline (same model, same chunk size) on
all three ROUGE metrics — 29/40, 31/40, 33/40 meetings — and on faithfulness by a wide
margin (**18 inversions vs 109**), while writing summaries ~8x shorter. On long meetings
(≥400 lines) it wins 9 of 10.

It **fails the revision probe (G1)**: given a decision reversed later in the meeting it
states the reversal but drops the identifying detail. Under an all-or-nothing rule the
recorded decision is therefore *ship the baseline*. Two further caveats travel with the
numbers: the on-device latency gate is a **projection, never measured on a phone**, and
faithfulness per-claim actually favours the baseline (7.3% vs 4.9%) — the agent wins on
absolute inversions partly because it asserts far less. Full detail:
[model card](https://hf.co/{MODEL_REPO}).

**This demo runs Q8_0 — the same artifact every number above was measured on.** A smaller
Q4_K_M build was withdrawn after measurement showed it keeps under half the quality margin.
Expect several seconds per chunk on Space CPU; that is the cost of running the real thing.
"""
    )

    ev = run_btn.click(
        run_demo,
        [transcript_box, example_dd],
        [
            transcript_html,
            ops_html,
            memory_html,
            prose_html,
            progress_html,
            run_btn,
            example_dd,
            stop_btn,
        ],
    )
    stop_btn.click(None, None, None, cancels=[ev])

# Guarded so `import app` does not start a server. HF Spaces runs this file as
# __main__, so the Space is unaffected -- but without the guard the module cannot be
# imported by a test or a REPL at all, which is how this demo's smoke test hung.
if __name__ == "__main__":
    demo.queue(max_size=8).launch()
