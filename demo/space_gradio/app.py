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

import json
import os
import pathlib
import tempfile
import time

import gradio as gr
import spaces
from examples import EXAMPLES
from huggingface_hub import hf_hub_download
from model_backend import ArcsumModel

from arcsum.chunker import CHUNK_TOKENS, iter_chunks
from arcsum.guards import apply_ops
from arcsum.memory import ARC_TOKENS, POINT_TOKENS, POINTS_CAP, Memory
from arcsum.ops import render_op
from arcsum.prompts import (
    build_step_prompt,
    build_synth_prompt,
    synth_system_prompt,
    tool_step_system_prompt,
)
from arcsum.prose import finalize
from arcsum.tokens import heuristic_token_len
from arcsum.toolcalls import parse_tool_calls
from arcsum.transcript import Utterance, parse_transcript

MODEL_REPO = os.environ.get("ARCSUM_MODEL_REPO", "Luigi/qwen35-0.8b-arcsum")
#: Q8_0 — the EVALUATED artifact, and now the only one published.
#: Q4_K_M used to be served here for speed. It was then MEASURED against Q8_0 on the same
#: 40 held-out meetings and is materially worse: the agent's margin over the map-reduce
#: baseline more than halves on ROUGE-1 (+0.077 -> +0.034, wins 29/40 -> 22/40) and its
#: summaries run ~30% shorter (226 vs 320 chars), i.e. it simply records less. A demo
#: running that quant under the card's Q8 numbers would misrepresent the system, so the
#: file was withdrawn from the model repo.
#: Cost: 1.15 GB instead of 688 MB, and slower on Space CPU. `ARCSUM_MODEL_FILE` can
#: still point elsewhere — but measure any replacement before trusting it.
#:
#: **`mixed-e3` was shipped here on 2026-09-02 and ROLLED BACK the same day.** It measured
#: better on every offline gate — revision probe 3/27 -> 8/27, real-ASR "curated" 17/20 ->
#: 19/20, all three G3 gates still passing — and then failed visibly in THIS demo on a real
#: ASR meeting: it kept 1 point where v5 kept 4, emitted 4 `restates dropped` churn events
#: (DROP the only point, re-ADD a near-identical one, five steps running), and synthesised
#: 553 characters of confident strategy prose from that single point, asserting competitive
#: positioning that appears nowhere in the memory.
#:
#: **Two eval failures let that through, and both are about CONFIGURATION, not metrics.**
#:
#: 1. The gates run on `llama-server` with `cache_prompt: false`; this demo runs
#:    llama-cpp-python with the KV cache live across calls. The prompt cache is KNOWN to
#:    change generation (measured: 167 vs 700 characters, same model, same seed,
#:    temperature 0). Under the GATED config `mixed-e3` handles this same transcript fine —
#:    4 points, 0 churn. The regression exists only in the config the product actually
#:    uses, which is the one nothing gates.
#: 2. `asr_gate.py` scores a meeting "curated" by summary LENGTH. A 553-character
#:    confabulation built from one churned point passes that test, so the metric that said
#:    19/20 was rewarding the failure.
#:
#: Do not re-promote `mixed-e3` on offline gate numbers alone. It needs a churn-aware and
#: cache-on measurement first.
MODEL_FILE = os.environ.get("ARCSUM_MODEL_FILE", "Qwen3.5-0.8B.Q8_0.gguf")

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


#: Layers to offload to CUDA. **0 (CPU) is the default** -- GPU is opt-in via
#: `ARCSUM_N_GPU_LAYERS` (-1 offloads every layer). The wheel in requirements.txt is a
#: dynamic-backend cu131 build, so one artifact serves both and this value alone picks the
#: backend at load time.
#:
#: Off by default deliberately: on ZeroGPU an `@spaces.GPU` call CONSUMES QUOTA whether or
#: not the model actually uses the device, so a Space that only needs CPU should not carry
#: the decorator on its hot path. See the conditional wrapping below.
GPU_LAYERS = int(os.environ.get("ARCSUM_N_GPU_LAYERS", "-1"))
#: Whether the UI toggle starts checked. GPU stays OFF by default: ZeroGPU quota is
#: finite and a CPU run costs none of it.
GPU_DEFAULT = os.environ.get("ARCSUM_GPU_DEFAULT", "0") not in ("0", "", "false", "False")


def get_model(n_gpu_layers: int) -> ArcsumModel:
    """Build the model, caching only in CPU mode.

    **A CUDA `Llama` must NOT be cached across ZeroGPU calls.** ZeroGPU attaches a device
    for the duration of an `@spaces.GPU` function and reclaims it on return; a `Llama`
    holding CUDA buffers from a previous allocation is invalid on the next call. In CPU
    mode there is no device to reclaim, so the cache is safe and worth keeping -- it
    avoids re-reading 833 MB per request.
    """
    global _model
    if n_gpu_layers != 0:
        path = hf_hub_download(MODEL_REPO, MODEL_FILE)
        return ArcsumModel(path, n_gpu_layers=n_gpu_layers)
    if _model is None:
        path = hf_hub_download(MODEL_REPO, MODEL_FILE)
        _model = ArcsumModel(path, n_gpu_layers=0)
    return _model


# --- rendering -------------------------------------------------------------------------


#: All colour is expressed through Gradio's own theme variables, so the demo follows the
#: visitor's light/dark preference instead of assuming a white page. The previous layout
#: hardcoded GitHub-light hex values (`#d0d7de`, `#ddf4ff`, `#eaeef2`), which render as
#: near-invisible light-on-light for any visitor whose browser prefers dark.
CSS = """
.ax-panel {
  border: 1px solid var(--border-color-primary);
  border-radius: 10px;
  background: var(--background-fill-secondary);
  display: flex; flex-direction: column;
  height: 440px; overflow: hidden;
  transition: border-color .2s, box-shadow .2s;
}
.ax-panel.ax-active {
  border-color: var(--color-accent);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--color-accent) 18%, transparent);
}
.ax-head {
  display: flex; justify-content: space-between; align-items: center; gap: 8px;
  padding: 9px 12px; border-bottom: 1px solid var(--border-color-primary);
  background: var(--background-fill-primary);
  position: sticky; top: 0;
}
.ax-title { font-weight: 600; font-size: .95rem; display: flex; align-items: center; gap: 7px; }
.ax-step {
  display: inline-grid; place-items: center; width: 19px; height: 19px;
  border-radius: 50%; font-size: .72rem; font-weight: 700;
  background: var(--color-accent); color: var(--button-primary-text-color, #fff);
}
.ax-badge { font-size: .8rem; opacity: .65; font-variant-numeric: tabular-nums; }
.ax-body { padding: 10px 12px; overflow-y: auto; flex: 1; }
.ax-dim { opacity: .55; font-size: .85em; font-variant-numeric: tabular-nums; }

/* memory */
.ax-arc {
  border-left: 3px solid var(--color-accent);
  background: color-mix(in srgb, var(--color-accent) 10%, transparent);
  padding: 7px 9px; border-radius: 0 6px 6px 0; margin-bottom: 10px; line-height: 1.5;
}
.ax-point {
  padding: 4px 2px; border-bottom: 1px solid var(--border-color-primary);
  font-size: .92em; line-height: 1.5;
}
.ax-point:last-child { border-bottom: 0; }
.ax-idx {
  display: inline-block; min-width: 1.4em; opacity: .45;
  font-variant-numeric: tabular-nums; font-size: .85em;
}

/* progress */
.ax-prog { margin: 2px 0 4px; }
.ax-track {
  background: var(--background-fill-secondary); border-radius: 999px; height: 7px;
  overflow: hidden; border: 1px solid var(--border-color-primary);
}
.ax-fill {
  background: var(--color-accent); height: 100%; border-radius: 999px;
  transition: width .25s ease;
}
.ax-meta {
  display: flex; gap: 9px; align-items: center; margin-top: 6px;
  font-size: .87rem; opacity: .8;
}
.ax-mode {
  border-radius: 999px; padding: 1px 8px; font-size: .74rem; font-weight: 600;
  letter-spacing: .02em; color: #fff;
}
.ax-mode-gpu { background: #1a7f37; }
.ax-mode-cpu { background: #5b6570; }

/* the summary is the product, so it gets the full width and a taller line height */
#ax-summary .ax-panel { height: auto; min-height: 190px; }
#ax-summary .ax-body { line-height: 1.85; font-size: 1.03em; }

/* header */
.ax-hero { padding: 10px 2px 4px; }
.ax-hero h1 {
  margin: 0; font-size: 1.5rem; font-weight: 700; letter-spacing: -.015em;
  display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap;
}
.ax-hero h1 span { font-size: .95rem; font-weight: 500; opacity: .6; letter-spacing: 0; }
.ax-hero p { margin: 7px 0 0; opacity: .78; max-width: 78ch; line-height: 1.6; }
.ax-flow {
  display: flex; align-items: center; gap: 7px; flex-wrap: wrap;
  margin-top: 12px; font-size: .84rem;
}
.ax-flow span {
  display: inline-flex; align-items: center; gap: 6px;
  border: 1px solid var(--border-color-primary); border-radius: 999px;
  padding: 3px 11px 3px 5px; background: var(--background-fill-secondary);
}
.ax-flow b {
  display: inline-grid; place-items: center; width: 17px; height: 17px;
  border-radius: 50%; font-size: .7rem;
  background: var(--color-accent); color: var(--button-primary-text-color, #fff);
}
.ax-flow i { opacity: .4; font-style: normal; }

/* stack the three live panels on narrow screens instead of crushing them */
@media (max-width: 860px) {
  .ax-panel { height: 320px; }
}
"""

def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _panel(title: str, status: str, body_html: str, *, active: bool = False,
           step_no: str = "") -> str:
    """One of the three live panels.

    `active` puts a blue rule on the panel currently doing work. With three panels
    updating at different rates it is otherwise genuinely hard to tell where to look --
    the reading step writes to the middle panel, the applier to the right one, and the
    transcript highlight moves on the left.

    **All colour comes from Gradio theme variables, never literal hex.** The panels used
    hardcoded `#d0d7de` borders and a `#ddf4ff` ARC block, which render as light-on-light
    inside a dark-themed Space — invisible to roughly half of visitors, since Gradio
    follows the browser preference by default.

    `step_no` numbers the panel in pipeline order. With four surfaces updating at
    different rates, the single most common confusion is not knowing which one to read
    first; the numbers make the dataflow explicit rather than implied by column position.
    """
    badge = f"<span class='ax-badge'>{_esc(status)}</span>" if status else ""
    step = f"<span class='ax-step'>{_esc(step_no)}</span>" if step_no else ""
    # NOTE: no nested same-type quotes inside the f-string expression. HF Spaces runs
    # Python 3.10, where PEP 701 does not apply and `f"{" x" if c else ""}"` is a hard
    # SyntaxError at import — the Space fails to boot. This venv is 3.12 and accepts it,
    # so the break is invisible locally; `test_demo_app.py` now parses this file at
    # feature_version (3, 10) to catch the class.
    active_cls = " ax-active" if active else ""
    return (
        f"<div class='ax-panel{active_cls}'>"
        f"<div class='ax-head'><span class='ax-title'>{step}{_esc(title)}</span>"
        f"{badge}</div><div class='ax-body'>{body_html}</div></div>"
    )


def render_transcript_html(utterances: list[Utterance], first: int, last: int) -> str:
    if not utterances:
        return _panel("Transcript", "", "<i class='ax-dim'>No transcript loaded.</i>", step_no="1")
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
    return _panel("Transcript", label, "".join(rows), step_no="1")


#: One colour per op kind, shared by the live view and the applied-memory view.
_OP_COLOUR = {"Arc": "#0969da", "Add": "#1a7f37", "Drop": "#cf222e"}


def render_ops_html(status: str, raw: str, live: bool) -> str:
    """Render the step's raw output.

    **Protocol-aware.** Every v1.0 checkpoint (`qwen-tools-v5` through `mixed-e3`) emits
    ONE line -- an `update_memory` tool call
    carrying JSON -- not the ADD/DROP/ARC edit lines the previous checkpoint produced. The
    old renderer coloured lines by their leading keyword and fell through to plain grey
    for anything else, so a tool call rendered as a single escaped blob of
    `<tool_call>{"name": ...}` and read on screen as raw markup.

    So: parse it and show the ops. While a step is still STREAMING the JSON is truncated
    and cannot parse, which is not an error -- fall back to the raw text with the wrapper
    stripped and let it wrap, so the user still sees tokens arriving.
    """
    if not raw:
        return _panel("Model output", status, "<i class='ax-dim'>Waiting…</i>", step_no="2",
                      active=live)

    rows: list[str] = []
    ops = parse_tool_calls(raw)
    parsed = [o for o in ops if type(o).__name__ != "Malformed"]
    if parsed:
        for op in parsed:
            kind = type(op).__name__
            rows.append(
                f"<div style='font-family:ui-monospace,monospace;font-size:.9em;"
                f"color:{_OP_COLOUR.get(kind, '#57606a')};padding:1px 0;"
                f"white-space:pre-wrap;word-break:break-word'>{_esc(render_op(op))}</div>"
            )
    else:
        # Streaming, or genuinely malformed. Strip the wrapper so the panel shows the
        # JSON being built rather than the tag noise around it.
        shown = raw.replace("<tool_call>", "").replace("</tool_call>", "").strip()
        rows.append(
            f"<div style='font-family:ui-monospace,monospace;font-size:.85em;"
            f"color:#57606a;white-space:pre-wrap;word-break:break-word;"
            f"opacity:.85'>{_esc(shown)}</div>"
        )
    body = "".join(rows) + ("<span style='opacity:.5'>▌</span>" if live else "")
    return _panel("Model output", status, body, active=live, step_no="2")


def render_memory_html(mem: Memory, n_chunks: int, step: int) -> str:
    arc_tok = heuristic_token_len(mem.arc) if mem.arc else 0
    arc = (
        f"<div class='ax-arc'><b>ARC</b> <span class='ax-dim'>"
        f"{arc_tok}/{ARC_TOKENS} tok</span><br>{_esc(mem.arc)}</div>"
        if mem.arc
        else "<div style='opacity:.5;margin-bottom:8px'><b>ARC</b> — empty</div>"
    )
    pts = (
        "".join(
            f"<div class='ax-point'>"
            f"<span class='ax-idx'>{i + 1}</span> {_esc(p.text)} "
            f"<span class='ax-dim'>({heuristic_token_len(p.text)}/{POINT_TOKENS})</span></div>"
            for i, p in enumerate(mem.points)
        )
        or "<div style='opacity:.5;padding:4px'>no points yet</div>"
    )
    head = (
        f"<b>POINTS</b> <span style='opacity:.6;font-size:.85em'>"
        f"{len(mem.points)}/{POINTS_CAP}</span>"
    )
    status = f"step {step}/{n_chunks}" if n_chunks else ""
    return _panel("External memory", status, arc + head + pts, step_no="3")


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
    return _panel("Final summary", "written from the memory alone", body, active=live,
                  step_no="4")


def _progress(pct: float, label: str, *, elapsed: float | None = None,
              mode: str | None = None) -> str:
    """Progress bar with elapsed time and a CPU/GPU badge.

    Elapsed time is not decoration: a CPU run of the 6-chunk example takes minutes, and
    the commonest way a live demo is misread is a user deciding it has hung. Showing the
    clock move makes "slow" legible as slow rather than broken.
    """
    bits = []
    if mode:
        bits.append(f"<span class='ax-mode ax-mode-{mode.lower()}'>{_esc(mode)}</span>")
    bits.append(f"<span>{_esc(label)}</span>")
    if elapsed is not None:
        bits.append(f"<span style='opacity:.55'>{elapsed:.0f}s</span>")
    return (
        f"<div class='ax-prog'><div class='ax-track'>"
        f"<div class='ax-fill' style='width:{pct:.1f}%'></div></div>"
        f"<div class='ax-meta'>{''.join(bits)}</div></div>"
    )


# --- the run loop ----------------------------------------------------------------------

#: (run_btn, example_dd, stop_btn, gpu_toggle) -- the tuple width must match the tail of
#: `run_demo`'s outputs list exactly, or Gradio silently misassigns updates to components.
_BUSY = (
    gr.update(interactive=False),
    gr.update(interactive=False),
    gr.update(interactive=True),
    gr.update(interactive=False),
)
_IDLE = (
    gr.update(interactive=True),
    gr.update(interactive=True),
    gr.update(interactive=False),
    gr.update(interactive=True),
)


def _run(custom_transcript: str, example_name: str, n_gpu_layers: int, log: dict):
    """`log` is a `gr.State` dict MUTATED IN PLACE.

    In place, and passed as an input rather than yielded as an output, so the export
    button can read it without widening every one of the generator's yield tuples --
    they must all stay the same width or Gradio misassigns updates to components.
    `gr.State` is per-session, so two concurrent users cannot see each other's run.
    """
    mode = "GPU" if n_gpu_layers != 0 else "CPU"
    # NOT `t0`: the chunk loop below reassigns `t0` for per-step timing, which would make
    # the progress bar's "elapsed" restart every step instead of tracking the whole run.
    t_run = time.time()
    log.clear()
    log.update({
        "schema": 1,
        "model": {"repo": MODEL_REPO, "file": MODEL_FILE, "n_gpu_layers": n_gpu_layers,
                  "mode": mode, "plain_chatml": True},
        "harness": {
            "chunk_tokens": CHUNK_TOKENS, "arc_tokens": ARC_TOKENS,
            "point_tokens": POINT_TOKENS, "points_cap": POINTS_CAP,
            "max_tokens_step": MAX_TOKENS_STEP, "max_tokens_synth": MAX_TOKENS_SYNTH,
            "synth_repeat_penalty": SYNTH_REPEAT_PENALTY,
        },
        "example": example_name, "steps": [], "started_at": t_run,
    })

    def _bar(pct: float, label: str) -> str:
        """Named `_bar`, not `prog`: the reading loop already binds a local `prog` to a
        rendered string, and shadowing this closure with it makes the SECOND chunk raise
        `'str' object is not callable`."""
        return _progress(pct, label, elapsed=time.time() - t_run, mode=mode)

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
        yield (*empty, _bar(0, f"Transcript format error: {exc}"), *_IDLE)
        return

    chunks = list(iter_chunks(utterances, budget=CHUNK_TOKENS, token_len=heuristic_token_len))
    mem = Memory(token_len=heuristic_token_len)
    sys_step = tool_step_system_prompt()

    yield (
        render_transcript_html(utterances, -1, -1),
        render_ops_html("", "", False),
        render_memory_html(mem, len(chunks), 0),
        render_prose_html("", False),
        _bar(0, "Loading the model…" if _model is None or n_gpu_layers else "Starting…"),
        *_BUSY,
    )
    model = get_model(n_gpu_layers)

    consecutive_nops = 0
    idx = 0
    for ci, chunk in enumerate(chunks):
        first = idx
        last = idx + len(chunk.utterances) - 1
        idx = last + 1
        t_html = render_transcript_html(utterances, first, last)
        m_html = render_memory_html(mem, len(chunks), ci)
        status = f"step {ci + 1}/{len(chunks)}"
        prog = _bar(ci / len(chunks) * 90, f"{status} — reading…")

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
        parsed = parse_tool_calls(raw)
        outcome = apply_ops(mem, parsed, chunk, consecutive_nops=consecutive_nops)
        consecutive_nops = consecutive_nops + 1 if outcome.nop_collapse else 0

        # Per-op verdicts are the point of the export: "the model emitted X and the
        # harness refused it because Y" is the question a debug log has to answer, and
        # it is invisible in the UI, which only shows what survived.
        log["steps"].append({
            "index": ci,
            "chunk_tokens": chunk.tokens,
            "system": sys_step,
            "prompt": user,
            "raw": raw,
            "seconds": round(elapsed, 2),
            "ops": [
                {"op": render_op(a.op), "applied": a.applied,
                 "reason": a.reason, "note": a.note}
                for a in outcome.results
            ],
            "nop_collapse": outcome.nop_collapse,
            "memory_after": {"arc": mem.arc, "points": [pt.text for pt in mem.points]},
        })

        done = f"{status} ({elapsed:.1f}s)"
        yield (
            t_html,
            render_ops_html(done, raw, False),
            render_memory_html(mem, len(chunks), ci + 1),
            render_prose_html("", False),
            _bar((ci + 1) / len(chunks) * 90, done),
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
        _bar(92, "SYNTHESIZE — writing the summary…"),
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
            _bar(96, "SYNTHESIZE — writing the summary…"),
            *_BUSY,
        )

    prose = finalize(_cut_at_turn_end(prose_raw), token_len=heuristic_token_len)
    log["synthesis"] = {
        "system": synth_system_prompt(),
        "prompt": build_synth_prompt(mem),
        "raw": prose_raw,
        # `finalize` can rewrite or reject prose (SPEC 3's output contract), so the raw
        # and final forms are both kept -- a mismatch between them IS the bug, when there
        # is one, and only the final form is visible in the UI.
        "final": prose.text,
        "refusals": list(getattr(prose, "refusals", []) or []),
    }
    log["total_seconds"] = round(time.time() - t_run, 2)
    yield (
        final_t,
        render_ops_html("reading done", "", False),
        final_m,
        render_prose_html(prose.text, False),
        _bar(100, "done"),
        *_IDLE,
    )


# --- UI --------------------------------------------------------------------------------

# --- ZeroGPU wiring --------------------------------------------------------------------
#
# ZeroGPU attaches a device only for the duration of an `@spaces.GPU` call, so when GPU is
# enabled the WHOLE agent loop must sit inside one -- reading steps and synthesis alike.
# `spaces.GPU` supports generators, which is what keeps the live streaming UI working.
#
# `duration` is the wall-clock ceiling for one call; exceeding it kills the request. A
# meeting is ~15 reading steps plus a synthesis, and `get_model` RELOADS the 833 MB model
# per call in GPU mode (a CUDA `Llama` cannot outlive its allocation), so the budget must
# cover that load too.
#
# When GPU is OFF the decorator is deliberately NOT applied to `run_demo`: it would consume
# ZeroGPU quota for a call that never touches the device. A no-op stub keeps the Space
# bootable, since ZeroGPU hard-fails at startup unless it detects at least one such
# function.
@spaces.GPU(duration=int(os.environ.get("ARCSUM_GPU_DURATION", "180")))
def _run_gpu(custom_transcript: str, example_name: str, log: dict):
    """GPU entry point. ZeroGPU attaches a device for the duration of THIS call, so the
    whole agent loop runs inside it and the model is built here (never cached -- the
    device is reclaimed on return, which would invalidate a cached CUDA model)."""
    yield from _run(custom_transcript, example_name, GPU_LAYERS, log)


def _run_cpu(custom_transcript: str, example_name: str, log: dict):
    """CPU entry point, deliberately NOT decorated. `@spaces.GPU` consumes ZeroGPU quota
    on every call whether or not the device is touched, so routing CPU runs through a
    decorated function would bill GPU time for CPU work. Keeping two entry points is the
    only way to make the toggle free when it is off."""
    yield from _run(custom_transcript, example_name, 0, log)


def export_log(log: dict):
    """Write the session's run log to a temp file and hand Gradio the path.

    Returns a DownloadButton update rather than a bare path so the button can also carry
    a useful filename. Refuses politely when there is nothing to export -- an empty file
    is a worse debugging experience than a clear message.
    """
    if not log or not log.get("steps"):
        gr.Warning("Nothing to export yet — run a transcript first.")
        return gr.update()
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(log.get("started_at", time.time())))
    path = pathlib.Path(tempfile.gettempdir()) / f"arcsum-debug-{stamp}.json"
    path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return gr.update(value=str(path))


def run_demo(custom_transcript: str, example_name: str, use_gpu: bool, log: dict):
    """Dispatch on the UI toggle. `_run_gpu` is still a real `@spaces.GPU` function, so
    ZeroGPU sees one at startup and the Space boots -- the old no-op stub is no longer
    needed."""
    yield from (_run_gpu if use_gpu else _run_cpu)(custom_transcript, example_name, log)


with gr.Blocks(title="arcsum — live agentic zh-TW meeting summarizer") as demo:
    gr.HTML(
        "<div class='ax-hero'>"
        "<h1>arcsum <span>live agentic meeting summarizer</span></h1>"
        "<p>A 1B model reads a Traditional-Chinese transcript in ~2,500-token chunks, "
        "curating a two-slot external memory as it goes. <b>No conversation history "
        "crosses steps</b> — that memory is the only thing carried forward, and the final "
        "summary is written from it alone.</p>"
        "<div class='ax-flow'>"
        "<span><b>1</b> Transcript</span><i>→</i>"
        "<span><b>2</b> Tool call</span><i>→</i>"
        "<span><b>3</b> Memory</span><i>→</i>"
        "<span><b>4</b> Summary</span>"
        "</div></div>"
    )
    with gr.Row(equal_height=True):
        example_dd = gr.Dropdown(
            choices=list(EXAMPLES.keys()),
            value=(list(EXAMPLES.keys()) or [None])[0],
            label="Transcript",
            scale=5,
        )
        with gr.Column(scale=2, min_width=170):
            run_btn = gr.Button("▶  Run", variant="primary", size="lg")
            stop_btn = gr.Button("■  Stop", interactive=False, size="sm")
        with gr.Column(scale=2, min_width=170):
            gpu_toggle = gr.Checkbox(
                value=GPU_DEFAULT,
                label="GPU acceleration",
                info="ZeroGPU — uses quota. Off = CPU, free but slower.",
            )
            export_btn = gr.DownloadButton("⬇  Debug log", size="sm")

    #: Per-session run log, mutated in place by `_run` and read by `export_log`. A
    #: module-level dict would leak one visitor's transcript into another's download.
    log_state = gr.State({})

    progress_html = gr.HTML(_progress(0, "Pick a transcript and press Run."))

    with gr.Accordion(
        "Paste your own zh-TW transcript (format: `speaker: text`, one per line)", open=False
    ):
        transcript_box = gr.Textbox(
            lines=8, show_label=False, placeholder="S1: 我們開始今天的會議。\nS2: 好的。"
        )

    with gr.Row(equal_height=True):
        transcript_html = gr.HTML(render_transcript_html([], -1, -1))
        ops_html = gr.HTML(render_ops_html("", "", False))
        memory_html = gr.HTML(render_memory_html(Memory(token_len=heuristic_token_len), 0, 0))
    with gr.Row(elem_id="ax-summary"):
        prose_html = gr.HTML(render_prose_html("", False))

    gr.Markdown(
        f"""
### What you are watching

The **middle panel** is the model's raw output — one batched `update_memory` tool call
per chunk, never prose, until the very end. The **right panel** is the harness's memory
after applying those ops deterministically (token caps enforced, malformed or unsafe ops
refused).

The design bet is what the memory buys: **a later chunk can overturn an earlier
conclusion.** Map-reduce structurally cannot do that — each window is summarised
independently, so a decision reversed at minute 90 never reaches the summary of minute 10.

### Honest status

Running **`qwen-tools-v5`**, measured on **40 held-out meetings** it was never trained or
tuned on. Against a fair map-reduce baseline — same model, same chunk size — it wins all
three ROUGE metrics (28/40, 29/40, 35/40) and on faithfulness by 16 inversions vs 58.

It **fails the revision probe: 3 of 27** scenarios. Given a decision reversed later in the
meeting it may report the superseded decision as if it still stood. Under an all-or-nothing
rule the recorded decision is *ship the baseline*; revision is not a capability to rely on
here.

**A newer checkpoint was served here briefly and rolled back**, which is worth stating
because it is the honest shape of this problem. `mixed-e3` beat v5 on every offline gate —
including nearly tripling the revision probe — and then, on a real ASR meeting in this very
demo, kept one memory point where v5 kept four and wrote 553 characters of confident prose
out of it. The gates missed it because they run with the model's prompt cache disabled
while this demo runs with it on, and because the real-ASR metric scored summaries by
LENGTH, which a confabulation passes easily.

Two further caveats: **faithfulness per-claim favours the baseline** — the agent wins on
absolute inversions partly because it asserts far fewer claims; and on-device latency is
19.0 min/meeting on an Oppo Reno 7 against a 20-minute ceiling, a 3% margin, with a
contended run measured at 21.6 min. Full detail:
[model card](https://hf.co/{MODEL_REPO}).

**This demo runs Q8_0 — the same artifact every number above was measured on.** A smaller
Q4_K_M build was withdrawn after measurement showed it keeps under half the quality margin.
Expect several seconds per chunk on Space CPU; that is the cost of running the real thing.
"""
    )

    ev = run_btn.click(
        run_demo,
        [transcript_box, example_dd, gpu_toggle, log_state],
        [
            transcript_html,
            ops_html,
            memory_html,
            prose_html,
            progress_html,
            run_btn,
            example_dd,
            stop_btn,
            gpu_toggle,
        ],
    )
    export_btn.click(export_log, [log_state], [export_btn])
    stop_btn.click(None, None, None, cancels=[ev])

# Guarded so `import app` does not start a server. HF Spaces runs this file as
# __main__, so the Space is unaffected -- but without the guard the module cannot be
# imported by a test or a REPL at all, which is how this demo's smoke test hung.
if __name__ == "__main__":
    # Gradio 6 moved `css` off the Blocks constructor onto `launch()`. Passing it to
    # Blocks only emits a UserWarning and silently drops the stylesheet, which is how a
    # theme-aware redesign can look like it landed while every panel still renders with
    # default styling.
    demo.queue(max_size=8).launch(css=CSS)
