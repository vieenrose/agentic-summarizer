/* arcsum WASM demo — runs the model entirely in your browser via wllama.
 *
 * The harness in arcsum.js is a verified port of the Python package (differential-tested
 * byte-for-byte on prompts, chunking, op parsing, memory eviction and prose cleanup), so
 * what runs here is the real mechanism, not a re-enactment.
 */

import { Wllama } from "https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.5/esm/index.js";
import * as A from "./arcsum.js";
import { EXAMPLES } from "./examples.js";

const CDN = "https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.5/esm/";
const CONFIG_PATHS = {
  "single-thread/wllama.wasm": CDN + "single-thread/wllama.wasm",
  "multi-thread/wllama.wasm": CDN + "multi-thread/wllama.wasm",
};

const MODEL_REPO = "Luigi/minicpm5-1b-arcsum";
// Q8_0 only. Q4_K_M was dropped after being MEASURED against it on the 40 held-out
// meetings: the agent's margin over the map-reduce baseline more than halved on rouge1
// (+0.077 -> +0.034, wins 29/40 -> 22/40), and Q4 summaries came out ~30% shorter
// (226 vs 320 chars), i.e. it records less. Shipping a quant that weak behind the model
// card's Q8 numbers would misrepresent the system.
// COST: 1.15 GB into the wasm heap instead of 688 MB. This front-end is NOT deployed;
// if it ever is, verify it still loads on a modest device first.
const MODEL_FILE = "MiniCPM5-1B.Q8_0.gguf";
const N_CTX = 4096;
const MAX_TOKENS_STEP = 512;
const MAX_TOKENS_SYNTH = 700;
/** Prose calls ONLY. Reading steps emit a fixed op vocabulary, so a repetition penalty
 *  there punishes the literal ADD/DROP/ARC tokens the format requires. */
const SYNTH_REPEAT_PENALTY = 1.1;

/** Chat-template turn markers. Stopping on these is what keeps a second, duplicate
 *  summary out of the panel — measured on the Gradio build before this one. */
const TURN_MARKERS = ["<|im_end|>", "<|im_start|>", "\nassistant\n", "\nuser\n"];

const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

function cutAtTurnEnd(text) {
  let out = text;
  for (const m of TURN_MARKERS) {
    const i = out.indexOf(m);
    if (i !== -1) out = out.slice(0, i);
  }
  return out.trim();
}

let wllama = null;
let running = false;
let cancelled = false;

// --- MiniCPM5 chat template (ChatML, thinking disabled) --------------------------------
// The <think>\n\n</think> pair is closed immediately: undefined, the model free-runs into
// its own reasoning mode. This reproduces llama-server's --jinja + enable_thinking=false.
function buildPrompt(system, user) {
  return (
    `<|im_start|>system\n${system}<|im_end|>\n` +
    `<|im_start|>user\n${user}<|im_end|>\n` +
    `<|im_start|>assistant\n<think>\n\n</think>\n\n`
  );
}

// --- rendering -------------------------------------------------------------------------

function renderTranscript(utts, first, last) {
  if (!utts.length) return `<i class="muted">No transcript loaded.</i>`;
  return utts
    .map((u, i) => {
      const on = i >= first && i <= last;
      return `<div class="line${on ? " active" : ""}"><b>${esc(u.speaker)}:</b> ${esc(u.text)}</div>`;
    })
    .join("");
}

function renderOps(raw, live) {
  if (!raw) return `<i class="muted">Waiting…</i>`;
  const body = raw
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean)
    .map((s) => {
      const cls = s.startsWith("ARC") ? "op-arc"
        : s.startsWith("ADD") ? "op-add"
        : s.startsWith("DROP") ? "op-drop" : "op-nop";
      return `<div class="op ${cls}">${esc(s)}</div>`;
    })
    .join("");
  return body + (live ? `<span class="caret">▌</span>` : "");
}

function renderMemory(mem) {
  const arcTok = mem.arc ? A.tokenLen(mem.arc) : 0;
  const arc = mem.arc
    ? `<div class="arc"><b>ARC</b> <span class="muted small">${arcTok}/${A.ARC_TOKENS} tok</span><br>${esc(mem.arc)}</div>`
    : `<div class="muted" style="margin-bottom:8px"><b>ARC</b> — empty</div>`;
  const pts = mem.points.length
    ? mem.points
        .map(
          (p, i) =>
            `<div class="pt"><span class="muted">${i + 1}.</span> ${esc(p.text)} <span class="muted small">(${A.tokenLen(p.text)}/${A.POINT_TOKENS})</span></div>`,
        )
        .join("")
    : `<div class="muted" style="padding:4px">no points yet</div>`;
  return (
    arc +
    `<b>POINTS</b> <span class="muted small">${mem.points.length}/${A.POINTS_CAP}</span>` +
    pts
  );
}

function setStatus(pct, label) {
  $("bar").style.width = `${pct}%`;
  $("status").textContent = label;
}

// --- model ------------------------------------------------------------------------------

async function ensureModel() {
  if (wllama) return wllama;
  setStatus(2, "Loading wllama…");
  wllama = new Wllama(CONFIG_PATHS);
  await wllama.loadModelFromHF(MODEL_REPO, MODEL_FILE, {
    n_ctx: N_CTX,
    n_threads: navigator.hardwareConcurrency > 1 ? Math.min(4, navigator.hardwareConcurrency) : 1,
    progressCallback: ({ loaded, total }) => {
      const pct = total ? (loaded / total) * 100 : 0;
      setStatus(pct * 0.35, `Downloading model… ${(loaded / 1e6) | 0}/${(total / 1e6) | 0} MB (cached after first run)`);
    },
  });
  return wllama;
}

/** Stream a completion, calling onToken with the cumulative text. */
async function generate(system, user, { maxTokens, repeatPenalty = 1.0, onToken }) {
  const prompt = buildPrompt(system, user);
  let acc = "";
  await wllama.createCompletion(prompt, {
    nPredict: maxTokens,
    sampling: { temp: 0.0, penalty_repeat: repeatPenalty },
    onNewToken: (_token, _piece, currentText, optionals) => {
      acc = currentText;
      if (cancelled) optionals.abortSignal();
      else if (TURN_MARKERS.some((m) => acc.includes(m))) optionals.abortSignal();
      else onToken(acc);
    },
  });
  return cutAtTurnEnd(acc);
}

// --- the run loop -------------------------------------------------------------------------

async function run() {
  if (running) return;
  running = true;
  cancelled = false;
  $("run").disabled = true;
  $("stop").disabled = false;
  $("example").disabled = true;

  try {
    const custom = $("custom").value.trim();
    const text = custom || EXAMPLES[$("example").value] || "";
    if (!text.trim()) {
      setStatus(0, "Pick an example or paste a transcript.");
      return;
    }

    const utts = A.parseTranscript(text);
    const chunks = A.iterChunks(utts, A.CHUNK_TOKENS);
    const mem = new A.Memory();

    $("transcript").innerHTML = renderTranscript(utts, -1, -1);
    $("memory").innerHTML = renderMemory(mem);
    $("ops").innerHTML = renderOps("", false);
    $("prose").innerHTML = `<i class="muted">Produced after the last chunk, from the memory alone.</i>`;

    await ensureModel();

    let cursor = 0;
    for (let ci = 0; ci < chunks.length; ci++) {
      if (cancelled) break;
      const chunk = chunks[ci];
      const first = cursor;
      const last = cursor + chunk.utterances.length - 1;
      cursor = last + 1;

      $("transcript").innerHTML = renderTranscript(utts, first, last);
      $("transcript").querySelector(".active")?.scrollIntoView({ block: "center" });
      const pct = 35 + (ci / chunks.length) * 55;
      setStatus(pct, `step ${ci + 1}/${chunks.length} — reading…`);

      const t0 = performance.now();
      const raw = await generate(A.STEP_SYS, A.buildStepPrompt(mem, chunk), {
        maxTokens: MAX_TOKENS_STEP,
        onToken: (t) => { $("ops").innerHTML = renderOps(cutAtTurnEnd(t), true); },
      });
      const secs = ((performance.now() - t0) / 1000).toFixed(1);

      // The real harness: parse, then apply deterministically with caps and guards.
      A.applyOps(mem, A.parseOps(raw), chunk.index);

      $("ops").innerHTML = renderOps(raw, false);
      $("memory").innerHTML = renderMemory(mem);
      setStatus(35 + ((ci + 1) / chunks.length) * 55, `step ${ci + 1}/${chunks.length} done (${secs}s)`);
    }

    if (cancelled) { setStatus(0, "Stopped."); return; }

    // SYNTHESIZE: the memory alone, no transcript.
    setStatus(92, "SYNTHESIZE — writing the summary…");
    const proseRaw = await generate(A.SYNTH_SYS, A.buildSynthPrompt(mem), {
      maxTokens: MAX_TOKENS_SYNTH,
      repeatPenalty: SYNTH_REPEAT_PENALTY,
      onToken: (t) => {
        $("prose").innerHTML = `<div class="prose-text">${esc(cutAtTurnEnd(t))}<span class="caret">▌</span></div>`;
      },
    });
    const prose = A.finalizeProse(proseRaw);
    $("prose").innerHTML =
      `<div class="prose-text">${esc(prose.text)}</div>` +
      `<div class="muted small" style="margin-top:10px">${prose.chars} characters · ~${prose.tokens} tokens</div>`;
    setStatus(100, "done");
  } catch (err) {
    console.error(err);
    setStatus(0, `Error: ${err.message || err}`);
  } finally {
    running = false;
    $("run").disabled = false;
    $("stop").disabled = true;
    $("example").disabled = false;
  }
}

// --- wiring ------------------------------------------------------------------------------

const sel = $("example");
for (const name of Object.keys(EXAMPLES)) {
  const opt = document.createElement("option");
  opt.value = name;
  opt.textContent = name;
  sel.appendChild(opt);
}
$("run").addEventListener("click", run);
$("stop").addEventListener("click", () => { cancelled = true; });
setStatus(0, "Pick a transcript and press Run. The model (~656 MB) downloads once, then caches.");
