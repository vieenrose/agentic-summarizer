/* arcsum harness — JavaScript port of the Python package (src/arcsum/).
 *
 * This is a PORT, not a re-imagining: the prompts are byte-identical to
 * `arcsum.prompts`, and the tokenizer estimate, chunker, memory caps, eviction and op
 * grammar all follow their Python definitions. That fidelity is the whole point — the
 * model was fine-tuned against these exact strings and this exact memory rendering, so
 * a "close enough" port would be feeding it out-of-distribution input and the demo
 * would misrepresent the system.
 *
 * Ported against arcsum @ PROMPT_VERSION sys-v1 / TOKENIZE_VERSION chartok-v1.
 * If either constant changes in Python, this file must be re-derived.
 */

export const PROMPT_VERSION = "sys-v1";

// --- tokens.py ------------------------------------------------------------------------

const IDEOGRAPH_RANGES = [
  [0x3400, 0x4dbf], // CJK Unified Ideographs Extension A
  [0x4e00, 0x9fff], // CJK Unified Ideographs (URO)
  [0xf900, 0xfaff], // CJK Compatibility Ideographs
];
const KANA_RANGES = [[0x3040, 0x30ff]];

function isCJK(cp) {
  for (const [lo, hi] of IDEOGRAPH_RANGES) if (cp >= lo && cp <= hi) return true;
  for (const [lo, hi] of KANA_RANGES) if (cp >= lo && cp <= hi) return true;
  return false;
}

/** NON-NORMATIVE budget estimate: ~1 token per CJK char, ~4 chars/token otherwise.
 *  Mirrors `tokens.heuristic_token_len`, including its integer ceiling division. */
export function tokenLen(text) {
  if (!text) return 0;
  let cjk = 0;
  let total = 0;
  for (const ch of text) {
    total += 1;
    if (isCJK(ch.codePointAt(0))) cjk += 1;
  }
  return cjk + Math.floor((total - cjk + 3) / 4);
}

/** NFKC + whitespace collapse (`tokens.normalise`). */
export function normalise(text) {
  return text.normalize("NFKC").split(/\s+/).filter(Boolean).join(" ");
}

// --- transcript.py --------------------------------------------------------------------

const SEP = ": ";
const MAX_SPEAKER_LEN = 40;
const UNK = "UNK";

/** Split on the FIRST ": ". NEVER throws — a non-conforming line becomes (UNK, line),
 *  which is what keeps `parseTranscript` total. */
export function parseLine(line) {
  const idx = line.indexOf(SEP);
  if (idx === -1 || idx > MAX_SPEAKER_LEN || idx === 0) return { speaker: UNK, text: line };
  return { speaker: line.slice(0, idx), text: line.slice(idx + SEP.length) };
}

export function parseTranscript(text) {
  return text
    .split(/\r?\n/)
    .filter((l) => l.trim())
    .map(parseLine);
}

export function renderUtterance(u) {
  return `${u.speaker}${SEP}${u.text}`;
}

// --- chunker.py -----------------------------------------------------------------------

export const CHUNK_TOKENS = 2500;
const OVERLAP_LINES = 2;
const SPLIT_SLACK = 64;
const NEWLINE_COST = 1;

function splitLong(u, budget) {
  const overhead = tokenLen(`${u.speaker}${SEP}`) + NEWLINE_COST;
  const room = budget - overhead;
  if (room <= 0 || tokenLen(renderUtterance(u)) + NEWLINE_COST <= budget) return [u];

  const words = u.text.split(" ");
  const units = words.length > 1 ? words : Array.from(u.text);
  const joiner = words.length > 1 ? " " : "";

  const pieces = [];
  let current = [];
  for (const unit of units) {
    const candidate = [...current, unit].join(joiner);
    if (current.length && tokenLen(candidate) > room) {
      pieces.push(current.join(joiner));
      current = [unit];
    } else {
      current.push(unit);
    }
  }
  if (current.length) pieces.push(current.join(joiner));

  const out = pieces.filter(Boolean).map((p) => ({ speaker: u.speaker, text: p }));
  return out.length ? out : [u];
}

/** Line-atomic, token-budgeted chunks with a 2-line rewind, per `chunker.iter_chunks`. */
export function iterChunks(utterances, budget = CHUNK_TOKENS) {
  if (!utterances.length) return [];

  let lines = [];
  for (const u of utterances) lines.push(...splitLong(u, budget));

  const chunks = [];
  let index = 0;
  let i = 0;
  while (i < lines.length) {
    const current = [];
    let used = 0;
    while (i < lines.length) {
      const cost = tokenLen(renderUtterance(lines[i])) + NEWLINE_COST;
      if (current.length && used + cost > budget) {
        const room = budget - used;
        const pieces = room > SPLIT_SLACK ? splitLong(lines[i], room) : [];
        if (pieces.length > 1) {
          lines.splice(i, 1, ...pieces);
          continue;
        }
        break;
      }
      current.push(lines[i]);
      used += cost;
      i += 1;
    }
    if (!current.length) break;

    chunks.push({ index, utterances: current, tokens: used });
    index += 1;
    if (i >= lines.length) break;
    // Rewind for overlap, clamped so the cursor always advances.
    i = Math.max(i - OVERLAP_LINES, i - current.length + 1);
  }
  return chunks;
}

export function renderChunk(chunk) {
  return chunk.utterances.map(renderUtterance).join("\n");
}

// --- memory.py ------------------------------------------------------------------------

export const ARC_TOKENS = 80;
export const POINT_TOKENS = 25;
export const POINTS_CAP = 16;
const MIN_PREFIX_TOKENS = 4;

/** Reduce to `cap` items by spreading EVENLY, never head-truncating (SPEC §4.1):
 *  head-truncating a time-ordered list drops the end of the meeting, where decisions
 *  land. Endpoints are always kept. */
export function spread(items, cap) {
  const n = items.length;
  if (n <= cap) return [...items];
  if (cap <= 0) return [];
  if (cap === 1) return [items[n - 1]];
  const seen = [];
  for (let i = 0; i < cap; i++) {
    // Python's round() is banker's rounding; the values here are k*(n-1)/(cap-1) and
    // collisions are resolved by the walk below, so Math.round is equivalent in effect.
    let p = Math.round((i * (n - 1)) / (cap - 1));
    while (seen.includes(p)) p += 1;
    seen.push(Math.min(p, n - 1));
  }
  const uniq = [...new Set(seen)].sort((a, b) => a - b);
  return uniq.map((i) => items[i]);
}

export class Memory {
  constructor() {
    this.arc = "";
    this.points = []; // [{text, chunk}]
  }

  /** Refuses empty, over-length, or UNCHANGED text. "Unchanged" mirrors `add_point`'s
   *  duplicate refusal: rewriting the arc to what it already says does no work, and
   *  reporting it as a substantive edit hides a stalled step. */
  setArc(text) {
    const cleaned = text.split(/\s+/).filter(Boolean).join(" ");
    if (!cleaned) return "empty arc";
    const n = tokenLen(cleaned);
    if (n > ARC_TOKENS) return `arc too long (${n} > ${ARC_TOKENS} tokens)`;
    if (this.arc && normalise(cleaned) === normalise(this.arc)) return "arc unchanged";
    this.arc = cleaned;
    return null;
  }

  addPoint(text, chunk) {
    const cleaned = text.split(/\s+/).filter(Boolean).join(" ");
    if (!cleaned) return "empty point";
    const n = tokenLen(cleaned);
    if (n > POINT_TOKENS) return `point too long (${n} > ${POINT_TOKENS} tokens)`;
    const key = normalise(cleaned);
    if (this.points.some((p) => normalise(p.text) === key)) return "duplicate point";
    this.points.push({ text: cleaned, chunk });
    return null;
  }

  /** Ambiguity is refusal: a prefix matching more than one point does NOT guess. */
  find(prefix) {
    const key = normalise(prefix);
    if (tokenLen(key) < MIN_PREFIX_TOKENS) return { index: null, reason: "prefix too short" };
    const hits = [];
    this.points.forEach((p, i) => {
      if (normalise(p.text).startsWith(key)) hits.push(i);
    });
    if (hits.length === 0) return { index: null, reason: "no point matches that prefix" };
    if (hits.length > 1) return { index: null, reason: "prefix is ambiguous" };
    return { index: hits[0], reason: null };
  }

  dropPoint(prefix) {
    const { index, reason } = this.find(prefix);
    if (index === null) return reason;
    this.points.splice(index, 1);
    return null;
  }

  enforceCaps() {
    if (this.points.length > POINTS_CAP) this.points = spread(this.points, POINTS_CAP);
  }
}

// --- render.py ------------------------------------------------------------------------

const EMPTY = "-";

export function renderMemory(mem) {
  const lines = [`ARC: ${mem.arc || EMPTY}`, "POINTS:"];
  if (mem.points.length) lines.push(...mem.points.map((p) => `- ${p.text}`));
  else lines.push(EMPTY);
  return lines.join("\n") + "\n";
}

// --- ops.py ---------------------------------------------------------------------------

const NOP_RE = /^NOP[\s。．.]*$/i;
const ARC_RE = /^ARC\s*[:：]\s*(.*)$/i;
const DROP_RE = /^DROP\s*(?:«(.*?)»|<<(.*?)>>|「(.*?)」|"(.*?)")\s*$/i;
const ADD_RE = /^ADD\s*(?:([-–—])\s*)?(.*)$/i;
/** A hallucinated `[m:ss]` anchor. v2 has no timestamps, but a 1B model may emit one
 *  from pretraining exposure; peeling it beats letting it corrupt the point. */
const JUNK_ANCHOR = /\s*[[［]\s*\d+\s*[:：]\s*\d{2}(?::\d{2})?\s*[\]］]\s*$/;

const stripJunk = (s) => s.replace(JUNK_ANCHOR, "").trim();

/** Line-local, first-match-wins in the order NOP -> ARC -> DROP -> ADD -> Malformed.
 *  NEVER throws. */
export function parseOps(text) {
  if (!text) return [];
  const ops = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) continue;
    ops.push(parseOpLine(line));
  }
  return ops;
}

function parseOpLine(line) {
  if (NOP_RE.test(line)) return { kind: "NOP", raw: line };

  let m = line.match(ARC_RE);
  if (m) {
    const text = stripJunk(m[1].trim());
    return text
      ? { kind: "ARC", text, raw: line }
      : { kind: "MALFORMED", reason: "empty arc", raw: line };
  }

  m = line.match(DROP_RE);
  if (m) {
    const prefix = (m[1] ?? m[2] ?? m[3] ?? m[4] ?? "").trim();
    return prefix
      ? { kind: "DROP", prefix, raw: line }
      : { kind: "MALFORMED", reason: "empty prefix", raw: line };
  }

  m = line.match(ADD_RE);
  if (m) {
    const point = stripJunk((m[2] ?? "").trim());
    return point
      ? { kind: "ADD", point, raw: line }
      : { kind: "MALFORMED", reason: "empty point", raw: line };
  }

  return { kind: "MALFORMED", reason: "unrecognised op", raw: line };
}

// --- guards.py (applier) --------------------------------------------------------------

/** Apply a step's ops IN PLACE, in emission order, returning a verdict per op.
 *  A refused op leaves memory unchanged. */
export function applyOps(mem, ops, chunkIndex) {
  const results = [];
  for (const op of ops) {
    if (op.kind === "NOP") {
      results.push({ op, ok: true, reason: null });
    } else if (op.kind === "MALFORMED") {
      results.push({ op, ok: false, reason: op.reason });
    } else if (op.kind === "ARC") {
      const reason = mem.setArc(op.text);
      results.push({ op, ok: reason === null, reason });
    } else if (op.kind === "ADD") {
      const reason = mem.addPoint(op.point, chunkIndex);
      results.push({ op, ok: reason === null, reason });
    } else if (op.kind === "DROP") {
      const reason = mem.dropPoint(op.prefix);
      results.push({ op, ok: reason === null, reason });
    }
  }
  mem.enforceCaps();
  return results;
}

// --- prompts.py (BYTE-IDENTICAL to the Python) ----------------------------------------

const CAPS_LINE = `ARC 上限 ${ARC_TOKENS} 個字；POINTS 最多 ${POINTS_CAP} 項，每項上限 ${POINT_TOKENS} 個字`;
const PREFIX_RULE = `前綴至少需要 ${MIN_PREFIX_TOKENS} 個字`;
export const PROSE_MAX_TOKENS = 1000;

export const STEP_SYS = `你是一個會議記錄助手，正在閱讀一份會議逐字稿的其中一段（CHUNK），並維護一份精簡的記憶（MEMORY）。

記憶由兩部分組成（${CAPS_LINE}）：
ARC：1 到 3 句話，描述會議目前為止的發展脈絡。
POINTS：關鍵論點、決議或承諾的清單。

閱讀完這段 CHUNK 後，只能用以下四種指令回覆，每行一個指令：

ADD - <重點內容>       新增一項重點
DROP «<前綴>»          移除先前記錄中，開頭符合此前綴的重點（${PREFIX_RULE}）
ARC: <文字>            以新內容取代目前的 ARC 摘要
NOP                    這段內容沒有值得記錄的新資訊

規則：
- 只能輸出上述四種指令，不要輸出其他文字、標點符號、時間戳記或說明。
- 若這段內容推翻或修正了先前的重點，先 DROP 舊的重點，再 ADD 新的重點——不要同時保留互相矛盾的兩項重點。
- 全部使用繁體中文書寫。`;

export const SYNTH_SYS = `你是一個會議記錄助手。以下是目前累積的會議記憶（ARC 與 POINTS）。

請根據這份記憶，寫出一段流暢連貫的繁體中文摘要，不超過 ${PROSE_MAX_TOKENS} 個字：
- 不使用條列式、不使用小標題、不加時間戳記。
- 只寫一段連續的文字，讀起來像一篇完整的會議摘要，而不是重點清單。
- 全部使用繁體中文書寫。`;

export function buildStepPrompt(mem, chunk) {
  return `MEMORY:\n${renderMemory(mem)}\nCHUNK:\n${renderChunk(chunk)}\n`;
}

export function buildSynthPrompt(mem) {
  return `MEMORY:\n${renderMemory(mem)}\n`;
}

// --- prose.py (the cleanup the product output actually goes through) ------------------

const BULLET_LINE = /^\s*(?:[-*•▪]|\d+[.)、])\s*/;
const HEADING_LINE = /^\s*#{1,6}\s*/;
/** A hallucinated harness-format label leaking into the prose ("ARC: ", "POINTS: "). */
const LABEL_LINE = /^\s*(?:TITLE|SUMMARY|ARC|POINTS|DECISIONS|ACTIONS|OPEN|TOPICS)\s*[:：]\s*/i;
const JUNK_ANCHOR_ANY = /\s*[[［]\s*\d+\s*[:：]\s*\d{2}(?::\d{2})?\s*[\]］]\s*/g;
const MD_EMPHASIS = /[*_`]{1,3}/g;

/** Strip bullets/headings/labels/anchors and collapse to one flowing block, per
 *  `prose.finalize`. Lines are joined with a SPACE and empty lines are NOT pre-filtered
 *  — the final whitespace collapse handles them. Joining with "" instead silently welds
 *  adjacent sentences together, which a differential test against the Python caught. */
export function finalizeProse(raw) {
  const lines = raw
    .split(/\r?\n/)
    .map((l) => l.replace(LABEL_LINE, "").replace(HEADING_LINE, "").replace(BULLET_LINE, ""));
  let text = lines.join(" ");
  text = text.replace(JUNK_ANCHOR_ANY, " ");
  text = text.replace(MD_EMPHASIS, "");
  text = text.split(/\s+/).filter(Boolean).join(" ");
  return { text, tokens: tokenLen(text), chars: text.length };
}
