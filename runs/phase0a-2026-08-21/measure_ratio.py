"""Phase 0a item 2: measure the en->zh-TW token ratio under MiniCPM5's real tokenizer.

Uses real MeetingBank English segment text (cached huuuyeah/meetingbank derivative --
fine for THIS units experiment per SPEC, which explicitly allows "any available
model" and doesn't require the authoritative Zenodo release for a ratio-only
measurement). Translation model: the already-running local Qwen3.8-27B llama-server
(port 8082) -- not TranslateGemma, which SPEC explicitly says this measurement must
not wait for.
"""
import json
import sys
import urllib.request

from datasets import load_dataset

sys.path.insert(0, "/home/luigi/agentic-summarizer/src")
from arcsum.tokens import hf_token_len

SYS_PROMPT = (
    "You are a professional English-to-Traditional-Chinese (zh-TW) translator. "
    "Translate the user text into fluent zh-TW COMPLETELY, sentence by sentence, "
    "omitting nothing. Output ONLY the translation, nothing else. Do not summarize "
    "or shorten."
)


def translate(text: str) -> str:
    body = {
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0,
        "max_tokens": 8000,
    }
    req = urllib.request.Request(
        "http://127.0.0.1:8082/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    choice = data["choices"][0]
    content = choice["message"].get("content") or ""
    return content, choice.get("finish_reason")


def main():
    ds = load_dataset("huuuyeah/meetingbank", split="train")
    # Pick 5 segments in a moderate size band (real content, bounded translation cost).
    candidates = [i for i in range(len(ds)) if 1500 <= len(ds[i]["transcript"]) <= 6000]
    picked = candidates[:5]
    print(f"picked indices: {picked}", file=sys.stderr)

    token_len = hf_token_len("openbmb/MiniCPM5-1B")

    total_en = 0
    total_zh = 0
    rows = []
    for i in picked:
        en_text = ds[i]["transcript"]
        en_tokens = token_len(en_text)
        zh_text, finish_reason = translate(en_text)
        zh_tokens = token_len(zh_text)
        ratio = zh_tokens / en_tokens if en_tokens else 0.0
        rows.append(
            {
                "idx": i,
                "uid": ds[i]["uid"],
                "en_chars": len(en_text),
                "en_tokens": en_tokens,
                "zh_chars": len(zh_text),
                "zh_tokens": zh_tokens,
                "ratio": ratio,
                "finish_reason": finish_reason,
            }
        )
        total_en += en_tokens
        total_zh += zh_tokens
        print(f"[{i}] en_tokens={en_tokens} zh_tokens={zh_tokens} ratio={ratio:.3f} finish={finish_reason}", file=sys.stderr)

    overall_ratio = total_zh / total_en if total_en else 0.0
    print(json.dumps({"rows": rows, "total_en": total_en, "total_zh": total_zh, "overall_ratio": overall_ratio}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
