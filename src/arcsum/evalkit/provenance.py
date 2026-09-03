"""What configuration produced a number — captured, hashed, and made refusable.

Every failure this module exists to prevent actually happened in this project, and none
of them was caught by process. They were caught by a number looking implausible, which
only works when the wrong number happens to look wrong.

1. **A server that failed to bind, answering as the previous model.** Killing one
   `llama-server` and starting the next in a single command: the kill had not completed,
   the new process exited on a port conflict, and the OLD model served the new
   checkpoint's measurement. Caught only because two checkpoints returned byte-identical
   numbers. `serving_identity()` reads `/props` and pins `model_path`, so a measurement
   always records which file actually answered.

2. **The wrong protocol.** `tools/score_reversals.py` defaulted to `--protocol edit` and
   scored a v1.0 tool-call checkpoint under the edit grammar: a clean-looking 0/27, which
   reads exactly like "this model cannot revise". The true value was 8/27.

3. **A number with no artifact.** A "memory details 10.0 -> 2.8" collapse motivated an
   entire retrain and does not reproduce; the real regression was a third that size. The
   recorded probe scores 3/27 and 12/27 likewise had nothing on disk, and re-ran as 3/27
   and 11/27.

4. **Comparing across corpus builds.** A model card records `v5` at 5/27 on "an
   independent 27-scenario probe"; the same checkpoint measured 3/27 here. The probe
   corpus had been regenerated in between. Nothing flagged the two as incomparable.

5. **Unrecorded checkpoint identity.** Every v1.0 number is a LAST-epoch measurement and
   none says so, which matters because epoch choice moves real-ASR curation by two
   meetings and is not consistent in direction across builds.

**The core idea is `comparison_key()`.** A scorecard's provenance splits in two: the
MODEL, which is what a comparison varies, and EVERYTHING ELSE, which a comparison must
hold fixed. Two scorecards are comparable exactly when their `comparison_key()` values
match. That is a refusal, not a warning — failure 4 above is what a warning gets you.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: Fields that describe the SUBJECT of a comparison rather than its conditions. Excluded
#: from `comparison_key()` — varying them is the entire point of an A/B.
MODEL_FIELDS = ("model_path", "model_sha256", "label", "checkpoint", "epoch")


@dataclass(frozen=True)
class CorpusFingerprint:
    """Identity of a corpus directory: which files, and what is in them.

    Content-hashed, not mtime-stamped. Failure 4 above was a regenerated probe corpus at
    the same path with the same file count — only the bytes differed.
    """

    path: str
    n_files: int
    content_sha256: str

    @classmethod
    def of(cls, directory: Path, pattern: str = "*.txt") -> CorpusFingerprint:
        files = sorted(directory.glob(pattern))
        h = hashlib.sha256()
        for f in files:
            # The NAME is hashed as well as the bytes: two corpora holding the same
            # documents under different meeting ids are not the same corpus, because the
            # paired statistics join on the id.
            h.update(f.name.encode("utf-8"))
            h.update(f.read_bytes())
        return cls(str(directory), len(files), h.hexdigest())


@dataclass(frozen=True)
class Provenance:
    """The complete answer to "what produced this number".

    Construct with `capture()`, which refuses rather than guessing when the server cannot
    be reached — an unreachable server is exactly the state that produced failure 1.
    """

    model_path: str
    model_sha256: str
    protocol: str
    prompt_version: str
    tokenize_version: str
    #: Sampling and serving knobs that change generation. `cache_prompt` is here because
    #: llama.cpp's prompt cache alone turned a 167-character answer into a 700-character
    #: one at temperature=0 with the same seed.
    generation: dict[str, object] = field(default_factory=dict)
    corpora: dict[str, CorpusFingerprint] = field(default_factory=dict)
    label: str = ""
    checkpoint: str = ""
    epoch: str = ""
    code_revision: str = ""

    def comparison_key(self) -> str:
        """Hash of everything a comparison must hold FIXED (i.e. excluding the model).

        Two `Provenance` values with equal keys describe measurements that differ only in
        the thing under test. Unequal keys mean the numbers are not comparable, however
        similar they look.
        """
        payload = {k: v for k, v in _plain(self).items() if k not in MODEL_FIELDS}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def differences(self, other: Provenance) -> dict[str, tuple[object, object]]:
        """Field-by-field differences, for explaining a refusal in terms a human can act
        on. Reporting only "hashes differ" would reproduce failure 4 with extra steps."""
        a, b = _plain(self), _plain(other)
        return {k: (a[k], b[k]) for k in sorted(a) if a[k] != b[k]}


def _plain(p: Provenance) -> dict[str, object]:
    d = asdict(p)
    d["corpora"] = {k: dict(v) if isinstance(v, dict) else v for k, v in d["corpora"].items()}
    return d


def serving_identity(
    base_url: str, *, opener: Callable[[str], object] | None = None
) -> tuple[str, str]:
    """`(model_path, sha256)` of the model ACTUALLY answering at `base_url`.

    The hash is computed only when the path is readable from this host; a remote server
    yields `""` rather than a fabricated value. Never returns a guess — if `/props` is
    unreachable this raises, because "measure anyway and hope the right server is up" is
    precisely failure 1.
    """
    url = base_url.rstrip("/") + "/props"
    try:
        raw = (opener or urllib.request.urlopen)(url)
        body = raw.read() if hasattr(raw, "read") else raw
        props = json.loads(body)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"cannot read {url}: {exc}. Refusing to record a measurement without knowing "
            "which model answered."
        ) from exc

    path = props.get("model_path") or props.get("default_generation_settings", {}).get("model")
    if not path:
        raise RuntimeError(f"{url} returned no model_path; cannot identify the served model")

    p = Path(path)
    if not p.is_file():
        return str(path), ""
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return str(path), h.hexdigest()


def local_identity(gguf_path: str) -> tuple[str, str]:
    """`(path, sha256)` for an in-process GGUF, where there is no server to interrogate.

    The HTTP path asks `/props` because the served file cannot otherwise be known. Here
    the caller names the file directly, so identity is just its content hash — but it is
    still HASHED rather than trusted, so a scorecard cannot claim a checkpoint it did not
    actually load.
    """
    p = Path(gguf_path)
    if not p.is_file():
        raise RuntimeError(f"no such model file: {gguf_path}")
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return str(p), h.hexdigest()


def code_revision() -> str:
    """Short git revision, with a `-dirty` suffix when the tree has uncommitted changes.

    A dirty tree is recorded, not refused: most measurements here are taken mid-change.
    But it must be VISIBLE, so a number that cannot be reproduced from any commit says so
    on its face.
    """
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return f"{rev}-dirty" if dirty else rev
    except (subprocess.SubprocessError, OSError):
        return ""


def capture(
    base_url: str,
    *,
    protocol: str,
    generation: dict[str, object],
    corpora: Iterable[tuple[str, Path]] = (),
    label: str = "",
    checkpoint: str = "",
    epoch: str = "",
    gguf_path: str = "",
    opener: Callable[[str], object] | None = None,
) -> Provenance:
    """Build a `Provenance` for a live server. Raises if the server cannot be identified.

    `epoch` is a free-text note (`"best (626)"`, `"last (939)"`) rather than an int: the
    useful thing to record is which SELECTION RULE produced the artifact, since best- and
    last-epoch exports of one run differ measurably and not in a consistent direction.
    """
    from arcsum.prompts import PROMPT_VERSION
    from arcsum.tokens import TOKENIZE_VERSION

    if protocol not in ("edit", "tool"):
        raise ValueError(f"protocol must be 'edit' or 'tool', got {protocol!r}")

    if gguf_path:
        path, digest = local_identity(gguf_path)
    else:
        path, digest = serving_identity(base_url, opener=opener)
    return Provenance(
        model_path=path,
        model_sha256=digest,
        protocol=protocol,
        prompt_version=PROMPT_VERSION,
        tokenize_version=TOKENIZE_VERSION,
        generation=dict(sorted(generation.items())),
        corpora={name: CorpusFingerprint.of(d) for name, d in corpora},
        label=label,
        checkpoint=checkpoint,
        epoch=epoch,
        code_revision=code_revision(),
    )
