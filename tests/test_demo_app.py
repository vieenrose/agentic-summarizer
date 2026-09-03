"""Pins the Gradio demo against breakage that is INVISIBLE on this workstation.

**The deploy target is Python 3.10; this venv is 3.12.** On 2026-09-03 a UI redesign
shipped `f"<div class='ax-panel{" ax-active" if active else ""}'>"` — nested same-type
quotes inside an f-string expression, which PEP 701 legalised in 3.12 and which is a hard
`SyntaxError` in 3.10. It imported cleanly here, passed lint, passed every test, and took
the Space down on boot with `SyntaxError: f-string: expecting '}'`.

Nothing in the suite could have caught it, because every check ran on the interpreter that
accepts it.

**The obvious guard does not work.** `ast.parse(..., feature_version=(3, 10))` accepts this
syntax happily — 3.12 replaced the f-string tokenizer wholesale, and `feature_version` only
gates a small set of grammar features. That was verified before relying on it, which is the
only reason this file contains a structural detector instead of a one-liner that would have
given false confidence.

The renderer tests below are the second half — they guard the theme-awareness the redesign
was FOR. The panels previously hardcoded GitHub-light hex values, which render as
near-invisible light-on-light for any visitor whose browser prefers a dark scheme, and
that is not something a developer on a light theme ever sees either.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys
import types

import pytest

DEMO = pathlib.Path(__file__).resolve().parent.parent / "demo" / "space_gradio"

def pep701_offences(src: str, filename: str = "<src>") -> list[tuple[int, str]]:
    """`(lineno, expression)` for every f-string expression reusing the f-string's own
    quote character — legal from 3.12, a hard SyntaxError before it.

    Detected structurally rather than by parsing under an older grammar, because
    `feature_version` does not reject it (checked).
    """
    tree = ast.parse(src, filename=filename)
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        seg = ast.get_source_segment(src, node)
        if not seg:
            continue
        quote = next((c for c in seg if c in "\"'"), None)
        if not quote:
            continue
        for child in node.values:
            if not isinstance(child, ast.FormattedValue):
                continue
            expr = ast.get_source_segment(src, child.value)
            if expr and quote in expr:
                out.append((node.lineno, expr))
    return out


def test_detector_catches_the_syntax_that_took_the_space_down():
    """A guard nobody has seen fail is a guard nobody knows works."""
    bad = 'x = f"<div class=\'p{" a" if c else ""}\'>"'
    good = 'cls = " a" if c else ""\nx = f"<div class=\'p{cls}\'>"'
    legal = "y = f\"{'a' if b else 'c'}\""  # different quote types: fine on 3.10
    assert pep701_offences(bad), "detector missed the real break"
    assert not pep701_offences(good)
    assert not pep701_offences(legal)


@pytest.mark.parametrize("path", sorted(DEMO.glob("*.py")), ids=lambda p: p.name)
def test_demo_uses_no_python_312_only_fstrings(path: pathlib.Path):
    """Every demo module must be parseable by the DEPLOY target, not just by this venv."""
    offences = pep701_offences(path.read_text(encoding="utf-8"), str(path))
    assert not offences, (
        f"{path.name} uses PEP 701 f-string nesting at line(s) "
        f"{[ln for ln, _ in offences]}: {[e for _, e in offences]}. "
        "HF Spaces runs Python 3.10, where this is a SyntaxError at import."
    )


@pytest.fixture(scope="module")
def app():
    """Import the demo with `spaces` stubbed — the ZeroGPU shim is not installed here."""
    gr = pytest.importorskip("gradio")
    assert gr  # imported for the skip condition only
    sys.modules.setdefault(
        "spaces", types.SimpleNamespace(GPU=lambda *a, **k: (lambda f: f)))
    sys.path.insert(0, str(DEMO))
    import app as app_module

    return app_module


def _memory(app):
    from arcsum.memory import Memory
    from arcsum.tokens import heuristic_token_len

    m = Memory(arc="市議會審議年度預算案並作成決議。", token_len=heuristic_token_len)
    m.add_point("市議會通過總預算 12 億元", 0)
    return m


HEX = re.compile(r"#[0-9a-fA-F]{6}")


def test_rendered_panels_carry_no_literal_colour(app):
    """Colour must come from Gradio theme variables. A literal hex here is a panel that
    renders wrong under the opposite colour scheme — and the developer never sees it."""
    html = (app._panel("T", "s", "body", step_no="1")
            + app.render_memory_html(_memory(app), 6, 2)
            + app.render_prose_html("摘要。", False))
    assert not HEX.findall(html), f"literal colour in rendered output: {HEX.findall(html)}"


def test_progress_badge_keeps_its_only_intentional_colour(app):
    """The CPU/GPU badge is the one deliberate exception: a semantic status colour that
    must stay legible on either ground, so it is defined in CSS, not inline."""
    html = app._progress(40.0, "reading", mode="CPU")
    assert "ax-mode-cpu" in html and not HEX.findall(html)


def test_every_live_panel_is_numbered(app):
    """The four surfaces update at different rates; the numbers are what say which to read
    first. A panel silently losing its number is a real regression in comprehensibility."""
    from arcsum.memory import Memory
    from arcsum.tokens import heuristic_token_len

    surfaces = [
        app.render_transcript_html([], -1, -1),
        app.render_ops_html("", "", False),
        app.render_memory_html(Memory(token_len=heuristic_token_len), 0, 0),
        app.render_prose_html("", False),
    ]
    for i, html in enumerate(surfaces, start=1):
        assert f"<span class='ax-step'>{i}</span>" in html, f"panel {i} lost its step number"


def test_css_is_attached_at_launch_not_to_blocks(app):
    """Gradio 6 moved `css` from the Blocks constructor to `launch()`. Passing it to Blocks
    only warns and DROPS the stylesheet, so the redesign looks applied locally and ships
    unstyled."""
    src = (DEMO / "app.py").read_text(encoding="utf-8")
    assert "launch(css=CSS)" in src
    assert "gr.Blocks(title=" in src and "css=CSS) as demo" not in src


def test_css_defines_both_schemes_through_theme_variables(app):
    for var in ("--border-color-primary", "--color-accent", "--background-fill-secondary"):
        assert var in app.CSS, f"{var} missing; panels would not follow the visitor's theme"
