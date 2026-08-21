"""arcsum — agentic zh-TW meeting summarizer with ARC+POINTS external memory.

`SPEC.md` is the normative contract. Where this code and the spec disagree, the spec wins.

No re-exports live here on purpose: every module is imported by its full path so the
import graph stays readable and the zero-core-dependency property is easy to verify.
"""

__version__ = "0.1.0"
