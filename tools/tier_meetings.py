#!/usr/bin/env python3
"""Print `meeting_id lang` lines for one eval tier (used by eval/run_tier.sh)."""
import json
import sys
from pathlib import Path

rows = json.loads((Path(__file__).resolve().parent.parent / "data/transcripts/manifest.json").read_text())
tier = sys.argv[1] if len(sys.argv) > 1 else "micro"
for r in rows:
    if r["split"] == tier:
        print(r["meeting_id"], r["lang"])
