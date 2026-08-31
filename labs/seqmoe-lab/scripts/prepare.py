#!/usr/bin/env python3
"""Build scientific caches (wiki symlink/copy, code harvest, math fallback)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seqmoe.data import prepare_scientific


def main() -> int:
    meta = prepare_scientific()
    print(json.dumps(meta, indent=2))
    if not meta["wiki_ok"]:
        print(
            f"wiki missing at {meta['wiki_path']}; --scientific will exit 2.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
