#!/usr/bin/env python3
"""Build scientific caches. --fetch tries GSM8K + CPython dump; never fakes full."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seqmoe.data import prepare_scientific
from seqmoe.dumps import fetch_dumps


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fetch",
        action="store_true",
        help="try GSM8K (math) + CPython stdlib (code) dumps; sidecar dump only if obtained",
    )
    args = p.parse_args()
    if args.fetch:
        status = fetch_dumps()
        print(json.dumps(status, indent=2))
        meta = prepare_scientific()
        print(json.dumps(meta, indent=2))
        if not meta["wiki_ok"]:
            return 2
        return 0 if meta["scientific_grade"] == "full" else 1
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
