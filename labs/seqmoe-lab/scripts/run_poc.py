#!/usr/bin/env python3
"""Sequential MoE PoC: pretrain → sequential experts → router → eval arms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seqmoe.config import (
    cpu_dump_seed_config,
    cpu_dump_skill_config,
    resolve_device,
    scientific_config,
    tiny_config,
    tiny_seed_config,
    tiny_skill_config,
)
from seqmoe.data import build_scientific_data, build_synth_data, wiki_path
from seqmoe.train import run_and_eval


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--tiny", action="store_true", help="synth, tiny_config")
    g.add_argument(
        "--tiny-seeds",
        action="store_true",
        help="synth, same-mix random FFN seeds (CPU only)",
    )
    g.add_argument(
        "--tiny-skills",
        action="store_true",
        help="synth, one cloned expert per skill code/math/fill (CPU only)",
    )
    g.add_argument("--scientific", action="store_true", help="wiki+code+math")
    g.add_argument(
        "--dump-skills",
        action="store_true",
        help="CPU: cloned code/math/wiki experts on real dumps",
    )
    g.add_argument(
        "--dump-seeds",
        action="store_true",
        help="CPU: random FFN seeds on the same dump mix",
    )
    p.add_argument("--device", default=None, help="cpu | cuda | cuda:N (bare cuda → cuda:1)")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--work-dir", default=None)
    args = p.parse_args()

    if args.dump_skills or args.dump_seeds:
        wiki = wiki_path()
        if not wiki.is_file() or wiki.stat().st_size < 10_000:
            print(f"wiki missing at {wiki}; dump runs need WikiText-2.", file=sys.stderr)
            return 2
        if args.device and args.device != "cpu":
            print("dump-skills/seeds stay on CPU", file=sys.stderr)
        cfg = cpu_dump_skill_config() if args.dump_skills else cpu_dump_seed_config()
        data = build_scientific_data(seed=cfg.train.seed)
    elif args.scientific:
        wiki = wiki_path()
        if not wiki.is_file() or wiki.stat().st_size < 10_000:
            print(
                f"wiki missing at {wiki}\n"
                "Copy WikiText-2 train there, or set SEQMOE_WIKI to the raw file.\n"
                "The scientific path does not fall back to synth.",
                file=sys.stderr,
            )
            return 2
        device = resolve_device(args.device or "cuda:1")
        cfg = scientific_config(device=device)
        data = build_scientific_data(seed=cfg.train.seed)
    else:
        data = None  # set below
        if args.tiny_seeds:
            if args.device and args.device != "cpu":
                print("tiny-seeds is CPU-only (other session holds the GPUs)", file=sys.stderr)
            cfg = tiny_seed_config(device="cpu")
            data = build_synth_data(seed=cfg.train.seed)
        elif args.tiny_skills:
            if args.device and args.device != "cpu":
                print("tiny-skills is CPU-only (other session holds the GPUs)", file=sys.stderr)
            cfg = tiny_skill_config(device="cpu")
            data = build_synth_data(seed=cfg.train.seed)
        else:
            device = resolve_device(args.device or "cpu")
            cfg = tiny_config(device=device)
            data = build_synth_data(seed=cfg.train.seed)

    if args.results_dir:
        cfg.train.results_dir = args.results_dir
    if args.run_name:
        cfg.train.run_name = args.run_name
    work = Path(args.work_dir) if args.work_dir else Path(cfg.train.results_dir) / cfg.train.run_name
    work.mkdir(parents=True, exist_ok=True)

    print(
        f"run_poc grade={data.scientific_grade} device={cfg.train.device} "
        f"code_source={data.code_source} math_source={data.math_source}"
    )
    payload = run_and_eval(
        cfg, data, work_dir=work, results_dir=cfg.train.results_dir
    )
    print(json.dumps({k: payload["arms"][k] for k in payload["arms"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
