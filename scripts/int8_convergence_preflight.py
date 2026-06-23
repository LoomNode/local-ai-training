#!/usr/bin/env python3
"""Run the hard-gate checks for the matched BF16/int8 convergence experiment."""

from local_ai_training.int8_convergence_preflight import main

if __name__ == "__main__":
    raise SystemExit(main())
