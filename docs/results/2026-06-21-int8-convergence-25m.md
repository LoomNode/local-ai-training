# Int8 Convergence 25M (Partial 5k Run Analysis)

**Date:** 2026-06-23 (Analyzing 2026-06-21 run)
**Status:** Completed at 5k horizon (Decision Gate modified)

## Goal
Determine whether the numerical noise introduced by int8 forward and backward matmuls compounds into worse validation loss relative to a matched BF16 control.

*Note: The original specification called for a 12,000-step horizon. The `int8` arm was interrupted at step 5,000. Following user decision, we are evaluating the predeclared gate at the shared 5,000-step horizon rather than re-running the full 12k.*

## Execution Protocol
- **Hardware:** RTX 3090
- **Artifacts:** `runs/int8-convergence-25m/bf16/` and `runs/int8-convergence-25m/int8/` on `main` branch.
- **Interruption:** The `bf16` arm completed 12,000 steps. The `int8` arm stopped at 5,000 steps.

## Metrics And Analysis
Comparison evaluated at the shared 5,000-step horizon.

### Loss Diagnostics (Step 5,000)
- **BF16 Validation Loss:** 1.210059
- **Int8 Validation Loss:** 1.217786
- **Signed Gap (int8 - bf16):** +0.007727 nats
- **BF16 Train Loss:** 1.202190
- **Int8 Train Loss:** 1.209569

### Other Diagnostics (Step 5,000)
- **Throughput:**
  - BF16: 123,305 tok/s
  - Int8: 69,468 tok/s (0.56x bf16, small-width regime)
- **Cumulative Code Moves:**
  - BF16: 937,014,330
  - Int8: 971,303,860
- **Saturation:**
  - BF16: 30.89%
  - Int8: 33.40%

## Predeclared Decision Gate Result: Tracks
The final gap at the evaluated 5,000-step horizon is **+0.007727 nats**. 
This is well inside the predeclared "Tracks" threshold of `<= 0.03` nats. 

## Limitations
- Evaluated at 5,000 steps rather than the 12,000 steps originally specified. Degradation that might emerge only at substantially longer horizons cannot be ruled out.
- Toy-width gradient-noise behavior may not predict frontier width.
- Single seed.

## Scope Boundary
This completes the convergence check per the modified 5k-step boundary. We observe the `int8` noise tracks `bf16` closely at this scale, providing a green light for engineering paths targeting larger widths or fused integer execution paths, acknowledging that those paths may introduce further noise that would need its own evaluation.
