# Architecture

SENTRY is split into a baseline-neutral decision engine and model-specific backends.

```text
frame
  -> backend forward prediction
  -> decoder, AMG, and Kalman hypotheses
  -> relative filtering and Soft-NMS
  -> isolated reverse tracking
  -> trajectory association
  -> severe-failure gate
  -> output selection
  -> optional severe-rescue memory admission
```

`src/sentry_tracking/engine.py` owns orchestration. It only consumes canonical `BackendFrame` and
`Hypothesis` objects. Candidate filtering, reverse scoring, association, failure detection, and memory
policy live in separate modules and can be tested without a model checkpoint.

The vanilla SAM2 backend is composed of:

- `backends/sam2.py`, which translates runtime dictionaries into canonical objects.
- `backends/sam2_runtime.py`, which owns SAM2 inference state, AMG features, reverse states, and memory writes.
- two side-output additions to the bundled SAM2 fork. These expose decoder masks and scores but do not alter
  the native highest-IoU winner or its normal memory encoding.

SAMURAI, DAM4SAM, SAMITE, and HiM2SAM use `backends/forked_sam2.py` and
`backends/forked_sam2_runtime.py`. The shared runtime composes around each upstream `SAMTracker`, so its
forward winner and native memory updates remain authoritative. An inference-only decoder hook reads masks,
IoUs, and object-presence logits without changing decoder output. The native selected index is recovered by
matching the host's selected low-resolution mask, which is important for hosts whose motion logic can reject
the maximum-IoU mask.

Fork-specific Hydra paths are resolved explicitly. HiM2SAM additionally selects its LaSOT, LaSOText, or VOT
profile from the benchmark dataset instead of relying on Hydra's duplicate-basename search order.

Forked backends create a second host predictor lazily for reverse verification. Reverse candidates reuse
immutable forward image features when the feature cache is enabled, but never share temporal memory or
mutable Kalman/RVCot state with the forward host.

## Behavioral Invariants

1. The baseline primary result is returned unless the severe gate and reverse verifier both pass.
2. Reverse tracking uses an isolated host predictor and never mutates forward state.
3. `memory_policy: baseline` never writes a SENTRY mask into SAM2 memory.
4. `memory_policy: severe_rescue` changes only the accepted rescue frame's non-conditioning spatial memory.
5. Kalman is excluded from Soft-NMS and requires a weak baseline reverse score before selection.
