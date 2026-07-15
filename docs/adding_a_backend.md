# Adding a Backend

A backend converts a host tracker into the protocol in `sentry_tracking.backends.protocol`.

The required forward methods return `BackendFrame`:

```python
class MyBackend:
    def initialize(self, frame, bbox, mask=None) -> BackendFrame:
        ...

    def track(self, frame) -> BackendFrame:
        ...
```

Full SENTRY operation additionally requires:

```python
    def generate_amg_hypotheses(self, frame) -> list[Hypothesis]:
        ...

    def reverse_track_window(self, frames, bbox=None, mask=None):
        ...

    def replace_current_memory(self, mask):
        ...
```

Reverse tracking must create isolated inference state. Memory replacement must update the current
non-conditioning spatial memory without converting the frame into a prompt or conditioning frame.

The release includes the following built-in names:

| Backend | Host |
| :--- | :--- |
| `sam2` | Vanilla SAM2 |
| `samurai` | SAMURAI |
| `dam4sam` | DAM4SAM |
| `samite` | SAMITE |
| `him2sam` | HiM2SAM |

The four forked hosts are implemented through the shared `ForkedSAM2Runtime`. New SAM2-family integrations
should prefer the same composition pattern: call the native host forward method unchanged, capture only
side outputs, and use a separate host predictor for reverse verification.

External backends can be selected without editing SENTRY:

```bash
python tools/run_benchmark.py \
  --backend my_package.adapter:MyBackend \
  --method sentry \
  --dataset-root /datasets/LaSOT \
  --output-dir outputs/my_backend
```

Bundled baselines use packages with conflicting `sam2` module names. Run each family in a separate process;
the backend loader intentionally rejects a process that already imported a different SAM2 fork.
