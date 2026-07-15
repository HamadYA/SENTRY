# Dataset Layouts

SENTRY uses the same public dataset keys and directory conventions as the unified baseline wrapper. Configure roots and outputs in `configs/paths.yaml`, copied from `configs/paths.example.yaml`.

## Box Tracking

All box datasets write one `<sequence>.txt` file per sequence. Each row is `[x,y,w,h]`, including the initialization frame.

### LaSOT and LaSOT Extension

```text
<root>/testing_set.txt
<root>/<category>/<sequence>/groundtruth.txt
<root>/<category>/<sequence>/img/*.jpg
```

### GOT-10k

Point the configured root at the evaluation split containing:

```text
<root>/list.txt
<root>/<sequence>/groundtruth.txt
<root>/<sequence>/*.jpg
```

### TrackingNet

```text
<root>/list.txt
<root>/anno/<sequence>.txt
<root>/frames/<sequence>/*.jpg
```

### TNL2K

```text
<root>/list.txt
<root>/<sequence>/groundtruth.txt
<root>/<sequence>/imgs/*.{jpg,png}
```

### LaTOT

```text
<root>/list.txt
<root>/<sequence>/<sequence>.txt
<root>/<sequence>/img/*.jpg
```

### OTB

```text
<root>/list.txt
<root>/<sequence>/groundtruth_rect.txt
<root>/<sequence>/img/*.jpg
```

## DiDi

DiDi requires the VOT toolkit and the dataset's native metadata. The root must contain `list.txt`; each VOT sequence must expose `first_frame_segm.txt` through its metadata root.

SENTRY writes both representations used by the baseline wrapper:

```text
<output>/<sequence>.txt                 # boxes
<output>/<sequence>/<sequence>.txt      # VOT mask trajectory
```

## Commands

Run a complete dataset:

```bash
python tools/run_benchmark.py \
  --method sentry \
  --backend sam2 \
  --dataset tnl2k \
  --tracker-name sam21-T \
  --sentry-config configs/sentry/default.yaml
```

Run one indexed sequence:

```bash
python tools/run_benchmark.py \
  --method sentry \
  --backend sam2 \
  --dataset got_10k \
  --sequence GOT-10k_Test_000001 \
  --tracker-name sam21-T
```

Existing result files are skipped. Pass `--overwrite` to recompute them.
