# Baselines for SENTRY

This directory contains the baseline trackers used for the experimental comparisons in:

**SENTRY: SAM2-Enhanced Neighbor-Aware and Temporally Reasoned Memory for Visual Tracking**

> **Important:** this folder is **not** the implementation of the proposed SENTRY method.  
> It contains third-party baseline backends and wrapper scripts used to run comparison methods under a shared evaluation interface. For the main SENTRY method, see the repository-level README and the main method code.

---

## What is in this folder?

```text
baseline/
├── sam2/              # SAM2 / SAM2.1 baseline backend
├── SAMURAI/           # SAMURAI baseline backend
├── DAM4SAM/           # DAM4SAM baseline backend
├── HiM2SAM/           # HiM2SAM baseline backend
├── SAMITE/            # SAMITE baseline backend
├── SAM2Long/          # SAM2Long baseline and VOS/VOT-style tools
├── unified/           # Shared runner used by run_all_models.py
├── utils/             # Shared dataset, mask, box, VOT, and visualization helpers
├── config.yaml        # Public template config; keep local paths out of this file
└── run_all_models.py  # Convenience wrapper for running one or more baselines
```

The goal of this directory is to make it easier to reproduce and compare the baseline methods used in the paper. Each backend keeps its own tracking logic, while `run_all_models.py` provides a common entry point for launching them on supported datasets.

---

## Supported baseline backends

| Backend folder | Method | Notes |
|---|---|---|
| `sam2/` | SAM2 / SAM2.1 baseline | Supports `sam21` and `sam2` model families with sizes `L`, `B`, `S`, and `T`. |
| `SAMURAI/` | SAMURAI | Integrated as a baseline backend. The wrapper supports the `sam21` family with sizes `L`, `B`, `S`, and `T`. |
| `DAM4SAM/` | DAM4SAM | Integrated as a baseline backend. Supports `sam21` and `sam2` families with sizes `L`, `B`, `S`, and `T`. |
| `HiM2SAM/` | HiM2SAM | Integrated as a baseline backend. The wrapper supports the `sam21` family with sizes `L`, `B`, `S`, and `T`. |
| `SAMITE/` | SAMITE | Integrated as a baseline backend. The wrapper supports the `sam21` family with sizes `L`, `B`, `S`, and `T`. |
| `SAM2Long/` | SAM2Long | Provides VOT-style tracking and semi-supervised VOS tools. Supports `sam21` and `sam2` families with sizes `L`, `B`, `S`, and `T`. |

---

## Supported dataset keys

The shared wrapper expects dataset names using the following keys:

```text
didi
got_10k
lasot
lasot_ext
trackingnet
tnl2k
latot
otb
```

Dataset roots are configured in `config.local.yaml`, as described below.

---

## Setup

Use the environment from the main SENTRY repository. This baseline directory does not define a separate environment.

From the repository root:

```bash
cd SENTRY
# Create and activate the environment following the main README.
# Then move into the baseline directory:
cd baseline
```

The baselines typically require Python, PyTorch, OpenCV, NumPy, SciPy, PyYAML, tqdm, pycocotools, and the official benchmark toolkits for the datasets you evaluate on.

---

## Checkpoints

Download or place SAM2 / SAM2.1 checkpoints in the repository-level `checkpoints/` directory, or set explicit checkpoint paths in your local config.

A typical layout is:

```text
SENTRY/
├── checkpoints/
│   ├── sam2.1_hiera_large.pt
│   ├── sam2.1_hiera_base_plus.pt
│   ├── sam2.1_hiera_small.pt
│   ├── sam2.1_hiera_tiny.pt
│   ├── sam2_hiera_large.pt
│   ├── sam2_hiera_base_plus.pt
│   ├── sam2_hiera_small.pt
│   └── sam2_hiera_tiny.pt
└── baseline/
    ├── config.yaml
    └── config.local.yaml
```

Because `config.local.yaml` is usually stored inside `baseline/`, checkpoint paths should normally be written relative to `baseline/`, for example:

```yaml
weights:
  sam21-L: ../checkpoints/sam2.1_hiera_large.pt
  sam21-B: ../checkpoints/sam2.1_hiera_base_plus.pt
  sam21-S: ../checkpoints/sam2.1_hiera_small.pt
  sam21-T: ../checkpoints/sam2.1_hiera_tiny.pt
  sam2-L: ../checkpoints/sam2_hiera_large.pt
  sam2-B: ../checkpoints/sam2_hiera_base_plus.pt
  sam2-S: ../checkpoints/sam2_hiera_small.pt
  sam2-T: ../checkpoints/sam2_hiera_tiny.pt
```

---

## Configuration

Do not put machine-specific paths in `config.yaml`. Instead, copy it and edit the local copy:

```bash
cp config.yaml config.local.yaml
```

Then fill in your dataset and checkpoint paths in `config.local.yaml`.

Minimal example:

```yaml
seed: 0
device: cuda:0

# Dataset roots.
# Keep these local paths out of version control.
lasot_dataset_path: /path/to/LaSOT
lasot_ext_dataset_path: /path/to/LaSOT_Extension
got_10k_dataset_path: /path/to/GOT-10k
trackingnet_dataset_path: /path/to/TrackingNet
tnl2k_dataset_path: /path/to/TNL2K
latot_dataset_path: /path/to/LaTOT
didi_dataset_path: /path/to/DiDi
otb_dataset_path: /path/to/OTB

# Optional nested dataset form.
datasets:
  lasot:
    path: /path/to/LaSOT
  tnl2k:
    path: /path/to/TNL2K
  didi:
    path: /path/to/DiDi

wrapper:
  default_dataset: lasot
  default_model_family: sam21
  default_model_size: L
  output_root: outputs/unified
  visualize: false
  fail_fast: false

  models:
    - name: sam2
      path: sam2
      type: standard
      enabled: true
    - name: SAMURAI
      path: SAMURAI
      type: standard
      enabled: true
    - name: HiM2SAM
      path: HiM2SAM
      type: standard
      enabled: true
    - name: DAM4SAM
      path: DAM4SAM
      type: standard
      enabled: true
    - name: SAMITE
      path: SAMITE
      type: standard
      enabled: true
    - name: SAM2Long
      path: SAM2Long
      type: sam2long
      enabled: true

weights:
  sam21-L: ../checkpoints/sam2.1_hiera_large.pt
  sam21-B: ../checkpoints/sam2.1_hiera_base_plus.pt
  sam21-S: ../checkpoints/sam2.1_hiera_small.pt
  sam21-T: ../checkpoints/sam2.1_hiera_tiny.pt
  sam2-L: ../checkpoints/sam2_hiera_large.pt
  sam2-B: ../checkpoints/sam2_hiera_base_plus.pt
  sam2-S: ../checkpoints/sam2_hiera_small.pt
  sam2-T: ../checkpoints/sam2_hiera_tiny.pt
```

---

## Quick start

All commands below assume you are inside:

```bash
SENTRY/baseline
```

List available baseline backends:

```bash
python run_all_models.py --config config.local.yaml --list-models
```

Run one baseline on one dataset:

```bash
python run_all_models.py \
  --config config.local.yaml \
  --models sam2 \
  --dataset lasot \
  --model-family sam21 \
  --model-size L
```

Run several baselines:

```bash
python run_all_models.py \
  --config config.local.yaml \
  --models sam2 SAMURAI DAM4SAM \
  --dataset tnl2k \
  --model-family sam21 \
  --model-size L
```

Run all enabled baselines:

```bash
python run_all_models.py \
  --config config.local.yaml \
  --models all \
  --dataset didi \
  --tracker-name sam21-L
```

Run a single sequence:

```bash
python run_all_models.py \
  --config config.local.yaml \
  --models DAM4SAM \
  --dataset lasot \
  --tracker-name sam21-L \
  --sequence <sequence-name>
```

Preview commands without running them:

```bash
python run_all_models.py \
  --config config.local.yaml \
  --models all \
  --dataset lasot \
  --model-family sam21 \
  --model-size L \
  --dry-run
```

Enable visualization mode:

```bash
python run_all_models.py \
  --config config.local.yaml \
  --models sam2 \
  --dataset didi \
  --tracker-name sam21-L \
  --sequence <sequence-name> \
  --visualize
```

---

## Choosing model family and size

You can select trackers in two equivalent ways.

### Option 1: model family and size

```bash
python run_all_models.py \
  --config config.local.yaml \
  --models sam2 \
  --dataset lasot \
  --model-family sam21 \
  --model-size L
```

This resolves to tracker name:

```text
sam21-L
```

Supported families:

```text
sam21
sam2
```

Supported sizes:

```text
L, B, S, T
```

### Option 2: explicit tracker name

```bash
python run_all_models.py \
  --config config.local.yaml \
  --models sam2 \
  --dataset lasot \
  --tracker-name sam21-L
```

Use `--tracker-name` when you want the exact tracker identifier.

---

## Outputs

By default, predictions are written under:

```text
outputs/unified/<backend>/<dataset>/<tracker-name>/
```

Example:

```text
outputs/unified/DAM4SAM/lasot/sam21-L/
```

The exact prediction format follows the corresponding benchmark or backend runner. Use the official benchmark evaluation toolkit to compute metrics from the generated predictions.

---

## Running backend scripts directly

The recommended entry point is `run_all_models.py`, but each backend also exposes direct scripts such as:

```text
run_didi.py
run_got10k.py
run_lasot.py
run_lasot_ext.py
run_latot.py
run_otb.py
run_tnl2k.py
run_trackingnet.py
```

Example direct run:

```bash
cd SAMURAI

SAM2_CONFIG=../config.local.yaml python run_lasot.py \
  --tracker_name sam21-L \
  --output_dir ../outputs/SAMURAI/lasot/sam21-L
```

Direct scripts are useful for debugging a specific backend. For reproducible paper-style comparisons, prefer the shared wrapper.

---

## SAM2Long direct usage

`SAM2Long` also provides VOT-style and VOS tools.

Example VOT-style command:

```bash
python SAM2Long/tools/vot_inference.py \
  --config_file config.local.yaml \
  --dataset_name lasot \
  --model_name sam2.1_hiera_l \
  --output_dir outputs/SAM2Long/lasot/sam21-L
```

Visualization example:

```bash
python SAM2Long/tools/vot_inference.py \
  --config_file config.local.yaml \
  --dataset_name didi \
  --model_name sam2.1_hiera_t \
  --sequence <sequence-name> \
  --visualize
```

For semi-supervised VOS commands, see:

```text
SAM2Long/tools/README.md
```

---

## Adding a new baseline backend

To add another baseline to this wrapper:

1. Create a new backend folder under `baseline/`.
2. Add dataset runner scripts using the same naming convention, for example `run_lasot.py` and `run_tnl2k.py`.
3. Add the backend to `wrapper.models` in `config.local.yaml`.
4. Test command construction with `--dry-run`.
5. Run a single sequence before launching full benchmark evaluation.

Example config entry:

```yaml
wrapper:
  models:
    - name: NewTracker
      path: NewTracker
      type: standard
      enabled: true
```

---

## Troubleshooting

### `No path configured for dataset ...`

Set the dataset root in `config.local.yaml`.

For example:

```yaml
lasot_dataset_path: /path/to/LaSOT
```

or:

```yaml
datasets:
  lasot:
    path: /path/to/LaSOT
```

### `Unknown model`

Check available model names:

```bash
python run_all_models.py --config config.local.yaml --list-models
```

Then use one of the listed names with `--models`.

### `missing directory`

The wrapper found a model entry in the config, but the corresponding backend folder is missing. Check the `path` field in `wrapper.models`.

### Checkpoint not found

Make sure the paths in the `weights:` section point to real checkpoint files. If your config is inside `baseline/`, paths such as `../checkpoints/...` usually refer to the repository-level checkpoint directory.

### Import errors in direct scripts

Run direct backend scripts from the backend folder and set:

```bash
export SAM2_CONFIG=../config.local.yaml
```

For wrapper-based runs, `run_all_models.py` sets the required paths automatically.

---

## Notes on third-party baselines

This directory contains baseline integrations and wrappers for comparison methods. The original methods, model code, checkpoints, and licenses belong to their respective authors. Please check and follow the upstream licenses and citation requirements for each baseline method.

When reporting results from this folder, cite both:

1. The SENTRY paper.
2. The original paper or repository for each baseline method used.

---

## Citation

If you use this repository or the baseline results in your research, please cite SENTRY:

```bibtex
@inproceedings{alansari2026sentry,
  title={SENTRY: SAM2-Enhanced Neighbor-Aware and Temporally Reasoned Memory for Visual Tracking},
  author={Alansari, Mohamad and Michael, Yonathan and AlMarzouqi, Hasan and Naseer, Muzammal and Werghi, Naoufel and Javed, Sajid},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

Please also cite the original baseline methods used in your experiments.

---

## Related files

- Main project README: `../README.md`
- Baseline config template: `config.yaml`
- Local config file: `config.local.yaml`
- Shared runner: `run_all_models.py`
- Shared backend dispatcher: `unified/child_runner.py`
- Shared utilities: `utils/`
