# SENTRY <img src="assets/sentry_logo.png" alt="SENTRY Logo" height="32">: SAM2-Enhanced Neighbor-Aware and Temporally Reasoned Memory for Visual Tracking

<!--
<p align="center">
  <a href="https://arxiv.org/abs/2606.24449"><img src="https://img.shields.io/badge/Paper-arXiv-B31B1B.svg"></a>
  <a href="https://hamadya.github.io/SENTRY/page/"><img src="https://img.shields.io/badge/Project-Page-blue.svg"></a>
  <a href="https://github.com/<ORG>/SENTRY"><img src="https://img.shields.io/badge/Code-GitHub-black.svg"></a>
</p>
-->

<p align="center">
  📄<a href="https://arxiv.org/abs/2606.24449">Paper</a>&nbsp;&nbsp;|&nbsp;&nbsp;
  🌐<a href="https://hamadya.github.io/SENTRY/page/">Project Page</a>&nbsp;&nbsp;
</p>

**Official repository for "SENTRY <img src="assets/sentry_logo.png" alt="SENTRY Logo" height="32">: SAM2-Enhanced Neighbor-Aware and Temporally Reasoned Memory for Visual Tracking" (ECCV 2026).**

**Authors:** Mohamad Alansari*, Yonathan Michael*, Hasan AlMarzouqi, Muzammal Naseer, Naoufel Werghi, and Sajid Javed

Khalifa University, Abu Dhabi, UAE; University of Western Australia, Australia. *Equal contribution.

---

## Table of Contents
- [News](#news)
- [Introduction](#introduction)
- [Method Overview](#method-overview)
- [Model Lineup](#model-lineup)
- [SAM2 Baselines](#sam2-baselines)
- [Getting Started](#getting-started)
- [Checkpoints](#checkpoints)
- [Quick Run](#quick-run)
- [Evaluation](#evaluation)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

---

## News
- 2026-06-18: SENTRY has been accepted to ECCV 2026.
- 2026-06-18: Project page is available at https://hamadya.github.io/SENTRY/.
- 2026-07-15: Code, inference scripts, and evaluation wrappers released.

### Release Plan and Checklist

We are releasing SENTRY code, configs, and evaluation scripts. Track progress here:

<details>
  <summary><b>View checklist</b></summary>

#### 1) Code and Inference
- [x] Release SENTRY code.
- [x] Add single-video tracking demo.
- [x] Add examples for SENTRY-S2, SENTRY-SR, and SENTRY-D4S.
- [ ] Add visualization scripts for masks, boxes, and tracklets.

#### 2) Third-Party Trackers
- [x] Add SAM2 environment and checkpoint preparation instructions.
- [x] Add SAMURAI integration instructions.
- [x] Add DAM4SAM integration instructions.
- [x] Add HiM2SAM integration instructions.
- [x] Add SAMITE integration instructions.

#### 3) Evaluation
- [x] Add benchmark preparation instructions.
- [x] Add evaluation scripts for LaSOT, LaSOText, TNL2K, GOT-10k, and TrackingNet.
- [x] Add direct DiDi evaluation and retain the VOT20, VOT22, and VOTS24 toolkit wrappers.
- [ ] Add scripts for VOS evaluation under the first-frame-mask protocol.

</details>

---

## Introduction

**SENTRY** is a training-free, plug-and-play memory-admission module for SAM2-based visual object tracking. It addresses a key failure mode in SAM2-style trackers: confidence-only mask selection can write incorrect masks into memory during occlusion, abrupt motion, or distractor interference, causing drift in later frames.

<p align="center">
  <img src="assets/teaser.png" width="85%">
</p>

SENTRY aggregates multiple candidate masks, backtracks them into short tracklets, and compares them against recent target and neighbor trajectories. The baseline prediction remains authoritative unless it is severely inconsistent and another identity passes strict temporal verification. The canonical configuration then replaces only that accepted rescue frame's non-conditioning spatial memory.

Key properties:

- **Training-free:** no retraining or finetuning is required.
- **Plug-and-play:** integrates with existing SAM2-based trackers.
- **Memory-safe:** preserves native memory by default and admits only independently verified severe rescues when enabled.
- **Neighbor-aware:** penalizes candidates that align with nearby distractors.
- **Real-time:** large-model SENTRY variants remain real-time on an NVIDIA A100.

---

## Method Overview

<p align="center">
  <img src="assets/method.png" width="90%">
</p>

SENTRY operates at the memory-write interface of a host tracker.

### 1. Candidate Mask Generation

For each frame, SENTRY builds a compact candidate pool from:

- SAM2 decoder hypotheses.
- Automatic Mask Generation (AMG) proposals.
- Soft-NMS filtered candidates for spatial diversity.
- A Kalman motion-prior fallback for severe occlusion or unreliable appearance.

### 2. Short-Horizon Temporal Reasoning

Each candidate is backtracked over a fixed temporal window to produce a candidate tracklet. SENTRY compares these tracklets with:

- the recent target trajectory, and
- neighbor tracklets representing distractors or non-selected candidates.

### 3. Cycle-Consistent Memory Admission

SENTRY selects the candidate whose backward-propagated trajectory is most consistent with the target trajectory while avoiding neighbor/distractor trajectories. The release default uses `memory_policy: severe_rescue`, which replaces only the accepted rescue frame's non-conditioning spatial memory. `configs/sentry/conservative.yaml` preserves native host memory.

---

## Model Lineup

SENTRY can be attached to multiple SAM2-based visual trackers.

| SENTRY variant | Host tracker | Backend | Description |
| :--- | :--- | :--- | :--- |
| **SENTRY-S2** | SAM2 | `sam2` | SENTRY applied to vanilla SAM2. |
| **SENTRY-SR** | SAMURAI | `samurai` | SENTRY applied to SAMURAI. |
| **SENTRY-D4S** | DAM4SAM | `dam4sam` | SENTRY applied to DAM4SAM. |
| **SENTRY-SA** | SAMITE | `samite` | SENTRY applied to SAMITE. |
| **SENTRY-HiM** | HiM2SAM | `him2sam` | SENTRY applied to HiM2SAM. |

All five SENTRY backends are bundled directly. Each host family is loaded lazily and must run in a separate
process because the upstream forks use the same top-level Python package name, `sam2`.

> **License note:** DAM4SAM and SAMITE did not publish an explicit software license in their upstream
> repositories when this release was prepared. Their inclusion does not create a license grant. See
> [`NOTICE`](NOTICE) before redistributing this repository.

The runner resolves each fork's native model configuration explicitly. HiM2SAM uses its LaSOT profile by
default, its LaSOText profile for `--dataset lasot_ext`, and its VOT profile for `--dataset didi`. The bundled
HiM2SAM VOT profile is available only with `--tracker-name sam21-L`. HiM2SAM also loads CoTracker3 through
`torch.hub` from a pinned upstream revision; its first run therefore needs network access or an existing
Torch Hub cache for that revision.

Supported SAM2 model scales:

| Scale | Name |
| :--- | :--- |
| T | Tiny |
| S | Small |
| B | Base |
| L | Large |

---

## SAM2 Baselines

This repository also includes a dedicated `baseline/` directory for running the
SAM2-based comparison methods used in the paper.

> **Important:** `baseline/` is not the main SENTRY method.  
> It is a unified inference wrapper for reproducing and evaluating the SAM2-based
> baselines under a common configuration and output format.

The baseline wrapper provides a shared entry point for running multiple SAM2-era
trackers, including:

| Baseline | Description |
| :--- | :--- |
| **SAM2** | Vanilla SAM2 tracker baseline. |
| **SAMURAI** | SAMURAI-style SAM2 tracking baseline. |
| **DAM4SAM** | DAM4SAM-style SAM2 tracking baseline. |
| **HiM2SAM** | HiM2SAM-style SAM2 tracking baseline. |
| **SAMITE** | SAMITE-style SAM2 tracking baseline. |
| **SAM2Long** | Long-video SAM2-based tracking/VOS baseline. |

The goal of this folder is to make baseline comparison reproducible. Each method
keeps its own tracking logic, while the wrapper provides a consistent interface
for dataset selection, checkpoint selection, output paths, and batch execution.

Typical layout:

```text
baseline/
├── sam2/              # SAM2 / SAM2.1 baseline backend
├── SAMURAI/           # SAMURAI baseline backend
├── DAM4SAM/           # DAM4SAM baseline backend
├── HiM2SAM/           # HiM2SAM baseline backend
├── SAMITE/            # SAMITE baseline backend
├── SAM2Long/          # SAM2Long baseline backend
├── unified/           # Shared baseline runner and dispatch logic
├── utils/             # Shared dataset, mask, box, VOT, and visualization helpers
├── config.yaml        # Public baseline config template
└── run_all_models.py  # Unified baseline inference entry point
```

The main method is an installable package under `src/sentry_tracking`. See
[`docs/architecture.md`](docs/architecture.md) and
[`docs/adding_a_backend.md`](docs/adding_a_backend.md).

---

## Getting Started

### Environment Setup

Clone the repository:

```bash
git clone https://github.com/HamadYA/SENTRY.git
cd SENTRY
```

Create the environment:

```bash
conda env create -f sentry.yml
conda activate sentry
python -m pip install -e ".[evaluation]"
```

SENTRY is designed to run from this source checkout using an editable install. A standalone wheel is not
distributed because the bundled tracker forks and their model configurations are resolved relative to the
repository. Install a CUDA-compatible PyTorch build for your platform before the editable install when the
default package-index build is not appropriate for your GPU.

Fallback manual installation:

```bash
conda create -n sentry python=3.10 -y
conda activate sentry
pip install --upgrade pip
pip install -r requirements.txt
python -m pip install -e ".[evaluation]"
```

A typical setup requires PyTorch, OpenCV, NumPy, SciPy, tqdm, matplotlib, pycocotools, and the official benchmark toolkits for evaluation.

---

## Checkpoints

Place sam2 checkpoints under `checkpoints/`.
```bash
cd checkpoints
bash download_ckpts.sh
```
Expected directory structure:

```bash
SENTRY/
├── checkpoints/
│   ├── sam2.1_hiera_tiny.pt
│   ├── sam2.1_hiera_small.pt
│   ├── sam2.1_hiera_base_plus.pt
│   ├── sam2.1_hiera_large.pt
├── configs/
├── tools/
├── scripts/
├── assets/
└── outputs/
```
---

## Quick Run

### Run SENTRY-S2 on a video

```bash
python tools/demo.py \
  --video assets/example_video.mp4 \
  --init-bbox 40 80 64 48 \
  --backend sam2 \
  --tracker-name sam21-T \
  --sentry-config configs/sentry/default.yaml \
  --output-dir outputs/sentry_s2_demo
```

Use the conservative AMG-off, baseline-memory policy:

```bash
python tools/demo.py \
  --video assets/example_video.mp4 \
  --init-bbox 40 80 64 48 \
  --tracker-name sam21-T \
  --sentry-config configs/sentry/conservative.yaml \
  --output-dir outputs/sentry_s2_conservative_demo
```

The included synthetic video has a moving red target initialized by the box above. The demo writes `boxes.txt` and an annotated `tracking.mp4`.

Use `--backend samurai`, `--backend dam4sam`, `--backend samite`, or `--backend him2sam` to run the
corresponding SENTRY host. The benchmark and demo runners resolve each fork's bundled model configuration;
`--model-config` remains available for an explicit host-specific override.

### Public SENTRY configurations

| Configuration | AMG | Rescue output | Rescue memory | Feature cache |
| :--- | :---: | :---: | :---: | :---: |
| `default.yaml` | Yes | Severe cases only | Severe cases only | Yes |
| `conservative.yaml` | No | Severe cases only | No | Yes |
| `shadow.yaml` | Yes | No | No | Yes |

`configs/sentry/default.yaml` is the canonical paper configuration. The Python API defaults match this file, so using SENTRY without an explicit policy YAML has the same behavior.

---

<a id="evaluation"></a>
## Evaluation

SENTRY is evaluated on bounding-box tracking, VOT-style tracking, and VOS benchmarks.

### Bounding-Box Tracking

Datasets:

- LaSOT
- LaSOText
- TNL2K
- GOT-10k
- TrackingNet

The structured release runner supports the same dataset keys as the unified baseline wrapper:

| Dataset key | Benchmark | Sequence index |
| :--- | :--- | :--- |
| `lasot` | LaSOT | `testing_set.txt` |
| `lasot_ext` | LaSOT Extension / LaSOText | `testing_set.txt` |
| `got_10k` | GOT-10k test | `list.txt` |
| `trackingnet` | TrackingNet test | `list.txt` |
| `tnl2k` | TNL2K test | `list.txt` |
| `latot` | LaTOT | `list.txt` |
| `otb` | OTB | `list.txt` |
| `didi` | DiDi mask tracking | `list.txt` |

Copy `configs/paths.example.yaml` to `configs/paths.yaml` once and set the
machine-local dataset, checkpoint, and output paths. The local file is ignored
by Git, and explicit command-line path arguments still override it.

```bash
python tools/run_benchmark.py \
  --method sentry \
  --backend sam2 \
  --dataset lasot \
  --tracker-name sam21-T \
  --sentry-config configs/sentry/default.yaml \
  --debug-log outputs/lasot/sentry_s2_t.jsonl
```

The same command runs every bundled paper variant by changing only the backend and output/debug locations:

```bash
# SENTRY-SR; use dam4sam, samite, or him2sam for the other host families.
python tools/run_benchmark.py \
  --method sentry \
  --backend samurai \
  --dataset lasot \
  --tracker-name sam21-T \
  --sentry-config configs/sentry/default.yaml \
  --debug-log outputs/lasot/sentry_sr_t.jsonl
```

Change `--dataset` to any key in the table; the corresponding root, output, and checkpoint are read from `configs/paths.yaml`. Use `--sequence NAME` to select one sequence from that dataset's official index. The runner reports progress, FPS, ETA, and average image-I/O, forward, candidate-generation, and reverse-verification times every 100 frames. Change the interval with `--progress-every N`.

Box benchmarks write one `<sequence>.txt` file per sequence in `[x,y,w,h]` format. DiDi additionally writes VOT mask trajectories under `<output>/<sequence>/<sequence>.txt`. Dataset layouts and examples are documented in [`docs/datasets.md`](docs/datasets.md).

SAM2 release configs enable `reverse_feature_cache_enabled` so reverse
candidates reuse the forward pass's immutable image-encoder features while
retaining independent candidate memory and propagation states. Set it to
`false` for an uncached equivalence or timing comparison.

Run `--method baseline` with the same arguments to validate baseline equivalence.

### VOT-Style Tracking

Datasets:

- VOT20
- VOT22
- VOTS24
- DiDi

DiDi is supported directly by `tools/run_benchmark.py`. The original interactive VOT20/VOT22/VOTS24 wrappers remain under `baseline/`; those toolkit-driven protocols are separate from the offline dataset runner.

### Main Results: Large Models

Bounding-box benchmarks:

| Method | LaSOT S | LaSOText S | TNL2K S | GOT-10k AO | TrackingNet S |
| :--- | ---: | ---: | ---: | ---: | ---: |
| SAM2-L | 68.5 | 56.8 | 56.7 | 80.8 | 85.3 |
| SENTRY-S2-L | 70.2 | 57.0 | 57.9 | 81.1 | 85.7 |
| SAMURAI-L | 74.2 | 61.0 | 50.6 | 81.7 | 85.3 |
| SENTRY-SR-L | 75.1 | 61.5 | 59.6 | 81.8 | 85.8 |
| DAM4SAM-L | 75.1 | 60.9 | 59.8 | 81.1 | 85.3 |
| SENTRY-D4S-L | **76.3** | **61.8** | **61.3** | **82.1** | **85.9** |

VOT-style benchmarks:

| Method | VOT20 Q | VOT22 Q | VOTS24 Q | DiDi Q |
| :--- | ---: | ---: | ---: | ---: |
| SAM2-L | 0.681 | 0.692 | 0.661 | 0.649 |
| SENTRY-S2-L | 0.690 | 0.701 | 0.669 | 0.660 |
| SAMURAI-L | 0.674 | 0.693 | 0.673 | 0.680 |
| SENTRY-SR-L | 0.686 | 0.715 | 0.681 | 0.693 |
| DAM4SAM-L | 0.723 | 0.750 | 0.711 | 0.694 |
| SENTRY-D4S-L | **0.732** | **0.759** | **0.715** | **0.705** |

Runtime on NVIDIA A100:

| Method | Base FPS | SENTRY FPS | FPS drop | Base VRAM | SENTRY VRAM |
| :--- | ---: | ---: | ---: | ---: | ---: |
| SAM2-L -> SENTRY-S2-L | 44.0 | 32.8 | 25.5% | 5.1 GB | 5.7 GB |
| SAMURAI-L -> SENTRY-SR-L | 40.6 | 30.9 | 24.0% | 5.2 GB | 5.7 GB |
| DAM4SAM-L -> SENTRY-D4S-L | 39.4 | 30.2 | 23.4% | 5.2 GB | 5.7 GB |


---

## 🙏 Acknowledgements

We would like to express our sincere gratitude to the authors and contributors of [SAM2](https://github.com/facebookresearch/sam2), [SAMURAI](https://github.com/yangchris11/samurai), [DAM4SAM](https://github.com/jovanavidenovic/DAM4SAM), [SAMITE](https://github.com/Sam1224/SAMITE), [HiM2SAM](https://github.com/LouisFinner/HiM2SAM), [SAM2Long](https://github.com/Mark12Ding/SAM2Long), [NeighborTrack](https://github.com/franktpmvu/NeighborTrack), and many other open-source efforts in visual object tracking and video object segmentation. Their impactful research, public implementations, checkpoints, and benchmarks have been invaluable to the development and evaluation of SENTRY.

We are also very grateful to the **ECCV 2026** reviewers, area chairs, organizers, and the broader **computer vision community** for their constructive feedback, service, and continued support of open scientific progress. We deeply appreciate being part of this community and hope SENTRY serves as a useful contribution for future research.

This work was supported in part by the Khalifa University Center for Autonomous Robotic Systems (KUCARS) under Award RC1-2018-KUCARS; in part by Silal Innovation Oasis through projects under grants 8475000023, 8475000024, 8475000025, and 8475000026; and in part by Khalifa University of Science and Technology through the Faculty Start-Ups under Project ID KU-INT-FSU-2005-8474000775.

---

## Citation

If you find SENTRY useful in your research, please consider citing our paper:

```bibtex
@inproceedings{alansari2026sentry,
  title={SENTRY: SAM2-Enhanced Neighbor-Aware and Temporally Reasoned Memory for Visual Tracking},
  author={Alansari, Mohamad and Michael, Yonathan and AlMarzouqi, Hasan and Naseer, Muzammal and Werghi, Naoufel and Javed, Sajid},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

GitHub can also generate citation metadata directly from [`CITATION.cff`](CITATION.cff).

---

## License

SENTRY-authored code and documentation are licensed under the [Apache License 2.0](LICENSE). Bundled and
runtime-loaded third-party components remain subject to their own terms; consult [`NOTICE`](NOTICE),
[`LICENSE_cctorch`](LICENSE_cctorch), and [`THIRD_PARTY_LICENSES`](THIRD_PARTY_LICENSES) before use or
redistribution. In particular, SAM2Long and CoTracker include non-commercial terms, while DAM4SAM and
SAMITE require upstream license clarification or permission before public redistribution.
