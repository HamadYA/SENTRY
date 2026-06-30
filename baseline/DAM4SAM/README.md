# DAM4SAM

This folder contains the DAM4SAM tracking backend.

Use it through the shared runner:

```bash
python ../run_all_models.py --config ../config.local.yaml --models DAM4SAM --dataset didi --model-family sam21 --model-size L
```

Supported wrapper families: `sam21` and `sam2`, with sizes `L`, `B`, `S`, and `T`. The wrapper uses the generic tracker names, while this backend maps them to DAM4SAM `pp` configs internally.

Direct scripts are still available and read the shared config through `SAM2_CONFIG`:

```bash
SAM2_CONFIG=../config.local.yaml python run_lasot.py --tracker_name sam21-L --output_dir ../outputs/DAM4SAM
```

DAM4SAM's distraction-aware memory uses VOT region helpers. Install the VOT toolkit if that code path is used.
