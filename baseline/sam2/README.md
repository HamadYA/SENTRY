# sam2

This folder contains the baseline SAM2/SAM2.1 tracking backend.

Use it through the shared runner:

```bash
python ../run_all_models.py --config ../config.local.yaml --models sam2 --dataset didi --model-family sam21 --model-size L
```

Supported wrapper families: `sam21` and `sam2`, with sizes `L`, `B`, `S`, and `T`.

The nested `sam2/` package contains this backend's model code and architecture utilities. Keep those files inside this folder; shared runner helpers live in the repository-level `utils/` package.

Direct scripts are still available:

```bash
SAM2_CONFIG=../config.local.yaml python run_lasot.py --tracker_name sam21-L --output_dir ../outputs/sam2
```
