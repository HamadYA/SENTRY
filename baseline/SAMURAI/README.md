# SAMURAI

This folder contains the SAMURAI tracking backend.

Use it through the shared runner:

```bash
python ../run_all_models.py --config ../config.local.yaml --models SAMURAI --dataset didi --model-family sam21 --model-size L
```

Supported wrapper family: `sam21`, with sizes `L`, `B`, `S`, and `T`.

Direct scripts are still available and read the shared config through `SAM2_CONFIG`:

```bash
SAM2_CONFIG=../config.local.yaml python run_lasot.py --tracker_name sam21-L --output_dir ../outputs/SAMURAI
```

The nested `sam2/` package is the SAMURAI model implementation. The root `utils/` package is only for shared runner helpers.
