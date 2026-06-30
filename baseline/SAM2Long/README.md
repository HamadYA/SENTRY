# SAM2Long

This folder contains the SAM2Long tracking and semi-supervised VOS tools.

Use the VOT-style tracker through the shared runner:

```bash
python ../run_all_models.py --config ../config.local.yaml --models SAM2Long --dataset didi --model-family sam21 --model-size L
```

Supported wrapper families: `sam21` and `sam2`, with sizes `L`, `B`, `S`, and `T`.

Direct VOT-style usage:

```bash
python tools/vot_inference.py \
  --config_file ../config.local.yaml \
  --dataset_name lasot \
  --model_name sam2.1_hiera_l \
  --output_dir ../outputs/SAM2Long/lasot
```

The tools also support visualization:

```bash
python tools/vot_inference.py \
  --config_file ../config.local.yaml \
  --dataset_name didi \
  --model_name sam2.1_hiera_t \
  --sequence <sequence-name> \
  --visualize
```

For semi-supervised VOS commands, see `tools/README.md`.
