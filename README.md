# MobileSAM_lite 

An unofficial Python package for MobileSAM and MobileSAMv2 runtime that adds support for lighter encoder models not available in the original implementation.

This package vendors the runtime code needed for inference:

- `mobilesamv2`
- `tinyvit`
- `efficientvit`
- `ultralytics` under `mobilesam_lite/_vendor/ultralytics`

It intentionally does not bundle model checkpoints. Download weights separately and pass the checkpoint path at runtime.

The optional `mobilesamv2.promt_mobilesamv2` module now resolves its Ultralytics dependency from the vendored package in `mobilesam_lite._vendor.ultralytics`.

## Install locally

```bash
pip install -e .
```

## Install with pypi
```bash
pip install mobilesam-lite
```


## Example

```python
import torch

from mobilesam_lite.mobile_sam import SamPredictor, sam_model_registry

model = sam_model_registry["vit_t"]("./weight/mobile_sam.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

predictor = SamPredictor(model)
```

## Verify an installed wheel

After installing the wheel into a clean environment, run:

```bash
python example_inference_mobilesam.py --checkpoint /path/to/mobile_sam.pt
```

You can also provide a real image:

```bash
python example_inference_mobilesam.py --checkpoint /path/to/mobile_sam.pt --image /path/to/image.jpg
```

The script prints the installed distribution version, the imported package path, and the output tensor shapes from one prediction call.

For the MobileSAMv2 decoder path, use:

```bash
python example_inference_mobilesamv2.py \
  --checkpoint /path/to/mobile_sam.pt \
  --prompt-decoder-checkpoint /path/to/Prompt_guided_Mask_Decoder.pt \
  --object-aware-model-checkpoint /path/to/ObjectAwareModel.pt
```

This script verifies the packaged MobileSAMv2 pipeline with `ObjectAwareModel` box proposals plus the prompt-guided decoder, and writes `boxes.png`, `mask_union.png`, `mask_union_overlay.png`, and `mask_overlay.png` into the chosen output directory.

## Reference: Official MobileSAM repository 
https://github.com/chaoningzhang/mobilesam
