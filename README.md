# MobileSAM_lite 
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/mobilesam-lite?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/mobilesam-lite)

An unofficial Python package for MobileSAM and MobileSAMv2 runtime that adds support for lighter encoder models not available in the original implementation.


## Install locally

```bash
pip install -e .
```

## Install with pypi
```bash
pip install mobilesam-lite
```

Download the model weight provided by original repo [checkpoints](https://drive.google.com/file/d/1dE-YAG-1mFCBmao2rHDp0n-PP4eH7SjE/view?usp=sharing). 

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

For a Jupyter notebook walkthrough that installs from `pip` and runs single-prompt mobilesam inference, see [examples/mobilesam_single_point_inference.ipynb](examples/mobilesam_single_point_inference.ipynb).
or open the notebook diectly in google colab to play wtih [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/bill2239/mobilesam_lite/blob/main/examples/mobilesam_single_point_inference.ipynb)

the segmentation result of the would be shown like ![MobileSAM seg-any output](asset/outmobilesam.png)

## Examples for inference

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
  --object-aware-model-checkpoint /path/to/ObjectAwareModel.pt \
  --image /path/to/image.jpg \
  --output-dir wheel_verify_mobilesamv2_output
```

This script runs the MobileSAMv2 seg-every pipeline with `ObjectAwareModel` box proposals plus the prompt-guided decoder.

Inputs:

- `--checkpoint`: image encoder checkpoint
- `--prompt-decoder-checkpoint`: `Prompt_guided_Mask_Decoder.pt`
- `--object-aware-model-checkpoint`: `ObjectAwareModel.pt`
- `--image`: optional input image path. If omitted, the script uses a synthetic test image.
- `--output-dir`: directory for generated visualizations
- `--encoder-type`: you can choose between supported image encoders tiny_vit, sam_vit_h, efficientvit_l0, efficientvit_l1, efficientvit_l2
- Optional tuning args: , `--imgsz`, `--iou`, `--conf`, `--retina`, `--decoder-batch-size`, `--min-box-area-ratio`, `--max-box-area-ratio`

Outputs:

- Console summary with device, input image shape, detected box count, filtered box count, mask tensor shape, and saved output path
- `boxes.png`: detected boxes after filtering
- `mask_union.png`: binary union of all predicted masks
- `mask_union_overlay.png`: union mask blended over the input image
- `mask_overlay.png`: per-mask color overlay for the seg-every result

Example for the MobileSAMv2 seg-every inference with segmentation mask overlay visulization:

Input image:

![MobileSAMv2 seg-every input](asset/input.png)

Output overlay:

![MobileSAMv2 seg-every output](asset/mask_overlay.png)

## Reference: Official MobileSAM repository 
https://github.com/chaoningzhang/mobilesam

If you find this repo useful to you please consider click the button below to donate and support my work!
[![Buy Me A Coffee](https://www.buymeacoffee.com/assets/img/custom_images/yellow_img.png)](https://buymeacoffee.com/bill2239)
