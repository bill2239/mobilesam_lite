import argparse
import importlib.metadata
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torchvision.io import read_image, write_png

from mobilesam_lite.mobile_sam import SamPredictor, sam_model_registry


def _load_image(image_path: Optional[str]) -> np.ndarray:
    if image_path is None:
        image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        image[256:768, 256:768, 1] = 255
        image[384:640, 384:640, 0] = 255
        return image

    image_tensor = read_image(str(image_path))
    if image_tensor.shape[0] == 1:
        image_tensor = image_tensor.repeat(3, 1, 1)
    if image_tensor.shape[0] > 3:
        image_tensor = image_tensor[:3]
    return image_tensor.permute(1, 2, 0).cpu().numpy()


def _save_visualizations(
    image: np.ndarray,
    mask: np.ndarray,
    box: np.ndarray,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    mask_rgb = np.zeros_like(image, dtype=np.uint8)
    mask_rgb[..., 1] = mask_uint8

    overlay = image.astype(np.float32).copy()
    overlay[mask > 0] = overlay[mask > 0] * 0.4 + np.array([0, 255, 0], dtype=np.float32) * 0.6
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    x0, y0, x1, y1 = box.astype(int)
    x0 = max(0, min(x0, image.shape[1] - 1))
    x1 = max(0, min(x1, image.shape[1] - 1))
    y0 = max(0, min(y0, image.shape[0] - 1))
    y1 = max(0, min(y1, image.shape[0] - 1))

    overlay[y0:y0 + 2, x0:x1 + 1] = np.array([255, 0, 0], dtype=np.uint8)
    overlay[y1 - 1:y1 + 1, x0:x1 + 1] = np.array([255, 0, 0], dtype=np.uint8)
    overlay[y0:y1 + 1, x0:x0 + 2] = np.array([255, 0, 0], dtype=np.uint8)
    overlay[y0:y1 + 1, x1 - 1:x1 + 1] = np.array([255, 0, 0], dtype=np.uint8)

    write_png(torch.from_numpy(image).permute(2, 0, 1).contiguous(), str(output_dir / "input.png"))
    write_png(torch.from_numpy(mask_uint8).unsqueeze(0).contiguous(), str(output_dir / "mask.png"))
    write_png(torch.from_numpy(mask_rgb).permute(2, 0, 1).contiguous(), str(output_dir / "mask_color.png"))
    write_png(torch.from_numpy(overlay).permute(2, 0, 1).contiguous(), str(output_dir / "overlay.png"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that the installed mobilesam_lite wheel can run a basic MobileSAM inference."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a MobileSAM checkpoint, e.g. ./weight/mobile_sam.pt",
    )
    parser.add_argument(
        "--image",
        help="Optional path to an input image. If omitted, a synthetic image is used.",
    )
    parser.add_argument(
        "--model-type",
        default="vit_t",
        choices=sorted(sam_model_registry.keys()),
        help="Model type exposed by mobilesam_lite.mobile_sam.sam_model_registry.",
    )
    parser.add_argument(
        "--output-dir",
        default="wheel_verify_output",
        help="Directory where input, mask, and overlay visualizations will be written.",
    )
    args = parser.parse_args()

    dist_name = "mobilesam_lite"
    try:
        version = importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise SystemExit(
            "The mobilesam_lite distribution is not installed in this environment."
        ) from exc

    import mobilesam_lite.mobile_sam as mobile_sam_pkg

    print(f"Installed distribution: {dist_name}=={version}")
    print(f"Imported package from: {Path(mobile_sam_pkg.__file__).resolve()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = _load_image(args.image)
    predictor = SamPredictor(sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device))
    predictor.model.eval()
    predictor.set_image(image)

    height, width = image.shape[:2]
    box = np.array([width * 0.2, height * 0.2, width * 0.8, height * 0.8], dtype=np.float32)
    masks, scores, logits = predictor.predict(box=box, multimask_output=True)
    best_idx = int(np.argmax(scores))
    output_dir = Path(args.output_dir)
    _save_visualizations(image, masks[best_idx], box, output_dir)

    print(f"Device: {device}")
    print(f"Input image shape: {image.shape}")
    print(f"Masks shape: {masks.shape}")
    print(f"Scores: {scores}")
    print(f"Logits shape: {logits.shape}")
    print(f"Saved visualization to: {output_dir.resolve()}")
    print("Wheel verification succeeded.")


if __name__ == "__main__":
    main()
