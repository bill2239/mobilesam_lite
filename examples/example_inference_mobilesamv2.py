import argparse
import importlib
import importlib.metadata
import sys
from pathlib import Path
from typing import Generator, List, Optional

import numpy as np
import torch
from torchvision.io import read_image, write_png


def _load_image(image_path: Optional[str]) -> np.ndarray:
    if image_path is None:
        image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        image[160:864, 160:864, 2] = 180
        image[256:768, 256:768, 1] = 255
        image[384:640, 384:640, 0] = 255
        return image

    image_tensor = read_image(str(image_path))
    if image_tensor.shape[0] == 1:
        image_tensor = image_tensor.repeat(3, 1, 1)
    if image_tensor.shape[0] > 3:
        image_tensor = image_tensor[:3]
    return image_tensor.permute(1, 2, 0).cpu().numpy()


def _draw_box_outline(image: np.ndarray, box: np.ndarray, color: np.ndarray) -> None:
    x0, y0, x1, y1 = box.astype(int)
    x0 = max(0, min(x0, image.shape[1] - 1))
    x1 = max(0, min(x1, image.shape[1] - 1))
    y0 = max(0, min(y0, image.shape[0] - 1))
    y1 = max(0, min(y1, image.shape[0] - 1))
    image[y0:y0 + 2, x0:x1 + 1] = color
    image[max(0, y1 - 1):y1 + 1, x0:x1 + 1] = color
    image[y0:y1 + 1, x0:x0 + 2] = color
    image[y0:y1 + 1, max(0, x1 - 1):x1 + 1] = color


def _save_single_mask_visualizations(
    image: np.ndarray,
    mask: np.ndarray,
    boxes: np.ndarray,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    mask_rgb = np.zeros_like(image, dtype=np.uint8)
    mask_rgb[..., 0] = mask_uint8

    overlay = image.astype(np.float32).copy()
    overlay[mask > 0] = overlay[mask > 0] * 0.45 + np.array([255, 80, 0], dtype=np.float32) * 0.55
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    for box in boxes:
        _draw_box_outline(overlay, box, np.array([0, 255, 255], dtype=np.uint8))

    write_png(torch.from_numpy(image).permute(2, 0, 1).contiguous(), str(output_dir / "input.png"))
    write_png(torch.from_numpy(mask_uint8).unsqueeze(0).contiguous(), str(output_dir / "mask.png"))
    write_png(torch.from_numpy(mask_rgb).permute(2, 0, 1).contiguous(), str(output_dir / "mask_color.png"))
    write_png(torch.from_numpy(overlay).permute(2, 0, 1).contiguous(), str(output_dir / "overlay.png"))


def _save_mobilesamv2_outputs(
    image: np.ndarray,
    boxes: np.ndarray,
    masks: torch.Tensor,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    boxes_image = image.copy()
    for box in boxes:
        _draw_box_outline(boxes_image, box, np.array([255, 0, 0], dtype=np.uint8))

    union_mask = masks.any(dim=0).cpu().numpy()
    union_uint8 = union_mask.astype(np.uint8) * 255

    union_overlay = image.astype(np.float32).copy()
    union_overlay[union_mask] = union_overlay[union_mask] * 0.35 + np.array([255, 0, 0], dtype=np.float32) * 0.65
    union_overlay = np.clip(union_overlay, 0, 255).astype(np.uint8)

    layered_overlay = image.astype(np.float32).copy()
    color_cycle = np.array(
        [
            [255, 80, 0],
            [0, 200, 255],
            [120, 255, 0],
            [255, 0, 180],
            [255, 220, 0],
        ],
        dtype=np.float32,
    )
    for idx, mask in enumerate(masks):
        mask_np = mask.bool().cpu().numpy()
        color = color_cycle[idx % len(color_cycle)]
        layered_overlay[mask_np] = layered_overlay[mask_np] * 0.7 + color * 0.3
    layered_overlay = np.clip(layered_overlay, 0, 255).astype(np.uint8)

    write_png(torch.from_numpy(boxes_image).permute(2, 0, 1).contiguous(), str(output_dir / "boxes.png"))
    write_png(torch.from_numpy(union_uint8).unsqueeze(0).contiguous(), str(output_dir / "mask_union.png"))
    write_png(torch.from_numpy(union_overlay).permute(2, 0, 1).contiguous(), str(output_dir / "mask_union_overlay.png"))
    write_png(torch.from_numpy(layered_overlay).permute(2, 0, 1).contiguous(), str(output_dir / "mask_overlay.png"))


def _prepare_imports() -> tuple[str, Path]:
    dist_name = "mobilesam_lite"
    version = importlib.metadata.version(dist_name)

    mobile_sam_pkg = importlib.import_module("mobilesam_lite.mobile_sam")
    package_root = Path(mobile_sam_pkg.__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

    return version, Path(mobile_sam_pkg.__file__).resolve()


def _filter_boxes(boxes: torch.Tensor, image_shape: tuple[int, ...], min_area_ratio: float, max_area_ratio: float) -> torch.Tensor:
    image_h, image_w = image_shape[:2]
    image_area = image_h * image_w
    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_ratios = box_areas / image_area
    keep = (area_ratios >= min_area_ratio) & (area_ratios <= max_area_ratio)
    return boxes[keep]


def _batch_iterator(batch_size: int, *args) -> Generator[List[torch.Tensor], None, None]:
    assert len(args) > 0 and all(len(a) == len(args[0]) for a in args), "Batched inputs must share length."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for batch_idx in range(n_batches):
        yield [arg[batch_idx * batch_size : (batch_idx + 1) * batch_size] for arg in args]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that the installed mobilesam_lite wheel can run a MobileSAMv2-style inference."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to the image encoder checkpoint.")
    parser.add_argument(
        "--prompt-decoder-checkpoint",
        required=True,
        help="Path to Prompt_guided_Mask_Decoder.pt.",
    )
    parser.add_argument(
        "--object-aware-model-checkpoint",
        required=True,
        help="Path to ObjectAwareModel.pt.",
    )
    parser.add_argument("--image", help="Optional path to an input image.")
    parser.add_argument(
        "--encoder-type",
        default="tiny_vit",
        choices=["tiny_vit", "sam_vit_h", "efficientvit_l0", "efficientvit_l1", "efficientvit_l2"],
        help="Image encoder to plug into the MobileSAMv2 decoder stack.",
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="ObjectAwareModel inference image size.")
    parser.add_argument("--iou", type=float, default=0.9, help="YOLO IoU threshold.")
    parser.add_argument("--conf", type=float, default=0.4, help="YOLO confidence threshold.")
    parser.add_argument("--retina", action="store_true", help="Enable retina masks in ObjectAwareModel.")
    parser.add_argument("--decoder-batch-size", type=int, default=64, help="Boxes per decoder batch.")
    parser.add_argument("--max-box-area-ratio", type=float, default=0.5, help="Drop overly large detections.")
    parser.add_argument("--min-box-area-ratio", type=float, default=0.0005, help="Drop tiny detections.")
    parser.add_argument(
        "--output-dir",
        default="wheel_verify_mobilesamv2_output",
        help="Directory where visualization files will be written.",
    )
    args = parser.parse_args()

    try:
        version, mobile_sam_import_path = _prepare_imports()
    except importlib.metadata.PackageNotFoundError as exc:
        raise SystemExit("The mobilesam_lite distribution is not installed in this environment.") from exc

    import mobilesamv2
    from mobilesamv2 import SamPredictor, sam_model_registry
    from mobilesamv2.promt_mobilesamv2 import ObjectAwareModel

    print("Installed distribution: mobilesam_lite==%s" % version)
    print("Imported bootstrap package from: %s" % mobile_sam_import_path)
    print("Imported mobilesamv2 from: %s" % Path(mobilesamv2.__file__).resolve())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = _load_image(args.image)

    prompt_guided_decoder = sam_model_registry["PromptGuidedDecoder"](args.prompt_decoder_checkpoint)
    model = sam_model_registry["vit_h"]()
    model.prompt_encoder = prompt_guided_decoder["PromtEncoder"]
    model.mask_decoder = prompt_guided_decoder["MaskDecoder"]
    model.image_encoder = sam_model_registry[args.encoder_type](args.checkpoint)
    model.to(device=device)
    model.eval()

    predictor = SamPredictor(model)
    predictor.set_image(image)
    object_aware_model = ObjectAwareModel(args.object_aware_model_checkpoint)
    obj_results = object_aware_model(
        image,
        device=device,
        retina_masks=args.retina,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
    )
    if obj_results is None or len(obj_results) == 0:
        raise SystemExit("ObjectAwareModel returned no results.")

    input_boxes = obj_results[0].boxes.xyxy
    if input_boxes.numel() == 0:
        raise SystemExit("ObjectAwareModel detected no boxes for this image.")

    filtered_boxes = _filter_boxes(input_boxes, image.shape, args.min_box_area_ratio, args.max_box_area_ratio)
    if filtered_boxes.numel() == 0:
        raise SystemExit("All detected boxes were filtered out by the area thresholds.")

    transformed_boxes = predictor.transform.apply_boxes_torch(filtered_boxes, predictor.original_size).to(device=device)
    mask_batches = []
    for (boxes_batch,) in _batch_iterator(args.decoder_batch_size, transformed_boxes):
        with torch.no_grad():
            batch_masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=boxes_batch,
                mask_input=None,
                multimask_output=False,
                return_logits=False,
            )
        mask_batches.append(batch_masks.squeeze(1).float())
    masks = torch.cat(mask_batches, dim=0)

    output_dir = Path(args.output_dir)
    _save_mobilesamv2_outputs(image, filtered_boxes.detach().cpu().numpy(), masks.bool(), output_dir)

    print("Device: %s" % device)
    print("Input image shape: %s" % (image.shape,))
    print("Detected boxes: %d" % input_boxes.shape[0])
    print("Boxes kept after filtering: %d" % filtered_boxes.shape[0])
    print("Masks shape: %s" % (tuple(masks.shape),))
    print("Saved visualization to: %s" % output_dir.resolve())
    print("MobileSAMv2 wheel verification succeeded.")


if __name__ == "__main__":
    main()
