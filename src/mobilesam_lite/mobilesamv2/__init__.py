# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import sys
from pathlib import Path


def _ensure_vendored_ultralytics() -> None:
    if "ultralytics" in sys.modules:
        return

    vendor_root = Path(__file__).resolve().parents[1] / "_vendor" / "ultralytics"
    spec = importlib.util.spec_from_file_location(
        "ultralytics",
        vendor_root / "__init__.py",
        submodule_search_locations=[str(vendor_root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load vendored ultralytics package from {vendor_root}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["ultralytics"] = module
    spec.loader.exec_module(module)


_ensure_vendored_ultralytics()

from .build_sam import (
    build_sam_vit_h,
    build_sam_vit_l,
    build_sam_vit_b,
    sam_model_registry,
)
from .predictor import SamPredictor
from .automatic_mask_generator import SamAutomaticMaskGenerator
