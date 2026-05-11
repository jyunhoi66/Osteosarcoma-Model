"""Extract 3D deep-learning features from CT images.

The script reads image volumes, optional ROI masks, and labels from an Excel
spreadsheet, then exports one feature vector per patient to a JSON file. The
output JSON follows the format used by the downstream K-fold training code:

``feature``
    Dictionary mapping sequential sample indices to feature vectors.
``label``
    Dictionary mapping sequential sample indices to integer class labels.
``name``
    Dictionary mapping sequential sample indices to patient/sample IDs.

Example
-------
python extract_dl_features.py \
    --image-dir /path/to/images \
    --roi-dir /path/to/rois \
    --label-xlsx /path/to/labels.xlsx \
    --checkpoint /path/to/pretrained_ckpt.pt \
    --output-json ../data_spaced/dl_features.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, EnsureTyped, Lambdad, LoadImaged, NormalizeIntensityd, Resized
from monai.utils import set_determinism

from swinMM import SSLHead, load_pretrained_model


LOGGER = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Swin Transformer features from 3D medical images."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("/home/dell/PycharmProjects/item1/湘雅二医院/train_spaced"),
        help="Directory containing image volumes.",
    )
    parser.add_argument(
        "--roi-dir",
        type=Path,
        default=Path("/home/dell/PycharmProjects/item1/湘雅二医院/roi_spaced"),
        help="Directory containing ROI masks with filenames matching image volumes.",
    )
    parser.add_argument(
        "--label-xlsx",
        type=Path,
        default=Path("/home/dell/PycharmProjects/item1/湘雅二医院/output1.xlsx"),
        help="Excel file containing sample IDs and labels.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/dell/PycharmProjects/HCC-CPS/骨肉瘤/swim transformer权重/pretrained_ckpt .pt"),
        help="Pretrained Swin checkpoint.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=SCRIPT_DIR.parent / "data_spaced" / "dl_features.json",
        help="Output JSON file.",
    )
    parser.add_argument("--id-column", type=int, default=0)
    parser.add_argument("--label-column", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda-visible-devices", type=str, default=None)
    parser.add_argument(
        "--spatial-size",
        type=int,
        nargs=3,
        default=(96, 96, 96),
        metavar=("D", "H", "W"),
        help="Resize all volumes to this spatial size.",
    )
    parser.add_argument(
        "--input-mode",
        choices=("image", "roi", "masked_image", "image_roi"),
        default="image",
        help=(
            "Model input construction. Use image_roi only with checkpoints/models "
            "configured for two input channels."
        ),
    )
    parser.add_argument(
        "--checkpoint-prefix",
        type=str,
        default="module.",
        help="Prefix removed from checkpoint keys before loading.",
    )
    return parser.parse_args()


def configure_logging(output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_json.parent / "extract_dl_features.log", mode="w"),
        ],
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)


def stem_without_nii_suffix(path: Path) -> str:
    """Return a stable sample ID for .nii, .nii.gz, and ordinary filenames."""
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def build_image_map(image_dir: Path, roi_dir: Path) -> dict[str, tuple[Path, Path]]:
    image_paths = sorted(path for path in image_dir.iterdir() if path.is_file())
    image_map: dict[str, tuple[Path, Path]] = {}

    for image_path in image_paths:
        sample_id = stem_without_nii_suffix(image_path)
        roi_path = roi_dir / image_path.name
        if not roi_path.exists():
            LOGGER.warning("Skipping %s because ROI file is missing.", sample_id)
            continue
        image_map[sample_id] = (image_path, roi_path)
    return image_map


def load_labels(label_xlsx: Path, id_column: int, label_column: int) -> dict[str, int]:
    label_table = pd.read_excel(label_xlsx)
    labels: dict[str, int] = {}

    for _, row in label_table.iterrows():
        sample_id = row.iloc[id_column]
        label = row.iloc[label_column]
        if pd.isna(sample_id) or pd.isna(label):
            continue
        labels[str(sample_id).strip()] = int(label)
    return labels


def build_records(
    image_map: dict[str, tuple[Path, Path]],
    labels: dict[str, int],
) -> list[dict[str, Any]]:
    records = []
    for sample_id in sorted(labels):
        if sample_id not in image_map:
            LOGGER.warning("Skipping %s because image file is missing.", sample_id)
            continue

        image_path, roi_path = image_map[sample_id]
        records.append(
            {
                "img": str(image_path),
                "roi": str(roi_path),
                "label": labels[sample_id],
                "id": sample_id,
            }
        )
    return records


def build_transforms(spatial_size: tuple[int, int, int]) -> Compose:
    return Compose(
        [
            LoadImaged(keys=("img", "roi"), ensure_channel_first=True),
            Resized(
                keys=("img", "roi"),
                spatial_size=spatial_size,
                mode=("area", "nearest"),
            ),
            NormalizeIntensityd(keys="img", nonzero=True, channel_wise=False),
            EnsureTyped(keys=("img", "roi")),
        ]
    )


def build_input_tensor(batch: dict[str, torch.Tensor], input_mode: str) -> torch.Tensor:
    image = batch["img"]
    roi = batch["roi"]

    if input_mode == "image":
        return image
    if input_mode == "roi":
        return roi
    if input_mode == "masked_image":
        return image * (roi > 0).to(image.dtype)
    if input_mode == "image_roi":
        return torch.cat((image, roi), dim=1)
    raise ValueError(f"Unsupported input_mode: {input_mode}")


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    input_mode: str,
) -> dict[str, dict[str, Any]]:
    model.eval()
    output = {"name": {}, "feature": {}, "label": {}}

    for batch_index, batch in enumerate(loader):
        inputs = build_input_tensor(batch, input_mode).to(device, non_blocking=True)
        features = model(inputs).detach().cpu().numpy()
        labels = batch["label"].detach().cpu().numpy()
        sample_ids = batch["id"]

        for item_index, feature in enumerate(features):
            output_index = str(len(output["feature"]))
            output["name"][output_index] = str(sample_ids[item_index])
            output["feature"][output_index] = feature.astype(float).tolist()
            output["label"][output_index] = int(labels[item_index])

        LOGGER.info("Processed batch %d/%d", batch_index + 1, len(loader))

    return output


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    configure_logging(args.output_json)
    set_seed(args.seed)

    image_map = build_image_map(args.image_dir, args.roi_dir)
    labels = load_labels(args.label_xlsx, args.id_column, args.label_column)
    records = build_records(image_map, labels)
    if not records:
        raise RuntimeError("No valid samples were found. Check image, ROI, and label paths.")

    LOGGER.info("Found %d valid samples.", len(records))
    dataset = Dataset(data=records, transform=build_transforms(tuple(args.spatial_size)))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SSLHead(n_class=2).to(device)
    load_pretrained_model(
        model=model,
        pretrained_path=str(args.checkpoint),
        prefix=args.checkpoint_prefix,
    )

    LOGGER.info("Starting feature extraction on %s.", device)
    output = extract_features(model, loader, device, args.input_mode)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    LOGGER.info("Saved features to %s.", args.output_json)


if __name__ == "__main__":
    main()
