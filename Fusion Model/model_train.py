"""K-fold training script for multimodal binary classification.

The expected JSON file contains two top-level dictionaries:

``feature``
    Mapping from sample index to a concatenated feature vector in the order
    ``[image_768, radiomics_7, tumor_768]``.
``label``
    Mapping from sample index to an integer class label.

Example
-------
python K_fold_train_json_auc.py \
    --data-json data_spaced/dl_radiomics_tumor_features.json \
    --vit-checkpoint ../../vit \
    --output-dir outputs/five_fold
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from monai.utils import set_determinism
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from features_fusion import BinaryClassifier


LOGGER = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class FeatureLayout:
    """Feature slice definition for the concatenated JSON feature vector."""

    image_dim: int = 768
    radiomics_dim: int = 7
    tumor_dim: int = 768

    @property
    def total_dim(self) -> int:
        return self.image_dim + self.radiomics_dim + self.tumor_dim

    @property
    def radiomics_start(self) -> int:
        return self.image_dim

    @property
    def tumor_start(self) -> int:
        return self.image_dim + self.radiomics_dim


class JsonFeatureDataset(Dataset):
    """Dataset wrapping pre-extracted multimodal features and class labels."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        if len(features) != len(labels):
            raise ValueError("features and labels must have the same length.")
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[index],
            "label": self.labels[index],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the multimodal fusion classifier with stratified K-fold CV."
    )
    parser.add_argument(
        "--data-json",
        type=Path,
        default=Path("data_spaced/dl_radiomics_tumor_features.json"),
        help="Path to the JSON file containing pre-extracted features and labels.",
    )
    parser.add_argument(
        "--vit-checkpoint",
        type=str,
        default="../../vit",
        help="Hugging Face ViT checkpoint name or local checkpoint directory.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/kfold"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-interval", type=int, default=2)
    parser.add_argument(
        "--class-weights",
        type=float,
        nargs=2,
        default=(0.45, 0.55),
        metavar=("CLASS_0", "CLASS_1"),
        help="Cross-entropy weights for the negative and positive classes.",
    )
    return parser.parse_args()


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "train.log", mode="w"),
        ],
    )


def resolve_project_path(path_like: str | Path) -> Path | str:
    """Resolve local relative paths from the script directory when possible."""
    path = Path(path_like)
    if path.is_absolute():
        return path

    candidate = SCRIPT_DIR / path
    if candidate.exists():
        return candidate
    return path_like


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)


def load_json_data(json_path: Path, layout: FeatureLayout) -> tuple[np.ndarray, np.ndarray]:
    """Load features and labels from the paper JSON feature format."""
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    sample_ids = sorted(data["feature"], key=lambda item: int(item))
    features = np.asarray([data["feature"][sample_id] for sample_id in sample_ids])
    labels = np.asarray([data["label"][sample_id] for sample_id in sample_ids], dtype=np.int64)

    if features.ndim != 2 or features.shape[1] != layout.total_dim:
        raise ValueError(
            f"Expected feature shape [N, {layout.total_dim}], got {features.shape}."
        )
    return features, labels


def split_modalities(
    features: torch.Tensor,
    layout: FeatureLayout,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split concatenated features into image, radiomics, and tumor tensors."""
    image_features = features[:, : layout.image_dim]
    radiomics_features = features[:, layout.radiomics_start : layout.tumor_start]
    tumor_features = features[:, layout.tumor_start : layout.total_dim]
    return image_features, radiomics_features, tumor_features


def build_dataloader(
    features: np.ndarray,
    labels: np.ndarray,
    indices: Iterable[int],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    indices = np.asarray(list(indices), dtype=np.int64)
    dataset = JsonFeatureDataset(features[indices], labels[indices])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    layout: FeatureLayout,
) -> float:
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        features = batch["features"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        image_features, radiomics_features, tumor_features = split_modalities(
            features,
            layout,
        )

        optimizer.zero_grad(set_to_none=True)
        logits = model(image_features, radiomics_features, tumor_features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    layout: FeatureLayout,
) -> dict[str, float]:
    model.eval()
    probabilities = []
    targets = []

    for batch in dataloader:
        features = batch["features"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        image_features, radiomics_features, tumor_features = split_modalities(
            features,
            layout,
        )

        logits = model(image_features, radiomics_features, tumor_features)
        probabilities.append(torch.softmax(logits, dim=1).cpu())
        targets.append(labels.cpu())

    y_prob = torch.cat(probabilities, dim=0)
    y_true = torch.cat(targets, dim=0)
    y_pred = y_prob.argmax(dim=1)

    return {
        "auc": roc_auc_score(y_true.numpy(), y_prob[:, 1].numpy()),
        "acc": (y_pred == y_true).float().mean().item(),
    }


def train_fold(
    fold_index: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
    layout: FeatureLayout,
    device: torch.device,
    writer: SummaryWriter,
) -> dict[str, float]:
    LOGGER.info("Fold %d/%d", fold_index + 1, args.n_splits)

    train_loader = build_dataloader(
        features,
        labels,
        train_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_loader = build_dataloader(
        features,
        labels,
        val_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    model = BinaryClassifier(
        img_dim=layout.image_dim,
        rad_dim=layout.radiomics_dim,
        tumor_dim=layout.tumor_dim,
        dropout=args.dropout,
        pretrained_model_name_or_path=args.vit_checkpoint,
    ).to(device)

    class_weights = torch.tensor(args.class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_metrics = {"auc": -1.0, "acc": -1.0, "epoch": -1}
    fold_dir = args.output_dir / f"fold_{fold_index + 1}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            layout,
        )
        writer.add_scalar(f"fold_{fold_index + 1}/train_loss", train_loss, epoch)
        LOGGER.info(
            "Fold %d | Epoch %03d/%03d | train_loss=%.4f",
            fold_index + 1,
            epoch,
            args.epochs,
            train_loss,
        )

        if epoch % args.val_interval != 0:
            continue

        metrics = evaluate(model, val_loader, device, layout)
        writer.add_scalar(f"fold_{fold_index + 1}/val_auc", metrics["auc"], epoch)
        writer.add_scalar(f"fold_{fold_index + 1}/val_acc", metrics["acc"], epoch)
        LOGGER.info(
            "Fold %d | Epoch %03d | val_auc=%.4f | val_acc=%.4f",
            fold_index + 1,
            epoch,
            metrics["auc"],
            metrics["acc"],
        )

        if metrics["auc"] > best_metrics["auc"]:
            best_metrics = {
                "auc": metrics["auc"],
                "acc": metrics["acc"],
                "epoch": epoch,
            }
            torch.save(model.state_dict(), fold_dir / "best_model.pth")
            LOGGER.info(
                "Fold %d | saved best checkpoint: auc=%.4f, acc=%.4f",
                fold_index + 1,
                metrics["auc"],
                metrics["acc"],
            )

    return best_metrics


def main() -> None:
    args = parse_args()
    args.data_json = Path(resolve_project_path(args.data_json))
    args.vit_checkpoint = str(resolve_project_path(args.vit_checkpoint))

    configure_logging(args.output_dir)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layout = FeatureLayout()
    LOGGER.info("Using device: %s", device)

    features, labels = load_json_data(args.data_json, layout)
    splitter = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed,
    )

    writer = SummaryWriter(log_dir=args.output_dir / "tensorboard")
    fold_results = []
    for fold_index, (train_indices, val_indices) in enumerate(
        splitter.split(features, labels)
    ):
        metrics = train_fold(
            fold_index,
            train_indices,
            val_indices,
            features,
            labels,
            args,
            layout,
            device,
            writer,
        )
        fold_results.append(metrics)
        LOGGER.info(
            "Fold %d complete | best_auc=%.4f | best_acc=%.4f | best_epoch=%d",
            fold_index + 1,
            metrics["auc"],
            metrics["acc"],
            metrics["epoch"],
        )

    writer.close()

    mean_auc = float(np.mean([item["auc"] for item in fold_results]))
    std_auc = float(np.std([item["auc"] for item in fold_results]))
    mean_acc = float(np.mean([item["acc"] for item in fold_results]))
    std_acc = float(np.std([item["acc"] for item in fold_results]))
    LOGGER.info("Cross-validation AUC: %.4f +/- %.4f", mean_auc, std_auc)
    LOGGER.info("Cross-validation ACC: %.4f +/- %.4f", mean_acc, std_acc)


if __name__ == "__main__":
    main()
