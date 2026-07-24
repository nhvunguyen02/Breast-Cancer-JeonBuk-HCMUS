# -*- coding: utf-8 -*-
"""
================================================================================
TN-MAMMO — PHÂN LOẠI MẬT ĐỘ MÔ VÚ 4 LỚP A/B/C/D (BI-RADS DENSITY)
Bản gộp 1 trang của thư mục `tn-mammo-bestmacro-hientai` (model E1 tốt nhất
hiện tại: Macro-F1 = 0.7022 trên tập test khóa TN-Mammo 132 ca).
================================================================================

BÀI TOÁN
--------
Cho MỘT ca chụp nhũ ảnh gồm đủ 4 view: L-CC, L-MLO, R-CC, R-MLO (vú trái/phải,
mỗi bên 2 góc chụp), hãy phân loại mật độ mô vú của bệnh nhân vào đúng một
trong 4 mức BI-RADS:

    A — gần như toàn mỡ            C — mô dày không đồng nhất
    B — mô sợi-tuyến rải rác       D — mô cực kỳ dày

Mật độ càng cao thì nguy cơ ung thư vú càng tăng và khối u càng dễ bị che
khuất trên nhũ ảnh, nên đây là bước bắt buộc trong tầm soát.

Điểm khó:
  1. Mất cân bằng lớp (A rất hiếm)      -> chỉ số chính là Macro-F1,
                                           loss dùng class-balanced focal.
  2. Nhãn có thứ tự A < B < C < D       -> thêm loss ordinal CORAL phụ trợ
                                           (chỉ dùng lúc train).
  3. Dữ liệu TN-Mammo ít (411 ca train) -> train trộn thêm VinDr (3975 ca),
                                           sampler giữ 60% mass cho domain TN.

SƠ ĐỒ KIẾN TRÚC HIỆN TẠI (E1)
-----------------------------
    L-CC ─┐
    L-MLO ┤   DenseNet121 (chia sẻ trọng số)      mean fusion
    R-CC  ┼──> encode từng view ──> [B,4,1024] ──> trái=(v1+v2)/2 ──┐
    R-MLO ┘        224x224                         phải=(v3+v4)/2   │
                                                   exam=(trái+phải)/2
                                                        [B,1024]
                                          ┌─────────────┴─────────────┐
                                          ▼                           ▼
                                 Flat head Linear(1024,4)    CORAL head (3 ngưỡng)
                                 => argmax => A/B/C/D        chỉ tính loss lúc train
                                 (DỰ ĐOÁN CUỐI CÙNG)         (KHÔNG dùng để decode)

    Loss = ClassBalancedFocal(flat) + 0.5 * coral_loss(ordinal)

KẾT QUẢ TEST KHÓA TN-MAMMO (132 ca — đã "đốt", không dùng lại để chọn model)
----------------------------------------------------------------------------
    Macro-F1 0.7022 | Acc 0.6818 | BalAcc 0.7454 | QWK 0.7643
    Within-one 1.0000 | Lỗi nghiêm trọng (lệch >= 2 bậc): 0
    Confusion: [[4,0,0,0],[2,15,9,0],[0,8,37,12],[0,0,11,34]]

GHI CHÚ KHI GỘP
---------------
- Thư mục gốc KHÔNG chứa module `tn_mammo/data/` (PhaseGDatasetAdapter,
  contracts, sampler) — phần đó nằm trên server train. File này tái dựng:
  contracts CORAL (mục 2), dataset đọc manifest CSV tối giản (mục 3) và
  sampler trộn domain (mục 4) đúng theo hành vi được mô tả.
- Chỉ giữ MEAN fusion (fusion mà checkpoint E1 dùng). Các biến thể thí nghiệm
  khác (MLP control, ipsilateral, bilateral) xem `src/tn_mammo/models/fusion.py`.
- Toàn bộ máy móc audit test khóa (SHA256, marker 1 lần, đối chiếu validation)
  trong `inference.py` gốc được rút gọn thành đánh giá thường.

Cách dùng:
    python tn_mammo_onepage.py train --config config.yaml --output-dir outputs/run1
    python tn_mammo_onepage.py eval  --checkpoint checkpoint/best_model.pt \
                                     --manifest <manifest.csv>
================================================================================
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import coral_loss
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import DenseNet121_Weights, densenet121

# ============================================================================
# 1. HẰNG SỐ
# ============================================================================
VIEW_ORDER = ("L_CC", "L_MLO", "R_CC", "R_MLO")
LABEL_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}
NUM_CLASSES = 4
FEATURE_DIM = 1024  # đầu ra DenseNet121 sau global average pooling

# Phân bố lớp của tập train TN (411 ca) — nguồn duy nhất để tính class weight.
TN_CLASS_COUNTS = [12, 81, 178, 140]


# ============================================================================
# 2. CONTRACTS CHO NHÃN ORDINAL (tái dựng từ tn_mammo.data.contracts)
# ============================================================================
def make_ordinal_targets(labels: torch.Tensor) -> torch.Tensor:
    """Nhãn lớp k -> vector 3 mức CORAL: mức j = 1 nếu label > j."""
    thresholds = torch.arange(NUM_CLASSES - 1, device=labels.device)
    return (labels.unsqueeze(1) > thresholds.unsqueeze(0)).float()


def make_binary_targets(labels: torch.Tensor) -> torch.Tensor:
    """Nhãn phụ A/B (0) với C/D (1) — chỉ dùng khi bật binary head (E2)."""
    return (labels >= 2).long()


def decode_coral_logits(
    logits: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    """Decode CORAL: đếm số ngưỡng vượt qua. KHÔNG dùng cho dự đoán cuối E1."""
    return (torch.sigmoid(logits) > threshold).sum(dim=1)


# ============================================================================
# 3. DATASET 4 VIEW ĐỌC MANIFEST CSV
#    (adapter gốc PhaseGDatasetAdapter nằm trên server; bản này tối giản,
#     manifest cần cột: case_id, label, L_CC, L_MLO, R_CC, R_MLO, [source])
# ============================================================================
class SquarePad:
    """Pad ảnh thành hình vuông mà không làm méo tỷ lệ giải phẫu."""

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        pad_left = (side - width) // 2
        pad_top = (side - height) // 2
        pad_right = side - width - pad_left
        pad_bottom = side - height - pad_top
        return ImageOps.expand(
            image,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=0,
        )


class FourViewManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int = 224,
        training: bool = False,
        validate_paths: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy manifest: {self.manifest_path}")

        self.dataframe = pd.read_csv(self.manifest_path, dtype=str)
        required_columns = {"case_id", "label", *VIEW_ORDER}
        missing_columns = sorted(required_columns - set(self.dataframe.columns))
        if missing_columns:
            raise ValueError(f"Manifest thiếu cột bắt buộc: {missing_columns}")
        if self.dataframe.empty:
            raise ValueError(f"Manifest rỗng: {self.manifest_path}")
        if self.dataframe["case_id"].isna().any():
            raise ValueError("Manifest có case_id bị thiếu.")
        duplicated = self.dataframe["case_id"].duplicated(keep=False)
        if duplicated.any():
            examples = self.dataframe.loc[duplicated, "case_id"].head(5).tolist()
            raise ValueError(f"Manifest có case_id trùng, ví dụ: {examples}")

        labels = self.dataframe["label"].str.strip().str.upper()
        invalid_labels = sorted(set(labels.dropna()) - set(LABEL_TO_INDEX))
        if invalid_labels or labels.isna().any():
            raise ValueError(f"Label không hợp lệ hoặc bị thiếu: {invalid_labels}")
        self.dataframe["label"] = labels

        manifest_dir = self.manifest_path.parent
        for view in VIEW_ORDER:
            if self.dataframe[view].isna().any():
                raise ValueError(f"Manifest có đường dẫn bị thiếu ở view {view}.")
            self.dataframe[view] = self.dataframe[view].map(
                lambda value: str(
                    (manifest_dir / str(value)).resolve()
                    if not Path(str(value)).expanduser().is_absolute()
                    else Path(str(value)).expanduser().resolve()
                )
            )

        if validate_paths:
            missing_paths = [
                path
                for view in VIEW_ORDER
                for path in self.dataframe[view].tolist()
                if not Path(path).is_file()
            ]
            if missing_paths:
                preview = missing_paths[:5]
                raise FileNotFoundError(
                    f"Thiếu {len(missing_paths)} file ảnh; ví dụ: {preview}"
                )

        # Không flip/rotate độc lập từng view vì sẽ phá consistency giải phẫu.
        photometric = (
            [transforms.ColorJitter(brightness=0.10, contrast=0.10)]
            if training
            else []
        )
        self.transform = transforms.Compose([
            SquarePad(),
            transforms.Resize((image_size, image_size), antialias=True),
            *photometric,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict:
        row = self.dataframe.iloc[index]
        image_tensors = []
        for view in VIEW_ORDER:
            image_path = Path(str(row[view]))
            try:
                with Image.open(image_path) as image:
                    image_tensors.append(self.transform(image.convert("RGB")))
            except Exception as exc:
                raise RuntimeError(
                    f"Không thể đọc case={row['case_id']} view={view}: {image_path}"
                ) from exc

        return {
            "views": torch.stack(image_tensors),
            "label": LABEL_TO_INDEX[str(row["label"])],
            "case_id": str(row["case_id"]),
            "source": str(row.get("source", "TN")).strip(),
        }


# ============================================================================
# 4. SAMPLER TRỘN DOMAIN TN / VinDr (tn_domain_ratio = 0.6)
# ============================================================================
def build_domain_sampler(
    domains: list[str],
    tn_ratio: float,
    num_samples: int,
    generator: torch.Generator | None = None,
) -> WeightedRandomSampler:
    """Trọng số sao cho tổng xác suất lấy mẫu domain TN đúng bằng tn_ratio."""
    if not 0.0 < tn_ratio < 1.0:
        raise ValueError(f"tn_ratio phải nằm trong (0, 1), nhận được {tn_ratio}")
    if num_samples <= 0:
        raise ValueError("num_samples phải > 0")

    domains_array = np.char.upper(np.char.strip(np.asarray(domains, dtype=str)))
    tn_mask = domains_array == "TN"
    tn_count = int(tn_mask.sum())
    other_count = len(domains) - tn_count
    if tn_count == 0 or other_count == 0:
        raise ValueError(
            f"Sampler cần cả TN và domain ngoài; tn_count={tn_count}, "
            f"other_count={other_count}"
        )

    weights = np.where(
        tn_mask,
        tn_ratio / tn_count,
        (1.0 - tn_ratio) / other_count,
    )

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )


# ============================================================================
# 5. MODEL: DenseNet121 chia sẻ + mean fusion + flat head + CORAL head
# ============================================================================
class FourViewDensityModel(nn.Module):
    """E1: mean fusion + flat A/B/C/D head + CORAL ordinal head phụ trợ.

    Mean fusion không thêm tham số nào ngoài backbone, nên state_dict
    tương thích chặt với checkpoint Phase-G/E0.
    """

    def __init__(
        self,
        use_ordinal_head: bool = True,
        imagenet_init: bool = False,
    ) -> None:
        super().__init__()

        self.backbone = densenet121(
            weights=DenseNet121_Weights.IMAGENET1K_V1
            if imagenet_init
            else None
        )
        self.backbone.classifier = nn.Linear(FEATURE_DIM, NUM_CLASSES)

        self.ordinal_head = (
            CoralLayer(FEATURE_DIM, NUM_CLASSES)
            if use_ordinal_head
            else None
        )

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone.features(images)
        features = F.relu(features)  # khớp forward chuẩn của torchvision
        features = F.adaptive_avg_pool2d(features, (1, 1))
        return torch.flatten(features, 1)  # [N, 1024]

    def forward(self, views: torch.Tensor) -> dict[str, torch.Tensor | None]:
        # views: [B, 4, 3, H, W] theo thứ tự L_CC, L_MLO, R_CC, R_MLO
        batch_size, num_views = views.shape[:2]

        view_features = self.encode_images(
            views.reshape(batch_size * num_views, *views.shape[2:])
        ).reshape(batch_size, num_views, FEATURE_DIM)

        # Mean fusion: trung bình 2 view mỗi bên, rồi trung bình 2 bên.
        left = view_features[:, 0:2].mean(dim=1)
        right = view_features[:, 2:4].mean(dim=1)
        exam_features = (left + right) / 2.0

        return {
            "flat_logits": self.backbone.classifier(exam_features),
            "ordinal_logits": (
                self.ordinal_head(exam_features)
                if self.ordinal_head is not None
                else None
            ),
            "exam_features": exam_features,
        }


# ============================================================================
# 6. LOSS: class-balanced focal (chính) + CORAL ordinal (phụ, lambda=0.5)
# ============================================================================
def class_balanced_weights(
    class_counts: list[int], beta: float = 0.99
) -> torch.Tensor:
    """Trọng số 'effective number of samples' (Cui et al. 2019)."""
    counts = torch.tensor(class_counts, dtype=torch.float64)
    weights = (1.0 - beta) / (1.0 - torch.pow(beta, counts))
    weights = weights / weights.sum() * len(class_counts)
    return weights.to(torch.float32)


class ClassBalancedFocalLoss(nn.Module):
    def __init__(
        self,
        class_counts: list[int],
        beta: float = 0.99,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer(
            "class_weights", class_balanced_weights(class_counts, beta)
        )

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, targets, weight=self.class_weights, reduction="none"
        )
        pt = torch.softmax(logits, dim=1).gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)
        return (torch.pow(1.0 - pt, self.gamma) * ce).mean()


class MultiTaskCriterion(nn.Module):
    def __init__(
        self,
        class_counts: list[int] = TN_CLASS_COUNTS,
        beta: float = 0.99,
        gamma: float = 2.0,
        lambda_ordinal: float = 0.5,
    ) -> None:
        super().__init__()
        self.lambda_ordinal = lambda_ordinal
        self.flat_loss = ClassBalancedFocalLoss(class_counts, beta, gamma)

    def forward(
        self,
        outputs: dict[str, torch.Tensor | None],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        total = self.flat_loss(outputs["flat_logits"], labels)

        if self.lambda_ordinal > 0 and outputs["ordinal_logits"] is not None:
            total = total + self.lambda_ordinal * coral_loss(
                outputs["ordinal_logits"],
                make_ordinal_targets(labels),
            )

        return total


# ============================================================================
# 7. METRICS: Macro-F1 (chính) + các chỉ số thứ tự (QWK, within-one, severe)
# ============================================================================
def compute_metrics(y_true, y_pred) -> dict:
    truth = np.asarray(y_true, dtype=np.int64)
    prediction = np.asarray(y_pred, dtype=np.int64)
    labels = list(range(NUM_CLASSES))

    precision, recall, class_f1, support = precision_recall_fscore_support(
        truth, prediction, labels=labels, zero_division=0
    )
    distance = np.abs(truth - prediction)

    return {
        "num_samples": int(len(truth)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(truth, prediction)
        ),
        "macro_f1": float(
            f1_score(truth, prediction, average="macro", zero_division=0)
        ),
        "qwk": float(
            cohen_kappa_score(
                truth, prediction, weights="quadratic", labels=labels
            )
        ),
        "within_one": float(np.mean(distance <= 1)),
        "severe_error_count": int(np.sum(distance >= 2)),
        "confusion_matrix": confusion_matrix(
            truth, prediction, labels=labels
        ).tolist(),
        "per_class": {
            INDEX_TO_LABEL[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(class_f1[i]),
                "support": int(support[i]),
            }
            for i in labels
        },
    }


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================================
# 8. VÒNG LẶP TRAIN / VALIDATE (chọn checkpoint theo Macro-F1 validation)
# ============================================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp):
    model.train()
    total_loss, total_samples = 0.0, 0

    for batch in loader:
        views = batch["views"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp
        ):
            loss = criterion(model(views), labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach()) * labels.shape[0]
        total_samples += labels.shape[0]

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    y_true, y_pred = [], []

    for batch in loader:
        views = batch["views"].to(device, non_blocking=True)
        labels = batch["label"].long()

        with torch.amp.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp
        ):
            outputs = model(views)

        # Dự đoán cuối = argmax flat head (CORAL không tham gia decode).
        predictions = outputs["flat_logits"].float().argmax(dim=1).cpu()

        y_true.extend(labels.tolist())
        y_pred.extend(predictions.tolist())

    return compute_metrics(y_true, y_pred)


def run_training(config: dict, output_dir: Path) -> None:
    seed = int(config["experiment"]["seed"])
    deterministic = bool(config["experiment"].get("deterministic", True))
    seed_everything(seed, deterministic=deterministic)
    generator = torch.Generator().manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(config["training"].get("amp", True)) and device.type == "cuda"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = int(config["data"]["image_size"])
    tn_train = FourViewManifestDataset(
        config["data"]["train"]["tn_manifest"], image_size, training=True
    )
    vindr_train = FourViewManifestDataset(
        config["data"]["train"]["vindr_manifest"], image_size, training=True
    )
    tn_valid = FourViewManifestDataset(
        config["data"]["validation"]["manifest"], image_size, training=False
    )

    combined = torch.utils.data.ConcatDataset([tn_train, vindr_train])
    sampler = build_domain_sampler(
        ["TN"] * len(tn_train) + ["VinDr"] * len(vindr_train),
        tn_ratio=float(config["data"]["tn_domain_ratio"]),
        num_samples=int(
            config["training"].get("sampler_num_samples", len(combined))
        ),
        generator=generator,
    )

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"].get("num_workers", 4))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": generator,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        combined, batch_size=batch_size, sampler=sampler, **loader_kwargs
    )
    valid_loader = DataLoader(
        tn_valid, batch_size=batch_size, shuffle=False, **loader_kwargs
    )

    model = FourViewDensityModel(
        use_ordinal_head=bool(config["model"].get("use_ordinal_head", True)),
        imagenet_init=bool(config["model"].get("imagenet_init", False)),
    )

    # E1 khởi tạo từ checkpoint E0 (Phase-G): thiếu key ordinal_head là hợp lệ.
    init_checkpoint = config["model"].get("initialization_checkpoint")
    if init_checkpoint:
        state = torch.load(
            init_checkpoint, map_location="cpu", weights_only=True
        )
        missing_keys, unexpected_keys = model.load_state_dict(
            state.get("model_state_dict", state), strict=False
        )
        allowed_missing = {key for key in missing_keys if key.startswith("ordinal_head.")}
        disallowed_missing = set(missing_keys) - allowed_missing
        if disallowed_missing or unexpected_keys:
            raise RuntimeError(
                "Checkpoint khởi tạo không tương thích: "
                f"missing={sorted(disallowed_missing)}, "
                f"unexpected={sorted(unexpected_keys)}"
            )

    model = model.to(device)
    criterion = MultiTaskCriterion(
        beta=float(config["loss"]["flat"]["beta"]),
        gamma=float(config["loss"]["flat"]["gamma"]),
        lambda_ordinal=float(config["loss"]["lambda_ordinal"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(config["training"]["scheduler"]["step_size"]),
        gamma=float(config["training"]["scheduler"]["gamma"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_macro_f1, patience_counter = -math.inf, 0
    patience = int(config["training"]["early_stopping_patience"])

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, amp
        )
        metrics = evaluate(model, valid_loader, device, amp)
        scheduler.step()

        improved = metrics["macro_f1"] > best_macro_f1
        if improved:
            best_macro_f1 = metrics["macro_f1"]
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "valid_metrics": metrics,
                    "config": config,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                },
                output_dir / "best_checkpoint.pt",
            )
        else:
            patience_counter += 1

        print(json.dumps({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "valid_macro_f1": round(metrics["macro_f1"], 4),
            "best_macro_f1": round(best_macro_f1, 4),
            "improved": improved,
        }))

        if patience_counter >= patience:
            print(f"[EARLY_STOP] epoch={epoch}")
            break


# ============================================================================
# 9. ĐÁNH GIÁ CHECKPOINT ĐÃ CHỌN
#    (bản gốc inference.py còn kiểm SHA256 checkpoint, đối chiếu lại metric
#     validation trước khi mở test khóa đúng MỘT lần — ở đây rút gọn)
# ============================================================================
def run_eval(checkpoint_path: Path, manifest_path: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    training_config = config.get("training", {})

    use_ordinal_head = bool(model_config.get("use_ordinal_head", True))
    image_size = int(data_config.get("image_size", 224))
    batch_size = int(training_config.get("eval_batch_size", 2))
    num_workers = int(training_config.get("num_workers", 4))

    model = FourViewDensityModel(use_ordinal_head=use_ordinal_head)
    model.load_state_dict(
        checkpoint.get("model_state_dict", checkpoint), strict=True
    )
    model = model.to(device)

    loader = DataLoader(
        FourViewManifestDataset(
            manifest_path, image_size=image_size, training=False
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    metrics = evaluate(model, loader, device, amp=device.type == "cuda")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


# ============================================================================
# 10. ENTRYPOINT
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--output-dir", required=True)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--manifest", required=True)

    args = parser.parse_args()

    if args.command == "train":
        config = yaml.safe_load(
            Path(args.config).read_text(encoding="utf-8")
        )
        run_training(config, Path(args.output_dir))
    else:
        run_eval(Path(args.checkpoint), Path(args.manifest))


if __name__ == "__main__":
    main()
