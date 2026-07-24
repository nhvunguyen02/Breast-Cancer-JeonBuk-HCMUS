# -*- coding: utf-8 -*-
"""Pipeline biến đổi ảnh cho 4 view mammogram."""
from __future__ import annotations

from PIL import Image, ImageOps
from torchvision import transforms


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


def build_four_view_transform(
    image_size: int, training: bool
) -> transforms.Compose:
    """Biến đổi dùng chung cho cả 4 view của một ca.

    Không flip/rotate độc lập từng view vì sẽ phá consistency giải phẫu;
    lúc train chỉ augment quang học (ColorJitter nhẹ).
    """
    photometric = (
        [transforms.ColorJitter(brightness=0.10, contrast=0.10)]
        if training
        else []
    )
    return transforms.Compose([
        SquarePad(),
        transforms.Resize((image_size, image_size), antialias=True),
        *photometric,
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
