from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

from phaseG_rerun.config import Config


class FourViewMammographyDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        view_columns: List[str],
        image_size: int,
        train: bool,
    ) -> None:
        self.dataframe = (
            dataframe
            .copy()
            .reset_index(drop=True)
        )

        self.view_columns = list(
            view_columns
        )

        required_columns = {
            "case_id",
            "label",
            "label_idx",
            "domain",
            *self.view_columns,
        }

        missing_columns = (
            required_columns
            - set(self.dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                "Dataset is missing columns: "
                f"{sorted(missing_columns)}"
            )

        transform_steps = [
            transforms.Resize(
                size=(
                    image_size,
                    image_size,
                ),
                antialias=True,
            ),
        ]

        if train:
            transform_steps.append(
                transforms.RandomHorizontalFlip(
                    p=0.5,
                )
            )

        transform_steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                ),
            ]
        )

        self.transform = transforms.Compose(
            transform_steps
        )

    def __len__(self) -> int:
        return len(
            self.dataframe
        )

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, object]:
        row = self.dataframe.iloc[
            index
        ]

        view_tensors = []

        for column in self.view_columns:
            image_path = Path(
                row[column]
            )

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            with Image.open(
                image_path
            ) as image:
                image = image.convert(
                    "RGB"
                )

                image_tensor = self.transform(
                    image
                )

            view_tensors.append(
                image_tensor
            )

        images = torch.stack(
            view_tensors,
            dim=0,
        )

        label = torch.tensor(
            int(row["label_idx"]),
            dtype=torch.long,
        )

        return {
            "images": images,
            "label": label,
            "case_id": str(
                row["case_id"]
            ),
            "domain": str(
                row["domain"]
            ),
        }


def validate_dataframe(
    dataframe: pd.DataFrame,
    name: str,
    view_columns: List[str],
) -> None:
    required_columns = {
        "case_id",
        "label",
        "label_idx",
        "domain",
        *view_columns,
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{name} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise ValueError(
            f"{name} is empty."
        )

    invalid_labels = sorted(
        set(
            dataframe["label_idx"]
            .astype(int)
            .unique()
        )
        - {0, 1, 2, 3}
    )

    if invalid_labels:
        raise ValueError(
            f"{name} contains invalid labels: "
            f"{invalid_labels}"
        )

    for column in view_columns:
        missing_paths = dataframe[
            column
        ].map(
            lambda path: not Path(
                str(path)
            ).is_file()
        )

        if missing_paths.any():
            examples = dataframe.loc[
                missing_paths,
                [
                    "case_id",
                    column,
                ],
            ].head(10)

            raise FileNotFoundError(
                f"{name} contains missing images "
                f"in {column}:\n"
                f"{examples.to_string(index=False)}"
            )


def load_dataframes(
    config: Config,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    tn_df = pd.read_csv(
        config.tn_split_csv,
        dtype={
            "case_id": str,
            "label": str,
        },
    )

    vindr_df = pd.read_csv(
        config.vindr_pool_csv,
        dtype={
            "case_id": str,
            "label": str,
        },
    )

    tn_df["domain"] = "TN"
    vindr_df["domain"] = "VinDr"

    tn_train_df = (
        tn_df[
            tn_df["split"] == "train"
        ]
        .copy()
        .reset_index(drop=True)
    )

    valid_df = (
        tn_df[
            tn_df["split"] == "valid"
        ]
        .copy()
        .reset_index(drop=True)
    )

    test_df = (
        tn_df[
            tn_df["split"] == "test"
        ]
        .copy()
        .reset_index(drop=True)
    )

    vindr_train_df = (
        vindr_df
        .copy()
        .reset_index(drop=True)
    )

    if len(tn_train_df) != 411:
        raise RuntimeError(
            "Expected 411 TN train cases, "
            f"found {len(tn_train_df)}."
        )

    if len(vindr_train_df) != 500:
        raise RuntimeError(
            "Expected 500 VinDr pool cases, "
            f"found {len(vindr_train_df)}."
        )

    if len(valid_df) != 133:
        raise RuntimeError(
            "Expected 133 TN validation cases, "
            f"found {len(valid_df)}."
        )

    if len(test_df) != 132:
        raise RuntimeError(
            "Expected 132 TN test cases, "
            f"found {len(test_df)}."
        )

    train_df = pd.concat(
        [
            tn_train_df,
            vindr_train_df,
        ],
        ignore_index=True,
    )

    validate_dataframe(
        dataframe=train_df,
        name="mixed train",
        view_columns=config.view_columns,
    )

    validate_dataframe(
        dataframe=valid_df,
        name="TN validation",
        view_columns=config.view_columns,
    )

    validate_dataframe(
        dataframe=test_df,
        name="TN test",
        view_columns=config.view_columns,
    )

    return (
        train_df,
        valid_df,
        test_df,
    )


def build_datasets(
    config: Config,
) -> Tuple[
    FourViewMammographyDataset,
    FourViewMammographyDataset,
    FourViewMammographyDataset,
]:
    train_df, valid_df, test_df = (
        load_dataframes(
            config
        )
    )

    train_dataset = FourViewMammographyDataset(
        dataframe=train_df,
        view_columns=config.view_columns,
        image_size=config.image_size,
        train=True,
    )

    valid_dataset = FourViewMammographyDataset(
        dataframe=valid_df,
        view_columns=config.view_columns,
        image_size=config.image_size,
        train=False,
    )

    test_dataset = FourViewMammographyDataset(
        dataframe=test_df,
        view_columns=config.view_columns,
        image_size=config.image_size,
        train=False,
    )

    return (
        train_dataset,
        valid_dataset,
        test_dataset,
    )


def build_domain_sampler(
    dataframe: pd.DataFrame,
    tn_domain_ratio: float,
    seed: int = 42,
) -> WeightedRandomSampler:
    domain_counts = (
        dataframe["domain"]
        .value_counts()
        .to_dict()
    )

    n_tn = int(
        domain_counts.get(
            "TN",
            0,
        )
    )

    n_vindr = int(
        domain_counts.get(
            "VinDr",
            0,
        )
    )

    if n_tn == 0:
        raise ValueError(
            "No TN training cases found."
        )

    if n_vindr == 0:
        raise ValueError(
            "No VinDr training cases found."
        )

    tn_weight = (
        tn_domain_ratio
        / n_tn
    )

    vindr_weight = (
        1.0 - tn_domain_ratio
    ) / n_vindr

    sample_weights = dataframe[
        "domain"
    ].map(
        {
            "TN": tn_weight,
            "VinDr": vindr_weight,
        }
    )

    if sample_weights.isna().any():
        invalid_domains = sorted(
            dataframe.loc[
                sample_weights.isna(),
                "domain",
            ].unique()
        )

        raise ValueError(
            "Unsupported training domains: "
            f"{invalid_domains}"
        )

    generator = torch.Generator()
    generator.manual_seed(
        seed
    )

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(
            sample_weights.to_numpy(),
            dtype=torch.double,
        ),
        num_samples=len(
            dataframe
        ),
        replacement=True,
        generator=generator,
    )

    return sampler