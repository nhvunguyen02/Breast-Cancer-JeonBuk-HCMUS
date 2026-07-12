import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]


def get_path_from_env(
    variable_name: str,
    default_path: Path,
) -> Path:
    value = os.getenv(variable_name)

    if value:
        return Path(value).expanduser().resolve()

    return default_path.resolve()


@dataclass
class Config:
    tn_split_csv: Path = field(
        default_factory=lambda: get_path_from_env(
            "TN_SPLIT_CSV",
            REPO_ROOT
            / "data"
            / "splits"
            / "tn_split_411_133_132_seed42.csv",
        )
    )

    vindr_pool_csv: Path = field(
        default_factory=lambda: get_path_from_env(
            "VINDR_POOL_CSV",
            REPO_ROOT
            / "data"
            / "splits"
            / "vindr_train_pool_500_seed42.csv",
        )
    )

    output_dir: Path = field(
        default_factory=lambda: get_path_from_env(
            "PHASEG_OUTPUT_DIR",
            REPO_ROOT
            / "outputs"
            / "phaseG_baseline_oldlike",
        )
    )

    log_dir: Path = field(
        default_factory=lambda: get_path_from_env(
            "PHASEG_LOG_DIR",
            REPO_ROOT
            / "logs"
            / "phaseG_baseline_oldlike",
        )
    )

    image_size: int = 224
    num_views: int = 4
    num_classes: int = 4

    batch_size: int = 2
    num_workers: int = 2

    epochs: int = 50
    patience: int = 10
    min_delta: float = 1e-4

    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    scheduler_step_size: int = 25
    scheduler_gamma: float = 0.5

    tn_domain_ratio: float = 0.6

    focal_gamma: float = 2.0
    cb_beta: float = 0.99

    seed: int = 42
    device: str = "cuda"
    use_amp: bool = False

    model_name: str = "densenet121"
    fusion: str = "mean_logits"

    view_columns: List[str] = field(
        default_factory=lambda: [
            "left_cc_path",
            "left_mlo_path",
            "right_cc_path",
            "right_mlo_path",
        ]
    )

    label_to_index: Dict[str, int] = field(
        default_factory=lambda: {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
        }
    )

    def create_directories(self) -> None:
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.log_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def validate(self) -> None:
        if not self.tn_split_csv.is_file():
            raise FileNotFoundError(
                "TN split CSV not found: "
                f"{self.tn_split_csv}"
            )

        if not self.vindr_pool_csv.is_file():
            raise FileNotFoundError(
                "VinDr pool CSV not found: "
                f"{self.vindr_pool_csv}"
            )

        if self.image_size <= 0:
            raise ValueError(
                "image_size must be positive."
            )

        if self.num_views != 4:
            raise ValueError(
                "Exactly four mammography views are required."
            )

        if self.num_classes != 4:
            raise ValueError(
                "Exactly four density classes are required."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if self.num_workers < 0:
            raise ValueError(
                "num_workers cannot be negative."
            )

        if self.epochs <= 0:
            raise ValueError(
                "epochs must be positive."
            )

        if self.patience <= 0:
            raise ValueError(
                "patience must be positive."
            )

        if self.min_delta < 0:
            raise ValueError(
                "min_delta cannot be negative."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay cannot be negative."
            )

        if self.scheduler_step_size <= 0:
            raise ValueError(
                "scheduler_step_size must be positive."
            )

        if not 0.0 < self.scheduler_gamma <= 1.0:
            raise ValueError(
                "scheduler_gamma must be in (0, 1]."
            )

        if not 0.0 < self.tn_domain_ratio < 1.0:
            raise ValueError(
                "tn_domain_ratio must be in (0, 1)."
            )

        if not 0.0 <= self.cb_beta < 1.0:
            raise ValueError(
                "cb_beta must be in [0, 1)."
            )

        if self.focal_gamma < 0:
            raise ValueError(
                "focal_gamma cannot be negative."
            )

        if len(self.view_columns) != self.num_views:
            raise ValueError(
                "view_columns must contain four columns."
            )

        if sorted(
            self.label_to_index.values()
        ) != list(range(self.num_classes)):
            raise ValueError(
                "label_to_index must map classes to 0, 1, 2, 3."
            )


config = Config()
