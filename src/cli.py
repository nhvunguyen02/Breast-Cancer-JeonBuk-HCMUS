"""Command-line argument parsing (kept torch-free so the entrypoint can set
CUDA_VISIBLE_DEVICES before torch is ever imported)."""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", type=str, default="1")
    parser.add_argument(
        "--backbone",
        type=str,
        default="densenet121",
        choices=["densenet121", "densenet169", "efficientnet_b0",
                 "efficientnet_b3", "efficientnet_b5", "resnet50",
                 "convnext_tiny", "convnext_small", "convnext_base"],
        help="Shared backbone for the 4-view model.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--tn-split-csv",
        type=str,
        default="data/splits/all_splits_with_paths.csv",
    )
    parser.add_argument(
        "--vindr-split-csv",
        type=str,
        default="data/splits/vindr_4view_density_split.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs",
    )

    parser.add_argument(
        "--tn-domain-ratio",
        type=float,
        default=0.5,
        help="Sampling mass for TN train. 0.5 means TN and VinDr appear equally often per epoch.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="cb_focal",
        choices=["ce", "focal", "cb_focal"],
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--cb-beta",
        type=float,
        default=0.999,
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        default="none",
        choices=["none", "brm"],
        help="none = raw resize + ImageNet norm; brm = BRM stage0 "
             "(crop-to-breast + in-mask p2-p98 normalize) before resize.",
    )
    parser.add_argument(
        "--brm-pectoral",
        action="store_true",
        help="With --preprocess brm, also remove the pectoral muscle in MLO "
             "views (conservative corner-anchored bright-triangle detector).",
    )
    parser.add_argument(
        "--masked-pool",
        action="store_true",
        help="Pool conv features only over the breast region (derived from the "
             "zeroed-background input) instead of a plain global average pool.",
    )
    parser.add_argument(
        "--attn-loss-weight",
        type=float,
        default=0.0,
        help="Weight of the attention-suppression loss (penalizes activation "
             "energy outside the breast mask). 0 = off.",
    )
    parser.add_argument(
        "--fusion",
        type=str,
        default="mean",
        choices=["mean", "gated"],
        help="How to combine the 4 views: 'mean' averages per-view logits; "
             "'gated' learns a gated-attention weight per view.",
    )

    return parser.parse_args()
