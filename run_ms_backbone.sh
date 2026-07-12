#!/bin/bash
# Multi-seed: ConvNeXt-Base vs DenseNet121 (raw, mean fusion).
set -e
cd /home/oem/Desktop/Ngoc/Breast-Cancer-JeonBuk-HCMUS
source /home/oem/miniconda3/etc/profile.d/conda.sh
conda activate ngoc_bc

CSV=data/splits/all_splits_with_paths.csv
COMMON="--tn-split-csv $CSV --preprocess none --fusion mean --gpu 3 --num-workers 16"

for SEED in 1 2 3; do
  echo ">>> DENSENET seed $SEED"
  python src/train_phaseG_mixed_loss.py --backbone densenet121 --batch-size 32 \
      --seed $SEED $COMMON --out-dir outputs_ms_dn > logs/msb_dn_$SEED.log 2>&1

  echo ">>> CONVNEXT-BASE seed $SEED"
  python src/train_phaseG_mixed_loss.py --backbone convnext_base --batch-size 16 \
      --seed $SEED $COMMON --out-dir outputs_ms_cnb > logs/msb_cnb_$SEED.log 2>&1
done
echo "ALL DONE"
