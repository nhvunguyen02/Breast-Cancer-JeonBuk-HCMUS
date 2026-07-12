#!/bin/bash
# Multi-seed comparison: Raw mean-fusion baseline vs crop+masked-pool+gated.
set -e
cd /home/oem/Desktop/Ngoc/Breast-Cancer-JeonBuk-HCMUS
source /home/oem/miniconda3/etc/profile.d/conda.sh
conda activate ngoc_bc

RAW_CSV=data/splits/all_splits_with_paths.csv
GATED_CSV=data/cache/brm_nonorm/all_splits_brm_nonorm.csv
COMMON="--preprocess none --gpu 3 --batch-size 32 --num-workers 16"

for SEED in 1 2 3; do
  echo ">>> RAW seed $SEED"
  python src/train_phaseG_mixed_loss.py --tn-split-csv $RAW_CSV --fusion mean \
      --seed $SEED $COMMON --out-dir outputs_ms_raw > logs/ms_raw_$SEED.log 2>&1

  echo ">>> GATED seed $SEED"
  python src/train_phaseG_mixed_loss.py --tn-split-csv $GATED_CSV --masked-pool --fusion gated \
      --seed $SEED $COMMON --out-dir outputs_ms_gated > logs/ms_gated_$SEED.log 2>&1
done
echo "ALL DONE"
