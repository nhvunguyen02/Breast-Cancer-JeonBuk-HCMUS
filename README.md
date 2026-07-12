
## Task

4-class breast density classification:

- A: Almost entirely fatty
- B: Scattered fibroglandular density
- C: Heterogeneously dense
- D: Extremely dense

Input is one mammography exam with four standard views:

- L-CC
- L-MLO
- R-CC
- R-MLO

## Layout

```text
src/
  train_phaseG_mixed_loss.py   # entrypoint: orchestrates the run
  cli.py                       # argparse (torch-free)
  constants.py                 # label / view constants
  data.py                      # split-CSV standardization, dataset, sampler
  models.py                    # DenseNet121 mean-fusion model
  losses.py                    # focal loss + class-weight helpers
  engine.py                    # train / eval loops, test metrics
  utils.py                     # seeding, param count, benchmark IO
```

## Run

```bash
python src/train_phaseG_mixed_loss.py --loss-type cb_focal --gpu 1
```

