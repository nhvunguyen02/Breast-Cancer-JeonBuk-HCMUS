# TN-Mammo Current Best Macro-F1

## Task

Four-class mammographic breast-density classification: A, B, C and D.

Each examination contains four mammogram views:

1. L-CC
2. L-MLO
3. R-CC
4. R-MLO

## Architecture

- Shared DenseNet121 backbone
- Full four-view mammogram input
- Mean feature fusion
- Flat four-class classification head
- Auxiliary CORAL ordinal loss during training only
- Final inference uses flat-head argmax
- Input resolution: 224 x 224

The CORAL output is not used as the final prediction decoder.
Final A/B/C/D predictions are produced by the flat four-class head.

## Final selected checkpoint

`checkpoint/best_model.pt`

The checkpoint was selected using TN-Mammo validation Macro-F1.

## TN-Mammo locked-test result

| Metric | Result |
|---|---:|
| Macro-F1 | **0.7022 (70.22%%)** |
| Accuracy | 0.6818 |
| Balanced accuracy | 0.7454 |
| QWK | 0.7643 |
| Within-one accuracy | 1.0000 |
| Severe errors | 0 |

Exact Macro-F1: `0.7021551131`.

Confusion matrix:

```text
[[4, 0, 0, 0],
 [2, 15, 9, 0],
 [0, 8, 37, 12],
 [0, 0, 11, 34]]
```

## Files

- `train.py`: model-training entrypoint
- `evaluate.py`: checkpoint evaluation entrypoint
- `config.yaml`: exact runtime configuration for the selected E1 run
- `src/tn_mammo/`: model, dataset, loss, metrics and utility modules
- `checkpoint/best_model.pt`: selected checkpoint
- `README.md`: how to run and follow the fusion experiment plan

## Evaluation protocol

The TN-Mammo locked test has already been evaluated and is treated as a historical burned test.
It must not be used for additional model selection, calibration or hyperparameter tuning.
