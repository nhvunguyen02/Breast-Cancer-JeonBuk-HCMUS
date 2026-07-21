# TN-Mammo Phase H — Final Locked Evaluation

## Pipeline overview

This folder contains the final Phase H artifacts for four-class mammographic breast-density classification:

- A: almost entirely fatty
- B: scattered fibroglandular density
- C: heterogeneously dense
- D: extremely dense

Each examination contains four full mammogram views in the fixed order:

1. L-CC
2. L-MLO
3. R-CC
4. R-MLO

## Final model

- Shared DenseNet121 backbone
- Four-view full mammogram input
- Mean feature fusion
- Four-class flat classification head
- CORAL ordinal auxiliary supervision
- Final decoder: flat-head argmax
- Image size: 224 × 224
- ImageNet initialization
- Primary selection metric: TN-validation Macro-F1

## Development protocol

- TN training: 411 cases
- TN validation: 133 cases
- VinDr source data used during mixed-domain training
- TN-domain sampling ratio: 0.60
- Class-balanced focal classification loss
- CORAL ordinal auxiliary loss
- AdamW optimizer
- Early stopping based on TN-validation Macro-F1

The final checkpoint was selected only from TN validation.

## Final results

| Dataset | Cases | Accuracy | Balanced Accuracy | Macro-F1 | QWK | C→D | D→C | Severe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TN validation | 133 | 0.7218 | 0.7367 | 0.7623 | 0.7843 | 15 | 11 | 0 |
| TN locked test | 132 | 0.6818 | 0.7454 | 0.7022 | 0.7643 | 12 | 11 | 0 |
| VinDr locked test | 992 | 0.6401 | 0.7374 | 0.5580 | 0.5360 | 284 | 6 | 1 |

## Test status

- TN test 132: burned historical test
- VinDr test 992: burned historical external test

These datasets must not be used for further model selection, calibration, threshold tuning, or hyperparameter optimization.

## Phase I

Phase I uses a new protocol:

- TN train and validation combined into 544 development cases
- Fixed five-fold case-level cross-validation
- Seeds 42, 43, and 44
- Separate inner validation for early stopping
- Out-of-fold evaluation
- ImageNet initialization
- No initialization from Phase H checkpoints

A new temporal or external holdout is required for an unbiased final claim.
