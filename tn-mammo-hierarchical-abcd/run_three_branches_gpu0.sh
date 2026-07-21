#!/usr/bin/env bash

set +e

source /mnt/hcmus/breast_vn/miniconda3/etc/profile.d/conda.sh
conda activate tnmammo

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
RUN_ID="task1_improve_$(date +%Y%m%d_%H%M%S)"
PIPE_DIR="${ROOT}/outputs/${RUN_ID}"
RESOLVED_DIR="${PIPE_DIR}/resolved"
SMOKE_DIR="${PIPE_DIR}/smoke"
mkdir -p "${PIPE_DIR}" "${RESOLVED_DIR}" "${SMOKE_DIR}" "${ROOT}/logs"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

FAILED_STEPS=()
SETUP_FAILED=0
TEST_PHASE_EXECUTED=false

run_step() {
  local NAME="$1"
  shift
  echo
  echo "============================================================"
  echo "[STEP_START] ${NAME} | $(date --iso-8601=seconds)"
  echo "[COMMAND] $*"
  "$@"
  local RC=$?
  if [[ "${RC}" -eq 0 ]]; then
    echo "[STEP_PASS] ${NAME} | $(date --iso-8601=seconds)"
  else
    echo "[STEP_FAIL] ${NAME} rc=${RC} | $(date --iso-8601=seconds)"
    FAILED_STEPS+=("${NAME}")
  fi
  return "${RC}"
}

CONFIG_R1="${ROOT}/configs/r1_soft_hierarchy.yaml"
CONFIG_R2="${ROOT}/configs/r2_cd_residual.yaml"
CONFIG_R3="${ROOT}/configs/r3_cd_specific_fusion.yaml"

cat <<EOF
[TASK1_IMPROVE] RUN_ID=${RUN_ID}
[TASK1_IMPROVE] ROOT=${ROOT}
[TASK1_IMPROVE] PIPE_DIR=${PIPE_DIR}
[TASK1_IMPROVE] GPU physical 0 via CUDA_VISIBLE_DEVICES=0
[TASK1_IMPROVE] Branch order: R1 soft hierarchy -> R2 C/D residual -> R3 C/D-specific fusion
[TASK1_IMPROVE] All branches start independently from ImageNet for fair ablation
[TASK1_IMPROVE] Model selection: TN validation macro-F1 only
[TASK1_IMPROVE] TN test is already exposed; new test results are exploratory reused-test reports only
EOF

run_step audit_environment \
  python3 -u -X faulthandler "${ROOT}/00_audit_environment.py" \
  --root "${ROOT}" \
  --output-dir "${PIPE_DIR}" \
  --config "${CONFIG_R1}" \
  --config "${CONFIG_R2}" \
  --config "${CONFIG_R3}" || SETUP_FAILED=1

if [[ "${SETUP_FAILED}" -eq 0 ]]; then
  run_step resolve_manifests \
    python3 -u -X faulthandler "${ROOT}/01_resolve_manifests.py" \
    --config "${CONFIG_R1}" \
    --output-dir "${RESOLVED_DIR}" || SETUP_FAILED=1
fi

if [[ "${SETUP_FAILED}" -eq 0 ]]; then
  run_step preprocessing_audit \
    python3 -u -X faulthandler "${ROOT}/02_dataset.py" \
    --audit \
    --manifest "${RESOLVED_DIR}/resolved_vindr_dev.csv" \
    --manifest "${RESOLVED_DIR}/resolved_tn_dev.csv" \
    --output "${PIPE_DIR}/preprocessing_audit.json" \
    --max-cases 4 || SETUP_FAILED=1
fi

if [[ "${SETUP_FAILED}" -eq 0 ]]; then
  run_step unit_tests \
    python3 -m pytest -q "${ROOT}/tests" || SETUP_FAILED=1
fi

BRANCH_NAMES=("R1_soft_hierarchy" "R2_cd_residual" "R3_cd_specific_fusion")
BRANCH_CONFIGS=("${CONFIG_R1}" "${CONFIG_R2}" "${CONFIG_R3}")

if [[ "${SETUP_FAILED}" -eq 0 ]]; then
  for INDEX in 0 1 2; do
    BRANCH="${BRANCH_NAMES[$INDEX]}"
    CONFIG="${BRANCH_CONFIGS[$INDEX]}"
    SMOKE_RUN="${SMOKE_DIR}/${BRANCH}"
    FULL_RUN="${PIPE_DIR}/${BRANCH}"

    run_step "${BRANCH}_smoke" \
      python3 -u -X faulthandler "${ROOT}/07_train_mixed.py" \
      --config "${CONFIG}" \
      --resolved-dir "${RESOLVED_DIR}" \
      --run-dir "${SMOKE_RUN}" \
      --smoke
    BRANCH_SMOKE_RC=$?

    if [[ "${BRANCH_SMOKE_RC}" -eq 0 ]]; then
      run_step "${BRANCH}_train" \
        python3 -u -X faulthandler "${ROOT}/07_train_mixed.py" \
        --config "${CONFIG}" \
        --resolved-dir "${RESOLVED_DIR}" \
        --run-dir "${FULL_RUN}"
      BRANCH_TRAIN_RC=$?
    else
      BRANCH_TRAIN_RC=1
    fi

    if [[ "${BRANCH_TRAIN_RC}" -eq 0 ]]; then
      run_step "${BRANCH}_tn_valid" \
        python3 -u -X faulthandler "${ROOT}/08_evaluate_valid.py" \
        --config "${FULL_RUN}/resolved_config.yaml" \
        --resolved-dir "${RESOLVED_DIR}" \
        --run-dir "${FULL_RUN}" \
        --checkpoint "${FULL_RUN}/best_tn_checkpoint.pt"
      BRANCH_VALID_RC=$?
    else
      BRANCH_VALID_RC=1
    fi

    if [[ "${BRANCH_VALID_RC}" -eq 0 ]]; then
      run_step "${BRANCH}_dashboard" \
        python3 -u -X faulthandler "${ROOT}/11_generate_reports.py" \
        --run-dir "${FULL_RUN}"
    fi
  done
fi

ALL_VALID_COMPLETE=1
for BRANCH in "${BRANCH_NAMES[@]}"; do
  if [[ ! -f "${PIPE_DIR}/${BRANCH}/VALIDATION_DONE.json" ]]; then
    ALL_VALID_COMPLETE=0
  fi
done

if [[ "${ALL_VALID_COMPLETE}" -eq 1 ]]; then
  run_step aggregate_validation \
    python3 -u -X faulthandler "${ROOT}/11_generate_reports.py" \
    --aggregate-root "${PIPE_DIR}"

  run_step freeze_three_branches_before_reused_test \
    python3 -u -X faulthandler "${ROOT}/09_freeze_branches.py" \
    --root "${PIPE_DIR}"
  FREEZE_RC=$?

  if [[ "${FREEZE_RC}" -eq 0 ]]; then
    TEST_PHASE_EXECUTED=true
    for INDEX in 0 1 2; do
      BRANCH="${BRANCH_NAMES[$INDEX]}"
      FULL_RUN="${PIPE_DIR}/${BRANCH}"
      run_step "${BRANCH}_exploratory_reused_tn_test" \
        python3 -u -X faulthandler "${ROOT}/10_evaluate_reused_test.py" \
        --root "${PIPE_DIR}" \
        --config "${FULL_RUN}/resolved_config.yaml" \
        --resolved-dir "${RESOLVED_DIR}" \
        --run-dir "${FULL_RUN}" \
        --checkpoint "${FULL_RUN}/best_tn_checkpoint.pt"
    done

    run_step aggregate_validation_and_reused_test \
      python3 -u -X faulthandler "${ROOT}/11_generate_reports.py" \
      --aggregate-root "${PIPE_DIR}"
  fi
else
  echo "[TEST_SKIPPED] Not all three validation branches completed; no reused TN-test evaluation was run."
fi

if [[ "${#FAILED_STEPS[@]}" -eq 0 ]]; then
  PIPELINE_STATUS=0
else
  PIPELINE_STATUS=1
fi

FAILED_TEXT=""
if [[ "${#FAILED_STEPS[@]}" -gt 0 ]]; then
  FAILED_TEXT=$(IFS=,; echo "${FAILED_STEPS[*]}")
fi

cat > "${PIPE_DIR}/PIPELINE_DONE.txt" <<EOF
TASK1_IMPROVEMENT_PIPELINE_DONE
time=$(date --iso-8601=seconds)
run_id=${RUN_ID}
pipeline_dir=${PIPE_DIR}
pipeline_status=${PIPELINE_STATUS}
setup_failed=${SETUP_FAILED}
failed_steps=${FAILED_TEXT}
selection_metric=TN_validation_macro_F1
test_phase_executed=${TEST_PHASE_EXECUTED}
test_scientific_status=EXPLORATORY_REUSED_TEST_NOT_NEW_LOCKED_ESTIMATE
no_post_test_tuning_permitted=true
EOF

cat "${PIPE_DIR}/PIPELINE_DONE.txt"
echo "[TASK1_IMPROVE] Pipeline finished."
