#!/bin/bash
set -e

SCRIPT_NAME=$(basename "$0" .sh)
RUN_TAG="${SCRIPT_NAME#run_}"
OUTPUT_ROOT="output_${RUN_TAG}"

DATA_ROOT="./dataset"
INPUT_SUBDIR="images"
LAP_LEVELS=2
DECOMP_FUNC="pn"

run_case () {
    CASE=$1
    DATASET=$2

    OUTPUT_DIR="${OUTPUT_ROOT}/${CASE}"

    echo "========== Running ${CASE} =========="
    echo "Dataset: ${DATASET}"
    echo "Output : ${OUTPUT_DIR}"

    python train_lap.py \
        -s "${DATASET}" \
        -i "${INPUT_SUBDIR}" \
        --eval \
        --backbone taming \
        --optimizer_type sparse_adam \
        --lap_n_levels "$LAP_LEVELS" \
        --decomposition_func "$DECOMP_FUNC" \
        --blur_up \
        --inherit_ratio 1.0 \
        --iter_plan 10000_20000 \
        --inherit_state only_xyz \
        --percent_dense 0.01 \
        -m "${OUTPUT_DIR}"

    python render_pn.py \
        -s "$DATASET" \
        -i "$INPUT_SUBDIR" \
        -m "$OUTPUT_DIR" \
        --eval \
        --backbone taming \
        --skip_train \
        --lap_n_levels "$LAP_LEVELS" \
        --decomposition_func "$DECOMP_FUNC" \
        --blur_up

    python metrics.py -m "${OUTPUT_DIR}/Reconstructed/"
    python get_result.py --base_dir "./${OUTPUT_DIR}/"
}

# 360_v2
run_case "bicycle"  "${DATA_ROOT}/360_v2/bicycle"
run_case "flowers"  "${DATA_ROOT}/360_v2/flowers"
run_case "garden"   "${DATA_ROOT}/360_v2/garden"
run_case "stump"    "${DATA_ROOT}/360_v2/stump"
run_case "treehill" "${DATA_ROOT}/360_v2/treehill"
run_case "room"     "${DATA_ROOT}/360_v2/room"
run_case "counter"  "${DATA_ROOT}/360_v2/counter"
run_case "kitchen"  "${DATA_ROOT}/360_v2/kitchen"
run_case "bonsai"   "${DATA_ROOT}/360_v2/bonsai"


python summarize_laplacian_results.py --base_dir "$OUTPUT_ROOT"

echo "All experiments finished!"
