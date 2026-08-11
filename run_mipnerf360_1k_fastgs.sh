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
    EXTRA_ARGS="$3"
    MULT_ARGS="$4"

    OUTPUT_DIR="${OUTPUT_ROOT}/${CASE}"

    echo "========== Running ${CASE} =========="
    echo "Dataset: ${DATASET}"
    echo "Output : ${OUTPUT_DIR}"

    python train_lap.py \
        -s "${DATASET}" \
        -i "${INPUT_SUBDIR}" \
        --eval \
        --densification_interval 100 \
        --optimizer_type default \
        --lap_n_levels "$LAP_LEVELS" \
        --decomposition_func "$DECOMP_FUNC" \
        --blur_up \
        --inherit_ratio 1.0 \
        --iter_plan 10000_20000 \
        --inherit_state only_xyz \
        --backbone fastgs \
        ${EXTRA_ARGS} \
        -m "${OUTPUT_DIR}"

    python render_pn.py \
        -s "$DATASET" \
        -i "$INPUT_SUBDIR" \
        -m "$OUTPUT_DIR" \
        --eval \
        --skip_train \
        --lap_n_levels "$LAP_LEVELS" \
        --decomposition_func "$DECOMP_FUNC" \
        --blur_up
    python metrics.py -m "${OUTPUT_DIR}/Reconstructed/"
    python get_result.py --base_dir "./${OUTPUT_DIR}/"
}

# 360_v2
run_case "bicycle_big"  "${DATA_ROOT}/360_v2/bicycle"  "--loss_thresh 0.05 --grad_abs_thresh 0.0004"  ""
run_case "flowers_big"  "${DATA_ROOT}/360_v2/flowers"  "--loss_thresh 0.05 --dense 0.005 --grad_abs_thresh 0.0005"  ""
run_case "garden_big"   "${DATA_ROOT}/360_v2/garden"   "--loss_thresh 0.03 --highfeature_lr 0.02 --grad_abs_thresh 0.0003" ""
run_case "stump_big"    "${DATA_ROOT}/360_v2/stump"    "--loss_thresh 0.05 --dense 0.004 --grad_abs_thresh 0.0005" ""
run_case "treehill_big" "${DATA_ROOT}/360_v2/treehill" "--loss_thresh 0.05 --dense 0.01 --grad_abs_thresh 0.0012" ""
run_case "room_big"     "${DATA_ROOT}/360_v2/room"     "--loss_thresh 0.05 --highfeature_lr 0.02 --grad_abs_thresh 0.0003" ""
run_case "counter_big"  "${DATA_ROOT}/360_v2/counter"  "--loss_thresh 0.05 --highfeature_lr 0.02 --grad_abs_thresh 0.0003" ""
run_case "kitchen_big"  "${DATA_ROOT}/360_v2/kitchen"  "--loss_thresh 0.05 --highfeature_lr 0.02 --grad_abs_thresh 0.00015" ""
run_case "bonsai_big"   "${DATA_ROOT}/360_v2/bonsai"   "--loss_thresh 0.05 --highfeature_lr 0.02 --grad_abs_thresh 0.00015" ""


python summarize_laplacian_results.py --base_dir "$OUTPUT_ROOT"

echo "All experiments finished!"
