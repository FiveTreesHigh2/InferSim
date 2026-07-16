#!/usr/bin/env bash

# Run the scripts under kernel_benchmark to collect PRO5000 benchmark data.
#
# Default behavior:
#   1. Write all new results to kernel_benchmark/tmp.
#   2. Do not modify committed files under bench_data.
#   3. Print the target bench_data path for each benchmark section.
#
#
# Environment variable overrides:
#   PYTHON_BIN           Python command; defaults to python3
#   CUDA_VISIBLE_DEVICES GPU to use; defaults to 0
#   CONFIG               Model configuration path
#   RESULT_DIR           Temporary result directory

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
# Model config.json path
CONFIG="${CONFIG:-${REPO_ROOT}/hf_configs/qwen3.5-35B-A3B_config.json}"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/tmp}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ "${RESULT_DIR}" != /* ]]; then
    RESULT_DIR="${PWD}/${RESULT_DIR}"
fi
mkdir -p "${RESULT_DIR}"
RESULT_DIR="$(cd "${RESULT_DIR}" && pwd)"
cd "${RESULT_DIR}"

merge_csv_files() {
    local output_file="$1"
    shift

    "${PYTHON_BIN}" - "${output_file}" "$@" <<'PY'
import pathlib
import sys

import pandas as pd

output = pathlib.Path(sys.argv[1])
inputs = [pathlib.Path(path) for path in sys.argv[2:]]
if not inputs:
    raise SystemExit("No CSV files to merge")

frames = [pd.read_csv(path) for path in inputs]
pd.concat(frames, ignore_index=True).to_csv(output, index=False)
PY
}

echo "Result directory: ${RESULT_DIR}"
echo "Model configuration: ${CONFIG}"

# -----------------------------------------------------------------------------
# 1. Device-memory bandwidth calibration
# -----------------------------------------------------------------------------
# This section does not generate a bench_data CSV. It checks whether the
# PRO5000 mem_bw setting in hardware/gpu.py matches measured memory bandwidth.
#
# Output log:
#   ${RESULT_DIR}/memory_bandwidth.log
# Related setting:
#   pro5000.mem_bw in hardware/gpu.py
echo
echo "[1/7] Measure PRO5000 device-memory bandwidth"
"${PYTHON_BIN}" "${SCRIPT_DIR}/memory_bandwidth.py" \
    2>&1 | tee "${RESULT_DIR}/memory_bandwidth.log"

# -----------------------------------------------------------------------------
# 2. GEMM
# -----------------------------------------------------------------------------
# Sweep all required M values for each (K, N) pair, then merge the results into
# one CSV file.
#
# Target data:
#   bench_data/gemm/pro5000/data.csv
echo
echo "[2/7] Measure GEMM"
gemm_parts=()
while read -r k n; do
    part="${RESULT_DIR}/gemm_${k}_${n}.csv"
    m_values=(
        1 2 4 8 16 32 64 128 224 256 512 1024
        4096 8192 16384 32768 65536 131072
    )
    # The committed (K=2048, N=5120) data starts at M=8. Do not generate the
    # M=1, 2, and 4 rows that are absent from the target CSV.
    if [[ "${k}" == "2048" && "${n}" == "5120" ]]; then
        m_values=(
            8 16 32 64 128 224 256 512 1024
            4096 8192 16384 32768 65536 131072
        )
    fi
    "${PYTHON_BIN}" "${SCRIPT_DIR}/flashinfer_gemm.py" \
        -k "${k}" \
        -n "${n}" \
        --gpu-tflops 536 \
        --m-values "${m_values[@]}" \
        --output "${part}"
    gemm_parts+=("${part}")
done <<'EOF'
512 2048
2048 1024
2048 5120
2048 9216
2048 12288
4096 2048
EOF

merge_csv_files "${RESULT_DIR}/gemm_data.csv" "${gemm_parts[@]}"

# -----------------------------------------------------------------------------
# 3. MHA decode
# -----------------------------------------------------------------------------
# Each invocation measures one (batch_size, kv_len) point. The list below
# covers all points in bench_data/mha/decode/pro5000/16-2-256.csv.
#
# Target data:
#   bench_data/mha/decode/pro5000/16-2-256.csv
echo
echo "[3/7] Measure MHA decode"
mha_decode_parts=()

run_mha_decode() {
    local batch_size="$1"
    shift

    local kv_len
    local part
    for kv_len in "$@"; do
        part="${RESULT_DIR}/mha_decode_bs${batch_size}_kv${kv_len}.csv"
        rm -f "${RESULT_DIR}/attention_benchmark.csv"
        "${PYTHON_BIN}" "${SCRIPT_DIR}/flashinfer_mha_decode.py" \
            --config-path "${CONFIG}" \
            --kv-cache-dtype bf16 \
            --tp-size 1 \
            --fp16-tflops 274 \
            --batch-size "${batch_size}" \
            --kv-len "${kv_len}"
        mv "${RESULT_DIR}/attention_benchmark.csv" "${part}"
        mha_decode_parts+=("${part}")
    done
}

run_mha_decode 1   1024 4096 5120 8192 16384 32768 65536 131072
run_mha_decode 8   64512
run_mha_decode 16  1024 4096 8192 16384 32768 65536 131072
run_mha_decode 32  1024 4096 8192 16384 32768 65536 131072
run_mha_decode 64  1024 4096 8192 16384 32768 65536 131072
run_mha_decode 128 1024 4096 5120 8192 16384 32768 65536
run_mha_decode 256 1024 4096 8192 16384
run_mha_decode 512 1024 4096 8192

merge_csv_files "${RESULT_DIR}/mha_decode_unsorted.csv" "${mha_decode_parts[@]}"
"${PYTHON_BIN}" - \
    "${RESULT_DIR}/mha_decode_unsorted.csv" \
    "${RESULT_DIR}/mha_decode.csv" <<'PY'
import pathlib
import sys

import pandas as pd

source = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
frame = pd.read_csv(source)
frame.sort_values(["batch_size", "kv_len"], inplace=True, ignore_index=True)
frame.to_csv(target, index=False)
PY

# -----------------------------------------------------------------------------
# 4. MHA prefill
# -----------------------------------------------------------------------------
# Temporary result:
#   ${RESULT_DIR}/mha_prefill_partial.csv
# Final target:
#   bench_data/mha/prefill/pro5000/16-2-256.csv
echo
echo "[4/7] Measure MHA prefill"
rm -f "${RESULT_DIR}/attention_benchmark.csv"
"${PYTHON_BIN}" "${SCRIPT_DIR}/flashinfer_mha_prefill.py" \
    --config-path "${CONFIG}"
mv "${RESULT_DIR}/attention_benchmark.csv" \
    "${RESULT_DIR}/mha_prefill_partial.csv"
echo "Temporary result: ${RESULT_DIR}/mha_prefill_partial.csv"
echo "Target file: ${REPO_ROOT}/bench_data/mha/prefill/pro5000/16-2-256.csv"

# -----------------------------------------------------------------------------
# 5. Grouped GEMM decode
# -----------------------------------------------------------------------------
# Generate decode grouped GEMM data with the Qwen3.5 MoE configuration,
# FP8 W8A8, and TP=1.
#
# Target data:
#   bench_data/grouped_gemm/decode/pro5000/data.csv
echo
echo "[5/7] Measure Grouped GEMM decode"
rm -f "${RESULT_DIR}/groupedgemm_decode.csv"
"${PYTHON_BIN}" "${SCRIPT_DIR}/sgl_grouped_gemm.py" \
    --config-path "${CONFIG}" \
    --mode decode \
    --num-gpus 1 \
    --tp-size 1 \
    --use-fp8-w8a8 \
    --gpu-tflops 536
mv "${RESULT_DIR}/groupedgemm_decode.csv" \
    "${RESULT_DIR}/grouped_gemm_decode.csv"

# -----------------------------------------------------------------------------
# 6. Grouped GEMM prefill
# -----------------------------------------------------------------------------
# Generate prefill data with the same model and precision settings as decode.
#
# Target data:
#   bench_data/grouped_gemm/prefill/pro5000/data.csv
echo
echo "[6/7] Measure Grouped GEMM prefill"
rm -f "${RESULT_DIR}/groupedgemm_prefill.csv"
"${PYTHON_BIN}" "${SCRIPT_DIR}/sgl_grouped_gemm.py" \
    --config-path "${CONFIG}" \
    --mode prefill \
    --num-gpus 1 \
    --tp-size 1 \
    --use-fp8-w8a8 \
    --gpu-tflops 536
mv "${RESULT_DIR}/groupedgemm_prefill.csv" \
    "${RESULT_DIR}/grouped_gemm_prefill.csv"

# -----------------------------------------------------------------------------
# 7. Raw GDN decode and prefill kernel logs
# -----------------------------------------------------------------------------
# Decode scripts:
#   sgl_causal_conv1d_update.py
#   sgl_gdn_update.py
# Final target:
#   bench_data/gdn/decode/pro5000/4-16-128-32-128.csv
#
# Prefill scripts:
#   sgl_causal_conv1d.py
#   sgl_chunk_gdn.py
# Final target:
#   bench_data/gdn/prefill/pro5000/4-16-128-32-128.csv
echo
echo "[7/7] Measure raw GDN decode and prefill kernel data"
"${PYTHON_BIN}" "${REPO_ROOT}/kernel_benchmark/sgl_causal_conv1d_update.py" \
    2>&1 | tee "${RESULT_DIR}/gdn_decode_causal_conv1d_update.log"
"${PYTHON_BIN}" "${REPO_ROOT}/kernel_benchmark/sgl_gdn_update.py" \
    2>&1 | tee "${RESULT_DIR}/gdn_decode_update.log"
"${PYTHON_BIN}" "${REPO_ROOT}/kernel_benchmark/sgl_causal_conv1d.py" \
    2>&1 | tee "${RESULT_DIR}/gdn_prefill_causal_conv1d.log"
"${PYTHON_BIN}" "${REPO_ROOT}/kernel_benchmark/sgl_chunk_gdn.py" \
    2>&1 | tee "${RESULT_DIR}/gdn_prefill_kernels.log"

echo "GDN decode log directory: ${RESULT_DIR}"
echo "GDN decode target: ${REPO_ROOT}/bench_data/gdn/decode/pro5000/4-16-128-32-128.csv"
echo "GDN prefill log directory: ${RESULT_DIR}"
echo "GDN prefill target: ${REPO_ROOT}/bench_data/gdn/prefill/pro5000/4-16-128-32-128.csv"

echo
echo "All benchmarks completed."
echo "Result directory: ${RESULT_DIR}"
