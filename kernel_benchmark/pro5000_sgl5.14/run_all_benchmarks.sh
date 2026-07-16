#!/usr/bin/env bash

# 使用 pro5000_sgl5.14 获取 Pro5000 数据。
#
# 默认行为：
#   1. 所有新结果写入 pro5000_sgl5.14/tmp；
#   2. 不修改仓库中已经提交的 bench_data；
#   3. 每一段会打印其结果对应的 bench_data 文件路径。
#
#
# 可覆盖的环境变量：
#   PYTHON_BIN          Python 命令，默认 python3
#   CUDA_VISIBLE_DEVICES 使用的 GPU，默认 0
#   CONFIG              模型配置路径
#   RESULT_DIR          临时结果目录

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

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
    raise SystemExit("没有可合并的 CSV 文件")

frames = [pd.read_csv(path) for path in inputs]
pd.concat(frames, ignore_index=True).to_csv(output, index=False)
PY
}

echo "结果暂存目录：${RESULT_DIR}"
echo "模型配置文件：${CONFIG}"

# -----------------------------------------------------------------------------
# 1. 显存带宽校准
# -----------------------------------------------------------------------------
# 这部分不会生成 bench_data CSV。它用于检查 hardware/gpu.py 中 Pro5000 的
# mem_bw 参数是否与实测显存带宽相符。
#
# 输出日志：
#   ${RESULT_DIR}/memory_bandwidth.log
# 对应代码参数：
#   hardware/gpu.py 中 pro5000.mem_bw
echo
echo "[1/7] 测量 Pro5000 显存带宽"
"${PYTHON_BIN}" "${SCRIPT_DIR}/memory_bandwidth.py" \
    2>&1 | tee "${RESULT_DIR}/memory_bandwidth.log"

# -----------------------------------------------------------------------------
# 2. 普通 GEMM
# -----------------------------------------------------------------------------
# 每组 (K, N) 会覆盖脚本内置的全部 M 值，最后合并成一个 CSV。
#
# 结果对应：
#   bench_data/gemm/pro5000/data.csv
echo
echo "[2/7] 测量普通 GEMM"
gemm_parts=()
while read -r k n; do
    part="${RESULT_DIR}/gemm_${k}_${n}.csv"
    m_values=(
        1 2 4 8 16 32 64 128 224 256 512 1024
        4096 8192 16384 32768 65536 131072
    )
    # (K=2048, N=5120) 的已提交数据从 M=8 开始；不生成目标 CSV
    # 中不存在的 M=1、2、4 三行。
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
# 每次运行现有脚本只测一个 (batch_size, kv_len) 点；下面的列表覆盖当前
# bench_data/mha/decode/pro5000/16-2-256.csv 中使用的全部点。
#
# 结果对应：
#   bench_data/mha/decode/pro5000/16-2-256.csv
echo
echo "[3/7] 测量 MHA decode"
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
# 临时结果：
#   ${RESULT_DIR}/mha_prefill_partial.csv
# 最终应放入：
#   bench_data/mha/prefill/pro5000/16-2-256.csv
echo
echo "[4/7] 测量 MHA prefill"
rm -f "${RESULT_DIR}/attention_benchmark.csv"
"${PYTHON_BIN}" "${SCRIPT_DIR}/flashinfer_mha_prefill.py" \
    --config-path "${CONFIG}"
mv "${RESULT_DIR}/attention_benchmark.csv" \
    "${RESULT_DIR}/mha_prefill_partial.csv"
echo "临时结果：${RESULT_DIR}/mha_prefill_partial.csv"
echo "目标文件：${REPO_ROOT}/bench_data/mha/prefill/pro5000/16-2-256.csv"

# -----------------------------------------------------------------------------
# 5. Grouped GEMM decode
# -----------------------------------------------------------------------------
# 使用 Qwen3.5 MoE 配置、FP8 W8A8、TP=1，生成 decode 阶段 grouped GEMM 数据。
#
# 结果对应：
#   bench_data/grouped_gemm/decode/pro5000/data.csv
echo
echo "[5/7] 测量 Grouped GEMM decode"
rm -f "${RESULT_DIR}/groupedgemm_decode.csv"
"${PYTHON_BIN}" "${SCRIPT_DIR}/sglang_fused_moe.py" \
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
# 使用与 decode 相同的模型和数值精度配置，生成 prefill 阶段数据。
#
# 结果对应：
#   bench_data/grouped_gemm/prefill/pro5000/data.csv
echo
echo "[6/7] 测量 Grouped GEMM prefill"
rm -f "${RESULT_DIR}/groupedgemm_prefill.csv"
"${PYTHON_BIN}" "${SCRIPT_DIR}/sglang_fused_moe.py" \
    --config-path "${CONFIG}" \
    --mode prefill \
    --num-gpus 1 \
    --tp-size 1 \
    --use-fp8-w8a8 \
    --gpu-tflops 536
mv "${RESULT_DIR}/groupedgemm_prefill.csv" \
    "${RESULT_DIR}/grouped_gemm_prefill.csv"

# -----------------------------------------------------------------------------
# 7. GDN decode / prefill 原始 kernel 日志
# -----------------------------------------------------------------------------
# Decode 使用：
#   sgl_causal_conv1d_update.py
#   sgl_gdn_update.py
# 最终应整理到：
#   bench_data/gdn/decode/pro5000/4-16-128-32-128.csv
#
# Prefill 使用：
#   sgl_causal_conv1d.py
#   sgl_chunk_gdn.py
# 最终应整理到：
#   bench_data/gdn/prefill/pro5000/4-16-128-32-128.csv
echo
echo "[7/7] 测量 GDN decode / prefill 原始 kernel 数据"
"${PYTHON_BIN}" "${REPO_ROOT}/kernel_benchmark/sgl_causal_conv1d_update.py" \
    2>&1 | tee "${RESULT_DIR}/gdn_decode_causal_conv1d_update.log"
"${PYTHON_BIN}" "${REPO_ROOT}/kernel_benchmark/sgl_gdn_update.py" \
    2>&1 | tee "${RESULT_DIR}/gdn_decode_update.log"
"${PYTHON_BIN}" "${REPO_ROOT}/kernel_benchmark/sgl_causal_conv1d.py" \
    2>&1 | tee "${RESULT_DIR}/gdn_prefill_causal_conv1d.log"
"${PYTHON_BIN}" "${REPO_ROOT}/kernel_benchmark/sgl_chunk_gdn.py" \
    2>&1 | tee "${RESULT_DIR}/gdn_prefill_kernels.log"

echo "GDN decode 日志目录：${RESULT_DIR}"
echo "GDN decode 最终目标：${REPO_ROOT}/bench_data/gdn/decode/pro5000/4-16-128-32-128.csv"
echo "GDN prefill 日志目录：${RESULT_DIR}"
echo "GDN prefill 最终目标：${REPO_ROOT}/bench_data/gdn/prefill/pro5000/4-16-128-32-128.csv"

echo
echo "全部 benchmark 已运行完成。"
echo "本次结果目录：${RESULT_DIR}"
