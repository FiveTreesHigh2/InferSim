#!/usr/bin/env python3
"""
Triton fused-MoE benchmark for sm120 (RTX PRO 5000).

Replaces deepgemm_grouped_gemm_{contiguous,masked}.py, which are dead on sm120:
sglang 0.5.14 explicitly disables DeepGEMM on sm120 and its AUTO MoE backend falls back
to the Triton fused MoE. So the Triton fused MoE is what actually runs on this card.

InferSim's moe.py only reads up_mfu/down_mfu from the grouped_gemm CSV (it computes the
MoE FLOPs itself), so we measure the fused-MoE latency, derive the effective MFU with the
SAME flop formula moe.py uses (self-consistent), and write both up_mfu and down_mfu.

Two MANUAL switches (add-back per request):
  --use-fp8-w8a8   : block-FP8 experts (matches the model). Off => bf16 experts.
  --use-cuda-graph : time via CUDA-graph replay (drops CPU launch overhead). Off => direct.
For feeding InferSim, use --use-fp8-w8a8 (the model IS block-FP8) and NO --use-cuda-graph
(keeps the same timing convention as the other bench_data). The switches exist for A/B
diagnostics; graphed / bf16 numbers should NOT be fed to the simulator.

Run (single card => --num-gpus 1 --tp-size 1 => ep_size 1 => 256 experts on one GPU):
  python kernel_benchmark/sgl_grouped_gemm.py --config-path hf_configs/qwen3.5-35B-A3B_config.json \
      --mode prefill --num-gpus 1 --tp-size 1 --use-fp8-w8a8 --use-cuda-graph
  python kernel_benchmark/sgl_grouped_gemm.py --config-path hf_configs/qwen3.5-35B-A3B_config.json \
      --mode decode  --num-gpus 1 --tp-size 1 --use-fp8-w8a8 --use-cuda-graph
Then:
  mkdir -p bench_data/grouped_gemm/prefill/pro5000 bench_data/grouped_gemm/decode/pro5000
  mv groupedgemm_prefill.csv bench_data/grouped_gemm/prefill/pro5000/data.csv
  mv groupedgemm_decode.csv  bench_data/grouped_gemm/decode/pro5000/data.csv
"""
import argparse
import os
import sys

import pandas as pd
import torch
import triton

from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

try:  # import path drifted across sglang versions; 0.5.14 resolves the first
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe
except ImportError:
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe

parent_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(os.path.abspath(parent_dir))
from config.model_config import ModelConfig  # noqa E402

BLOCK = 128


def _round_up(x, m):
    return ((x + m - 1) // m) * m


def per_block_cast_to_fp8(x):
    # x: (N, K) -> fp8 (N,K), continuous fp32 scale (N/128, K/128). Block-FP8 [128,128].
    assert x.dim() == 2
    m, n = x.shape
    xp = torch.zeros((_round_up(m, BLOCK), _round_up(n, BLOCK)), dtype=x.dtype, device=x.device)
    xp[:m, :n] = x
    xv = xp.view(-1, BLOCK, xp.size(1) // BLOCK, BLOCK)
    amax = xv.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = amax / 448.0
    xs = (xv * (1.0 / sf)).to(torch.float8_e4m3fn)
    return xs.view_as(xp)[:m, :n].contiguous(), sf.view(xv.size(0), xv.size(2))


def build_experts(num_experts, N, K, use_fp8, dtype=torch.bfloat16):
    # use_fp8=True -> (w_fp8 [E,N,K], w_scale [E,N/128,K/128]); else -> (w_bf16 [E,N,K], None)
    if not use_fp8:
        return torch.randn(num_experts, N, K, dtype=dtype, device="cuda"), None
    w_fp8 = torch.empty(num_experts, N, K, dtype=torch.float8_e4m3fn, device="cuda")
    w_scale = torch.empty(num_experts, N // BLOCK, K // BLOCK, dtype=torch.float32, device="cuda")
    for e in range(num_experts):
        we = torch.randn(N, K, dtype=dtype, device="cuda")
        w_fp8[e], w_scale[e] = per_block_cast_to_fp8(we)
    return w_fp8, w_scale


def bench_moe(tokens, hidden, num_local_experts, topk, w1, w1s, w2, w2s,
              use_fp8, use_cuda_graph):
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    gating = torch.randn(tokens, num_local_experts, dtype=torch.float32, device="cuda")
    # routing done once, OUTSIDE the timed region (we time only the expert GEMMs)
    topk_output = select_experts(
        hidden_states=x,
        router_logits=gating,
        topk_config=TopKConfig(top_k=topk, renormalize=False),
    )

    moe_kwargs = dict(use_fp8_w8a8=use_fp8)  # inplace is set via MoeRunnerConfig; default is fine
    if use_fp8:
        moe_kwargs.update(w1_scale=w1s, w2_scale=w2s, block_shape=[BLOCK, BLOCK])

    def call():
        return fused_moe(x, w1, w2, topk_output, **moe_kwargs)

    for _ in range(5):
        call()
    torch.cuda.synchronize()

    if use_cuda_graph:
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    call()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                call()
            torch.cuda.synchronize()
            ms = triton.testing.do_bench(lambda: g.replay())
            return ms * 1e3
        except Exception as ex:
            print(f"   [warn] CUDA-graph capture failed ({type(ex).__name__}); timing directly")

    ms = triton.testing.do_bench(call)  # median ms, L2-flushed
    return ms * 1e3  # -> us


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-path", type=str, required=True)
    p.add_argument("--mode", choices=["prefill", "decode"], required=True)
    p.add_argument("--num-gpus", type=int, default=1, help="world size")
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--use-fp8-w8a8", action="store_true",
                   help="block-FP8 experts (matches the model). Off => bf16.")
    p.add_argument("--use-cuda-graph", action="store_true",
                   help="time via CUDA-graph replay (diagnostic; do NOT feed to simulator).")
    p.add_argument("--gpu-tflops", type=int, default=None,
                   help="peak TFLOPS for MFU. Default: 536 (fp8) / 274 (bf16). Must match "
                        "how main.py runs (fp8_tflops if --use-fp8-gemm, else fp16_tflops).")
    args = p.parse_args()

    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    peak = args.gpu_tflops if args.gpu_tflops is not None else (536 if args.use_fp8_w8a8 else 274)

    cfg = ModelConfig(args.config_path)
    ep_size = args.num_gpus // args.tp_size
    num_experts = cfg.num_routed_experts
    num_local_experts = num_experts // ep_size
    hidden = cfg.hidden_size
    shard_interm = cfg.intermediate_size // args.tp_size  # matches moe.py: intermediate // tp
    topk = cfg.num_experts_per_tok

    print(f"mode={args.mode} fp8={args.use_fp8_w8a8} cuda_graph={args.use_cuda_graph} "
          f"peak_tflops={peak}")
    print(f"experts={num_experts} local={num_local_experts} hidden={hidden} "
          f"shard_interm={shard_interm} topk={topk} ep_size={ep_size}")

    # Expert weights (built once, reused): w1=gate+up (2I,H), w2=down (H,I)
    w1, w1s = build_experts(num_local_experts, shard_interm * 2, hidden, args.use_fp8_w8a8)
    w2, w2s = build_experts(num_local_experts, hidden, shard_interm, args.use_fp8_w8a8)

    if args.mode == "prefill":
        token_sweep = [512, 1024, 2048, 4096, 8192, 16384,32768]
        col6_name = "seq_len_per_gpu"
    else:
        token_sweep = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        col6_name = "batch_size_per_gpu"

    rows = []
    for tokens in token_sweep:
        try:
            us = bench_moe(tokens, hidden, num_local_experts, topk,
                           w1, w1s, w2, w2s, args.use_fp8_w8a8, args.use_cuda_graph)
        except Exception as e:
            print(f" > tokens={tokens:6} FAILED: {type(e).__name__}: {str(e)[:160]}")
            continue
        # moe.py flop model: gemm_flops(1,H,I)*tokens*topk*3 = 2*H*I * tokens*topk*3
        moe_flops = 2 * hidden * shard_interm * tokens * topk * 3
        tflops = moe_flops / (us * 1e-6) / 1e12
        mfu = tflops / peak
        tokens_per_expert = round(tokens * topk / num_local_experts)
        print(f" > tokens={tokens:6} tok/expert={tokens_per_expert:5} | "
              f"{us:9.2f} us | {tflops:6.1f} TFLOPS | mfu={mfu:.4f}")
        rows.append({
            "num_experts": num_experts,
            "num_gpus": args.num_gpus,
            "num_local_experts": num_local_experts,
            "topk": topk,
            "hidden_size": hidden,
            "intermediate_size": shard_interm,
            col6_name: tokens,
            "tokens_per_expert": tokens_per_expert,
            "up_proj_us": round(us, 3),      # ignored by simulator (kept for readability)
            "up_mfu": round(mfu, 6),
            "down_proj_us": round(us, 3),    # ignored by simulator
            "down_mfu": round(mfu, 6),       # combined MFU in both; sim takes max()
        })

    out = f"groupedgemm_{args.mode}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out} ({len(rows)} rows). "
          f"mv to bench_data/grouped_gemm/{args.mode}/pro5000/data.csv")


if __name__ == "__main__":
    main()
