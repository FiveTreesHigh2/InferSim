# Adapt from https://github.com/sgl-project/sglang/blob/main/benchmark/kernels/decoding_attention_triton/triton_flashinfer_cudnn.py
import argparse
import os
import sys

import pandas as pd
import torch
import torch.utils.benchmark as benchmark

parent_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(os.path.abspath(parent_dir))

from config.model_config import ModelConfig  # noqa E402
from flops.flops import get_mha_gflops  # noqa E402


def benchmark_forward(
    fn,
    *inputs,
    repeats=10,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    return t, m


def time_fwd(func, *args, **kwargs):
    time_f = benchmark_forward(func, *args, **kwargs)
    return time_f[1].mean * 1e6

def prefill_attention_flashinfer():
    from flashinfer import single_prefill_with_kv_cache

    class FlashinferPrefill(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, num_q_heads, num_kv_heads, head_dim, warmup=10):
            seq_len = q.shape[0]
            scale = head_dim ** -0.5
            qi = q.reshape(seq_len, num_q_heads, head_dim)
            ki = k.reshape(seq_len, num_kv_heads, head_dim)
            vi = v.reshape(seq_len, num_kv_heads, head_dim)

            def run():
                return single_prefill_with_kv_cache(
                    qi, ki, vi, causal=True, sm_scale=scale
                )

            for _ in range(warmup):
                o = run()
            torch.cuda.synchronize()
            f = time_fwd(run)
            return f, o

    return FlashinferPrefill

def main(args):
    config = ModelConfig(args.config_path)
    fp16_tflops = 274
    head_dim = config.head_dim
    num_q_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    if num_q_heads == num_kv_heads:
        attn_type = "MHA"
    else:
        attn_type = "GQA"
    dtype = torch.bfloat16

    seq_lens = [1024, 4096, 8192, 16384, 32768]
    results = []

    attn_fa = prefill_attention_flashinfer().apply
    for seq_len in seq_lens:
        q = torch.randn(seq_len, num_q_heads, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device="cuda")

        attn_core_gflops, other_gflops = get_mha_gflops(config, 1, seq_len, 1)
        attn_core_gflops = attn_core_gflops * seq_len / 2

        us_fa, _ = attn_fa(q, k, v, num_q_heads, num_kv_heads, head_dim)
        mfu = attn_core_gflops * 1e3 / (fp16_tflops * us_fa)
        print(
            attn_type,
            "  ",
            num_q_heads,
            "  ",
            head_dim,
            "  ",
            seq_len,
            "  ",
            us_fa,
            "  ",
            mfu,
        )

        results.append(
            {
                "dtype": "bf16",
                "seq_len": seq_len,
                "latency_us": round(us_fa, 3),
                "mfu": round(mfu, 3),
            }
        )

    df = pd.DataFrame(results)
    df.to_csv("attention_benchmark.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        help="The path of the hf model config.json",
        required=True,
    )

    args = parser.parse_args()
    main(args)
