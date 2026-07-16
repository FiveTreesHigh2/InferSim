#!/usr/bin/env python3
"""
Columns (unchanged): m,k,n,latency_us,mfu
Run (sweep m for one (k,n) shape):
    python kernel_benchmark/flashinfer_gemm.py -k 2048 -n 9216 --gpu-tflops 536 \
        --output gemm_2048_9216.csv
"""
import argparse
import random

import pandas as pd
import torch

from flashinfer.gemm import gemm_fp8_nt_groupwise

def _round_up(x, m):
    return ((x + m - 1) // m) * m


def per_token_cast_to_fp8(x):
    # x: (m, k), k % 128 == 0 -> fp8 (m,k), continuous fp32 scale (m, k//128)  [K-major]
    # Continuous (NON-ue8m0) scales: matches original use_ue8m0=False and the
    # flashinfer cutlass test; sm120 cutlass block-scaled can't take e8m0fnu scales.
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    xv = x.view(m, -1, 128)
    amax = xv.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    sf = amax / 448.0
    return (xv * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), sf


def per_block_cast_to_fp8(x):
    # x: (n, k) -> fp8 (n,k), continuous fp32 scale (n//128, k//128)  [K-major]
    assert x.dim() == 2
    m, n = x.shape
    xp = torch.zeros((_round_up(m, 128), _round_up(n, 128)), dtype=x.dtype, device=x.device)
    xp[:m, :n] = x
    xv = xp.view(-1, 128, xp.size(1) // 128, 128)
    amax = xv.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = amax / 448.0
    xs = (xv * (1.0 / sf)).to(torch.float8_e4m3fn)
    return xs.view_as(xp)[:m, :n].contiguous(), sf.view(xv.size(0), xv.size(2))


def _event_fallback_us(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters  # ms -> us per iter


def _kernel_self_us(evt):
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(evt, attr, None)
        if v:
            return float(v)
    return 0.0


# Substrings that identify non-compute GPU ops to skip (esp. the L2-flush memset).
_DENY = ("memset", "memcpy", "fill", "zero", "elementwise", "copy", "reduce")
_PRINTED_KERNEL = False


def bench_us(fn, num_tests=30, flush_l2=True):
    global _PRINTED_KERNEL
    try:
        flush_n = int(8e9 // 4)
        fn()  # autotune / lazy-compile warmup
        sched = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            schedule=sched,
            acc_events=True,
        ) as prof:
            for _ in range(2):
                for _ in range(num_tests):
                    if flush_l2:
                        torch.empty(flush_n, dtype=torch.int, device="cuda").zero_()
                    fn()
                torch.cuda.synchronize()
                prof.step()
        best_key, best_total, best_count = None, -1.0, 1
        for e in prof.key_averages():
            name = e.key.lower()
            if any(d in name for d in _DENY):
                continue
            t = _kernel_self_us(e)
            if t > best_total:
                best_key, best_total, best_count = e.key, t, max(int(e.count), 1)
        if best_total > 0:
            if not _PRINTED_KERNEL:
                print(f"   [timing] measuring kernel: {best_key[:90]}")
                _PRINTED_KERNEL = True
            return best_total / best_count  # us per call, pure kernel time
    except Exception as ex:
        print(f"   [warn] profiler timing failed ({type(ex).__name__}); using CUDA-event fallback")
    return _event_fallback_us(fn)


def test_gemm(m, k, n):
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    a_fp8, a_scale = per_token_cast_to_fp8(a)          # (m,k), (m,k//128)
    b_fp8, b_scale = per_block_cast_to_fp8(b)          # (n,k), (n//128,k//128)

    # NT: out = a @ b.T ; groupwise 1x128 (A) / 128x128 (B), K-major scales
    def run():
        return gemm_fp8_nt_groupwise(
            a_fp8, b_fp8, a_scale, b_scale,
            scale_major_mode="K",
            scale_granularity_mnk=(1, 128, 128),
            out_dtype=torch.bfloat16,
        )

    out = run()                                        # correctness sanity (non-fatal)
    torch.cuda.synchronize()
    ref = (a.float() @ b.float().t()).to(torch.bfloat16)
    rel = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean().clamp(1e-6)
    if rel > 0.05:
        print(f"   [warn] m={m} k={k} n={n} rel_err={rel:.4f} (block-FP8 quant error; timing still valid)")

    t_us = bench_us(run)
    tflops = 2 * m * n * k / (t_us * 1e-6) / 1e12
    print(f" > m={m:6} k={k:5} n={n:5} | {t_us:8.2f} us | {tflops:6.0f} TFLOPS")
    return t_us, tflops


def main(args):
    results = []
    default_m_values = [
        1, 2, 4, 8, 16, 32, 64, 128, 224, 256, 512, 1024,
        4096, 8192, 16384, 32768, 64 * 1024, 128 * 1024,
    ]
    for m in args.m_values or default_m_values:
        try:
            t, tflops = test_gemm(m, args.k, args.n)
            results.append({
                "m": m, "k": args.k, "n": args.n,
                "latency_us": round(t, 3),
                # Small-batch GEMMs can have MFU below 0.001. Keep enough
                # precision to avoid writing 0.000 and dividing by zero later.
                "mfu": round(tflops / args.gpu_tflops, 6),
            })
        except Exception as e:
            print(f" > m={m:6} k={args.k} n={args.n} | FAILED: {type(e).__name__}: {str(e)[:160]}")
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nWrote {args.output} ({len(results)} rows).")


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=2048, help="[m,k] * [k,n]  (must be %128==0)")
    parser.add_argument("-n", type=int, default=2048, help="[m,k] * [k,n]  (must be %128==0)")
    parser.add_argument("--gpu-tflops", type=int, default=536, help="GPU FP8 peak TFLOPS (pro5000=536)")
    parser.add_argument(
        "--m-values", type=int, nargs="+", default=None,
        help="Optional m values to benchmark, for example: --m-values 1 2 4",
    )
    parser.add_argument("--output", default="gemm.csv", help="Output CSV path")
    args = parser.parse_args()
    assert args.k % 128 == 0 and args.n % 128 == 0, "k and n must be multiples of 128"
    main(args)
