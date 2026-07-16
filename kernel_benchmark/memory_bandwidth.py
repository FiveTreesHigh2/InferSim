#!/usr/bin/env python3
"""Measure sustained PRO5000 device-memory bandwidth with PyTorch CUDA kernels.

The benchmark reports two common bandwidth measurements:

copy_: device-to-device copy-engine bandwidth.
add_out: an SM kernel that reads one large tensor and writes one large
  tensor.  This is generally the more relevant upper bound for kernels such as
  attention, although real paged-KV access can achieve less.

Both measurements count one source read plus one destination write, following
the STREAM-copy convention.  Use a working set much larger than L2 cache.

Example:
    python3 kernel_benchmark/memory_bandwidth.py   --size-gib 2   \
        --warmup 10   --iterations 30   --rounds 7   \
        --peak-gbps 1345   --kv-batch-size 8   --kv-len 64512   --kv-heads 2   --head-dim 256   --kv-dtype-bytes 2
"""

import argparse
import statistics

import torch


GIB = 1024**3


def time_cuda(fn, warmup: int, iterations: int, rounds: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end) / iterations)
    return times_ms


def summarize(name: str, times_ms: list[float], traffic_bytes: int, peak_gbps: float) -> float:
    bandwidths = [traffic_bytes / (t / 1e3) / 1e9 for t in times_ms]
    median_gbps = statistics.median(bandwidths)
    print(f"\n{name}")
    print(f"  median latency:       {statistics.median(times_ms):.3f} ms")
    print(f"  median bandwidth:     {median_gbps:.1f} GB/s")
    print(f"  min/max bandwidth:    {min(bandwidths):.1f} / {max(bandwidths):.1f} GB/s")
    print(f"  fraction of peak:     {median_gbps / peak_gbps:.3f}")
    return median_gbps


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)

    dtype = torch.float32
    element_size = torch.empty((), dtype=dtype).element_size()
    numel = int(args.size_gib * GIB) // element_size
    tensor_bytes = numel * element_size
    traffic_bytes = tensor_bytes * 2  # one read + one write

    print(f"GPU:                    {props.name}")
    print(f"CUDA device:            {device}")
    print(f"Total GPU memory:        {props.total_memory / GIB:.2f} GiB")
    print(f"Tensor size:             {tensor_bytes / GIB:.2f} GiB")
    print(f"Traffic per iteration:   {traffic_bytes / GIB:.2f} GiB (read + write)")
    print(f"Configured peak:         {args.peak_gbps:.1f} GB/s")
    print(f"Simulator 0.8 target:    {args.peak_gbps * 0.8:.1f} GB/s")

    src = torch.empty(numel, dtype=dtype, device=device)
    dst = torch.empty_like(src)
    src.fill_(1.0)
    dst.zero_()
    torch.cuda.synchronize()

    copy_times = time_cuda(
        lambda: dst.copy_(src), args.warmup, args.iterations, args.rounds
    )
    copy_gbps = summarize(
        "D2D copy_ (copy engine)", copy_times, traffic_bytes, args.peak_gbps
    )

    add_times = time_cuda(
        lambda: torch.add(src, 1.0, out=dst),
        args.warmup,
        args.iterations,
        args.rounds,
    )
    add_gbps = summarize(
        "add_out (SM global load + store)", add_times, traffic_bytes, args.peak_gbps
    )

    kv_bytes = (
        args.kv_batch_size
        * args.kv_len
        * 2  # K and V
        * args.kv_heads
        * args.head_dim
        * args.kv_dtype_bytes
    )
    print("\nDecode KV reference (one Full Attention layer)")
    print(f"  shape parameters:      bs={args.kv_batch_size}, kv_len={args.kv_len}, "
          f"kv_heads={args.kv_heads}, head_dim={args.head_dim}")
    print(f"  K+V bytes:             {kv_bytes / 1e9:.3f} GB ({kv_bytes / GIB:.3f} GiB)")
    print(f"  time at peak:          {kv_bytes / (args.peak_gbps * 1e9) * 1e6:.1f} us")
    print(f"  time at peak * 0.8:    {kv_bytes / (args.peak_gbps * 0.8 * 1e9) * 1e6:.1f} us")
    print(f"  time at copy result:   {kv_bytes / (copy_gbps * 1e9) * 1e6:.1f} us")
    print(f"  time at SM result:     {kv_bytes / (add_gbps * 1e9) * 1e6:.1f} us")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--size-gib",
        type=float,
        default=2.0,
        help="Size of each source/destination tensor. Keep this much larger than L2.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument(
        "--peak-gbps",
        type=float,
        default=1345.0,
        help="Decimal GB/s from the GPU specification.",
    )
    parser.add_argument("--kv-batch-size", type=int, default=8)
    parser.add_argument("--kv-len", type=int, default=64512)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--kv-dtype-bytes", type=int, default=2)
    main(parser.parse_args())
