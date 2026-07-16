import argparse

from config.model_config import ModelConfig
from models.hybrid_model import HybridModel
from models.model import Model


def mfu_value(value):
    value = float(value)
    if not 0 < value <= 1:
        raise argparse.ArgumentTypeError("MFU must be in the range (0, 1]")
    return value


def non_negative_float(value):
    value = float(value)
    if not 0 <= value < float("inf"):
        raise argparse.ArgumentTypeError("value must be finite and greater than or equal to 0")
    return value


def main(args):
    config = ModelConfig(args.config_path)

    enable_shared_expert_overlap = getattr(
        args, "enable_shared_expert_overlap", False
    )
    if enable_shared_expert_overlap and not (
        config.is_qwen3_5_moe and config.is_hybrid_linear
    ):
        raise ValueError(
            "--enable-shared-expert-overlap is only supported for the "
            "Qwen3.5 MoE HybridModel"
        )

    print("\n{s:{c}^{n}}".format(s=" Simulator Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Device type:", args.device_type))
    print("{:<40} {:<10}".format("World size:", args.world_size))
    print("{:<40} {:<10}".format("TP size:", args.tp_size))
    print("{:<40} {:<10}".format("Attn type:", config.attn_type))
    print("{:<40} {:<10}".format("Use FP8 GEMM:", args.use_fp8_gemm))
    print("{:<40} {:<10}".format("Use FP8 KV:", args.use_fp8_kv))
    if config.is_qwen3_5_moe and config.is_hybrid_linear:
        print(
            "{:<40} {:<10}".format(
                "Shared expert overlap:", enable_shared_expert_overlap
            )
        )

    if config.is_hybrid_linear:
        model = HybridModel(args, config)
        print("{:<40} {:<10}".format("Model type: ", "HybridModel"))
    else:
        model = Model(args, config)
    model.print_weights_info()
    model.print_kvcache_info()
    model.print_flops_info()

    if not args.decode_only:
        model.prefill()

    if not args.prefill_only:
        model.decoding()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        help="The path of the hf model config.json",
        required=True,
    )
    parser.add_argument(
        "--device-type",
        type=str,
        default="H20",
        choices=["H20", "H800", "H200", "GB200", "PRO5000"],
        help="Device type",
    )
    parser.add_argument("--world-size", type=int, default=1, help="Num of GPUs")
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size. If >1, both attention and MoE use TP; if =1, attention uses DP and MoE uses EP.",
    )
    parser.add_argument("--num-nodes", type=int, default=1, help="Num of nodes")
    parser.add_argument(
        "--max-prefill-tokens",
        type=int,
        default=4096,
        help="Max prefill tokens per GPU",
    )
    parser.add_argument(
        "--prefill-attn-mfu",
        type=mfu_value,
        default=None,
        help="Override prefill attention MFU. If omitted, read MFU from bench data.",
    )
    parser.add_argument(
        "--prefill-moe-mfu",
        type=mfu_value,
        default=None,
        help="Override prefill routed MoE/FFN MFU. If omitted, read MFU from bench data.",
    )
    parser.add_argument(
        "--decode-bs",
        type=int,
        help="Decoding batchsize per GPU. If not specified, bs = tgs * tpot.",
    )
    parser.add_argument(
        "--target-tgs", type=float, default=2560, help="Target tokens/s per GPU"
    )
    parser.add_argument("--target-tpot", type=float, default=50, help="TPOT in ms")
    parser.add_argument(
        "--target-isl", type=int, default=4096, help="Input sequence length, in tokens"
    )
    parser.add_argument(
        "--target-osl", type=int, default=2048, help="Output sequence length, in tokens"
    )
    parser.add_argument("--use-fp8-gemm", action="store_true", help="Use fp8 gemm")
    parser.add_argument("--use-fp8-kv", action="store_true", help="Use fp8 kvcache")
    parser.add_argument(
        "--decode-scheduler-overhead-ms",
        type=non_negative_float,
        default=None,
        help=(
            "Override decode scheduler overhead in milliseconds. If omitted, "
            "use the existing model default."
        ),
    )
    parser.add_argument(
        "--enable-shared-expert-overlap",
        action="store_true",
        help=(
            "Model Qwen3.5 MoE Decode shared and routed experts as parallel "
            "CUDA-stream paths. Only supported for the Qwen3.5 MoE HybridModel."
        ),
    )
    parser.add_argument("--enable-deepep", action="store_true", help="Enable DeepEP")
    parser.add_argument(
        "--enable-tbo", action="store_true", help="Enable two batch overlap"
    )
    parser.add_argument(
        "--sm-ratio",
        type=float,
        default=108 / 132,
        help="In TBO DeepEP normal mode, the SM ratio used for computation",
    )
    parser.add_argument(
        "--prefill-only", action="store_true", help="Only simulate prefill"
    )
    parser.add_argument(
        "--decode-only", action="store_true", help="Only simulate decoding"
    )
    args = parser.parse_args()
    main(args)
