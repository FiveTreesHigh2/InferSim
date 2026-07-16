from config.model_config import ModelConfig
from hardware.gpu import GPU


def get_mha_params_size(config: ModelConfig, use_fp8: bool, tp_size: int):
    # TP shards heads; hidden_size is NOT sharded (row/col-parallel + allreduce)
    tp_num_heads = config.num_attention_heads // tp_size
    tp_num_kv_heads = config.num_key_value_heads // tp_size
    q_multiplier = 2 if config.attn_output_gate else 1
    wq = config.hidden_size * q_multiplier * tp_num_heads * config.head_dim
    wk = config.hidden_size * tp_num_kv_heads * config.head_dim
    wv = config.hidden_size * tp_num_kv_heads * config.head_dim
    wo = config.hidden_size * tp_num_heads * config.head_dim
    if use_fp8:
        return wq + wk + wv + wo
    return 2 * (wq + wk + wv + wo)


def get_mla_params_size(config: ModelConfig, use_fp8: bool, tp_size: int):
    # TP shards attention heads; hidden_size and lora ranks are NOT sharded
    tp_num_heads = config.num_attention_heads // tp_size
    wq_down = config.hidden_size * config.q_lora_rank
    wq_up = config.q_lora_rank * tp_num_heads * config.qk_head_dim
    wkv_down = config.hidden_size * config.kv_lora_rank
    wkv_up = (
        config.kv_lora_rank
        * tp_num_heads
        * (config.qk_nope_head_dim + config.v_head_dim)
    )
    wo = config.hidden_size * tp_num_heads * config.v_head_dim
    if use_fp8:
        return wq_down + wq_up + wkv_down + wkv_up + wo
    return 2 * (wq_down + wq_up + wkv_down + wkv_up + wo)


def get_gdn_params_size(config: ModelConfig, use_fp8: bool, tp_size: int):
    # NOTE: linear-attn head counts are currently NOT sharded in this model,
    # so tp_size has no effect here. Parameter kept for interface consistency.
    del tp_size
    wq = config.hidden_size * config.linear_num_key_heads * config.linear_key_head_dim
    wk = wq
    wv = (
        config.hidden_size
        * config.linear_num_value_heads
        * config.linear_value_head_dim
    )
    wz = wv
    wa = config.hidden_size * config.linear_num_value_heads
    wb = wa
    s = wq + wk + wv + wz + wa + wb
    wconv = (
        config.linear_num_key_heads
        * config.linear_key_head_dim
        * config.linear_conv_kernel_dim
    )
    wconv += (
        config.linear_num_key_heads
        * config.linear_key_head_dim
        * config.linear_conv_kernel_dim
    )
    wconv += (
        config.linear_num_value_heads
        * config.linear_value_head_dim
        * config.linear_conv_kernel_dim
    )
    if use_fp8:
        return s + wconv
    return 2 * s + wconv


def get_attn_params_size(config: ModelConfig, use_fp8: bool, tp_size: int):
    if config.attn_type == "MHA/GQA":
        return get_mha_params_size(config, use_fp8, tp_size)
    elif config.attn_type == "MLA":
        return get_mla_params_size(config, use_fp8, tp_size)


def get_linear_attn_params_size(config: ModelConfig, use_fp8: bool, tp_size: int):
    return get_gdn_params_size(config, use_fp8, tp_size)


def get_expert_params_size(config: ModelConfig, use_fp8: bool, tp_size: int):
    # TP shards intermediate_size; hidden_size is NOT sharded
    tp_intermediate_size = config.intermediate_size // tp_size
    w = 3 * config.hidden_size * tp_intermediate_size
    if not use_fp8:
        w *= 2
    return w


def load_attn_weights_time(config: ModelConfig, use_fp8: bool, gpu: GPU, tp_size: int):
    size = get_attn_params_size(config, use_fp8, tp_size)
    return size / 1024 / 1024 / 1024 / gpu.mem_bw


def get_expected_active_experts(
    config: ModelConfig, num_gpus: int, tp_size: int, num_tokens=None
):
    """Estimate local routed experts touched by a uniformly routed batch."""
    ep_size = num_gpus // tp_size
    num_local_experts = config.num_routed_experts / ep_size
    if num_tokens is None:
        return num_local_experts

    topk = min(config.num_experts_per_tok, num_local_experts)
    probability_not_selected = (1 - topk / num_local_experts) ** num_tokens
    return num_local_experts * (1 - probability_not_selected)


def load_moe_weights_time(
    config: ModelConfig,
    use_fp8: bool,
    gpu: GPU,
    num_gpus,
    tp_size=1,
    num_tokens=None,
):
    size = get_expert_params_size(config, use_fp8, tp_size)
    size *= get_expected_active_experts(config, num_gpus, tp_size, num_tokens)
    return size / 1024 / 1024 / 1024 / gpu.mem_bw
