"""Parameter role inference for heterogeneous model families.

Roles are *semantic hints* derived from tensor names and shapes. They are used by
alignment (to decide which tensors are even candidates for correspondence),
merging (to apply per-role policies), inspection, and provenance. A role is never
treated as proof of mathematical equivalence — it only narrows the search space.
"""

from __future__ import annotations

import re
from enum import Enum


class ParameterRole(str, Enum):
    EMBEDDING = "embedding"
    POSITIONAL_EMBEDDING = "positional_embedding"
    ATTENTION_QKV = "attention_qkv"
    ATTENTION_Q = "attention_q"
    ATTENTION_K = "attention_k"
    ATTENTION_V = "attention_v"
    ATTENTION_OUT = "attention_output"
    MLP_INPUT = "mlp_input"
    MLP_OUTPUT = "mlp_output"
    EXPERT_ROUTER = "expert_router"
    EXPERT_WEIGHT = "expert_weight"
    LAYERNORM = "layernorm"
    RMSNORM = "rmsnorm"
    CONV = "conv"
    DEPTHWISE_CONV = "depthwise_conv"
    RNN = "recurrent"
    LINEAR = "linear"
    BIAS = "bias"
    HEAD = "output_head"
    SCALAR = "scalar"
    UNCLASSIFIED = "unclassified"

    @property
    def is_matrix(self) -> bool:
        return self in {
            ParameterRole.EMBEDDING,
            ParameterRole.ATTENTION_QKV,
            ParameterRole.ATTENTION_Q,
            ParameterRole.ATTENTION_K,
            ParameterRole.ATTENTION_V,
            ParameterRole.ATTENTION_OUT,
            ParameterRole.MLP_INPUT,
            ParameterRole.MLP_OUTPUT,
            ParameterRole.EXPERT_ROUTER,
            ParameterRole.EXPERT_WEIGHT,
            ParameterRole.CONV,
            ParameterRole.RNN,
            ParameterRole.LINEAR,
            ParameterRole.HEAD,
        }

    @property
    def is_bias_like(self) -> bool:
        return self in {ParameterRole.BIAS, ParameterRole.LAYERNORM, ParameterRole.RMSNORM}


# Ordered rules: (regex, role). First match wins, so specific patterns precede
# generic ones.
_RULES: list[tuple[re.Pattern[str], ParameterRole]] = [
    (re.compile(r"(?:^|[._/])(?:pos(?:ition)?_?(?:emb|embed|embedding)|wpe|position_embeddings)"), ParameterRole.POSITIONAL_EMBEDDING),
    (re.compile(r"(?:^|[._/])(?:tok(?:en)?_?emb(?:ed(?:ding)?)?|wte|word_embeddings|embed_tokens|embedding)"), ParameterRole.EMBEDDING),
    (re.compile(r"(?:^|[._/])(?:lm_head|output_head|classifier|head_proj|final_logits)"), ParameterRole.HEAD),
    (re.compile(r"(?:^|[._/])(?:qkv|in_proj|c_attn|query_key_value|qkv_proj)"), ParameterRole.ATTENTION_QKV),
    (re.compile(r"(?:^|[._/])(?:q_proj|query|wq|q_lin)"), ParameterRole.ATTENTION_Q),
    (re.compile(r"(?:^|[._/])(?:k_proj|key|wk|k_lin)"), ParameterRole.ATTENTION_K),
    (re.compile(r"(?:^|[._/])(?:v_proj|value|wv|v_lin)"), ParameterRole.ATTENTION_V),
    (re.compile(r"(?:^|[._/])(?:o_proj|out_proj|dense|wo|attention\.output|attn_out)"), ParameterRole.ATTENTION_OUT),
    (re.compile(r"(?:^|[._/])(?:gate_proj|up_proj|fc1|fc_in|w1|w3|mlp_in|intermediate\.dense|ffn_in|input_proj)"), ParameterRole.MLP_INPUT),
    (re.compile(r"(?:^|[._/])(?:down_proj|fc2|fc_out|w2|mlp_out|output\.dense|ffn_out)"), ParameterRole.MLP_OUTPUT),
    (re.compile(r"(?:^|[._/])(?:router|gate|routing|moe_gate)"), ParameterRole.EXPERT_ROUTER),
    (re.compile(r"(?:^|[._/])(?:experts?\.|expert_\d|moe)"), ParameterRole.EXPERT_WEIGHT),
    (re.compile(r"(?:^|[._/])(?:rms_?norm|rmsnorm)"), ParameterRole.RMSNORM),
    (re.compile(r"(?:^|[._/])(?:layernorm|layer_norm|ln_?\d|ln_f|norm\d*|norm$)"), ParameterRole.LAYERNORM),
    (re.compile(r"(?:^|[._/])(?:depthwise|dw_conv)"), ParameterRole.DEPTHWISE_CONV),
    (re.compile(r"(?:^|[._/])(?:conv|convs?\d|downsample)"), ParameterRole.CONV),
    (re.compile(r"(?:^|[._/])(?:rnn|gru|lstm|recurrent|rnn_cell|hidden2hidden)"), ParameterRole.RNN),
    (re.compile(r"(?:^|[._/])(?:linear|dense|proj|matmul|weight$)"), ParameterRole.LINEAR),
]


def infer_role(name: str, shape: tuple[int, ...] | list[int]) -> ParameterRole:
    """Infer a parameter role from its name (primary) and shape (secondary)."""
    lowered = name.lower()
    dims = tuple(int(d) for d in shape)
    for pattern, role in _RULES:
        if pattern.search(lowered):
            if role.is_matrix and len(dims) < 2:
                continue
            if role in (ParameterRole.LAYERNORM, ParameterRole.RMSNORM) and len(dims) > 1:
                # e.g. "model.norm.weight" is 1-D, but a fused 2-D "norm" is odd;
                # treat 2-D norm-named tensors as linear projections.
                return ParameterRole.LINEAR
            return role
    if len(dims) == 1:
        return ParameterRole.BIAS
    if len(dims) == 0:
        return ParameterRole.SCALAR
    if len(dims) == 3:
        return ParameterRole.LINEAR
    if len(dims) == 4:
        return ParameterRole.CONV
    if len(dims) == 2:
        return ParameterRole.LINEAR
    return ParameterRole.UNCLASSIFIED


def canonical_stem(name: str) -> str:
    """Normalize a tensor name to a comparable stem (used for correspondence)."""
    stem = name.lower()
    stem = re.sub(r"^model\.", "", stem)
    stem = re.sub(r"\.\d+\.", ".N.", stem)
    stem = re.sub(r"\.\d+$", ".N", stem)
    stem = re.sub(r"(?:^|[._/])layers?\.N\.", "", stem)
    stem = re.sub(r"(?:^|[._/])h\.N\.", "", stem)
    return stem


def suffix_kind(name: str) -> str:
    tail = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return tail
