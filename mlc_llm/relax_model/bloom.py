import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import tvm
from tvm import relax, te
from tvm.relax.testing import nn
from tvm.script import relax as R

from ..quantization import ParamQuantKind, QuantizationScheme
from .commons import create_metadata_func
from .modules import ModuleList
from .param_manager import ParamManager


@dataclass
class BloomConfig:
    """
    Description of transfomers.models.bloom.configuration_bloom
    ----
    This is the configuration class to store the configuration of a [`BloomModel`]. It is used to instantiate a Bloom
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to the Bloom architecture
    [bigscience/bloom](https://huggingface.co/bigscience/bloom).
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 250880):
            Vocabulary size of the Bloom model. Defines the maximum number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`BloomModel`]. Check [this
            discussion](https://huggingface.co/bigscience/bloom/discussions/120#633d28389addb8530b406c2a) on how the
            `vocab_size` has been defined.
        hidden_size (`int`, *optional*, defaults to 64):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 2):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        apply_residual_connection_post_layernorm (`bool`, *optional*, defaults to `False`):
            If enabled, use the layer norm of the hidden states as the residual in the transformer blocks
        hidden_dropout (`float`, *optional*, defaults to 0.1):
            Dropout rate of the dropout function on the bias dropout.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            Dropout rate applied to the attention probs
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        pretraining_tp (`int`, *optional*, defaults to `1`):
            Experimental feature. Tensor parallelism rank used during pretraining with Megatron. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232). Note also that this is enabled only when
            `slow_but_exact=True`.
        slow_but_exact (`bool`, *optional*, defaults to `False`):
            Experimental feature. Whether to use slow but exact implementation of the attention mechanism. While
            merging the TP rank tensors, due to slicing operations the results may be slightly different between the
            model trained on Megatron and our model. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232). A solution to obtain more accurate results is to
            enable this feature. Enabling this will hurt the computational time of the inference. Will be probably
            resolved in the future once the main model has been fine-tuned with TP_rank=1.
    Example:
    ```python
    >>> from transformers import BloomConfig, BloomModel
    >>> # Initializing a Bloom configuration
    >>> configuration = BloomConfig()
    >>> # Initializing a model (with random weights) from the configuration
    >>> model = BloomModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "bloom"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_hidden_layers": "n_layer",
        "num_attention_heads": "n_head",
    }

    def __init__(
        self,
        vocab_size=250880,
        hidden_size=64,
        n_layer=2,
        n_head=8,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        unk_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
        apply_residual_connection_post_layernorm=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        pretraining_tp=1,  # TP rank used when training with megatron
        slow_but_exact=False,
        dtype = 'float32',
        **kwargs,
    ):
        self.vocab_size = vocab_size
        # Backward compatibility with n_embed kwarg
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size = hidden_size if n_embed is None else n_embed
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.pretraining_tp = pretraining_tp
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.unk_token_id = unk_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.slow_but_exact = slow_but_exact
        self.dtype = dtype
        for k,v in BloomConfig.attribute_map.items():
            if k in kwargs:
                setattr(self, v, kwargs[k])

class LayerNorm(nn.Module):
    def __init__(
        self,
        hidden_size,
        dtype,
        eps=1e-5,
    ):
        super().__init__()
        self.dtype = dtype
        self.eps = eps
        self.weight = nn.Parameter((hidden_size,), dtype=self.dtype, name="weight")
        self.bias = nn.Parameter((hidden_size,), dtype=self.dtype, name="bias")

    def forward(self, x: relax.Expr) -> relax.Var:
        from tvm.relax.op.nn import layer_norm
        if x.struct_info.dtype != self.dtype:
            x = nn.emit(relax.op.astype(x, self.dtype))
        x = nn.emit(
            layer_norm(
                x,
                gamma=self.weight,
                beta=self.bias,
                axes=-1,
                epsilon=self.eps,
            )
        )
        return x


class Linear(nn.Module):
    def __init__(self, in_features, out_features, dtype: str, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            (out_features, in_features), dtype=dtype, name="linear_weight"
        )
        if bias:
            self.bias = nn.Parameter((out_features,), dtype=dtype, name="linear_bias")
        else:
            self.bias = None

    def forward(self, input: relax.Expr) -> relax.Var:
        return nn.emit(relax.op.linear(input, self.weight, self.bias))


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dtype: str):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            (num_embeddings, embedding_dim), dtype=dtype, name="embedding_weight"
        )

    def forward(self, x: relax.Expr) -> relax.Var:
        from tvm.relax.op import reshape, take

        ndim = x.struct_info.ndim
        if ndim == 1:
            return nn.emit(take(self.weight, x, axis=0))
        else:
            x_shape = x.struct_info.shape.values
            emb_size = self.weight.struct_info.shape.values[-1]
            x = nn.emit(reshape(x, shape=[-1]))
            embedding = nn.emit(take(self.weight, x, axis=0))
            return nn.emit(reshape(embedding, [*x_shape, emb_size]))

class BloomGelu(nn.Module):
    r"""BloomGelu is same with the relax.op.gelu_tanh

    Compare
    ----
        relax.op.gelu_tanh formula: 
            .. math::
                \text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt(2 / \pi) * (x + 0.044715 * x^3)))
        transformers.models.bloom.modeling_bloom.BloomGelu formula:
            .. math::
                \text{GELU}(x) = x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))
    
    """
    def __init__(self):
        pass

    def forward(self, x: relax.Expr):
        from tvm.relax.op.nn import gelu_tanh
        assert x.struct_info.dtype == 'float32', 'relax.op.nn.gelu_tanh requires float input tensor'
        return nn.emit(gelu_tanh(x))

class BloomMLP(nn.Module):
    def __init__(self, hidden_size: int, dtype: str):
        self.dtype = dtype
        self.dense_h_to_4h = Linear(hidden_size, 4 * hidden_size, dtype=dtype)
        self.gelu_impl = BloomGelu()
        self.dense_4h_to_h = Linear(4 * hidden_size, hidden_size, dtype=dtype)

    def forward(self, x: relax.Expr):
        from tvm.relax.op import astype
        ln1_tensor = self.dense_h_to_4h(x)
        if ln1_tensor.struct_info.dtype != 'float32':
            ln1_tensor = nn.emit(astype(ln1_tensor, 'float32'))
        gelu_tensor = self.gelu_impl(ln1_tensor)
        if gelu_tensor.struct_info.dtype != self.dtype:
            gelu_tensor = nn.emit(astype(gelu_tensor, self.dtype))
        hidden_states = self.dense_4h_to_h(gelu_tensor)
        # hidden_states = self.dense_4h_to_h(self.gelu_impl(self.dense_h_to_4h(x)))

        return hidden_states

def build_alibi_tensor(batch_size, seq_length, num_heads, dtype="float32"):
    """
    tvm implementation of transformers.model.bloom.modeling_bloom.build_alibi_tensor

    Description of transformers.bloom
    ----
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.
    Args:
    Returns tensor shaped (batch_size, num_heads, 1, max_seq_len)
        attention_mask (`torch.Tensor`):
            Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
        num_heads (`int`, *required*):
            number of heads
        dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):
            dtype of the output tensor
    """
    import math
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
    powers = nn.emit(relax.op.arange(1, 1 + closest_power_of_2, dtype=dtype))
    slopes = nn.emit(relax.op.power(relax.const(base, dtype=dtype), powers))

    if closest_power_of_2 != num_heads:
        extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = nn.emit(relax.op.arange(1, 1 + 2 * num_remaining_heads, 2, dtype=dtype))
        slopes = nn.emit(relax.op.concat([slopes, relax.op.power(relax.const(extra_base, dtype=dtype), extra_powers)], axis=0))

    cumsum_mask = nn.emit(relax.op.reshape(relax.op.repeat(relax.op.arange(0, seq_length, dtype=dtype), repeats=batch_size.value, axis=0) , (batch_size, seq_length)))
    arange_tensor = nn.emit(relax.op.expand_dims(cumsum_mask, axis=1))

    slope_tensor = nn.emit(relax.op.expand_dims(slopes, axis=-1))
    alibi = nn.emit(relax.op.astype(relax.op.multiply(slope_tensor, arange_tensor), dtype))
    alibi = nn.emit(relax.op.reshape(alibi, (batch_size, num_heads, 1, seq_length)))

    return alibi

class BloomAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: BloomConfig):
        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0
        # 3 * self.hidden_size -> [num_heads * 3 * head_dim]
        # linear output: [bsz, seq_length, num_heads * 3 * head_dim]
        self.query_key_value = Linear(self.hidden_size, 3 * self.hidden_size, dtype = config.dtype, bias=True)

        # same with o_proj in llama
        self.dense = Linear(self.hidden_size, self.hidden_size, dtype = config.dtype)

    def _split_heads(self, fused_qkv: relax.Expr) -> Tuple[relax.Expr, relax.Expr, relax.Expr]:
        """
        Split the last dimension into (num_heads, head_dim)
        Args:
            fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]
        Returns:
            query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """

        from tvm.relax.op import reshape
        bsz, seq_length, three_times_hidden_size = fused_qkv.struct_info.shape
        reshape_qkv = nn.emit(reshape(fused_qkv, (bsz, seq_length, self.num_heads, 3, self.head_dim)))
        def slice_qkv_te(x:te.Tensor, indice):
            return te.compute((bsz, seq_length, self.num_heads, self.head_dim), lambda b,s,n,d: x[b, s, n, indice, d], name = "slice_qkv")
        q_tensor = nn.emit_te(slice_qkv_te, reshape_qkv, 0, primfunc_name_hint="slice_q")
        k_tensor = nn.emit_te(slice_qkv_te, reshape_qkv, 1, primfunc_name_hint="slice_k")
        v_tensor = nn.emit_te(slice_qkv_te, reshape_qkv, 2, primfunc_name_hint="slice_v")

        return q_tensor, k_tensor, v_tensor


    def forward(
        self,
        hidden_states: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_value: Tuple[relax.Expr],
        alibi: relax.Expr,
        attention_mask: Optional[relax.Expr] = None,
    ) -> Tuple[relax.Expr, Optional[relax.Expr], Optional[Tuple[relax.Expr]]]:
        from tvm.relax.op import astype, matmul, maximum, permute_dims, reshape, squeeze, add
        from tvm.relax.op.nn import softmax

        bsz, q_len, _ = hidden_states.struct_info.shape
        assert bsz == 1, "Only support batch size 1 at this moment."

        # [batch_size, seq_length, num_heads * 3 * head_dim]
        fused_qkv = self.query_key_value(hidden_states)
        ## [batch_size, seq_len, num_heads, head_dim] 
        query_states, key_states, value_states = self._split_heads(fused_qkv)

        kv_seq_len = all_seq_len_shape.struct_info.values[0]

        # [bsz, seq_length, num_heads, head_dim]
        kv_states_shape = key_states.struct_info.shape
        kv_states_dtype = key_states.struct_info.dtype
        assert kv_states_shape[0] == 1  # bsz
        # [bsz, kv_seq_len, num_heads, head_dim]
        kv_states_shape = R.shape(
            [kv_states_shape[0], kv_seq_len, kv_states_shape[2], kv_states_shape[3]]
        )
        # [kv_seq_len, num_heads, head_dim]
        kv_cache_shape = R.shape([kv_seq_len, kv_states_shape[2], kv_states_shape[3]])

        # [seq_length, num_heads, head_dim]
        squeezed_key = nn.emit(squeeze(key_states, axis=0))
        # [seq_length, num_heads, head_dim]
        squeezed_value = nn.emit(squeeze(value_states, axis=0))

        k_cache, v_cache = past_key_value
        f_kv_cache_append = relax.extern("vm.builtin.attention_kv_cache_append")
        k_cache = nn.emit(
            relax.Call(
                f_kv_cache_append,
                args=[k_cache, squeezed_key],
                sinfo_args=[relax.ObjectStructInfo()],
            )
        )
        v_cache = nn.emit(
            relax.Call(
                f_kv_cache_append,
                args=[v_cache, squeezed_value],
                sinfo_args=[relax.ObjectStructInfo()],
            )
        )
        past_key_value = (k_cache, v_cache)
        f_kv_cache_view = relax.extern("vm.builtin.attention_kv_cache_view")
        k_cache = nn.emit(
            relax.Call(
                f_kv_cache_view,
                args=[k_cache, kv_cache_shape],
                sinfo_args=[R.Tensor(kv_cache_shape, kv_states_dtype)],
            )
        )
        v_cache = nn.emit(
            relax.Call(
                f_kv_cache_view,
                args=[v_cache, kv_cache_shape],
                sinfo_args=[R.Tensor(kv_cache_shape, kv_states_dtype)],
            )
        )
        # [bsz, kv_seq_len, num_heads, head_dim]
        key_states = nn.emit(reshape(k_cache, kv_states_shape))
        # [bsz, kv_seq_len, num_heads, head_dim]
        value_states = nn.emit(reshape(v_cache, kv_states_shape))

        # [bsz, seq_len, num_heads, head_dim] -> [bsz, num_heads, seq_len, head_dim]
        query_states = nn.emit(permute_dims(query_states, [0, 2, 1, 3]))
        # [bsz, kv_seq_len, num_heads, head_dim] -> [bsz, num_heads, kv_seq_len, head_dim]
        key_states = nn.emit(permute_dims(key_states, [0, 2, 1, 3]))
        # [bsz, kv_seq_len, num_heads, head_dim] -> [bsz, num_heads, kv_seq_len, head_dim]
        value_states = nn.emit(permute_dims(value_states, [0, 2, 1, 3]))

        attn_weights = nn.emit(
            # matmul(query, key): [bsz, num_heads, seq_len, head_dim] @ [bsz, num_heads, head_dim, kv_seq_len]
            # result.shape = [bsz, num_heads, seq_len, kv_seq_len]
            add((matmul(query_states, permute_dims(key_states, [0, 1, 3, 2]))), alibi)
            / relax.const(math.sqrt(self.head_dim), query_states.struct_info.dtype)
        )
        # print(f"attn_weights.shape = {attn_weights.struct_info.shape}")
        tvm.ir.assert_structural_equal(
            attn_weights.struct_info.shape.values,
            (bsz, tvm.tir.IntImm("int64", self.num_heads), q_len, kv_seq_len),
        )
        tvm.ir.assert_structural_equal(
            attention_mask.struct_info.shape.values,
            (bsz, tvm.tir.IntImm("int64", 1), q_len, kv_seq_len),
        )

        attn_weights = nn.emit(
            maximum(
                attn_weights,
                relax.const(
                    tvm.tir.min_value(attn_weights.struct_info.dtype).value,
                    attn_weights.struct_info.dtype,
                ),
            )
        )
        attn_weights = nn.emit(relax.op.minimum(attn_weights, attention_mask))

        # upcast attention to fp32
        if attn_weights.struct_info.dtype != "float32":
            attn_weights = astype(attn_weights, "float32")
        attn_weights = nn.emit(softmax(attn_weights, axis=-1))
        if attn_weights.struct_info.dtype != query_states.struct_info.dtype:
            attn_weights = astype(attn_weights, query_states.struct_info.dtype)
        # [bsz, num_heads, seq_len, head_dim]
        attn_output = nn.emit(matmul(attn_weights, value_states))

        tvm.ir.assert_structural_equal(
            attn_output.struct_info.shape.values,
            (
                bsz,
                tvm.tir.IntImm("int64", self.num_heads),
                q_len,
                tvm.tir.IntImm("int64", self.head_dim),
            ),
        )

        # [bsz, seq_len, num_heads, head_dim]
        attn_output = permute_dims(attn_output, [0, 2, 1, 3])
        # same to transform.bloom_model._merge_heads() function
        # [bsz, seq_len, num_heads * head_dim]
        attn_output = reshape(attn_output, (bsz, q_len, self.hidden_size))

        attn_output = self.dense(attn_output)
        return attn_output, ((None, None) if past_key_value is None else past_key_value)


class BloomBlock(nn.Module):
    def __init__(self, config: BloomConfig):
        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head

        self.input_layernorm = LayerNorm(self.hidden_size, dtype=config.dtype, eps=config.layer_norm_epsilon)
        self.self_attention = BloomAttention(config)
        self.post_attention_layernorm = LayerNorm(self.hidden_size, dtype=config.dtype, eps=config.layer_norm_epsilon)

        self.mlp = BloomMLP(
            hidden_size=self.hidden_size,
            dtype=config.dtype,
        )

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm

    def forward(
        self,
        hidden_states: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_value: Tuple[relax.Expr],
        alibi: relax.Expr,
        attention_mask: Optional[relax.Expr] = None,
    ) -> Tuple[relax.Expr, Optional[Tuple[relax.Expr, relax.Expr]]]:
        # hidden_states: [batch_size, seq_length, hidden_size]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Layer norm post the self attention.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # Self Attention
        hidden_states, present_key_value = self.self_attention(
            hidden_states=layernorm_output,
            past_key_value=past_key_value,
            alibi=alibi,
            attention_mask=attention_mask,
            all_seq_len_shape=all_seq_len_shape,
        )

        layernorm_output = self.post_attention_layernorm(hidden_states)

        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        hidden_states = self.mlp(layernorm_output)

        hidden_states = nn.emit(residual + hidden_states)

        return hidden_states, present_key_value


def _make_causal_mask(input_ids_shape, dtype, src_len):
    from tvm.relax.op import broadcast_to, full, triu

    bsz, tgt_len = input_ids_shape

    def min_max_triu_te():
        return te.compute(
            (tgt_len, tgt_len),
            lambda i, j: tvm.tir.Select(
                j > i, tvm.tir.min_value(dtype), tvm.tir.max_value(dtype)
            ),
            name="make_diag_mask_te",
        )

    mask = nn.emit_te(min_max_triu_te)
    diag_mask = nn.emit(broadcast_to(mask, (bsz, 1, tgt_len, tgt_len)))
    if src_len == tgt_len:
        return diag_mask

    def extend_te(x, tgt_len, src_len):
        return te.compute(
            (bsz, 1, tgt_len, src_len),
            lambda b, _, i, j: te.if_then_else(
                j < src_len - tgt_len,
                tvm.tir.max_value(dtype),
                x[b, _, i, j - (src_len - tgt_len)],
            ),
            name="concat_te",
        )

    return nn.emit_te(extend_te, diag_mask, tgt_len, src_len)


class BloomEmbedTokens(nn.Module):
    def __init__(self, config: BloomConfig):
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, dtype=config.dtype
        )

    def forward(self, input_ids: relax.Expr):
        inputs_embeds = self.embed_tokens(input_ids)
        return inputs_embeds


class BloomEmbedTokensWrapper(nn.Module):
    def __init__(self, config: BloomConfig):
        # build a wrapper to ensure that the naming of the embed_tokens parameter is consistent
        self.model = BloomEmbedTokens(config)

    def forward(self, input_ids: relax.Expr):
        inputs_embeds = self.model(input_ids)
        return inputs_embeds


class BloomModel(nn.Module):
    def __init__(self, config: BloomConfig, sep_embed: bool = False):
        # TODO: no padding_idx in config.json
        self.padding_idx = 3
        self.vocab_size = config.vocab_size
        self.embed_tokens = None
        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head
        self.dtype = config.dtype

        if not sep_embed:
            self.word_embeddings = Embedding(
                config.vocab_size, config.hidden_size, dtype=config.dtype
            )
        self.word_embeddings_layernorm = LayerNorm(self.embed_dim, dtype=config.dtype, eps=config.layer_norm_epsilon)
        self.h = ModuleList(
            [BloomBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = LayerNorm(self.embed_dim, dtype=config.dtype, eps=config.layer_norm_epsilon)

    def _prepare_decoder_attention_mask(self, input_shape, src_len, dtype):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if isinstance(input_shape[-1], tvm.tir.Var) or input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(input_shape, dtype, src_len)
        else:
            # Get src_len from input parameters
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            bsz, tgt_len = input_shape
            combined_attention_mask = nn.emit(
                relax.op.full(
                    (bsz, 1, tgt_len, src_len),
                    relax.const(tvm.tir.max_value(dtype).value, dtype),
                    dtype,
                )
            )
        return combined_attention_mask

    def forward(
        self,
        inputs: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_values: relax.Expr,
    ):
        if self.word_embeddings:
            inputs_embeds = self.word_embeddings(inputs)
        else:
            inputs_embeds = inputs
        # layer_norm inputs
        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        # retrieve input_ids
        batch_size, seq_length, _ = inputs_embeds.struct_info.shape
        # ref to !https://github.com/mlc-ai/mlc-llm/blob/main/cpp/llm_chat.cc#L556
        # seq_length_with_past = kv_seq_length + seq_length
        seq_length_with_past = all_seq_len_shape.struct_info.values[0]

        # embed positions
        # same to transformers.models.bloom.modeling_bloom._prepare_attn_mask
        attention_mask = self._prepare_decoder_attention_mask(
            (batch_size, seq_length),
            seq_length_with_past,
            inputs_embeds.struct_info.dtype,
        )

        # build alibit tensor
        alibi = build_alibi_tensor(batch_size, seq_length_with_past, self.num_heads, dtype = self.dtype)

        # decoder layers
        next_decoder_cache = ()

        for idx, decoder_layer in enumerate(self.h):
            assert past_key_values is not None
            past_key_value = (past_key_values[idx * 2], past_key_values[idx * 2 + 1])

            hidden_states, key_value_cache = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                all_seq_len_shape=all_seq_len_shape,
                alibi=alibi,
            )
            next_decoder_cache += key_value_cache

        hidden_states = self.ln_f(hidden_states)

        assert len(next_decoder_cache) == len(self.h) * 2
        return hidden_states, next_decoder_cache


class BloomForCausalLM(nn.Module):
    def __init__(self, config: BloomConfig, sep_embed: bool = False):
        self.transformer = BloomModel(config, sep_embed)
        self.lm_head = Linear(
            config.hidden_size, config.vocab_size, dtype=config.dtype, bias=False
        )

    def forward(
        self,
        inputs: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_values: relax.Expr,
    ):
        # hidden_states: [bsz, seq_len, hidden_size]
        hidden_states, key_value_cache = self.transformer(
            inputs=inputs,
            all_seq_len_shape=all_seq_len_shape,
            past_key_values=past_key_values,
        )

        def te_slicing(x: te.Tensor):
            return te.compute(
                shape=(1, 1, x.shape[-1]),
                fcompute=lambda i, j, k: x[i, x.shape[1] - 1, k],
                name="slice",
            )

        logits = self.lm_head(
            nn.emit_te(te_slicing, hidden_states, primfunc_name_hint="slice")
        )
        if logits.struct_info.dtype != "float32":
            logits = nn.emit(relax.op.astype(logits, "float32"))

        return logits, key_value_cache


def get_param_quant_kind(
    name: str, param_info: relax.TensorStructInfo
) -> ParamQuantKind:
    if "word_embeddings.weight" in name:
        return ParamQuantKind.embedding_table
    elif "lm_head.weight" in name:
        return ParamQuantKind.final_fc_weight
    elif param_info.ndim == 2 and name.endswith(".weight"):
        return ParamQuantKind.linear_weight
    else:
        return ParamQuantKind.others


def create_embed_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: BloomConfig,
    quant_scheme: QuantizationScheme,
) -> None:
    func_name = "embed"

    bsz = 1
    seq_len = tvm.tir.Var("n", "int64")
    with bb.function(func_name):
        model = BloomEmbedTokensWrapper(config)
        param_manager.register_params(
            model, func_name, quant_scheme, get_param_quant_kind
        )

        input_ids = nn.Placeholder((bsz, seq_len), dtype="int32", name="input_ids")
        with bb.dataflow():
            inputs_embeds = model(input_ids)
            params = [input_ids] + model.parameters()
            gv = bb.emit_output(inputs_embeds)
        bb.emit_func_output(gv, params)

    mod = bb.get()
    gv = mod.get_global_var(func_name)
    bb.update_func(gv, mod[gv].with_attr("num_input", 1))


def create_encoding_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: BloomConfig,
    quant_scheme: QuantizationScheme,
    sep_embed: bool = False,
) -> None:
    func_name = "prefill_with_embed" if sep_embed else "prefill"

    bsz = 1
    seq_len = tvm.tir.Var("n", "int64")
    all_seq_len = tvm.tir.Var("m", "int64")
    hidden_size = config.hidden_size
    with bb.function(func_name):
        model = BloomForCausalLM(config, sep_embed)
        param_manager.register_params(
            model, func_name, quant_scheme, get_param_quant_kind
        )

        inputs = (
            nn.Placeholder(
                (bsz, seq_len, hidden_size), dtype=config.dtype, name="inputs_embeds"
            )
            if sep_embed
            else nn.Placeholder((bsz, seq_len), dtype="int32", name="input_ids")
        )
        all_seq_len_shape = relax.Var(
            "all_seq_len", relax.ShapeStructInfo((all_seq_len,))
        )
        past_key_values = relax.Var(
            "kv_cache",
            relax.TupleStructInfo(
                [relax.ObjectStructInfo() for _ in range(config.n_layer * 2)]
            ),
        )
        with bb.dataflow():
            logits, key_value_cache = model(
                inputs, all_seq_len_shape, past_key_values=past_key_values
            )
            params = [
                inputs,
                all_seq_len_shape,
                past_key_values,
            ] + model.parameters()
            gv = bb.emit_output((logits, relax.Tuple(key_value_cache)))
        bb.emit_func_output(gv, params)

    mod = bb.get()
    gv = mod.get_global_var(func_name)
    bb.update_func(gv, mod[gv].with_attr("num_input", 3))


def create_decoding_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: BloomConfig,
    quant_scheme: QuantizationScheme,
) -> None:
    func_name = "decode"

    bsz = 1
    all_seq_len = tvm.tir.Var("n", "int64")

    with bb.function(func_name):
        model = BloomForCausalLM(config)
        param_manager.register_params(
            model, func_name, quant_scheme, get_param_quant_kind
        )

        input_ids = nn.Placeholder((bsz, 1), dtype="int32", name="input_ids")
        all_seq_len_shape = relax.Var(
            "all_seq_len", relax.ShapeStructInfo((all_seq_len,))
        )
        past_key_values = relax.Var(
            "kv_cache",
            relax.TupleStructInfo(
                [relax.ObjectStructInfo() for _ in range(config.n_layer * 2)]
            ),
        )
        with bb.dataflow():
            logits, key_value_cache = model(
                input_ids, all_seq_len_shape, past_key_values=past_key_values
            )
            params = [
                input_ids,
                all_seq_len_shape,
                past_key_values,
            ] + model.parameters()
            gv = bb.emit_output((logits, relax.Tuple(key_value_cache)))
        bb.emit_func_output(gv, params)

    mod = bb.get()
    gv = mod.get_global_var(func_name)
    bb.update_func(gv, mod[gv].with_attr("num_input", 3))


def create_kv_cache_func(bb: relax.BlockBuilder, config: BloomConfig) -> None:
    init_shape = relax.ShapeExpr(
        (
            config.max_sequence_length,
            config.n_head,
            config.hidden_size // config.n_head,
        )
    )
    with bb.function("create_kv_cache", []):
        with bb.dataflow():
            zeros = bb.emit(relax.op.zeros(init_shape, config.dtype))
            caches = []
            f_kv_cache_create = relax.extern("vm.builtin.attention_kv_cache_create")
            for _ in range(config.n_layer * 2):
                caches.append(
                    bb.emit(
                        relax.Call(
                            f_kv_cache_create,
                            args=[zeros, init_shape, relax.PrimValue(0)],
                            sinfo_args=[relax.ObjectStructInfo()],
                        )
                    )
                )
            gv = bb.emit_output(caches)
        bb.emit_func_output(gv)


def create_softmax_func(bb: relax.BlockBuilder, config: BloomConfig) -> None:
    with bb.function("softmax_with_temperature"):
        logits = nn.Placeholder(
            (1, 1, config.vocab_size), dtype="float32", name="logits"
        )
        temperature = nn.Placeholder((), dtype="float32", name="temperature")
        with bb.dataflow():
            div = bb.emit(relax.op.divide(logits, temperature))
            softmax = bb.emit(relax.op.nn.softmax(div, axis=-1))
            gv = bb.emit_output(softmax)
        bb.emit_func_output(gv, [logits, temperature])


def get_model(args, hf_config):
    model_name = args.model
    model_path = args.model_path
    dtype = args.quantization.model_dtype
    max_seq_len = args.max_seq_len
    sep_embed = args.sep_embed

    supported_model_name = model_name.lower()
    if (
        supported_model_name.startswith("bloom-")
        or supported_model_name.startswith("bloomz-")
    ):
        config = BloomConfig(**hf_config, dtype=dtype)
        print(hf_config)
        print("===")
        attrs = dir(config)
        for attr in attrs:
            if not attr.startswith("__"):
                value = getattr(config, attr)
                print(f"{attr}: {value}")
        if max_seq_len != -1:
            config.max_sequence_length = max_seq_len

        param_manager = ParamManager()
        bb = relax.BlockBuilder()
        if sep_embed:
            create_embed_func(bb, param_manager, config, args.quantization)
        create_encoding_func(bb, param_manager, config, args.quantization, sep_embed)
        create_decoding_func(bb, param_manager, config, args.quantization)
        create_kv_cache_func(bb, config)
        create_softmax_func(bb, config)
        create_metadata_func(
            bb,
            model_name=model_name,
            max_window_size=config.max_sequence_length,
            stop_tokens=[2],
            add_prefix_space=False,
        )

        mod = bb.get()
        for gv in mod.functions:
            func = mod[gv]
            if isinstance(func, relax.Function):
                mod[gv] = func.with_attr(
                    "tir_var_upper_bound",
                    {
                        "n": config.max_sequence_length,
                        "m": config.max_sequence_length,
                    },
                )

        def f_convert_pname_fwd(pname: str) -> List[str]:
                return [pname]

        def f_convert_param_bkwd(torch_name, raw_param):
            transformer_header = "transformer."
            pname = torch_name
            if not pname.startswith(transformer_header):
                pname = transformer_header + torch_name
            res = [(pname, raw_param.astype(dtype))]
            ## TODO bloom模型中 lm_head可能会与word_embeddings layer参数共享, 因此pytorch.model.bin中没有lm_head.weight参数
            ## 当前默认认为参数共享, 需适配不共享的情况
            if "word_embeddings.weight" in torch_name:
                res = res + [("lm_head.weight", raw_param.astype(dtype))]

            return res

        mod = param_manager.transform_module(
            mod,
            args.model_path,
            f_convert_pname_fwd,
            f_convert_param_bkwd,
        )

        device = tvm.cpu()
        param_list = [None] * len(param_manager.param_names)
        if args.quantization.pre_quantized:
            param_list = args.quantization.load_quantized_params(
                args.model_path,
                param_list,
                param_manager.pidx2pname,
                device
            )

        return mod, param_manager, param_list

    raise ValueError(f"Unsupported model: {model_name}")