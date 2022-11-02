# Copyright 2021 The T5X Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal decoder-only Transformer model."""

from typing import Any, Optional, Sequence

from flax import linen as nn
from flax import struct
import jax.numpy as jnp
from t5x.examples.scalable_decoder_only import layers
from flax.linen import partitioning as nn_partitioning
import jax
from jax.ad_checkpoint import checkpoint_name
with_sharding_constraint = nn_partitioning.with_sharding_constraint
scan_with_axes = nn_partitioning.scan_with_axes
remat = nn_partitioning.remat
ScanIn = nn_partitioning.ScanIn


@struct.dataclass
class TransformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  vocab_size: int
  # Activation dtypes.
  dtype: Any = jnp.float32
  emb_dim: int = 512
  num_heads: int = 8
  num_layers: int = 6
  head_dim: int = 64
  mlp_dim: int = 2048
  # Activation functions are retrieved from Flax.
  mlp_activations: Sequence[str] = ('relu',)
  dropout_rate: float = 0.1
  # If `True`, the embedding weights are used in the decoder output layer.
  logits_via_embedding: bool = False
  remat_policy: str = 'none'
  scan_layers: bool = True
  param_scan_axis: int = 1

class DecoderLayer(nn.Module):
  """Transformer decoder layer."""
  config: TransformerConfig

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               decoder_mask: Optional[jnp.ndarray] = None,
               deterministic: bool = False,
               decode: bool = False,
               max_decode_length: Optional[int] = None,
               prefill: bool = False,
               prefill_lengths: Optional[jnp.ndarray] = None):
    """Applies decoder block module."""
    cfg = self.config

    # Relative position embedding as attention biases.
    l = max_decode_length if decode and max_decode_length else inputs.shape[-2]
    decoder_bias = layers.RelativePositionBiases(
        num_buckets=32,
        max_distance=128,
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        embedding_init=nn.initializers.variance_scaling(1.0, 'fan_avg',
                                                        'uniform'),
        name='relpos_bias')(l, l, False)

    # `inputs` is layer input with a shape [batch, length, emb_dim].
    inputs = with_sharding_constraint(inputs, ('batch', 'length', 'embed'))
    x = layers.LayerNorm(
        dtype=cfg.dtype, name='pre_self_attention_layer_norm')(
            inputs)
    lnx = with_sharding_constraint(inputs, ('batch', 'length', 'embed'))

    # Self-attention block
    attn = layers.MultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        head_dim=cfg.head_dim,
        dropout_rate=cfg.dropout_rate,
        name='self_attention')(
            lnx,
            lnx,
            decoder_mask,
            decoder_bias,
            deterministic=deterministic,
            decode=decode,
            prefill=prefill,
            prefill_lengths=prefill_lengths)
    attn = checkpoint_name(attn, 'out_proj')

    attn = nn.Dropout(
        rate=cfg.dropout_rate,
        broadcast_dims=(-2,),
        name='post_self_attention_dropout')(
            attn, deterministic=deterministic)
    attn = with_sharding_constraint(attn, ('batch', 'length', 'embed'))
    # Parallel attention formulation
    # cf: https://arxiv.org/pdf/2204.02311.pdf
    # y = input + MLP(LayerNorm(input)) + Attention(LayerNorm(input))
    # MLP block.
    mlp = layers.MlpBlock(
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        name='mlp',
    )(lnx, deterministic=deterministic)
    mlp = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,), name='post_mlp_dropout')(
            mlp, deterministic=deterministic)
    mlp = with_sharding_constraint(mlp, ('batch', 'length', 'embed'))
    y = inputs + mlp + attn
    y = with_sharding_constraint(y, ('batch', 'length', 'embed'))

    if cfg.scan_layers:
      return y, None
    else:
      return y


class Decoder(nn.Module):
  """A stack of decoder layers."""
  config: TransformerConfig

  @nn.compact
  def __call__(self,
               decoder_input_tokens: jnp.ndarray,
               decoder_target_tokens: jnp.ndarray,
               decoder_segment_ids: Optional[jnp.ndarray] = None,
               decoder_positions: Optional[jnp.ndarray] = None,
               decoder_causal_attention: Optional[jnp.ndarray] = None,
               *,
               enable_dropout: bool = True,
               decode: bool = False,
               max_decode_length: Optional[int] = None,
               prefill: Optional[bool] = None,
               prefill_lengths: Optional[jnp.ndarray] = None):
    """Applies LanguageModel on the inputs.

    For a decoder-only architecture with the notion of "prefix", e.g., a prefix
    LM where the prefix corresponds to the "inputs" of a supervised dataset, we
    perform the "prefill" operation to fill the autoregressive cache
    corresponding to the prefix region in one go. Then the autoregressive
    decoding starts after the prefix. This makes the decoding process more
    efficient. In addition, it gives an option to use bidirectional attention in
    the prefix region because the cache is filled simultaneously.

    Args:
      decoder_input_tokens: input token to the decoder.
      decoder_target_tokens: target token to the decoder.
      decoder_segment_ids: decoder segmentation info for packed examples.
      decoder_positions: decoder subsequence positions for packed examples.
      decoder_causal_attention: a binary mask indicating the portion of the
        sequence to apply bidirectional attention to instead of causal. As an
        example, useful to specify the "inputs" portion of a concatenated
        sequence for a prefix LM.
      enable_dropout: enables dropout if set to True.
      decode: whether to prepare and use an autoregressive cache as opposed to
        using teacher-forcing.
      max_decode_length: maximum sequence length to be decoded.
      prefill: whether to run a partial sequence to prefill the cache.
      prefill_lengths: an array of shape [batch] denoting the length of each
        partial sequence we are filling in the cache.

    Returns:
      logits array.
    """
    cfg = self.config
    deterministic = not enable_dropout
    assert decoder_input_tokens.ndim == 2  # [batch, len]

    if decode:
      decoder_mask = None
    else:
      decoder_mask = layers.make_decoder_mask(
          decoder_target_tokens=decoder_target_tokens,
          dtype=cfg.dtype,
          decoder_causal_attention=decoder_causal_attention,
          decoder_segment_ids=decoder_segment_ids)

    embedding = layers.Embed(
        num_embeddings=cfg.vocab_size,
        features=cfg.emb_dim,
        dtype=cfg.dtype,
        attend_dtype=jnp.float32,  # for logit training stability
        embedding_init=nn.initializers.normal(stddev=1.0),
        one_hot=False,
        name='token_embedder')
    y = embedding(decoder_input_tokens.astype('int32'))

    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,), name='input_dropout')(
            y, deterministic=deterministic)
    y = y.astype(cfg.dtype)

    BlockLayer = DecoderLayer

    if cfg.remat_policy not in (None, 'none'):
      if cfg.remat_policy == 'minimal':
        # Example of using a combination of checkpoint policies  
        policy_1 = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
        policy_2 = jax.checkpoint_policies.save_only_these_names(
          'combined_qkv_proj', 'query_proj', 'value_proj', 'key_proj',
         'out_proj', 'context')
        policy = jax.checkpoint_policies.save_from_both_policies(policy_1, policy_2)
      else: #full
        policy = jax.checkpoint_policies.nothing_saveable
      BlockLayer = remat(  # pylint: disable=invalid-name
          BlockLayer,
          prevent_cse=not cfg.scan_layers,
          policy=policy,
          static_argnums=(2, 3, 5))
    if cfg.scan_layers:
      initializing = self.is_mutable_collection('params')
      params_spec = (
          cfg.param_scan_axis if initializing else ScanIn(cfg.param_scan_axis))
      cache_spec = 0
      y, _ = scan_with_axes(
          BlockLayer,
          variable_axes={
              'params': params_spec,
              'cache': cache_spec
          },
          split_rngs={
              'params': True,
              'dropout': True
          },
          in_axes=(nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast,
                   nn.broadcast, nn.broadcast),
          length=cfg.num_layers,
          axis_name='layers')(
              config=cfg, name='layers')(
                  y, decoder_mask,
                  deterministic, decode, max_decode_length,
                  prefill, prefill_lengths
              )
    else:
      for lyr in range(cfg.num_layers):
        # [batch, length, emb_dim] -> [batch, length, emb_dim]
        y = BlockLayer(
            config=cfg,
            name=f'layers_{lyr}')(
                y,
                decoder_mask=decoder_mask,
                deterministic=deterministic,
                decode=decode,
                max_decode_length=max_decode_length,
                prefill=prefill,
                prefill_lengths=prefill_lengths)

    y = layers.LayerNorm(dtype=cfg.dtype, name='decoder_norm')(y)
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,), name='output_dropout')(
            y, deterministic=deterministic)

    # [batch, length, emb_dim] -> [batch, length, vocab_size]
    if cfg.logits_via_embedding:
      # Use the transpose of embedding matrix for the logit transform.
      logits = embedding.attend(y)
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(y.shape[-1])
    else:
      # Use a separate dense layer for the logit transform.
      logits = layers.DenseGeneral(
          cfg.vocab_size,
          dtype=jnp.float32,  # Use float32 for stabiliity.
          kernel_axes=('embed', 'vocab'),
          name='logits_dense')(
              y)
    return logits


# TODO(hwchung): remove this after figuring out the name scope issue.
class DecoderWrapper(nn.Module):
  """Thin wrapper for the outer "decoder/" name scope."""

  config: TransformerConfig
  scan_layers: bool = struct.field(init=False)

  def setup(self):
    self.decoder = Decoder(self.config, name='decoder')

  def __call__(self, *args, **kwargs):
    return self.decoder(*args, **kwargs)
