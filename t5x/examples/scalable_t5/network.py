# Copyright 2023 The T5X Authors.
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

"""T5.1.1 Transformer model."""

from typing import Any, Sequence

from flax import linen as nn
from flax import struct
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from t5x.examples.scalable_t5 import layers

with_sharding_constraint = nn_partitioning.with_sharding_constraint
scan_with_axes = nn_partitioning.scan_with_axes
remat = nn_partitioning.remat
ScanIn = nn_partitioning.ScanIn


@struct.dataclass
class T5Config:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  vocab_size: int
  # Activation dtypes.
  dtype: Any = jnp.float32
  emb_dim: int = 512
  num_heads: int = 8
  num_encoder_layers: int = 6
  num_decoder_layers: int = 6
  head_dim: int = 64
  mlp_dim: int = 2048
  # Activation functions are retrieved from Flax.
  mlp_activations: Sequence[str] = ('relu',)
  dropout_rate: float = 0.1
  # If `True`, the embedding weights are used in the decoder output layer.
  logits_via_embedding: bool = False
  # minimal, full, or none
  remat_policy: str = 'none'
  scan_layers: bool = True
  param_scan_axis: int = 1


class EncoderLayer(nn.Module):
  """Transformer encoder layer."""
  config: T5Config

  @nn.compact
  def __call__(self, inputs, encoder_mask=None, deterministic=False):
    cfg = self.config

    # Relative position embedding as attention biases.
    encoder_bias = layers.RelativePositionBiases(
        num_buckets=32,
        max_distance=128,
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        embedding_init=nn.initializers.variance_scaling(1.0, 'fan_avg',
                                                        'uniform'),
        name='relpos_bias')(inputs.shape[-2], inputs.shape[-2], True)

    # Attention block.
    assert inputs.ndim == 3
    inputs = with_sharding_constraint(inputs, ('batch', 'length', 'embed'))
    x = layers.LayerNorm(
        dtype=cfg.dtype, name='pre_attention_layer_norm')(
            inputs)
    x = with_sharding_constraint(x, ('batch', 'length', 'embed'))
    # [batch, length, emb_dim] -> [batch, length, emb_dim]
    x = layers.MultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        head_dim=cfg.head_dim,
        dropout_rate=cfg.dropout_rate,
        name='attention')(
            x, x, encoder_mask, encoder_bias, deterministic=deterministic)
    x = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic)
    x = x + inputs
    x = with_sharding_constraint(x, ('batch', 'length', 'embed'))

    # MLP block.
    y = layers.LayerNorm(dtype=cfg.dtype, name='pre_mlp_layer_norm')(x)
    y = with_sharding_constraint(y, ('batch', 'length', 'embed'))
    # [batch, length, emb_dim] -> [batch, length, emb_dim]
    y = layers.MlpBlock(
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        name='mlp',
    )(y, deterministic=deterministic)
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)
    y = y + x
    y = with_sharding_constraint(y, ('batch', 'length', 'embed'))

    if cfg.scan_layers:
      return y, None
    else:
      return y


class DecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""
  config: T5Config

  @nn.compact
  def __call__(self,
               inputs,
               encoded,
               decoder_mask=None,
               encoder_decoder_mask=None,
               deterministic=False,
               decode=False,
               max_decode_length=None):
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

    inputs = with_sharding_constraint(inputs, ('batch', 'length', 'embed'))

    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    x = layers.LayerNorm(
        dtype=cfg.dtype, name='pre_self_attention_layer_norm')(
            inputs)
    x = with_sharding_constraint(x, ('batch', 'length', 'embed'))

    # Self-attention block
    x = layers.MultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        head_dim=cfg.head_dim,
        dropout_rate=cfg.dropout_rate,
        name='self_attention')(
            x,
            x,
            decoder_mask,
            decoder_bias,
            deterministic=deterministic,
            decode=decode)
    x = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic)
    x = x + inputs
    x = with_sharding_constraint(x, ('batch', 'length', 'embed'))

    # Encoder-Decoder block.
    y = layers.LayerNorm(
        dtype=cfg.dtype, name='pre_cross_attention_layer_norm')(
            x)
    y = with_sharding_constraint(y, ('batch', 'length', 'embed'))
    y = layers.MultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        head_dim=cfg.head_dim,
        dropout_rate=cfg.dropout_rate,
        name='encoder_decoder_attention')(
            y, encoded, encoder_decoder_mask, deterministic=deterministic)
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)
    y = y + x
    y = with_sharding_constraint(y, ('batch', 'length', 'embed'))

    # MLP block.
    z = layers.LayerNorm(dtype=cfg.dtype, name='pre_mlp_layer_norm')(y)
    z = with_sharding_constraint(z, ('batch', 'length', 'embed'))
    z = layers.MlpBlock(
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        name='mlp',
    )(z, deterministic=deterministic)
    z = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            z, deterministic=deterministic)
    z = z + y
    z = with_sharding_constraint(z, ('batch', 'length', 'embed'))

    if cfg.scan_layers:
      return z, None
    else:
      return z


class Encoder(nn.Module):
  """A stack of encoder layers."""
  config: T5Config
  shared_embedding: nn.Module

  @nn.compact
  def __call__(self,
               encoder_input_tokens,
               encoder_mask=None,
               deterministic=False):
    cfg = self.config
    assert encoder_input_tokens.ndim == 2  # [batch, length]

    # [batch, length] -> [batch, length, emb_dim]
    x = self.shared_embedding(encoder_input_tokens.astype('int32'))
    x = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic)
    x = x.astype(cfg.dtype)

    BlockLayer = EncoderLayer

    if cfg.remat_policy not in (None, 'none'):
      if cfg.remat_policy == 'minimal':
        policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
      else:
        policy = None
      BlockLayer = remat(  # pylint: disable=invalid-name
          BlockLayer,
          prevent_cse=not cfg.scan_layers,
          policy=policy,
          static_argnums=(2,))

    if cfg.scan_layers:
      initializing = self.is_mutable_collection('params')
      params_spec = (
          cfg.param_scan_axis if initializing else ScanIn(cfg.param_scan_axis))
      cache_spec = 0
      x, _ = scan_with_axes(
          BlockLayer,
          variable_axes={
              'params': params_spec,
              'cache': cache_spec,
          },
          split_rngs={
              'params': True,
              'dropout': True
          },
          in_axes=(nn.broadcast, nn.broadcast),
          length=cfg.num_encoder_layers,
          axis_name='layers')(
              config=cfg, name='encoder')(x, encoder_mask, deterministic)
    else:
      for lyr in range(cfg.num_encoder_layers):
        # [batch, length, emb_dim] -> [batch, length, emb_dim]
        x = BlockLayer(
            config=cfg, name=f'layers_{lyr}')(x, encoder_mask, deterministic)

    x = layers.LayerNorm(dtype=cfg.dtype, name='encoder_norm')(x)
    return nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)


class Decoder(nn.Module):
  """A stack of decoder layers as a part of an encoder-decoder architecture."""
  config: T5Config
  shared_embedding: nn.Module

  @nn.compact
  def __call__(self,
               encoded,
               decoder_input_tokens,
               decoder_positions=None,
               decoder_mask=None,
               encoder_decoder_mask=None,
               deterministic=False,
               decode=False,
               max_decode_length=None):
    cfg = self.config
    assert decoder_input_tokens.ndim == 2  # [batch, len]

    # [batch, length] -> [batch, length, emb_dim]
    y = self.shared_embedding(decoder_input_tokens.astype('int32'))
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)
    y = y.astype(cfg.dtype)

    BlockLayer = DecoderLayer

    if cfg.remat_policy not in (None, 'none'):
      if cfg.remat_policy == 'minimal':
        policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
      else:
        policy = None
      BlockLayer = remat(  # pylint: disable=invalid-name
          BlockLayer,
          prevent_cse=not cfg.scan_layers,
          policy=policy,
          static_argnums=(4, 5, 6))
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
          length=cfg.num_decoder_layers,
          axis_name='layers')(
              config=cfg,
              name='decoder')(y, encoded, decoder_mask, encoder_decoder_mask,
                              deterministic, decode, max_decode_length)
    else:
      for lyr in range(cfg.num_decoder_layers):
        # [batch, length, emb_dim] -> [batch, length, emb_dim]
        y = BlockLayer(
            config=cfg, name=f'layers_{lyr}')(
                y,
                encoded,
                decoder_mask=decoder_mask,
                encoder_decoder_mask=encoder_decoder_mask,
                deterministic=deterministic,
                decode=decode,
                max_decode_length=max_decode_length)

    y = layers.LayerNorm(dtype=cfg.dtype, name='decoder_norm')(y)
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)

    # [batch, length, emb_dim] -> [batch, length, vocab_size]
    if cfg.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = self.shared_embedding.attend(y)
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(y.shape[-1])
    else:
      logits = layers.DenseGeneral(
          cfg.vocab_size,
          dtype=jnp.float32,  # Use float32 for stabiliity.
          kernel_axes=('embed', 'vocab'),
          name='logits_dense')(
              y)
    return logits


class Transformer(nn.Module):
  """An encoder-decoder Transformer model."""
  config: T5Config
  # needed only for janky models.py scan_layers detection.
  scan_layers: bool = struct.field(init=False)

  def __post_init__(self):
    super().__post_init__()
    # needed only for janky models.py scan_layers detection.
    object.__setattr__(self, 'scan_layers',
                       object.__getattribute__(self, 'config').scan_layers)

  def setup(self):
    cfg = self.config
    self.shared_embedding = layers.Embed(
        num_embeddings=cfg.vocab_size,
        features=cfg.emb_dim,
        dtype=cfg.dtype,
        attend_dtype=jnp.float32,  # for logit training stability
        embedding_init=nn.initializers.normal(stddev=1.0),
        one_hot=True,
        name='token_embedder')

    self.encoder = Encoder(config=cfg, shared_embedding=self.shared_embedding)
    self.decoder = Decoder(config=cfg, shared_embedding=self.shared_embedding)

  def encode(self,
             encoder_input_tokens,
             encoder_segment_ids=None,
             enable_dropout=True):
    """Applies Transformer encoder-branch on the inputs."""
    cfg = self.config
    assert encoder_input_tokens.ndim == 2  # (batch, len)

    # Make padding attention mask.
    encoder_mask = layers.make_attention_mask(
        encoder_input_tokens > 0, encoder_input_tokens > 0, dtype=cfg.dtype)
    # Add segmentation block-diagonal attention mask if using segmented data.
    if encoder_segment_ids is not None:
      encoder_mask = layers.combine_masks(
          encoder_mask,
          layers.make_attention_mask(
              encoder_segment_ids,
              encoder_segment_ids,
              jnp.equal,
              dtype=cfg.dtype))

    return self.encoder(
        encoder_input_tokens, encoder_mask, deterministic=not enable_dropout)

  def decode(
      self,
      encoded,
      encoder_input_tokens,  # only needed for masks
      decoder_input_tokens,
      decoder_target_tokens,
      encoder_segment_ids=None,
      decoder_segment_ids=None,
      decoder_positions=None,
      enable_dropout=True,
      decode=False,
      max_decode_length=None):
    """Applies Transformer decoder-branch on encoded-input and target."""
    cfg = self.config

    # Make padding attention masks.
    if decode:
      # Do not mask decoder attention based on targets padding at
      # decoding/inference time.
      decoder_mask = None
      encoder_decoder_mask = layers.make_attention_mask(
          jnp.ones_like(decoder_target_tokens),
          encoder_input_tokens > 0,
          dtype=cfg.dtype)
    else:
      decoder_mask = layers.make_decoder_mask(
          decoder_target_tokens=decoder_target_tokens,
          dtype=cfg.dtype,
          decoder_segment_ids=decoder_segment_ids)
      encoder_decoder_mask = layers.make_attention_mask(
          decoder_target_tokens > 0, encoder_input_tokens > 0, dtype=cfg.dtype)

    # Add segmentation block-diagonal attention masks if using segmented data.
    if encoder_segment_ids is not None:
      if decode:
        raise ValueError(
            'During decoding, packing should not be used but '
            '`encoder_segment_ids` was passed to `Transformer.decode`.')

      encoder_decoder_mask = layers.combine_masks(
          encoder_decoder_mask,
          layers.make_attention_mask(
              decoder_segment_ids,
              encoder_segment_ids,
              jnp.equal,
              dtype=cfg.dtype))

    logits = self.decoder(
        encoded,
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask,
        deterministic=not enable_dropout,
        decode=decode,
        max_decode_length=max_decode_length)
    return logits

  def __call__(self,
               encoder_input_tokens,
               decoder_input_tokens,
               decoder_target_tokens,
               encoder_segment_ids=None,
               decoder_segment_ids=None,
               encoder_positions=None,
               decoder_positions=None,
               *,
               enable_dropout: bool = True,
               decode: bool = False):
    """Applies Transformer model on the inputs.

    This method requires both decoder_target_tokens and decoder_input_tokens,
    which is a shifted version of the former. For a packed dataset, it usually
    has additional processing applied. For example, the first element of each
    sequence has id 0 instead of the shifted EOS id from the previous sequence.

    Args:
      encoder_input_tokens: input data to the encoder.
      decoder_input_tokens: input token to the decoder.
      decoder_target_tokens: target token to the decoder.
      encoder_segment_ids: encoder segmentation info for packed examples.
      decoder_segment_ids: decoder segmentation info for packed examples.
      encoder_positions: encoder subsequence positions for packed examples.
      decoder_positions: decoder subsequence positions for packed examples.
      enable_dropout: Ensables dropout if set to True.
      decode: Whether to prepare and use an autoregressive cache.

    Returns:
      logits array from full transformer.
    """
    encoded = self.encode(
        encoder_input_tokens,
        encoder_segment_ids=encoder_segment_ids,
        enable_dropout=enable_dropout)

    return self.decode(
        encoded,
        encoder_input_tokens,  # only used for masks
        decoder_input_tokens,
        decoder_target_tokens,
        encoder_segment_ids=encoder_segment_ids,
        decoder_segment_ids=decoder_segment_ids,
        decoder_positions=decoder_positions,
        enable_dropout=enable_dropout,
        decode=decode)
