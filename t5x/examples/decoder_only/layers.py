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

"""Dense attention classes and mask/weighting functions."""

# pylint: disable=attribute-defined-outside-init,g-bare-generic

import dataclasses
import functools
import operator
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

from flax import linen as nn
import flax.core.variables as variables
from flax.linen import partitioning as nn_partitioning
from flax.training import common_utils
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import numpy as np


# from flax.linen.partitioning import param_with_axes, with_sharding_constraint
param_with_axes = nn_partitioning.param_with_axes
with_sharding_constraint = nn_partitioning.with_sharding_constraint


# Type annotations
Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Sequence[int]
Activation = Callable[..., Array]
# Parameter initializers.
Initializer = Callable[[PRNGKey, Shape, DType], Array]

default_embed_init = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'normal', out_axis=0)


def dot_product_attention(query: Array,
                          key: Array,
                          value: Array,
                          bias: Optional[Array] = None,
                          dropout_rng: Optional[PRNGKey] = None,
                          dropout_rate: float = 0.,
                          deterministic: bool = False,
                          dtype: DType = jnp.float32,
                          float32_logits: bool = False):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  Args:
    query: queries for calculating attention with shape of `[batch, q_length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch, kv_length,
      num_heads, qk_depth_per_head]`.
    value: values to be used in attention with shape of `[batch, kv_length,
      num_heads, v_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch, num_heads, q_length, kv_length]` This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: float32)
    float32_logits: bool, if True then compute logits in float32 to avoid
      numerical issues with bfloat16.

  Returns:
    Output of shape `[batch, length, num_heads, v_depth_per_head]`.
  """
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
      'q, k, v batch dims must match.')
  assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
      'q, k, v num_heads must match.')
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # Casting logits and softmax computation for float32 for model stability.
  if float32_logits:
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)

  # `attn_weights`: [batch, num_heads, q_length, kv_length]
  attn_weights = jnp.einsum('bqhd,bkhd->bhqk', query, key)

  # Apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias.astype(attn_weights.dtype)

  # Normalize the attention weights across `kv_length` dimension.
  attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  # Apply attention dropout.
  if not deterministic and dropout_rate > 0.:
    keep_prob = 1.0 - dropout_rate
    # T5 broadcasts along the "length" dim, but unclear which one that
    # corresponds to in positional dimensions here, assuming query dim.
    dropout_shape = list(attn_weights.shape)
    dropout_shape[-2] = 1
    keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
    keep = jnp.broadcast_to(keep, attn_weights.shape)
    multiplier = (
        keep.astype(attn_weights.dtype) / jnp.asarray(keep_prob, dtype=dtype))
    attn_weights = attn_weights * multiplier

  # Take the linear combination of `value`.
  return jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)


class MultiHeadDotProductAttention(nn.Module):
  """Multi-head dot-product attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      head_dim: dimension of each head.
      dtype: the dtype of the computation.
      dropout_rate: dropout rate
      kernel_init: initializer for the kernel of the Dense layers.
      float32_logits: bool, if True then compute logits in float32 to avoid
        numerical issues with bfloat16.
  """
  num_heads: int
  head_dim: int
  dtype: DType = jnp.float32
  dropout_rate: float = 0.
  kernel_init: Initializer = nn.initializers.variance_scaling(
      1.0, 'fan_in', 'normal')
  float32_logits: bool = False

  def update_cache_prefill(
      self, key: Array, value: Array, cached_key: variables.Variable,
      cached_value: variables.Variable, cache_index: variables.Variable,
      prefill_lengths: Array
  ) -> Tuple[Array, Array, Array, Array, Array, Array]:
    """Update the autoregressive cache for multiple timesteps at once.

    This is useful for things like a prefix-lm where the encoder section of the
    input is visible bidirectionally. The key and value for this section need to
    be computed in a single shot, as a step by step approach would result in
    causal attention.

    Args:
      key: The calculated key used in attention. [batch..., length, num_heads,
        features_per_head]
      value: The calculated value used in attention. [batch..., length,
        num_heads, features_per_head]
      cached_key: The cache of previous keys. [batch..., num_heads,
        features_per_head, length]
      cached_value: The cache of previous values. [batch..., num_heads,
        features_per_head, length]
      cache_index: The timestep that we are currently calculating the key and
        value for. [batch]
      prefill_lengths: The number of timesteps we should fill in the cache.
        [batch]

    Returns:
      The key, value, and the last timestep we just filled in the cache.
      We also return the new cache values for now because assigning to a
      variable inside of a method doesn't work. These returns will be removed
      eventually.
    """
    # Make a reference to the data underlaying the variable for ease of
    # use.
    cache_index.value = prefill_lengths
    # Note, the cache index is now a vector of batch size so that each example
    # can start just after its prefix, which can be different lengths for
    # different examples.
    cur_index = cache_index.value
    # Move the sequence dimension to the end to match the cache shapes.
    key_cached = jnp.moveaxis(key, -3, -1)
    value_cached = jnp.moveaxis(value, -3, -1)
    # Reshape the index so the batch is at the beginning. The default
    # broadcasting behavior is to add singleton dims to the front, but we need
    # them at the end.
    batch_first_index = jnp.reshape(
        cur_index, (-1,) + tuple(1 for _ in range(cached_key.value.ndim - 1)))
    # Calculate a mask that will set any position past the prefix to zero
    # when applied to the key.
    key_mask = (
        lax.broadcasted_iota(jnp.int32, cached_key.value.shape,
                             cached_key.value.ndim - 1) < batch_first_index)
    value_mask = (
        lax.broadcasted_iota(jnp.int32, cached_value.value.shape,
                             cached_value.value.ndim - 1) < batch_first_index)
    # Set the caches with the calculated key and values but hide anything
    # past the prefix.
    cached_key_value = key_cached * key_mask
    cached_value_value = value_cached * value_mask
    # TODO(hwchung): remove the return values once direct assignment to
    # variables inside a method is possible.
    return (key, value, cur_index, cached_key_value, cached_value_value,
            prefill_lengths)

  def update_cache_decode(
      self, key: Array, value: Array, cached_key: variables.Variable,
      cached_value: variables.Variable, cache_index: variables.Variable
  ) -> Tuple[Array, Array, Array, Array, Array, Array]:
    """Update the next timestep in the autoregressive cache.

    This is used during step by step decoding where each key and value we get
    are a single (the next) timestep.

    Args:
      key: The calculated key used in attention. [batch..., 1, num_heads,
        features_per_head]
      value: The calculated value used in attention. [batch..., 1, num_heads,
        features_per_head]
      cached_key: The cache of previous keys. [batch..., num_heads,
        features_per_head, length]
      cached_value: The cache of previous values. [batch..., num_heads,
        features_per_head, length]
      cache_index: The timestep that we are currently calculating the key and
        value for. [batch] if we are decoding after doing a prefill or [1] if we
        are starting with step-by-step decoding.

    Returns:
      The key, value, and the last timestep we just filled in the cache. Note:
      this index is the last timestep we just fill, the actual value of the
      `cache_index` is already increased to point to the next timestep to fill.
      We also return the new cache values for now because assigning to a
      variable inside of a method doesn't work. These returns will be removed
      eventually.
    """
    cache_length = cached_key.value.shape[-1]
    # Create a OHE of the current index. NOTE: the index is increased
    # below.
    # Note: We reshape the index into a column vector so that it will work
    # if the index is a scalar or a vector with different cache positions
    # from different elements in a batch.
    cur_index = jnp.reshape(cache_index.value, (-1,))
    one_hot_indices = jax.nn.one_hot(cur_index, cache_length, dtype=key.dtype)
    # In order to update the key, value caches with the current key and
    # value, we move the length axis to the back, similar to what we did
    # for the cached ones above.
    # Note these are currently the key and value of a single position,
    # since we feed one position at a time.
    one_token_key = jnp.moveaxis(key, -3, -1)
    one_token_value = jnp.moveaxis(value, -3, -1)
    # The one hot indices are now either [1, length] for a scalar index or
    # [batch size, length] for examples where there are different lengths
    # of prefixes. We need to add dims for num_heads and num_features as
    # broadcasting doesn't work for the batched version.
    one_hot_indices = jnp.expand_dims(
        jnp.expand_dims(one_hot_indices, axis=1), axis=1)
    # Update key, value caches with our new 1d spatial slices.
    # We implement an efficient scatter into the cache via one-hot
    # broadcast and addition.
    # Key/Value have seq lengths of 1 while one_hot has a seq_length
    # of length. key/value will broadcast their value to each timestep
    # and the onehot will mask all but the correct timesteps.
    key = cached_key.value + one_token_key * one_hot_indices
    value = cached_value.value + one_token_value * one_hot_indices
    cached_key_value = key
    cached_value_value = value
    cache_index_value = cache_index.value + 1
    # Move the keys and values back to their original shapes.
    key = jnp.moveaxis(key, -1, -3)
    value = jnp.moveaxis(value, -1, -3)
    # TODO(hwchung): remove the return values once direct assignment to
    # variables inside a method is possible.
    return (key, value, cur_index, cached_key_value, cached_value_value,
            cache_index_value)

  @nn.compact
  def __call__(self,
               inputs_q: Array,
               inputs_kv: Array,
               mask: Optional[Array] = None,
               bias: Optional[Array] = None,
               *,
               decode: bool = False,
               deterministic: bool = False,
               prefill: bool = False,
               prefill_lengths: Optional[Array] = None) -> Array:
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    There are two modes: decoding and non-decoding (e.g., training). The mode is
    determined by `decode`.

    During decoding mode, this method is called twice, by `init` and
    `apply`. In the former, inputs_q: `[batch..., length, qkv_features]` and
    inputs_kv: `[batch..., length, qkv_features]`.

    During apply, query, key and value all have the shape: `[batch * beam, 1,
    qkv_features]` where the batch dimension is added to include multiple beams.
    Note that the batch dimension is different during the `init` and `apply`
    calls.  This is because the cached variables are directly passed-in during
    `apply` method. In other words, the cache variables such as `cached_key` are
    initialized with `batch` dim, expanded by tiling in the beam search function
    to `batch * beam` dimension, and passed to the `apply` method as part of a
    variable dict.

    Args:
      inputs_q: input queries of shape `[batch, q_length, embed]`.
      inputs_kv: key/values of shape `[batch, kv_length, embed]`.
      mask: attention mask of shape `[batch, num_heads, q_length, kv_length]`.
      bias: attention bias of shape `[batch, num_heads, q_length, kv_length]`.
      decode: whether to prepare and use an autoregressive cache.
      deterministic: whether deterministic or not (to apply dropout)
      prefill: whether to run a partial sequence to prefill the cache.
      prefill_lengths: an array of shape [batch] denoting the length of each
        partial sequence we are filling in the cache.

    Returns:
      output of shape `[batch, q_length, embed]`.
    """
    projection = functools.partial(
        DenseGeneral,
        axis=-1,
        features=(self.num_heads, self.head_dim),
        kernel_axes=('embed', 'joined_kv'),
        dtype=self.dtype)

    # NOTE: T5 does not explicitly rescale the attention logits by
    #       1/sqrt(depth_kq)!  This is folded into the initializers of the
    #       linear transformations, which is equivalent under Adafactor.
    depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)
    query_init = lambda *args: self.kernel_init(*args) / depth_scaling

    # Project inputs_q to multi-headed q/k/v
    # dimensions are then [batch, length, num_heads, head_dim]
    query = projection(kernel_init=query_init, name='query')(inputs_q)
    key = projection(kernel_init=self.kernel_init, name='key')(inputs_kv)
    value = projection(kernel_init=self.kernel_init, name='value')(inputs_kv)

    query = with_sharding_constraint(query, ('batch', 'length', 'heads', 'kv'))
    key = with_sharding_constraint(key, ('batch', 'length', 'heads', 'kv'))
    value = with_sharding_constraint(value, ('batch', 'length', 'heads', 'kv'))

    if prefill and decode:
      raise ValueError('prefill and decode cannot both be true at the same'
                       'time. If you are using a prefix LM with bidirectional '
                       'attention on the inputs, please make a call with '
                       'prefill=True that includes an attention mask that '
                       'covers your inputs first and then make your decoding '
                       'calls.')
    if prefill or decode:
      # Detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      # The key and value have dimension
      # [batch..., length, num_heads, features_per_head], but we cache them as
      # [batch..., num_heads, features_per_head, length] as a TPU fusion
      # optimization. This also enable the "scatter via one-hot broadcast"
      # trick, which means we do a one-hot broadcast instead of a scatter/gather
      # operations, which gives a 3-4x speedup in practice.
      swap_dims = lambda x: x[:-3] + tuple(x[i] for i in [-2, -1, -3])
      cached_key = self.variable('cache', 'cached_key', jnp.zeros,
                                 swap_dims(key.shape), key.dtype)
      cached_value = self.variable('cache', 'cached_value', jnp.zeros,
                                   swap_dims(value.shape), value.dtype)
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.int32))
      if is_initialized:
        # Here we are in "apply()".
        *batch_dims, num_heads, features_per_head, length = (
            cached_key.value.shape)
        if prefill:
          if prefill_lengths is None:
            # Figure out how far each element in the batch fills the cache based
            # on the mask. We index each element in the batch, the first head
            # dim (because this is always set to one), and the first query
            # vector. If there is any prefix at all, the first element in the
            # prefix would be part of it.
            prefill_lengths = jnp.sum(
                mask[:, 0, 0, :], axis=-1).astype(cache_index.value.dtype)
          (key, value, cur_index, cached_key_value, cached_value_value,
           cache_index_value) = self.update_cache_prefill(
               key, value, cached_key, cached_value, cache_index,
               prefill_lengths)
        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        elif decode:
          # Check the shape of the cached key against the input query.
          expected_shape = tuple(batch_dims) + (1, num_heads, features_per_head)
          if expected_shape != query.shape:
            raise ValueError('Autoregressive cache shape error, '
                             'expected query shape %s instead got %s.' %
                             (expected_shape, query.shape))
          (key, value, cur_index, cached_key_value, cached_value_value,
           cache_index_value) = self.update_cache_decode(
               key, value, cached_key, cached_value, cache_index)
          # Enforcing the Causal mask over previous positions and selecting only
          # the bias value for the current index is only needed during decode
          # mode where a single example is feed at a time. In prefill mode we
          # uses these as provided, that same way it is done in a normal forward
          # pass, like when computing logits during training.

          # Causal mask for cached decoder self-attention: our single query
          # position should only attend to those key positions that have already
          # been generated and cached, not the remaining zero elements.

          # (1, 1, length) represent (head dim, query length, key length)
          # query length is 1 because during decoding we deal with one
          # index.
          # The same mask is applied to all batch elements and heads.
          #
          # Add trailing dims to the current index so it can either
          # broadcast over the batch dim or it can just be batch size.
          mask = combine_masks(
              mask,
              jnp.broadcast_to(
                  jnp.arange(length),
                  tuple(batch_dims) +
                  (1, 1, length)) <= jnp.reshape(cur_index, (-1, 1, 1, 1)))
          # Grab the correct relative attention bias during decoding. This is
          # only required during single step decoding.
          if bias is not None:
            # The bias is a full attention matrix, but during decoding we only
            # have to take a slice of it.
            # This is equivalent to `bias[..., cur_index:cur_index+1, :]`.  If
            # we are doing prefix decoding where `cur_index` is a vector the
            # result will be `[batch, heads, 1, :]`. If `cur_index` is a scalar
            # like in encdec decoding, the result will be `[1, heads, 1, :]`.
            # We use a one-hot einsum rather than a slice to avoid introducing a
            # Gather op that is currently lowered poorly by SPMD passes, adding
            # expensive all-reduce and all-gather operations.

            bias = jnp.einsum(
                'bq, bhqk->bhk',
                common_utils.onehot(cur_index, num_classes=length), bias)
            bias = jnp.expand_dims(bias, 2)

        # Currently, updating a variable inside of a method is not handled
        # in flax, so we return the actual values and assign them in the main
        # compacted call for now.
        # TODO(brianlester,levskaya): Move variable assignment inside of the
        # cache update functions once variable references are tracked across
        # transform boundaries.
        cache_index.value = cache_index_value
        cached_key.value = cached_key_value
        cached_value.value = cached_value_value

    # Convert the boolean attention mask to an attention bias.
    if mask is not None:
      # attention mask in the form of attention bias
      attention_bias = lax.select(
          mask > 0,
          jnp.full(mask.shape, 0.).astype(self.dtype),
          jnp.full(mask.shape, -1e10).astype(self.dtype))
    else:
      attention_bias = None

    # Add provided bias term (e.g. relative position embedding).
    if bias is not None:
      attention_bias = combine_biases(attention_bias, bias)

    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')

    # Apply attention.
    x = dot_product_attention(
        query,
        key,
        value,
        bias=attention_bias,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        deterministic=deterministic,
        dtype=self.dtype,
        float32_logits=self.float32_logits)

    # Back to the original inputs dimensions.
    out = DenseGeneral(
        features=inputs_q.shape[-1],  # output dim is set to the input dim.
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        kernel_axes=('joined_kv', 'embed'),
        dtype=self.dtype,
        name='out')(
            x)
    return out


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple([ax if ax >= 0 else ndim + ax for ax in axes])


def _canonicalize_tuple(x):
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)


# ------------------------------------------------------------------------------
# DenseGeneral for attention layers.
# ------------------------------------------------------------------------------
class DenseGeneral(nn.Module):
  """A linear transformation (without bias) with flexible axes.

    Attributes:
      features: tuple with numbers of output features.
      axis: tuple with axes to apply the transformation on.
      dtype: the dtype of the computation (default: float32).
      kernel_init: initializer function for the weight matrix.
  """
  features: Union[Iterable[int], int]
  axis: Union[Iterable[int], int] = -1
  dtype: DType = jnp.float32
  kernel_init: Initializer = nn.initializers.variance_scaling(
      1.0, 'fan_in', 'truncated_normal')
  kernel_axes: Tuple[str, ...] = ()

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    features = _canonicalize_tuple(self.features)
    axis = _canonicalize_tuple(self.axis)

    inputs = jnp.asarray(inputs, self.dtype)
    axis = _normalize_axes(axis, inputs.ndim)

    kernel_shape = tuple([inputs.shape[ax] for ax in axis]) + features
    kernel_param_shape = (np.prod([inputs.shape[ax] for ax in axis]),
                          np.prod(features))
    kernel = param_with_axes(
        'kernel',
        self.kernel_init,
        kernel_param_shape,
        jnp.float32,
        axes=self.kernel_axes)
    kernel = jnp.asarray(kernel, self.dtype)
    kernel = jnp.reshape(kernel, kernel_shape)

    contract_ind = tuple(range(0, len(axis)))
    return lax.dot_general(inputs, kernel, ((axis, contract_ind), ((), ())))


def _convert_to_activation_function(
    fn_or_string: Union[str, Callable]) -> Callable:
  """Convert a string to an activation function."""
  if fn_or_string == 'linear':
    return lambda x: x
  elif isinstance(fn_or_string, str):
    return getattr(nn, fn_or_string)
  elif callable(fn_or_string):
    return fn_or_string
  else:
    raise ValueError("don't know how to convert %s to an activation function" %
                     (fn_or_string,))


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    intermediate_dim: Shared dimension of hidden layers.
    activations: Type of activations for each layer.  Each element is either
      'linear', a string function name in flax.linen, or a function.
    kernel_init: Kernel function, passed to the dense layers.
    deterministic: Whether the dropout layers should be deterministic.
    intermediate_dropout_rate: Dropout rate used after the intermediate layers.
    dtype: Type for the dense layer.
  """
  intermediate_dim: int = 2048
  activations: Sequence[Union[str, Callable]] = ('relu',)
  kernel_init: Initializer = nn.initializers.variance_scaling(
      1.0, 'fan_in', 'truncated_normal')
  intermediate_dropout_rate: float = 0.1
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs, decode: bool = False, deterministic: bool = False):
    """Applies Transformer MlpBlock module."""
    # Iterate over specified MLP input activation functions.
    # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
    activations = []
    for idx, act_fn in enumerate(self.activations):
      dense_name = 'wi' if len(self.activations) == 1 else f'wi_{idx}'
      x = DenseGeneral(
          self.intermediate_dim,
          dtype=self.dtype,
          kernel_init=self.kernel_init,
          kernel_axes=('embed', 'mlp'),
          name=dense_name)(
              inputs)
      x = _convert_to_activation_function(act_fn)(x)
      activations.append(x)

    # Take elementwise product of above intermediate activations.
    x = functools.reduce(operator.mul, activations)
    # Apply dropout and final dense output projection.
    x = nn.Dropout(
        rate=self.intermediate_dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic)  # Broadcast along length.
    x = with_sharding_constraint(x, ('batch', 'length', 'mlp'))
    output = DenseGeneral(
        inputs.shape[-1],
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        kernel_axes=('mlp', 'embed'),
        name='wo')(
            x)
    return output


class Embed(nn.Module):
  """A parameterized function from integers [0, n) to d-dimensional vectors.

  Attributes:
    num_embeddings: number of embeddings.
    features: number of feature dimensions for each embedding.
    dtype: the dtype of the embedding vectors (default: float32).
    embedding_init: embedding initializer.
    one_hot: performs the gather with a one-hot contraction rather than a true
      gather. This is currently needed for SPMD partitioning.
  """
  num_embeddings: int
  features: int
  cast_input_dtype: Optional[DType] = None
  dtype: DType = jnp.float32
  attend_dtype: Optional[DType] = None
  embedding_init: Initializer = default_embed_init
  one_hot: bool = False
  embedding: Array = dataclasses.field(init=False)

  def setup(self):
    self.embedding = param_with_axes(
        'embedding',
        self.embedding_init, (self.num_embeddings, self.features),
        jnp.float32,
        axes=('vocab', 'embed'))

  def __call__(self, inputs: Array) -> Array:
    """Embeds the inputs along the last dimension.

    Args:
      inputs: input data, all dimensions are considered batch dimensions.

    Returns:
      Output which is embedded input data.  The output shape follows the input,
      with an additional `features` dimension appended.
    """
    if self.cast_input_dtype:
      inputs = inputs.astype(self.cast_input_dtype)
    if not jnp.issubdtype(inputs.dtype, jnp.integer):
      raise ValueError('Input type must be an integer or unsigned integer.')
    if self.one_hot:
      iota = lax.iota(jnp.int32, self.num_embeddings)
      one_hot = jnp.array(inputs[..., jnp.newaxis] == iota, dtype=self.dtype)
      output = jnp.dot(one_hot, jnp.asarray(self.embedding, self.dtype))
    else:
      output = jnp.asarray(self.embedding, self.dtype)[inputs]
      output = with_sharding_constraint(output, ('batch', 'length', 'embed'))
    return output

  def attend(self, query: Array) -> Array:
    """Attend over the embedding using a query array.

    Args:
      query: array with last dimension equal the feature depth `features` of the
        embedding.

    Returns:
      An array with final dim `num_embeddings` corresponding to the batched
      inner-product of the array of query vectors against each embedding.
      Commonly used for weight-sharing between embeddings and logit transform
      in NLP models.
    """
    dtype = self.attend_dtype if self.attend_dtype is not None else self.dtype
    return jnp.dot(query, jnp.asarray(self.embedding, dtype).T)


class RelativePositionBiases(nn.Module):
  """Adds T5-style relative positional embeddings to the attention logits.

  Attributes:
    num_buckets: Number of buckets to bucket distances between key and query
      positions into.
    max_distance: Maximum distance before everything is lumped into the last
      distance bucket.
    num_heads: Number of heads in the attention layer. Each head will get a
      different relative position weighting.
    dtype: Type of arrays through this module.
    embedding_init: initializer for relative embedding table.
  """
  num_buckets: int
  max_distance: int
  num_heads: int
  dtype: Any
  embedding_init: Callable[..., Array] = nn.linear.default_embed_init

  @staticmethod
  def _relative_position_bucket(relative_position,
                                bidirectional=True,
                                num_buckets=32,
                                max_distance=128):
    """Translate relative position to a bucket number for relative attention.

    The relative position is defined as memory_position - query_position, i.e.
    the distance in tokens from the attending position to the attended-to
    position.  If bidirectional=False, then positive relative positions are
    invalid.
    We use smaller buckets for small absolute relative_position and larger
    buckets for larger absolute relative_positions.  All relative
    positions >=max_distance  map to the same bucket.  All relative
    positions <=-max_distance map to the same bucket.  This should allow for
    more graceful generalization to longer sequences than the model has been
    trained on.

    Args:
      relative_position: an int32 array
      bidirectional: a boolean - whether the attention is bidirectional
      num_buckets: an integer
      max_distance: an integer

    Returns:
      a Tensor with the same shape as relative_position, containing int32
        values in the range [0, num_buckets)
    """
    ret = 0
    n = -relative_position
    if bidirectional:
      num_buckets //= 2
      ret += (n < 0).astype(np.int32) * num_buckets
      n = np.abs(n)
    else:
      n = np.maximum(n, 0)
    # now n is in the range [0, inf)
    max_exact = num_buckets // 2
    is_small = (n < max_exact)
    val_if_large = max_exact + (
        np.log(n.astype(np.float32) / max_exact + np.finfo(np.float32).eps) /
        np.log(max_distance / max_exact) *
        (num_buckets - max_exact)).astype(np.int32)
    val_if_large = np.minimum(val_if_large, num_buckets - 1)
    ret += np.where(is_small, n, val_if_large)
    return ret

  @nn.compact
  def __call__(self, qlen, klen, bidirectional=True, decode=False):
    """Produce relative position embedding attention biases.

    Args:
      qlen: attention query length.
      klen: attention key length.
      bidirectional: whether to allow positive memory-query relative position
        embeddings.
      decode: whether to cache relative position bias during autoregressive
        decoding.

    Returns:
      output: `(1, num_heads, q_len, k_len)` attention bias
    """
    # bidirectional embeddings don't make sense when decoding (and break cache).
    if decode and bidirectional:
      raise ValueError(
          'bidirectional RelativePositionBiases are not supported when '
          '`decode=True`.')

    # We only cache the bias if the model was already initialized, i.e. if this
    # module is called with `model.apply` and `decode = True`. We raise an error
    # if called with `model.init` and `decode = True`, since this can cache
    # incorrect positional embeddings produced by random parameters.
    is_initialized = self.has_variable('params', 'rel_embedding')
    if decode and not is_initialized:
      raise ValueError(
          'decode-mode cannot be enabled during init. use model.apply to '
          'initialize the decoding cache.')

    # Return pre-computed relative position bias in cache during decode steps.
    if decode and self.has_variable('cache', 'cached_bias'):
      cached_bias = self.get_variable('cache', 'cached_bias')
      expected_bias_shape = (1, self.num_heads, qlen, klen)
      if cached_bias.shape != expected_bias_shape:
        raise ValueError(f'The cached relative position attention bias was '
                         f'expected to have shape {expected_bias_shape} but '
                         f'instead has the shape {cached_bias.shape}.')
      return cached_bias

    # TODO(levskaya): should we be computing this w. numpy as a program
    # constant?
    context_position = np.arange(qlen, dtype=jnp.int32)[:, None]
    memory_position = np.arange(klen, dtype=jnp.int32)[None, :]
    relative_position = memory_position - context_position  # shape (qlen, klen)
    rp_bucket = self._relative_position_bucket(
        relative_position,
        bidirectional=bidirectional,
        num_buckets=self.num_buckets,
        max_distance=self.max_distance)
    relative_attention_bias = param_with_axes(
        'rel_embedding',
        self.embedding_init, (self.num_heads, self.num_buckets),
        jnp.float32,
        axes=('heads', 'relpos_buckets'))

    relative_attention_bias = jnp.asarray(relative_attention_bias, self.dtype)
    # Instead of using a slow gather, we create a leading-dimension one-hot
    # array from rp_bucket and use it to perform the gather-equivalent via a
    # contraction, i.e.:
    # (num_head, num_buckets) x (num_buckets one-hot, qlen, klen).
    # This is equivalent to relative_attention_bias[:, rp_bucket]
    bcast_iota = lax.broadcasted_iota(jnp.int32, (self.num_buckets, 1, 1), 0)
    rp_bucket_one_hot = jnp.array(
        rp_bucket[jnp.newaxis, ...] == bcast_iota, dtype=self.dtype)
    # --> shape (qlen, klen, num_heads)
    values = lax.dot_general(
        relative_attention_bias,
        rp_bucket_one_hot,
        (
            ((1,), (0,)),  # rhs, lhs contracting dims
            ((), ())))  # no batched dims
    # Add a singleton batch dimension.
    # --> shape (1, num_heads, qlen, klen)
    out = values[jnp.newaxis, ...]

    # Store computed relative position bias in cache after first calculation.
    if decode:
      _ = self.variable('cache', 'cached_bias', lambda: out)

    return out


# ------------------------------------------------------------------------------
# T5 Layernorm - no subtraction of mean or bias.
# ------------------------------------------------------------------------------
class LayerNorm(nn.Module):
  """T5 Layer normalization operating on the last axis of the input data."""
  epsilon: float = 1e-6
  dtype: Any = jnp.float32
  scale_init: Initializer = nn.initializers.ones

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    x = jnp.asarray(x, jnp.float32)
    features = x.shape[-1]
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
    scale = param_with_axes(
        'scale', self.scale_init, (features,), jnp.float32, axes=('embed',))

    scale = jnp.asarray(scale, self.dtype)
    return y * scale


# ------------------------------------------------------------------------------
# Mask-making utility functions.
# ------------------------------------------------------------------------------
def make_attention_mask(query_input: Array,
                        key_input: Array,
                        pairwise_fn: Callable = jnp.multiply,
                        extra_batch_dims: int = 0,
                        dtype: DType = jnp.float32) -> Array:
  """Mask-making helper for attention weights.

  In case of 1d inputs (i.e., `[batch, len_q]`, `[batch, len_kv]`, the
  attention weights will be `[batch, heads, len_q, len_kv]` and this
  function will produce `[batch, 1, len_q, len_kv]`.

  Args:
    query_input: a batched, flat input of query_length size
    key_input: a batched, flat input of key_length size
    pairwise_fn: broadcasting elementwise comparison function
    extra_batch_dims: number of extra batch dims to add singleton axes for, none
      by default
    dtype: mask return dtype

  Returns:
    A `[batch, 1, len_q, len_kv]` shaped mask for 1d attention.
  """
  # [batch, len_q, len_kv]
  mask = pairwise_fn(
      # [batch, len_q] -> [batch, len_q, 1]
      jnp.expand_dims(query_input, axis=-1),
      # [batch, len_q] -> [batch, 1, len_kv]
      jnp.expand_dims(key_input, axis=-2))

  # [batch, 1, len_q, len_kv]. This creates the head dim.
  mask = jnp.expand_dims(mask, axis=-3)
  mask = jnp.expand_dims(mask, axis=tuple(range(extra_batch_dims)))
  return mask.astype(dtype)


def make_causal_mask(x: Array,
                     extra_batch_dims: int = 0,
                     dtype: DType = jnp.float32) -> Array:
  """Make a causal mask for self-attention.

  In case of 1d inputs (i.e., `[batch, len]`, the self-attention weights
  will be `[batch, heads, len, len]` and this function will produce a
  causal mask of shape `[batch, 1, len, len]`.

  Note that a causal mask does not depend on the values of x; it only depends on
  the shape. If x has padding elements, they will not be treated in a special
  manner.

  Args:
    x: input array of shape `[batch, len]`
    extra_batch_dims: number of batch dims to add singleton axes for, none by
      default
    dtype: mask return dtype

  Returns:
    A `[batch, 1, len, len]` shaped causal mask for 1d attention.
  """
  idxs = jnp.broadcast_to(jnp.arange(x.shape[-1], dtype=jnp.int32), x.shape)
  return make_attention_mask(
      idxs,
      idxs,
      jnp.greater_equal,
      extra_batch_dims=extra_batch_dims,
      dtype=dtype)


def combine_masks(*masks: Optional[Array], dtype: DType = jnp.float32):
  """Combine attention masks.

  Args:
    *masks: set of attention mask arguments to combine, some can be None.
    dtype: final mask dtype

  Returns:
    Combined mask, reduced by logical and, returns None if no masks given.
  """
  masks = [m for m in masks if m is not None]
  if not masks:
    return None
  assert all(map(lambda x: x.ndim == masks[0].ndim, masks)), (
      f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
  mask, *other_masks = masks
  for other_mask in other_masks:
    mask = jnp.logical_and(mask, other_mask)
  return mask.astype(dtype)


def combine_biases(*masks: Optional[Array]):
  """Combine attention biases.

  Args:
    *masks: set of attention bias arguments to combine, some can be None.

  Returns:
    Combined mask, reduced by summation, returns None if no masks given.
  """
  masks = [m for m in masks if m is not None]
  if not masks:
    return None
  assert all(map(lambda x: x.ndim == masks[0].ndim, masks)), (
      f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
  mask, *other_masks = masks
  for other_mask in other_masks:
    mask = mask + other_mask
  return mask


def make_decoder_mask(decoder_target_tokens: Array,
                      dtype: DType,
                      decoder_causal_attention: Optional[Array] = None,
                      decoder_segment_ids: Optional[Array] = None) -> Array:
  """Compute the self-attention mask for a decoder.

  Decoder mask is formed by combining a causal mask, a padding mask and an
  optional packing mask. If decoder_causal_attention is passed, it makes the
  masking non-causal for positions that have value of 1.

  A prefix LM is applied to a dataset which has a notion of "inputs" and
  "targets", e.g., a machine translation task. The inputs and targets are
  concatenated to form a new target. `decoder_target_tokens` is the concatenated
  decoder output tokens.

  The "inputs" portion of the concatenated sequence can attend to other "inputs"
  tokens even for those at a later time steps. In order to control this
  behavior, `decoder_causal_attention` is necessary. This is a binary mask with
  a value of 1 indicating that the position belonged to "inputs" portion of the
  original dataset.

  Example:

    Suppose we have a dataset with two examples.

    ds = [{"inputs": [6, 7], "targets": [8]},
          {"inputs": [3, 4], "targets": [5]}]

    After the data preprocessing with packing, the two examples are packed into
    one example with the following three fields (some fields are skipped for
    simplicity).

       decoder_target_tokens = [[6, 7, 8, 3, 4, 5, 0]]
         decoder_segment_ids = [[1, 1, 1, 2, 2, 2, 0]]
    decoder_causal_attention = [[1, 1, 0, 1, 1, 0, 0]]

    where each array has [batch, length] shape with batch size being 1. Then,
    this function computes the following mask.

                      mask = [[[[1, 1, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0],
                                [0, 0, 0, 1, 1, 0, 0],
                                [0, 0, 0, 1, 1, 0, 0],
                                [0, 0, 0, 1, 1, 1, 0],
                                [0, 0, 0, 0, 0, 0, 0]]]]

    mask[b, 1, :, :] represents the mask for the example `b` in the batch.
    Because mask is for a self-attention layer, the mask's shape is a square of
    shape [query length, key length].

    mask[b, 1, i, j] = 1 means that the query token at position i can attend to
    the key token at position j.

  Args:
    decoder_target_tokens: decoder output tokens. [batch, length]
    dtype: dtype of the output mask.
    decoder_causal_attention: a binary mask indicating which position should
      only attend to earlier positions in the sequence. Others will attend
      bidirectionally. [batch, length]
    decoder_segment_ids: decoder segmentation info for packed examples. [batch,
      length]

  Returns:
    the combined decoder mask.
  """
  masks = []
  # The same mask is applied to all attention heads. So the head dimension is 1,
  # i.e., the mask will be broadcast along the heads dim.
  # [batch, 1, length, length]
  causal_mask = make_causal_mask(decoder_target_tokens, dtype=dtype)

  # Positions with value 1 in `decoder_causal_attneition` can attend
  # bidirectionally.
  if decoder_causal_attention is not None:
    # [batch, 1, length, length]
    inputs_mask = make_attention_mask(
        decoder_causal_attention,
        decoder_causal_attention,
        jnp.logical_and,
        dtype=dtype)
    masks.append(jnp.logical_or(causal_mask, inputs_mask).astype(dtype))
  else:
    masks.append(causal_mask)

  # Padding mask.
  masks.append(
      make_attention_mask(
          decoder_target_tokens > 0, decoder_target_tokens > 0, dtype=dtype))

  # Packing mask
  if decoder_segment_ids is not None:
    masks.append(
        make_attention_mask(
            decoder_segment_ids, decoder_segment_ids, jnp.equal, dtype=dtype))

  return combine_masks(*masks, dtype=dtype)  # pytype: disable=bad-return-type  # jax-ndarray
