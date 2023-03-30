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

"""Tests for attention classes."""

import dataclasses
from typing import Optional
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
from flax.core import freeze
from flax.linen import partitioning as nn_partitioning
import jax
from jax import random
from jax.nn import initializers
import jax.numpy as jnp
import numpy as np
from t5x.examples.scalable_t5 import layers

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()

Array = jnp.ndarray
AxisMetadata = nn_partitioning.AxisMetadata  # pylint: disable=invalid-name


class SelfAttention(layers.MultiHeadDotProductAttention):
  """Self-attention special case of multi-head dot-product attention."""

  @nn.compact
  def __call__(self,
               inputs_q: Array,
               mask: Optional[Array] = None,
               bias: Optional[Array] = None,
               deterministic: bool = False):
    return super().__call__(
        inputs_q, inputs_q, mask, bias, deterministic=deterministic)


@dataclasses.dataclass(frozen=True)
class SelfAttentionArgs:
  num_heads: int = 1
  batch_size: int = 2
  # qkv_features: int = 3
  head_dim: int = 3
  # out_features: int = 4
  q_len: int = 5
  features: int = 6
  dropout_rate: float = 0.1
  deterministic: bool = False
  decode: bool = False
  float32_logits: bool = False

  def __post_init__(self):
    # If we are doing decoding, the query length should be 1, because are doing
    # autoregressive decoding where we feed one position at a time.
    assert not self.decode or self.q_len == 1

  def init_args(self):
    return dict(
        num_heads=self.num_heads,
        head_dim=self.head_dim,
        dropout_rate=self.dropout_rate,
        float32_logits=self.float32_logits)

  def apply_args(self):
    inputs_q = jnp.ones((self.batch_size, self.q_len, self.features))
    mask = jnp.ones((self.batch_size, self.num_heads, self.q_len, self.q_len))
    bias = jnp.ones((self.batch_size, self.num_heads, self.q_len, self.q_len))
    return {
        'inputs_q': inputs_q,
        'mask': mask,
        'bias': bias,
        'deterministic': self.deterministic
    }


class AttentionTest(parameterized.TestCase):

  def test_dot_product_attention_shape(self):
    # This test only checks for shape but tries to make sure all code paths are
    # reached.
    dropout_rng = random.PRNGKey(0)
    batch_size, num_heads, q_len, kv_len, qk_depth, v_depth = 1, 2, 3, 4, 5, 6

    query = jnp.ones((batch_size, q_len, num_heads, qk_depth))
    key = jnp.ones((batch_size, kv_len, num_heads, qk_depth))
    value = jnp.ones((batch_size, kv_len, num_heads, v_depth))
    bias = jnp.ones((batch_size, num_heads, q_len, kv_len))

    args = dict(
        query=query,
        key=key,
        value=value,
        bias=bias,
        dropout_rng=dropout_rng,
        dropout_rate=0.5,
        deterministic=False,
    )

    output = layers.dot_product_attention(**args)
    self.assertEqual(output.shape, (batch_size, q_len, num_heads, v_depth))

  def test_make_attention_mask_multiply_pairwise_fn(self):
    decoder_target_tokens = jnp.array([[7, 0, 0], [8, 5, 0]])
    attention_mask = layers.make_attention_mask(
        decoder_target_tokens > 0, decoder_target_tokens > 0, dtype=jnp.int32)
    expected0 = jnp.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    expected1 = jnp.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    self.assertEqual(attention_mask.shape, (2, 1, 3, 3))
    np.testing.assert_array_equal(attention_mask[0, 0], expected0)
    np.testing.assert_array_equal(attention_mask[1, 0], expected1)

  def test_make_attention_mask_equal_pairwise_fn(self):
    segment_ids = jnp.array([[1, 1, 2, 2, 2, 0], [1, 1, 1, 2, 0, 0]])
    attention_mask = layers.make_attention_mask(
        segment_ids, segment_ids, pairwise_fn=jnp.equal, dtype=jnp.int32)
    # Padding is not treated in a special way. So they need to be zeroed out
    # separately.
    expected0 = jnp.array([[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0],
                           [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 1, 0],
                           [0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 1]])
    expected1 = jnp.array([[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0],
                           [1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1]])
    self.assertEqual(attention_mask.shape, (2, 1, 6, 6))
    np.testing.assert_array_equal(attention_mask[0, 0], expected0)
    np.testing.assert_array_equal(attention_mask[1, 0], expected1)

  def test_make_causal_mask_with_padding(self):
    x = jnp.array([[7, 0, 0], [8, 5, 0]])
    y = layers.make_causal_mask(x)
    self.assertEqual(y.shape, (2, 1, 3, 3))
    # Padding is not treated in a special way. So they need to be zeroed out
    # separately.
    expected_y = jnp.array([[[1., 0., 0.], [1., 1., 0.], [1., 1., 1.]]],
                           jnp.float32)
    np.testing.assert_allclose(y[0], expected_y)
    np.testing.assert_allclose(y[1], expected_y)

  def test_make_causal_mask_extra_batch_dims(self):
    x = jnp.ones((3, 3, 5))
    y = layers.make_causal_mask(x, extra_batch_dims=2)
    self.assertEqual(y.shape, (1, 1, 3, 3, 1, 5, 5))

  def test_make_causal_mask(self):
    x = jnp.ones((1, 3))
    y = layers.make_causal_mask(x)
    self.assertEqual(y.shape, (1, 1, 3, 3))
    expected_y = jnp.array([[[[1., 0., 0.], [1., 1., 0.], [1., 1., 1.]]]],
                           jnp.float32)
    np.testing.assert_allclose(y, expected_y)

  def test_combine_masks(self):
    masks = [
        jnp.array([0, 1, 0, 1], jnp.float32), None,
        jnp.array([1, 1, 1, 1], jnp.float32),
        jnp.array([1, 1, 1, 0], jnp.float32)
    ]
    y = layers.combine_masks(*masks)
    np.testing.assert_allclose(y, jnp.array([0, 1, 0, 0], jnp.float32))

  def test_combine_biases(self):
    masks = [
        jnp.array([0, 1, 0, 1], jnp.float32), None,
        jnp.array([0, 1, 1, 1], jnp.float32),
        jnp.array([0, 1, 1, 0], jnp.float32)
    ]
    y = layers.combine_biases(*masks)
    np.testing.assert_allclose(y, jnp.array([0, 3, 2, 2], jnp.float32))

  def test_make_decoder_mask_lm_unpacked(self):
    decoder_target_tokens = jnp.array([6, 7, 3, 0])
    mask = layers.make_decoder_mask(
        decoder_target_tokens=decoder_target_tokens, dtype=jnp.float32)
    expected_mask = jnp.array([[[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0],
                                [0, 0, 0, 0]]])
    np.testing.assert_array_equal(mask, expected_mask)

  def test_make_decoder_mask_lm_packed(self):
    decoder_target_tokens = jnp.array([[6, 7, 3, 4, 5, 0]])
    decoder_segment_ids = jnp.array([[1, 1, 1, 2, 2, 0]])
    mask = layers.make_decoder_mask(
        decoder_target_tokens=decoder_target_tokens,
        dtype=jnp.float32,
        decoder_segment_ids=decoder_segment_ids)
    expected_mask = jnp.array([[[[1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0],
                                 [1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0]]]])
    np.testing.assert_array_equal(mask, expected_mask)

  def test_make_decoder_mask_prefix_lm_unpacked(self):
    decoder_target_tokens = jnp.array([[5, 6, 7, 3, 4, 0]])
    decoder_causal_attention = jnp.array([[1, 1, 1, 0, 0, 0]])
    mask = layers.make_decoder_mask(
        decoder_target_tokens=decoder_target_tokens,
        dtype=jnp.float32,
        decoder_causal_attention=decoder_causal_attention)
    expected_mask = jnp.array(
        [[[[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0],
           [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0]]]],
        dtype=jnp.float32)
    np.testing.assert_array_equal(mask, expected_mask)

  def test_make_decoder_mask_prefix_lm_packed(self):
    decoder_target_tokens = jnp.array([[5, 6, 7, 8, 3, 4, 0]])
    decoder_segment_ids = jnp.array([[1, 1, 1, 2, 2, 2, 0]])
    decoder_causal_attention = jnp.array([[1, 1, 0, 1, 1, 0, 0]])
    mask = layers.make_decoder_mask(
        decoder_target_tokens=decoder_target_tokens,
        dtype=jnp.float32,
        decoder_causal_attention=decoder_causal_attention,
        decoder_segment_ids=decoder_segment_ids)
    expected_mask = jnp.array([[[[1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0],
                                 [1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0],
                                 [0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 0],
                                 [0, 0, 0, 0, 0, 0, 0]]]])
    np.testing.assert_array_equal(mask, expected_mask)

  def test_make_decoder_mask_prefix_lm_unpacked_multiple_elements(self):
    decoder_target_tokens = jnp.array([[6, 7, 3, 0], [4, 5, 0, 0]])
    decoder_causal_attention = jnp.array([[1, 1, 0, 0], [1, 0, 0, 0]])
    mask = layers.make_decoder_mask(
        decoder_target_tokens=decoder_target_tokens,
        dtype=jnp.float32,
        decoder_causal_attention=decoder_causal_attention)
    expected_mask0 = jnp.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0],
                                [0, 0, 0, 0]])
    expected_mask1 = jnp.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0]])
    self.assertEqual(mask.shape, (2, 1, 4, 4))
    np.testing.assert_array_equal(mask[0, 0], expected_mask0)
    np.testing.assert_array_equal(mask[1, 0], expected_mask1)

  def test_make_decoder_mask_composite_causal_attention(self):
    decoder_target_tokens = jnp.array([[6, 7, 3, 4, 8, 9, 0]])
    decoder_causal_attention = jnp.array([[1, 1, 0, 0, 1, 1, 0]])
    mask = layers.make_decoder_mask(
        decoder_target_tokens=decoder_target_tokens,
        dtype=jnp.float32,
        decoder_causal_attention=decoder_causal_attention)
    expected_mask0 = jnp.array([[1, 1, 0, 0, 1, 1, 0], [1, 1, 0, 0, 1, 1, 0],
                                [1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0],
                                [1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 0],
                                [0, 0, 0, 0, 0, 0, 0]])

    self.assertEqual(mask.shape, (1, 1, 7, 7))
    np.testing.assert_array_equal(mask[0, 0], expected_mask0)

  def test_make_decoder_mask_composite_causal_attention_packed(self):
    decoder_target_tokens = jnp.array([[6, 7, 3, 4, 8, 9, 2, 3, 4]])
    decoder_segment_ids = jnp.array([[1, 1, 1, 1, 1, 1, 2, 2, 2]])
    decoder_causal_attention = jnp.array([[1, 1, 0, 0, 1, 1, 1, 1, 0]])
    mask = layers.make_decoder_mask(
        decoder_target_tokens=decoder_target_tokens,
        dtype=jnp.float32,
        decoder_causal_attention=decoder_causal_attention,
        decoder_segment_ids=decoder_segment_ids)
    expected_mask0 = jnp.array([[1, 1, 0, 0, 1, 1, 0, 0, 0],
                                [1, 1, 0, 0, 1, 1, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 1, 0, 0, 0, 0, 0],
                                [1, 1, 1, 1, 1, 1, 0, 0, 0],
                                [1, 1, 1, 1, 1, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1, 1, 0],
                                [0, 0, 0, 0, 0, 0, 1, 1, 0],
                                [0, 0, 0, 0, 0, 0, 1, 1, 1]])

    self.assertEqual(mask.shape, (1, 1, 9, 9))
    np.testing.assert_array_equal(mask[0, 0], expected_mask0)

  @parameterized.parameters({'f': 20}, {'f': 22})
  def test_multihead_dot_product_attention(self, f):
    # b: batch, f: emb_dim, q: q_len, k: kv_len, h: num_head, d: head_dim
    b, q, h, d, k = 2, 3, 4, 5, 6

    base_args = SelfAttentionArgs(num_heads=h, head_dim=d, dropout_rate=0)
    args = base_args.init_args()

    np.random.seed(0)
    inputs_q = np.random.randn(b, q, f)
    inputs_kv = np.random.randn(b, k, f)

    # Projection: [b, q, f] -> [b, q, h, d]
    # So the kernels have to be [f, h, d]
    query_kernel = np.random.randn(f, h, d)
    key_kernel = np.random.randn(f, h, d)
    value_kernel = np.random.randn(f, h, d)
    # `out` calculation: [b, q, h, d] -> [b, q, f]
    # So kernel has to be [h, d, f]
    out_kernel = np.random.randn(h, d, f)

    params = {
        'query': {
            'kernel': query_kernel
        },
        'key': {
            'kernel': key_kernel
        },
        'value': {
            'kernel': value_kernel
        },
        'out': {
            'kernel': out_kernel
        }
    }
    y = layers.MultiHeadDotProductAttention(**args).apply(
        {'params': freeze(params)}, inputs_q, inputs_kv)

    query = np.einsum('bqf,fhd->bqhd', inputs_q, query_kernel)
    key = np.einsum('bkf,fhd->bkhd', inputs_kv, key_kernel)
    value = np.einsum('bkf,fhd->bkhd', inputs_kv, value_kernel)
    logits = np.einsum('bqhd,bkhd->bhqk', query, key)
    weights = nn.softmax(logits, axis=-1)
    combined_value = np.einsum('bhqk,bkhd->bqhd', weights, value)
    y_expected = np.einsum('bqhd,hdf->bqf', combined_value, out_kernel)
    np.testing.assert_allclose(y, y_expected, rtol=1e-5, atol=1e-5)

  def test_multihead_dot_product_attention_caching(self):
    # b: batch, f: qkv_features, k: kv_len, h: num_head, d: head_dim
    b, h, d, k = 2, 3, 4, 5
    f = h * d

    base_args = SelfAttentionArgs(num_heads=h, head_dim=d, dropout_rate=0)
    args = base_args.init_args()

    cache = {
        'cached_key': np.zeros((b, h, d, k)),
        'cached_value': np.zeros((b, h, d, k)),
        'cache_index': np.array(0)
    }
    inputs_q = np.random.randn(b, 1, f)
    inputs_kv = np.random.randn(b, 1, f)

    # Mock dense general such that q, k, v projections are replaced by simple
    # reshaping.
    def mock_dense_general(self, x, **kwargs):  # pylint: disable=unused-argument
      return x.reshape(b, -1, h, d)

    with mock.patch.object(
        layers.DenseGeneral, '__call__', new=mock_dense_general):
      _, mutated = layers.MultiHeadDotProductAttention(**args).apply(
          {'cache': freeze(cache)},
          inputs_q,
          inputs_kv,
          decode=True,
          mutable=['cache'])
      updated_cache = mutated['cache']

    # Perform the same mocked projection to generate the expected cache.
    # (key|value): [b, 1, h, d]
    key = mock_dense_general(None, inputs_kv)
    value = mock_dense_general(None, inputs_kv)

    # cached_(key|value): [b, h, d, k]
    cache['cached_key'][:, :, :, 0] = key[:, 0, :, :]
    cache['cached_value'][:, :, :, 0] = value[:, 0, :, :]
    cache['cache_index'] = np.array(1)
    for name, array in cache.items():
      np.testing.assert_allclose(array, updated_cache[name])

  def test_dot_product_attention(self):
    # b: batch, f: emb_dim, q: q_len, k: kv_len, h: num_head, d: head_dim
    b, q, h, d, k = 2, 3, 4, 5, 6
    np.random.seed(0)
    query = np.random.randn(b, q, h, d)
    key = np.random.randn(b, k, h, d)
    value = np.random.randn(b, k, h, d)
    bias = np.random.randn(b, h, q, k)
    attn_out = layers.dot_product_attention(query, key, value, bias=bias)
    logits = np.einsum('bqhd,bkhd->bhqk', query, key)
    weights = jax.nn.softmax(logits + bias, axis=-1)
    expected = np.einsum('bhqk,bkhd->bqhd', weights, value)
    np.testing.assert_allclose(attn_out, expected, atol=1e-6)


class EmbeddingTest(parameterized.TestCase):

  def test_embedder_raises_exception_for_incorrect_input_type(self):
    """Tests that inputs are integers and that an exception is raised if not."""
    embed = layers.Embed(num_embeddings=10, features=5)
    inputs = np.expand_dims(np.arange(5, dtype=np.int64), 1)
    variables = embed.init(jax.random.PRNGKey(0), inputs)
    bad_inputs = inputs.astype(np.float32)
    with self.assertRaisesRegex(
        ValueError, 'Input type must be an integer or unsigned integer.'):
      _ = embed.apply(variables, bad_inputs)

  @parameterized.named_parameters(
      {
          'testcase_name': 'with_ones',
          'init_fn': jax.nn.initializers.ones,
          'num_embeddings': 10,
          'features': 5,
          'matrix_sum': 5 * 10,
      }, {
          'testcase_name': 'with_zeros',
          'init_fn': jax.nn.initializers.zeros,
          'num_embeddings': 10,
          'features': 5,
          'matrix_sum': 0,
      })
  def test_embedding_initializes_correctly(self, init_fn, num_embeddings,
                                           features, matrix_sum):
    """Tests if the Embed class initializes with the requested initializer."""
    embed = layers.Embed(
        num_embeddings=num_embeddings,
        features=features,
        embedding_init=init_fn)
    inputs = np.expand_dims(np.arange(5, dtype=np.int64), 1)
    variables = embed.init(jax.random.PRNGKey(0), inputs)
    embedding_matrix = variables['params']['embedding']
    self.assertEqual(int(np.sum(embedding_matrix)), matrix_sum)

  def test_embedding_matrix_shape(self):
    """Tests that the embedding matrix has the right shape."""
    num_embeddings = 10
    features = 5
    embed = layers.Embed(num_embeddings=num_embeddings, features=features)
    inputs = np.expand_dims(np.arange(features, dtype=np.int64), 1)
    variables = embed.init(jax.random.PRNGKey(0), inputs)
    embedding_matrix = variables['params']['embedding']
    self.assertEqual((num_embeddings, features), embedding_matrix.shape)

  def test_embedding_attend(self):
    """Tests that attending with ones returns sum of embedding vectors."""
    features = 5
    embed = layers.Embed(num_embeddings=10, features=features)
    inputs = np.array([[1]], dtype=np.int64)
    variables = embed.init(jax.random.PRNGKey(0), inputs)
    query = np.ones(features, dtype=np.float32)
    result = embed.apply(variables, query, method=embed.attend)
    expected = np.sum(variables['params']['embedding'], -1)
    np.testing.assert_array_almost_equal(result, expected)


class DenseTest(parameterized.TestCase):

  def test_dense_general_no_bias(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 3))
    model = layers.DenseGeneral(
        features=4,
        kernel_init=lambda k, s, d, ai, ao: initializers.ones(k, s, d),
    )
    y, _ = model.init_with_output(rng, x)
    self.assertEqual(y.shape, (1, 4))
    np.testing.assert_allclose(y, np.full((1, 4), 3.))

  def test_dense_general_two_features(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 3))
    model = layers.DenseGeneral(
        features=(2, 2),
        kernel_init=lambda k, s, d, ai, ao: initializers.ones(k, s, d),
    )
    y, _ = model.init_with_output(rng, x)
    # We transform the last input dimension to two output dimensions (2, 2).
    np.testing.assert_allclose(y, np.full((1, 2, 2), 3.))

  def test_dense_general_two_axes(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 2, 2))
    model = layers.DenseGeneral(
        features=3,
        axis=(-2, 2),  # Note: this is the same as (1, 2).
        kernel_init=lambda k, s, d, ai, ao: initializers.ones(k, s, d),
    )
    y, _ = model.init_with_output(rng, x)
    # We transform the last two input dimensions (2, 2) to one output dimension.
    np.testing.assert_allclose(y, np.full((1, 3), 4.))

  def test_mlp_same_out_dim(self):
    module = layers.MlpBlock(
        intermediate_dim=4,
        activations=('relu',),
        kernel_init=layers.nd_dense_init(1.0, 'fan_avg', 'uniform'),
        dtype=jnp.float32,
    )
    inputs = np.array(
        [
            # Batch 1.
            [[1, 1], [1, 1], [1, 2]],
            # Batch 2.
            [[2, 2], [3, 1], [2, 2]],
        ],
        dtype=np.float32)
    params = module.init(random.PRNGKey(0), inputs, deterministic=True)
    self.assertEqual(
        jax.tree_map(lambda a: a.tolist(), params), {
            'params': {
                'wi': {
                    'kernel': [[
                        -0.8675811290740967, 0.08417510986328125,
                        0.022586345672607422, -0.9124102592468262
                    ],
                               [
                                   -0.19464373588562012, 0.49809837341308594,
                                   0.7808468341827393, 0.9267289638519287
                               ]],
                },
                'wo': {
                    'kernel': [[0.01154780387878418, 0.1397249698638916],
                               [0.974980354309082, 0.5903260707855225],
                               [-0.05997943878173828, 0.616570234298706],
                               [0.2934272289276123, 0.8181164264678955]],
                },
            },
            'params_axes': {
                'wi': {
                    'kernel_axes': AxisMetadata(names=('embed', 'mlp')),
                },
                'wo': {
                    'kernel_axes': AxisMetadata(names=('mlp', 'embed')),
                },
            },
        })
    result = module.apply(params, inputs, deterministic=True)
    np.testing.assert_allclose(
        result.tolist(),
        [[[0.5237172245979309, 0.8508185744285583],
          [0.5237172245979309, 0.8508185744285583],
          [1.2344461679458618, 2.3844780921936035]],
         [[1.0474344491958618, 1.7016371488571167],
          [0.6809444427490234, 0.9663378596305847],
          [1.0474344491958618, 1.7016371488571167]]],
        rtol=1e-6,
    )


class RelativePositionBiasesTest(absltest.TestCase):

  def setUp(self):
    self.num_heads = 3
    self.query_len = 5
    self.key_len = 7
    self.relative_attention = layers.RelativePositionBiases(
        num_buckets=12,
        max_distance=10,
        num_heads=3,
        dtype=jnp.float32,
    )
    super(RelativePositionBiasesTest, self).setUp()

  def test_relative_attention_bidirectional_params(self):
    """Tests that bidirectional relative position biases have expected params."""
    params = self.relative_attention.init(
        random.PRNGKey(0), self.query_len, self.key_len, bidirectional=True)
    param_shapes = jax.tree_map(lambda x: x.shape, params)
    self.assertEqual(
        param_shapes, {
            'params': {
                'rel_embedding': (3, 12),
            },
            'params_axes': {
                'rel_embedding_axes':
                    AxisMetadata(names=('heads', 'relpos_buckets')),
            }
        })

  def test_regression_relative_attention_bidirectional_values(self):
    """Tests that bidirectional relative position biases match expected values.

    See top docstring note on matching T5X behavior for these regression tests.
    """
    outputs, unused_params = self.relative_attention.init_with_output(
        random.PRNGKey(0), self.query_len, self.key_len, bidirectional=True)
    self.assertEqual(outputs.shape,
                     (1, self.num_heads, self.query_len, self.key_len))
    self.assertAlmostEqual(outputs[0, 0, 0, 0], 0.55764728, places=5)
    self.assertAlmostEqual(outputs[0, 1, 2, 1], -0.10935841, places=5)
    self.assertAlmostEqual(outputs[0, 1, 4, 6], 0.14510104, places=5)
    self.assertAlmostEqual(outputs[0, 2, 4, 6], -0.36783996, places=5)

  def test_relative_attention_unidirectional_params(self):
    """Tests that unidirectional relative position biases have expected params."""
    params = self.relative_attention.init(
        random.PRNGKey(0), self.query_len, self.key_len, bidirectional=False)
    param_shapes = jax.tree_map(lambda x: x.shape, params)
    self.assertEqual(
        param_shapes, {
            'params': {
                'rel_embedding': (3, 12),
            },
            'params_axes': {
                'rel_embedding_axes':
                    AxisMetadata(names=('heads', 'relpos_buckets')),
            }
        })

  def test_regression_relative_attention_unidirectional_values(self):
    """Tests that unidirectional relative position biases match expected values.

    See top docstring note on matching T5X behavior for these regression tests.
    """
    outputs, unused_params = self.relative_attention.init_with_output(
        random.PRNGKey(0), self.query_len, self.key_len, bidirectional=False)
    self.assertEqual(outputs.shape,
                     (1, self.num_heads, self.query_len, self.key_len))
    self.assertAlmostEqual(outputs[0, 0, 0, 0], 0.55764728, places=5)
    self.assertAlmostEqual(outputs[0, 1, 2, 1], -0.10935841, places=5)
    self.assertAlmostEqual(outputs[0, 1, 4, 6], -0.13101986, places=5)
    self.assertAlmostEqual(outputs[0, 2, 4, 6], 0.39296466, places=5)


if __name__ == '__main__':
  absltest.main()
