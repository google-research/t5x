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

"""Tests for t5x.models."""

import functools
from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import flax
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import t5.data.tasks  # pylint:disable=unused-import
from t5x import decoding
from t5x import models
from t5x import partitioning
from t5x import test_utils
from t5x import trainer as trainer_lib
from t5x import utils
import tensorflow as tf

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()

PartitionSpec = partitioning.PartitionSpec


class ModelsTest(parameterized.TestCase):

  def test_remove_prefix(self):
    sequences = np.array([[1, 2, 3, 4, 5, 6, 7, 0], [6, 7, 8, 9, 10, 11, 0, 0]])
    prefix_lengths = np.array([2, 4])
    expected = [[3, 4, 5, 6, 7, 0, 0, 0], [10, 11, 0, 0, 0, 0, 0, 0]]
    remove_prefix = jax.jit(models.remove_prefix)
    actual = remove_prefix(sequences, prefix_lengths)
    np.testing.assert_array_equal(actual, expected)

  def test_remove_prefix_zero_len_prefix(self):
    sequences = np.array([[1, 2, 3, 4, 5, 6, 7, 0], [6, 7, 8, 9, 10, 11, 0, 0]])
    prefix_lengths = np.array([0, 0])
    remove_prefix = jax.jit(models.remove_prefix)
    actual = remove_prefix(sequences, prefix_lengths)
    # The expected output is the original sequences.
    np.testing.assert_array_equal(actual, sequences)

  def test_count_packed_examples(self):
    segment_ids_1 = np.array([[1, 1, 2, 2, 0, 0], [1, 1, 1, 1, 1, 1]])
    actual_1 = models.count_packed_examples(segment_ids_1)
    # `actual_1` is DeviceArray(3, dtype=int32).
    np.testing.assert_array_equal(actual_1, [3])

    segment_ids_2 = np.array([[1, 1, 3, 3, 0, 0], [2, 2, 2, 2, 2, 2],
                              [2, 7, 7, 7, 7, 0]])
    actual_2 = models.count_packed_examples(segment_ids_2)
    # `actual_2` is DeviceArray(5, dtype=int32).
    np.testing.assert_array_equal(actual_2, [5])


BATCH_SIZE, ENCODER_LEN, MAX_DECODE_LEN, EMBED_DIM = 2, 3, 4, 5


class EncoderDecoderModelTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='no_types',
          shapes={
              'encoder_input_tokens': [1, 512],
              'decoder_input_tokens': [1, 62]
          },
          types=None),
      dict(
          testcase_name='int32',
          shapes={
              'encoder_input_tokens': [1, 512],
              'decoder_input_tokens': [1, 62]
          },
          types={
              'encoder_input_tokens': jnp.int32,
              'decoder_input_tokens': jnp.int32
          }),
      dict(
          testcase_name='float32',
          shapes={
              'encoder_input_tokens': [1, 512],
              'decoder_input_tokens': [1, 62],
              'encoder_positions': [1, 512],
              'decoder_positions': [1, 62],
          },
          types={
              'encoder_input_tokens': jnp.int32,
              'decoder_input_tokens': jnp.int32,
              'encoder_positions': jnp.int32,
              'decoder_positions': jnp.int32
          }),
      dict(
          testcase_name='float32_segment_ids',
          shapes={
              'encoder_input_tokens': [1, 512],
              'decoder_input_tokens': [1, 62],
              'encoder_segment_ids': [1, 512],
              'decoder_segment_ids': [1, 62],
          },
          types={
              'encoder_input_tokens': jnp.int32,
              'decoder_input_tokens': jnp.int32,
              'encoder_segment_ids': jnp.int32,
              'decoder_segment_ids': jnp.int32
          }),
  )
  def test_get_initial_variables_shapes_and_types(self, shapes, types):
    mock_transformer = mock.Mock()
    mock_transformer.init.return_value = {'params': {}}
    mock_optimizer_def = mock.Mock()
    rng = mock.Mock()

    def mock_init(self):
      self.module = mock_transformer
      self.optimizer_def = mock_optimizer_def

    with mock.patch.object(
        models.EncoderDecoderModel, '__init__', new=mock_init):
      model = models.EncoderDecoderModel()
      model.get_initial_variables(rng, shapes, types)

    if types is None:
      encoder_input = jnp.ones(
          shapes['encoder_input_tokens'], dtype=jnp.float32)
      decoder_input = jnp.ones(
          shapes['decoder_input_tokens'], dtype=jnp.float32)
    else:
      encoder_input = jnp.ones(
          shapes['encoder_input_tokens'], dtype=types['encoder_input_tokens'])
      decoder_input = jnp.ones(
          shapes['decoder_input_tokens'], dtype=types['decoder_input_tokens'])

    # Using `.assert_called_once_with` doesn't work because the simple
    # comparison it does for the array arguments fail (truth value of an array
    # is ambiguous).
    called_with = mock_transformer.init.call_args
    self.assertEqual(called_with[0][0], rng)
    np.testing.assert_allclose(called_with[0][1], encoder_input)
    np.testing.assert_allclose(called_with[0][2], decoder_input)
    np.testing.assert_allclose(called_with[0][3], decoder_input)

    if 'encoder_positions' in shapes:
      encoder_positions = jnp.ones(
          shapes['encoder_positions'], dtype=types['encoder_positions'])
      np.testing.assert_allclose(called_with[1]['encoder_positions'],
                                 encoder_positions)
    else:
      self.assertIsNone(called_with[1]['encoder_positions'])
    if 'decoder_positions' in shapes:
      decoder_positions = jnp.ones(
          shapes['decoder_positions'], dtype=types['decoder_positions'])
      np.testing.assert_allclose(called_with[1]['decoder_positions'],
                                 decoder_positions)
    else:
      self.assertIsNone(called_with[1]['decoder_positions'])

    if 'encoder_segment_ids' in shapes:
      encoder_positions = jnp.ones(
          shapes['encoder_segment_ids'], dtype=types['encoder_segment_ids'])
      np.testing.assert_allclose(called_with[1]['encoder_segment_ids'],
                                 encoder_positions)
    else:
      self.assertIsNone(called_with[1]['encoder_segment_ids'])
    if 'decoder_segment_ids' in shapes:
      decoder_segment_ids = jnp.ones(
          shapes['decoder_segment_ids'], dtype=types['decoder_segment_ids'])
      np.testing.assert_allclose(called_with[1]['decoder_segment_ids'],
                                 decoder_segment_ids)
    else:
      self.assertIsNone(called_with[1]['decoder_segment_ids'])

    self.assertFalse(called_with[1]['decode'])
    self.assertFalse(called_with[1]['enable_dropout'])

  @parameterized.named_parameters(
      dict(testcase_name='no_force_decoding', prompt_with_targets=False),
      dict(testcase_name='force_decoding', prompt_with_targets=True),
  )
  def test_prompt_with_targets(self, prompt_with_targets):
    batch_size, encoder_len, max_decode_len, emb_dim = 2, 3, 4, 5
    batch = {
        'encoder_input_tokens':
            np.zeros((batch_size, encoder_len), dtype=np.int32),
        'decoder_input_tokens':
            np.full([batch_size, max_decode_len], 2, dtype=np.int32)
    }

    # These dummy logits represent the probability distribution where all the
    # probability mass is in one item (i.e., degenerate distribution). For
    # batch element 0, it is vocabulary index 3.
    # We test `_predict_step` to avoid having to define a task and its
    # vocabulary.
    dummy_logits = jnp.expand_dims(
        jnp.array([[-1e7, -1e7, -1e7, 0, -1e7], [-1e7, -1e7, -1e7, -1e7, 0]]),
        axis=1)

    mock_decode_fn = mock.Mock()
    mock_decode_fn.return_value = (np.full([batch_size, max_decode_len, 1],
                                           3,
                                           dtype=np.int32),
                                   np.full([batch_size, 1],
                                           1.0,
                                           dtype=np.float32))

    class MockModule:

      def __init__(self):
        self.dtype = jnp.float32

      def apply(self, *args, method=None, **kwargs):
        del args, kwargs
        if method is None:  # use for module.`__call__`
          return (dummy_logits, {'cache': {}})
        else:
          return method()

      def encode(self):
        return jnp.zeros((batch_size, encoder_len, emb_dim))

      def decode(self):
        return (dummy_logits, {'cache': {}})

    def mock_init(self):
      self.module = MockModule()
      self.module.scan_layers = False
      self._input_vocabulary = mock.Mock(eos_id=1)
      self._output_vocabulary = mock.Mock(eos_id=1)
      self._decode_fn = mock_decode_fn

    with mock.patch.object(
        models.EncoderDecoderModel, '__init__', new=mock_init):
      model = models.EncoderDecoderModel()

    model.predict_batch_with_aux({},
                                 batch,
                                 prompt_with_targets=prompt_with_targets)

    if prompt_with_targets:
      expected_inputs = batch['decoder_input_tokens']
    else:
      expected_inputs = np.zeros([batch_size, max_decode_len], dtype=np.int32)

    assert mock_decode_fn.call_count == 1
    # Look at the kwargs call list for inputs, assert_called_with doesn't
    # work well with np.array comparison.
    np.testing.assert_array_equal(mock_decode_fn.mock_calls[0][2]['inputs'],
                                  expected_inputs)

  def test_predict_batch_loop_and_caches_are_equal(self):
    vocab_size = 50
    lengths = np.array([[2], [3]])
    batch_size, beam_size, encoder_len, max_decode_len = 2, 2, 3, 7
    batch = {
        'encoder_input_tokens':
            np.zeros((batch_size, encoder_len), dtype=np.int32),
        'decoder_target_tokens':
            np.zeros((batch_size, encoder_len), dtype=np.int32),
        'decoder_input_tokens':
            np.concatenate(
                [
                    np.expand_dims(
                        np.concatenate(
                            [[0],
                             np.arange(9, 9 + lengths[0][0], dtype=np.int32),
                             np.zeros((max_decode_len - lengths[0][0] - 1),
                                      dtype=np.int32)]),
                        axis=0),  # First element
                    np.expand_dims(
                        np.concatenate(
                            [[0],
                             np.arange(3, 3 + lengths[1][0], dtype=np.int32),
                             np.zeros((max_decode_len - lengths[1][0] - 1),
                                      dtype=np.int32)]),
                        axis=0)  # Second element
                ],
                axis=0),
    }

    model = test_utils.get_t5_test_model(vocab_size=50)
    module = model.module
    params = module.init(
        jax.random.PRNGKey(0),
        jnp.ones((batch_size, encoder_len)),
        jnp.ones((batch_size, max_decode_len)),
        jnp.ones((batch_size, max_decode_len)),
        enable_dropout=False)['params']

    def mock_init(self):
      self.module = module
      # Set the EOS token to be larger then the vocabulary size. This forces the
      # model to decode all the way to `max_decode_length`, allowing us to test
      # behavior when one element reaches the end before the others.
      self._output_vocabulary = mock.Mock(eos_id=vocab_size + 12)
      self._decode_fn = decoding.beam_search

    with mock.patch.object(
        models.EncoderDecoderModel, '__init__', new=mock_init):
      model = models.EncoderDecoderModel()

    with mock.patch.object(
        model, '_compute_logits_from_slice',
        autospec=True) as tokens_to_logits_mock:
      # Make the side effect of the mock, call the method on the class, with the
      # instance partialed in as `self`. This lets us call the actual code,
      # while recording the inputs, without an infinite loop you would get
      # calling `instance.method`
      tokens_to_logits_mock.side_effect = functools.partial(
          models.EncoderDecoderModel._compute_logits_from_slice, model)
      # Disable jit, so that the `lax.while_loop` isn't traced, as the
      # collection of tracers in the mock call_args would generally trigger a
      # tracer leak error.
      with jax.disable_jit():
        _ = model.predict_batch_with_aux(
            params, batch, prompt_with_targets=True, num_decodes=2)

    # Collect all the input tokens to our tokens_to_logits function
    all_inputs = []
    all_cache_keys = []  # Collect all the cache keys
    all_cache_values = []  # Collect all the cache values
    # Currently force decoding generates logits at every step. We should have
    # `max_decode_length` calls to our tokens -> logits func.
    self.assertLen(tokens_to_logits_mock.call_args_list, max_decode_len)
    for tokens_call in tokens_to_logits_mock.call_args_list:
      decoding_state: decoding.DecodingState = tokens_call[0][0]
      # Inputs: [B, 1]
      inputs = decoding_state.cur_token
      cache = decoding_state.cache
      cache = flax.core.unfreeze(cache)
      # Cache: [B * Be, 1] * #Layers
      cache_keys = [
          v for k, v in traverse_util.flatten_dict(cache).items()
          if k[-1] == 'cached_key'
      ]
      cache_values = [
          v for k, v in traverse_util.flatten_dict(cache).items()
          if k[-1] == 'cached_value'
      ]
      all_inputs.append(inputs)
      all_cache_keys.append(cache_keys)
      all_cache_values.append(cache_values)
    # Convert inputs to a single block [B, DL, Be]
    all_inputs = np.concatenate(all_inputs, axis=1)
    # Convert caches into a single block per layer [B * Be, DL] * L
    all_cache_keys = [np.stack(c, axis=1) for c in zip(*all_cache_keys)]
    all_cache_values = [np.stack(c, axis=1) for c in zip(*all_cache_values)]

    # Make sure that for each batch, the cache for each beam is identical when
    # prompt is being forced.
    for b in range(batch_size):
      for i, input_token in enumerate(all_inputs[b * beam_size]):
        if i < lengths[b]:
          self.assertEqual(input_token, batch['decoder_input_tokens'][b][i])
          # For all layers.
          for cache_keys in all_cache_keys:
            np.testing.assert_array_equal(cache_keys[b * beam_size][i],
                                          cache_keys[b * beam_size + 1][i])
          for cache_values in all_cache_values:
            np.testing.assert_array_equal(cache_values[b * beam_size][i],
                                          cache_values[b * beam_size + 1][i])

  def test_score_batch(self):
    encoder_input_tokens = jnp.ones((2, 3))
    # For this test, decoder input and target tokens are dummy values.
    decoder_input_tokens = jnp.array([[1, 2, 1, 0], [0, 1, 0, 2]])
    decoder_target_tokens = jnp.array([[1, 2, 1, 0], [0, 1, 0, 2]])
    decoder_loss_weights = jnp.array([[1, 1, 1, 0], [0, 1, 0, 1]])
    logits = jnp.arange(0, 24).reshape((2, 4, 3))
    params = {'foo': jnp.zeros(3)}

    mock_transformer = mock.Mock()
    mock_transformer.apply.return_value = logits
    mock_transformer.dtype = jnp.float32

    batch = {
        'encoder_input_tokens': encoder_input_tokens,
        'decoder_input_tokens': decoder_input_tokens,
        'decoder_target_tokens': decoder_target_tokens,
        'decoder_loss_weights': decoder_loss_weights
    }

    def mock_init(self):
      self.module = mock_transformer

    with mock.patch.object(
        models.EncoderDecoderModel, '__init__', new=mock_init):
      model = models.EncoderDecoderModel()
      res = model.score_batch(params, batch)

    mock_transformer.apply.assert_called_with({'params': params},
                                              encoder_input_tokens,
                                              decoder_input_tokens,
                                              decoder_target_tokens,
                                              encoder_segment_ids=None,
                                              decoder_segment_ids=None,
                                              encoder_positions=None,
                                              decoder_positions=None,
                                              decode=False,
                                              enable_dropout=False,
                                              rngs=None,
                                              mutable=False)
    np.testing.assert_allclose(res, [-3.222973, -1.815315], rtol=1e-4)

  def test_score_batch_can_return_intermediates(self):
    encoder_input_tokens = jnp.ones((2, 3))
    # For this test, decoder input and target tokens are dummy values.
    decoder_input_tokens = jnp.array([[1, 2, 1, 0], [0, 1, 0, 2]])
    decoder_target_tokens = jnp.array([[1, 2, 1, 0], [0, 1, 0, 2]])
    decoder_loss_weights = jnp.array([[1, 1, 1, 0], [0, 1, 0, 1]])
    logits = jnp.arange(0, 24).reshape((2, 4, 3))
    modified_variables = {'intermediates': {'bar': jnp.ones(5)}}
    params = {'foo': jnp.zeros(3)}

    mock_transformer = mock.Mock()
    mock_transformer.apply.return_value = (logits, modified_variables)
    mock_transformer.dtype = jnp.float32

    batch = {
        'encoder_input_tokens': encoder_input_tokens,
        'decoder_input_tokens': decoder_input_tokens,
        'decoder_target_tokens': decoder_target_tokens,
        'decoder_loss_weights': decoder_loss_weights
    }

    def mock_init(self):
      self.module = mock_transformer

    with mock.patch.object(
        models.EncoderDecoderModel, '__init__', new=mock_init):
      model = models.EncoderDecoderModel()
      scores, intermediates = model.score_batch(
          params, batch, return_intermediates=True)

    mock_transformer.apply.assert_called_with({'params': params},
                                              encoder_input_tokens,
                                              decoder_input_tokens,
                                              decoder_target_tokens,
                                              encoder_segment_ids=None,
                                              decoder_segment_ids=None,
                                              encoder_positions=None,
                                              decoder_positions=None,
                                              decode=False,
                                              enable_dropout=False,
                                              rngs=None,
                                              mutable=['intermediates'])
    np.testing.assert_allclose(scores, [-3.222973, -1.815315], rtol=1e-4)
    # Incumbent intermediates are passed out unchanged.
    np.testing.assert_allclose(intermediates['bar'], jnp.ones(5))
    # A new collection of decoder intermediates are inserted by score_batch()
    np.testing.assert_allclose(intermediates['decoder']['loss_weights'][0],
                               decoder_loss_weights)
    np.testing.assert_allclose(intermediates['decoder']['target_tokens'][0],
                               decoder_target_tokens)

  def test_train_transformer_wmt(self):
    # Dummy input data
    input_shape = (16, 8)
    encoder_input_tokens = np.ones(shape=input_shape, dtype=np.float32)
    decoder_input_tokens = 5 * np.ones(shape=input_shape, dtype=np.float32)
    decoder_target_tokens = 5 * np.ones(input_shape, dtype=np.float32)
    # input_data = {'inputs': inputs, 'targets': targets}
    input_data = {
        'encoder_input_tokens': encoder_input_tokens,
        'decoder_input_tokens': decoder_input_tokens,
        'decoder_target_tokens': decoder_target_tokens
    }

    partitioner = partitioning.PjitPartitioner(num_partitions=1)

    model = test_utils.get_t5_test_model()

    ds_iter = tf.data.Dataset.from_tensors(input_data).as_numpy_iterator()
    input_shapes = {k: input_shape for k in input_data}

    train_state_initializer = utils.TrainStateInitializer(
        optimizer_def=model.optimizer_def,
        init_fn=model.get_initial_variables,
        input_shapes=input_shapes,
        partitioner=partitioner)
    train_state_axes = train_state_initializer.train_state_axes

    trainer = trainer_lib.Trainer(
        model,
        train_state=train_state_initializer.from_scratch(jax.random.PRNGKey(0)),
        partitioner=partitioner,
        eval_names=[],
        summary_dir=None,
        train_state_axes=train_state_axes,
        rng=jax.random.PRNGKey(0),
        learning_rate_fn=lambda x: 0.001,
        num_microbatches=1)

    trainer.train(ds_iter, 1)
    logging.info('optimizer after first step %s', trainer.train_state.params)


  @parameterized.parameters(
      {'decode_fn': decoding.beam_search},
      {'decode_fn': functools.partial(decoding.temperature_sample, topk=4)})
  def test_predict_batch(self, decode_fn):
    batch_size, encoder_len, max_decode_len, emb_dim = 2, 3, 4, 5
    batch = {
        'encoder_input_tokens':
            np.zeros((batch_size, encoder_len), dtype=np.int32),
        'decoder_input_tokens':
            np.zeros((batch_size, max_decode_len), dtype=np.int32)
    }

    # These dummy logits represent the probability distribution where all the
    # probability mass is in one item (i.e., degenerate distribution). For
    # batch element 0, it is vocabulary index 2.
    # We test `_predict_step` to avoid having to define a task and its
    # vocabulary.
    dummy_logits = jnp.expand_dims(
        jnp.array([[-1e7, -1e7, 0, -1e7], [-1e7, -1e7, -1e7, 0]]), axis=1)

    class MockModule:

      def __init__(self):
        self.dtype = jnp.float32

      def apply(self, *args, method=None, **kwargs):
        del args, kwargs
        if method is None:  # use for module.`__call__`
          return (dummy_logits, {'cache': {}})
        else:
          return method()

      def encode(self):
        return jnp.zeros((batch_size, encoder_len, emb_dim))

      def decode(self):
        return (dummy_logits, {'cache': {}})

    def mock_init(self):
      self.module = MockModule()
      self.module.scan_layers = False
      self._input_vocabulary = mock.Mock(eos_id=1)
      self._output_vocabulary = mock.Mock(eos_id=1)
      self._decode_fn = decode_fn

    with mock.patch.object(
        models.EncoderDecoderModel, '__init__', new=mock_init):
      model = models.EncoderDecoderModel()

    actual = model.predict_batch({}, batch)
    # The predicted token for the first batch element is always 2 and it is 3
    # for the second batch element.
    expected = [[2] * max_decode_len, [3] * max_decode_len]
    np.testing.assert_array_equal(actual, expected)

  def test_predict_batch_rng(self):
    batch = {
        'encoder_input_tokens': np.zeros((2, 1), dtype=np.int32),
        'decoder_input_tokens': np.zeros((2, 2), dtype=np.int32)
    }

    decode_fn_mock = mock.Mock(
        return_value=(np.zeros((2, 2, 3)), np.zeros((2, 2))))

    def mock_init(self):
      self.module = mock.Mock(
          apply=mock.Mock(side_effect=lambda *_, **kwargs: (  # pylint:disable=g-long-lambda,g-long-ternary
              np.zeros((2, 2)), {
                  'cache': None
              }) if 'mutable' in kwargs else np.zeros((2, 2))))
      self._output_vocabulary = mock.Mock(eos_id=1)
      self._decode_fn = decode_fn_mock

    with mock.patch.object(
        models.EncoderDecoderModel, '__init__', new=mock_init):
      model = models.EncoderDecoderModel()

    # No RNG
    model.predict_batch({}, batch)
    _, decode_fn_kwargs = decode_fn_mock.call_args
    self.assertNotIn('decode_rng', decode_fn_kwargs)

    # No RNG (w/ aux)
    model.predict_batch_with_aux({}, batch)
    _, decode_fn_kwargs = decode_fn_mock.call_args
    self.assertNotIn('decode_rng', decode_fn_kwargs)

    # decoder_params RNG
    model.predict_batch_with_aux({}, batch, decoder_params={'decode_rng': 3})
    _, decode_fn_kwargs = decode_fn_mock.call_args
    self.assertEqual(decode_fn_kwargs['decode_rng'], 3)

    # rng RNG
    model.predict_batch({}, batch, rng=4)
    _, decode_fn_kwargs = decode_fn_mock.call_args
    self.assertEqual(decode_fn_kwargs['decode_rng'], 4)

    # rng RNG (w/ aux)
    model.predict_batch_with_aux({}, batch, rng=4)
    _, decode_fn_kwargs = decode_fn_mock.call_args
    self.assertEqual(decode_fn_kwargs['decode_rng'], 4)

    # Both
    with self.assertRaisesWithLiteralMatch(
        ValueError, 'Got RNG both from the `rng` argument (4) and '
        "`decoder_params['decode_rng']` (3). Please specify one or the other."):
      model.predict_batch_with_aux({},
                                   batch,
                                   rng=4,
                                   decoder_params={'decode_rng': 3})

  @parameterized.named_parameters(
      dict(
          testcase_name='int32',
          batch={
              'encoder_input_tokens':
                  np.zeros((BATCH_SIZE, ENCODER_LEN), dtype=np.int32),
              'decoder_input_tokens':
                  np.zeros((BATCH_SIZE, MAX_DECODE_LEN), dtype=np.int32)
          }),
      dict(
          testcase_name='float32',
          batch={
              'encoder_input_tokens':
                  np.zeros((BATCH_SIZE, ENCODER_LEN), dtype=np.float32),
              'decoder_input_tokens':
                  np.zeros((BATCH_SIZE, MAX_DECODE_LEN), dtype=np.float32)
          }))
  def test_predict_batch_fake_input_shapes_and_types(self, batch):

    # These dummy logits represent the probability distribution where all the
    # probability mass is in one item (i.e., degenerate distribution). For
    # batch element 0, it is vocabulary index 2.
    # We test `_predict_step` to avoid having to define a task and its
    # vocabulary.
    dummy_logits = jnp.ones((2, 1, 4), jnp.float32)

    class MockModule:

      def __init__(self):
        self.dtype = jnp.float32
        self.call_args_list = []

      def apply(self, *args, method=None, **kwargs):
        # Not sure why this isn't a real Mock so just record the args/kwargs
        self.call_args_list.append({'args': args, 'kwargs': kwargs})
        del args, kwargs
        if method is None:  # use for module.`__call__`
          return (dummy_logits, {'cache': {}})
        else:
          return method()

      def encode(self):
        return jnp.zeros((BATCH_SIZE, ENCODER_LEN, EMBED_DIM))

      def decode(self):
        return (dummy_logits, {'cache': {}})

    def mock_init(self):
      self.module = MockModule()
      self.module.scan_layers = False
      self._input_vocabulary = mock.Mock(eos_id=1)
      self._output_vocabulary = mock.Mock(eos_id=1)
      self._decode_fn = decoding.beam_search
      self._inputs_bidirectional_attention = False

    with mock.patch.object(
        models.EncoderDecoderModel, '__init__', new=mock_init):
      model = models.EncoderDecoderModel()
    model.predict_batch({}, batch)

    fake_inputs = jnp.ones_like(batch['encoder_input_tokens'])
    fake_target = jnp.ones_like(batch['decoder_input_tokens'])

    cache_init_call = model.module.call_args_list[1]
    self.assertEqual(cache_init_call['args'][0], {'params': {}})
    call_kwargs = cache_init_call['kwargs']
    np.testing.assert_allclose(
        call_kwargs.pop('encoder_input_tokens'), fake_inputs
    )
    np.testing.assert_allclose(
        call_kwargs.pop('decoder_input_tokens'), fake_target
    )
    np.testing.assert_allclose(
        call_kwargs.pop('decoder_target_tokens'), fake_target
    )
    self.assertEqual(
        call_kwargs,
        {
            'decode': True,
            'enable_dropout': False,
            'mutable': ['cache'],
        },
    )


class DecoderOnlyModelTest(parameterized.TestCase):



  def test_predict_batch_visible_in_prefill(self):
    batch_size = 2
    seq_len = 10
    lengths = np.array([[6], [3]])
    batch = {
        'decoder_input_tokens':
            np.tile(
                np.expand_dims(np.arange(seq_len, dtype=np.int32), axis=0),
                (batch_size, 1)),
        'decoder_causal_attention':
            (lengths > np.arange(seq_len)).astype(np.int32)
    }

    dummy_logits = jnp.expand_dims(
        jnp.array([[-1e7, -1e7, 0, -1e7], [-1e7, -1e7, -1e7, 0]]), axis=1)

    mock_module = mock.Mock()
    mock_module.apply.return_value = (dummy_logits, {'cache': {}})
    mock_module.dtype = jnp.float32

    def mock_init(self):
      self.module = mock_module
      self._output_vocabulary = mock.Mock(eos_id=1)
      self._decode_fn = functools.partial(decoding.temperature_sample, topk=4)
      self._inputs_bidirectional_attention = False

    with mock.patch.object(models.DecoderOnlyModel, '__init__', new=mock_init):
      model = models.DecoderOnlyModel()

    model.predict_batch({}, batch)
    prefill_call = mock_module.apply.call_args_list[1]
    kwargs = prefill_call[1]
    inputs = prefill_call[1]['decoder_input_tokens']
    # Note that, for the prefill call, we use 'decoder_causal_attention' as
    # 'decoder_target_tokens'.
    targets = prefill_call[1]['decoder_target_tokens']
    self.assertTrue(kwargs['prefill'])
    np.testing.assert_array_equal(kwargs['prefill_lengths'],
                                  np.squeeze(lengths - 1, axis=-1))
    # Test that the non padding values of the "targets" cover all of the input,
    # you it will all be considered in the attention mask.
    np.testing.assert_array_equal(inputs * targets, inputs)
    # Check that the first value of the target is 1, the first value of the
    # inputs is always 0 so the masking check wouldn't catch it if the target
    # had a 0 in the first location.
    np.testing.assert_array_equal(targets[:, 0], np.ones_like(targets[:, 0]))
    # Test that the targets are properly removed. Our input is a sequence from 0
    # onward, so our largest value (the last input) should be equal by its
    # position (which is 1 - length). If we didn't mask the target correctly,
    # we would expect a larger value in the max.
    np.testing.assert_array_equal(
        np.max(inputs, axis=1), np.squeeze(lengths - 1, axis=-1))


  def test_predict_batch(self):
    batch = {
        'decoder_input_tokens':
            np.array([[0, 3, 4, 5, 6, 0, 0], [0, 7, 8, 9, 0, 0, 0]]),
        'decoder_causal_attention':
            np.array([[1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0]])
    }

    # These dummy logits represent the probability distribution where all the
    # probability mass is in one item (i.e., degenerate distribution). For
    # batch element 0, it is vocabulary index 2.
    # We test `_predict_step` to avoid having to define a task and its
    # vocabulary.
    dummy_logits = jnp.expand_dims(
        jnp.array([[-1e7, -1e7, 0, -1e7], [-1e7, -1e7, -1e7, 0]]), axis=1)

    mock_module = mock.Mock()
    mock_module.apply.return_value = (dummy_logits, {'cache': {}})
    mock_module.dtype = jnp.float32

    def mock_init(self):
      self.module = mock_module
      self._output_vocabulary = mock.Mock(eos_id=1)
      self._decode_fn = functools.partial(decoding.temperature_sample, topk=4)
      self._inputs_bidirectional_attention = False

    with mock.patch.object(models.DecoderOnlyModel, '__init__', new=mock_init):
      model = models.DecoderOnlyModel()

    actual = model.predict_batch({}, batch)

    expected = [[2, 2, 2, 2, 2, 0, 0], [3, 3, 3, 3, 3, 3, 0]]

    # The expected progression of the first element of 'decoder_input_tokens':
    # [0, 3, 4, 5, 6, 0, 0] -> [0, 3, 4, 0, 0, 0, 0] ->
    # [3, 4, 2, 2, 2, 2, 2] -> [2, 2, 2, 2, 2, 0, 0]

    # The expected progression of the second element of 'decoder_input_tokens':
    # [0, 7, 8, 9, 0, 0, 0] -> [0, 7, 0, 0, 0, 0, 0] ->
    # [7, 3, 3, 3, 3, 3, 3] -> [3, 3, 3, 3, 3, 3, 0]

    np.testing.assert_array_equal(actual, expected)

  def test_predict_batch_rng(self):
    batch = {
        'decoder_input_tokens': np.zeros((2, 2), dtype=np.int32),
        'decoder_causal_attention': np.zeros((2, 2), dtype=np.int32)
    }

    decode_fn_mock = mock.Mock(
        return_value=(np.zeros((2, 2, 3)), np.zeros((2, 2))))

    def mock_init(self):
      self.module = mock.Mock(
          apply=mock.Mock(side_effect=lambda *_, **kwargs: (  # pylint:disable=g-long-lambda,g-long-ternary
              np.zeros((2, 2)), {
                  'cache': None
              }) if 'mutable' in kwargs else np.zeros((2, 2))))
      self._output_vocabulary = mock.Mock(eos_id=1)
      self._decode_fn = decode_fn_mock
      self._inputs_bidirectional_attention = False

    with mock.patch.object(models.DecoderOnlyModel, '__init__', new=mock_init):
      model = models.DecoderOnlyModel()

    # No RNG
    model.predict_batch({}, batch)
    _, decode_fn_kwargs = decode_fn_mock.call_args
    self.assertNotIn('decode_rng', decode_fn_kwargs)

    # No RNG (w/ aux)
    model.predict_batch_with_aux({}, batch)
    _, decode_fn_kwargs = decode_fn_mock.call_args
    self.assertNotIn('decode_rng', decode_fn_kwargs)

    # decoder_params RNG
    model.predict_batch_with_aux({}, batch, decoder_params={'decode_rng': 3})
    _, decode_fn_kwargs = decode_fn_mock.call_args
    self.assertEqual(decode_fn_kwargs['decode_rng'], 3)

    # rng RNG
    model.predict_batch({}, batch, rng=4)
    _, decode_fn_kwargs = decode_fn_mock.call_args
    self.assertEqual(decode_fn_kwargs['decode_rng'], 4)

    # rng RNG (w/ aux)
    model.predict_batch_with_aux({}, batch, rng=4)
    _, decode_fn_kwargs = decode_fn_mock.call_args
    self.assertEqual(decode_fn_kwargs['decode_rng'], 4)

    # Both
    with self.assertRaisesWithLiteralMatch(
        ValueError, 'Got RNG both from the `rng` argument (4) and '
        "`decoder_params['decode_rng']` (3). Please specify one or the other."):
      model.predict_batch_with_aux({},
                                   batch,
                                   rng=4,
                                   decoder_params={'decode_rng': 3})

  def test_predict_batch_num_decodes_temperature_sample(self):
    batch = {
        'decoder_input_tokens': np.array([
            [0, 3, 4, 5, 6, 0, 0],
        ]),
        'decoder_causal_attention': np.array([
            [1, 1, 1, 0, 0, 0, 0],
        ])
    }

    # These dummy logits represent the probability distribution where all the
    # probability mass is in one item (i.e., degenerate distribution). For
    # batch element 0, it is vocabulary index 2. We have two samples.
    # Technically these should be identical since the prompts are the same, but
    # this makes testing easier.
    dummy_logits = jnp.expand_dims(
        jnp.array([[-1e7, -1e7, 0, -1e7], [-1e7, -1e7, -1e7, 0]]), axis=1)

    mock_module = mock.Mock()
    mock_module.apply.return_value = (dummy_logits, {'cache': {}})
    mock_module.dtype = jnp.float32

    def mock_init(self):
      self.module = mock_module
      self._output_vocabulary = mock.Mock(eos_id=1)
      self._decode_fn = functools.partial(decoding.temperature_sample, topk=4)
      self._inputs_bidirectional_attention = False

    with mock.patch.object(models.DecoderOnlyModel, '__init__', new=mock_init):
      model = models.DecoderOnlyModel()

    actual_output, aux = model.predict_batch_with_aux({},
                                                      batch,
                                                      num_decodes=2,
                                                      return_all_decodes=True)

    expected_output = [[[2, 2, 2, 2, 2, 0, 0], [3, 3, 3, 3, 3, 0, 0]]]
    expected_scores = [[0., 0.]]

    # The expected progression of the first element of 'decoder_input_tokens':
    # [0, 3, 4, 5, 6, 0, 0] -> [0, 3, 4, 0, 0, 0, 0] ->
    # [3, 4, 2, 2, 2, 2, 2] -> [2, 2, 2, 2, 2, 0, 0]

    # The expected progression of the second element of 'decoder_input_tokens':
    # [0, 7, 8, 9, 0, 0, 0] -> [0, 7, 0, 0, 0, 0, 0] ->
    # [7, 3, 3, 3, 3, 3, 3] -> [3, 3, 3, 3, 3, 3, 0]

    np.testing.assert_array_equal(actual_output, expected_output)
    np.testing.assert_array_equal(aux['scores'], expected_scores)

  def test_predict_batch_fake_input_shapes_and_types(self):
    # The input and causal attention actually have to be int32 for this test,
    # even though the cache init should work with any types the `inputs` that
    # is created from multiplying the causal attention and the input tokens
    # needs to be an int or the decoding will fail.
    batch = {
        'decoder_input_tokens':
            np.array([[0, 3, 4, 5, 6, 0, 0], [0, 7, 8, 9, 0, 0, 0]],
                     dtype=np.int32),
        'decoder_causal_attention':
            np.array([[1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0]],
                     dtype=np.int32)
    }

    dummy_logits = jnp.ones((2, 1, 5), jnp.float32)

    mock_module = mock.Mock()
    mock_module.apply.return_value = (dummy_logits, {'cache': {}})
    mock_module.dtype = jnp.float32

    def mock_init(self):
      self.module = mock_module
      self._output_vocabulary = mock.Mock(eos_id=1)
      self._decode_fn = functools.partial(decoding.temperature_sample, topk=4)
      self._inputs_bidirectional_attention = False

    with mock.patch.object(models.DecoderOnlyModel, '__init__', new=mock_init):
      model = models.DecoderOnlyModel()

    model.predict_batch({}, batch)

    fake_target = jnp.ones_like(batch['decoder_input_tokens'])

    cache_init_call = mock_module.apply.call_args_list[0]

    self.assertEqual(cache_init_call[0][0], {'params': {}})
    np.testing.assert_allclose(cache_init_call[0][1], fake_target)
    np.testing.assert_allclose(cache_init_call[0][2], fake_target)
    self.assertEqual(cache_init_call[1], {
        'decode': True,
        'enable_dropout': False,
        'mutable': ['cache']
    })

  @parameterized.named_parameters(
      dict(
          testcase_name='no_types',
          shapes={'decoder_input_tokens': [1, 62]},
          types=None),
      dict(
          testcase_name='int32',
          shapes={'decoder_input_tokens': [1, 62]},
          types={'decoder_input_tokens': jnp.int32}),
      dict(
          testcase_name='float32',
          shapes={'decoder_input_tokens': [1, 62]},
          types={'decoder_input_tokens': jnp.int32}),
  )
  def test_get_initial_variables_shapes_and_types(self, shapes, types):
    mock_lm = mock.Mock()
    mock_lm.init.return_value = {'params': {}}
    mock_optimizer_def = mock.Mock()
    rng = mock.Mock()

    def mock_init(self):
      self.module = mock_lm
      self.optimizer_def = mock_optimizer_def

    with mock.patch.object(models.DecoderOnlyModel, '__init__', new=mock_init):
      model = models.DecoderOnlyModel()
      model.get_initial_variables(rng, shapes, types)

    if types is None:
      decoder_input = jnp.ones(
          shapes['decoder_input_tokens'], dtype=jnp.float32)
    else:
      decoder_input = jnp.ones(
          shapes['decoder_input_tokens'], dtype=types['decoder_input_tokens'])

    # Using `.assert_called_once_with` doesn't work because the simple
    # comparison it does for the array arguments fail (truth value of an array
    # is ambiguous).
    called_with = mock_lm.init.call_args
    self.assertEqual(called_with[0][0], rng)
    np.testing.assert_allclose(called_with[0][1], decoder_input)
    np.testing.assert_allclose(called_with[0][2], decoder_input)
    self.assertEqual(mock_lm.init.call_args[1], {'enable_dropout': False})


if __name__ == '__main__':
  absltest.main()
