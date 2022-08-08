# Copyright 2022 The T5X Authors.
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

"""Tests for models."""

import functools
from unittest import mock

from absl.testing import absltest
from clu import metrics as clu_metrics_lib
from flax import core as flax_core
import jax.numpy as jnp
import numpy as np
from t5x import decoding
from t5x import metrics as metrics_lib

from t5x.contrib.moe import models

Accuracy = clu_metrics_lib.Accuracy
Average = clu_metrics_lib.Average
AveragePerStep = metrics_lib.AveragePerStep
FrozenDict = flax_core.frozen_dict.FrozenDict


class ModelsTest(absltest.TestCase):

  def test_expert_losses(self):
    diversity_metrics = {
        'auxiliary_loss': jnp.array([1., 2.]),
        'router_z_loss': jnp.array([0., 1.]),
        'fraction_tokens_left_behind': jnp.array([0.5, 0.5]),
        'expert_usage': jnp.array([0.5, 0.5]),
        'router_confidence': jnp.array([0.5, 0.5])
    }
    aux_loss, router_z_loss = models._expert_losses(
        diversity_metrics, auxiliary_loss_factor=0.1, router_z_loss_factor=10)

    self.assertEqual(aux_loss, 0.15)
    self.assertEqual(router_z_loss, 5.)

  def test_expert_metrics(self):
    diversity_metrics = {
        'auxiliary_loss': jnp.array([1., 2.]),
        'router_z_loss': jnp.array([0., 1.]),
        'fraction_tokens_left_behind': jnp.array([1., 0.5]),
        'expert_usage': jnp.array([0.7, 0.5]),
        'router_confidence': jnp.array([0.5, 0.5])
    }
    actual_metrics = models._expert_metrics(
        diversity_metrics,
        total_loss=100.,
        z_loss=1.,
        auxiliary_loss=3.,
        router_z_loss=7.,
        num_tokens=2)
    actual_metrics = metrics_lib.set_step_metrics_num_steps(actual_metrics, 1)
    actual_computed_metrics = {
        k: v.compute() for k, v in actual_metrics.items()
    }

    expected_metrics = {
        'cross_ent_loss': 89.0,
        'cross_ent_loss_per_all_target_tokens': 44.5,
        'experts/auxiliary_loss': 3.,
        'experts/expert_usage': 0.6,
        'experts/fraction_tokens_left_behind': 0.75,
        'experts/router_confidence': 0.5,
        'experts/router_z_loss': 7.
    }
    self.assertEqual(actual_computed_metrics, expected_metrics)

  def test_extract_diversity_metrics(self):
    state = flax_core.freeze({
        'intermediates': {
            'moe_layer_0': {
                'auxiliary_loss': (jnp.array([[0.2]]), jnp.array([[0.1]])),
                'router_z_loss': (jnp.array([[0.1]]), jnp.array([[0.2]])),
                'router_confidence': (jnp.array([[0.4]]), jnp.array([[0.2]])),
                'expert_usage': (jnp.array([[0.9]]), jnp.array([[0.2]])),
                'fraction_tokens_left_behind': (jnp.array([[0.1]]),
                                                jnp.array([[0.2]])),
            },
            'moe_layer_1': {
                'auxiliary_loss': (jnp.array([[0.2]]), jnp.array([[0.]])),
                'router_z_loss': (jnp.array([[0.1]]), jnp.array([[0.]])),
                'router_confidence': (jnp.array([[0.4]]), jnp.array([[0.5]])),
                'expert_usage': (jnp.array([[0.9]]), jnp.array([[0.8]])),
                'fraction_tokens_left_behind': (jnp.array([[0.1]]),
                                                jnp.array([[0.3]])),
            }
        }
    })
    extracted_metrics = models._extract_diversity_metrics(state)

    expected_raw_metrics = {
        'auxiliary_loss':
            jnp.array([[[0.2]], [[0.05]]], dtype=jnp.float32),
        'router_z_loss':
            jnp.array([[[0.1]], [[0.1]]], dtype=jnp.float32),
        'fraction_tokens_left_behind':
            jnp.array([[[0.1]], [[0.25]]], dtype=jnp.float32),
        'expert_usage':
            jnp.array([[[0.9]], [[0.5]]], dtype=jnp.float32),
        'router_confidence':
            jnp.array([[[0.4]], [[0.35]]], dtype=jnp.float32)
    }
    for metric, expected_value in expected_raw_metrics.items():
      np.testing.assert_allclose(extracted_metrics[metric], expected_value)

  def test_extract_from_non_expert_model(self):
    empty_state = FrozenDict({'intermediates': {}})
    with self.assertRaisesRegex(ValueError, 'Unable to find expert metric'):
      models._extract_diversity_metrics(empty_state)

  def test_encoder_decoder_model(self):
    encoder_input_tokens = jnp.ones((2, 3))
    decoder_input_tokens = jnp.array([[1, 2, 1, 0], [0, 1, 0, 2]])
    decoder_target_tokens = jnp.array([[1, 2, 1, 0], [0, 1, 0, 2]])
    decoder_loss_weights = jnp.array([[1, 1, 1, 0], [0, 1, 0, 1]])
    dummy_logits = jnp.arange(0, 24).reshape((2, 4, 3))
    params = {'foo': jnp.zeros(3)}

    mock_transformer = mock.Mock()
    mock_transformer.apply.return_value = dummy_logits
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
        models.MoeEncoderDecoderModel, '__init__', new=mock_init):
      model = models.MoeEncoderDecoderModel()
      result = model.score_batch(params, batch)

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
    np.testing.assert_allclose(result, [-3.2228181, -1.8152122], rtol=1e-5)

  def test_decoder_only_model(self):
    batch = {
        'decoder_input_tokens':
            jnp.array([[0, 3, 4, 5, 6, 0, 0], [0, 7, 8, 9, 0, 0, 0]]),
        'decoder_causal_attention':
            jnp.array([[1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0]]),
    }
    params = {}

    dummy_logits = jnp.expand_dims(
        jnp.array([[-1e7, -1e7, 0, -1e7], [-1e7, -1e7, -1e7, 0]]), axis=1)

    mock_transformer = mock.Mock()
    mock_transformer.apply.return_value = (dummy_logits, {'cache': {}})
    mock_transformer.dtype = jnp.float32

    def mock_init(self):
      self.module = mock_transformer
      self._output_vocabulary = mock.Mock(eos_id=1)
      self._decode_fn = functools.partial(decoding.temperature_sample, topk=4)
      self._inputs_bidirectional_attention = False

    with mock.patch.object(
        models.MoeDecoderOnlyModel, '__init__', new=mock_init):
      model = models.MoeDecoderOnlyModel()

    actual = model.predict_batch(params, batch)
    expected = [[2, 2, 2, 2, 2, 0, 0], [3, 3, 3, 3, 3, 3, 0]]
    np.testing.assert_array_equal(actual, expected)

  def test_moe_loss_fn(self):
    batch = {
        'encoder_input_tokens': jnp.ones((2, 3)),
        'decoder_input_tokens': jnp.array([[1, 2, 1, 0], [0, 1, 0, 2]]),
        'decoder_target_tokens': jnp.array([[1, 2, 1, 0], [0, 1, 0, 2]]),
        'decoder_loss_weights': jnp.array([[1, 1, 1, 0], [0, 1, 0, 1]])
    }
    logits = jnp.arange(0, 24).reshape((2, 4, 3))
    state = flax_core.freeze({
        'intermediates': {
            'auxiliary_loss': (jnp.array([[0.2]]),),
            'router_z_loss': (jnp.array([[0.1]]),),
            'router_confidence': (jnp.array([[0.4]]),),
            'expert_usage': (jnp.array([[0.9]]),),
            'fraction_tokens_left_behind': (jnp.array([[0.1]]),),
        }
    })

    loss, metrics = models._moe_loss_fn(
        batch,
        logits,
        state,
        label_smoothing=0.,
        z_loss=0.,
        loss_normalizing_factor=None,
        aux_loss_factor=0.1,
        router_z_loss_factor=0.01)

    self.assertAlmostEqual(loss, 5.0590305)
    self.assertContainsSubset(
        {
            'experts/auxiliary_loss':
                AveragePerStep(total=jnp.array(0.02), steps=1),
            'experts/router_z_loss':
                AveragePerStep(total=jnp.array(0.001), steps=1),
            'experts/router_confidence':
                AveragePerStep(total=jnp.array(0.5), steps=1),
            'experts/expert_usage':
                AveragePerStep(total=jnp.array(0.9), steps=1),
            'experts/fraction_tokens_left_behind':
                AveragePerStep(total=jnp.array(0.1), steps=1),
            'accuracy':
                Accuracy(total=jnp.array(2.), count=jnp.array(5)),
            'cross_ent_loss':
                AveragePerStep(steps=1, total=jnp.array(5.0590305)),
            'loss':
                AveragePerStep(steps=1, total=jnp.array(5.0590305)),
            'timing/seqs_per_second':
                metrics_lib.TimeRate(duration=None, numerator=2),
            'timing/steps_per_second':
                metrics_lib.StepsPerTime(duration=None, steps=1),
        }, metrics)


if __name__ == '__main__':
  absltest.main()
