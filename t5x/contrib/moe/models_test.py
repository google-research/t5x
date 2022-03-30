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

from unittest import mock

from absl.testing import absltest
from clu import metrics as clu_metrics_lib
from flax import core as flax_core
import jax.numpy as jnp
import numpy as np
from t5x import metrics as metrics_lib

from t5x.contrib.moe import models

Accuracy = clu_metrics_lib.Accuracy
AveragePerStep = metrics_lib.AveragePerStep
ExpertMetrics = models.ExpertMetrics
FrozenDict = flax_core.frozen_dict.FrozenDict


class ModelsTest(absltest.TestCase):

  def test_expert_losses(self):
    diversity_metrics = [
        ExpertMetrics(
            auxiliary_loss=1.,
            router_z_loss=0.,
            fraction_tokens_left_behind=0.5,
            expert_usage=0.5,
            router_confidence=0.5),
        ExpertMetrics(
            auxiliary_loss=2.,
            router_z_loss=1.,
            fraction_tokens_left_behind=0.5,
            expert_usage=0.5,
            router_confidence=0.5)
    ]
    aux_loss, router_z_loss = models._expert_losses(
        diversity_metrics, auxiliary_loss_factor=0.1, router_z_loss_factor=10)

    self.assertEqual(aux_loss, 0.15)
    self.assertEqual(router_z_loss, 5.)

  def test_expert_metrics(self):
    diversity_metrics = [
        ExpertMetrics(
            auxiliary_loss=1.,
            router_z_loss=0.,
            fraction_tokens_left_behind=1.,
            expert_usage=0.7,
            router_confidence=0.5),
        ExpertMetrics(
            auxiliary_loss=2.,
            router_z_loss=1.,
            fraction_tokens_left_behind=0.5,
            expert_usage=0.5,
            router_confidence=0.5)
    ]
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

  def test_extract_from_non_expert_model(self):
    empty_state = FrozenDict({'intermediates': {}})
    with self.assertRaisesRegex(ValueError,
                                'Unable to find any expert diversity metrics.'):
      models._extract_diversity_metrics(empty_state)

  def test_model(self):
    encoder_input_tokens = jnp.ones((2, 3))
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


if __name__ == '__main__':
  absltest.main()
