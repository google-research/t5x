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

"""Tests for t5x.losses."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from t5x import losses


class LossTest(absltest.TestCase):

  def test_xent(self):

    def lossfn(logits, targets, weights):
      loss, z_loss, weight_sum = losses.compute_weighted_cross_entropy(
          logits,
          targets,
          weights,
          label_smoothing=0.1,
          z_loss=0.1,
          loss_normalizing_factor=0.1)
      return loss, (z_loss, weight_sum)

    batch_size = 2
    length = 4
    vocab_size = 8
    logits = np.random.normal(size=(batch_size, length,
                                    vocab_size)).astype(np.float32)
    targets = np.random.randint(0, vocab_size, size=(batch_size, length))
    weights = np.ones_like(targets)
    out = jax.jit(jax.value_and_grad(lossfn, has_aux=True))(logits, targets,
                                                            weights)
    (loss, (z_loss, weight_sum)), dlogits = out
    # Just a smoke test for now
    # TODO(t5x): Expand test
    print(jax.device_get(((loss, (z_loss, weight_sum)), dlogits)))


class SpecialLossNormalizingFactorTest(absltest.TestCase):

  def test_num_real_target_tokens(self):
    batch = {
        'decoder_target_tokens':
            jnp.asarray([[1, 2, 3, 4, 0], [5, 6, 0, 0, 0]], jnp.int32)
    }

    (output_lnf,
     output_loss_weights) = losses.get_loss_normalizing_factor_and_weights(
         loss_normalizing_factor=losses.SpecialLossNormalizingFactor
         .NUM_REAL_TARGET_TOKENS,
         batch=batch)

    np.testing.assert_allclose(output_lnf, 6.0, rtol=1e-3)
    np.testing.assert_allclose(
        output_loss_weights,
        np.array([[1.0, 1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0]],
                 dtype=np.float32),
        rtol=1e-3)

  def test_num_total_target_tokens(self):
    batch = {
        'decoder_target_tokens':
            jnp.asarray([[1, 2, 3, 4, 0], [5, 6, 0, 0, 0]], jnp.int32)
    }

    (output_lnf,
     output_loss_weights) = losses.get_loss_normalizing_factor_and_weights(
         loss_normalizing_factor=losses.SpecialLossNormalizingFactor
         .NUM_TOTAL_TARGET_TOKENS,
         batch=batch)

    np.testing.assert_allclose(output_lnf, 10.0, rtol=1e-3)
    np.testing.assert_allclose(
        output_loss_weights,
        np.array([[1.0, 1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0]],
                 dtype=np.float32),
        rtol=1e-3)

  def test_average_per_sequence(self):
    batch = {
        'decoder_target_tokens':
            jnp.asarray([[1, 2, 3, 4, 0], [5, 6, 0, 0, 0]], jnp.int32)
    }

    (output_lnf,
     output_loss_weights) = losses.get_loss_normalizing_factor_and_weights(
         loss_normalizing_factor=losses.SpecialLossNormalizingFactor
         .AVERAGE_PER_SEQUENCE,
         batch=batch)

    np.testing.assert_allclose(output_lnf, 2.0, rtol=1e-3)
    np.testing.assert_allclose(
        output_loss_weights,
        jnp.asarray([[0.25, 0.25, 0.25, 0.25, 0.0], [0.5, 0.5, 0.0, 0.0, 0.0]],
                    jnp.float32),
        rtol=1e-3)

  def test_average_per_sequence_with_weights(self):
    batch = {
        'decoder_target_tokens':
            jnp.asarray([[1, 2, 3, 4, 0], [5, 6, 0, 0, 0]], jnp.int32),
        'decoder_loss_weights':
            jnp.asarray([[0.5, 1.0, 0.25, 2.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0]],
                        jnp.float32)
    }

    (output_lnf,
     output_loss_weights) = losses.get_loss_normalizing_factor_and_weights(
         loss_normalizing_factor=losses.SpecialLossNormalizingFactor
         .AVERAGE_PER_SEQUENCE,
         batch=batch)

    np.testing.assert_allclose(output_lnf, 2.0, rtol=1e-3)
    np.testing.assert_allclose(
        output_loss_weights,
        jnp.asarray([[0.5 / 3.75, 1.0 / 3.75, 0.25 / 3.75, 2.0 / 3.75, 0.0],
                     [1.0 / 2.0, 1.0 / 2.0, 0.0, 0.0, 0.0]], jnp.float32),
        rtol=1e-3)

  def test_sum_weights_per_segment(self):
    weights = jnp.asarray(
        [[0.5, 1.0, 0.25, 2.0, 1.5], [1.0, 2.0, 3.0, 4.0, 5.0]], jnp.float32)
    positions = jnp.asarray([[0, 1, 2, 0, 0], [0, 0, 1, 0, 0]])
    segment_ids = jnp.asarray([[1, 1, 1, 2, 3], [1, 2, 2, 3, 0]])

    norm_vec = losses._sum_weights_per_segment(positions, segment_ids, weights)

    np.testing.assert_allclose(
        norm_vec,
        jnp.asarray([[1.75, 1.75, 1.75, 2.0, 1.5], [1.0, 5.0, 5.0, 4.0, 0.0]],
                    jnp.float32),
        rtol=1e-3)

  def test_average_per_sequence_with_weights_with_packing(self):
    batch = {
        'decoder_target_tokens':
            jnp.asarray([[1, 2, 3, 4, 5], [5, 6, 7, 8, 0]], jnp.int32),
        'decoder_loss_weights':
            jnp.asarray([[0.5, 1.0, 0.25, 2.0, 1.5], [1.0, 2.0, 3.0, 4.0, 5.0]],
                        jnp.float32),
        'decoder_positions':
            jnp.asarray([[0, 1, 2, 0, 0], [0, 0, 1, 0, 0]]),
        'decoder_segment_ids':
            jnp.asarray([[1, 1, 1, 2, 3], [1, 2, 2, 3, 0]])
    }

    (output_lnf,
     output_loss_weights) = losses.get_loss_normalizing_factor_and_weights(
         loss_normalizing_factor=losses.SpecialLossNormalizingFactor
         .AVERAGE_PER_SEQUENCE,
         batch=batch)

    np.testing.assert_allclose(output_lnf, 6.0, rtol=1e-3)
    np.testing.assert_allclose(
        output_loss_weights,
        jnp.asarray(
            [[0.5 / 1.75, 1.0 / 1.75, 0.25 / 1.75, 2.0 / 2.0, 1.5 / 1.5],
             [1.0 / 1.0, 2.0 / 5.0, 3.0 / 5.0, 4.0 / 4.0, 0.0]], jnp.float32),
        rtol=1e-3)

if __name__ == '__main__':
  absltest.main()
