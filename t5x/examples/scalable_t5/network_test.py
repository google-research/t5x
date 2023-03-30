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

"""Tests for network."""

import os

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
import seqio
from t5x import adafactor
from t5x import models
from t5x import test_utils
from t5x.examples.scalable_t5 import network

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()

FLAGS = flags.FLAGS


def get_test_model(emb_dim,
                   head_dim,
                   num_heads,
                   mlp_dim,
                   dtype='float32',
                   vocab_size=32128,
                   num_encoder_layers=2,
                   num_decoder_layers=2):
  config = network.T5Config(
      num_encoder_layers=num_encoder_layers,
      num_decoder_layers=num_decoder_layers,
      vocab_size=vocab_size,
      dropout_rate=0,
      emb_dim=emb_dim,
      num_heads=num_heads,
      head_dim=head_dim,
      mlp_dim=mlp_dim,
      dtype=dtype,
      mlp_activations=('gelu', 'linear'))
  module = network.Transformer(config=config)
  vocab = seqio.test_utils.sentencepiece_vocab()
  optimizer_def = adafactor.Adafactor()
  return models.EncoderDecoderModel(
      module, vocab, vocab, optimizer_def=optimizer_def)


class NetworkTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    batch_size, max_decode_len, input_len = 2, 3, 4
    self.input_shapes = {
        'encoder_input_tokens': (batch_size, input_len),
        'decoder_input_tokens': (batch_size, max_decode_len)
    }
    np.random.seed(42)
    self.batch = {
        'encoder_input_tokens':
            np.random.randint(3, 10, size=(batch_size, input_len)),
        'decoder_input_tokens':
            np.random.randint(3, 10, size=(batch_size, max_decode_len)),
        'decoder_target_tokens':
            np.random.randint(3, 10, size=(batch_size, max_decode_len))
    }

  def test_regression(self):
    model = get_test_model(
        emb_dim=13,
        head_dim=64,
        num_heads=8,
        mlp_dim=2048,
        vocab_size=10,
        num_encoder_layers=3)
    params = model.get_initial_variables(
        jax.random.PRNGKey(0), self.input_shapes)['params']
    loss, _ = model.loss_fn(params, self.batch, jax.random.PRNGKey(1))

    self.assertAlmostEqual(loss, 16.45335, delta=0.05)
    predicted, scores = model.predict_batch_with_aux(params, self.batch)
    np.testing.assert_array_equal(predicted, [[7, 1, 0], [7, 1, 0]])
    np.testing.assert_allclose(
        scores['scores'], [-1.240393, -2.035653], rtol=1e-2)



if __name__ == '__main__':
  absltest.main()
