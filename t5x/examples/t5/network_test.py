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

"""Tests for network."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
import seqio
from t5x import adafactor
from t5x import models
from t5x import test_utils
from t5x.examples.t5 import network


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

  def test_regression(self):
    batch, max_decode_len, input_len = 2, 3, 4
    emb_dim, num_heads, head_dim, mlp_dim, vocab_size = 13, 8, 64, 2048, 10
    model = get_test_model(
        emb_dim,
        head_dim,
        num_heads,
        mlp_dim,
        vocab_size=vocab_size,
        num_encoder_layers=3)

    input_shapes = {
        'encoder_input_tokens': (batch, input_len),
        'decoder_input_tokens': (batch, max_decode_len)
    }
    params = model.get_initial_variables(jax.random.PRNGKey(42),
                                         input_shapes)['params']

    np.random.seed(0)
    batch = {
        'encoder_input_tokens':
            np.random.randint(3, 10, size=(batch, input_len)),
        'decoder_input_tokens':
            np.random.randint(3, 10, size=(batch, max_decode_len)),
        'decoder_target_tokens':
            np.random.randint(3, 10, size=(batch, max_decode_len))
    }
    loss, _ = jax.jit(model.loss_fn)(params, batch, jax.random.PRNGKey(1))
    self.assertAlmostEqual(loss, 18.088945, delta=0.05)

    predicted, scores = model.predict_batch_with_aux(params, batch)
    np.testing.assert_array_equal(predicted, [[7, 1, 0], [1, 0, 0]])
    np.testing.assert_allclose(
        scores['scores'], [-3.0401115, -1.9265753], rtol=1e-3)



if __name__ == '__main__':
  absltest.main()
