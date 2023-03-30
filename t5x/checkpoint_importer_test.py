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

"""Tests for t5x.checkpoint_importer."""

import json
import os

from absl import flags
from absl.testing import absltest
import jax
import numpy as np
from t5x import checkpoint_importer
import tensorflow as tf


class CheckpointImporterTest(absltest.TestCase):

  def test_rel_embeddings_shared_layers(self):
    # This represents a ckpt where the Mesh TensorFlow's
    # transformer_layers.SelfAttention.relative_attention_type = "bias_shared",
    # i.e., the same relative attention parameters are shared by all layers
    # within the (en|de)coder.
    ckpt_data = {
        'encoder/block_000/layer_000/SelfAttention/relative_attention_bias':
            1,
        'decoder/block_000/layer_000/SelfAttention/relative_attention_bias':
            2,
        'decoder/block_000/layer_000/SelfAttention/relative_attention_bias_slot_v':
            3,
    }
    t5_data = checkpoint_importer.t5_importer.apply(ckpt_data)
    t5_data = checkpoint_importer._maybe_correct_relpos_bias(t5_data)
    expected = {
        'target/encoder/relpos_bias/rel_embedding': 1,
        'target/decoder/relpos_bias/rel_embedding': 2,
        'state/param_states/decoder/relpos_bias/rel_embedding/v': 3,
    }
    self.assertEqual(t5_data, expected)

  def test_rel_embeddings_per_layer(self):
    # This represents a ckpt where the Mesh TensorFlow's
    # transformer_layers.SelfAttention.relative_attention_type = "bias", i.e.,
    # each layer has its own relative attention parameters.
    ckpt_data = {
        'encoder/block_000/layer_000/SelfAttention/relative_attention_bias':
            1,
        'encoder/block_001/layer_000/SelfAttention/relative_attention_bias':
            2,
        'decoder/block_000/layer_000/SelfAttention/relative_attention_bias':
            3,
        'decoder/block_000/layer_000/SelfAttention/relative_attention_bias_slot_v':
            4,
        'decoder/block_011/layer_000/SelfAttention/relative_attention_bias':
            5
    }
    t5_data = checkpoint_importer.t5_importer.apply(ckpt_data)
    t5_data = checkpoint_importer._maybe_correct_relpos_bias(t5_data)
    expected = {
        'target/encoder/layers_0/relpos_bias/rel_embedding': 1,
        'target/encoder/layers_1/relpos_bias/rel_embedding': 2,
        'target/decoder/layers_0/relpos_bias/rel_embedding': 3,
        'state/param_states/decoder/layers_0/relpos_bias/rel_embedding/v': 4,
        'target/decoder/layers_11/relpos_bias/rel_embedding': 5,
    }
    self.assertEqual(t5_data, expected)


if __name__ == '__main__':
  absltest.main()
