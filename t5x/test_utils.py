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

"""Testing helpers for fake device meshes and data chunking."""

import dataclasses
import itertools
import operator
from typing import Sequence, Tuple
from flax import optim
import jax
import numpy as np
import seqio
from t5x import adafactor
from t5x import models
from t5x import train_state as train_state_lib
from t5x.examples.t5 import network
import tensorflow.compat.v2 as tf


# Mock JAX devices
@dataclasses.dataclass
class CpuDevice:
  id: int
  process_index: int
  device_kind: str = 'cpu'
  platform: str = 'cpu'


@dataclasses.dataclass
class GpuDevice:
  id: int
  process_index: int
  device_kind: str = 'gpu'
  platform: str = 'Tesla V100-SXM2-16GB'


@dataclasses.dataclass
class TpuDevice:
  id: int
  process_index: int
  coords: Sequence[int]
  core_on_chip: int
  device_kind: str = 'TPU v3'
  platform: str = 'tpu'


# Mock TPU device meshes.
def coords_to_idx(coords: Tuple[int, ...], bounds: Tuple[int, ...]) -> int:
  """Convert grid coordinates to linear index given a dimension ordering.

  Args:
    coords: coordinates in minor to major ordering.
    bounds: coordinate grid bonuds in SAME minor to major ordering as above.

  Returns:
    Linear index for grid point.
  """
  # Calculate stride multipliers.
  strides = tuple(itertools.accumulate((1,) + bounds[:-1], operator.mul))
  # Sum linear index from strides and coords
  return sum(jax.tree_multimap(lambda x, y: x * y, coords, strides))


def make_devices(nx: int,
                 ny: int,
                 nz: int,
                 nc: int = 2,
                 host_layout: Tuple[int, ...] = (2, 2, 1, 2),
                 kind='TPU v3'):
  """Create mock TPU devices."""
  devices = []
  device_bounds = (nx, ny, nz, nc)
  hnx, hny, hnz, hnc = jax.tree_multimap(lambda a, b: a // b, device_bounds,
                                         host_layout)
  for x, y, z, c in itertools.product(*map(range, device_bounds)):
    hx, hy, hz, hc = jax.tree_multimap(lambda a, b: a // b, (x, y, z, c),
                                       host_layout)
    # TODO(levskaya, jekbradbury): verify this id/host ordering on TPU v4
    device_id = coords_to_idx((c, x, y, z), (nc, nx, ny, nz))  # pytype: disable=wrong-arg-types
    process_index = coords_to_idx((hc, hx, hy, hz), (hnc, hnx, hny, hnz))  # pytype: disable=wrong-arg-types
    devices.append(
        TpuDevice(
            id=device_id,
            process_index=process_index,
            coords=(x, y, z),
            core_on_chip=c,
            platform='tpu',
            device_kind=kind))
  return devices


def get_t5_test_model(**config_overrides) -> models.EncoderDecoderModel:
  """Returns a tiny T5 1.1 model to use for testing."""
  tiny_config = network.T5Config(
      vocab_size=32128,
      dtype='bfloat16',
      emb_dim=8,
      num_heads=4,
      num_encoder_layers=2,
      num_decoder_layers=2,
      head_dim=3,
      mlp_dim=16,
      mlp_activations=('gelu', 'linear'),
      dropout_rate=0.0,
      logits_via_embedding=False,
  )

  tiny_config = dataclasses.replace(tiny_config, **config_overrides)
  sentencepiece_model_file = 'gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model'
  vocabulary = seqio.SentencePieceVocabulary(sentencepiece_model_file)
  return models.EncoderDecoderModel(
      module=network.Transformer(tiny_config),
      input_vocabulary=vocabulary,
      output_vocabulary=vocabulary,
      optimizer_def=adafactor.Adafactor(
          decay_rate=0.8,
          step_offset=0,
          logical_factor_rules=adafactor.standard_logical_factor_rules()))


def get_fake_vocab():
  """Creates fake vocabulary compatible with `get_fake_tokenized_dataset`."""

  @dataclasses.dataclass
  class DummyVocab:
    vocab_size: int = 128
    eos_id: int = 1

  vocab = DummyVocab()
  return (vocab, vocab)


# Text preprocessed and tokenized.
_FAKE_TOKENIZED_DATASET = {
    'train': [
        {
            'inputs': (3, 13, 7, 14, 15, 9, 4, 16),
            'inputs_pretokenized': 'complete: this',
            'targets': (3, 8, 6, 3, 5, 10),
            'targets_pretokenized': 'is a test'
        },
        {
            'inputs': (3, 13, 7, 14, 15, 9, 4, 16),
            'inputs_pretokenized': 'complete: that',
            'targets': (17, 5, 6, 3, 5, 10),
            'targets_pretokenized': 'was a test'
        },
        {
            'inputs': (3, 13, 7, 14, 15, 9, 4, 16),
            'inputs_pretokenized': 'complete: those',
            'targets': (17, 4, 23, 4, 10, 6),
            'targets_pretokenized': 'were tests'
        },
    ],
    # Notice that we repeat consecutively each examples 4 times,
    # this needed for tests like infer_tests to validate determinism.
    'validation': [{
        'inputs': (3, 13, 7, 14, 15, 9, 4, 16),
        'inputs_pretokenized': 'complete: this',
        'targets': (3, 8, 6, 3, 5, 3, 25, 5),
        'targets_pretokenized': 'is a validation',
    }] * 4 + [{
        'inputs': (3, 13, 7, 14, 15, 9, 4, 17),
        'inputs_pretokenized': 'complete: that',
        'targets': (17, 5, 6, 3, 5, 22, 7, 24),
        'targets_pretokenized': 'was another validation',
    }] * 4
}


def get_fake_tokenized_dataset(*_, split='validation', **__):
  """Creates fake dataset compatible with T5X models inputs."""

  if split == 'test':
    split = 'validation'
  output_types = {
      'inputs': tf.int32,
      'targets': tf.int32,
      'inputs_pretokenized': tf.string,
      'targets_pretokenized': tf.string
  }
  output_shapes = {
      'inputs': [None],
      'targets': [None],
      'inputs_pretokenized': [],
      'targets_pretokenized': []
  }
  ds = tf.data.Dataset.from_generator(lambda: _FAKE_TOKENIZED_DATASET[split],
                                      output_types, output_shapes)
  if split == 'train':
    ds = ds.repeat(None)
  return ds


def assert_same(tree_a, tree_b):
  """Asserts that both trees are the same."""
  tree_a, tree_b = jax.device_get((tree_a, tree_b))
  jax.tree_multimap(np.testing.assert_array_equal, tree_a, tree_b)


def get_train_state_from_variables(variables,
                                   optimizer_def=optim.Adafactor(0.0)):
  """Returns a default Train State with Adafactor optimizer."""
  optimizer = optimizer_def.create(variables['params'])
  return train_state_lib.FlaxOptimTrainState(optimizer)
