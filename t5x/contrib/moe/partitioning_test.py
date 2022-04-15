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

"""Tests for partitioning."""

from typing import Any

from absl.testing import absltest
from flax import core as flax_core
from flax import optim
from flax.linen import partitioning as flax_partitioning
import jax
import numpy as np
from t5x import train_state as train_state_lib

from t5x.contrib.moe import partitioning
from t5x.contrib.moe import training_utils

AxisMetadata = flax_partitioning.AxisMetadata
FlaxOptimTrainState = train_state_lib.FlaxOptimTrainState
InferenceState = train_state_lib.InferenceState
PartitionSpec = partitioning.PartitionSpec
PRNGKey = Any


class LogicalAdam(optim.Adam):
  """Subclass of Adam optimizer with T5X logical axis partitioning support."""

  def derive_logical_axes(self, optimizer_state, param_logical_axes):
    """Derives optimizer logical partitioning from model logical partitions."""
    del param_logical_axes  # Return fixed axes for test
    optimizer_logical_axes = {
        'state': {
            'param_states': {
                'logits_dense': {
                    'grad_ema': None,
                    'grad_sq_ema': None
                },
                'mlp': {
                    'wo': {
                        'kernel': {
                            'grad_ema': PartitionSpec('embed', 'mlp'),
                            'grad_sq_ema': None
                        }
                    }
                }
            },
            'step': None
        },
        'target': {
            'logits_dense': PartitionSpec('vocab', 'embed'),
            'mlp': {
                'wo': {
                    'kernel': PartitionSpec('embed', 'mlp'),
                },
            },
        }
    }
    return optimizer_state.restore_state(optimizer_logical_axes)


def create_optimizer():
  """Creates simple Adam optimizer."""
  target = {
      'logits_dense': np.ones((16, 16), np.float32),
      'mlp': {
          'wo': {
              'kernel': np.ones((32, 16), np.float32)
          }
      }
  }
  return LogicalAdam(learning_rate=1e-4).create(target)


class PartitioningTest(absltest.TestCase):

  def test_logical_axes_for_moe_partitioner(self):
    partitioner = partitioning.MoePjitPartitioner(num_partitions=1)

    optimizer = create_optimizer()
    train_state = FlaxOptimTrainState(
        optimizer,
        params_axes={
            'logits_dense_axes': AxisMetadata(names=('vocab', 'embed')),
            'mlp': {
                'wo': {
                    'kernel_axes': AxisMetadata(names=('embed', 'mlp'))
                }
            }
        })

    logical_axes = partitioner.get_logical_axes(train_state)

    # No updates to state. Should match what derive_logical_axes() returns.
    jax.tree_map(self.assertIsNone, logical_axes.param_states['logits_dense'])
    self.assertEqual(logical_axes.param_states['mlp']['wo']['kernel'].grad_ema,
                     PartitionSpec('embed', 'mlp'))
    self.assertIsNone(
        logical_axes.param_states['mlp']['wo']['kernel'].grad_sq_ema)

    self.assertEqual(
        logical_axes.params, {
            'logits_dense': PartitionSpec('vocab', 'embed'),
            'mlp': {
                'wo': {
                    'kernel': PartitionSpec('embed', 'mlp')
                }
            }
        })

  def test_logical_axes_for_moe_partitioner_with_overrides(self):
    partitioner = partitioning.MoePjitPartitioner(
        num_partitions=1, state_filter_fn=training_utils.match_fn(r'.*mlp.*'))

    optimizer = create_optimizer()
    train_state = FlaxOptimTrainState(
        optimizer,
        params_axes={
            'logits_dense_axes': AxisMetadata(names=('vocab', 'embed')),
            'mlp': {
                'wo': {
                    'kernel_axes': AxisMetadata(names=('embed', 'mlp'))
                }
            }
        })

    logical_axes = partitioner.get_logical_axes(train_state)

    jax.tree_map(self.assertIsNone, logical_axes.param_states['logits_dense'])
    # 'mlp' params should be prepended with 'expert' spec because
    # state_filter_fn matches '.*mlp.*'.
    self.assertEqual(logical_axes.param_states['mlp']['wo']['kernel'].grad_ema,
                     PartitionSpec('expert', 'embed', 'mlp'))
    self.assertEqual(
        logical_axes.param_states['mlp']['wo']['kernel'].grad_sq_ema,
        PartitionSpec('expert',))

    self.assertEqual(
        logical_axes.params, {
            'logits_dense': PartitionSpec('vocab', 'embed'),
            'mlp': {
                'wo': {
                    'kernel': PartitionSpec('embed', 'mlp')
                }
            }
        })

  def test_inference_state_logical_axes(self):
    partitioner = partitioning.MoePjitPartitioner(num_partitions=1)

    model_variables = flax_core.freeze({
        'params': {
            'dense': {
                'bias': np.zeros(4),
                'kernel': np.zeros((2, 4))
            }
        },
        'params_axes': {
            'dense': {
                'bias_axes': AxisMetadata(names=('embed',)),
                'kernel_axes': AxisMetadata(names=('vocab', 'embed')),
            }
        },
    })
    train_state = InferenceState.create(model_variables)
    logical_axes = partitioner.get_logical_axes(train_state)

    # No "expert axis" overrides to InferenceState. Partition specs should match
    # input axis metadata.
    self.assertEqual(
        logical_axes,
        InferenceState(
            step=None,
            params=flax_core.FrozenDict({
                'dense': {
                    'bias': PartitionSpec('embed',),
                    'kernel': PartitionSpec('vocab', 'embed'),
                },
            })))

  def test_logical_axis_rules(self):
    self.assertEqual(partitioning.standard_logical_axis_rules(), (
        ('expert', 'data'),
        ('expert_mlp', 'model'),
        ('expert_group', None),
        ('unmodeled', None),
    ))


if __name__ == '__main__':
  absltest.main()
