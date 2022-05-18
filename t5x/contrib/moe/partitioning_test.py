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

from t5x.contrib.moe import partitioning as moe_partitioning
from t5x.contrib.moe import training_utils

mock = absltest.mock

AxisMetadata = flax_partitioning.AxisMetadata
DataLayout = moe_partitioning.DataLayout
FlaxOptimTrainState = train_state_lib.FlaxOptimTrainState
InferenceState = train_state_lib.InferenceState
PartitionSpec = moe_partitioning.PartitionSpec
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

  def test_default_data_layout(self):
    # No expert replication required. Use default data layout.
    partitioner = moe_partitioning.MoePjitPartitioner(
        num_experts=8, num_partitions=1)
    self.assertFalse(partitioner.two_data_axes)
    self.assertEqual(
        partitioner.get_data_layout(batch_size=32),
        DataLayout(
            batch_size=32,
            shard_id=0,
            num_shards=1,
            is_first_host_in_replica_set=True))

  def test_two_data_axis_layout_override(self):
    partitioner = moe_partitioning.MoePjitPartitioner(
        num_experts=8, num_partitions=1)
    # Force override case to check layout is valid.
    partitioner.two_data_axes = True
    partitioner._data_axis = ('data', 'model')
    self.assertEqual(
        partitioner.get_data_layout(batch_size=8),
        DataLayout(
            batch_size=8,
            shard_id=0,
            num_shards=1,
            is_first_host_in_replica_set=True))

  def test_logical_axes_for_moe_partitioner_no_overrides(self):
    partitioner = moe_partitioning.MoePjitPartitioner(
        num_experts=8,
        num_partitions=1,
        state_filter_fn=training_utils.match_fn(r'no_state_matching'))

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
    partitioner = moe_partitioning.MoePjitPartitioner(
        num_experts=8,
        num_partitions=1,
        state_filter_fn=training_utils.match_fn(r'.*mlp.*'))

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
    partitioner = moe_partitioning.MoePjitPartitioner(
        num_experts=8, num_partitions=1)

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

    # No expert axis overrides to InferenceState. Partition specs should match
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

  @mock.patch('jax.device_count')
  def test_overridden_logical_axis_rules(self, device_count: int):
    device_count.return_value = 4
    # Fewer experts than devices --> modified axis rules with two 'batch' axes.
    self.assertEqual(
        moe_partitioning.standard_logical_axis_rules(
            num_experts=1,
            num_partitions=1,
            model_parallel_submesh=None,
            additional_rules=[('additional', 'model'),
                              ('expert_magic', 'data')]),
        [
            ('batch', ('data', 'model')),  # Shard batch over entire mesh
            # No sharding of weights over model axis.
            ('vocab', None),
            ('embed', None),
            ('mlp', None),
            ('heads', None),
            ('kv', None),
            ('joined_kv', None),
            ('relpos_buckets', None),
            ('abspos_buckets', None),
            ('length', None),
            ('layers', None),
            ('stack', None),
            ('mlp_activations', None),
            ('expert', 'data'),  # Shard experts over "first" data axis only
            ('expert_mlp', None),
            ('expert_group', None),
            # Experts replicated along "second" data axis
            ('expert_replicas', 'model'),
            ('unmodeled', None),
            ('additional', None),
            ('expert_magic', 'data'),
        ])

  def test_default_logical_axis(self):
    # Model parallelism used --> default logical axis rules.
    self.assertEqual(
        moe_partitioning.standard_logical_axis_rules(
            num_experts=1,
            num_partitions=2,
            model_parallel_submesh=None,
            additional_rules=[('additional', 'model')]),
        [
            ('batch', 'data'),  # Shard batch over single data axis
            # Default model annotations used.
            ('vocab', 'model'),
            ('embed', None),
            ('mlp', 'model'),
            ('heads', 'model'),
            ('kv', None),
            ('joined_kv', 'model'),
            ('relpos_buckets', None),
            ('abspos_buckets', None),
            ('length', None),
            ('layers', None),
            ('stack', None),
            ('mlp_activations', None),
            ('expert', 'data'),  # Shard experts along data axis
            ('expert_mlp', 'model'),
            ('expert_group', None),
            ('expert_replicas', None),
            ('unmodeled', None),
            ('additional', 'model'),
        ])

  def test_2d_parameter_sharding_unsupported(self):
    with self.assertRaisesRegex(ValueError, 'is not supported for MoE.'):
      moe_partitioning.standard_logical_axis_rules(
          num_experts=4, num_partitions=1, parameter_partitioning_dims=2)

  def test_data_partition_spec(self):
    self.assertEqual(
        moe_partitioning.data_partition_spec(two_data_axes=False),
        PartitionSpec('data',))
    self.assertEqual(
        moe_partitioning.data_partition_spec(two_data_axes=True),
        PartitionSpec(('data', 'model'),))

  @mock.patch('jax.device_count')
  def test_when_to_override_model_axis(self, device_count: int):
    device_count.return_value = 4

    # More experts than devices.
    self.assertFalse(
        moe_partitioning._override_model_axis(
            num_experts=8, num_partitions=1, model_parallel_submesh=None))

    # Fewer experts than devices.
    self.assertTrue(
        moe_partitioning._override_model_axis(
            num_experts=1, num_partitions=1, model_parallel_submesh=None))

    # Model parallelism used.
    self.assertFalse(
        moe_partitioning._override_model_axis(
            num_experts=1, num_partitions=2, model_parallel_submesh=None))

  def test_axis_resource_overrides(self):
    input_resources = (PartitionSpec('data'), PartitionSpec('model'), None,
                       PartitionSpec('unrecognized'))
    overridden_resources = moe_partitioning._override_partition_specs(
        input_resources)
    # "data" -> ("data", "model"). "model" -> None.
    self.assertEqual(overridden_resources, (PartitionSpec(
        ('data', 'model'),), None, None, PartitionSpec('unrecognized',)))

if __name__ == '__main__':
  absltest.main()
