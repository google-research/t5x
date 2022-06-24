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
from flax.linen import partitioning as flax_partitioning
import jax
import numpy as np
import optax
from t5x import optimizers
from t5x import test_utils
from t5x import train_state as train_state_lib
from t5x.contrib.moe import partitioning as moe_partitioning
from t5x.contrib.moe import training_utils

mock = absltest.mock

AxisMetadata = flax_partitioning.AxisMetadata
DataLayout = moe_partitioning.DataLayout
FlaxOptimTrainState = train_state_lib.FlaxOptimTrainState
FrozenDict = flax_core.frozen_dict.FrozenDict
InferenceState = train_state_lib.InferenceState
PartitionSpec = moe_partitioning.PartitionSpec
PRNGKey = Any


def create_train_state() -> FlaxOptimTrainState:
  """Creates simple Adam optimizer."""
  model_variables = flax_core.freeze({
      'params': {
          'logits_dense': np.ones((16, 16), np.float32),
          'mlp': {
              'wo': {
                  'kernel': np.ones((32, 16), np.float32)
              }
          }
      },
      'params_axes': {
          'logits_dense_axes': AxisMetadata(names=('vocab', 'embed')),
          'mlp': {
              'wo': {
                  'kernel_axes': AxisMetadata(names=('embed', 'mlp'))
              }
          }
      }
  })

  optimizer_def = optimizers.adamw(learning_rate=1e-4)
  return FlaxOptimTrainState.create(optimizer_def, model_variables)


class PartitioningTest(absltest.TestCase):

  @mock.patch('jax.local_devices')
  @mock.patch('jax.devices')
  @mock.patch('jax._src.lib.xla_bridge.process_index')
  def test_default_mesh(self, process_index_fn, devices_fn, local_devices_fn):
    # Mesh with 8 devices.
    devices = test_utils.make_devices(2, 2, 1, 2, kind='TPU v3')
    devices_fn.return_value = devices
    local_devices_fn.return_value = [d for d in devices if d.process_index == 0]
    process_index_fn.return_value = 0

    with self.subTest(name='more_experts_than_devices'):
      mesh = moe_partitioning.default_moe_mesh(num_experts=16, num_partitions=1)
      self.assertEqual(mesh.devices.shape, (1, 8, 1))
      self.assertEqual(mesh.axis_names, ('data', 'expert', 'model'))

    with self.subTest(name='equal_experts_and_devices'):
      mesh = moe_partitioning.default_moe_mesh(num_experts=8, num_partitions=1)
      self.assertEqual(mesh.devices.shape, (1, 8, 1))
      self.assertEqual(mesh.axis_names, ('data', 'expert', 'model'))

    with self.subTest(name='fewer_experts_than_devices'):
      mesh = moe_partitioning.default_moe_mesh(num_experts=4, num_partitions=1)
      self.assertEqual(mesh.devices.shape, (2, 4, 1))
      self.assertEqual(mesh.axis_names, ('data', 'expert', 'model'))

    with self.subTest(name='nontrivial_model_partitions'):
      mesh = moe_partitioning.default_moe_mesh(num_experts=8, num_partitions=4)
      self.assertEqual(mesh.devices.shape, (1, 2, 4))
      self.assertEqual(mesh.axis_names, ('data', 'expert', 'model'))

    with self.subTest(name='specified_model_parallel_submesh'):
      mesh = moe_partitioning.default_moe_mesh(
          num_experts=8, model_parallel_submesh=(1, 1, 1, 2))
      self.assertEqual(mesh.devices.shape, (1, 4, 2))
      self.assertEqual(mesh.axis_names, ('data', 'expert', 'model'))

  def test_gpu_mesh(self):
    mesh = moe_partitioning.get_gpu_mesh()
    self.assertEqual(mesh.devices.shape, (1, jax.device_count(), 1))
    self.assertEqual(mesh.axis_names, ('data', 'expert', 'model'))

  def test_cpu_mesh(self):
    mesh = moe_partitioning.get_cpu_mesh()
    self.assertEqual(mesh.devices.shape, (1, jax.device_count(), 1))
    self.assertEqual(mesh.axis_names, ('data', 'expert', 'model'))

  @mock.patch('jax.local_devices')
  @mock.patch('jax.devices')
  @mock.patch('jax._src.lib.xla_bridge.process_index')
  def test_local_chunker_data_layout(self, process_index_fn, devices_fn,
                                     local_devices_fn):
    # Mesh with 32 devices.
    devices = test_utils.make_devices(4, 4, 1, 2, kind='TPU v3')
    devices_fn.return_value = devices
    local_devices_fn.return_value = [d for d in devices if d.process_index == 0]

    for process_index, shard_id in zip([0, 1, 2, 3], [0, 2, 1, 3]):
      process_index_fn.return_value = process_index
      partitioner = moe_partitioning.MoePjitPartitioner(
          num_experts=8, num_partitions=1)
      self.assertEqual(
          partitioner.get_data_layout(batch_size=32),
          DataLayout(
              batch_size=32,
              shard_id=shard_id,
              num_shards=4,
              is_first_host_in_replica_set=True))

  def test_logical_axes_for_moe_partitioner_no_overrides(self):
    partitioner = moe_partitioning.MoePjitPartitioner(
        num_experts=8,
        num_partitions=1,
        state_filter_fn=training_utils.match_fn(r'no_state_matching'))

    train_state = create_train_state()
    logical_axes = partitioner.get_logical_axes(train_state)

    # No updates to state.
    self.assertEqual(logical_axes.param_states, (optax.ScaleByAdamState(
        count=None,
        mu=FrozenDict({
            'logits_dense': PartitionSpec('vocab', 'embed'),
            'mlp': {
                'wo': {
                    'kernel': PartitionSpec('embed', 'mlp'),
                },
            },
        }),
        nu=FrozenDict({
            'logits_dense': PartitionSpec('vocab', 'embed'),
            'mlp': {
                'wo': {
                    'kernel': PartitionSpec('embed', 'mlp'),
                },
            },
        })), optax.EmptyState(), optax.EmptyState()))

    # Target (params) should be unchanged.
    self.assertEqual(
        logical_axes.params,
        FrozenDict({
            'logits_dense': PartitionSpec('vocab', 'embed'),
            'mlp': {
                'wo': {
                    'kernel': PartitionSpec('embed', 'mlp'),
                },
            },
        }))

  def test_logical_axes_for_moe_partitioner_with_overrides(self):
    partitioner = moe_partitioning.MoePjitPartitioner(
        num_experts=8,
        num_partitions=1,
        state_filter_fn=training_utils.match_fn(r'.*mlp.*'))

    train_state = create_train_state()
    logical_axes = partitioner.get_logical_axes(train_state)

    # 'mlp' params should be prepended with 'expert' spec because
    # state_filter_fn matches '.*mlp.*'.
    self.assertEqual(logical_axes.param_states, (optax.ScaleByAdamState(
        count=None,
        mu=FrozenDict({
            'logits_dense': PartitionSpec('vocab', 'embed'),
            'mlp': {
                'wo': {
                    'kernel': PartitionSpec('expert', 'embed', 'mlp'),
                },
            },
        }),
        nu=FrozenDict({
            'logits_dense': PartitionSpec('vocab', 'embed'),
            'mlp': {
                'wo': {
                    'kernel': PartitionSpec('expert', 'embed', 'mlp'),
                },
            },
        })), optax.EmptyState(), optax.EmptyState()))

    # Target (params) should be unchanged.
    self.assertEqual(
        logical_axes.params,
        FrozenDict({
            'logits_dense': PartitionSpec('vocab', 'embed'),
            'mlp': {
                'wo': {
                    'kernel': PartitionSpec('embed', 'mlp'),
                },
            },
        }))

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

  def test_logical_axis_rules(self):
    self.assertEqual(
        moe_partitioning.standard_logical_axis_rules(
            additional_rules=[('additional', 'model'), ('expert_magic',
                                                        'data')]),
        [
            ('batch', ('data', 'expert')),  # Shard batch over entire mesh
            # No sharding of weights over model axis.
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
            ('expert', 'expert'),  # Shard experts along expert axis
            ('expert_mlp', 'model'),
            ('expert_group', None),
            # Experts replicated along "pure" data axis
            ('expert_replicas', 'data'),
            ('unmodeled', None),
            ('additional', 'model'),
            ('expert_magic', 'data'),
        ])

  def test_data_partition_spec(self):
    partitioner = moe_partitioning.MoePjitPartitioner(
        num_experts=2, num_partitions=1)
    self.assertEqual(partitioner.data_partition_spec,
                     PartitionSpec(('data', 'expert'),))

  def test_axis_resource_overrides(self):
    input_resources = (PartitionSpec('data'), PartitionSpec('model'),
                       PartitionSpec('expert'), None,
                       PartitionSpec('unrecognized'))
    overridden_resources = moe_partitioning._override_partition_specs(
        input_resources)
    # 'data' -> ('data', 'expert').
    self.assertEqual(
        overridden_resources,
        (PartitionSpec(('data', 'expert'),), PartitionSpec('model'),
         PartitionSpec('expert'), None, PartitionSpec('unrecognized',)))

if __name__ == '__main__':
  absltest.main()
