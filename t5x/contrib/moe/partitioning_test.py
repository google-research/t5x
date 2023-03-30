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

"""Tests for partitioning."""

from typing import Any
import unittest

from absl.testing import absltest
import flax
from flax import core as flax_core
from flax.linen import partitioning as flax_partitioning
import jax
import numpy as np
import optax
from t5x import adafactor
from t5x import optimizers
from t5x import partitioning as base_partitioning
from t5x import test_utils
from t5x import train_state as train_state_lib
from t5x.contrib.moe import partitioning as moe_partitioning
from t5x.contrib.moe import training_utils

mock = absltest.mock

AxisMetadata = flax_partitioning.AxisMetadata
DataLayout = moe_partitioning.DataLayout
FlaxOptimTrainState = train_state_lib.FlaxOptimTrainState
FrozenDict = flax_core.frozen_dict.FrozenDict
FrozenVariableDict = flax_core.scope.FrozenVariableDict
InferenceState = train_state_lib.InferenceState
PartitionSpec = moe_partitioning.PartitionSpec
PRNGKey = Any


def create_model_variables() -> FrozenVariableDict:
  """Creates simple model variables."""
  return flax_core.freeze({
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


def create_train_state() -> FlaxOptimTrainState:
  """Creates simple Adam optimizer train state."""
  optimizer_def = optimizers.adamw(learning_rate=1e-4)
  return FlaxOptimTrainState.create(optimizer_def, create_model_variables())


def create_adafactor_train_state(factored: bool = True) -> FlaxOptimTrainState:
  """Creates MultiOptimizer train state."""
  optimizer_def = adafactor.Adafactor(learning_rate=0.1, factored=factored)
  return FlaxOptimTrainState.create(optimizer_def, create_model_variables())


def create_multioptimizer_train_state(
    factored: bool = True) -> FlaxOptimTrainState:
  """Creates MultiOptimizer train state."""

  def _is_mlp(path):
    return 'mlp' in path

  mlp_vars = flax.traverse_util.ModelParamTraversal(
      lambda path, _: not _is_mlp(path))
  non_mlp_vars = flax.traverse_util.ModelParamTraversal(
      lambda path, _: _is_mlp(path))
  scaled_opt = adafactor.Adafactor(learning_rate=0.1, factored=factored)
  unscaled_opt = adafactor.Adafactor(
      learning_rate=0.1, multiply_by_parameter_scale=False, factored=factored)

  optimizer_def = optimizers.MultiOptimizer(
      ((mlp_vars, scaled_opt), (non_mlp_vars, unscaled_opt)))

  return FlaxOptimTrainState.create(optimizer_def, create_model_variables())


class PartitioningTest(absltest.TestCase):

  @unittest.skipIf(jax.__version_info__ < (0, 4, 5), 'Test requires jax 0.4.5')
  @mock.patch('jax.local_devices')
  @mock.patch('jax.devices')
  @mock.patch(f'{jax.process_index.__module__}.process_index')
  def test_default_mesh(self, process_index_fn, devices_fn, local_devices_fn):
    # Mesh with 8 devices.
    devices = test_utils.make_devices(2, 2, 1, 2, kind='TPU v3')
    devices_fn.return_value = devices
    local_devices_fn.return_value = [d for d in devices if d.process_index == 0]
    process_index_fn.return_value = 0

    with self.subTest(name='more_experts_than_devices'):
      mesh = moe_partitioning.default_moe_mesh(
          num_expert_partitions=16, num_partitions=1)
      self.assertEqual(mesh.devices.shape, (1, 8, 1))
      self.assertEqual(mesh.axis_names, ('data', 'expert', 'model'))

    with self.subTest(name='equal_experts_and_devices'):
      mesh = moe_partitioning.default_moe_mesh(
          num_expert_partitions=8, num_partitions=1)
      self.assertEqual(mesh.devices.shape, (1, 8, 1))
      self.assertEqual(mesh.axis_names, ('data', 'expert', 'model'))

    with self.subTest(name='fewer_experts_than_devices'):
      mesh = moe_partitioning.default_moe_mesh(
          num_expert_partitions=4, num_partitions=1)
      self.assertEqual(mesh.devices.shape, (2, 4, 1))
      self.assertEqual(mesh.axis_names, ('data', 'expert', 'model'))

    with self.subTest(name='nontrivial_model_partitions'):
      mesh = moe_partitioning.default_moe_mesh(
          num_expert_partitions=8, num_partitions=4)
      self.assertEqual(mesh.devices.shape, (1, 2, 4))
      self.assertEqual(mesh.axis_names, ('data', 'expert', 'model'))

    with self.subTest(name='specified_model_parallel_submesh'):
      mesh = moe_partitioning.default_moe_mesh(
          num_expert_partitions=8, model_parallel_submesh=(1, 1, 1, 2))
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

  @unittest.skipIf(jax.__version_info__ < (0, 4, 5), 'Test requires jax 0.4.5')
  @mock.patch('jax.local_devices')
  @mock.patch('jax.devices')
  @mock.patch(f'{jax.process_index.__module__}.process_index')
  def test_local_chunker_moe_usage(self, process_index_fn, devices_fn,
                                   local_devices_fn):
    # The MoE partitioning library uses a 2D "data" mesh spanning ('expert',
    # 'data') axes, so we reshape the batch across this 2D "data" mesh when
    # computing replica ids from local chunk info. In this test, we check that
    # the replica ids constructed in this manner are equivalent to the default
    # replica id (over a single 'data' mesh axis).

    # Mesh with 32 devices.
    devices = test_utils.make_devices(2, 2, 1, 2, kind='TPU v3')
    devices_fn.return_value = devices
    local_devices_fn.return_value = [d for d in devices if d.process_index == 0]
    process_index_fn.return_value = 0

    num_expert_partitions = 8
    moe_mesh = moe_partitioning.default_moe_mesh(
        num_expert_partitions=num_expert_partitions, num_partitions=2)
    moe_chunker = base_partitioning.LocalChunker(moe_mesh)

    base_mesh = base_partitioning.default_mesh(num_partitions=2)
    base_chunker = base_partitioning.LocalChunker(base_mesh)

    for batch_size in [8, 16, 32, 64]:
      moe_global_array_shape = (batch_size // num_expert_partitions,
                                num_expert_partitions)
      moe_replica_id = moe_chunker.get_local_chunk_info(
          moe_global_array_shape, ('expert', 'data')).replica_id
      base_global_array_shape = (batch_size,)
      base_replica_id = base_chunker.get_local_chunk_info(
          base_global_array_shape, ('data',)).replica_id
      self.assertEqual(moe_replica_id, base_replica_id)

  @unittest.skipIf(jax.__version_info__ < (0, 4, 5), 'Test requires jax 0.4.5')
  @mock.patch('jax.local_devices')
  @mock.patch('jax.devices')
  @mock.patch(f'{jax.process_index.__module__}.process_index')
  def test_local_chunker_data_layout(self, process_index_fn, devices_fn,
                                     local_devices_fn):
    # Mesh with 32 devices.
    devices = test_utils.make_devices(4, 4, 1, 2, kind='TPU v3')
    devices_fn.return_value = devices
    local_devices_fn.return_value = [d for d in devices if d.process_index == 0]

    for process_index, shard_id in zip([0, 1, 2, 3], [0, 2, 1, 3]):
      process_index_fn.return_value = process_index
      partitioner = moe_partitioning.MoePjitPartitioner(
          num_expert_partitions=8, num_partitions=1)
      self.assertEqual(
          partitioner.get_data_layout(batch_size=32),
          DataLayout(
              batch_size=32,
              shard_id=shard_id,
              num_shards=4,
              is_first_host_in_replica_set=True))

  def test_logical_axes_for_moe_partitioner_no_overrides(self):
    partitioner = moe_partitioning.MoePjitPartitioner(
        num_expert_partitions=8,
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
        num_expert_partitions=8,
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
        num_expert_partitions=8, num_partitions=1)

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

  def test_infer_state_function(self):

    with self.subTest(name='optax'):
      optax_train_state = create_train_state()
      self.assertIsNone(
          moe_partitioning._infer_state_filter_fn(optax_train_state))

    with self.subTest(name='factored_adafactor'):
      adafactor_train_state = create_adafactor_train_state(factored=True)
      match_fn = moe_partitioning._infer_state_filter_fn(adafactor_train_state)
      self.assertTrue(match_fn('expert/kernel/v_col'))
      self.assertTrue(match_fn('expert/kernel/v_row'))
      self.assertFalse(match_fn('expert/kernel/m'))
      self.assertFalse(match_fn('kernel/v_col'))

    with self.subTest(name='unfactored_adafactor'):
      adafactor_train_state = create_adafactor_train_state(factored=False)
      self.assertIsNone(
          moe_partitioning._infer_state_filter_fn(adafactor_train_state))

    with self.subTest(name='factored_adafactor_multi_optimizer'):
      multi_opt_train_state = create_multioptimizer_train_state(factored=True)
      match_fn = moe_partitioning._infer_state_filter_fn(multi_opt_train_state)
      self.assertTrue(match_fn('expert/kernel/v_col'))
      self.assertTrue(match_fn('expert/kernel/v_row'))
      self.assertFalse(match_fn('expert/kernel/m'))
      self.assertFalse(match_fn('kernel/v_col'))

    with self.subTest(name='unfactored_adafactor_multi_optimizer'):
      multi_opt_train_state = create_multioptimizer_train_state(factored=False)
      self.assertIsNone(
          moe_partitioning._infer_state_filter_fn(multi_opt_train_state))

    with self.subTest(name='mixed_factoring_adafactor_multi_optimizer'):
      true_vars = flax.traverse_util.ModelParamTraversal(lambda p, _: True)
      false_vars = flax.traverse_util.ModelParamTraversal(lambda p, _: False)
      factored_opt = adafactor.Adafactor(learning_rate=0.1, factored=True)
      unfactored_opt = adafactor.Adafactor(learning_rate=1., factored=False)
      optimizer_def = optimizers.MultiOptimizer(
          ((true_vars, factored_opt), (false_vars, unfactored_opt)))
      multi_opt_train_state = FlaxOptimTrainState.create(
          optimizer_def, create_model_variables())

      with self.assertRaisesRegex(
          ValueError,
          'all suboptimizers must be either factored or unfactored'):
        _ = moe_partitioning._infer_state_filter_fn(multi_opt_train_state)

  def test_logical_axis_rules(self):
    self.assertEqual(
        moe_partitioning.standard_logical_axis_rules(
            additional_rules=[('additional', 'model'), ('expert_magic',
                                                        'data')]),
        [
            ('batch', ('expert', 'data')),  # Shard batch over entire mesh
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
            # Experts replicated along "pure" data axis
            ('expert_replicas', 'data'),
            ('unmodeled', None),
            ('additional', 'model'),
            ('expert_magic', 'data'),
        ])

  def test_data_partition_spec(self):
    partitioner = moe_partitioning.MoePjitPartitioner(
        num_expert_partitions=2, num_partitions=1)
    self.assertEqual(partitioner.data_partition_spec,
                     PartitionSpec(('expert', 'data'),))

  def test_axis_resource_overrides(self):

    with self.subTest(name='sequence_of_resources'):
      input_resources = (PartitionSpec('data'), PartitionSpec('model'),
                         PartitionSpec('expert'), None,
                         PartitionSpec('unrecognized'))
      # 'data' -> ('expert', 'data').
      self.assertEqual(
          moe_partitioning.override_partition_specs(input_resources),
          (PartitionSpec(('expert', 'data'),), PartitionSpec('model'),
           PartitionSpec('expert'), None, PartitionSpec('unrecognized',)))

    with self.subTest(name='single_resource'):
      # 'data' -> ('expert', 'data').
      self.assertEqual(
          moe_partitioning.override_partition_specs(PartitionSpec('data',)),
          PartitionSpec(('expert', 'data'),))

    with self.subTest(name='no_override'):
      # 'data' -> ('expert', 'data').
      self.assertEqual(
          moe_partitioning.override_partition_specs(
              PartitionSpec(('expert', 'data'))),
          PartitionSpec(('expert', 'data'),))

    with self.subTest(name='no_resource'):
      self.assertIsNone(moe_partitioning.override_partition_specs(None))

  def test_compute_num_model_partitions(self):

    with self.subTest(name='no_model_parallel_submesh'):
      self.assertEqual(
          moe_partitioning.compute_num_model_partitions(
              num_model_partitions=2, model_parallel_submesh=None), 2)

    with self.subTest(name='no_model_partitions'):
      self.assertEqual(
          moe_partitioning.compute_num_model_partitions(
              num_model_partitions=None, model_parallel_submesh=(1, 2, 1, 2)),
          4)

    with self.subTest(name='partitions_and_submesh'):
      self.assertEqual(
          moe_partitioning.compute_num_model_partitions(
              num_model_partitions=4, model_parallel_submesh=(1, 2, 1, 2)), 4)

    with self.subTest(name='inconsistent_partitions'):
      with self.assertRaisesRegex(
          ValueError,
          'num_model_partitions and model_parallel_submesh are inconsistent.'):
        _ = moe_partitioning.compute_num_model_partitions(
            num_model_partitions=1, model_parallel_submesh=(1, 2, 1, 2))

    with self.subTest(name='no_submesh_or_partitions'):
      with self.assertRaisesRegex(
          ValueError,
          'At least one of num_model_partitions and model_parallel_submesh must '
          'be specified.'):
        _ = moe_partitioning.compute_num_model_partitions(
            num_model_partitions=None, model_parallel_submesh=None)


if __name__ == '__main__':
  absltest.main()
