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

"""Tests for checkpoints."""

import functools
import itertools
from typing import Any, Mapping, Optional, Tuple
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from t5x import optimizers
from t5x import partitioning as base_partitioning
from t5x import state_utils
from t5x import test_utils
from t5x import train_state as train_state_lib
from t5x.contrib.moe import checkpoints
from t5x.contrib.moe import partitioning as moe_partitioning
import tensorflow as tf

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()

mock = absltest.mock
PartitionSpec = base_partitioning.PartitionSpec

FlaxOptimTrainState = train_state_lib.FlaxOptimTrainState


def create_sharded_array(
    arr: np.ndarray,
    global_mesh: jax.sharding.Mesh,
    mesh_axes: PartitionSpec,
) -> jax.Array:
  """Converts NumPy array into sharded JAX Array."""
  return jax.make_array_from_callback(
      arr.shape,
      jax.sharding.NamedSharding(global_mesh, mesh_axes),
      lambda idx: arr[idx],
  )


def make_train_state(
    *,
    step: Optional[int],
    params: Mapping[str, Any],
    param_states: Mapping[str, Any],
    flax_optimizer_def: optimizers.OptimizerDefType = optimizers.sgd(0.1),
) -> FlaxOptimTrainState:
  """Helper to construct train state for testing."""
  optimizer = optimizers.Optimizer(
      flax_optimizer_def,
      state=optimizers.OptimizerState(step=step, param_states=param_states),
      target=params,
  )
  return FlaxOptimTrainState(optimizer)


def shard_train_state(
    *,
    train_state: FlaxOptimTrainState,
    global_mesh: Optional[jax.sharding.Mesh],
    mesh_axes: Optional[PartitionSpec],
) -> FlaxOptimTrainState:
  """Helper to construct a sharded train state from NumPy arrays."""
  return jax.tree_map(
      functools.partial(
          create_sharded_array, global_mesh=global_mesh, mesh_axes=mesh_axes
      ),
      train_state,
      is_leaf=lambda x: isinstance(x, np.ndarray),
  )


def host_count_to_layout(host_count: int) -> Tuple[int, int, int, int]:
  """Determines the host layout from the host count."""
  return {
      1: (2, 2, 1, 2),
      2: (4, 2, 1, 2),
      4: (4, 4, 1, 2),
      8: (4, 8, 1, 2),
      16: (8, 8, 1, 2),
      32: (8, 16, 1, 2),
  }[host_count]


class CheckpointsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.num_experts = 32

    # The dense model is the checkpointed model that we seek to restore as a
    # sparse model. The dense train state does NOT need to be sharded because
    # it is only used for saving and validation.
    self.dense_model_train_state = make_train_state(
        step=np.int32(42),
        params={
            'mlp': {
                'kernel': np.arange(128, dtype=np.float32).reshape((8, 16)),
            },
            'attention': {
                'kernel': np.arange(64, dtype=np.float32).reshape((8, 8)),
            },
        },
        param_states={
            'mlp': {
                'kernel': 2 * np.arange(64, dtype=np.uint8),
            },
            'attention': {
                'kernel': 3 * np.arange(64, dtype=np.uint8),
            },
        },
    )
    self.dense_model_mesh_axes = make_train_state(
        step=None,
        params={
            'mlp': {
                'kernel': PartitionSpec(None, 'model'),
            },
            'attention': {
                'kernel': PartitionSpec(None, 'model'),
            },
        },
        param_states={
            'mlp': {
                'kernel': None,
            },
            'attention': {
                'kernel': None,
            },
        },
    )

    # The sparse model is the model that we want to restore into. It has two
    # differences relative to the dense model:
    # (1) 'mlp' --> 'expert'
    # (2) 'expert' kernel has self.num_experts copies of the 'mlp' parameters.
    # We will need to shard this train state into a JAX Array.
    self.sparse_model_train_state = make_train_state(
        step=np.int32(42),
        params={
            'expert': {
                'kernel': np.repeat(
                    np.expand_dims(
                        np.arange(128, dtype=np.float32).reshape((8, 16)),
                        axis=0,
                    ),
                    self.num_experts,
                    axis=0,
                ),
            },
            'attention': {
                'kernel': np.arange(64, dtype=np.float32).reshape((8, 8)),
            },
        },
        param_states={
            'expert': {
                'kernel': np.repeat(
                    np.expand_dims(2 * np.arange(64, dtype=np.uint8), axis=0),
                    self.num_experts,
                    axis=0,
                ),
            },
            'attention': {
                'kernel': 3 * np.arange(64, dtype=np.uint8),
            },
        },
    )
    # Axes are the same as the dense model axes, except that we have an
    # additional 'expert' axis for the expert kernels.
    self.sparse_model_mesh_axes = make_train_state(
        step=None,
        params={
            'expert': {
                'kernel': PartitionSpec('expert', None, 'model'),
            },
            'attention': {
                'kernel': PartitionSpec(None, 'model'),
            },
        },
        param_states={
            'expert': {
                'kernel': PartitionSpec('expert', None),
            },
            'attention': {
                'kernel': None,
            },
        },
    )

    self.ds = tf.data.Dataset.range(1024)

    self.checkpoints_dir = self.create_tempdir()
    self.tmp_dir = self.checkpoints_dir.full_path

  @unittest.skipIf(jax.__version_info__ < (0, 4, 5), 'Test requires jax 0.4.5')
  @mock.patch(f'{jax.process_index.__module__}.process_index')
  @mock.patch('jax.devices')
  @mock.patch('jax.local_devices')
  def get_partitioner(
      self,
      process_index,
      host_count,
      num_partitions,
      local_devices_fn,
      devices_fn,
      process_index_fn,
      mesh_axes,
      params_on_devices: bool = True,
  ):
    devices = test_utils.make_devices(*host_count_to_layout(host_count))
    devices_fn.return_value = devices
    local_devices = [d for d in devices if d.process_index == 0]
    local_devices_fn.return_value = local_devices
    process_index_fn.return_value = process_index
    num_partitions_to_mps = {
        1: (1, 1, 1, 1),
        2: (1, 1, 1, 2),
        4: (2, 1, 1, 2),
        16: (4, 2, 1, 2),
    }
    mesh = moe_partitioning.default_moe_mesh(
        num_expert_partitions=self.num_experts,
        num_partitions=num_partitions,
        model_parallel_submesh=num_partitions_to_mps[num_partitions],
    )
    local_chunker = base_partitioning.LocalChunker(mesh)

    class TestPartitioner(base_partitioning.BasePartitioner):

      def __init__(self):
        self.move_params_to_devices_calls = 0
        super().__init__(
            num_partitions, None, params_on_devices=params_on_devices
        )

      @property
      def _local_chunker(self):
        return local_chunker

      @property
      def _mesh(self):
        return mesh

      def partition(
          self,
          fn,
          in_axis_resources,
          out_axis_resources,
          static_argnums=(),
          donate_argnums=(),
      ):
        raise NotImplementedError

      def compile(self, partitioned_fn, *args):
        raise NotImplementedError

      def move_params_to_devices(self, train_state, train_state_axes):
        assert params_on_devices
        return train_state

      def get_mesh_axes(self, train_state):
        return mesh_axes

    return TestPartitioner()

  # pylint:disable=no-value-for-parameter
  @mock.patch(
      'jax.experimental.multihost_utils.sync_global_devices', return_value=None
  )
  @mock.patch('time.time', return_value=0)
  @mock.patch('jax.host_count')
  @mock.patch('jax.process_index')
  def call_host_checkpointer(
      self,
      train_state,
      process_index,
      host_count,
      partitioner,
      fn,
      save_dtype,
      ds_iter,
      mock_process_index,
      mock_host_count,
      unused_mock_host_time,
      unused_mock_sync_devices,
      restore_dtype=np.float32,
  ):
    mock_process_index.return_value = process_index
    mock_host_count.return_value = host_count

    checkpointer = checkpoints.UpcycleCheckpointer(
        train_state,
        partitioner,
        checkpoints_dir=self.tmp_dir,
        num_experts=self.num_experts,
        dataset_iterator=ds_iter,
        save_dtype=save_dtype,
        restore_dtype=restore_dtype,
    )
    return fn(checkpointer)

  def validate_restore(
      self,
      host_count,
      num_partitions,
      step=42,
      checkpoint_dataset=False,
      expected_restore_dtype=np.float32,
      lazy_parameters=False,
  ):
    """Verifies that UpcycleCheckpointer correctly sparsifies checkpoint."""
    global_mesh = test_utils.create_global_mesh(
        host_count_to_layout(host_count), ('data', 'expert', 'model')
    )

    # We want to restore into the sparse model train state.
    sharded_sparse_model_train_state = shard_train_state(
        train_state=self.sparse_model_train_state,
        global_mesh=global_mesh,
        mesh_axes=self.sparse_model_mesh_axes,
    )

    # We map params of saved (dense) model to restored (sparse) model.
    assignment_map = (
        (r'(.*)expert(.*)', r'\1mlp\2'),
        (r'(.*)attention(.*)', r'\1attention\2'),
    )
    # Turn `assignment_map` into a transformation function.
    assignment_map_fn = functools.partial(
        state_utils.apply_assignment_map, assignment_map=assignment_map
    )

    for i in range(host_count):
      partitioner = self.get_partitioner(
          i,
          host_count,
          num_partitions,
          params_on_devices=not lazy_parameters,
          mesh_axes=self.sparse_model_mesh_axes,
      )

      ds_iter = iter(self.ds)

      actual_train_state = self.call_host_checkpointer(
          sharded_sparse_model_train_state,
          i,
          host_count,
          partitioner,
          lambda c: c.restore(  # pylint: disable=g-long-lambda
              step=step,
              lazy_parameters=lazy_parameters,
              state_transformation_fns=(assignment_map_fn,),
          ),
          np.float32,
          ds_iter if checkpoint_dataset else None,
          restore_dtype=expected_restore_dtype,
      )
      if lazy_parameters:
        actual_train_state = jax.tree_map(lambda x: x.get(), actual_train_state)

      # Validate.

      # Optimizer should be the same between actual (sparse) and original
      # (dense) train states.
      self.assertEqual(
          actual_train_state._optimizer.optimizer_def,
          self.dense_model_train_state._optimizer.optimizer_def,
      )
      self.assertEqual(actual_train_state.step, step)
      self.assertEqual(actual_train_state.step.dtype, np.int32)
      self.assertEqual(actual_train_state._optimizer.state.step.dtype, np.int32)

      # Experts are sharded along the 'expert' axis, so each host loads a
      # fraction of the expert parameters.
      experts_per_host = self.num_experts // host_count
      expected_per_host_params = {
          'expert': {
              'kernel': np.repeat(
                  np.expand_dims(
                      np.arange(128, dtype=np.float32).reshape((8, 16)), axis=0
                  ),
                  experts_per_host,
                  axis=0,
              ),
          },
          'attention': {
              'kernel': np.arange(64, dtype=np.float32).reshape((8, 8)),
          },
      }
      expected_per_host_param_states = {
          'expert': {
              'kernel': np.repeat(
                  np.expand_dims(2 * np.arange(64, dtype=np.uint8), axis=0),
                  experts_per_host,
                  axis=0,
              ),
          },
          'attention': {
              'kernel': 3 * np.arange(64, dtype=np.uint8),
          },
      }

      jax.tree_map(
          np.testing.assert_array_equal,
          actual_train_state.params,
          expected_per_host_params,
      )
      jax.tree_map(
          np.testing.assert_array_equal,
          actual_train_state.param_states,
          expected_per_host_param_states,
      )

      self.assertEqual(
          actual_train_state.param_states['attention']['kernel'].dtype, np.uint8
      )
      self.assertEqual(
          actual_train_state.param_states['expert']['kernel'].dtype, np.uint8
      )

      self.assertSameElements(
          actual_train_state.params, ('attention', 'expert')
      )

      self.assertTrue(
          all(
              jax.tree_leaves(
                  jax.tree_map(
                      lambda x: x.dtype == expected_restore_dtype,
                      actual_train_state.params,
                  )
              )
          )
      )

      expected_params = sharded_sparse_model_train_state.params
      expected_param_states = sharded_sparse_model_train_state.param_states

      mlp_slice = partitioner.get_local_chunk_info(
          expected_params['expert']['kernel'].shape, ('expert', None, 'model')
      ).slice
      np.testing.assert_equal(
          actual_train_state.params['expert']['kernel'],
          expected_params['expert']['kernel'][mlp_slice],
      )

      attn_slice = partitioner.get_local_chunk_info(
          expected_params['attention']['kernel'].shape, (None, 'model')
      ).slice
      np.testing.assert_equal(
          actual_train_state.params['attention']['kernel'],
          expected_params['attention']['kernel'][attn_slice],
      )

      mlp_state_slice = partitioner.get_local_chunk_info(
          expected_param_states['expert']['kernel'].shape, ('expert', None)
      ).slice
      np.testing.assert_equal(
          actual_train_state.param_states['expert']['kernel'],
          expected_param_states['expert']['kernel'][mlp_state_slice],
      )

      if checkpoint_dataset:
        ds_shard_id = partitioner.get_data_layout().shard_id
        # The next value from the restored iterator should equal the replica
        # set id.
        self.assertEqual(next(ds_iter).numpy(), ds_shard_id)

  def save(
      self,
      host_count,
      num_partitions,
      step=42,
      save_dtype=np.float32,
      checkpoint_dataset=False,
      disable_partitioning=False,
  ):
    """We do not validate saves; UpcycleCheckpointer only overwrites restore."""
    # We save a dense model. We will try to restore it as a sparse model.
    global_mesh = test_utils.create_global_mesh(
        host_count_to_layout(host_count), ('data', 'model')
    )

    sharded_dense_model_train_state = shard_train_state(
        train_state=self.dense_model_train_state,
        global_mesh=global_mesh,
        mesh_axes=self.dense_model_mesh_axes,
    )

    params = sharded_dense_model_train_state.params
    param_states = sharded_dense_model_train_state.param_states
    optimizer_def = sharded_dense_model_train_state._optimizer.optimizer_def
    # Update these on each save.
    step = np.int32(step)

    # Save the parameters and optimizer states.
    # Each host sets its partition to its host number + 1.
    # Go in reverse since host 0 renames the directory.
    for i in reversed(range(host_count)):
      partitioner = self.get_partitioner(
          i,
          host_count,
          num_partitions,
          mesh_axes=jax.tree_map(lambda x: None, self.dense_model_mesh_axes)
          if disable_partitioning
          else self.dense_model_mesh_axes,
      )
      data_layout = partitioner.get_data_layout()
      ds_shard_id = data_layout.shard_id

      mlp_chunk = partitioner.get_local_chunk_info(
          params['mlp']['kernel'].shape, (None, 'model')
      )
      attn_chunk = partitioner.get_local_chunk_info(
          params['attention']['kernel'].shape, (None, 'model')
      )

      ds_iter = iter(self.ds)

      # pylint:disable=cell-var-from-loop
      def _save_ckpt(checkpointer):
        # Set the checkpoint so that the next value on restore will be the
        # replica set id.
        for _ in range(ds_shard_id):
          next(ds_iter)

        train_state = make_train_state(
            step=step,
            # We save the dense model params.
            params={
                'mlp': {
                    'kernel': params['mlp']['kernel'][mlp_chunk.slice],
                },
                'attention': {
                    'kernel': params['attention']['kernel'][attn_chunk.slice],
                },
            },
            param_states=param_states,
            flax_optimizer_def=optimizer_def,
        )
        checkpointer.save(train_state)

      # pylint:enable=cell-var-from-loop

      # Call host checkpointer with dense model train state.
      self.call_host_checkpointer(
          sharded_dense_model_train_state,
          i,
          host_count,
          partitioner,
          _save_ckpt,
          save_dtype,
          ds_iter if checkpoint_dataset else None,
      )

  # (host_count, num_partitions)
  TOPOLOGIES = [
      (1, 1),  # 1 host, 1 partition
      (1, 2),  # 1 host, 2 partitions
      (2, 1),  # 2 hosts, 1 partition
      (2, 2),  # 2 hosts, 2 partitions
      (4, 4),  # 4 hosts, 4 partitions
      (4, 1),  # 4 hosts, 1 partition
      (4, 2),  # 4 hosts, 2 partitions
      (8, 2),  # 8 hosts, 2 partitions
  ]

  DTYPES = [
      jnp.int32,
      jnp.float32,
      jnp.bfloat16,
      jnp.uint32,
      jnp.int64,
      jnp.float64,
  ]

  @parameterized.parameters(itertools.product(TOPOLOGIES, TOPOLOGIES))
  def test_save_restore(self, save_topology, restore_topology):
    self.save(*save_topology)
    self.validate_restore(*restore_topology)

  @parameterized.parameters(itertools.product(TOPOLOGIES, TOPOLOGIES))
  def test_save_restore_lazy(self, save_topology, restore_topology):
    self.save(*save_topology)
    self.validate_restore(*restore_topology, lazy_parameters=True)

  @parameterized.parameters(TOPOLOGIES)
  def test_save_restore_dataset(self, *topology):
    # Note that we must use the same number of replica sets on save/restore.
    self.save(*topology, checkpoint_dataset=True)
    self.validate_restore(*topology, checkpoint_dataset=True)

  @parameterized.parameters(itertools.product(DTYPES, DTYPES))
  def test_save_as_type(self, save_dtype, restore_dtype):
    self.save(1, 1, save_dtype=save_dtype)
    self.validate_restore(1, 1, expected_restore_dtype=restore_dtype)

  @parameterized.parameters(TOPOLOGIES)
  def test_save_non_partitioned_restore_partitioned(self, *restore_topology):
    # Save without partitioning.
    self.save(2, 1, disable_partitioning=True)
    # Restore with partitioning.
    self.validate_restore(*restore_topology)


if __name__ == '__main__':
  absltest.main()
