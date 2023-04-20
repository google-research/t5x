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

"""Tests for t5x.checkpoints."""
# TODO(b/234480674): Deprecate this test.
import concurrent.futures
import functools
import itertools
import os
from typing import Any, Mapping
import unittest

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from flax import serialization
from flax import traverse_util
from flax.metrics import tensorboard
import jax
import jax.numpy as jnp
import numpy as np
from t5x import checkpoints
from t5x import optimizers
from t5x import partitioning
from t5x import state_utils
from t5x import test_utils
from t5x import train_state as train_state_lib
from t5x import utils
import tensorflow as tf
from tensorflow.io import gfile
import tensorstore as ts

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()

mock = absltest.mock
PartitionSpec = partitioning.PartitionSpec
FLAGS = flags.FLAGS
LazyArray = checkpoints.LazyArray

TESTDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testdata')

FlaxOptimTrainState = train_state_lib.FlaxOptimTrainState


def make_train_state(
    *,
    step: int,
    params: Mapping[str, Any],
    param_states: Mapping[str, Any],
    flax_optimizer_def: optimizers.OptimizerDefType = optimizers.sgd(0.1)
) -> FlaxOptimTrainState:
  """Helper to construct a train state for testing."""
  optimizer = optimizers.Optimizer(
      flax_optimizer_def,
      state=optimizers.OptimizerState(step=step, param_states=param_states),
      target=params)
  return FlaxOptimTrainState(optimizer)


def make_train_state_multi_optimizer(params: Mapping[str, Any],
                                     param_states: Mapping[str, Any],
                                     step: int) -> FlaxOptimTrainState:
  """Helper to construct a train state with multi optimizer for testing."""
  optimizer = optimizers.Optimizer(
      optimizers.MultiOptimizer([
          (traverse_util.ModelParamTraversal(
              lambda path, _: 'kernel' not in path), optimizers.sgd(0.1)),
      ]),
      state=optimizers.OptimizerState(step=step, param_states=param_states),
      target=params)
  return FlaxOptimTrainState(optimizer)


def update_train_state_step(train_state: FlaxOptimTrainState,
                            step: int) -> FlaxOptimTrainState:
  """Helper to update the step inside TrainState."""
  state_dict = train_state.state_dict()
  state_dict['state']['step'] = step
  return train_state.restore_state(state_dict)


class CheckpointChunkShapeTest(absltest.TestCase):

  def test_simple(self):
    self.assertEqual([4096, 4096],
                     checkpoints._choose_chunk_shape([4096, 4096], 4096 * 4096))

    self.assertEqual([4096, 4096],
                     checkpoints._choose_chunk_shape([8192, 8192], 4096 * 4096))

    self.assertEqual([4096, 2731],
                     checkpoints._choose_chunk_shape([8192, 8193], 4096 * 4096))

    self.assertEqual([4096], checkpoints._choose_chunk_shape([8192], 4096))

    self.assertEqual([2731], checkpoints._choose_chunk_shape([8193], 4096))


class LegacyCheckpointsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.train_state = make_train_state(
        step=np.int32(42),
        params={
            'bias': np.arange(4, dtype=jnp.bfloat16).reshape((4, 1)),
            'kernel': np.arange(32, dtype=np.float32).reshape((2, 16))
        },
        param_states={
            'bias': np.int32(1),
            'kernel': np.array([1, 2], np.uint8)
        })
    self.train_state_multi_optimizer = make_train_state_multi_optimizer(
        step=np.int32(42),
        params={
            'bias': np.arange(4, dtype=jnp.bfloat16).reshape((4, 1)),
            'kernel': np.arange(32, dtype=np.float32).reshape((2, 16))
        },
        param_states={
            'bias': np.int32(1),
            'kernel': None
        })
    self.default_mesh_axes = make_train_state(
        step=None,
        params={
            'bias': PartitionSpec('model', None),
            'kernel': PartitionSpec(None, 'model')
        },
        param_states={
            'bias': None,
            'kernel': None
        })

    self.ds = tf.data.Dataset.range(1024)

    self.checkpoints_dir = self.create_tempdir()
    self.tmp_dir = self.checkpoints_dir.full_path

    fake_checkpoints = self.create_tempdir()
    self.fake_checkpoints = fake_checkpoints.full_path
    self.steps = (0, 100, 200)
    for step in self.steps:
      step_dir = fake_checkpoints.mkdir(f'checkpoint_{step}')
      step_dir.create_file('checkpoint')

  @unittest.skipIf(jax.__version_info__ < (0, 4, 5), 'Test requires jax 0.4.5')
  @mock.patch(f'{jax.process_index.__module__}.process_index')
  @mock.patch('jax.devices')
  @mock.patch('jax.local_devices')
  def get_partitioner(self,
                      process_index,
                      host_count,
                      num_partitions,
                      local_devices_fn,
                      devices_fn,
                      process_index_fn,
                      params_on_devices: bool = True,
                      mesh_axes=None):
    host_count_to_layout = {
        1: (2, 2, 1, 2),
        2: (4, 2, 1, 2),
        4: (4, 4, 1, 2),
        8: (4, 8, 1, 2),
        16: (8, 8, 1, 2),
        32: (8, 16, 1, 2)
    }
    devices = test_utils.make_devices(*host_count_to_layout[host_count])
    devices_fn.return_value = devices
    local_devices = [d for d in devices if d.process_index == 0]
    local_devices_fn.return_value = local_devices
    process_index_fn.return_value = process_index
    num_partitions_to_mps = {
        1: (1, 1, 1, 1),
        2: (1, 1, 1, 2),
        4: (2, 1, 1, 2),
        16: (4, 2, 1, 2)
    }
    mesh = partitioning.get_mesh(
        model_parallel_submesh=num_partitions_to_mps[num_partitions],
        input_devices=devices,
        input_local_devices=local_devices)
    mesh_axes = mesh_axes or self.default_mesh_axes
    local_chunker = partitioning.LocalChunker(mesh)

    class TestPartitioner(partitioning.BasePartitioner):

      def __init__(self):
        self.move_params_to_devices_calls = 0
        super().__init__(
            num_partitions, None, params_on_devices=params_on_devices)

      @property
      def _local_chunker(self):
        return local_chunker

      @property
      def _mesh(self):
        return mesh

      def partition(self,
                    fn,
                    in_axis_resources,
                    out_axis_resources,
                    static_argnums=(),
                    donate_argnums=()):
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
      'jax.experimental.multihost_utils.sync_global_devices', return_value=None)
  @mock.patch('time.time', return_value=0)
  @mock.patch('jax.host_count')
  @mock.patch('jax.process_index')
  def call_host_checkpointer(self,
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
                             restore_dtype=np.float32):
    mock_process_index.return_value = process_index
    mock_host_count.return_value = host_count

    checkpointer = checkpoints.Checkpointer(
        self.train_state,
        partitioner,
        self.tmp_dir,
        ds_iter,
        save_dtype=save_dtype,
        restore_dtype=restore_dtype,
        use_gda=False)
    return fn(checkpointer)

  # pylint:disable=no-value-for-parameter
  @mock.patch(
      'jax.experimental.multihost_utils.sync_global_devices', return_value=None)
  @mock.patch('time.time', return_value=0)
  @mock.patch('jax.host_count')
  @mock.patch('jax.process_index')
  def call_host_multioptimizer_checkpointer(self, process_index, host_count,
                                            partitioner, fn, save_dtype,
                                            ds_iter, mock_process_index,
                                            mock_host_count,
                                            unused_mock_host_time,
                                            unused_mock_sync_devices):
    mock_process_index.return_value = process_index
    mock_host_count.return_value = host_count

    checkpointer = checkpoints.Checkpointer(
        self.train_state_multi_optimizer,
        partitioner,
        self.tmp_dir,
        ds_iter,
        save_dtype=save_dtype,
        use_gda=False)
    return fn(checkpointer)

  def test_get_parameter_infos(self):
    train_state = make_train_state(
        params={
            'bias': np.ones((8192, 8192), np.float32),
            'kernel': np.ones((2, 16), np.float32)
        },
        param_states={
            'bias': np.int32(1),
            'kernel': np.array([1, 2])
        },
        step=np.int32(42))
    # host 3 of a 4x4 with mesh 'model' dim == 16
    partitioner = self.get_partitioner(3, 4, 16)
    checkpointer = checkpoints.Checkpointer(
        train_state, partitioner, self.tmp_dir, use_gda=False)

    expected_parameter_infos = {
        'state': {
            'step':
                checkpoints._ParameterInfo(
                    name='state/step', shape=(), ts_spec=None, local_chunk_info=None, axes=None),
            'param_states': {
                'bias':
                    checkpoints._ParameterInfo(
                        name='state/param_states/bias',
                        shape=(),
                        ts_spec=None,
                        local_chunk_info=None, axes=None),
                'kernel':
                    checkpoints._ParameterInfo(
                        name='state/param_states/kernel',
                        shape=(2,),
                        ts_spec=None,
                        local_chunk_info=None, axes=None)
            }
        },
        'target': {
            'bias':
                checkpoints._ParameterInfo(
                    name='target/bias',
                    shape=(8192, 8192),
                    ts_spec=ts.Spec({
                        'driver': 'zarr',
                        'dtype': 'float32',
                        'kvstore': {  # pylint:disable=duplicate-key
                            'driver': 'file',
                            'path': 'target.bias',
                        },
                        'metadata': {
                            'chunks': [4096, 4096],
                            'compressor': {
                                'id': 'gzip'
                            },
                            'shape': [8192, 8192],
                        },
                    }),
                    local_chunk_info=partitioning.LocalChunkInfo(
                        slice=(slice(4096, 8192, None), slice(None, None,
                                                              None)),
                        replica_id=1), axes=PartitionSpec('model', None)),
            'kernel':
                checkpoints._ParameterInfo(
                    name='target/kernel',
                    shape=(2, 16),
                    ts_spec=ts.Spec({
                        'driver': 'zarr',
                        'dtype': 'float32',
                        'kvstore': {  # pylint:disable=duplicate-key
                            'driver': 'file',
                            'path': 'target.kernel',
                        },
                        'metadata': {
                            'chunks': [2, 8],
                            'compressor': {
                                'id': 'gzip'
                            },
                            'shape': [2, 16],
                        },
                    }),
                    local_chunk_info=partitioning.LocalChunkInfo(
                        slice=(slice(None, None, None), slice(8, 16, None)),
                        replica_id=1), axes=PartitionSpec(None, 'model'))
        }
    }  # pyformat: disable
    jax.tree_map(self.assertEqual, checkpointer._get_parameter_infos(),
                 expected_parameter_infos)

  def test_get_multioptimizer_parameter_infos(self):
    train_state = make_train_state(
        step=np.int32(42),
        params={
            'bias': np.ones((8192, 8192), jnp.bfloat16),
            'kernel': np.ones((2, 16), np.float32)
        },
        param_states={
            'bias': np.int32(1),
            # The parameter state for Kernel is `None` as if we have a
            # multioptimizer that is not updating this parameter.
            'kernel': None
        })
    # host 3 of a 4x4 with mesh 'model' dim == 16
    partitioner = self.get_partitioner(3, 4, 16)
    checkpointer = checkpoints.Checkpointer(
        train_state, partitioner, self.tmp_dir, use_gda=False)
    kernel_state_info = (
        checkpointer._get_parameter_infos()['state']['param_states']['kernel'])
    self.assertIsNone(kernel_state_info)

  def test_all_steps(self):
    partitioner = self.get_partitioner(0, 1, 1)
    checkpointer = self.call_host_checkpointer(0, 1, partitioner, lambda c: c,
                                               np.float32, None)

    self.assertIsNone(checkpointer.latest_step())
    for step in ['0', '42', '10', '999.tmp-0', '100']:
      d = os.path.join(checkpointer.checkpoints_dir, f'checkpoint_{step}')
      gfile.makedirs(d)
      ckpt = os.path.join(d, 'checkpoint')
      with gfile.GFile(ckpt, 'w') as f:
        f.write('')
    self.assertSequenceEqual(
        checkpoints.all_steps(checkpointer.checkpoints_dir + '/'),
        [0, 10, 42, 100])

  def test_all_latest_step(self):
    partitioner = self.get_partitioner(0, 1, 1)
    checkpointer = self.call_host_checkpointer(0, 1, partitioner, lambda c: c,
                                               np.float32, None)

    self.assertIsNone(checkpointer.latest_step())

    for step in ['0', '42', '10', '999.tmp-0', '100']:
      d = os.path.join(checkpointer.checkpoints_dir, f'checkpoint_{step}')
      gfile.makedirs(d)
      ckpt = os.path.join(d, 'checkpoint')
      with gfile.GFile(ckpt, 'w') as f:
        f.write('')

    self.assertSequenceEqual(checkpointer.all_steps(), [0, 10, 42, 100])
    self.assertEqual(checkpointer.latest_step(), 100)

    # Remove checkpoint file for step 100 (but leave directory).
    gfile.remove(ckpt)
    self.assertSequenceEqual(checkpointer.all_steps(), [0, 10, 42])
    self.assertEqual(checkpointer.latest_step(), 42)

  def test_all_latest_step_public(self):
    self.assertIsNone(checkpoints.latest_step(self.tmp_dir))

    for step in ['0', '42', '10', '999.tmp-0', '100']:
      d = os.path.join(self.tmp_dir, f'checkpoint_{step}')
      gfile.makedirs(d)
      ckpt = os.path.join(d, 'checkpoint')
      with gfile.GFile(ckpt, 'w') as f:
        f.write('')

    self.assertSequenceEqual(
        checkpoints.all_steps(self.tmp_dir), [0, 10, 42, 100])
    self.assertEqual(checkpoints.latest_step(self.tmp_dir), 100)

    # Remove checkpoint file for step 100 (but leave directory).
    gfile.remove(ckpt)
    self.assertSequenceEqual(checkpoints.all_steps(self.tmp_dir), [0, 10, 42])
    self.assertEqual(checkpoints.latest_step(self.tmp_dir), 42)

  def validate_restore(self,
                       host_count,
                       num_partitions,
                       step=42,
                       checkpoint_dataset=False,
                       expected_restore_dtype=np.float32,
                       lazy_parameters=False,
                       disable_partitioning=False):
    params = self.train_state.params
    param_states = self.train_state.param_states

    for i in range(host_count):
      partitioner = self.get_partitioner(
          i,
          host_count,
          num_partitions,
          params_on_devices=not lazy_parameters,
          mesh_axes=jax.tree_map(lambda x: None, self.default_mesh_axes)
          if disable_partitioning else None)
      ds_shard_id = partitioner.get_data_layout().shard_id

      bias_slice = partitioner.get_local_chunk_info(params['bias'].shape,
                                                    ('model', None)).slice
      kernel_slice = partitioner.get_local_chunk_info(params['kernel'].shape,
                                                      (None, 'model')).slice

      ds_iter = iter(self.ds)

      actual_train_state = self.call_host_checkpointer(
          i,
          host_count,
          partitioner,
          lambda c: c.restore(  # pylint: disable=g-long-lambda
              step=step,
              lazy_parameters=lazy_parameters),
          np.float32,
          ds_iter if checkpoint_dataset else None,
          restore_dtype=expected_restore_dtype)
      if lazy_parameters:
        actual_train_state = jax.tree_map(lambda x: x.get(), actual_train_state)
      self.assertEqual(actual_train_state._optimizer.optimizer_def,
                       self.train_state._optimizer.optimizer_def)

      self.assertEqual(actual_train_state.step, step)
      self.assertEqual(actual_train_state.step.dtype, np.int32)
      self.assertEqual(actual_train_state._optimizer.state.step.dtype, np.int32)
      jax.tree_map(np.testing.assert_array_equal,
                   actual_train_state.param_states, param_states)
      self.assertEqual(actual_train_state.param_states['kernel'].dtype,
                       np.uint8)
      self.assertSameElements(actual_train_state.params, ('bias', 'kernel'))
      self.assertTrue(
          all(
              jax.tree_leaves(
                  jax.tree_map(lambda x: x.dtype == expected_restore_dtype,
                               actual_train_state.params))))
      np.testing.assert_equal(actual_train_state.params['bias'],
                              params['bias'][bias_slice])
      np.testing.assert_equal(actual_train_state.params['kernel'],
                              params['kernel'][kernel_slice])
      if checkpoint_dataset:
        # The next value from the restored iterator should equal the
        # replica set id.
        self.assertEqual(next(ds_iter).numpy(), ds_shard_id)

  def validate_multioptimizer_restore(self,
                                      host_count,
                                      num_partitions,
                                      step=42,
                                      checkpoint_dataset=False,
                                      expected_restore_dtype=np.float32):
    params = self.train_state_multi_optimizer.params
    param_states = self.train_state_multi_optimizer.param_states

    for i in range(host_count):
      partitioner = self.get_partitioner(i, host_count, num_partitions)
      ds_shard_id = partitioner.get_data_layout().shard_id

      bias_slice = partitioner.get_local_chunk_info(params['bias'].shape,
                                                    ('model', None)).slice
      kernel_slice = partitioner.get_local_chunk_info(params['kernel'].shape,
                                                      (None, 'model')).slice

      ds_iter = iter(self.ds)

      actual_train_state = self.call_host_multioptimizer_checkpointer(
          i, host_count, partitioner, lambda c: c.restore(step=step),
          np.float32, ds_iter if checkpoint_dataset else None)
      actual_optimizer = actual_train_state._optimizer  # pylint: disable=protected-access
      actual_step = actual_train_state.step
      actual_params = actual_train_state.params
      actual_param_states = actual_train_state.param_states
      self.assertEqual(
          actual_optimizer.optimizer_def,
          self.train_state_multi_optimizer._optimizer.optimizer_def)
      self.assertEqual(actual_optimizer.state.step.dtype, np.int32)
      jax.tree_map(lambda x: self.assertEqual(x.dtype, expected_restore_dtype),
                   actual_optimizer.target)
      self.assertEqual(actual_step, step)
      self.assertEqual(actual_step.dtype, np.int32)
      jax.tree_map(np.testing.assert_array_equal, actual_param_states,
                   param_states)
      self.assertSameElements(actual_params, ('bias', 'kernel'))
      self.assertTrue(
          all(
              jax.tree_leaves(
                  jax.tree_map(lambda x: x.dtype == expected_restore_dtype,
                               actual_params))))
      np.testing.assert_equal(actual_params['bias'], params['bias'][bias_slice])
      np.testing.assert_equal(actual_params['kernel'],
                              params['kernel'][kernel_slice])
      if checkpoint_dataset:
        # The next value from the restored iterator should equal the
        # replica set id.
        self.assertEqual(next(ds_iter).numpy(), ds_shard_id)

  def validate_save(self,
                    host_count,
                    num_partitions,
                    step=42,
                    save_dtype=np.float32,
                    checkpoint_dataset=False,
                    multi_optimizer=False,
                    disable_partitioning=False):
    if multi_optimizer:
      params = self.train_state_multi_optimizer.params
      param_states = self.train_state_multi_optimizer.param_states
      optimizer_def = self.train_state_multi_optimizer._optimizer.optimizer_def
    else:
      params = self.train_state.params
      param_states = self.train_state.param_states
      optimizer_def = self.train_state._optimizer.optimizer_def
    # Update these on each save.
    step = np.int32(step)
    expected_bias = np.zeros((4, 1), save_dtype)
    expected_kernel = np.zeros((2, 16), save_dtype)

    bias_tspec = {
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': f'{self.tmp_dir}/checkpoint_{step}.tmp-0/target.bias',
        }
    }
    kernel_tspec = {
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': f'{self.tmp_dir}/checkpoint_{step}.tmp-0/target.kernel',
        }
    }

    # Test save.
    # Each host sets its partition to its host number + 1.
    # Go in reverse since host 0 renames the directory.
    for i in reversed(range(host_count)):
      partitioner = self.get_partitioner(
          i,
          host_count,
          num_partitions,
          mesh_axes=jax.tree_map(lambda x: None, self.default_mesh_axes)
          if disable_partitioning else None)
      data_layout = partitioner.get_data_layout()
      num_ds_shards = data_layout.num_shards
      ds_shard_id = data_layout.shard_id
      chunk_id_for_shard = partitioner.get_local_chunk_info(
          jnp.ones((num_ds_shards,)), ['data']).replica_id

      bias_chunk = partitioner.get_local_chunk_info(params['bias'].shape,
                                                    ('model', None))
      kernel_chunk = partitioner.get_local_chunk_info(params['kernel'].shape,
                                                      (None, 'model'))

      ds_iter = iter(self.ds)

      # pylint:disable=cell-var-from-loop
      def _save_ckpt(checkpointer):
        # Set the checkpoint so that the next value on restore will be the
        # replica set id.
        for _ in range(ds_shard_id):
          next(ds_iter)

        train_state = make_train_state(
            step=step,
            params={
                'bias': params['bias'][bias_chunk.slice],
                'kernel': params['kernel'][kernel_chunk.slice]
            },
            param_states=param_states,
            flax_optimizer_def=optimizer_def)
        checkpointer.save(train_state)

      # pylint:enable=cell-var-from-loop

      self.call_host_checkpointer(i, host_count, partitioner, _save_ckpt,
                                  save_dtype,
                                  ds_iter if checkpoint_dataset else None)

      if disable_partitioning:
        continue

      # Read the current TensorStore.
      if i == 0:
        # Host 0 moves the files.
        bias_tspec['kvstore']['path'] = (
            bias_tspec['kvstore']['path'].replace('.tmp-0', ''))
        kernel_tspec['kvstore']['path'] = (
            kernel_tspec['kvstore']['path'].replace('.tmp-0', ''))

      if checkpoint_dataset:
        ckpt_dir = f'{self.tmp_dir}/checkpoint_{step}'
        if i != 0:
          ckpt_dir += '.tmp-0'
        ds_ckpt_glob = gfile.glob(ckpt_dir + '/train_ds-' +
                                  f'{ds_shard_id:03}-of-{num_ds_shards:03}*')
        if chunk_id_for_shard == 0:
          self.assertLen(ds_ckpt_glob, 2)
        else:
          self.assertEmpty(ds_ckpt_glob)

      # only replica_id=0 is saved for each array chunk
      if bias_chunk.replica_id == 0:
        current_bias = ts.open(bias_tspec).result().read().result().view(
            save_dtype)
        expected_bias[bias_chunk.slice] = (params['bias'][bias_chunk.slice])
        np.testing.assert_equal(current_bias, expected_bias)

      if kernel_chunk.replica_id == 0:
        current_kernel = ts.open(kernel_tspec).result().read().result().view(
            save_dtype)
        expected_kernel[kernel_chunk.slice] = (
            params['kernel'][kernel_chunk.slice])
        np.testing.assert_equal(current_kernel, expected_kernel)

    with gfile.GFile(f'{self.tmp_dir}/checkpoint_{step}/checkpoint', 'rb') as f:
      ckpt_contents = serialization.msgpack_restore(f.read())
    self.assertEqual(ckpt_contents['version'], checkpoints.VERSION)
    jax.tree_map(np.testing.assert_allclose,
                 ckpt_contents['optimizer']['state']['param_states'],
                 param_states)
    self.assertEqual(ckpt_contents['optimizer']['state']['step'].dtype,
                     np.int32)
    if disable_partitioning:
      # Parameters should also be in the msgpack checkpoint file.
      jax.tree_map(np.testing.assert_allclose,
                   ckpt_contents['optimizer']['target'],
                   jax.tree_map(lambda arr: arr.astype(save_dtype), params))

    # Jax tree maps ignore Nones so actually check this value is None
    if multi_optimizer:
      self.assertIsNone(
          ckpt_contents['optimizer']['state']['param_states']['kernel'])

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
      jnp.int32, jnp.float32, jnp.bfloat16, jnp.uint32, jnp.int64, jnp.float64
  ]

  @parameterized.parameters(itertools.product(TOPOLOGIES, TOPOLOGIES))
  def test_save_restore(self, save_topology, restore_topology):
    self.validate_save(*save_topology)
    self.validate_restore(*restore_topology)

  @parameterized.parameters(itertools.product(TOPOLOGIES, TOPOLOGIES))
  def test_save_restore_lazy(self, save_topology, restore_topology):
    self.validate_save(*save_topology)
    self.validate_restore(*restore_topology, lazy_parameters=True)

  @parameterized.parameters(itertools.product(TOPOLOGIES, TOPOLOGIES))
  def test_save_multioptimizer_restore(self, save_topology, restore_topology):
    self.validate_save(*save_topology)
    self.validate_multioptimizer_restore(*restore_topology)

  @parameterized.parameters(itertools.product(TOPOLOGIES, TOPOLOGIES))
  def test_multioptimizer_save_multioptimizer_restore(self, save_topology,
                                                      restore_topology):
    self.validate_save(*save_topology, multi_optimizer=True)
    self.validate_multioptimizer_restore(*restore_topology)

  def test_load_t5x_checkpoint(self):
    self.validate_save(1, 1)
    ckpt = checkpoints.load_t5x_checkpoint(self.tmp_dir)
    jax.tree_map(np.testing.assert_array_equal, self.train_state.state_dict(),
                 ckpt)

  def test_load_t5x_checkpoint_of_multioptimizer(self):
    self.validate_save(1, 1, multi_optimizer=True)
    ckpt = checkpoints.load_t5x_checkpoint(self.tmp_dir)
    jax.tree_map(np.testing.assert_array_equal,
                 self.train_state_multi_optimizer.state_dict(), ckpt)
    # Jax tree maps ignore Nones so actually check this value is None
    self.assertIsNone(ckpt['state']['param_states']['kernel'])

  def test_load_t5x_checkpoint_lazy(self):
    self.validate_save(1, 1)
    ckpt = checkpoints.load_t5x_checkpoint(self.tmp_dir)
    lazy_ckpt = checkpoints.load_t5x_checkpoint(
        self.tmp_dir, lazy_parameters=True)
    lazy_loaded_ckpt = jax.tree_map(lambda x: x.get(), lazy_ckpt)
    jax.tree_map(np.testing.assert_array_equal, ckpt, lazy_loaded_ckpt)

  def test_load_t5x_checkpoint_of_multioptimizer_lazy(self):
    self.validate_save(1, 1, multi_optimizer=True)
    ckpt = checkpoints.load_t5x_checkpoint(self.tmp_dir)
    lazy_ckpt = checkpoints.load_t5x_checkpoint(
        self.tmp_dir, lazy_parameters=True)
    lazy_loaded_ckpt = jax.tree_map(lambda x: x.get(), lazy_ckpt)
    jax.tree_map(np.testing.assert_array_equal, ckpt, lazy_loaded_ckpt)
    # Jax tree maps ignore Nones so actually check this value is None
    self.assertIsNone(lazy_loaded_ckpt['state']['param_states']['kernel'])

  @parameterized.parameters(TOPOLOGIES)
  def test_save_restore_dataset(self, *topology):
    # Note that we must use the same number of replica sets on save/restore.
    self.validate_save(*topology, checkpoint_dataset=True)
    self.validate_restore(*topology, checkpoint_dataset=True)

  @parameterized.parameters(itertools.product(DTYPES, DTYPES))
  def test_save_as_type(self, save_dtype, restore_dtype):
    self.validate_save(1, 1, save_dtype=save_dtype)
    self.validate_restore(1, 1, expected_restore_dtype=restore_dtype)

  @parameterized.parameters(TOPOLOGIES)
  def test_reload_wrong_shape(self, *restore_topology):
    self.validate_save(1, 1)
    self.train_state = make_train_state(
        step=np.int32(42),
        params={
            'bias': np.arange(4, dtype=jnp.bfloat16).reshape((4, 1)),
            'kernel': np.arange(32, dtype=np.float32).reshape((4, 8))
        },
        param_states={
            'bias': np.int32(1),
            'kernel': np.array([1, 2])
        })
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'Shape of `target/kernel` in checkpoint (2, 16) does not match '
        'expected (4, 8).'):
      self.validate_restore(*restore_topology)

  @parameterized.parameters(TOPOLOGIES)
  def test_save_partitioned_restore_non_partitioned(self, *restore_topology):
    # Save with default partitioning.
    self.validate_save(2, 2)
    # Restore without partitioning.
    self.validate_restore(*restore_topology, disable_partitioning=True)

  @parameterized.parameters(TOPOLOGIES)
  def test_save_non_partitioned_restore_partitioned(self, *restore_topology):
    # Save without partitioning.
    self.validate_save(2, 1, disable_partitioning=True)
    # Restore with partitioning.
    self.validate_restore(*restore_topology)

  @parameterized.parameters(TOPOLOGIES)
  def test_save_non_partitioned_restore_non_partitioned(self,
                                                        *restore_topology):
    # Save without partitioning.
    self.validate_save(2, 1, disable_partitioning=True)
    # Restore with partitioning.
    self.validate_restore(*restore_topology, disable_partitioning=True)

  @mock.patch('time.time', return_value=0)
  def test_keep(self, unused_mock_time):
    no_partitions_partitioner = self.get_partitioner(0, 1, 1)
    train_state = self.train_state
    checkpointer = checkpoints.Checkpointer(
        train_state,
        no_partitions_partitioner,
        self.tmp_dir,
        keep=2,
        use_gda=False)

    checkpointer.save(update_train_state_step(train_state, 42))
    self.assertSequenceEqual(checkpointer.all_steps(), [42])

    checkpointer.save(update_train_state_step(train_state, 43))
    self.assertSequenceEqual(checkpointer.all_steps(), [42, 43])

    checkpointer.save(update_train_state_step(train_state, 44))
    self.assertSequenceEqual(checkpointer.all_steps(), [43, 44])

    checkpointer.keep = 1
    checkpointer.save(update_train_state_step(train_state, 45))
    self.assertSequenceEqual(checkpointer.all_steps(), [45])

    checkpointer.keep = 3
    checkpointer.save(update_train_state_step(train_state, 46))
    self.assertSequenceEqual(checkpointer.all_steps(), [45, 46])

  @mock.patch('time.time', return_value=0)
  def test_keep_pinned(self, unused_mock_time):
    no_partitions_partitioner = self.get_partitioner(0, 1, 1)
    train_state = self.train_state
    checkpointer = checkpoints.Checkpointer(
        train_state,
        no_partitions_partitioner,
        self.tmp_dir,
        keep=1,
        use_gda=False)

    checkpointer.save(update_train_state_step(train_state, 42))
    self.assertSequenceEqual(checkpointer.all_steps(), [42])

    # Mark the checkpoint as pinned by creating the ALWAYS KEEP file.
    ckpt_dir = self.checkpoints_dir.mkdir(f'checkpoint_{42}')
    ckpt_dir.create_file('PINNED')

    checkpointer.save(update_train_state_step(train_state, 43))

    # Assert both the pinned and the most recent checkpoints are saved.
    self.assertSequenceEqual(checkpointer.all_steps(), [42, 43])

    checkpointer.save(update_train_state_step(train_state, 44))

    # Assert the non-pinned checkpoint gets deleted, but the pinned and the most
    # recent one are still saved.
    self.assertSequenceEqual(checkpointer.all_steps(), [42, 44])

  @mock.patch('time.time', return_value=0)
  def test_keep_dataset_checkpoints(self, unused_mock_time):
    no_partitions_partitioner = self.get_partitioner(0, 1, 1)
    train_state = self.train_state
    dataset_iterator = iter(tf.data.Dataset.range(10))
    checkpointer = checkpoints.Checkpointer(
        train_state,
        no_partitions_partitioner,
        self.tmp_dir,
        dataset_iterator=dataset_iterator,
        keep=2,
        keep_dataset_checkpoints=1,
        use_gda=False)

    checkpointer.save(update_train_state_step(train_state, 42))
    self.assertSequenceEqual(checkpointer.all_steps(), [42])
    self.assertSequenceEqual(checkpointer.all_dataset_checkpoint_steps(), [42])

    checkpointer.save(update_train_state_step(train_state, 43))
    self.assertSequenceEqual(checkpointer.all_steps(), [42, 43])
    self.assertSequenceEqual(checkpointer.all_dataset_checkpoint_steps(), [43])

    checkpointer.save(update_train_state_step(train_state, 44))
    self.assertSequenceEqual(checkpointer.all_steps(), [43, 44])
    self.assertSequenceEqual(checkpointer.all_dataset_checkpoint_steps(), [44])

    checkpointer.keep = 1
    checkpointer.save(update_train_state_step(train_state, 45))
    self.assertSequenceEqual(checkpointer.all_steps(), [45])
    self.assertSequenceEqual(checkpointer.all_dataset_checkpoint_steps(), [45])

    checkpointer.keep = 3
    checkpointer.save(update_train_state_step(train_state, 46))
    self.assertSequenceEqual(checkpointer.all_steps(), [45, 46])
    self.assertSequenceEqual(checkpointer.all_dataset_checkpoint_steps(), [46])

  @mock.patch('time.time', return_value=0)
  def test_keep_dataset_checkpoints_pinned(self, unused_mock_time):
    no_partitions_partitioner = self.get_partitioner(0, 1, 1)
    train_state = self.train_state
    dataset_iterator = iter(tf.data.Dataset.range(10))
    checkpointer = checkpoints.Checkpointer(
        train_state,
        no_partitions_partitioner,
        self.tmp_dir,
        dataset_iterator=dataset_iterator,
        keep=1,
        keep_dataset_checkpoints=1,
        use_gda=False)

    checkpointer.save(update_train_state_step(train_state, 42))
    self.assertSequenceEqual(checkpointer.all_steps(), [42])

    # Mark the checkpoint as pinned by creating the ALWAYS KEEP file.
    ckpt_dir = self.checkpoints_dir.mkdir(f'checkpoint_{42}')
    ckpt_dir.create_file('PINNED')

    checkpointer.save(update_train_state_step(train_state, 43))

    # Assert both the pinned and the most recent checkpoints are saved.
    self.assertSequenceEqual(checkpointer.all_steps(), [42, 43])
    self.assertSequenceEqual(checkpointer.all_dataset_checkpoint_steps(),
                             [42, 43])

    checkpointer.save(update_train_state_step(train_state, 44))

    # Assert the non-pinned checkpoint gets deleted, but the pinned and the most
    # recent one are still saved.
    self.assertSequenceEqual(checkpointer.all_steps(), [42, 44])
    self.assertSequenceEqual(checkpointer.all_dataset_checkpoint_steps(),
                             [42, 44])

  @mock.patch('time.time', return_value=0)
  def test_keep_with_save_best_checkpointer(self, unused_mock_time):
    no_partitions_partitioner = self.get_partitioner(0, 1, 1)
    train_state = self.train_state

    checkpointer = checkpoints.SaveBestCheckpointer(
        train_state,
        no_partitions_partitioner,
        self.tmp_dir,
        keep=2,
        metric_name_to_monitor='train/accuracy',
        metric_mode='max',
        keep_checkpoints_without_metrics=False,
        use_gda=False)

    # Test that without a valid set of metrics deletion falls back to oldest
    # step (since keep_checkpoints_without_metrics is set to False).
    checkpointer.save(update_train_state_step(train_state, 41))
    self.assertSequenceEqual(checkpointer.all_steps(), [41])
    checkpointer.save(update_train_state_step(train_state, 42))
    self.assertSequenceEqual(checkpointer.all_steps(), [41, 42])
    checkpointer.save(update_train_state_step(train_state, 43))
    self.assertSequenceEqual(checkpointer.all_steps(), [41, 42, 43])
    checkpointer.save(update_train_state_step(train_state, 44))
    self.assertSequenceEqual(checkpointer.all_steps(), [42, 43, 44])

    # Now create some metrics for steps 42, 43 and 44.
    summary_writer = tensorboard.SummaryWriter(
        os.path.join(self.tmp_dir, 'train'))
    summary_writer.scalar('accuracy', 0.9, 42)
    summary_writer.scalar('accuracy', 0.8, 43)
    summary_writer.scalar('accuracy', 0.7, 44)

    # Verify that both the newest (without a metrics) and best accuracy
    # checkpoints are kept.
    checkpointer.save(update_train_state_step(train_state, 45))
    self.assertSequenceEqual(checkpointer.all_steps(), [42, 43, 45])

    # Change mode to `min` and check that the checkpoints with highest accuracy
    # are removed.
    checkpointer._metric_mode = 'min'

    # Add metrics to newly created checkpoint as well as a new checkpoint.
    summary_writer.scalar('accuracy', 0.95, 45)
    checkpointer.save(update_train_state_step(train_state, 46))
    summary_writer.scalar('accuracy', 0.99, 46)
    checkpointer.save(update_train_state_step(train_state, 47))
    self.assertSequenceEqual(checkpointer.all_steps(), [42, 43, 47])

  @mock.patch('time.time', return_value=0)
  def test_keep_pinned_save_best_checkpointer(self, unused_mock_time):
    no_partitions_partitioner = self.get_partitioner(0, 1, 1)
    train_state = self.train_state

    checkpointer = checkpoints.SaveBestCheckpointer(
        train_state,
        no_partitions_partitioner,
        self.tmp_dir,
        keep=2,
        metric_name_to_monitor='train/accuracy',
        metric_mode='max',
        keep_checkpoints_without_metrics=False,
        use_gda=False)

    summary_writer = tensorboard.SummaryWriter(
        os.path.join(self.tmp_dir, 'train'))

    checkpointer.save(update_train_state_step(train_state, 42))
    summary_writer.scalar('accuracy', 0.9, 42)
    checkpointer.save(update_train_state_step(train_state, 43))
    summary_writer.scalar('accuracy', 0.7, 43)
    checkpointer.save(update_train_state_step(train_state, 44))
    summary_writer.scalar('accuracy', 0.8, 44)
    self.assertSequenceEqual(checkpointer.all_steps(), [42, 43, 44])

    # Mark checkpoint 43 as always keep.
    ckpt_dir = self.checkpoints_dir.mkdir(f'checkpoint_{43}')
    always_keep_ckpt_43 = ckpt_dir.create_file('PINNED')

    # Verify that the pinned checkpoint 43 is always saved even though it does
    # not have the best metrics, and keep = 2.
    checkpointer.save(update_train_state_step(train_state, 45))
    self.assertSequenceEqual(checkpointer.all_steps(), [42, 43, 44, 45])
    checkpointer.save(update_train_state_step(train_state, 46))
    summary_writer.scalar('accuracy', 0.6, 46)

    # Remove the ALWAYS KEEP file for checkpoint 43.
    gfile.rmtree(always_keep_ckpt_43.full_path)

    # Checkpoint 43 should get deleted in the next update since it is not
    # pinned and does not have the best metrics.
    checkpointer.save(update_train_state_step(train_state, 47))
    self.assertSequenceEqual(checkpointer.all_steps(), [42, 44, 47])

  @mock.patch('time.time', return_value=0)
  def test_keep_pinned_save_best_checkpointer_missing_metrics(
      self, unused_mock_time):
    """Test for `keep_checkpoints_without_metrics` behavior."""
    no_partitions_partitioner = self.get_partitioner(0, 1, 1)
    train_state = self.train_state

    # Use SaveBestCheckpointer with default keep_checkpoints_without_metrics.
    checkpointer = checkpoints.SaveBestCheckpointer(
        train_state,
        no_partitions_partitioner,
        self.tmp_dir,
        keep=1,
        metric_name_to_monitor='train/accuracy',
        metric_mode='max',
        use_gda=False)

    # Pre-create metrics for only some of the steps.
    summary_writer = tensorboard.SummaryWriter(
        os.path.join(self.tmp_dir, 'train'))
    summary_writer.scalar('accuracy', 0.5, 43)
    summary_writer.scalar('accuracy', 0.4, 44)
    summary_writer.scalar('accuracy', 0.8, 45)
    summary_writer.scalar('accuracy', 0.3, 46)

    # Verify that we keep checkpoints for 41 and 42 even without metrics.
    checkpointer.save(update_train_state_step(train_state, 41))
    checkpointer.save(update_train_state_step(train_state, 42))
    checkpointer.save(update_train_state_step(train_state, 43))
    self.assertSequenceEqual(checkpointer.all_steps(), [41, 42, 43])

    # Mark 41 and 43 checkpoints as pinned / to not be removed.
    ckpt_dir_41 = self.checkpoints_dir.mkdir(f'checkpoint_{41}')
    ckpt_dir_41.create_file('PINNED')
    ckpt_dir_43 = self.checkpoints_dir.mkdir(f'checkpoint_{43}')
    ckpt_dir_43.create_file('PINNED')

    # Checkpoints 41 and 43 should always be kept because they are pinned.
    checkpointer.save(update_train_state_step(train_state, 44))
    self.assertSequenceEqual(checkpointer.all_steps(), [41, 42, 43, 44])
    # Checkpoint 44 should get deleted on next save. 43 is saved inspite of
    #  it's low accuracy because it is pinned.
    checkpointer.save(update_train_state_step(train_state, 45))
    self.assertSequenceEqual(checkpointer.all_steps(), [41, 42, 43, 45])

  @mock.patch('time.time', return_value=0)
  def test_save_best_checkpointer_from_restart(self, unused_mock_time):
    """Emulate restart/preempt condition."""
    no_partitions_partitioner = self.get_partitioner(0, 1, 1)
    train_state = self.train_state

    # First, create a checkpointer that saves all checkpoints.
    checkpointer = checkpoints.Checkpointer(
        train_state,
        no_partitions_partitioner,
        self.tmp_dir,
        keep=None,
        use_gda=False)

    # Create a series of checkpoints. Create many checkpoints to stress test
    # event collection (some methods employ lossy/sampling collection).
    for i in range(100):
      checkpointer.save(update_train_state_step(train_state, i))
    self.assertSequenceEqual(checkpointer.all_steps(), list(range(100)))

    # Now create some metrics for all steps, with high metrics on specific
    # steps.
    summary_writer = tensorboard.SummaryWriter(
        os.path.join(self.tmp_dir, 'train'))
    for i in range(100):
      if i in (42, 53):
        summary_writer.scalar('accuracy', i * 0.01, i)
      else:
        summary_writer.scalar('accuracy', i * 0.001, i)

    # Replace checkpointer with SaveBest variant.
    checkpointer = checkpoints.SaveBestCheckpointer(
        train_state,
        no_partitions_partitioner,
        self.tmp_dir,
        keep=2,
        metric_name_to_monitor='train/accuracy',
        metric_mode='max',
        use_gda=False)

    # Verify that pre-existing metrics are read and the appropriate checkpoints
    # are deleted.
    checkpointer.save(update_train_state_step(train_state, 101))
    self.assertSequenceEqual(checkpointer.all_steps(), [42, 53, 101])

  def test_save_best_checkpointer_force_keep_period(self):
    no_partitions_partitioner = self.get_partitioner(0, 1, 1)
    train_state = self.train_state

    checkpointer = checkpoints.SaveBestCheckpointer(
        train_state,
        no_partitions_partitioner,
        self.tmp_dir,
        keep=2,
        metric_name_to_monitor='train/accuracy',
        metric_mode='max',
        keep_checkpoints_without_metrics=False,
        force_keep_period=3,
        use_gda=False)

    summary_writer = tensorboard.SummaryWriter(
        os.path.join(self.tmp_dir, 'train'))

    # save checkpoints 0..9 with increasing accuracy
    dict_actual_steps = {}
    for c in range(10):
      checkpointer.save(update_train_state_step(train_state, c))
      summary_writer.scalar('accuracy', c / 100, c)
      dict_actual_steps[c] = checkpointer.all_steps()

    # Check when the last step=8 is not divisible by the keep_period=3
    actual_steps_8 = dict_actual_steps[8]
    expected_steps_8 = [0, 3, 5, 6, 7, 8]
    self.assertSequenceEqual(actual_steps_8, expected_steps_8)

    # Check when the last step=9 is divisible by the keep_period=3
    actual_steps_9 = dict_actual_steps[9]
    expected_steps_9 = [0, 3, 6, 7, 8, 9]
    self.assertSequenceEqual(actual_steps_9, expected_steps_9)

  @mock.patch('time.time', return_value=0)
  def test_save_best_checkpointer_missing_metrics(self, unused_mock_time):
    """Test for `keep_checkpoints_without_metrics` behavior."""
    no_partitions_partitioner = self.get_partitioner(0, 1, 1)
    train_state = self.train_state

    # Replace checkpointer with SaveBest variant.
    checkpointer = checkpoints.SaveBestCheckpointer(
        train_state,
        no_partitions_partitioner,
        self.tmp_dir,
        keep=1,
        metric_name_to_monitor='train/accuracy',
        metric_mode='max',
        use_gda=False)

    # Pre-create metrics for only some of the steps.
    summary_writer = tensorboard.SummaryWriter(
        os.path.join(self.tmp_dir, 'train'))
    summary_writer.scalar('accuracy', 0.6, 43)
    summary_writer.scalar('accuracy', 0.5, 44)
    summary_writer.scalar('accuracy', 0.4, 45)

    # Verify that we always keep checkpoints for 41 and 42 (no metrics) and that
    # number to keep applies to other checkpoints.
    checkpointer.save(update_train_state_step(train_state, 41))
    self.assertSequenceEqual(checkpointer.all_steps(), [41])
    checkpointer.save(update_train_state_step(train_state, 42))
    self.assertSequenceEqual(checkpointer.all_steps(), [41, 42])
    checkpointer.save(update_train_state_step(train_state, 43))
    self.assertSequenceEqual(checkpointer.all_steps(), [41, 42, 43])
    checkpointer.save(update_train_state_step(train_state, 44))
    self.assertSequenceEqual(checkpointer.all_steps(), [41, 42, 43, 44])
    # Checkpoint 44 should get deleted on next save.
    checkpointer.save(update_train_state_step(train_state, 45))
    self.assertSequenceEqual(checkpointer.all_steps(), [41, 42, 43, 45])

    # When switching keep_checkpoints_without_metrics to False, we should see
    # checkpoints 41 and 42 also be deleted.
    checkpointer._keep_checkpoints_without_metrics = False
    checkpointer.save(update_train_state_step(train_state, 46))
    self.assertSequenceEqual(checkpointer.all_steps(), [43, 46])

  def test_assignment_map(self):
    self.validate_save(1, 1)
    # Change optimizer
    optimizer = optimizers.Optimizer(
        optimizers.sgd(0.1),
        state=optimizers.OptimizerState(
            step=np.int32(42),
            param_states={
                'bias': np.int32(1),
                'kernel': np.array([1, 2], np.uint8)
            }),
        target={
            'bias': np.arange(4, dtype=jnp.bfloat16).reshape((4, 1)),
            'layer1': {
                'bias': np.arange(4, dtype=jnp.bfloat16).reshape((4, 1)),
                'kernel': np.arange(32, dtype=np.float32).reshape((2, 16))
            },
            'layer2': {
                'bias': np.arange(32, dtype=np.float32).reshape((2, 16)),
                'kernel': np.arange(32, dtype=np.float32).reshape((2, 16))
            }
        })
    self.train_state = FlaxOptimTrainState(optimizer)

    actual_train_state = self.call_host_checkpointer(
        0,
        1,
        self.get_partitioner(
            0, 1, 1, mesh_axes=jax.tree_map(lambda x: None, self.train_state)),
        lambda c: c.restore(  # pylint:disable=g-long-lambda
            step=42,
            state_transformation_fns=[
                functools.partial(
                    state_utils.apply_assignment_map,
                    assignment_map=[('target/layer2/bias', 'target/kernel'),
                                    ('target/layer\\d/(.*)', 'target/\\1')])
            ]),
        np.float32,
        None)
    self.assertEqual(actual_train_state.step, 42)
    self.assertEqual(actual_train_state._optimizer.optimizer_def,
                     self.train_state._optimizer.optimizer_def)
    jax.tree_map(np.testing.assert_array_equal, actual_train_state.param_states,
                 self.train_state.param_states)
    jax.tree_map(np.testing.assert_array_equal, actual_train_state.params,
                 self.train_state.params)

  def test_assignment_map_unused(self):
    self.validate_save(1, 1)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Unused patterns in `assignment_map`: {'target/layer\\d/(.*)'}"):
      self.call_host_checkpointer(
          0,
          1,
          self.get_partitioner(0, 1, 1),
          lambda c: c.restore(  # pylint:disable=g-long-lambda
              step=42,
              state_transformation_fns=[
                  functools.partial(
                      state_utils.apply_assignment_map,
                      assignment_map=[('target/layer\\d/(.*)', 'target/\\1')])
              ]),
          np.float32,
          None)

  def test_assignment_map_noexists(self):
    self.validate_save(1, 1)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Parameter 'target/layer/bias' does not exist in restore checkpoint. "
        "Must be one of: ['state/param_states/bias', "
        "'state/param_states/kernel', 'state/step', 'target/bias', "
        "'target/kernel']"):
      self.call_host_checkpointer(
          0,
          1,
          self.get_partitioner(0, 1, 1),
          lambda c: c.restore(  # pylint:disable=g-long-lambda
              step=42,
              state_transformation_fns=[
                  functools.partial(
                      state_utils.apply_assignment_map,
                      assignment_map=[('target/(.*)', 'target/layer/\\1')])
              ]),
          np.float32,
          None)

  def test_assignment_map_partial_restore(self):
    self.validate_save(1, 1)
    # Change optimizer
    optimizer = optimizers.Optimizer(
        optimizers.sgd(0.1),
        state=optimizers.OptimizerState(
            step=np.int32(42),
            param_states={
                'bias': np.int32(1),
                'kernel': np.array([1, 2], np.uint8)
            }),
        target={
            'bias': np.arange(4, dtype=jnp.bfloat16).reshape((4, 1)),
            'layer1': {
                'bias': np.arange(4, dtype=jnp.bfloat16).reshape((4, 1)),
                'kernel': np.arange(32, dtype=np.float32).reshape((2, 16))
            },
            'layer2': {
                'bias': np.arange(32, dtype=np.float32).reshape((2, 16)),
                'kernel': np.arange(32, dtype=np.float32).reshape((2, 16))
            }
        })
    self.train_state = FlaxOptimTrainState(optimizer)

    actual_train_state = self.call_host_checkpointer(
        0,
        1,
        self.get_partitioner(
            0, 1, 1, mesh_axes=jax.tree_map(lambda x: None, self.train_state)),
        lambda c: c.restore(  # pylint:disable=g-long-lambda
            step=42,
            state_transformation_fns=[
                functools.partial(
                    state_utils.apply_assignment_map,
                    assignment_map=[
                        # Restore only the target kernels.
                        (r'target/layer(\d+)/kernel', r'target/kernel'),
                        (r'target.*bias', None),
                        (r'state.*', None)])
            ],
            fallback_state={
                # Initialize biases and optimizer state "from scratch"
                'target': {
                    'bias': np.arange(4, dtype=jnp.bfloat16).reshape((4, 1)),
                    'layer1': {
                        'bias': np.arange(4, dtype=jnp.bfloat16).reshape((4, 1)),
                    },
                    'layer2': {
                        'bias': np.arange(32, dtype=np.float32).reshape((2, 16)),
                    }
                },
                'state': {
                    'step': 1337,  # Note: original optimizer is step=42
                    'param_states': {
                        'bias': 1,
                        'kernel': np.array([1, 2], np.uint8)
                    }
                }
            }),
        np.float32,
        None)
    self.assertEqual(actual_train_state._optimizer.optimizer_def,
                     self.train_state._optimizer.optimizer_def)
    self.assertEqual(actual_train_state.step, 1337)  # note: from-scratch
    jax.tree_map(np.testing.assert_array_equal, actual_train_state.param_states,
                 self.train_state.param_states)
    jax.tree_map(np.testing.assert_array_equal, actual_train_state.params,
                 self.train_state.params)

  def verify_restore_checkpoint_from_path(
      self,
      path,
      model,
      decoder_only=False,
      partitioner_class=partitioning.PjitPartitioner):
    partitioner = partitioner_class(num_partitions=1)

    input_features = {'decoder_input_tokens': tf.zeros([2, 8])}
    if not decoder_only:
      input_features['encoder_input_tokens'] = tf.zeros([2, 8])
    train_ds = tf.data.Dataset.from_tensors(input_features)

    train_state_initializer = utils.TrainStateInitializer(
        optimizer_def=model.optimizer_def,
        init_fn=model.get_initial_variables,
        input_shapes={k: v.shape for k, v in train_ds.element_spec.items()},
        partitioner=partitioner)

    restored = list(
        train_state_initializer.from_checkpoints([
            utils.RestoreCheckpointConfig(
                mode='specific', path=path, use_gda=False)
        ]))
    self.assertLen(restored, 1)
    state, _ = restored[0]
    return state

  def test_checkpointer_in_threaded_env(self):
    """Tests use of asyncio in checkpointer works with non-main threads."""
    executor = concurrent.futures.thread.ThreadPoolExecutor(max_workers=1)
    save = executor.submit(self.validate_save, 1, 1)
    save.result()
    restore = executor.submit(self.validate_restore, 1, 1)
    restore.result()

  def test_find_checkpoint(self):
    # `model_dir` with no step
    self.assertEqual(
        checkpoints.find_checkpoint(self.fake_checkpoints),
        os.path.join(self.fake_checkpoints, f'checkpoint_{self.steps[-1]}',
                     'checkpoint'))
    # `model_dir` with step
    step = 100
    self.assertEqual(
        checkpoints.find_checkpoint(self.fake_checkpoints, step),
        os.path.join(self.fake_checkpoints, f'checkpoint_{step}', 'checkpoint'))
    # checkpoint_dir
    self.assertEqual(
        checkpoints.find_checkpoint(
            os.path.join(self.fake_checkpoints, f'checkpoint_{step}')),
        os.path.join(self.fake_checkpoints, f'checkpoint_{step}', 'checkpoint'))
    # checkpoint_dir with step
    with self.assertRaises(ValueError):
      _ = checkpoints.find_checkpoint(
          os.path.join(self.fake_checkpoints, f'checkpoint_{step}'), 1000),
    # checkpoint_file
    path = os.path.join(self.fake_checkpoints, f'checkpoint_{step}',
                        'checkpoint')
    self.assertEqual(checkpoints.find_checkpoint(path), path)
    # checkpoint_file with step
    self.assertEqual(checkpoints.find_checkpoint(path, 1000), path)
    # Error with step
    with self.assertRaises(ValueError):
      checkpoints.find_checkpoint(self.fake_checkpoints, 1000)
    # Error
    with self.assertRaises(ValueError):
      checkpoints.find_checkpoint(
          os.path.join(self.fake_checkpoints, 'checkpoint'))

  def test_restore_tf_as_t5x(self):
    checkpoint_path = os.path.join(TESTDATA, 'mtf_tiny_t5')
    partitioner = self.get_partitioner(0, 1, 1)
    with self.assertRaisesRegex(
        ValueError,
        'Attempting to restore a TensorFlow checkpoint as a native T5X '
        'checkpoint. Use `restore_from_tf_checkpoint` instead. Path: .*'):
      self.call_host_checkpointer(0, 1, partitioner,
                                  lambda c: c.restore(path=checkpoint_path),
                                  np.float32, None)

  def test_restore_from_invalid_path(self):
    with self.assertRaisesRegex(ValueError,
                                r'Path is not a valid T5X checkpoint: .*'):
      self.verify_restore_checkpoint_from_path(TESTDATA,
                                               test_utils.get_t5_test_model())

    with self.assertRaisesRegex(ValueError,
                                r'Path is not a valid T5X checkpoint: .*'):
      self.verify_restore_checkpoint_from_path(
          os.path.join(TESTDATA, 'checkpoint'), test_utils.get_t5_test_model())

  def test_save_lazy_optimizer(self):
    # Call save one to get the parameters onto disk
    self.validate_save(1, 1)
    # Load the parameters in a lazy way
    partitioner = self.get_partitioner(0, 1, 1, params_on_devices=False)
    step = 42
    train_state = self.call_host_checkpointer(
        0,
        1,
        partitioner,
        lambda c: c.restore(  # pylint: disable=g-long-lambda
            step=step, lazy_parameters=True),
        np.float32,
        None)
    # Increment the step so we can save it
    new_step = train_state.step.get() + 1
    state_dict = train_state.state_dict()
    state_dict['state']['step'] = new_step
    train_state = train_state.restore_state(state_dict)

    # Save the train state that is made of lazy parameters.
    self.call_host_checkpointer(
        0, 1, partitioner,
        lambda c: c.save(train_state=train_state, concurrent_gb=2), np.float32,
        None)

    # Load what we just saved to inspect values
    loaded_train_state = checkpoints.load_t5x_checkpoint(
        self.tmp_dir, step=new_step)
    # Make sure the parameters are the same.
    train_state = jax.tree_map(
        lambda x: x.get()  # pylint: disable=g-long-lambda
        if isinstance(x, LazyArray) else x,
        train_state)
    jax.tree_map(np.testing.assert_allclose, train_state.state_dict(),
                 loaded_train_state)

  def test_update_ts_from_gfile_to_gcs(self):
    ckpt_contents = {
        'version': 3,
        'optimizer': {
            'target': {
                'unsharded_param': np.ones((5, 5), dtype=np.int32),
                'sharded_param': {
                    'driver': 'zarr',
                    'dtype': 'float32',
                    'kvstore': {
                        'driver': 'file',
                        'path': 'target.sharded_param'
                    },
                    'metadata': {
                        'chunks': [768, 768],
                        'compressor': {
                            'id': 'gzip',
                            'level': 1
                        },
                        'shape': [768, 768]
                    }
                }
            }
        }
    }

    expected = {
        'version': 3,
        'optimizer': {
            'target': {
                # np.ndarray should not change
                'unsharded_param': np.ones((5, 5), dtype=np.int32),
                'sharded_param': {
                    'driver': 'zarr',
                    'dtype': 'float32',
                    'kvstore': {
                        'bucket': 't5x-dummy-bucket',
                        'driver': 'gcs',
                        'path': 'target.sharded_param'
                    },
                    'metadata': {
                        'chunks': [768, 768],
                        'compressor': {
                            'id': 'gzip',
                            'level': 1
                        },
                        'shape': [768, 768]
                    }
                }
            }
        }
    }
    actual = checkpoints._maybe_update_ts_from_file_to_gcs(ckpt_contents)
    jax.tree_map(np.testing.assert_array_equal, actual, expected)

  def test_update_ts_from_gcs_to_file(self):
    ckpt_contents = {
        'version': 3,
        'optimizer': {
            'target': {
                # np.ndarray should not change
                'unsharded_param': np.ones((5, 5), dtype=np.int32),
                'sharded_param': {
                    'driver': 'zarr',
                    'dtype': 'float32',
                    'kvstore': {
                        'bucket': 't5x-dummy-bucket',
                        'driver': 'gcs',
                        'path': 'target.sharded_param'
                    },
                    'metadata': {
                        'chunks': [768, 768],
                        'compressor': {
                            'id': 'gzip',
                            'level': 1
                        },
                        'shape': [768, 768]
                    },
                }
            }
        }
    }

    driver = 'file'
    expected = {
        'version': 3,
        'optimizer': {
            'target': {
                'unsharded_param': np.ones((5, 5), dtype=np.int32),
                'sharded_param': {
                    'driver': 'zarr',
                    'dtype': 'float32',
                    'kvstore': {
                        'driver': driver,
                        'path': 'target.sharded_param'
                    },
                    'metadata': {
                        'chunks': [768, 768],
                        'compressor': {
                            'id': 'gzip',
                            'level': 1
                        },
                        'shape': [768, 768]
                    }
                }
            }
        }
    }

    actual = checkpoints._maybe_update_ts_from_gcs_to_file(ckpt_contents)
    jax.tree_map(np.testing.assert_array_equal, actual, expected)

  def assert_update_ts_path_from_relative_to_absolute(self, ts_spec_dict,
                                                      expected, ckpt_dir):
    """Tests that `ts_spec_dict` gets updated with `ckpt_dir` to `expected`."""

    # Test with normalization (corresponds to tensorstore>=0.1.14)
    normalized_ts_spec_dict = ts.Spec(ts_spec_dict).to_json()
    checkpoints._update_ts_path_from_relative_to_absolute(
        ckpt_dir, normalized_ts_spec_dict)
    normalized_ts_spec_dict = ts.Spec(normalized_ts_spec_dict).to_json()
    normalized_expected = ts.Spec(expected).to_json()
    jax.tree_map(np.testing.assert_array_equal, normalized_ts_spec_dict,
                 normalized_expected)

    # Test without normalization (corresponds to tensorstore<0.1.14)
    checkpoints._update_ts_path_from_relative_to_absolute(
        ckpt_dir, ts_spec_dict)
    jax.tree_map(np.testing.assert_array_equal, ts_spec_dict, expected)

  def test_update_ts_path_from_relative_to_absolute_gfile(self):
    ts_spec_dict = {
        'driver': 'zarr',
        'dtype': 'float32',
        'kvstore': {
            'driver': 'file',
            'path': 'target.encoder.layers_0.attention.query.kernel'
        },
        'metadata': {
            'chunks': [768, 768],
            'compressor': {
                'id': 'gzip',
                'level': 1
            },
            'shape': [768, 768]
        }
    }

    expected = {
        'driver': 'zarr',
        'dtype': 'float32',
        'kvstore': {
            'driver': 'file',
            # Path becomes absolute.
            'path': '/dir1/dir2/target.encoder.layers_0.attention.query.kernel'
        },
        'metadata': {
            'chunks': [768, 768],
            'compressor': {
                'id': 'gzip',
                'level': 1
            },
            'shape': [768, 768]
        }
    }
    ckpt_dir = '/dir1/dir2'

    self.assert_update_ts_path_from_relative_to_absolute(
        ts_spec_dict, expected, ckpt_dir)

  def test_update_ts_path_from_relative_to_absolute_gcs(self):
    ts_spec_dict = {
        'driver': 'zarr',
        'dtype': 'float32',
        'kvstore': {
            'bucket': 't5x-dummy-bucket',
            'driver': 'gcs'
        },
        'metadata': {
            'chunks': [768, 768],
            'compressor': {
                'id': 'gzip',
                'level': 1
            },
            'shape': [768, 768]
        },
        'path': 'target.encoder.layers_0.attention.query.kernel',
        'transform': {
            'input_exclusive_max': [[768], [768]],
            'input_inclusive_min': [0, 0]
        }
    }

    expected = {
        'driver': 'zarr',
        'dtype': 'float32',
        'kvstore': {
            'bucket': 'test-bucket',  # bucket should be changed.
            'driver': 'gcs'
        },
        'metadata': {
            'chunks': [768, 768],
            'compressor': {
                'id': 'gzip',
                'level': 1
            },
            'shape': [768, 768]
        },
        # Path becomes absolute without the "gs://bucket" portion stripped.
        'path': 'dir1/dir2/target.encoder.layers_0.attention.query.kernel',
        'transform': {
            'input_exclusive_max': [[768], [768]],
            'input_inclusive_min': [0, 0]
        }
    }

    ckpt_dir = 'gs://test-bucket/dir1/dir2'

    self.assert_update_ts_path_from_relative_to_absolute(
        ts_spec_dict, expected, ckpt_dir)

  def test_restore_tf_checkpoint(self):
    self.verify_restore_checkpoint_from_path(
        os.path.join(TESTDATA, 'mtf_tiny_t5/model.ckpt-0'),
        test_utils.get_t5_test_model(
            emb_dim=32, head_dim=64, num_heads=2, mlp_dim=64))

  def test_restore_tf_checkpoint_wrong_config(self):
    with self.assertRaisesRegex(ValueError, r'Variable .* has shape .* != .*'):
      self.verify_restore_checkpoint_from_path(
          os.path.join(TESTDATA, 'mtf_tiny_t5/model.ckpt-0'),
          test_utils.get_t5_test_model())

  def test_convert_tf_checkpoint(self):
    checkpoint_path = os.path.join(TESTDATA, 'mtf_tiny_t5/model.ckpt-0')

    # Minimal setup to create an optimizer with the matching config.
    model = test_utils.get_t5_test_model(
        emb_dim=32, head_dim=64, num_heads=2, mlp_dim=64)

    partitioner = partitioning.PjitPartitioner(num_partitions=1)

    def initialize_params_fn(rng):
      initial_variables = model.get_initial_variables(
          rng=rng,
          input_shapes={
              'encoder_input_tokens': (2, 512),
              'decoder_input_tokens': (2, 114),
          })
      return FlaxOptimTrainState.create(model.optimizer_def, initial_variables)

    train_state = jax.eval_shape(initialize_params_fn, jax.random.PRNGKey(0))
    checkpointer = checkpoints.Checkpointer(
        train_state, partitioner, self.tmp_dir, use_gda=False)
    _ = checkpointer.convert_from_tf_checkpoint(checkpoint_path)

  def test_load_matched(self):
    checkpoint = os.path.join(TESTDATA, 'test_t5_tiny.checkpoint_0')
    train_state = self.verify_restore_checkpoint_from_path(
        checkpoint, test_utils.get_t5_test_model())
    state_dict = train_state._optimizer.state_dict()
    ckpt = checkpoints.load_t5x_checkpoint(checkpoint)
    jax.tree_map(np.testing.assert_array_equal, state_dict, ckpt)



if __name__ == '__main__':
  absltest.main()
