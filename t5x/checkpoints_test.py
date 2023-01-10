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

"""Tests for checkpointing."""

import functools
import math
import os
from typing import Any, Mapping
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from flax.metrics import tensorboard
import jax
from jax.experimental.global_device_array import GlobalDeviceArray
from jax.experimental.maps import Mesh
from jax.experimental.pjit import pjit
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
from t5x import checkpoints
from t5x import optimizers
from t5x import partitioning
from t5x import state_utils
from t5x import test_utils
from t5x import train_state as train_state_lib
from t5x import utils
from t5x.partitioning import PartitionSpec as P
import tensorflow as tf

jax.config.update('jax_parallel_functions_output_gda', True)


TESTDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testdata')
assert_tree_equal = orbax.checkpoint.test_utils.assert_tree_equal


def make_train_state_base(
    *,
    step: int,
    params: Mapping[str, Any],
    param_states: Mapping[str, Any],
    flax_optimizer_def: optimizers.OptimizerDefType = optimizers.sgd(0.1)
) -> train_state_lib.TrainState:
  """Helper to construct a train state for testing."""
  optimizer = optimizers.Optimizer(
      flax_optimizer_def,
      state=optimizers.OptimizerState(step=step, param_states=param_states),
      target=params)
  return train_state_lib.FlaxOptimTrainState(optimizer)


def make_train_state_non_gda(global_mesh,
                             global_input_shape,
                             mesh_axes,
                             step=42,
                             dtype=np.float32,
                             shard=False):
  bias = np.ones(global_input_shape, dtype=dtype)
  kernel = np.arange(
      math.prod(global_input_shape), dtype=dtype).reshape(global_input_shape)

  if shard:
    jax.config.update('jax_parallel_functions_output_gda', False)
    partition_array = pjit(
        lambda x: x, in_axis_resources=(None,), out_axis_resources=mesh_axes)
    with global_mesh:
      kernel = partition_array(kernel)
    jax.config.update('jax_parallel_functions_output_gda', True)

  train_state = make_train_state_base(
      step=np.int32(step),
      params={
          'bias': bias * 2,
          'kernel': kernel * 2
      },
      param_states={  # only cast targets (above)
          'bias': bias.astype(np.float32),
          'kernel': kernel.astype(np.float32)
      })
  return train_state


def create_sharded_array(arr, global_shape, global_mesh, mesh_axes):

  def cb(index):
    return arr[index]

  if np.isscalar(arr):
    return arr
  if jax.config.jax_array:
    return jax.make_array_from_callback(
        global_shape, jax.sharding.NamedSharding(global_mesh, mesh_axes), cb)
  else:
    return GlobalDeviceArray.from_callback(global_shape, global_mesh, mesh_axes,
                                           cb)


def make_train_state(global_mesh,
                     global_input_shape,
                     mesh_axes,
                     step=42,
                     dtype=np.float32):
  train_state = make_train_state_non_gda(
      global_mesh, global_input_shape, mesh_axes, step=step, dtype=dtype)

  return jax.tree_map(
      functools.partial(
          create_sharded_array,
          global_shape=global_input_shape,
          global_mesh=global_mesh,
          mesh_axes=mesh_axes),
      train_state,
      is_leaf=lambda x: isinstance(x, np.ndarray))


def all_gda_shards(gda):
  global_array = np.zeros(gda.shape)
  for shard in gda.global_shards:
    global_array[shard.index] = shard.data
  return global_array


class FakePartitioner(partitioning.BasePartitioner):

  def __init__(self, mesh, mesh_axes):
    super().__init__(num_partitions=1)
    self._global_mesh = mesh
    self._mesh_axes = mesh_axes
    self._local_chunker = partitioning.LocalChunker(self.mesh)

  def get_data_layout(self):
    return partitioning.DataLayout(
        batch_size=None,
        shard_id=1,
        num_shards=1,
        is_first_host_in_replica_set=True)

  @property
  def mesh(self):
    return self._global_mesh

  @property
  def params_on_devices(self):
    return False

  def move_params_to_devices(self, train_state, train_state_axes):
    return train_state

  def get_mesh_axes(self, train_state):
    mesh_axes = jax.tree_map(lambda _: self._mesh_axes, train_state)
    return mesh_axes.replace_step(None)

  def _local_chunker(self):
    return self._local_chunker

  def partition(self,
                fn,
                in_axis_resources,
                out_axis_resources,
                static_argnums=(),
                donate_argnums=()):
    pjitted = pjit(
        fn,
        in_axis_resources=in_axis_resources,
        out_axis_resources=out_axis_resources,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums)
    return partitioning.PjittedFnWithContext(pjitted, self.mesh)

  def compile(self, partitioned_fn, *args):
    return None


class CheckpointsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.tmp_dir = self.create_tempdir().full_path

  def make_lazy_array(self, arr):

    async def get_fn():
      return arr

    return checkpoints.LazyAwaitableArray.from_array(arr, get_fn=get_fn)

  def pytree_as_gda(self, pytree, global_mesh, global_input_shape, mesh_axes):

    return jax.tree_map(
        functools.partial(
            create_sharded_array,
            global_shape=global_input_shape,
            global_mesh=global_mesh,
            mesh_axes=mesh_axes),
        pytree,
        is_leaf=lambda x: isinstance(x, np.ndarray))

  def save(self, checkpointer, train_state):
    checkpointer.save(train_state)

  def restore(self,
              checkpointer,
              step=None,
              path=None,
              lazy_parameters=False,
              state_transformation_fns=(),
              fallback_state=None):
    return checkpointer.restore(
        step=step,
        path=path,
        lazy_parameters=lazy_parameters,
        state_transformation_fns=state_transformation_fns,
        fallback_state=fallback_state)

  def restore_tf(self, checkpointer, path):
    return checkpointer.restore_from_tf_checkpoint(path)

  def validate_save_restore(self,
                            train_state,
                            global_mesh,
                            mesh_axes,
                            step=42,
                            save_dtype=np.float32,
                            restore_dtype=np.float32,
                            lazy_parameters=False,
                            dataset_iterator=None):
    step = np.int32(step)

    checkpointer = checkpoints.Checkpointer(
        train_state,
        FakePartitioner(global_mesh, mesh_axes),
        self.tmp_dir,
        dataset_iterator=dataset_iterator,
        save_dtype=save_dtype,
        restore_dtype=restore_dtype)
    self.save(checkpointer, train_state)
    if dataset_iterator is not None:
      for _ in range(10):
        next(dataset_iterator)
      self.assertEqual(10, next(dataset_iterator))

    restored_train_state = self.restore(
        checkpointer, step=step, lazy_parameters=lazy_parameters)

    def _assert_is_lazy(arr):
      self.assertIsInstance(arr, orbax.checkpoint.lazy_utils.LazyValue)

    if lazy_parameters:
      jax.tree_map(_assert_is_lazy, restored_train_state)
    assert_tree_equal(self, train_state, restored_train_state)
    if dataset_iterator is not None:
      self.assertEqual(0, next(dataset_iterator))

  def test_basic(self):
    global_mesh = test_utils.create_global_mesh((4, 2), ('x', 'y'))
    mesh_axes = P('x', 'y')
    global_input_shape = (8, 2)

    train_state = make_train_state(global_mesh, global_input_shape, mesh_axes)
    self.validate_save_restore(train_state, global_mesh, mesh_axes)

  def test_with_dataset(self):
    global_mesh = test_utils.create_global_mesh((4, 2), ('x', 'y'))
    mesh_axes = P('x', 'y')
    global_input_shape = (8, 2)

    train_state = make_train_state(global_mesh, global_input_shape, mesh_axes)
    dataset_iterator = iter(tf.data.Dataset.range(64))
    self.validate_save_restore(
        train_state, global_mesh, mesh_axes, dataset_iterator=dataset_iterator)

  def test_lazy(self):
    global_mesh = test_utils.create_global_mesh((4, 2), ('x', 'y'))
    mesh_axes = P('x', 'y')
    global_input_shape = (8, 2)

    train_state = make_train_state(global_mesh, global_input_shape, mesh_axes)
    train_state = jax.tree_map(self.make_lazy_array, train_state)
    self.validate_save_restore(
        train_state, global_mesh, mesh_axes, lazy_parameters=True)

  def test_non_partitioned(self):
    global_mesh = Mesh(np.asarray(jax.devices()), ('x',))
    mesh_axes = P(None,)
    global_input_shape = (8,)

    train_state = make_train_state(global_mesh, global_input_shape, mesh_axes)
    self.validate_save_restore(train_state, global_mesh, mesh_axes)

  @parameterized.named_parameters(
      (
          'bfloat16',
          jnp.bfloat16,
      ),
      (
          'float32',
          np.float32,
      ),
      (
          'int32',
          np.int32,
      ),
  )
  def test_params_restore_as_type(self, dtype):
    global_mesh = test_utils.create_global_mesh((4, 2), ('x', 'y'))
    mesh_axes = P('x', 'y')
    global_input_shape = (8, 2)

    train_state = make_train_state(
        global_mesh, global_input_shape, mesh_axes, dtype=dtype)
    self.validate_save_restore(
        train_state,
        global_mesh,
        mesh_axes,
        save_dtype=np.float32,
        restore_dtype=dtype)

  def test_assignment_map_and_fallback(self, save_fn=None, restore_fn=None):
    global_mesh = test_utils.create_global_mesh((4, 2), ('x', 'y'))
    mesh_axes = P('x', 'y')
    global_input_shape = (8, 2)

    original_bias = np.ones(global_input_shape) * 10
    original_kernel = np.arange(16).reshape(global_input_shape) * 10
    fallback_bias = np.arange(16).reshape(global_input_shape)
    fallback_kernel = np.ones(16).reshape(global_input_shape) * 2

    original_train_state = make_train_state_base(
        step=np.int32(42),
        params={
            'bias': original_bias,
            'kernel': original_kernel,
        },
        param_states={
            'bias': original_bias,
            'kernel': original_kernel,
        })
    original_train_state = self.pytree_as_gda(original_train_state, global_mesh,
                                              global_input_shape, mesh_axes)
    checkpointer = checkpoints.Checkpointer(
        original_train_state, FakePartitioner(global_mesh, mesh_axes),
        self.tmp_dir)
    if save_fn is None:
      save_fn = self.save
    save_fn(checkpointer, original_train_state)

    # train_state should be restored as the following. Compare with
    # fallback_state below and original_train_state above.
    optimizer = optimizers.Optimizer(
        optimizers.sgd(0.1),
        state=optimizers.OptimizerState(
            step=np.int32(42),  # original step number
            param_states={
                'bias': fallback_bias,
                'kernel': fallback_kernel,
            }),
        target={
            'bias': fallback_bias,
            'layer2': {
                'bias': fallback_bias,
                'kernel': original_kernel,
            },
            'layer3': {
                'bias': fallback_bias,
                'kernel': original_kernel,
            }
        })
    train_state = train_state_lib.FlaxOptimTrainState(optimizer)
    train_state = self.pytree_as_gda(train_state, global_mesh,
                                     global_input_shape, mesh_axes)

    checkpointer = checkpoints.Checkpointer(
        train_state,
        FakePartitioner(global_mesh, mesh_axes),
        self.tmp_dir,
        restore_dtype=np.int32)

    fallback_state = {
        'target': {
            'bias': fallback_bias,
            'layer2': {
                'bias': fallback_bias,
            },
            'layer3': {
                'bias': fallback_bias,
            },
        },
        'state': {
            'step': np.int32(1337),  # Note: original optimizer is step=42
            'param_states': {
                'bias': fallback_bias,
                'kernel': fallback_kernel,
            }
        }
    }
    fallback_state = self.pytree_as_gda(fallback_state, global_mesh,
                                        global_input_shape, mesh_axes)
    if restore_fn is None:
      restore_fn = self.restore
    restored_train_state = restore_fn(
        checkpointer,
        step=42,
        state_transformation_fns=[
            functools.partial(
                state_utils.apply_assignment_map,
                assignment_map=[
                    # Restore only the target kernels.
                    (r'target/layer(\d+)/kernel', r'target/kernel'),
                    (r'target.*bias', None),
                    (r'state.*', None)
                ])
        ],
        fallback_state=fallback_state)

    self.assertEqual(restored_train_state._optimizer.optimizer_def,
                     train_state._optimizer.optimizer_def)
    self.assertEqual(restored_train_state.step, 1337)  # not 42, as saved.
    assert_tree_equal(self, restored_train_state.param_states,
                      train_state.param_states)
    assert_tree_equal(self, restored_train_state.params, train_state.params)

  @test_utils.with_mesh([('x', 4), ('y', 2)])
  def test_restore_from_path(self):
    global_mesh = test_utils.create_global_mesh((4, 2), ('x', 'y'))
    mesh_axes = P('x', 'y')
    global_input_shape = (8, 2)
    train_state = make_train_state(global_mesh, global_input_shape, mesh_axes)

    step = 42
    checkpoint_dir = self.create_tempdir(name='existing_checkpoint').full_path
    dataset_iterator = iter(tf.data.Dataset.range(64))
    checkpointer = checkpoints.Checkpointer(
        train_state,
        FakePartitioner(global_mesh, mesh_axes),
        checkpoint_dir,
        dataset_iterator=dataset_iterator)
    self.save(checkpointer, train_state)

    # Advance iterator state to compare in-memory with restored state.
    for _ in range(10):
      next(dataset_iterator)
    self.assertEqual(10, next(dataset_iterator))

    checkpointer = checkpoints.Checkpointer(
        train_state,
        FakePartitioner(global_mesh, mesh_axes),
        self.tmp_dir,
        dataset_iterator=dataset_iterator)
    restored_train_state = self.restore(
        checkpointer, path=checkpoints.get_checkpoint_dir(checkpoint_dir, step))
    assert_tree_equal(self, train_state, restored_train_state)
    self.assertEqual(0, next(dataset_iterator))

  def test_restore_tf_checkpoint(self):
    partitioner = partitioning.PjitPartitioner(num_partitions=1)
    input_features = {
        'decoder_input_tokens': tf.ones([2, 8]),
        'encoder_input_tokens': tf.ones([2, 8])
    }
    train_ds = tf.data.Dataset.from_tensors(input_features)
    model = test_utils.get_t5_test_model(
        emb_dim=32, head_dim=64, num_heads=2, mlp_dim=64)
    train_state_initializer = utils.TrainStateInitializer(
        optimizer_def=model.optimizer_def,
        init_fn=model.get_initial_variables,
        input_shapes={k: v.shape for k, v in train_ds.element_spec.items()},
        partitioner=partitioner)

    checkpointer = checkpoints.Checkpointer(
        train_state_initializer.global_train_state_shape,
        partitioner,
        self.tmp_dir,
        restore_dtype=np.int32)

    restored_train_state = self.restore_tf(
        checkpointer, os.path.join(TESTDATA, 'mtf_tiny_t5'))

    def check_type(arr):
      self.assertIsInstance(arr, (np.int32, jax.Array, GlobalDeviceArray))

    jax.tree_util.tree_map(check_type, restored_train_state)
    self.assertEqual(restored_train_state.step, 0)


class OrbaxCheckpointsTest(CheckpointsTest):

  def checkpoint_manager(self,
                         checkpointer,
                         save_dtype=None,
                         restore_dtype=None):
    return checkpoints.CheckpointManager(
        checkpointer.checkpoints_dir,
        checkpointer._train_state,
        checkpointer._partitioner,
        dataset_iterator=checkpointer._original_dataset_iterator,
        save_dtype=save_dtype,
        restore_dtype=restore_dtype)

  def save(self, checkpointer, train_state):
    self.checkpoint_manager(checkpointer).save(train_state)

  def restore(self,
              checkpointer,
              step=None,
              path=None,
              lazy_parameters=False,
              state_transformation_fns=(),
              fallback_state=None,
              dtype=None):
    return self.checkpoint_manager(
        checkpointer, restore_dtype=dtype).restore(
            step=step,
            path=path,
            lazy_parameters=lazy_parameters,
            state_transformation_fns=state_transformation_fns,
            fallback_state=fallback_state)

  def restore_tf(self, checkpointer, path):
    return self.checkpoint_manager(checkpointer).restore_from_tf_checkpoint(
        path)

  def test_backwards_compatibility(self):
    global_mesh = test_utils.create_global_mesh((2, 2), ('x', 'y'))
    mesh_axes = P('x', 'y')
    global_input_shape = (2, 2)
    step = np.int32(12)
    train_state = make_train_state(
        global_mesh, global_input_shape, mesh_axes, step=step)

    checkpointer = checkpoints.Checkpointer(
        train_state, FakePartitioner(global_mesh, mesh_axes), self.tmp_dir)
    checkpointer.save(train_state)

    manager = checkpoints.CheckpointManager(
        self.tmp_dir, train_state, FakePartitioner(global_mesh, mesh_axes))
    restored_train_state = manager.restore(step=step)

    assert_tree_equal(self, train_state, restored_train_state)

  def test_rollback_compatibility(self):
    # Ensures that we can still restore checkpoints written with Orbax if we
    # roll back to non-Orbax code.
    global_mesh = test_utils.create_global_mesh((2, 2), ('x', 'y'))
    mesh_axes = P('x', 'y')
    global_input_shape = (2, 2)
    step = np.int32(256)
    train_state = make_train_state(
        global_mesh, global_input_shape, mesh_axes, step=step)

    manager = checkpoints.CheckpointManager(
        self.tmp_dir, train_state, FakePartitioner(global_mesh, mesh_axes))
    manager.save(train_state)

    checkpointer = checkpoints.Checkpointer(
        jax.eval_shape(lambda x: x, train_state),
        FakePartitioner(global_mesh, mesh_axes), self.tmp_dir)
    restored_train_state = checkpointer.restore(step=step)

    assert_tree_equal(self, train_state, restored_train_state)

  def test_assignment_map_and_fallback_old_format(self):

    orbax_restore = self.restore

    def restore_fn(checkpointer, *args, **kwargs):
      return orbax_restore(checkpointer, *args, **kwargs)

    super().test_assignment_map_and_fallback(
        save_fn=super().save, restore_fn=restore_fn)

  def test_with_dataset_old_format(self):
    global_mesh = test_utils.create_global_mesh((4, 2), ('x', 'y'))
    mesh_axes = P('x', 'y')
    global_input_shape = (8, 2)

    train_state = make_train_state(global_mesh, global_input_shape, mesh_axes)
    dataset_iterator = iter(tf.data.Dataset.range(64))

    step = np.int32(42)

    checkpointer = checkpoints.Checkpointer(
        train_state,
        FakePartitioner(global_mesh, mesh_axes),
        self.tmp_dir,
        dataset_iterator=dataset_iterator)
    checkpointer.save(train_state)
    for _ in range(10):
      next(dataset_iterator)
    self.assertEqual(10, next(dataset_iterator))

    restored_train_state = self.restore(checkpointer, step=step)

    assert_tree_equal(self, restored_train_state, train_state)
    self.assertEqual(0, next(dataset_iterator))

  def test_steps(self):
    global_mesh = test_utils.create_global_mesh((4,), ('x',))
    mesh_axes = P('x',)
    global_input_shape = (8,)
    train_state = make_train_state(
        global_mesh, global_input_shape, mesh_axes, step=0)

    manager = checkpoints.CheckpointManager(
        self.tmp_dir, train_state, FakePartitioner(global_mesh, mesh_axes))
    num_steps = 5
    for step in range(num_steps):
      train_state = make_train_state(
          global_mesh, global_input_shape, mesh_axes, step=step)
      manager.save(train_state)

    self.assertSequenceEqual(range(num_steps), manager.all_steps())
    self.assertEqual(num_steps - 1, manager.latest_step())

  def test_metrics(self):
    global_mesh = test_utils.create_global_mesh((4, 2), ('x', 'y'))
    mesh_axes = P('x', 'y')
    global_input_shape = (8, 2)
    train_state = make_train_state(global_mesh, global_input_shape, mesh_axes)
    manager = checkpoints.BestCheckpointManager(
        self.tmp_dir,
        train_state,
        FakePartitioner(global_mesh, mesh_axes),
        keep=2,
        metric_name_to_monitor='train/accuracy',
        metric_mode='max',
        keep_checkpoints_without_metrics=True)
    self.assertTrue(manager._track_best)

    self.assertTrue(manager.save(train_state.replace_step(41)))
    self.assertSequenceEqual(manager.all_steps(), [41])
    self.assertTrue(manager.save(train_state.replace_step(42)))
    self.assertSequenceEqual(manager.all_steps(), [41, 42])
    self.assertTrue(manager.save(train_state.replace_step(43)))
    self.assertSequenceEqual(manager.all_steps(), [41, 42, 43])

    summary_writer = tensorboard.SummaryWriter(
        os.path.join(self.tmp_dir, 'train'))
    summary_writer.scalar('accuracy', 0.5, 41)
    summary_writer.scalar('accuracy', 0.6, 42)
    summary_writer.scalar('accuracy', 0.4, 43)
    summary_writer.scalar('accuracy', 0.8, 44)

    self.assertTrue(manager.save(train_state.replace_step(44)))
    # Only keep 2 best checkpoints, with highest accuracy.
    self.assertSequenceEqual(manager.all_steps(), [42, 44])

    # Change mode to `min` and check that the checkpoints with highest accuracy
    # are removed.
    manager._options.best_mode = 'min'

    summary_writer.scalar('accuracy', 0.1, 45)
    summary_writer.scalar('accuracy', 0.9, 48)
    self.assertTrue(manager.save(train_state.replace_step(45)))
    self.assertTrue(manager.save(train_state.replace_step(48)))
    self.assertSequenceEqual(manager.all_steps(), [42, 45])

  def test_metrics_existing_steps(self):
    global_mesh = test_utils.create_global_mesh((4, 2), ('x', 'y'))
    mesh_axes = P('x', 'y')
    global_input_shape = (8, 2)
    train_state = make_train_state(global_mesh, global_input_shape, mesh_axes)
    summary_writer = tensorboard.SummaryWriter(
        os.path.join(self.tmp_dir, 'train'))

    manager = checkpoints.CheckpointManager(
        self.tmp_dir,
        train_state,
        FakePartitioner(global_mesh, mesh_axes),
        keep=2,
        period=2)
    self.assertTrue(manager.save(train_state.replace_step(0)))
    self.assertFalse(manager.save(train_state.replace_step(1)))
    self.assertTrue(manager.save(train_state.replace_step(2)))
    self.assertFalse(manager.save(train_state.replace_step(3)))

    self.assertSameElements([0, 2], manager.all_steps())
    summary_writer.scalar('accuracy', 0.1, 0)
    summary_writer.scalar('accuracy', 0.2, 2)

    manager = checkpoints.BestCheckpointManager(
        self.tmp_dir,
        train_state,
        FakePartitioner(global_mesh, mesh_axes),
        keep=2,
        metric_name_to_monitor='train/accuracy',
        metric_mode='max',
        keep_checkpoints_without_metrics=True)
    self.assertTrue(manager._track_best)

    self.assertTrue(manager.save(train_state.replace_step(4)))
    self.assertSequenceEqual(manager.all_steps(), [0, 2, 4])
    self.assertTrue(manager.save(train_state.replace_step(5)))
    self.assertSequenceEqual(manager.all_steps(), [0, 2, 4, 5])
    self.assertTrue(manager.save(train_state.replace_step(6)))
    self.assertSequenceEqual(manager.all_steps(), [0, 2, 4, 5, 6])

    summary_writer.scalar('accuracy', 0.5, 4)
    summary_writer.scalar('accuracy', 0.6, 5)
    summary_writer.scalar('accuracy', 0.4, 6)
    summary_writer.scalar('accuracy', 0.8, 7)

    self.assertTrue(manager.save(train_state.replace_step(7)))
    # Only keep 2 best checkpoints, with highest accuracy.
    self.assertSequenceEqual(manager.all_steps(), [5, 7])

    # Change mode to `min` and check that the checkpoints with highest accuracy
    # are removed.
    manager._options.best_mode = 'min'

    summary_writer.scalar('accuracy', 0.1, 8)
    summary_writer.scalar('accuracy', 0.9, 9)
    self.assertTrue(manager.save(train_state.replace_step(8)))
    self.assertTrue(manager.save(train_state.replace_step(9)))
    self.assertSequenceEqual(manager.all_steps(), [5, 8])

  def test_delete_without_metrics(self):
    global_mesh = test_utils.create_global_mesh((4, 2), ('x', 'y'))
    mesh_axes = P('x', 'y')
    global_input_shape = (8, 2)
    train_state = make_train_state(global_mesh, global_input_shape, mesh_axes)
    manager = checkpoints.BestCheckpointManager(
        self.tmp_dir,
        train_state,
        FakePartitioner(global_mesh, mesh_axes),
        keep=2,
        metric_name_to_monitor='train/accuracy',
        metric_mode='max',
        keep_checkpoints_without_metrics=False)

    manager.save(train_state.replace_step(41))
    self.assertSequenceEqual(manager.all_steps(), [41])
    manager.save(train_state.replace_step(42))
    self.assertSequenceEqual(manager.all_steps(), [41, 42])
    manager.save(train_state.replace_step(43))
    self.assertSequenceEqual(manager.all_steps(), [42, 43])

  def test_end_to_end(self):
    global_mesh = test_utils.create_global_mesh((4, 2), ('x', 'y'))
    mesh_axes = P('x', 'y')
    global_input_shape = (8, 2)
    partitioner = FakePartitioner(global_mesh, mesh_axes)
    init_step = 10
    train_state = make_train_state(
        global_mesh, global_input_shape, mesh_axes, step=init_step)
    train_state_shape = jax.eval_shape(lambda: train_state)
    checkpoint_dir = self.create_tempdir(name='my_checkpoints').full_path
    # Save a checkpoint for preliminary restoration.
    checkpoints.CheckpointManager(checkpoint_dir, train_state_shape,
                                  partitioner).save(train_state)

    keep = 3
    period = 2
    save_cfg = utils.SaveCheckpointConfig(
        keep=keep, period=period, save_dataset=False)
    restore_cfg = utils.RestoreCheckpointConfig(
        path='',  # Path not used in this test.
        restore_dataset=False)

    manager = utils.create_checkpoint_manager(
        save_cfg=save_cfg,
        restore_cfg=restore_cfg,
        train_state_shape=train_state_shape,
        partitioner=partitioner,
        ds_iter=mock.Mock(),
        model_dir=checkpoint_dir)

    restored = manager.restore(step=init_step)
    assert_tree_equal(self, train_state, restored)

    self.assertEqual(manager._options.save_interval_steps, 2)

    max_step = 20
    for step in range(init_step + period, max_step + 1):
      if step % period == 0:
        self.assertTrue(manager.should_save(step))
      else:
        self.assertFalse(manager.should_save(step))
      self.assertEqual(
          manager.save(train_state.replace_step(step)), step % period == 0)
      self.assertEqual(manager.latest_step(),
                       step if step % period == 0 else step - 1)
    self.assertSameElements(manager.all_steps(), [16, 18, 20])

    restored = manager.restore(
        path=checkpoints.get_checkpoint_dir(checkpoint_dir, step))
    assert_tree_equal(self, train_state.replace_step(max_step), restored)

  def test_cleanup(self):

    def _add_checkpoint_info(*args, **kwargs):
      del args, kwargs
      pass  # Do nothing to simulate failure of finalization.

    global_mesh = test_utils.create_global_mesh((2, 2), ('x', 'y'))
    mesh_axes = P('x', 'y')
    global_input_shape = (2, 2)
    step = np.int32(0)
    train_state = make_train_state(
        global_mesh, global_input_shape, mesh_axes, step=step)

    with mock.patch.object(
        checkpoints.CheckpointManager, '_add_checkpoint_info',
        autospec=True) as commit_callback:
      commit_callback.side_effect = _add_checkpoint_info
      checkpoint_manager = checkpoints.CheckpointManager(
          self.tmp_dir, train_state, FakePartitioner(global_mesh, mesh_axes))
      checkpoint_manager.save(train_state)
      # Step 0 not finalized.
      tmp_checkpoint_pattern = ('checkpoint_*' +
                                orbax.checkpoint.utils.TMP_DIR_SUFFIX + '*')
      self.assertNotEmpty(
          list(checkpoint_manager.directory.glob(tmp_checkpoint_pattern)))

    checkpoint_manager = checkpoints.CheckpointManager(
        self.tmp_dir, train_state, FakePartitioner(global_mesh, mesh_axes))
    self.assertEmpty(
        list(checkpoint_manager.directory.glob(tmp_checkpoint_pattern)))


class DatasetCheckpointHandlerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self.directory = epath.Path(
        self.create_tempdir(name='checkpointing_test').full_path)
    self.dataset = tf.data.Dataset.range(64)

  def test_save_restore(self):
    checkpointer = checkpoints.DatasetCheckpointHandler(
        checkpoint_filename='ckpt')
    iterator = iter(self.dataset)
    # Change iterator state to check restoration of original state.
    for _ in range(10):
      next(iterator)
    checkpointer.save(self.directory, iterator)
    restored = checkpointer.restore(self.directory, iter(self.dataset))
    self.assertEqual(10, next(restored).numpy())

  # We stub out jax.process_index() and jax.process_count() with the
  # dataset_checkpoint_handler module. This will not affect other modules
  # (which would break JAX multihost utils).
  @mock.patch.object(checkpoints, 'jax')
  def test_save_restore_multihost(self, jax_mock):
    jax_mock.process_count.return_value = 2
    handler = checkpoints.DatasetCheckpointHandler(checkpoint_filename='ckpt')

    # Process 0 - save().
    jax_mock.process_index.return_value = 0
    iterator = iter(self.dataset)
    # Change iterator state to check restoration of original state.
    for _ in range(10):
      next(iterator)
    handler.save(self.directory, iterator)
    # Sub-directory with checkpoint for this host was created.
    self.assertIn('process_0-of-2', [p.name for p in self.directory.iterdir()])

    # Process 1 - save().
    jax_mock.process_index.return_value = 1
    iterator = iter(self.dataset)
    # Change iterator state to check restoration of original state.
    for _ in range(5):
      next(iterator)
    handler.save(self.directory, iterator)
    # Sub-directory with checkpoint for this host was created.
    self.assertIn('process_1-of-2', [p.name for p in self.directory.iterdir()])

    # Process 0 - restore()
    jax_mock.process_index.return_value = 0
    restored = handler.restore(self.directory, iter(self.dataset))
    self.assertEqual(10, next(restored).numpy())

    # Process 1 - restore()
    jax_mock.process_index.return_value = 1
    restored = handler.restore(self.directory, iter(self.dataset))
    self.assertEqual(5, next(restored).numpy())


if __name__ == '__main__':
  absltest.main()
