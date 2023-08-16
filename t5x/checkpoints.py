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

"""Utilities for reading and writing sharded checkpoints.

The checkpointing utilities here can be used in two ways. The first is to use
the `Checkpointer` class. This requires having an optimizer and various
partitioning utilities setup, but allows for reading and writing of partitioned
parameters. It also allows different hosts to read different parameter
partitions in a multi-host setup, which results in much faster reads. This is
normally used during training where you have already created an optimizer based
on a config.

The second way is to use the `load_t5x_checkpoint` function. This doesn't
require an optimizer to get given up front so it is useful for things like
debugging and analysis of learned weights. However, this means that we cannot do
partitioned reads so loading will be slower than that `Checkpointer` class.
"""
import asyncio
import dataclasses
import functools
import os
import re
import subprocess
import time
import typing
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

from absl import logging
import clu.data
from etils import epath
import flax
from flax import serialization
from flax import traverse_util
import jax
from jax import monitoring
import jax.config
from jax.experimental import multihost_utils
from jax.experimental.array_serialization import serialization as array_serialization
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from t5x import checkpoint_importer
from t5x import checkpoint_utils
from t5x import optimizers
from t5x import partitioning
from t5x import state_utils
from t5x import train_state as train_state_lib
import tensorflow as tf
from tensorflow.io import gfile
import tensorstore as ts
import typing_extensions


PartitionSpec = partitioning.PartitionSpec
PyTree = Any
PyTreeDef = jax.tree_util.PyTreeDef
LazyArray = checkpoint_importer.LazyArray
LazyAwaitableArray = checkpoint_importer.LazyAwaitableArray
LazyThreadPoolArray = checkpoint_importer.LazyThreadPoolArray

# Version 3 is used since 2021-06-10, compared to version 2 the only change is
# that `bfloat16` arrays are written in Tensorstore using its native `bfloat16`
# support instead of casting them to `uint16`.
VERSION = 3
# Desired chunk size is 64MiB.
# This is large enough to keep CNS happy but small enough to support a wide
# range of partitionings.
_DESIRED_CHUNK_SIZE_BYTES = 64 * 1024 * 1024
# TODO(levskaya, adarob): how should we handle stacked/fused variables??
_TRAIN_DS_PREFIX = 'train_ds'
_READ_CHECKPOINT_EVENT: str = '/jax/checkpoint/read/durations_sec'
_WRITE_CHECKPOINT_EVENT: str = '/jax/checkpoint/write/durations_sec'
_TS_CONTEXT = ts.Context({'file_io_concurrency': {'limit': 128}})


def _choose_chunk_shape(write_shape: Sequence[int],
                        target_elements: int) -> List[int]:
  """Chooses a chunk shape that evenly divides write_shape.

  The chunk shape is chosen such that the total number of elements is less than
  or equal to `target_elements`, but is otherwise as large as possible.

  This uses a greedy algorithm that attempts to split the largest dimensions
  first.

  Args:
    write_shape: Write shape for which to choose a chunk shape.
    target_elements: Desired number of elements in chosen chunk shape.  Must be
      >= 1.

  Returns:
    List of length `len(write_shape)` specifying the chosen chunk shape.
  """
  assert target_elements >= 1
  rank = len(write_shape)

  # `dim_factors[i]` is the list of divisors of `write_shape[i]`
  dim_factors = [
      [i for i in range(1, size + 1) if size % i == 0] for size in write_shape
  ]

  # The current chunk shape is:
  # [dim_factors[i][-1] for i in range(rank)]

  def get_total_elements():
    """Returns the number of elements in the current chunk shape."""
    total_elements = 1
    for i in range(rank):
      total_elements *= dim_factors[i][-1]
    return total_elements

  # Reduce the current chunk shape until the desired number of elements is
  # reached.
  while get_total_elements() > target_elements:
    # Greedily reduce the largest dimension.  This is not guaranteed to bring us
    # the closest to `target_elements`, but is simple to implement and should
    # work well enough.
    dim_to_reduce = -1
    dim_to_reduce_size = 1
    for i in range(rank):
      size = dim_factors[i][-1]
      if size > dim_to_reduce_size:
        dim_to_reduce_size = size
        dim_to_reduce = i
    # Can only fail to choose `dim_to_reduce` if all dimensions have size of 1.
    # But that cannot happen since `target_elements >= 1`.
    assert dim_to_reduce_size > 1
    dim_factors[dim_to_reduce].pop()
  return [dim_factors[i][-1] for i in range(rank)]


@dataclasses.dataclass
class _ParameterInfo:
  """Information needed to read/write and slice a partitioned parameter."""
  # The unique parameter name.
  name: str
  # The shape of the parameter.
  shape: Tuple[int]
  # The TensoreStore Spec containing the minimal information for read/write.
  ts_spec: Optional[ts.Spec]
  # The LocalChunkInfo for the part of the parameter local to this host.
  local_chunk_info: Optional[partitioning.LocalChunkInfo]
  # PartitionSpec mesh axes
  axes: Optional[partitioning.PartitionSpec] = None


def register_ts_spec_for_serialization():
  # Register functions with flax.serialization to handle `ts.Spec`.
  def is_dict(s):
    return isinstance(s, (dict, flax.core.FrozenDict))

  serialization.register_serialization_state(
      ts.Spec,
      ty_to_state_dict=lambda t: t.to_json(),
      # The parameter may have been written to tensorstore or msgpack.
      # If the former, a dict of the spec will be stored. If the latter it will
      # be the value itself.
      ty_from_state_dict=lambda t, s: ts.Spec(s) if is_dict(s) else s,
      override=True,
  )


register_ts_spec_for_serialization()


def _run_future_tree(future_tree):
  """Block until all futures are resolved on this host."""
  future_leaves, treedef = jax.tree_util.tree_flatten(future_tree)

  async def run():
    return await asyncio.gather(*future_leaves)

  leaves = asyncio.run(run())
  return jax.tree_util.tree_unflatten(treedef, leaves)


def all_steps(checkpoints_dir: str) -> Sequence[int]:
  """Returns list of available step numbers in ascending order."""
  glob_pattern = os.path.join(checkpoints_dir, 'checkpoint_*')
  checkpoint_paths = gfile.glob(glob_pattern)
  re_pattern = re.compile(r'.*/checkpoint_(\d+)$')
  matches = [re_pattern.match(ckpt) for ckpt in checkpoint_paths]
  return sorted(int(match.group(1)) for match in matches if match)


def all_dataset_checkpoint_steps(checkpoints_dir: str) -> Sequence[int]:
  """Returns available dataset checkpoint step numbers in ascending order."""
  glob_pattern = os.path.join(checkpoints_dir, 'checkpoint_*',
                              f'{_TRAIN_DS_PREFIX}-*')
  train_ds_paths = gfile.glob(glob_pattern)
  re_pattern = re.compile(r'.*/checkpoint_(\d+)/.*$')
  matches = [re_pattern.match(path) for path in train_ds_paths]
  return sorted(set(int(match.group(1)) for match in matches if match))


def latest_step(checkpoints_dir: str) -> Optional[int]:
  """Returns latest step number or None if no checkpoints exist."""
  steps = all_steps(checkpoints_dir)
  if not steps:
    return None
  return steps[-1]


def get_local_data(x):
  """Get local buffer for input data."""
  if isinstance(x, jax.Array) and not isinstance(x, jax.core.Tracer):
    return x.addressable_data(0)
  else:
    return x


def _sync_global_devices(name: str) -> None:
  """Sync across all hosts/devices."""
  # Internal mock TPU handling
  multihost_utils.sync_global_devices(name)


def get_checkpoint_dir(
    checkpoints_dir: epath.PathLike,
    step: int,
    step_format_fixed_length: Optional[int] = None,
) -> epath.PathLike:
  """Returns path to a checkpoint dir given a parent directory and step."""
  step_str = (
      f'{step:0{step_format_fixed_length}d}'
      if step_format_fixed_length is not None
      else str(step)
  )
  return os.path.join(checkpoints_dir, f'checkpoint_{step_str}')


def get_step_from_checkpoint_dir(checkpoints_dir: str) -> Tuple[str, int]:
  """Returns a step number and the parent directory."""
  if checkpoints_dir.endswith('/'):
    checkpoints_dir = checkpoints_dir[:-1]
  parent, checkpoint = os.path.split(checkpoints_dir)
  if 'checkpoint_' not in checkpoint:
    raise ValueError('Found improperly formatted checkpoint directory.')
  return parent, int(checkpoint.replace('checkpoint_', ''))


def _cast(target: PyTree, dtype: jnp.dtype):
  """Cast arrays in target to dtype."""

  def maybe_cast(x):
    if isinstance(x, (int, str)):
      # Ignore common non-array types that shouldn't be cast.
      return x
    elif x.dtype == dtype:
      return x
    elif isinstance(x, jax.ShapeDtypeStruct):
      return jax.ShapeDtypeStruct(x.shape, dtype)
    else:
      return x.astype(dtype)

  return jax.tree_util.tree_map(maybe_cast, target)


def _update_ts_path_from_relative_to_absolute(
    ckpt_dir: str, ts_spec_dict: MutableMapping[str, Any]):
  """Update (in-place) the path and gcs bucket (if applicable) in a TS Spec."""

  # Handle `gs://` paths.
  m = re.fullmatch('^gs://([^/]*)/(.*)$', ckpt_dir, re.DOTALL)
  if m is not None:
    if ts_spec_dict['kvstore']['driver'] != 'gcs':
      raise ValueError(f'Incorrect TensorStore Spec.  '
                       f'Expects kvstore driver to be "gcs" for {ckpt_dir}.  '
                       f'Got {ts_spec_dict}')
    bucket = m.group(1)
    ckpt_dir = m.group(2)
    ts_spec_dict['kvstore']['bucket'] = bucket

  # Update the path with `ckpt_dir`

  if 'path' in ts_spec_dict['kvstore']:
    # tensorstore>=0.1.14 format
    ts_spec_dict['kvstore']['path'] = os.path.join(
        ckpt_dir, ts_spec_dict['kvstore']['path'])
  elif 'path' in ts_spec_dict:
    # tensorstore<0.1.14 format
    ts_spec_dict['path'] = os.path.join(ckpt_dir, ts_spec_dict['path'])
  else:
    raise ValueError(
        'Incorrect TensorStore Spec. Expects "path" to be a key of spec or '
        f'`spec["kvstore"]`. Got {ts_spec_dict}')


def _maybe_update_ts_from_file_to_gcs(ckpt_contents):
  """Updates the TensorStore driver from gfile to gcs."""

  def _gfile_to_gcs_driver(arr_or_ts_spec_dict):
    """Converts the ts.Spec dict using gfile driver to gcs driver."""
    if not isinstance(arr_or_ts_spec_dict, dict):
      return arr_or_ts_spec_dict

    if arr_or_ts_spec_dict['kvstore']['driver'] in ('file', 'gfile'):
      ts_spec_dict = arr_or_ts_spec_dict
      path = ts_spec_dict['kvstore'].pop('path')
      # This will be updated to the actual bucket in `_read_ts`.
      ts_spec_dict['kvstore'] = {
          'bucket': 't5x-dummy-bucket',
          'driver': 'gcs',
          'path': path
      }
    else:
      if arr_or_ts_spec_dict['kvstore']['driver'] != 'gcs':
        raise ValueError('Unsupported TensoreStore driver. Got '
                         f'{arr_or_ts_spec_dict["kvstore"]["driver"]}.')
      ts_spec_dict = arr_or_ts_spec_dict

    return ts_spec_dict

  def _is_leaf(value):
    return not isinstance(
        value, dict) or set(value.keys()) >= {'driver', 'kvstore', 'metadata'}

  return jax.tree_util.tree_map(
      _gfile_to_gcs_driver, ckpt_contents, is_leaf=_is_leaf)


def _maybe_update_ts_from_gcs_to_file(ckpt_contents):
  """Updates the TensorStore driver to gfile or file if different."""

  # if saved in gcs, change to file
  def _gcs_to_file_driver(arr_or_ts_spec_dict):
    if not isinstance(arr_or_ts_spec_dict, dict):
      return arr_or_ts_spec_dict

    if arr_or_ts_spec_dict['kvstore']['driver'] == 'gcs':
      ts_spec_dict = arr_or_ts_spec_dict
      path = ts_spec_dict['kvstore'].pop('path')
      driver = 'file'
      ts_spec_dict['kvstore'] = {'path': path, 'driver': driver}
    elif arr_or_ts_spec_dict['kvstore']['driver'] == 'gfile':
      ts_spec_dict = arr_or_ts_spec_dict
      driver = 'file'
      ts_spec_dict['kvstore']['driver'] = driver
    elif arr_or_ts_spec_dict['kvstore']['driver'] == 'file':
      ts_spec_dict = arr_or_ts_spec_dict
    else:
      raise ValueError('Unsupported TensoreStore driver. Got '
                       f'{arr_or_ts_spec_dict["kvstore"]["driver"]}.')

    return ts_spec_dict

  def _is_leaf(value):
    return not isinstance(
        value, dict) or set(value.keys()) >= {'driver', 'kvstore', 'metadata'}

  return jax.tree_util.tree_map(
      _gcs_to_file_driver, ckpt_contents, is_leaf=_is_leaf)


def _get_spec(directory: str, arr: Any, name: str,
              metadata: Dict[str, Any]) -> ts.Spec:
  """Get ts.Spec from array and name information."""

  if os.fspath(directory).startswith('gs://'):
    spec = {
        'driver': 'zarr',
        'dtype': jnp.dtype(arr.dtype).name,
        'kvstore': {
            'driver': 'gcs',
            # We always write with a dummy bucket and dynamically update the
            # bucket information. This makes the checkpoint files portable
            # and not bind to the bucket that it was originally written to.
            'bucket': 't5x-dummy-bucket',
        },
        'path': name.replace('/', '.'),
        'metadata': metadata,
    }
  else:
    spec = {
        'driver': 'zarr',
        'dtype': jnp.dtype(arr.dtype).name,
        'kvstore': {
            'driver': 'file',
            'path': name.replace('/', '.')
        },
        'metadata': metadata,
    }

  return ts.Spec(spec)


def _sharding_matches(arr: Any, target_sharding: jax.sharding.Sharding) -> bool:
  if not isinstance(arr, jax.Array):
    return False
  sharding = arr.sharding
  return sharding.is_equivalent_to(target_sharding, arr.ndim)


def _maybe_make_sharded_array(
    arr: Any,
    mesh: Optional[jax.sharding.Mesh],
    axes: Optional[PartitionSpec] = None,
    restore_dtype: Optional[jnp.dtype] = None,
    params_on_devices: bool = True,
) -> Any:
  """Makes a sharded array from non-sharded array if necessary.

  Args:
    arr: array to maybe shard.
    mesh: jax.sharding.Mesh.
    axes: mesh_axes.
    restore_dtype: type to restore as.
    params_on_devices: If true, the array will be placed on device. Otherwise,
      it will be stored in the host(s) RAM.

  Returns:
    Sharded or unsharded array.
  """
  if axes is None:
    axes = PartitionSpec(None)
  assert mesh is not None, 'jax.sharding.Mesh should be provided.'
  target_sharding = jax.sharding.NamedSharding(mesh, axes)
  if _sharding_matches(arr, target_sharding):
    return arr
  if isinstance(arr, (np.ndarray, jnp.ndarray)):
    if restore_dtype is not None:
      arr = arr.astype(restore_dtype)
    if not params_on_devices:
      return arr
    arr = jax.make_array_from_callback(
        arr.shape, target_sharding, lambda idx: arr[idx]
    )
  return arr


class _BytesConditionVariable(object):
  """Wraps a condition variable to control concurrency based on bytes."""

  def __init__(self, num_bytes):
    self._max_bytes = num_bytes
    self._num_bytes = num_bytes
    self._cv = asyncio.Condition(lock=asyncio.Lock())

  async def wait_for_bytes(self, n_bytes):
    async with self._cv:
      await self._cv.wait_for(lambda: self._num_bytes > n_bytes)
      self._num_bytes -= n_bytes
      assert self._num_bytes >= 0

  async def return_bytes(self, n_bytes):
    async with self._cv:
      self._num_bytes += n_bytes
      assert self._num_bytes <= self._max_bytes
      self._cv.notify_all()


class SaveStateTransformationFn(typing_extensions.Protocol):

  def __call__(
      self, state_dict: PyTree, parameter_infos: PyTree
  ) -> Tuple[PyTree, PyTree]:
    """Transforms the state and param info, e.g., by remapping parameters.

    Args:
      state_dict: State in the current model.
      parameter_infos: PyTree containing `_ParameterInfo` objects.

    Returns:
      A tuple whose first element is the result of transforming `state_dict` and
      whose second element is the result of transforming `parameter_infos`.
    """


class RestoreStateTransformationFn(typing_extensions.Protocol):

  def __call__(
      self,
      state_dict: PyTree,
      target_state_dict: PyTree,
      *,
      is_resuming: bool = False,
  ) -> PyTree:
    """Transforms the given checkpoint state, e.g., by remapping parameters.

    Args:
      state_dict: State to transform, which could be from a previous version of
        the model.
      target_state_dict: State in the current model.
      is_resuming: `True` iff this restore call is due to a job resuming after
        being temporarily stopped due to, for example, a preemption. This is
        useful when there is restore logic that should run when restoring from
        some pre-existing checkpoint, but that should not run again when
        resuming from a newly-written checkpoint.

    Returns:
      The result of transforming the `state_dict`.
    """


class _TfDataCheckpointer:

  def __init__(self, dataset_iterator: tf.data.Iterator):
    self._dataset_ckpt = tf.train.Checkpoint(ds=dataset_iterator)

  def save(self, filename: str):
    self._dataset_ckpt.write(filename)

  def load(self, filename: str):
    self._dataset_ckpt.read(filename).assert_consumed()


# TODO(b/216649487): Replace with CheckpointManager.
class Checkpointer(object):
  """Handles saving and restoring potentially-sharded T5X checkpoints.

  Checkpoints are stored using a combination of msgpack (via flax.serialization)
  and TensorStore.

  Parameters (and other objects) that are not partitioned are written to the
  msgpack binary directly (by host 0). Partitioned parameters are each written
  to their own TensorStore, with each host writing their portion to the same
  TensorStore in parallel. If a partition is written on multiple hosts, the
  partition is further sharded across these replicas to avoid additional
  overhead. In place of the parameter, a `tensorstore.Spec` is written to the
  msgpack (by host 0) as a reference to be used during restore. Note that the
  path of the array being written is relative. This makes the checkpoints
  portable. In other words, even if the checkpoint files are moved to a new
  directory, they can still be loaded. Because the path is relative, the
  checkpoint directory information has to be dynamically provided. This is done
  by `_update_ts_path_from_relative_to_absolute`.

  For TensorStore driver using Google Cloud Storage (GCS) Key-Value Storage
  Layer, the GCS bucket information is necessary. When a checkpoint is written
  using the gcs driver, we don't want to hardcode the bucket information in the
  resulting file in order to maintain the portability. Therefore, we use a dummy
  bucket name of "t5x-dummy-bucket". When reading or writing the checkpoint, the
  bucket information is parsed from the checkpoint directory and the bucket
  information is dynamically updated.

  Attributes:
    checkpoints_dir: a path to a directory to save checkpoints in and restore
      them from.
    keep: an optional maximum number of checkpoints to keep. If more than this
      number of checkpoints exist after a save, the oldest ones will be
      automatically deleted to save space.
    restore_dtype: optional dtype to cast targets to after restoring.
    save_dtype: dtype to cast targets to before saving.
    keep_dataset_checkpoints: an optional maximum number of data iterators to
      keep. If more than this number of data iterators exist after a save, the
      oldest ones will be automatically deleted to save space.
  """

  def __init__(
      self,
      train_state: train_state_lib.TrainState,
      partitioner: partitioning.BasePartitioner,
      checkpoints_dir: epath.PathLike,
      dataset_iterator: Optional[
          Union[tf.data.Iterator, clu.data.dataset_iterator.DatasetIterator]
      ] = None,
      *,
      keep: Optional[int] = None,
      save_dtype: jnp.dtype = np.float32,
      restore_dtype: Optional[jnp.dtype] = None,
      keep_dataset_checkpoints: Optional[int] = None,
  ):
    """Checkpointer constructor.

    Args:
      train_state: A train state to be used to determine the structure of the
        parameter tree, and the *full* (non-partitioned) parameter shapes and
        dtypes. Saved and restored train states must match this structure.
      partitioner: the partitioner to use for determining the local chunks
        mapping or to perform params partitioning on restore.
      checkpoints_dir: a path to a directory to save checkpoints in and restore
        them from.
      dataset_iterator: an optional iterator to save/restore.
      keep: an optional maximum number of checkpoints to keep. If more than this
        number of checkpoints exist after a save, the oldest ones will be
        automatically deleted to save space.
      save_dtype: dtype to cast targets to before saving.
      restore_dtype: optional dtype to cast targets to after restoring. If None,
        no parameter casting is performed.
      keep_dataset_checkpoints: an optional maximum number of data iterators to
        keep. If more than this number of data iterators exist after a save, the
        oldest ones will be automatically deleted to save space.
    """
    self._train_state = train_state
    self._partitioner = partitioner
    self.checkpoints_dir = checkpoints_dir
    self.keep = keep
    self.keep_dataset_checkpoints = keep_dataset_checkpoints
    # Immutable due to use in `_get_parameter_infos`
    self._save_dtype = save_dtype
    self.restore_dtype = restore_dtype
    self._original_dataset_iterator = dataset_iterator
    if isinstance(dataset_iterator, tf.data.Iterator):
      dataset_iterator = _TfDataCheckpointer(dataset_iterator)
    elif isinstance(dataset_iterator,
                    clu.data.dataset_iterator.TfDatasetIterator):
      assert dataset_iterator._checkpoint
    self._dataset_iterator = dataset_iterator

    data_layout = partitioner.get_data_layout()
    self._dataset_ckpt_name = (
        f'{_TRAIN_DS_PREFIX}-'
        f'{data_layout.shard_id:03}-of-{data_layout.num_shards:03}')
    self._should_write_dataset_ckpt = (
        dataset_iterator and data_layout.is_first_host_in_replica_set)

    self._parameter_infos = self._get_parameter_infos()

    asyncio.set_event_loop(asyncio.new_event_loop())

  def _get_state_dict_for_save(
      self,
      state_dict: Dict[str, Any],
      lazy_load: bool = True) -> MutableMapping[str, Any]:
    """Gets the optimizer state dict."""

    def _lazy_load_device_array(arr):
      if isinstance(arr, jax.Array):
        if len(arr.sharding.device_set) == 1:
          return LazyThreadPoolArray(arr.shape, arr.dtype,
                                     lambda: np.array(arr))
      return arr

    if lazy_load:
      state_dict = jax.tree_util.tree_map(_lazy_load_device_array, state_dict)
    return state_dict

  def _get_parameter_infos(self):
    """Generates the state dict of _ParameterInfos for the Optimizer.

    We generate a state dict (matching the shape of the optimizer state dict)
    that stores a _ParameterInfo for each parameter array.

    The _ParameterInfo contains the TensorStore spec for the parameter array and
    the LocalChunkInfo describing the slice of the array local to this host.

    Returns:
      The state dict of _ParameterInfo objects.
    """

    def _get_param_info(name: str, arr: Any, axes: partitioning.PartitionSpec):
      # If a node in your model is None it is probably a param_state that is not
      # used because of a MultiOptimizer. We don't want to have any parameter
      # info for it because it shouldn't be saved or restored.
      if arr is None:
        return None
      # Pass-through empty dict leaves, which occur with optax EmptyState().
      if isinstance(arr, dict) and not arr:
        return {}

      if axes is None:
        return _ParameterInfo(
            name=name,
            shape=arr.shape,
            ts_spec=None,
            local_chunk_info=None,
            axes=None)

      if isinstance(arr, jax.Array):
        local_chunk_info = None
        metadata = array_serialization._get_metadata(arr)  # pylint: disable=protected-access
        del metadata['dtype']
      else:
        local_chunk_info = self._partitioner.get_local_chunk_info(
            arr.shape, axes)
        write_shape = [
            si if sl == slice(None) else sl.stop - sl.start
            for si, sl in zip(arr.shape, local_chunk_info.slice)
        ]
        # TODO(levskaya, adarob): how should we handle stacked/fused variables??
        chunk_shape = _choose_chunk_shape(
            write_shape,
            target_elements=_DESIRED_CHUNK_SIZE_BYTES / arr.dtype.itemsize)

        metadata = {
            'compressor': {
                'id': 'gzip'
            },
            'shape': arr.shape,
            'chunks': np.array(chunk_shape),
        }

      spec = _get_spec(self.checkpoints_dir, arr, name, metadata)

      return _ParameterInfo(
          name,
          shape=arr.shape,
          ts_spec=spec,
          local_chunk_info=local_chunk_info,
          axes=axes)

    # Create a tree of param names as the keys on the path to each leaf
    # separated by "/".
    param_names = traverse_util.unflatten_dict({
        k: '/'.join(k) for k in traverse_util.flatten_dict(
            self._train_state.state_dict(), keep_empty_nodes=True)
    })

    return jax.tree_util.tree_map(
        _get_param_info, param_names,
        self._get_state_dict_for_save(self._train_state.state_dict()),
        self._partitioner.get_mesh_axes(self._train_state).state_dict())

  def _get_checkpoint_dir(self, step: int) -> epath.PathLike:
    return get_checkpoint_dir(self.checkpoints_dir, step)

  def all_steps(self) -> Sequence[int]:
    """Returns list of available step numbers in ascending order."""
    return all_steps(self.checkpoints_dir)

  def all_dataset_checkpoint_steps(self) -> Sequence[int]:
    """Returns list of available step numbers in ascending order."""
    return all_dataset_checkpoint_steps(self.checkpoints_dir)

  def latest_step(self) -> Optional[int]:
    """Returns latest step number or None if no checkpoints exist."""
    return latest_step(self.checkpoints_dir)

  def _remove_old_dataset_checkpoints(self):
    """Deletes old dataset checkpoints if there are more than allowed."""
    if self.keep_dataset_checkpoints:
      existing_steps = self.all_dataset_checkpoint_steps()
      to_remove = len(existing_steps) - self.keep_dataset_checkpoints
      if to_remove > 0:
        for step in existing_steps[:to_remove]:
          checkpoint_utils.remove_dataset_checkpoint(
              self._get_checkpoint_dir(step), _TRAIN_DS_PREFIX)

  def _remove_old_checkpoints(self):
    """Deletes oldest checkpoints if there are more than keep_checkpoints."""
    if not self.keep:
      return
    existing_steps = self.all_steps()
    to_remove = len(existing_steps) - self.keep
    if to_remove <= 0:
      return

    for step in existing_steps[:to_remove]:
      checkpoint_utils.remove_checkpoint_dir(self._get_checkpoint_dir(step))

  def save(self,
           train_state: train_state_lib.TrainState,
           state_transformation_fns: Sequence[SaveStateTransformationFn] = (),
           *,
           concurrent_gb: int = 128):
    """Saves a checkpoint for the given train state.

    Args:
      train_state: the train state to save. May contain a combination of
        LazyArray objects and arrays (e.g., np.ndarray, jax.DeviceArray)
      state_transformation_fns: Transformations to apply, in order, to the state
        before writing.
      concurrent_gb: the approximate number of gigabytes of partitionable
        parameters to process in parallel. Useful to preserve RAM.
    """
    start_time = time.time()
    step = train_state.step
    step = step.get() if isinstance(step, LazyArray) else step
    step = get_local_data(step)
    # Integer, to avoid side effects in the checkpoint path.
    step = int(step)

    # Share a timestamp across devices.
    timestamp = multihost_utils.broadcast_one_to_all(np.int32(time.time()))

    final_dir = os.path.join(self.checkpoints_dir, f'checkpoint_{step}')
    tmp_dir = final_dir + f'.tmp-{timestamp}'

    if gfile.exists(final_dir):
      logging.info(
          'Skipping save checkpoint for step %d (directory %s already exists)',
          step, final_dir)
      return

    logging.info('Saving checkpoint for step %d to %s', step, tmp_dir)

    if jax.process_index() == 0:
      gfile.makedirs(tmp_dir)
    # Block all hosts until directory is ready.
    _sync_global_devices(f'checkpointer:make_dir:{tmp_dir}')

    written_state_dict = self._write_state_to_tensorstore(
        tmp_dir, train_state, concurrent_gb, state_transformation_fns)

    if self._should_write_dataset_ckpt:
      logging.info("Writing dataset iterator state to '%s'.",
                   self._dataset_ckpt_name)
      try:
        self._dataset_iterator.save(
            os.path.join(tmp_dir, self._dataset_ckpt_name))
      except tf.errors.FailedPreconditionError as e:
        logging.error(
            'Input pipeline must be stateless in order to checkpoint. Cache '
            'stateful steps offline or disable iterator checkpointing.')
        raise e

    # Block until complete on all hosts.
    _sync_global_devices(f'checkpointer:tensorstore_write_complete:{tmp_dir}')

    if jax.process_index() == 0:
      written_state_dict = jax.tree_util.tree_map(get_local_data,
                                                  written_state_dict)

      # Write msgpack file in host 0 only
      msgpack_bytes = serialization.to_bytes({
          'version': VERSION,
          'optimizer': written_state_dict
      })
      with gfile.GFile(os.path.join(tmp_dir, 'checkpoint'), 'wb') as fp:
        fp.write(msgpack_bytes)

      # Finalize checkpoint directory.
      if final_dir.startswith('gs://'):
        subprocess.run(['gsutil', '-m', 'mv', tmp_dir, final_dir],
                       stdout=subprocess.DEVNULL,
                       check=True)
      else:
        gfile.rename(tmp_dir, final_dir)
      logging.info('Saved checkpoint for step %d to %s', step, final_dir)

      # Remove old checkpoints, if necessary.
      self._remove_old_checkpoints()
      self._remove_old_dataset_checkpoints()

    # Block until complete on all hosts.
    _sync_global_devices(f'checkpointer:write_complete:{final_dir}')

    end_time = time.time()
    monitoring.record_event_duration_secs(_WRITE_CHECKPOINT_EVENT,
                                          end_time - start_time)
    ocp.utils.record_saved_duration(start_time)

  def _write_state_to_tensorstore(
      self,
      ckpt_dir: str,
      train_state: train_state_lib.TrainState,
      concurrent_gb: int,
      state_transformation_fns: Sequence[SaveStateTransformationFn],
  ) -> Mapping[str, Any]:
    """Writes extracted state from train state to Tensorstore."""
    concurrent_bytes = concurrent_gb * 10**9

    async def _write_array(maybe_arr: Any,
                           param_info: Optional[_ParameterInfo],
                           cast: bool = False):
      """Maybe write to TensorStore, returning object to write to msgpack.

      Args:
        maybe_arr: array or LazyArray to be written
        param_info: ParameterInfo object. If None (or if param_info.ts_spec is
          None), the array will be immediately returned without writing to
          tensorstore. This is because array is None or is not partitioned, and
          should be written separately.
        cast: if True, performs cast operation using self._save_dtype.

      Returns:
        Tensorstore spec corresponding to the written array.
      """
      bytes_cv = _BytesConditionVariable(concurrent_bytes)

      if isinstance(maybe_arr, LazyArray):
        maybe_arr = await maybe_arr.get_async()

      if param_info is None or param_info.ts_spec is None:
        # Write to the msgpack file on host 0.
        return maybe_arr

      arr = maybe_arr
      # Wait until memory is available.
      if isinstance(arr, jax.Array):
        n_bytes = sum(
            [
                shard.data.nbytes
                for shard in arr.addressable_shards
                if shard.replica_id == 0
            ]
        )
      else:
        n_bytes = arr.nbytes
      if n_bytes > concurrent_bytes:
        logging.warning(
            (
                'Temporarily increasing the concurrency limits from %d bytes to'
                ' %d bytes to fit %s.'
            ),
            concurrent_bytes,
            n_bytes,
            param_info.name,
        )
        n_bytes = concurrent_bytes
      await bytes_cv.wait_for_bytes(n_bytes)

      tmp_ts_spec_dict = param_info.ts_spec.to_json()
      if cast:
        # Set desired destination dtype.
        tmp_ts_spec_dict['dtype'] = jnp.dtype(self._save_dtype).name
      param_info.ts_spec = ts.Spec(tmp_ts_spec_dict)
      # Path and gcs bucket (if applicable) information is updated in-place.
      _update_ts_path_from_relative_to_absolute(ckpt_dir, tmp_ts_spec_dict)
      if cast:
        # Set up casting spec.
        tmp_ts_spec_dict = {
            'base': tmp_ts_spec_dict,
            'driver': 'cast',
            'dtype': jnp.dtype(arr.dtype).name,  # dtype before cast
        }

      if isinstance(arr, jax.Array):
        await array_serialization.async_serialize(arr, tmp_ts_spec_dict)
      else:
        # Array is assumed to be replicated on all hosts in this case.
        t = await ts.open(
            tmp_ts_spec_dict,
            create=True,
            open=True,
            context=ts.Context({'file_io_concurrency': {'limit': 128}}),
        )
        await t.write(arr)
      await bytes_cv.return_bytes(n_bytes)

      # N.B. we return the original ts_spec (before
      # `_update_ts_path_from_relative_to_absolute` was called). This is because
      # we'd like to keep the path as relative, i.e., it doesn't hardcode the
      # directory that the checkpoint was originally written. This makes the
      # checkpoints portable.
      return param_info.ts_spec

    transformed_state_dict, transformed_parameter_infos = (
        _transform_state_and_infos(train_state.state_dict(),
                                   self._parameter_infos,
                                   state_transformation_fns))

    state_dict_for_save = self._get_state_dict_for_save(transformed_state_dict)

    def _cast_arr_if_not_partitioned(maybe_arr, param_info):
      if param_info is None or param_info.ts_spec is None:
        return _cast(maybe_arr, self._save_dtype)
      return maybe_arr

    state_dict_for_save['target'] = jax.tree_util.tree_map(  # pytype: disable=unsupported-operands  # dynamic-method-lookup
        _cast_arr_if_not_partitioned, state_dict_for_save['target'],
        transformed_parameter_infos['target'])
    future_written_state = {}
    for k in state_dict_for_save.keys():
      # ensure that only 'target' is cast
      future_written_state[k] = jax.tree_util.tree_map(
          functools.partial(
              _write_array,
              cast=(k == 'target' and self._save_dtype is not None)),
          state_dict_for_save[k], transformed_parameter_infos[k])

    # Block until complete on this host.
    written_state_dict = _run_future_tree(future_written_state)

    # Block until complete on all hosts.
    _sync_global_devices(f'checkpointer:ts_write_complete:{ckpt_dir}')

    return written_state_dict

  def _transform_state_and_infos(
      self,
      state_dict: PyTree,
      parameter_infos: PyTree,
      state_transformation_fns: Sequence[SaveStateTransformationFn],
  ) -> Tuple[PyTree, PyTree]:
    """Applies transformations to the state dict and parameter infos PyTrees."""
    return _transform_state_and_infos(state_dict, parameter_infos,
                                      state_transformation_fns)

  def restore(
      self,
      step: Optional[int] = None,
      path: Optional[str] = None,
      state_transformation_fns: Sequence[RestoreStateTransformationFn] = (),
      fallback_state: Optional[Mapping[str, Any]] = None,
      lazy_parameters: bool = False) -> train_state_lib.TrainState:
    """Restores the host-specific parameters in an Optimizer.

    Either `step` or `path` can be specified, but not both. If neither are
    specified, restores from the latest checkpoint in the checkpoints directory.

    Args:
      step: the optional step number to restore from.
      path: an optional absolute path to a checkpoint file to restore from.
      state_transformation_fns: Transformations to apply, in order, to the state
        after reading.
      fallback_state: a state dict of an optimizer to fall back to for loading
        params that do not exist in the checkpoint (after applying all
        `state_transformation_fns`), but do exist in `Checkpointer.optimizer`.
        The union of `fallback_state` and state loaded from the checkpoint must
        match `Checkpointer.optimizer`.
      lazy_parameters: whether to load the parameters as LazyArrays to preserve
        memory.

    Returns:
      The restored train state.

    Raises:
      ValueError if both `step` and `path` are specified.
      ValueError if checkpoint at `path` or `step` does not exist.
      ValueError if `step` and `path` are not specified and no checkpoint is
        found in the checkpoints directory.
    """
    start_time = time.time()
    if lazy_parameters and self._partitioner.params_on_devices:
      raise ValueError('Lazy Parameters cannot be copied to devices, please '
                       'set partitioner.params_on_devices=False.')
    if step is not None and path is not None:
      raise ValueError('At most one of `step` or `path` may be provided.')
    if path:
      ckpt_path = path
    else:
      if step is None:
        step = self.latest_step()
        if not step:
          raise ValueError(f'No checkpoints found in {self.checkpoints_dir}.')
      ckpt_path = self._get_checkpoint_dir(step)

    if gfile.isdir(ckpt_path):
      ckpt_dir = ckpt_path
      if gfile.isdir(os.path.join(ckpt_dir, _STATE_KEY)):
        ckpt_path = os.path.join(ckpt_path, _STATE_KEY, 'checkpoint')
      else:
        ckpt_path = os.path.join(ckpt_path, 'checkpoint')
    else:
      ckpt_dir = os.path.dirname(ckpt_path)

    if not gfile.exists(ckpt_path) or gfile.isdir(ckpt_path):
      raise ValueError(f'Path is not a valid T5X checkpoint: {ckpt_path}')

    ckpt_type = checkpoint_utils.detect_checkpoint_type(
        ckpt_path, expected=checkpoint_utils.CheckpointTypes.T5X
    )
    if ckpt_type is checkpoint_utils.CheckpointTypes.T5X_TF:
      raise ValueError(
          'Attempting to restore a TensorFlow checkpoint as a native T5X '
          'checkpoint. Use `restore_from_tf_checkpoint` instead. Path: '
          f'{ckpt_path}'
      )
    # Don't error out here for Orbax-detected checkpoint (there are edge cases
    # where all values are stored in the msgpack and the checkpoint file can be
    # loaded by both the Orbax and T5X checkpointer).
    logging.info('Restoring from checkpoint: %s', ckpt_path)

    with gfile.GFile(ckpt_path, 'rb') as fp:
      # TODO(adarob): Use threaded reading as in flax.checkpoints.
      # `ckpt_contents['optimizer']` is a pytree with a realized np.array for
      # leaves (params or states) written as msgpack and a ts.Spec (in a dict)
      # for leaves written by TensorStore.
      ckpt_contents = serialization.msgpack_restore(fp.read())

    # If reading a ckpt that was written with gfile driver but the current
    # session uses the gcs driver, convert the ckpt's driver to gcs.
    if os.fspath(ckpt_dir).startswith('gs://'):
      ckpt_contents = _maybe_update_ts_from_file_to_gcs(ckpt_contents)
    # If a ckpt was saved in gcs and is being loaded locally, then convert the
    # driver to file or gfile. If the ckpt was not saved in gcs, do not change.
    else:
      ckpt_contents = _maybe_update_ts_from_gcs_to_file(ckpt_contents)

    ckpt_state_dict = self._get_optimizer_state_dict(ckpt_contents,
                                                     state_transformation_fns)

    # The state dict may contain TensorStore specs that need to be read.
    dummy_spec = ts.Spec({'driver': 'zarr', 'kvstore': {'driver': 'memory'}})

    # `dummy_written_state_dict` is a pytree with a `dummy_spec` for leaves
    # (params or states) written as msgpack and a ts.Spec (in a dict) for leaves
    # written by TensorStore.
    dummy_written_state_dict = jax.tree_util.tree_map(
        lambda x: x.ts_spec or dummy_spec,
        self._parameter_infos,
    )

    if fallback_state is None:
      restore_parameter_infos = self._parameter_infos
    else:
      # If `fallback_state` was specified, restore only the subset
      # of parameters matched by `self._get_optimizer_state_dict`. The
      # rest will be provided by `fallback_state`.
      dummy_written_state_dict = state_utils.intersect_state(
          dummy_written_state_dict, ckpt_state_dict)
      restore_parameter_infos = state_utils.intersect_state(
          self._parameter_infos, ckpt_state_dict)

    restore_parameter_infos_flat = state_utils.flatten_state_dict(
        restore_parameter_infos)
    for key in restore_parameter_infos_flat.keys():
      logging.info('Restoring key from ckpt: %s', key)

    # NB: `serialization.from_state_dict` doesn't check whether the shapes match
    # at the leaf level. Non-partitioned leaves (e.g., optimizer states) can
    # load arrays with inconsistent shapes.
    # `written_state_dict` is a pytree with a realized np.array for leaves
    # (params or states) written as msgpack and a `ts.Spec` for leaves written
    # by TensorStore.
    written_state_dict = serialization.from_state_dict(dummy_written_state_dict,
                                                       ckpt_state_dict)
    state_dict = self._read_state_from_tensorstore(
        ckpt_path,
        written_state_dict,
        restore_parameter_infos=restore_parameter_infos,
        lazy_parameters=lazy_parameters)

    # If `fallback_state` was specified, then fill the missing parameters.
    if fallback_state is not None:
      state_dict = state_utils.merge_state(state_dict, fallback_state)

    for key in state_utils.flatten_state_dict(state_dict).keys():
      if key not in restore_parameter_infos_flat:
        logging.info('Not restoring key from ckpt: %s', key)

    if self._dataset_iterator:
      logging.info("Restoring dataset iterator from '%s'.",
                   self._dataset_ckpt_name)
      self._dataset_iterator.load(
          os.path.join(ckpt_dir, self._dataset_ckpt_name))

    restored_train_state = self._restore_train_state(state_dict)

    end_time = time.time()
    monitoring.record_event_duration_secs(_READ_CHECKPOINT_EVENT,
                                          end_time - start_time)
    return restored_train_state

  def _restore_train_state(
      self,
      state_dict: optimizers.OptimizerStateType) -> train_state_lib.TrainState:
    """Restores a TrainState from an Optimizer state_dict."""
    return self._train_state.restore_state(state_dict)

  def _create_lazy_awaitable_array(
      self, param_info: _ParameterInfo, maybe_ts_spec: Any, ckpt_path: str,
      restore_dtype: Optional[jnp.dtype]) -> LazyAwaitableArray:
    """Creates LazyArray from tensorstore.

    Does not materialize the array immediately.

    Args:
      param_info: Information about how to read the parameter, host based sliced
        reads and the like.
      maybe_ts_spec: The tensorstore spec to read the parameter or some other
        object. If this is an array then we will do a host based sliced read on
        it (provided the param_info says to). Anything else we just return.
      ckpt_path: A base location to use when resolving the relative paths in the
        tensorstore spec.
      restore_dtype: type to restore as. None indicates that no cast is
        requested.

    Returns:
      LazyArray object.
    """
    mesh = self._partitioner.mesh
    axes = param_info.axes

    async def get_fn():
      nonlocal mesh
      nonlocal axes
      arr = await _read_ts(
          param_info,
          maybe_ts_spec,
          ckpt_path=ckpt_path,
          restore_dtype=restore_dtype,
          mesh=mesh,
          axes=axes,
          params_on_devices=self._partitioner.params_on_devices,
      )
      return _maybe_make_sharded_array(
          arr,
          mesh,
          axes=axes,
          restore_dtype=restore_dtype,
          params_on_devices=self._partitioner.params_on_devices,
      )

    return LazyAwaitableArray.from_tensor_store_spec_or_array(
        maybe_ts_spec, get_fn, dtype=restore_dtype)

  def _read_state_from_tensorstore(
      self,
      ckpt_path: str,
      written_state_dict: Mapping[str, Any],
      restore_parameter_infos: Optional[Mapping[str, Any]] = None,
      lazy_parameters: bool = False,
  ) -> Mapping[str, Any]:
    """Sets up lazy reads from Tensorstore and returns them as a state_dict."""
    if restore_parameter_infos is None:
      restore_parameter_infos = self._parameter_infos

    # Replace TensorStore Specs with the lazy array values.
    state_dict = {}
    for k in written_state_dict.keys():
      # ensure that only 'target' is cast
      restore_dtype = self.restore_dtype if k == 'target' else None
      state_dict[k] = jax.tree_util.tree_map(
          functools.partial(
              self._create_lazy_awaitable_array,
              ckpt_path=ckpt_path,
              restore_dtype=restore_dtype), restore_parameter_infos[k],
          written_state_dict[k])

    if not lazy_parameters:
      future_state_dict = jax.tree_util.tree_map(lambda x: x.get_async(),
                                                 state_dict)
      state_dict = _run_future_tree(future_state_dict)

    if self.restore_dtype is not None:
      if 'target' not in state_dict:
        raise ValueError(
            f'restore_dtype={self.restore_dtype} was specified, but no `target`'
            ' parameters were loaded.'
        )
      state_dict['target'] = _cast(state_dict['target'], self.restore_dtype)

    return state_dict

  def restore_from_tf_checkpoint(
      self,
      path_or_dir: str,
      strict: bool = True,
      translator: Optional[checkpoint_importer.CheckpointTranslator] = None
  ) -> train_state_lib.TrainState:
    """Restore from a TensorFlow-based T5 checkpoint."""
    start_time = time.time()
    full_state_dict = checkpoint_importer.restore_from_t5_checkpoint(
        self._train_state.state_dict(),
        path_or_dir,
        lazy_parameters=False,
        strict=strict,
        translator=translator)
    full_state_dict = dict(full_state_dict)

    def _partition_parameter(maybe_arr: Any, param_info: _ParameterInfo):
      if isinstance(maybe_arr, np.ndarray) and param_info:
        arr = maybe_arr
        to_sharded_array = self._partitioner.partition(
            lambda x: x,
            in_axis_resources=None,
            out_axis_resources=param_info.axes,
        )
        return to_sharded_array(arr)
      return maybe_arr

    if self.restore_dtype is not None:
      full_state_dict['target'] = _cast(full_state_dict['target'],
                                        self.restore_dtype)
    state_dict = jax.tree_util.tree_map(_partition_parameter, full_state_dict,
                                        self._parameter_infos)

    restored_train_state = self._restore_train_state(state_dict)

    end_time = time.time()
    monitoring.record_event_duration_secs(_READ_CHECKPOINT_EVENT,
                                          end_time - start_time)

    return restored_train_state

  def convert_from_tf_checkpoint(
      self,
      path_or_dir: str,
      *,
      state_transformation_fns: Sequence[SaveStateTransformationFn] = (),
      concurrent_gb: int = 16,
      translator: Optional[checkpoint_importer.CheckpointTranslator] = None):
    """Convert from a TensorFlow-based T5 checkpoint."""
    full_state_dict = checkpoint_importer.restore_from_t5_checkpoint(
        self._train_state.state_dict(),
        path_or_dir,
        lazy_parameters=True,
        translator=translator)
    train_state = self._train_state.restore_state(full_state_dict)
    self.save(
        train_state,
        state_transformation_fns=state_transformation_fns,
        concurrent_gb=concurrent_gb)

  def _get_optimizer_state_dict(
      self,
      ckpt_contents: PyTree,
      state_transformation_fns: Sequence[RestoreStateTransformationFn],
  ):
    return _get_optimizer_state_dict(ckpt_contents,
                                     self._train_state.state_dict(),
                                     state_transformation_fns)


class CheckpointerConstructor(typing_extensions.Protocol):
  """A function that returns a checkpoints.Checkpointer.

  This type annotation allows users to partially bind args to the constructors
  of Checkpointer subclasses without triggering type errors.
  """

  def __call__(self,
               train_state: train_state_lib.TrainState,
               partitioner: partitioning.BasePartitioner,
               checkpoints_dir: str,
               dataset_iterator: Optional[tf.data.Iterator] = None,
               *,
               keep: Optional[int] = None,
               save_dtype: jnp.dtype = np.float32,
               restore_dtype: Optional[jnp.dtype] = None,
               keep_dataset_checkpoints: Optional[int] = None) -> Checkpointer:
    """Checkpointer constructor.

    Args:
      train_state: A train state to be used to determine the structure of the
        parameter tree, and the *full* (non-partitioned) parameter shapes and
        dtypes. Saved and restored train states must match this structure.
      partitioner: the partitioner to use for determining the local chunks
        mapping or to perform params partitioning on restore.
      checkpoints_dir: a path to a directory to save checkpoints in and restore
        them from.
      dataset_iterator: an optional iterator to save/restore.
      keep: an optional maximum number of checkpoints to keep. If more than this
        number of checkpoints exist after a save, the oldest ones will be
        automatically deleted to save space.
      save_dtype: dtype to cast targets to before saving.
      restore_dtype: optional dtype to cast targets to after restoring. If None,
        no parameter casting is performed.
      keep_dataset_checkpoints: an optional maximum number of data iterators to
        keep. If more than this number of data iterators exist after a save, the
        oldest ones will be automatically deleted to save space.
    """
    pass


def populate_metrics_for_steps(checkpoints_dir: str, metric_name: str,
                               steps: Iterable[int]) -> Mapping[int, float]:
  """Iterate through summary event files and return metrics for `steps`."""

  metric_run, metric_tag = None, None

  def _try_fill_metric_run_and_tag_names(metric_name: str,
                                         run_keys: Iterable[str]) -> bool:
    """Extract metric run and tag names by matching one of the `run_keys`.

    This function tries to greedily split user-provided metric_name_to_monitor
    into {run} and {tag} components. It does so by trying to match all available
    {run}/{tag} names in the provided run_keys. If successful, populates
    metric_run and metric_tag.

    Args:
      metric_name: metric name to monitor.
      run_keys: Set of run keys to test for.

    Returns:
      Whether metric name prefix matches one of the run keys, and, as a
      side-effect, populates metric_run and metric_tag.
    """
    nonlocal metric_run
    nonlocal metric_tag

    # Query existing events for different run and tags to match with user
    # provided metric name.
    m = metric_name.split('/')
    possible_run_names = ['/'.join(m[:i]) for i in range(1, len(m))]
    for key in run_keys:
      for possible_run_name in possible_run_names:
        if key == possible_run_name:
          metric_run = possible_run_name
          metric_tag = metric_name[len(metric_run) + 1:]
          break

    if metric_run and metric_tag:
      return True
    return False

  metrics_by_step = {}
  for subdir in io_wrapper.GetLogdirSubdirectories(checkpoints_dir):
    rpath = os.path.relpath(subdir, checkpoints_dir)
    # Skip runs that do not match user-specified metric.
    if ((not metric_run and
         not _try_fill_metric_run_and_tag_names(metric_name, (rpath,))) or
        metric_run != rpath):
      logging.info('Skipping events in %s', subdir)
      continue

    logging.info('Looking for events in %s', subdir)
    loader = directory_watcher.DirectoryWatcher(
        subdir, event_file_loader.EventFileLoader,
        io_wrapper.IsTensorFlowEventsFile)
    for event in loader.Load():
      # Skip metric collection of events for unavailable checkpoints or for
      # unmonitored tags.
      if (event.step not in steps or not event.summary.value or
          event.summary.value[0].tag != metric_tag):
        continue
      metric_value = tf.make_ndarray(event.summary.value[0].tensor)
      metrics_by_step[event.step] = metric_value

  return metrics_by_step


# TODO(b/216649487): Replace with BestCheckpointManager.
class SaveBestCheckpointer(Checkpointer):
  """A Checkpointer class that keeps checkpoints based on 'best' metrics.

  This extends the standard Checkpointer to garbage collect checkpoints based on
  metric values, instead of step recency. It uses TensorBoard summary files to
  determine best values for a given user configured metric name. Events are read
  and parsed using TensorBoard's event_processing packages.

  The metric name must be of the form `{run_name}/{tag_name}`. For example,
  'train/accuracy' or 'inference_eval/glue_cola_v002/eval/accuracy'.

  A few important features of this checkpointer:

  - Fallback behavior. It is not possible to verify whether metric names are
    valid during initialization, since some metrics may get written out after
    some time (e.g., during an evaluation). As such, when user provided metric
    names are not found, this checkpointer can be configured for two fall back
    strategies: (1) if `keep_checkpoints_without_metrics` is False, we use to
    the "most recent checkpoint" strategy from the standard checkpointer, (2)
    if `keep_checkpoints_without_metrics` is True, we keep all checkpoints until
    metrics become available (potentially indefinitely if summary files have
    been deleted or corrupted).

  - The number of checkpoints to keep is always increased by 1. Since its
    crucial to always keep the latest checkpoint (for recovery purposes) we
    always store the latest checkpoint plus `keep` number of best checkpoints.

  - It is assumed that TensorBoard summaries (event) files share a common root
    directory with `checkpoint_dir`, which is the directory passed to the
    the logdir crawler that searches for event files.

  Attributes:
    checkpoints_dir: a path to a directory to save checkpoints in and restore
      them from.
    keep: an optional maximum number of checkpoints to keep. If more than this
      number of checkpoints exist after a save, the oldest ones will be
      automatically deleted to save space.
    restore_dtype: optional dtype to cast targets to after restoring.
    save_dtype: dtype to cast targets to before saving.
    metric_name_to_monitor: Name of metric to monitor. Must be in the format
      {run_name}/{tag_name} (e.g., 'train/accuracy',
      'inference_eval/glue_cola_v002/eval/accuracy').
    metric_mode: Mode to use to compare metric values. One of 'max' or 'min'.
    keep_checkpoints_without_metrics: Whether to always keep (or delete)
      checkpoints for which a metric value has not been found.
    force_keep_period: When removing checkpoints, skip those who step is
      divisible by force_keep_period (step % force_keep_period == 0).
    keep_dataset_checkpoints: an optional maximum number of data iterators to
      keep. If more than this number of data iterators exist after a save, the
      oldest ones will be automatically deleted to save space.
  """

  def __init__(self,
               train_state: train_state_lib.TrainState,
               partitioner: partitioning.BasePartitioner,
               checkpoints_dir: str,
               dataset_iterator: Optional[tf.data.Iterator] = None,
               *,
               keep: Optional[int] = None,
               save_dtype: jnp.dtype = np.float32,
               restore_dtype: Optional[jnp.dtype] = None,
               metric_name_to_monitor: str = 'train/accuracy',
               metric_mode: str = 'max',
               keep_checkpoints_without_metrics: bool = True,
               force_keep_period: Optional[int] = None,
               keep_dataset_checkpoints: Optional[int] = None):
    super().__init__(
        train_state,
        partitioner,
        checkpoints_dir,
        dataset_iterator,
        keep=keep,
        save_dtype=save_dtype,
        restore_dtype=restore_dtype,
        keep_dataset_checkpoints=keep_dataset_checkpoints)
    if metric_mode not in ('max', 'min'):
      raise ValueError('Unsupported `metric_mode`: %s' % metric_mode)

    self._metric_name_to_monitor = metric_name_to_monitor
    self._metric_mode = metric_mode
    self._keep_checkpoints_without_metrics = keep_checkpoints_without_metrics
    self._force_keep_period = force_keep_period
    logging.info('Using SaveBestCheckpointer to keep %s best (%s) metric %s',
                 keep, metric_mode, metric_name_to_monitor)

  def _filter_out_force_keep_period_steps(self, existing_steps):
    """Filter out steps that are divisible by keep_period excluding the last."""
    if not existing_steps:
      return existing_steps

    # Don't filter out the last step.
    last_step = existing_steps.pop()  # pytype: disable=attribute-error  # dynamic-method-lookup
    existing_steps = [
        s for s in existing_steps if s % self._force_keep_period != 0
    ]
    return existing_steps + [last_step]

  def _remove_old_checkpoints(self):
    """Deletes checkpoints if there are more than keep_checkpoints."""
    if not self.keep:
      return

    existing_steps = self.all_steps()
    if self._force_keep_period:
      # Ignore checkpoints whose step is divisible by the keep period.
      existing_steps = self._filter_out_force_keep_period_steps(existing_steps)

    # Artificially add 1 to `keep` since we always keep the latest checkpoint.
    if len(existing_steps) <= self.keep + 1:
      return

    # Synchronous fetch of new events for existing_steps.
    metrics_by_step = populate_metrics_for_steps(self.checkpoints_dir,
                                                 self._metric_name_to_monitor,
                                                 existing_steps)
    logging.info('SaveBestcheckpointer: collected metrics %s', metrics_by_step)

    # Re-sort existing_steps by metric values while always keeping the latest
    # checkpoint.
    latest_checkpoint = existing_steps[-1]
    existing_steps = existing_steps[:-1]

    if self._keep_checkpoints_without_metrics:
      existing_steps = list(
          filter(lambda s: s in metrics_by_step, existing_steps))

    to_remove = len(existing_steps) - self.keep
    if to_remove <= 0:
      return

    # For any remaining steps without metrics, we assign a low/high value which
    # will make them candidate for removal. If no metrics are found this sorting
    # should preserve current order (oldest first).
    not_found_value = float('-inf' if self._metric_mode == 'max' else 'inf')
    existing_steps = sorted(
        existing_steps,
        key=lambda step: metrics_by_step.get(step, not_found_value),
        reverse=(self._metric_mode != 'max'))
    existing_steps.append(latest_checkpoint)

    for step in existing_steps[:to_remove]:
      checkpoint_utils.remove_checkpoint_dir(self._get_checkpoint_dir(step))


def _no_optimizer_state(ckpt_contents: PyTree, use_orbax_format: bool) -> bool:
  if use_orbax_format:
    return True
  try:
    version = ckpt_contents.get('version', 0)
    return version == 0
  except Exception as e:
    raise ValueError('Failed to get version') from e


def _should_apply_transform_fns(
    ckpt_contents: PyTree, use_orbax_format: bool
) -> bool:
  if use_orbax_format:
    return True
  try:
    version = ckpt_contents.get('version', 0)
    return version >= 2
  except Exception as e:
    raise ValueError('Failed to get version') from e


def _get_optimizer_state_dict(
    ckpt_contents: PyTree,
    optimizer_state: Mapping[str, Any],
    state_transformation_fns: Sequence[RestoreStateTransformationFn],
    use_orbax_format: bool = False,
):
  """Extracts optimizer state dict contents and applies assignment map."""
  if _no_optimizer_state(ckpt_contents, use_orbax_format):
    # This is a standard Flax checkpoint and may require remapping below.
    ckpt_optimizer_state = ckpt_contents
  else:
    ckpt_optimizer_state = ckpt_contents['optimizer']

  if _should_apply_transform_fns(ckpt_contents, use_orbax_format):
    for fn in state_transformation_fns:
      ckpt_optimizer_state = fn(ckpt_optimizer_state, optimizer_state)
    return ckpt_optimizer_state
  else:
    version = ckpt_contents.get('version', 0)  # pylint: disable=unreachable
    raise ValueError('Checkpoint versions earlier than 2 are not supported. '  # pylint: disable=unreachable
                     f'Got version: {version}')


def _transform_state_and_infos(
    state_dict: PyTree,
    parameter_infos: PyTree,
    state_transformation_fns: Sequence[SaveStateTransformationFn],
) -> Tuple[PyTree, PyTree]:
  """Applies transformations to the state dict and parameter infos PyTrees."""
  for fn in state_transformation_fns:
    state_dict, parameter_infos = fn(state_dict, parameter_infos)
  return state_dict, parameter_infos


async def _read_ts(
    param_info: _ParameterInfo,
    maybe_tspec: Any,
    ckpt_path: str,
    restore_dtype: Optional[jnp.dtype] = None,
    mesh: Optional[jax.sharding.Mesh] = None,
    axes: Optional[PartitionSpec] = None,
    params_on_devices: bool = True,
):
  """Read from a tensorstore.

  If both `mesh` and `axes` are provided, the method will attempt to restore the
  array as a jax.Array.

  Note:
    We use param_infos as the first argument because this function is only used
    in `jax.tree_util.tree_map` calls. In a tree multimap if the leaf of the
    first tree is `None` then is is ignored, even if the second tree has a
    subtree at that point. This means that when we are using something like a
    MultiOptimizer we can set the parameter info for a variable to `None` and
    we can skip processing it, even if the checkpoint has a subtree with things
    like optimizer state variables in it.

  Args:
    param_info: Information about how to read the parameter, host based sliced
      reads and the like.
    maybe_tspec: The tensorstore spec to read the parameter or some other
      object. If this is an array then we will do a host based sliced read on it
      (provided the param_info says to). Anything else we just return.
    ckpt_path: A base location to use when resolving the relative paths in the
      tensorstore spec.
    restore_dtype: type to restore as. None indicates that no cast is requested.
    mesh: jax.sharding.Mesh object for GDA restoration.
    axes: jax.sharding.MeshAxes object for GDA restoration.
    params_on_devices: Whether parameters should be allowed to be deserialized
      to devices.

  Returns:
    The array. Depending on the value `maybe_tspec` it might be read from
    tensorstore, or it might be returned as is. Depending on the values in
    param_info (specifically the `local_chunk_info`) it might be the full value
    or a specific slice.
  """
  # If saved as a numpy array, but a partitioned read is requested, return a
  # slice of the array for that host. Otherwise, return the whole thing.
  if isinstance(maybe_tspec, np.ndarray) and param_info:
    return maybe_tspec
  # If we have anything else that isn't a tensorstore spec just return it.
  elif not isinstance(maybe_tspec, ts.Spec):
    return maybe_tspec

  tmp_ts_spec_dict = maybe_tspec.to_json()
  # Remove non-required params so that we can open Tensorstore
  # that was created with a different set of params.
  del tmp_ts_spec_dict['metadata']['chunks']
  del tmp_ts_spec_dict['metadata']['compressor']

  # Convert the relative path in the spec to a path based on the checkpoint
  # location. Path and gcs bucket (if applicable) information is updated
  # in-place.
  _update_ts_path_from_relative_to_absolute(
      os.path.dirname(ckpt_path), tmp_ts_spec_dict)

  if param_info.shape is not None:
    ts_spec_arr_shape = tuple(tmp_ts_spec_dict['metadata']['shape'])
    # Check that the shapes of the array on disk match the expected shape based
    # on the optimizer that is being restored.
    if ts_spec_arr_shape != param_info.shape:
      raise ValueError(f'Shape of `{param_info.name}` in checkpoint '
                       f'{ts_spec_arr_shape} does not match expected '
                       f'{param_info.shape}.')

  if ('dtype' in tmp_ts_spec_dict and tmp_ts_spec_dict['dtype']
      == 'uint16') or ('dtype' in tmp_ts_spec_dict['metadata'] and
                       tmp_ts_spec_dict['metadata']['dtype'] == '<u2'):
    error_message = (
        'Found unsupported uint16 type in Tensorstore spec: '
        f'{tmp_ts_spec_dict}. Please update saved types to bfloat16.')
    raise ValueError(error_message)

  if restore_dtype is not None:
    tmp_ts_spec_dict = {
        'base': tmp_ts_spec_dict,
        'driver': 'cast',
        'dtype': jnp.dtype(restore_dtype).name
    }

  if params_on_devices:
    arr = await array_serialization.async_deserialize(
        jax.sharding.NamedSharding(mesh, axes),
        tmp_ts_spec_dict,
    )
  else:
    t = await ts.open(tmp_ts_spec_dict, open=True)
    arr = await t.read()
  return arr


def fake_param_info(maybe_tspec: Any) -> Optional[_ParameterInfo]:
  """Create _ParameterInfo that results in a full read."""
  # tspec is only None for `param_states` where the associated variable
  # is not updated by any optimizers. By setting the parameter info for
  # this to None, we can later short circut processing these subtrees
  # during loading.
  if maybe_tspec is None:
    return None
  local_chunk_info = None
  tspec = None
  if isinstance(maybe_tspec, ts.Spec):
    tspec = maybe_tspec
    local_chunk_info = partitioning.LocalChunkInfo(
        slice=(slice(None, None),), replica_id=0)
  return _ParameterInfo(
      name='',  # We don't ever use the name.
      shape=tuple(tspec.to_json()['metadata']['shape']) if tspec else None,
      # We just believe the spec in the file.
      ts_spec=tspec,
      local_chunk_info=local_chunk_info,
      axes=None)


def find_checkpoint(
    path: epath.PathLike, step: Optional[int] = None
) -> epath.PathLike:
  """Find the checkpoint file based on paths and steps.

  Args:
    path: The location of the checkpoint. Can point to the `model_dir`, the
      checkpoint dir with a step, or the actual checkpoint file.
    step: The step to load. Only used if you are pointing to the `model_dir`

  Raises:
    ValueError if the checkpoint file can't be found.

  Returns:
    The path to the checkpoint file.
  """
  # If you aren't pointing at the msgpack checkpoint file
  if gfile.isdir(path):
    # If you didn't specify a step
    if step is None:
      # Try to get the most recent step.
      step = latest_step(path)
      # If you found a step then you were pointing at model_dir, set the path to
      # the msgpack file in the checkpoint dir.
      if step:
        path = get_checkpoint_dir(path, step)
    # You gave a step, use it.
    else:
      path = get_checkpoint_dir(path, step)
    # Whether you supplied a step, found a step, or were already pointing at the
    # step, you are not pointing at a step directory, so now point to the
    # msgpack file.
    path = os.path.join(path, 'checkpoint')
  # You weren't point to a dir so you were pointing at the msgpack file.
  # Check that we found a checkpoint file.
  if not gfile.exists(path) or gfile.isdir(path):
    raise ValueError(f'Path is not a valid checkpoint: {path}')
  return path


def load_t5x_checkpoint(
    path: str,
    step: Optional[int] = None,
    state_transformation_fns: Sequence[RestoreStateTransformationFn] = (),
    remap: bool = True,
    restore_dtype: Optional[jnp.dtype] = None,
    lazy_parameters: bool = False,
) -> PyTree:
  """Load a T5X checkpoint without pre-defining the optimizer.

  Note:
    This only works for T5X checkpoints, not TF checkpoints.

  Args:
    path: The location of the checkpoint.
    step: The checkpoint from which step should be loaded.
    state_transformation_fns: Transformations to apply, in order, to the state
      after reading.
    remap: Whether to rename the checkpoint variables to the newest version.
    restore_dtype: optional dtype to cast targets to after restoring. If None,
      no parameter casting is performed.
    lazy_parameters: whether to load the parameters as LazyArrays to preserve
      memory.

  Returns:
    A nested dictionary of weights and parameter states from the checkpoint.
  """
  start_time = time.time()
  path = find_checkpoint(path, step)
  logging.info('Restoring from checkpoint: %s', path)

  # The msgpack file will have all the info we need about the parameter layout.
  with gfile.GFile(path, 'rb') as fp:
    ckpt_contents = serialization.msgpack_restore(fp.read())

  # If reading a ckpt that was written with gfile driver but the current
  # session uses the gcs driver, convert the ckpt's driver to gcs.
  if os.fspath(path).startswith('gs://'):
    ckpt_contents = _maybe_update_ts_from_file_to_gcs(ckpt_contents)
  # If a ckpt was saved in gcs and is being loaded locally, then convert the
  # driver to file or gfile. If the ckpt was not saved in gcs, do not change.
  else:
    ckpt_contents = _maybe_update_ts_from_gcs_to_file(ckpt_contents)

  # Remap that variable names to the most recent formatting.
  if remap:
    ckpt_optimizer_state = _get_optimizer_state_dict(ckpt_contents, {},
                                                     state_transformation_fns)
  # If we aren't remapping names we at least need to index into the checkpoint
  # file blob to make sure we are only dealing with the optimizer state.
  else:
    # Grab a subsection of the file depending on the version.
    version = ckpt_contents.get('version', 0)
    if version == 0:
      ckpt_optimizer_state = ckpt_contents
    else:
      ckpt_optimizer_state = ckpt_contents['optimizer']

  # Replace all dicts of tensorstore specs with actual `ts.Spec`s.
  # When a checkpoint was trained using a MultiOptimizer, some of the parameter
  # states may be set to `None` (when a parameter was untouched by any
  # optimizer). We still needs references to these in our state so we keep
  # empty nodes.
  ckpt_optimizer_state_with_specs = (
      state_utils.flatten_state_dict(
          ckpt_optimizer_state, keep_empty_nodes=True))
  ckpt_optimizer_state_with_specs = {
      k: ts.Spec(v) if isinstance(v, dict) else v
      for k, v in ckpt_optimizer_state_with_specs.items()
  }

  # Create fake parameter info that results in reading the whole variable.
  param_infos = {
      k: fake_param_info(v) for k, v in ckpt_optimizer_state_with_specs.items()
  }

  ckpt_optimizer_state_with_specs = traverse_util.unflatten_dict(
      ckpt_optimizer_state_with_specs, sep='/')
  param_infos = traverse_util.unflatten_dict(param_infos, sep='/')

  def _create_lazy_awaitable_array(
      param_info: _ParameterInfo, maybe_ts_spec: Any, ckpt_path: str,
      restore_dtype: Optional[jnp.dtype]) -> LazyAwaitableArray:
    get_fn = functools.partial(
        _read_ts,
        param_info,
        maybe_ts_spec,
        ckpt_path=ckpt_path,
        restore_dtype=restore_dtype,
        params_on_devices=False,
    )
    return LazyAwaitableArray.from_tensor_store_spec_or_array(
        maybe_ts_spec, get_fn, dtype=restore_dtype)

  state_dict = jax.tree_util.tree_map(
      functools.partial(
          _create_lazy_awaitable_array,
          ckpt_path=path,
          restore_dtype=restore_dtype), param_infos,
      ckpt_optimizer_state_with_specs)

  if not lazy_parameters:
    future_state_dict = jax.tree_util.tree_map(lambda x: x.get_async(),
                                               state_dict)
    state_dict = _run_future_tree(future_state_dict)

  if restore_dtype is not None:
    state_dict['target'] = _cast(state_dict['target'], restore_dtype)

  end_time = time.time()
  monitoring.record_event_duration_secs(_READ_CHECKPOINT_EVENT,
                                        end_time - start_time)
  return state_dict


_OPTIMIZER_KEY = 'optimizer'
_VERSION_KEY = 'version'
_CHECKPOINTS_SUBDIR = 'checkpoints'
_STATE_KEY = 'state'
_DATASET_KEY = 'dataset'
_METRICS_KEY = 'metrics'
_FLAX_CHECKPOINT_FILE = 'checkpoint'


@dataclasses.dataclass
class _OrbaxParamInfo:
  name: str
  mesh_axes: partitioning.PartitionSpec


class DatasetCheckpointHandler(ocp.CheckpointHandler):
  """A CheckpointHandler implementation that handles tf.data.Iterator."""

  def __init__(self, checkpoint_filename: str):
    self._checkpoint_filename = checkpoint_filename

  def save(self, directory: epath.Path, item: tf.data.Iterator):
    """Saves the given item.

    Args:
      directory: save location directory.
      item: a tf.data.Iterator to be saved.
    """
    if jax.process_count() > 1:
      directory /= f'process_{jax.process_index()}-of-{jax.process_count()}'
      directory.mkdir(parents=False, exist_ok=False)
    ckpt = tf.train.Checkpoint(ds=item)
    ckpt.write(os.fspath(directory / self._checkpoint_filename))
    multihost_utils.sync_global_devices('DatasetCheckpointHandler:save')

  def restore(self,
              directory: epath.Path,
              item: Optional[tf.data.Iterator] = None) -> tf.data.Iterator:
    """Restores the given item.

    Args:
      directory: restore location directory.
      item: a tf.data.Iterator to be restored. Not Optional

    Returns:
      a tf.data.Iterator restored from `directory`.
    """
    if item is None:
      raise ValueError('Must provide item to restore')
    if jax.process_count() > 1:
      directory /= f'process_{jax.process_index()}-of-{jax.process_count()}'
    ckpt = tf.train.Checkpoint(ds=item)
    ckpt.read(os.fspath(directory /
                        self._checkpoint_filename)).assert_consumed()
    return item

  def structure(self, directory: epath.Path) -> Any:
    """Unimplemented. See parent class."""
    pass


def _step_from_train_state(train_state: train_state_lib.TrainState) -> int:
  step = train_state.step
  step = step.get() if isinstance(step, LazyArray) else step
  step = get_local_data(step)
  # Integer, to avoid side effects in the checkpoint path.
  return int(step)


def _construct_save_args(
    param_info: _OrbaxParamInfo, dtype: jnp.dtype
) -> ocp.SaveArgs:
  """Create SaveArgs for Orbax saving."""
  if param_info.name.split('.')[0] != 'target':
    dtype = None
  return ocp.SaveArgs(aggregate=param_info.mesh_axes is None, dtype=dtype)


def _construct_restore_args(
    param_info: _OrbaxParamInfo,
    dtype: jnp.dtype,
    mesh: jax.sharding.Mesh,
) -> ocp.RestoreArgs:
  """Create RestoreArgs for Orbax restoration."""
  if not isinstance(param_info, _OrbaxParamInfo):  # from fallback
    return ocp.RestoreArgs(dtype=dtype)
  if param_info.name.split('.')[0] != 'target':
    dtype = None
  if param_info.mesh_axes is None:
    return ocp.RestoreArgs(dtype=dtype)
  return ocp.ArrayRestoreArgs(
      mesh=mesh,
      mesh_axes=param_info.mesh_axes,
      dtype=dtype,
  )


def _construct_orbax_param_infos(
    train_state: train_state_lib.TrainState,
    partitioner: partitioning.BasePartitioner,
) -> PyTree:
  """Construct _OrbaxParamInfo tree for TrainState parameters."""
  param_names = traverse_util.unflatten_dict(
      {
          k: '/'.join(k)
          for k in traverse_util.flatten_dict(
              train_state.state_dict(), keep_empty_nodes=True
          )
      }
  )
  mesh_axes = partitioner.get_mesh_axes(train_state).state_dict()
  return jax.tree_util.tree_map(_OrbaxParamInfo, param_names, mesh_axes)


def _construct_orbax_restoration_transforms(
    manager: ocp.CheckpointManager,
    step: int,
    directory: epath.Path,
    state_dict: PyTree,
    state_transformation_fns: Sequence[RestoreStateTransformationFn],
    restore_args: PyTree,
) -> Tuple[PyTree, Any, PyTree]:
  """Construct transformations and restoration arguments for Orbax classes."""
  # After transforms, may be a subset of keys: only the ones we actually need
  # to restore.
  state_subdir = manager._get_save_directory(  # pylint: disable=protected-access
      step, directory, key_name=_STATE_KEY
  )
  assert state_subdir.is_dir()
  use_orbax_format = state_subdir.stem == _STATE_KEY  # Standard Orbax format
  checkpointer = typing.cast(
      ocp.Checkpointer, manager._checkpointers[_STATE_KEY]  # pylint: disable=protected-access
  )
  handler = typing.cast(
      ocp.PyTreeCheckpointHandler, checkpointer._handler  # pylint: disable=protected-access
  )
  structure = handler._read_aggregate_file(  # pylint: disable=protected-access
      state_subdir
  )
  # Note: Ideally we would use Orbax's `transform_fn` to do this logic, but
  # the problem is we need to modify `restore_args`, and there isn't a great
  # way to do that within Orbax.
  state_dict_to_restore = _get_optimizer_state_dict(
      structure,
      state_dict,
      state_transformation_fns,
      use_orbax_format=use_orbax_format,
  )
  # After transformations, state_dict_to_restore may still have extra keys
  # relative to item (the eventual restoration structure). Extraneous keys
  # need to be dropped.
  state_dict_to_restore = state_utils.intersect_state(
      state_dict_to_restore, state_dict
  )
  restore_args = state_utils.intersect_state(
      restore_args, state_dict_to_restore
  )

  def _transform_fn(
      item_: PyTree, structure_: PyTree, param_infos_: PyTree
  ) -> Tuple[PyTree, PyTree]:
    # When this function is called from within PyTreeCheckpointHandler,
    # transforms will already have been performed (see above), but use this
    # function to hack param_infos to return the needed values.
    # This structure is unneeded, because we already restored and transformed
    # it.
    del structure_, param_infos_

    def _make_orbax_internal_metadata(value: Any, args: ocp.RestoreArgs):
      if ocp.utils.leaf_is_placeholder(value):
        if isinstance(args, ocp.ArrayRestoreArgs):
          restore_type = 'jax.Array'
        else:
          restore_type = 'np.ndarray'
        return ocp.pytree_checkpoint_handler._InternalValueMetadata(  # pylint: disable=protected-access
            restore_type=restore_type
        )
      else:
        return ocp.pytree_checkpoint_handler._InternalValueMetadata(  # pylint: disable=protected-access
            restore_type=None,
            skip_deserialize=True,
            aggregate_value=value,
        )

    directory_ = manager._get_save_directory(  # pylint: disable=protected-access
        step, directory, key_name=_STATE_KEY
    )

    def _modify_orbax_param_info(info, value):
      if ocp.utils.leaf_is_placeholder(value):
        name = ocp.utils.name_from_leaf_placeholder(value)
        return dataclasses.replace(info, path=directory_ / name)
      return info

    item_ = jax.tree_util.tree_map(
        _make_orbax_internal_metadata, item_, restore_args
    )
    param_infos_, _ = ocp.pytree_checkpoint_handler._get_restore_parameters(  # pylint: disable=protected-access
        directory_,
        None,
        item_,
        None,
        None,
    )
    param_infos_ = jax.tree_util.tree_map(
        _modify_orbax_param_info, param_infos_, state_dict_to_restore
    )
    return item_, param_infos_

  return state_dict_to_restore, restore_args, _transform_fn


def _restore_from_tf_checkpoint(
    full_state_dict: PyTree,
    param_infos: PyTree,
    train_state: train_state_lib.TrainState,
    partitioner: partitioning.BasePartitioner,
    restore_dtype: jnp.dtype,
) -> train_state_lib.TrainState:
  """Restore from a TensorFlow-based T5 checkpoint."""
  full_state_dict = dict(full_state_dict)

  def _partition_parameter(maybe_arr: Any, param_info: _OrbaxParamInfo):
    if isinstance(maybe_arr, np.ndarray) and param_info:
      arr = maybe_arr
      to_sharded = partitioner.partition(
          lambda x: x,
          in_axis_resources=None,
          out_axis_resources=param_info.mesh_axes,
      )
      return to_sharded(arr)
    return maybe_arr

  if restore_dtype is not None:
    full_state_dict['target'] = _cast(full_state_dict['target'], restore_dtype)
  state_dict = jax.tree_util.tree_map(
      _partition_parameter,
      full_state_dict,
      param_infos,
  )

  return train_state.restore_state(state_dict)


class OrbaxCheckpointManagerInterface:
  """Wrapper for ocp.CheckpointManager."""

  class _CheckpointManagerImpl(ocp.CheckpointManager):
    """CheckpointManager implementation to deal with metrics update."""

    def _get_old_steps_to_remove(self):
      """Update metrics for Orbax management, if available."""
      if self._track_best:
        metric_name_to_monitor = self._options.metric_name_to_monitor  # pytype: disable=attribute-error
        step_to_metric = populate_metrics_for_steps(
            os.fspath(self.directory),
            metric_name_to_monitor,
            self.all_steps(),
        )
        for info in self._checkpoints:
          if info.step in step_to_metric:
            metrics = {metric_name_to_monitor: step_to_metric[info.step]}
            info.metrics = metrics
      super()._get_old_steps_to_remove()

  def __init__(
      self,
      directory: str,
      train_state: train_state_lib.TrainState,
      partitioner: partitioning.BasePartitioner,
      dataset_iterator: Optional[tf.data.Iterator] = None,
      save_dtype: Optional[jnp.dtype] = None,
      restore_dtype: Optional[jnp.dtype] = None,
      keep: Optional[int] = None,
      period: Optional[int] = 1,
      checkpoint_steps: Optional[Sequence[int]] = None,
      keep_dataset_checkpoints: Optional[int] = None,
      force_keep_period: Optional[int] = None,
      metric_name_to_monitor: Optional[str] = None,
      metric_mode: str = 'max',
      keep_checkpoints_without_metrics: bool = True,
  ):
    """Performs Orbax setup given standard arguments from T5X."""
    del checkpoint_steps
    del keep_dataset_checkpoints
    self._train_state = train_state
    self._partitioner = partitioner
    self._dataset_iterator = dataset_iterator
    self._save_dtype = save_dtype
    self._restore_dtype = restore_dtype
    self._tmp_directory: Optional[epath.PathLike] = None

    data_layout = partitioner.get_data_layout()
    dataset_ckpt_name = (
        f'{_TRAIN_DS_PREFIX}-'
        f'{data_layout.shard_id:03}-of-{data_layout.num_shards:03}'
    )
    self._should_write_dataset_ckpt = (
        self._dataset_iterator and data_layout.is_first_host_in_replica_set
    )

    checkpointers = {
        _STATE_KEY: ocp.Checkpointer(
            # TODO(b/273803615) Enable OCDBT.
            ocp.PyTreeCheckpointHandler(use_ocdbt=False)
        ),
    }
    if self._should_write_dataset_ckpt:
      checkpointers[_DATASET_KEY] = ocp.Checkpointer(
          DatasetCheckpointHandler(checkpoint_filename=dataset_ckpt_name)
      )

    def best_fn(metrics):
      return metrics[metric_name_to_monitor]

    options = ocp.CheckpointManagerOptions(
        max_to_keep=keep,
        save_interval_steps=period,
        keep_period=force_keep_period,
        best_fn=best_fn if metric_name_to_monitor is not None else None,
        best_mode=metric_mode,
        keep_checkpoints_without_metrics=keep_checkpoints_without_metrics,
        cleanup_tmp_directories=True,
        step_prefix='checkpoint',
    )
    options.metric_name_to_monitor = metric_name_to_monitor

    if not gfile.isdir(directory):
      directory = os.path.dirname(directory)
    self._manager = self._CheckpointManagerImpl(
        directory=directory, checkpointers=checkpointers, options=options
    )

  @property
  def directory(self) -> epath.Path:
    return self._manager.directory

  def all_steps(self) -> Sequence[int]:
    return self._manager.all_steps()

  def latest_step(self) -> Optional[int]:
    return self._manager.latest_step()

  def should_save(self, step: int) -> bool:
    return self._manager.should_save(step)

  def save(
      self,
      train_state: train_state_lib.TrainState,
      state_transformation_fns: Sequence[SaveStateTransformationFn] = (),
      force: bool = True,
  ) -> bool:
    """Saves a checkpoint for the given train state.

    Args:
      train_state: the train state to save. May contain a combination of
        LazyArray objects and arrays (e.g., np.ndarray, jax.DeviceArray)
      state_transformation_fns: Transformations to apply, in order, to the state
        before writing.
      force: Saves regardless of whether should_save is False. True by default
        because should_save logic is handled externally to this class in T5X.
        This is because of a feature that decouples actual step and step offset.

    Returns:
      Whether the save was performed or not.
    """
    start_time = time.time()
    step = _step_from_train_state(train_state)
    if not force and not self._manager.should_save(step):
      return False

    # TODO(b/216649487) Test save-time state_transformation_fns.
    state_dict, param_infos = _transform_state_and_infos(
        train_state.state_dict(),
        _construct_orbax_param_infos(self._train_state, self._partitioner),
        state_transformation_fns,
    )

    # Arguments for saving interpretable by Orbax.
    save_args = jax.tree_util.tree_map(
        functools.partial(_construct_save_args, dtype=self._save_dtype),
        param_infos,
    )
    # If the params are to be aggregated, then get locally addressable data.
    state_dict = jax.tree_util.tree_map(
        lambda v, arg: get_local_data(v) if arg.aggregate else v,
        state_dict,
        save_args,
    )

    # Separate savable items.
    items = {_STATE_KEY: state_dict}
    if self._should_write_dataset_ckpt:
      items[_DATASET_KEY] = self._dataset_iterator
    save_kwargs = {
        _STATE_KEY: {
            'save_args': save_args,
        },
    }
    saved = self._manager.save(
        step, items, save_kwargs=save_kwargs, force=force
    )

    # Record JAX monitoring events.
    end_time = time.time()
    monitoring.record_event_duration_secs(
        _WRITE_CHECKPOINT_EVENT, end_time - start_time
    )
    ocp.utils.record_saved_duration(start_time)

    return saved

  def restore(
      self,
      step: Optional[int] = None,
      path: Optional[str] = None,
      fallback_state: Optional[Mapping[str, Any]] = None,
      state_transformation_fns: Sequence[RestoreStateTransformationFn] = (),
      lazy_parameters: Optional[bool] = False,
  ) -> train_state_lib.TrainState:
    """Restores a TrainState from the given step or path.

    Note: can only provide one of `step` or `path`.

    Args:
      step: the step number to restore from.
      path: the full path to restore from.
      fallback_state: a state dict of an optimizer to fall back to for loading
        params that do not exist in the checkpoint (after applying all
        `state_transformation_fns`), but do exist in `Checkpointer.optimizer`.
        The union of `fallback_state` and state loaded from the checkpoint must
        match `Checkpointer.optimizer`.
      state_transformation_fns: Transformations to apply, in order, to the state
        after reading.
      lazy_parameters: whether to load the parameters as LazyArrays to preserve
        memory.

    Returns:
      The restored train state.
    """
    if lazy_parameters:
      logging.warning('Orbax does not support lazy restoration.')
    start_time = time.time()
    if step is not None and path is not None:
      raise ValueError('Can only provide `step` or `path` but not both.')
    directory = self.directory
    if path is not None:
      directory, step = get_step_from_checkpoint_dir(os.fspath(path))

    # Check for legacy T5X checkpoint: If so, use legacy T5X
    # checkpointer to restore the state. The following exclusive features of T5X
    # checkpoint are skipped: DatasetIterator, [add more here when discovered]
    try:
      ckpt_path = find_checkpoint(directory, step)
    except ValueError:
      # `find_checkpoint` fails if the `.checkpoint` file isn't directly in
      # the checkpoint directory. In this case, leave path as None and skip
      # the legacy T5X checkpoint check.
      ckpt_path = None

    if ckpt_path is not None:
      ckpt_type = checkpoint_utils.detect_checkpoint_type(
          ckpt_path, expected=checkpoint_utils.CheckpointTypes.ORBAX
      )
      if ckpt_type is checkpoint_utils.CheckpointTypes.T5X_TF:
        raise ValueError(
            'Attempting to restore a TensorFlow checkpoint as a native T5X '
            'checkpoint. Use `restore_from_tf_checkpoint` instead. Path: '
            + ckpt_path
        )
      elif ckpt_type is checkpoint_utils.CheckpointTypes.T5X:
        legacy_checkpointer = Checkpointer(
            self._train_state,
            self._partitioner,
            self.directory,
            restore_dtype=self._restore_dtype,
        )
        return legacy_checkpointer.restore(path=path)

    state_dict = self._train_state.state_dict()
    # Returns a state dict rather than a train state.
    param_infos = _construct_orbax_param_infos(
        self._train_state, self._partitioner
    )
    # Construct restoration arguments interpretable by Orbax.
    restore_args = jax.tree_util.tree_map(
        functools.partial(
            _construct_restore_args,
            dtype=self._restore_dtype,
            mesh=self._partitioner.mesh,
        ),
        param_infos,
    )
    # Handle T5X transformation functions, since they are specified differently
    # than native Orbax transformation functions.
    state_dict_to_restore, restore_args, transform_fn = (
        _construct_orbax_restoration_transforms(
            self._manager,
            step,
            directory,
            state_dict,
            state_transformation_fns,
            restore_args,
        )
    )

    # Construct separate items to restore.
    items = {_STATE_KEY: state_dict_to_restore}
    if self._should_write_dataset_ckpt:
      items[_DATASET_KEY] = self._dataset_iterator
    restore_kwargs = {
        _STATE_KEY: {
            'restore_args': restore_args,
            'legacy_transform_fn': transform_fn,
        },
    }
    restored = self._manager.restore(
        step, items, restore_kwargs=restore_kwargs, directory=directory
    )
    state_dict = restored[_STATE_KEY]
    if self._should_write_dataset_ckpt:
      self._dataset_iterator = restored[_DATASET_KEY]

    # Merge restored state dict with fallback state to fill in any remaining
    # params.
    if fallback_state is not None:
      state_dict = state_utils.merge_state(state_dict, fallback_state)

    # After restoration, some values may still be non-sharded arrays from
    # fallback state.
    def _maybe_make_sharded_array_helper(arr, info):
      return _maybe_make_sharded_array(
          arr,
          self._partitioner.mesh,
          axes=info.mesh_axes,
          restore_dtype=self._restore_dtype,
      )

    state_dict = jax.tree_util.tree_map(
        _maybe_make_sharded_array_helper, state_dict, param_infos
    )

    train_state = self._train_state.restore_state(state_dict)

    end_time = time.time()
    monitoring.record_event_duration_secs(
        _READ_CHECKPOINT_EVENT, end_time - start_time
    )

    return train_state

  def restore_from_tf_checkpoint(
      self,
      path_or_dir: str,
      strict: bool = True,
      translator: Optional[checkpoint_importer.CheckpointTranslator] = None,
  ) -> train_state_lib.TrainState:
    """Restore from a TensorFlow-based T5 checkpoint."""
    full_state_dict = checkpoint_importer.restore_from_t5_checkpoint(
        self._train_state.state_dict(),
        path_or_dir,
        strict=strict,
        translator=translator,
    )
    return _restore_from_tf_checkpoint(
        full_state_dict,
        _construct_orbax_param_infos(self._train_state, self._partitioner),
        self._train_state,
        self._partitioner,
        self._restore_dtype,
    )


class CheckpointManagerConstructor(typing_extensions.Protocol):
  """A function that returns a checkpoints.CheckpointManager.

  This type annotation allows users to partially bind args to the constructors
  of CheckpointManager subclasses without triggering type errors.
  """

  def __call__(
      self,
      directory: str,
      train_state: train_state_lib.TrainState,
      partitioner: partitioning.BasePartitioner,
      dataset_iterator: Optional[tf.data.Iterator] = None,
      save_dtype: Optional[jnp.dtype] = None,
      restore_dtype: Optional[jnp.dtype] = None,
      keep: Optional[int] = None,
      period: Optional[int] = None,
      force_keep_period: Optional[int] = None,
      checkpoint_steps: Optional[Sequence[int]] = None,
  ) -> OrbaxCheckpointManagerInterface:
    """CheckpointManager constructor."""
    pass
