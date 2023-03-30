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

"""General utility functions for t5x."""
import collections
import collections.abc
from concurrent.futures import thread
import contextlib
import dataclasses
import functools
import importlib
import inspect
import os
import re
import time
import typing
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple, Type, Union, cast
import warnings

from absl import flags
from absl import logging
import clu.data
from flax import traverse_util
import flax.core
from flax.core import scope as flax_scope
from flax.linen import partitioning as flax_partitioning
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
import seqio
from t5x import checkpoints
from t5x import optimizers
from t5x import partitioning
from t5x import state_utils
from t5x import train_state as train_state_lib
import tensorflow as tf
from tensorflow.io import gfile
import typing_extensions


FLAGS = flags.FLAGS

# Remove _ShardedDeviceArray when users of t5x have their types updated
_ShardedDeviceArray = Any
Array = Union[np.ndarray, jnp.ndarray, _ShardedDeviceArray, tf.Tensor]
PyTree = Any
PartitionSpec = partitioning.PartitionSpec
DType = Union[np.dtype, type(jnp.bfloat16)]
Shape = Tuple[int, ...]

# TODO(adarob): Remove namespace mapping after client gin files are updated.
TensorBoardLogger = seqio.TensorBoardLogger
get_local_data = checkpoints.get_local_data


class EvaluatorConstructor(typing_extensions.Protocol):
  """A function that returns an Evaluator.

  This protocol represents the actual callsite for the seqio.Evaluator c'tor
  in this file. It allows users to bind additional args with partial() and
  pass that partial into the fn without causing type check issues.
  """

  def __call__(
      self,
      mixture_or_task_name: str,
      feature_converter: seqio.FeatureConverter,
      eval_split: str,
      use_cached: bool,
      seed: Optional[int],
      sequence_length: Optional[Mapping[str, int]],
      log_dir: Optional[str],
      use_memory_cache: bool,
  ) -> seqio.Evaluator:
    """The call for the seqio.Evaluator c'tor in this file.

    Args:
      mixture_or_task_name: a registered task or mixture name.
      feature_converter: a feature converter object to use to convert the task
        features to model features. Must be a subclass of
        seqio.FeatureConverter.
      eval_split: evaluation split. Typically "validation" or "test".
      use_cached: whether to use the cached dataset instead of processing it on
        the fly.
      seed: random seed used for dataset shuffle and preprocessing. This is
        usually not needed since eval datasets aren't shuffled and shouldn't use
        stochastic operations. It is only useful for in certain data sources
        such as `FewshotDataSource` where the training examples are randomly
        selected during evaluation.
      sequence_length: an optional length specification. If specified, these
        will be the hard-limit on the evaluation data used for prediction. If
        none of the preprocessors depend on the sequence length, it can be left
        unspecified and the maximum length for each feature will be used. These
        lengths are computed while caching the datasets.
      log_dir: the directory to log outputs to. Required if `logger_cls` is
        non-empty.
      use_memory_cache: whether to use tf.data.Dataset#cache. May cause memory
        issues for large datasets.

    Returns:
      A seqio.Evaluator.
    """
    ...


# -----------------------------------------------------------------------------
# Configurations
# -----------------------------------------------------------------------------
@dataclasses.dataclass
class SaveCheckpointConfig:
  """Configuration for saving model checkpoints."""

  # The dtype to save ('float32' or 'bfloat16').
  dtype: str = 'float32'
  # Number of steps between writing checkpoints.
  period: Optional[int] = None
  # Number of most recent checkpoints to keep, or None to keep them all.
  keep: Optional[int] = None
  # Number of dataset checkpoints to keep, or None to keep them all.
  # Note: Dataset checkpoints are also affected by `keep`.
  keep_dataset_checkpoints: Optional[int] = None
  # Whether to save dataset checkpoints.
  save_dataset: bool = False
  # The checkpointer class to use.
  checkpointer_cls: checkpoints.CheckpointerConstructor = (
      checkpoints.Checkpointer
  )
  # Transformations to apply, in order, to the state before writing.
  state_transformation_fns: Sequence[checkpoints.SaveStateTransformationFn] = (
      dataclasses.field(default_factory=list)
  )
  # Enable GDA in this Checkpointer.
  use_gda: bool = True
  # CheckpointManager implementation to use.
  checkpoint_manager_cls: checkpoints.CheckpointManagerConstructor = (
      checkpoints.CheckpointManager
  )

  def __post_init__(self):
    if self.dtype not in (None, 'float32', 'bfloat16'):
      raise ValueError(
          "`SaveCheckpointConfig.dtype` must be one of None, 'float32' or "
          f"'bfloat16'. Got {self.dtype}."
      )


@dataclasses.dataclass
class RestoreCheckpointConfig:
  """Configuration for restoring model from checkpoint."""

  # Path(s) to checkpoint to restore from or directory (depending on `mode`).
  path: Union[str, Sequence[str]]
  # One of 'specific', 'latest', or 'all'.
  #   specific: load the checkpoint specified by `path`.
  #   latest: load most recent checkpoint in the directory specified by `path`.
  #   all: sequentially load all of checkpoints in the directory `path`.
  mode: str = 'latest'
  # An optional sequence of (pattern, replacement) regex pairs. The pattern
  # matches parameters in the model and the replacement matches the checkpoint
  # (after substitutions). The replacement may be None, in which case the
  # parameter can be dropped. Use `fallback_to_scratch` to fill them in with
  # newly initialized values.
  assignment_map: Optional[Sequence[Tuple[str, Optional[str]]]] = None
  # Whether to restore all optimizer parameters from the checkpoint.
  strict: bool = True
  # Whether to initialize parameters that are in the model being restored but
  # are missing from the checkpoint (after `assignment_map` is applied).
  fallback_to_scratch: bool = False
  # The dtype to restore ('float32' or 'bfloat16'), or None to load as saved.
  dtype: Optional[str] = None
  # Whether to restore the dataset checkpoint. Fails if checkpoint not present.
  restore_dataset: bool = False
  # The checkpointer class to use.
  checkpointer_cls: checkpoints.CheckpointerConstructor = (
      checkpoints.Checkpointer
  )
  # Transformations to apply, in order, to the state after reading. These will
  # be applied after the `assignment_map` transformations.
  state_transformation_fns: Sequence[
      checkpoints.RestoreStateTransformationFn
  ] = ()
  # Enable GDA in this Checkpointer.
  use_gda: bool = True
  # CheckpointManager implementation to use.
  checkpoint_manager_cls: checkpoints.CheckpointManagerConstructor = (
      checkpoints.CheckpointManager
  )

  def __post_init__(self):
    if self.mode not in ('specific', 'latest', 'all'):
      raise ValueError(
          "`RestoreCheckpointConfig.mode` must be one of 'specific', 'latest', "
          f"or 'all'. Got {self.mode}."
      )
    if self.dtype not in (None, 'float32', 'bfloat16', 'float16'):
      raise ValueError(
          "`RestoreCheckpointConfig.dtype` must be one of `None`, 'float32', "
          f"'float16' or 'bfloat16'. Got {self.dtype}."
      )
    if self.assignment_map is not None:
      # Turns `assignment_map` into a transformation function.
      assignment_map_fn = functools.partial(
          state_utils.apply_assignment_map, assignment_map=self.assignment_map
      )
      # Prepends the `assignment_map` transformation to the front of the list.
      self.state_transformation_fns = (
          assignment_map_fn,
          *self.state_transformation_fns,
      )


@dataclasses.dataclass
class CheckpointConfig:
  """Configuration for checkpointing of model and dataset."""

  save: Optional[SaveCheckpointConfig] = None
  restore: Optional[RestoreCheckpointConfig] = None


class LegacyCheckpointer(orbax.checkpoint.Checkpointer):
  """Implementation of Checkpointer interface for T5X.

  Relies on underlying save_checkpointer and restore_checkpointer, which are
  t5x.checkpoints.Checkpointer objects.
  """

  def __init__(
      self,
      *,
      save_checkpointer: Optional[checkpoints.Checkpointer] = None,
      restore_checkpointer: checkpoints.Checkpointer,
      strict: Optional[bool] = False,
  ):
    self._save_checkpointer = save_checkpointer
    self._restore_checkpointer = restore_checkpointer
    self._strict = strict

  def save(
      self,
      path: str,
      item: train_state_lib.TrainState,
      force: bool = False,
      state_transformation_fns: Sequence[
          checkpoints.SaveStateTransformationFn
      ] = (),
      *,
      concurrent_gb: int = 128,
  ):
    """Performs save operation using save_checkpointer.

    Args:
      path: path to save item to.
      item: a TrainState PyTree to save.
      force: unused.
      state_transformation_fns: Transformations to apply, in order, to the state
        before writing.
      concurrent_gb: the approximate number of gigabytes of partitionable
        parameters to process in parallel. Useful to preserve RAM.
    """
    train_state = item
    del path  # stored in save_checkpointer
    # dataset_iterator is also saved, but is provided in checkpointer init
    if self._save_checkpointer is None:
      raise ValueError(
          "`_save_checkpointer` is not set up. Can't save checkpoints."
      )
    self._save_checkpointer.save(
        train_state, state_transformation_fns, concurrent_gb=concurrent_gb
    )

  def restore(
      self,
      path: str,
      item: Optional[train_state_lib.TrainState] = None,
      state_transformation_fns: Sequence[
          checkpoints.RestoreStateTransformationFn
      ] = (),
      fallback_state: Optional[Mapping[str, Any]] = None,
      lazy_parameters: bool = False,
  ) -> train_state_lib.TrainState:
    """Performs restore operation using restore_checkpointer.

    Determines whether the indicated path is a Tensorflow checkpoint.

    Args:
      path: the string path to restore from.
      item: a TrainState PyTree to restore. Unused.
      state_transformation_fns: Transformations to apply, in order, to the state
        before writing.
      fallback_state: a state dict of an optimizer to fall back to for loading
        params that do not exist in the checkpoint (after applying all
        `state_transformation_fns`), but do exist in `Checkpointer.optimizer`.
        The union of `fallback_state` and state loaded from the checkpoint must
        match `Checkpointer.optimizer`.
      lazy_parameters: whether to load the parameters as LazyArrays to preserve
        memory.

    Returns:
      The restored train state.
    """
    del item  # not needed for restore in T5X
    from_tensorflow = gfile.exists(path + '.index')
    if from_tensorflow and state_transformation_fns:
      raise ValueError(
          'Cannot initialize from a TensorFlow checkpoint using '
          '`state_transformation_fns`.'
      )
    if from_tensorflow:
      logging.info(
          'Initializing parameters from TensorFlow checkpoint %s', path
      )
      return self._restore_checkpointer.restore_from_tf_checkpoint(
          path, strict=self._strict
      )
    return self._restore_checkpointer.restore(
        path=path,
        state_transformation_fns=state_transformation_fns,
        fallback_state=fallback_state,
        lazy_parameters=lazy_parameters,
    )


def create_checkpoint_manager(
    *,
    save_cfg: Optional[SaveCheckpointConfig] = None,
    restore_cfg: Optional[RestoreCheckpointConfig] = None,
    train_state: train_state_lib.TrainState,
    partitioner: partitioning.BasePartitioner,
    ds_iter: Optional[
        Union[tf.data.Iterator, clu.data.dataset_iterator.DatasetIterator]
    ] = None,
    model_dir: Optional[str] = None,
):
  """Creates Orbax CheckpointManager."""
  if save_cfg is not None and restore_cfg is not None:
    if (
        save_cfg.checkpoint_manager_cls
        is not restore_cfg.checkpoint_manager_cls
    ):
      msg = (
          'Must provide matching configurations of `checkpoint_manager_cls` in '
          '`save_cfg` and `restore_cfg`.'
      )
      raise ValueError(msg)

  def _get_default_args(cls_or_fcn):
    signature = inspect.signature(cls_or_fcn)
    # Only get certain parameters needed for BestCheckpointManager
    # configuration. These are the parameters of SaveBestCheckpointer that are
    # not shared by regular Checkpointer. This whole approach is very hacky, but
    # prevents us from needing to migrate every user to a new checkpoint config,
    # which is the only alternative.
    # Arguments aside from these should be set via CheckpointConfig, not gin.

    def _is_relevant_arg(key: str):
      return key in {
          'metric_name_to_monitor',
          'metric_mode',
          'keep_checkpoints_without_metrics',
          'force_keep_period',
      }

    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty and _is_relevant_arg(k)
        # Without the filtering by name specified above, we would have duplicate
        # parameters being passed, which would give an error.
    }

  def _get_checkpoint_manager_cls(cfg):
    checkpoint_manager_cls = cfg.checkpoint_manager_cls
    extra_kwargs = {}

    # Sometimes, the user pass in a `functools.partial` of
    # `SaveBestCheckpointer` which will cause issubclass to raise an Exception
    # since `functools.partial` is not a class.
    if isinstance(cfg.checkpointer_cls, functools.partial):
      func_to_check = cast(functools.partial, cfg.checkpointer_cls).func
      if issubclass(
          # pylint: disable-next=g-bare-generic
          cast(type, func_to_check),
          checkpoints.SaveBestCheckpointer,
      ):
        checkpoint_manager_cls = checkpoints.BestCheckpointManager
      # Note, this is intentionally moved out of the above if statement compared
      # to the condition below. This is because we need to handle the kwargs
      # differently since it's a functools.partial.
      extra_kwargs = _get_default_args(cfg.checkpointer_cls)
    else:
      if issubclass(cfg.checkpointer_cls, checkpoints.SaveBestCheckpointer):
        checkpoint_manager_cls = checkpoints.BestCheckpointManager
        extra_kwargs = _get_default_args(cfg.checkpointer_cls.__init__)

    return checkpoint_manager_cls, extra_kwargs

  save_dtype = None
  restore_dtype = None
  period = None
  keep = None
  should_save_restore_dataset = False
  checkpoint_manager_cls = None
  if save_cfg is not None:
    should_save_restore_dataset |= save_cfg.save_dataset
    save_dtype = save_cfg.dtype
    keep = save_cfg.keep
    period = save_cfg.period
    checkpoint_manager_cls, extra_kwargs = _get_checkpoint_manager_cls(save_cfg)
  if restore_cfg is not None:
    should_save_restore_dataset |= restore_cfg.restore_dataset
    restore_dtype = restore_cfg.dtype
    # If already set, configuration from save_cfg takes precendence. If
    # checkpoint_manager_cls is base CheckpointManager, give it a chance to be
    # reset to something more specialized.
    if (
        checkpoint_manager_cls is None
        or checkpoint_manager_cls == checkpoints.CheckpointManager
    ):
      checkpoint_manager_cls, extra_kwargs = _get_checkpoint_manager_cls(
          restore_cfg
      )
  ds_iter = ds_iter if should_save_restore_dataset else None

  return checkpoint_manager_cls(
      directory=model_dir,
      train_state=train_state,
      partitioner=partitioner,
      dataset_iterator=ds_iter,
      save_dtype=save_dtype,
      restore_dtype=restore_dtype,
      keep=keep,
      period=period,
      **extra_kwargs,
  )


class LegacyCheckpointManager(orbax.checkpoint.CheckpointManager):
  """Implementation of CheckpointManager interface for T5X.

  Uses underlying LegacyCheckpointer to handle save/restore for Dataset and
  TrainState.
  """

  def __init__(
      self,
      *,
      save_cfg: Optional[SaveCheckpointConfig],
      restore_cfg: Optional[RestoreCheckpointConfig],
      train_state_shape: train_state_lib.TrainState,
      partitioner: partitioning.BasePartitioner,
      ds_iter: Optional[
          Union[tf.data.Iterator, clu.data.dataset_iterator.DatasetIterator]
      ] = None,
      model_dir: Optional[str] = None,
      use_gda: Optional[bool] = True,
  ):
    if save_cfg is not None:
      if save_cfg.save_dataset:
        assert ds_iter is not None
      save_checkpointer = save_cfg.checkpointer_cls(
          train_state=train_state_shape,
          partitioner=partitioner,
          checkpoints_dir=model_dir,
          dataset_iterator=ds_iter if save_cfg.save_dataset else None,
          save_dtype=save_cfg.dtype,
          keep=save_cfg.keep,
          use_gda=save_cfg.use_gda,
          keep_dataset_checkpoints=save_cfg.keep_dataset_checkpoints,
      )
    else:
      save_checkpointer = None

    if restore_cfg:
      restore_checkpointer = restore_cfg.checkpointer_cls(
          train_state=train_state_shape,
          partitioner=partitioner,
          checkpoints_dir='',  # unused for restore
          dataset_iterator=ds_iter if restore_cfg.restore_dataset else None,
          restore_dtype=jnp.dtype(restore_cfg.dtype)
          if restore_cfg.dtype
          else None,
          use_gda=use_gda and restore_cfg.use_gda,
      )
      strict = restore_cfg.strict
    else:
      restore_checkpointer = None
      strict = False

    self._checkpointer = LegacyCheckpointer(
        save_checkpointer=save_checkpointer,
        restore_checkpointer=restore_checkpointer,
        strict=strict,
    )

  def save(
      self,
      train_state: train_state_lib.TrainState,
      state_transformation_fns: Sequence[
          checkpoints.SaveStateTransformationFn
      ] = (),
  ):
    """Performs save operation.

    Args:
      train_state: a TrainState PyTree to save.
      state_transformation_fns: Transformations to apply, in order, to the state
        before writing.
    """
    self._checkpointer.save(
        path='',  # not used
        item=train_state,
        state_transformation_fns=state_transformation_fns,
    )

  def restore(
      self,
      paths: Sequence[str],
      restore_cfg: Optional[RestoreCheckpointConfig] = None,
      fallback_state: Optional[Mapping[str, Any]] = None,
  ) -> Optional[
      Union[train_state_lib.TrainState, Sequence[train_state_lib.TrainState]]
  ]:
    """Performs restore operation using restore_checkpointer.

    Determines whether the indicated path is a Tensorflow checkpoint.

    Args:
      paths: A sequence of paths to restore from.
      restore_cfg: RestoreCheckpointConfig specifying restoration information.
      fallback_state: a state dict of an optimizer to fall back to for loading
        params that do not exist in the checkpoint (after applying all
        `state_transformation_fns`), but do exist in `Checkpointer.optimizer`.
        The union of `fallback_state` and state loaded from the checkpoint must
        match `Checkpointer.optimizer`.

    Returns:
      The restored TrainState if only one TrainState can be restored from the
      given paths, otherwise a sequence of TrainStates. May return None.
    """
    if restore_cfg is None or paths is None:
      return None

    restored = []
    for path in paths:
      logging.info(
          'Initializing parameters from specific T5X checkpoint %s', path
      )
      restored.append(
          self._checkpointer.restore(
              path=path,
              item=None,  # not used
              state_transformation_fns=restore_cfg.state_transformation_fns,
              fallback_state=fallback_state,
          )
      )

    if len(restored) == 1:
      restored = restored[0]
    return restored


def restore(
    checkpoint_manager: checkpoints.CheckpointManager,
    paths: Sequence[str],
    restore_cfg: RestoreCheckpointConfig,
    fallback_state: Optional[Mapping[str, Any]] = None,
) -> Union[train_state_lib.TrainState, Sequence[train_state_lib.TrainState]]:
  """Performs restore operation using restore_checkpointer.

  Determines whether the indicated path is a Tensorflow checkpoint.

  Args:
    checkpoint_manager: Orbax CheckpointManager
    paths: A sequence of paths to restore from.
    restore_cfg: RestoreCheckpointConfig specifying restoration information.
    fallback_state: a state dict of an optimizer to fall back to for loading
      params that do not exist in the checkpoint (after applying all
      `state_transformation_fns`), but do exist in `Checkpointer.optimizer`. The
      union of `fallback_state` and state loaded from the checkpoint must match
      `Checkpointer.optimizer`.

  Returns:
    The restored TrainState if only one TrainState can be restored from the
    given paths, otherwise a sequence of TrainStates.
  """
  if restore_cfg is None or paths is None:
    return None

  state_transformation_fns = restore_cfg.state_transformation_fns
  restored_checkpoints = []
  for path in paths:
    logging.info(
        'Initializing parameters from specific T5X checkpoint %s', path
    )

    from_tensorflow = gfile.exists(path + '.index')
    if from_tensorflow and state_transformation_fns:
      raise ValueError(
          'Cannot initialize from a TensorFlow checkpoint using '
          '`state_transformation_fns`.'
      )
    if from_tensorflow:
      logging.info(
          'Initializing parameters from TensorFlow checkpoint %s', path
      )
      return checkpoint_manager.restore_from_tf_checkpoint(
          path, strict=restore_cfg.strict
      )

    restored = checkpoint_manager.restore(
        path=path,
        state_transformation_fns=state_transformation_fns,
        fallback_state=fallback_state,
    )
    restored_checkpoints.append(restored)

  if len(restored_checkpoints) == 1:
    restored_checkpoints = restored_checkpoints[0]
  return restored_checkpoints


@dataclasses.dataclass
class DatasetConfig:
  """Configuration for loading a dataset from a SeqIO Task or Mixture."""

  mixture_or_task_name: Union[str, seqio.Task, seqio.Mixture]
  task_feature_lengths: Mapping[str, int]
  split: str
  batch_size: int  # Number of examples per batch.
  shuffle: bool
  seed: Optional[int]
  # Whether to use a precomputed version of the dataset from a cache dir.
  use_cached: bool = False
  pack: bool = False
  # Whether to use tensor2tensor custom ops for more efficient packing.
  use_custom_packing_ops: bool = False
  # An optional module to import for registering the referenced Mixture or Task.
  # DEPRECATED.
  module: Optional[str] = None
  # Whether to cache the dataset in memory (only applies to evaluation data).
  use_memory_cache: bool = True
  # Whether to trim output features from tasks.
  trim_output_features: bool = True


def _hashed_index(x) -> int:
  # This works for both `pjit`/`xmap` indices and `pmap` indices (which might
  # have an integer instead of a slice).
  assert all(v.step is None for v in x if isinstance(v, slice))
  return hash(
      tuple((v.start, v.stop) if isinstance(v, slice) else v for v in x)
  )


def _get_index_mappings(device_to_idxs):
  """Get device and host to index set mappings for GDA construction."""
  host_to_idxs = collections.defaultdict(list)
  idx_to_devices = collections.defaultdict(list)
  for d, idx in device_to_idxs.items():
    hashed_idx = _hashed_index(idx)
    # Only need one copy of each idx, since they are unique. Need to maintain
    # original ordering though.
    if hashed_idx not in host_to_idxs[d.process_index]:
      host_to_idxs[d.process_index].append(hashed_idx)
    # Index may correspond to multiple devices.
    idx_to_devices[hashed_idx].append(d)

  assert jax.process_index() in host_to_idxs
  for h1, idxs1 in host_to_idxs.items():
    for idx in idxs1:
      assert idx in idx_to_devices
    for h2, idxs2 in host_to_idxs.items():
      if h1 == h2:
        continue
      assert not (set(idxs1) & set(idxs2)) or set(idxs1) == set(idxs2)

  return host_to_idxs, idx_to_devices


def _create_sharded_array(
    partitioner: partitioning.BasePartitioner,
    global_shapes: PyTree,
    host_arrays: PyTree,
) -> PyTree:
  """Create jax.Array from input arrays.

  Example:

  Consider a case where the global input array has length 128. The global mesh
  specifies that the data dimension be sharded into 8 shards. This means we want
  shards of length 16. The data_layout, defined by the partitioner object,
  specifies that the data should be divided into two shards, one per host. Each
  host will have a local slice of the data (length 64).

  In this function, we will divide the local array into 4 shards of length 16.
  Each of these will be placed onto a separate device. If the sharding had
  specified only 4 global shards instead of 8, we would have divided our local
  array into only 2 shards. In this case, the first shard would be placed on the
  first two devices (replicated) and the second on the following two devices.

  Args:
    partitioner: Partitioner object containing mesh and mesh_axes
    global_shapes: PyTree matching host_arrays specifying global shape of each
      array.
    host_arrays: PyTree of LOCAL arrays (not global) that should be converted to
      jax.Array.

  Returns:
    PyTree matching host_arrays of jax.Array.
  """
  global_mesh = partitioner.mesh
  axes = partitioner.data_partition_spec
  local_devices = global_mesh.local_devices
  local_device_count = jax.local_device_count()

  # Global input array is already split into per-host shards.
  def _put_to_devices(x, global_shape):
    # Mapping of device to index slice from *global* array.

    device_to_idxs = jax.sharding.NamedSharding(
        global_mesh, axes
    ).devices_indices_map(global_shape)
    # Mapping of host to a set of unique index slices for that host.
    # Mapping of index slice to a list of devices onto which the slice should be
    # placed.
    host_to_idxs, idx_to_devices = _get_index_mappings(device_to_idxs)

    shard_length = jax.sharding.NamedSharding(global_mesh, axes).shard_shape(
        global_shape
    )[0]
    num_shards = len(x) // shard_length
    try:
      local_array_shards = np.split(x, num_shards, axis=0)
    except ValueError as array_split_error:
      raise ValueError(
          f'Unable to put to devices shape {x.shape} with '
          f'local device count {local_device_count}'
      ) from array_split_error

    # Construct mapping of device to index in the split local array.
    device_to_split_array_idx = {}
    i = 0
    for idx in host_to_idxs[jax.process_index()]:
      assert idx in idx_to_devices
      for d in idx_to_devices[idx]:
        device_to_split_array_idx[d] = i % len(local_array_shards)
      i += 1

    device_buffers = []
    for d in local_devices:
      assert d in device_to_split_array_idx
      i = device_to_split_array_idx[d]
      device_buffers.append(jax.device_put(local_array_shards[i], d))

    return device_buffers

  device_buffers = jax.tree_map(_put_to_devices, host_arrays, global_shapes)

  def _jax_array(dbs, global_shape):
    return jax.make_array_from_single_device_arrays(
        global_shape, jax.sharding.NamedSharding(global_mesh, axes), dbs
    )

  return jax.tree_map(
      _jax_array,
      device_buffers,
      global_shapes,
      is_leaf=lambda x: isinstance(x, (list, tuple)),
  )


class ShardedDatasetIterator(clu.data.dataset_iterator.DatasetIterator):
  """A wrapper iterator that returns sharded arrays."""

  def __init__(
      self,
      iterator: clu.data.dataset_iterator.DatasetIterator,
      partitioner: partitioning.BasePartitioner,
      global_shapes: PyTree,
  ):
    self._iterator = iterator
    self._global_shapes = global_shapes
    self._partitioner = partitioner

  def __next__(self):
    return _create_sharded_array(
        self._partitioner, self._global_shapes, next(self._iterator)
    )

  def reset(self):
    return self._iterator.reset()

  @property
  def element_spec(self):
    return self._iterator.element_spec

  def save(self, filename):
    return self._iterator.save(filename)

  def restore(self, filename):
    return self._iterator.restore(filename)

  @property
  def iterator(self):
    return (
        self._iterator.iterator
        if isinstance(
            self._iterator, clu.data.dataset_iterator.TfDatasetIterator
        )
        else self._iterator
    )


def prepare_train_iter(
    train_iter: Union[
        tf.data.Dataset, clu.data.dataset_iterator.DatasetIterator
    ],
    *,
    use_gda: bool,
    partitioner,
    checkpoint_cfg,
    data_layout,
) -> clu.data.dataset_iterator.PeekableDatasetIterator:
  """Prepares the training input iterator."""
  if isinstance(train_iter, tf.data.Dataset):
    train_iter = clu.data.dataset_iterator.TfDatasetIterator(
        train_iter, checkpoint=True
    )
  elif not isinstance(train_iter, clu.data.dataset_iterator.DatasetIterator):
    raise ValueError(
        f'get_dataset_fn returned unsupported type {type(train_iter)}.'
    )


  input_shapes = jax.tree_map(
      lambda x: (data_layout.batch_size, *x.shape[1:]), train_iter.element_spec
  )
  if use_gda:
    train_iter = ShardedDatasetIterator(train_iter, partitioner, input_shapes)
  return clu.data.dataset_iterator.PeekableDatasetIterator(train_iter)


def sync_global_devices(name: str) -> None:
  """Creates a barrier with given name across all hosts/devices."""
  # Internal mock TPU handling
  multihost_utils.sync_global_devices(name)


def multihost_assert_equal(input_tree, fail_message: str = ''):
  """Verifies that all the hosts have the same tree of values."""
  # Internal mock TPU handling
  multihost_utils.assert_equal(input_tree, fail_message)


# ------------------------------------------------------------------------------
# Fast *nondeterministic* hardware RNG for faster Dropout
# ------------------------------------------------------------------------------
def _hardware_uniform(
    rng_key: Array,
    shape: Shape,
    dtype: jnp.dtype = np.float32,
    minval: Array = np.float32(0),
    maxval: Array = np.float32(1),
) -> Array:
  """Random uniform method that uses non-deterministic accelerator hardware."""
  del rng_key  # non-deterministic prng.
  minval = jax.lax.convert_element_type(minval, dtype)
  maxval = jax.lax.convert_element_type(maxval, dtype)
  return jax.lax.rng_uniform(minval, maxval, shape)


# For dropout-only hardware rng.
def _hardware_bernoulli(
    rng_key: Array,
    p: Union[np.ndarray, np.floating] = np.float32(0.5),
    shape: Shape = (),
) -> Array:
  del rng_key  # non-deterministic prng.
  return jax.lax.rng_uniform(0.0, 1.0, shape) < p


def set_hardware_rng_ops():
  jax.config.update('jax_default_prng_impl', 'unsafe_rbg')


# -----------------------------------------------------------------------------
# Training utility functions.
# -----------------------------------------------------------------------------


def get_zeros_batch_like_spec(
    batch_spec: Mapping[str, jax.ShapeDtypeStruct]
) -> Mapping[str, jnp.ndarray]:
  return {k: jnp.zeros(t.shape, t.dtype) for k, t in batch_spec.items()}


def get_zeros_batch_like_dataset(
    dataset: tf.data.Dataset, batch_size=None
) -> Mapping[str, jnp.ndarray]:
  reshape = lambda s: (batch_size,) + s[1:] if batch_size else tuple(s)
  batch_spec = {
      k: jax.ShapeDtypeStruct(reshape(t.shape), t.dtype.as_numpy_dtype)
      for k, t in dataset.element_spec.items()
  }
  return get_zeros_batch_like_spec(batch_spec)


class InitFnCallable(typing_extensions.Protocol):
  """A callable that initializes model variables."""

  def __call__(
      self,
      rng: Array,
      input_shapes: Mapping[str, Array],
      input_types: Optional[Mapping[str, DType]],
  ) -> flax_scope.FrozenVariableDict:
    ...


class LearningRateCallable(typing_extensions.Protocol):

  def __call__(self, step: jnp.ndarray) -> jnp.ndarray:
    ...


def create_learning_rate_scheduler(
    factors: str = 'constant * linear_warmup * rsqrt_decay',
    base_learning_rate: float = 0.5,
    warmup_steps: int = 1000,
    decay_factor: float = 0.5,
    steps_per_decay: int = 20000,
    steps_per_cycle: int = 100000,
    step_offset: int = 0,
    min_learning_rate: float = 1e-8,
) -> LearningRateCallable:
  """Creates learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * linear_decay: linear decay from warmup_steps with decay_factor slope. Note
      this option implies 'constant * linear_warmup', and should not be used in
      in conjunction with `constant` or `linear_warmup` factors.
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * rsqrt_normalized_decay: divide by square root of max(step/warmup_steps, 1)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.

  Args:
    factors: string, factors separated by '*' that defines the schedule.
    base_learning_rate: float, the starting constant for the lr schedule.
    warmup_steps: int, how many steps to warm up for in the warmup schedule.
    decay_factor: float, the amount to decay the learning rate by.
    steps_per_decay: int, how often to decay the learning rate.
    steps_per_cycle: int, steps per cycle when using cosine decay.
    step_offset: int, an offset that the step parameters to this function are
      relative to.
    min_learning_rate: float, minimum learning rate to output. Useful for cases
      when a decay function is (mis)configured to decay to non-positive values.

  Returns:
    a function learning_rate(step): float -> {'learning_rate': float}, the
    step-dependent lr.
  """
  factors = [n.strip() for n in factors.split('*')]

  def step_fn(step: jnp.ndarray) -> jnp.ndarray:
    """Step to learning rate function."""
    step = jnp.maximum(0, step - step_offset)
    ret = 1.0
    for name in factors:
      if name == 'constant':
        ret *= base_learning_rate
      elif name == 'linear_warmup':
        ret *= jnp.minimum(1.0, step / warmup_steps)
      elif name == 'linear_decay':
        ret *= base_learning_rate * jnp.minimum(
            step / warmup_steps, 1.0 + decay_factor * (warmup_steps - step)
        )
      elif name == 'rsqrt_decay':
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'rsqrt_normalized_decay':
        ret *= jnp.sqrt(warmup_steps)
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'decay_every':
        ret *= decay_factor ** (step // steps_per_decay)
      elif name == 'cosine_decay':
        progress = jnp.maximum(
            0.0, (step - warmup_steps) / float(steps_per_cycle)
        )
        ret *= jnp.maximum(
            0.0, 0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0)))
        )
      else:
        raise ValueError('Unknown factor %s.' % name)
    ret = jnp.maximum(ret, min_learning_rate)
    return jnp.asarray(ret, dtype=jnp.float32)

  return step_fn


def get_first_valid_restore_config_and_paths(
    restore_cfgs: Sequence[RestoreCheckpointConfig],
) -> Tuple[Optional[RestoreCheckpointConfig], Sequence[str]]:
  """Returns first valid restore_cfg and the paths to restore.

  Args:
    restore_cfgs: a sequence of `RestoreCheckpointConfig` objects, which should
      be filtered to determine the first valid object.

  Returns:
    Tuple of valid RestoreCheckpointConfig and a sequence of paths.
    If the first config encountered has mode 'specfic', it is immediately
    returned, along with its specified paths.
    If the mode is 'all' or 'latest', checks to ensure that there are valid
    checkpoints at each of the provided paths and filters the returned paths
    accordingly.
  """
  for restore_cfg in restore_cfgs:
    paths = (
        [restore_cfg.path]
        if isinstance(restore_cfg.path, str)
        else restore_cfg.path
    )
    if restore_cfg.mode == 'specific':
      return restore_cfg, paths
    elif restore_cfg.mode in ('all', 'latest'):
      for ckpt_dir in paths:
        if not gfile.isdir(ckpt_dir):
          raise ValueError(
              'Checkpoint path(s) must be valid directories when using '
              "restore mode 'all' or 'latest'."
          )
        # Check if this is a TensorFlow checkpoint dir.
        tf_ckpt_state = tf.train.get_checkpoint_state(ckpt_dir)

        if tf_ckpt_state:
          ckpt_paths = tf_ckpt_state.all_model_checkpoint_paths
        else:
          ckpt_paths = [
              os.path.join(ckpt_dir, f'checkpoint_{step}')
              for step in checkpoints.all_steps(ckpt_dir)
          ]
        if not ckpt_paths:
          logging.info(
              'No checkpoints found in specified directory: %s', ckpt_dir
          )
          continue
        if restore_cfg.mode == 'latest':
          logging.info('Using latest T5X checkpoint.')
          ckpt_paths = ckpt_paths[-1:]
        return restore_cfg, ckpt_paths
    else:
      logging.error('Unsupported checkpoint restore mode: %s', restore_cfg.mode)
  return None, []


def get_fallback_state(
    restore_cfg: RestoreCheckpointConfig,
    init_fn: Callable[[jnp.ndarray], Mapping[str, Any]],
    init_rng: jnp.ndarray,
) -> Optional[Mapping[str, Any]]:
  """Returns the fallback_state that can be used in restore()."""
  if restore_cfg is None:
    return
  if restore_cfg.fallback_to_scratch:
    if not restore_cfg.state_transformation_fns:
      raise ValueError(
          '`state_transformation_fns` must be provided with '
          '`fallback_to_scratch`'
      )
    if init_rng is None:
      raise ValueError(
          'An `init_rng` must be provided with `fallback_to_scratch`'
      )
    fallback_state = init_fn(init_rng)
  else:
    fallback_state = None
  return fallback_state


class TrainStateInitializer:
  """Helper for initializing partitioned TrainState from checkpoints or scratch.

  Common use cases:

  * To restore from a single checkpoint, use `from_checkpoint`.
  * To iterate over multiple checkpoints without recompiling the model,
    use `from_checkpoints`.
  * To initialize from scratch, use `from_scratch`.
  * To restore from a checkpoint with a fallback to initializing from scratch,
    use `from_checkpoint_or_scratch`.

  Attributes:
    global_train_state_shape: a TrainState containing the global (unpartitioned)
      shape  (in `jax.ShapeDtypeStruct`) of each parameter instead of its value.
    train_state_axes: a TrainState object containing a PartitionSpec (or None)
      for each parameter, in place of the parameter itself.
  """

  # TODO(adarob): Replace input_shapes and input_types with sample batch.
  def __init__(
      self,
      optimizer_def: Optional[optimizers.OptimizerDefType],
      init_fn: InitFnCallable,
      input_shapes: Mapping[str, Array],
      partitioner: partitioning.BasePartitioner,
      input_types: Optional[Mapping[str, DType]] = None,
  ):
    """TrainStateInitializer constructor.

    Args:
      optimizer_def: Optimizer def to be initialized, or None to create a
        `InferenceState` without an optimizer.
      init_fn: callable that initializes model variables from a PRNGKey and the
        input shapes.
      input_shapes: a mapping from key to array shape for each feature in the
        global (unsharded) input batch.
      partitioner: the partitioner to use.
      input_types: a mapping from key to array type for each feature in the
        global (unshared) input batch. If not provided, the type is assumed to
        be `jnp.float32`.
    """

    def initialize_train_state(rng: Array):
      initial_variables = init_fn(
          rng=rng, input_shapes=input_shapes, input_types=input_types
      )
      if optimizer_def:
        return train_state_lib.FlaxOptimTrainState.create(
            optimizer_def, initial_variables
        )
      return train_state_lib.InferenceState.create(initial_variables)

    self._partitioner = partitioner
    self.global_train_state_shape = jax.eval_shape(
        initialize_train_state, rng=jax.random.PRNGKey(0)
    )
    self.train_state_axes = partitioner.get_mesh_axes(
        self.global_train_state_shape
    )
    self._initialize_train_state = initialize_train_state

    # Currently scanned layers require passing annotations through to the
    # point of the scan transformation to resolve an XLA SPMD issue.

    # init_fn is always(?) equal to model.get_initial_variables, fetch the model
    # instance from the bound method.
    model = init_fn.__self__  # pytype: disable=attribute-error
    if (
        hasattr(model, 'module')
        and hasattr(model.module, 'scan_layers')
        and model.module.scan_layers
    ):
      if hasattr(model.module, 'spmd_annotations'):
        # update top-level module with spmd annotations.
        model.module = model.module.clone(
            parent=None, spmd_annotations=self.train_state_axes.params
        )

  def from_scratch(self, init_rng: Array) -> train_state_lib.TrainState:
    """Initializes the partitioned Optimizer from scratch."""
    logging.info('Initializing parameters from scratch.')

    # If pretraining and no checkpoint imported, we jit the (sharded-) init
    # function to minimize fragmentation. We use the same partition
    # setup as the training step/loop to initialize everything "in-place" and
    # avoid communication or OOM.
    p_initialize_train_state_fn = self._partitioner.partition(
        self._initialize_train_state,
        in_axis_resources=None,
        out_axis_resources=self.train_state_axes,
    )
    return p_initialize_train_state_fn(init_rng)

  # TODO(b/216650048) deprecate this function and use orbax.
  def from_checkpoints(
      self,
      restore_cfgs: Sequence[RestoreCheckpointConfig],
      ds_iter: Optional[tf.data.Iterator] = None,
      init_rng: Optional[jnp.ndarray] = None,
  ) -> Iterable[Tuple[train_state_lib.TrainState, str]]:
    """Yields 0 or more restored partitioned Optimizers, and maybe datasets.

    The manner in which parameters are initialized depends on `restore_cfgs` and
    `restore_cfgs` is iterated over and the first config that matches one or
    more existing checkpoints is used to generate restored optimizers from the
    checkpoint(s). Any remaining configs are ignored.

    Args:
      restore_cfgs: ordered sequence of configurations specifying checkpoint(s)
        to restore from. The first config to match a checkpoint will be used.
      ds_iter: a tf.data.Iterator for the input data, or None. If provided, the
        referenced iterator's state may be silently restored (depending on the
        config's `restore_dataset` value) along with the optimizer.
      init_rng: for initializing parameters from scratch when they are not
        available in the checkpoint and `fallback_to_scratch` is True

    Yields:
      TrainState with initialized optimizer, with parameters copied to devices.
      Path to restored checkpoint.
    """

    def _restore_path(path, cfg):
      restore_checkpointer = cfg.checkpointer_cls(
          train_state=self.global_train_state_shape,
          partitioner=self._partitioner,
          checkpoints_dir='',  # unused for restore
          dataset_iterator=ds_iter if cfg.restore_dataset else None,
          restore_dtype=jnp.dtype(cfg.dtype) if cfg.dtype else None,
          use_gda=cfg.use_gda,
      )

      from_tensorflow = gfile.exists(path + '.index')
      if from_tensorflow and cfg.state_transformation_fns:
        raise ValueError(
            'Cannot initialize from a TensorFlow checkpoint using '
            '`state_transformation_fns`.'
        )
      if from_tensorflow:
        logging.info(
            'Initializing parameters from TensorFlow checkpoint %s', path
        )
        return restore_checkpointer.restore_from_tf_checkpoint(
            path, strict=cfg.strict
        )

      else:
        fallback_state = get_fallback_state(
            cfg, lambda rng: self.from_scratch(rng).state_dict(), init_rng
        )

        logging.info(
            'Initializing parameters from specific T5X checkpoint %s', path
        )
        return restore_checkpointer.restore(
            path=path,
            state_transformation_fns=cfg.state_transformation_fns,
            fallback_state=fallback_state,
        )

    restore_cfg, paths = get_first_valid_restore_config_and_paths(restore_cfgs)
    for path in paths:
      yield _restore_path(path, restore_cfg), path

  def from_checkpoint(
      self,
      ckpt_cfgs: Sequence[RestoreCheckpointConfig],
      *,
      ds_iter: Optional[tf.data.Iterator] = None,
      init_rng: Optional[jnp.ndarray] = None,
  ) -> Optional[train_state_lib.TrainState]:
    """Restores (at most) 1 checkpoint using `from_checkpoints`, or dies."""
    train_states = [
        state
        for state, _ in self.from_checkpoints(
            ckpt_cfgs, ds_iter=ds_iter, init_rng=init_rng
        )
    ]
    if len(train_states) > 1:
      raise ValueError(
          f'Expected at most 1 checkpoint but got {len(train_states)} for '
          f'config(s): {ckpt_cfgs}'
      )
    return (train_states[0]) if train_states else None

  def from_checkpoint_or_scratch(
      self,
      ckpt_cfgs: Sequence[RestoreCheckpointConfig],
      *,
      init_rng: Array,
      ds_iter: Optional[tf.data.Iterator] = None,
  ) -> Optional[train_state_lib.TrainState]:
    """Initializes from checkpoint, if found, or from scratch."""
    return self.from_checkpoint(
        ckpt_cfgs, ds_iter=ds_iter, init_rng=init_rng
    ) or self.from_scratch(init_rng)


# -----------------------------------------------------------------------------
# Logging utility functions
# -----------------------------------------------------------------------------


def log_model_info(
    log_file: Optional[str],
    full_train_state: train_state_lib.TrainState,
    partitioner: partitioning.BasePartitioner,
):
  """Log the variable shapes information and optionally write it to a file."""
  # Only write logs on host 0.
  if jax.process_index() != 0:
    return

  state_dict = full_train_state.state_dict()
  total_num_params = jax.tree_util.tree_reduce(
      np.add, jax.tree_map(np.size, state_dict['target'])
  )

  logical_axes = partitioner.get_logical_axes(full_train_state).state_dict()

  mesh_axes = jax.tree_map(
      lambda x: tuple(x) if x is not None else None,
      partitioner.get_mesh_axes(full_train_state).state_dict(),
  )

  def _log_info_and_write_to_file(writer, format_str, *args):
    logging.info(format_str, *args)
    if writer is not None:
      writer.write(format_str % args + '\n')

  with contextlib.ExitStack() as stack:
    writer = (
        stack.enter_context(gfile.GFile(log_file, 'w'))
        if log_file is not None
        else None
    )

    # Log params
    def _log_variable(
        name: str,
        arr: Optional[np.ndarray],
        logical_axes: Optional[partitioning.AxisNames],
        mesh_axes: Optional[partitioning.PartitionSpec],
    ):
      # Log nothing on empty dict leaves, which occur with optax EmptyState().
      if isinstance(arr, dict) and not arr:
        return
      if arr is None:
        _log_info_and_write_to_file(writer, 'Variable    %-80s None', name)
        return
      if logical_axes is None or len(logical_axes) != len(arr.shape):
        shape_str = str(arr.shape)
      else:
        shape_str = '({})'.format(
            ', '.join(
                f'{name}={dimension}'
                for name, dimension in zip(logical_axes, arr.shape)
            )
        )
      _log_info_and_write_to_file(
          writer,
          'Variable %-80s size %-12s shape %-40s partition spec %s',
          name,
          arr.size,
          shape_str,
          mesh_axes,
      )

    jax.tree_map(
        _log_variable,
        state_utils.get_name_tree(state_dict['target'], keep_empty_nodes=True),
        state_dict['target'],
        logical_axes['target'],
        mesh_axes['target'],
    )

    _log_info_and_write_to_file(
        writer, 'Total number of parameters: %d', total_num_params
    )

    # Add a blank line between params and states.
    _log_info_and_write_to_file(writer, '')

    jax.tree_map(
        _log_variable,
        state_utils.get_name_tree(state_dict['state'], keep_empty_nodes=True),
        state_dict['state'],
        logical_axes['state'],
        mesh_axes['state'],
    )


# -----------------------------------------------------------------------------
# Utility functions for prediction and evaluation.
# -----------------------------------------------------------------------------


class InferStepWithRngCallable(typing_extensions.Protocol):

  def __call__(
      self,
      params: Mapping[str, Any],
      batch: Mapping[str, jnp.ndarray],
      rng: jnp.ndarray = None,  # pytype: disable=annotation-type-mismatch  # jax-ndarray
  ) -> PyTree:
    """Runs an inference step returning a prediction or score."""
    ...


class InferStepWithoutRngCallable(typing_extensions.Protocol):

  def __call__(
      self, params: Mapping[str, Any], batch: Mapping[str, jnp.ndarray]
  ) -> PyTree:
    """Runs an inference step returning a prediction or score."""
    ...


InferStepCallable = Union[InferStepWithRngCallable, InferStepWithoutRngCallable]

# NOTE: We're not more prescriptive than PyTree because that's what
# InferStepCallable expects.
_InferFnResult = Sequence[Tuple[int, PyTree]]
_InferFnWithAuxResult = Tuple[_InferFnResult, Mapping[str, Sequence[Any]]]


class InferFnCallable(typing_extensions.Protocol):

  def __call__(
      self,
      ds: tf.data.Dataset,
      train_state: train_state_lib.TrainState,
      rng: Optional[jnp.ndarray] = None,
  ) -> Union[_InferFnResult, _InferFnWithAuxResult]:
    """Runs inference on the dataset."""
    ...


def _remove_padding(all_inferences, all_indices):
  """Remove padded examples.

  Args:
    all_inferences: PyTree[total_examples + padding_count, ...].
    all_indices: [total_examples + padding_count].

  Returns:
    all_inferences in shape PyTree[total_examples, ...].
    all_indices in shape [total_exmamples].
  """
  non_pad_idxs = np.where(all_indices >= 0)
  all_indices = all_indices[non_pad_idxs]
  all_inferences = jax.tree_map(lambda x: x[non_pad_idxs], all_inferences)
  return all_inferences, all_indices


def get_infer_fn(
    infer_step: InferStepCallable,
    batch_size: int,
    train_state_axes: train_state_lib.TrainState,
    partitioner: partitioning.BasePartitioner,
    keep_aux_as_numpy: bool = False,
) -> InferFnCallable:
  """Get prediction function for the SeqIO evaluator.

  The returned prediction function should take in an enumerated dataset, make
  predictions and return in an enumerated form with the original indices and
  examples zipped together. This ensures that the predictions are compared to
  the targets in a correct order even if the dataset is sharded across
  multiple hosts and gathered in a nondeterministic way.

  jax.process_index == 0 is used as a "main host", i.e., it gathers all
  inference results and returns.

  Shape notation:
    Per replica set num replicas: R
    Per replica set batch size: B
    Number of replica sets: H
    Length: L

    Some transformations have shape transformation annotation, e.g.,
    [B, L] -> [R, B/R, L].

  Args:
    infer_step: a callable that executes one prediction step. Should not yet be
      partitioned or pmapped.
    batch_size: the number of examples in the global infer batch.
    train_state_axes: Partitioning info for the train state object.
    partitioner: partitioner to use.
    keep_aux_as_numpy: bool. whether to leave aux values as numpy arrays; can be
      used to save space when saving bfloat16s

  Returns:
    predict_fn: a callable which takes in the enumerated infer dataset and an
      optimizer and runs the prediction.
  """

  def infer_step_with_indices(params, batch, rng, indices):
    if 'rng' in inspect.signature(infer_step).parameters:
      res = typing.cast(InferStepWithRngCallable, infer_step)(
          params, batch, rng
      )
    else:
      res = typing.cast(InferStepWithoutRngCallable, infer_step)(params, batch)
    return indices, res

  partitioned_infer_step = partitioner.partition(
      infer_step_with_indices,
      in_axis_resources=(
          train_state_axes.params,
          partitioner.data_partition_spec,
          None,
          partitioner.data_partition_spec,
      ),
      out_axis_resources=(None, None),
  )

  data_layout = partitioner.get_data_layout(batch_size)
  shard_id = data_layout.shard_id
  num_shards = data_layout.num_shards

  per_shard_batch_size = batch_size // num_shards

  def infer_fn(
      ds: tf.data.Dataset,
      train_state: train_state_lib.TrainState,
      rng: Optional[jnp.ndarray] = None,
  ):
    ds_shapes = jax.tree_map(lambda x: jnp.array(x.shape), ds.element_spec)
    multihost_assert_equal(
        ds_shapes,
        (
            'Dataset element shapes do not agree across hosts. '
            'This could be an indication that the dataset is nondeterministic.'
        ),
    )
    try:
      original_ds_length = len(ds)
      dataset_remainder = original_ds_length % batch_size  # pytype:disable=wrong-arg-types
      logging.info('length of dataset = %s', len(ds))
    except TypeError as e:
      if str(e).endswith('dataset length is unknown.'):
        logging.warning(
            'The following error is likely due to the use of TensorFlow v1 in '
            'your dataset pipeline. Verify you are not importing from '
            '`tf.compat.v1` as part of your pipeline.'
        )
      raise e

    if dataset_remainder:
      dataset_pad_amt = batch_size - dataset_remainder
      logging.info(
          'Padding infer dataset with %d examples for even per-replica shards.',
          dataset_pad_amt,
      )
      # Pad with the first example using an index of -1 so seqio will ignore.
      pad_ds = (
          ds.take(1)
          .map(lambda i, x: (np.int64(-1), x))
          .cache()
          .repeat(dataset_pad_amt)
      )
      ds = ds.concatenate(pad_ds)

    # Shard the infer dataset across replica sets.
    sharded_ds = ds.shard(num_shards, shard_id).batch(
        per_shard_batch_size, drop_remainder=True
    )
    multihost_assert_equal(
        jnp.array(len(sharded_ds)), 'Dataset lengths do not agree across hosts.'
    )

    logging.info(
        (
            'The infer dataset is sharded into %d shards with per-shard '
            'batch size of %d'
        ),
        num_shards,
        per_shard_batch_size,
    )

    # Run inference for each replica set.
    batched_results, all_indices = [], []
    for index, infer_batch in sharded_ds.as_numpy_iterator():
      if rng is None:
        step_rng = None
      else:
        step_rng, rng = jax.random.split(rng)
      # Run fast inference on batch.
      # [B, ...] -> [B * shard_count, ...]
      # partitioned_infer_step executes infer_step on sharded batched data, and
      # returns de-sharded batched indices and result replicated on all hosts.

      if jax.process_count() > 1:
        inputs = multihost_utils.host_local_array_to_global_array(
            (infer_batch, step_rng, index),
            partitioner.mesh,
            (
                partitioner.data_partition_spec,
                None,
                partitioner.data_partition_spec,
            ),
        )
        batch_indices, batch_result = partitioned_infer_step(
            train_state.params, *inputs
        )
        batch_indices, batch_result = (
            multihost_utils.global_array_to_host_local_array(
                (batch_indices, batch_result), partitioner.mesh, (None, None)
            )
        )
      else:
        batch_indices, batch_result = partitioned_infer_step(
            train_state.params, infer_batch, step_rng, index
        )
        logging.info('Inference of batch %s done.', index)

      def _copy_to_host_async(x):
        if hasattr(x, 'addressable_data'):
          # Array is fully replicated.
          x.addressable_data(0).copy_to_host_async()
          return x.addressable_data(0)
        else:
          x.copy_to_host_async()
          return x

      try:
        batch_result = jax.tree_map(_copy_to_host_async, batch_result)
        batch_indices = jax.tree_map(_copy_to_host_async, batch_indices)
      except AttributeError:
        # Similar to jax.device_get, we skip transfers for non DeviceArrays.
        pass

      batched_results.append(batch_result)
      all_indices.append(batch_indices)

    logging.info('Inference of all batches done.')
    all_inferences = batched_results

    # List[B * shard_count, ...] -> [B * shard_count * batch_count, ...]
    all_inferences = jax.tree_map(
        lambda *args: np.concatenate(args), *all_inferences
    )
    all_indices = np.concatenate(all_indices)

    all_inferences, all_indices = _remove_padding(all_inferences, all_indices)

    # Results are returned from infer_step out of order due to shard operation.
    # Note: remove padding first, as -1 indices would mess up this operation.
    # Note: all_inferences may be a PyTree, not just an array, e.g. if
    # `infer_step` is `model.predict_batch_with_aux`.
    all_inferences = jax.tree_map(lambda x: x[all_indices], all_inferences)
    all_indices = all_indices[all_indices]

    # aux_values is supposed to be a dictionary that maps strings to a set of
    # auxiliary values.
    #
    # We don't want to flatten/unflatten the aux values. We want to preserve the
    # unflattened values with the type List[Mapping[str, Sequence[Any]]]. We do
    # this as a memory optimization to avoid lots of redundant keys if we'd
    # instead had List[Mapping[str, Any]].
    #
    # It has shape Mapping[str, [B * shard_count * batch_count, ...]]. That is,
    # the first dimension of each of the values in aux_values is equal to
    # len(all_inferences).
    aux_values = None
    if (
        isinstance(all_inferences, tuple)
        and len(all_inferences) == 2
        and isinstance(all_inferences[1], Mapping)
    ):
      all_inferences, aux_values = all_inferences

    # Translate to List[...] by flattening inferences making sure to
    # preserve structure of individual elements (inferences are not assumed to
    # be simple np.array). Finally, zip inferences with corresponding indices
    # and convert leaf np.arrays into lists.
    all_inferences, struct = jax.tree_util.tree_flatten(all_inferences)
    all_inferences = map(
        functools.partial(jax.tree_util.tree_unflatten, struct),
        zip(*all_inferences),
    )
    indices_and_outputs = list(zip(all_indices, all_inferences))
    indices_and_outputs = jax.tree_map(
        lambda x: np.array(x).tolist(), indices_and_outputs
    )
    if len(indices_and_outputs) != original_ds_length:
      raise ValueError(
          'Size of indices_and_outputs does not match length of original '
          'dataset: %d versus %d'
          % (len(indices_and_outputs), original_ds_length)
      )

    if aux_values is None:
      return indices_and_outputs
    else:
      if keep_aux_as_numpy:
        aux_values = jax.tree_map(lambda x: list(np.array(x)), aux_values)
      else:
        aux_values = jax.tree_map(lambda x: np.array(x).tolist(), aux_values)
      return indices_and_outputs, aux_values

  return infer_fn


# -----------------------------------------------------------------------------
# SeqIO utility functions.
# -----------------------------------------------------------------------------


def import_module(module: str):
  """Imports the given module at runtime."""
  logging.info('Importing %s.', module)
  try:
    importlib.import_module(module)
  except RuntimeError as e:
    if (
        str(e)
        == 'Attempted to add a new configurable after the config was locked.'
    ):
      raise RuntimeError(
          'Your Task/Mixture module contains gin configurables that must be '
          'loaded before gin flag parsing. One fix is to add '
          f"'import {module}' in your gin file."
      ) from e
    raise e


def get_vocabulary(
    cfg: DatasetConfig,
) -> Tuple[seqio.Vocabulary, seqio.Vocabulary]:
  """Returns `seqio.Vocabulary` objects associated with the `Mixture`/`Task`.

  Args:
    cfg: the DatasetConfig specifying which mixture or task to get the
      vocabularies for.

  Returns:
    A tuple of seqio.Vocabulary for inputs and targets.

  Raises:
    ValueError: if inputs and targets are not both present and vocabularies
      are different.
  """
  if cfg.module:
    warnings.warn(
        (
            'The use of `DatasetConfig.module` and `MIXTURE_OR_TASK_MODULE` is '
            'deprecated in favor of importing the module directly or via gin.'
        ),
        DeprecationWarning,
    )
    import_module(cfg.module)

  if isinstance(cfg.mixture_or_task_name, seqio.DatasetProviderBase):
    mixture_or_task = cfg.mixture_or_task_name
  else:
    mixture_or_task = seqio.get_mixture_or_task(cfg.mixture_or_task_name)
  features = mixture_or_task.output_features

  if 'inputs' in features and 'targets' in features:
    return (features['inputs'].vocabulary, features['targets'].vocabulary)

  # If a mix of PassThroughVocabularies and other Vocabularies are specified,
  # use the non-PassThroughVocabularies.
  # TODO(b/185912004): Remove this once a more general solution is implemented.
  vocabularies = list(
      f.vocabulary
      for f in features.values()
      if not isinstance(f.vocabulary, seqio.PassThroughVocabulary)
  )

  # Otherwise, if all of the vocabs are PassThroughVocabularies, use those.
  if not vocabularies:
    vocabularies = list(f.vocabulary for f in features.values())

  # If there still aren't any vocabularies, raise an error.
  if not vocabularies:
    raise ValueError(
        '"inputs" and "targets" are not both present, and '
        'no vocabularies were set for any features.'
    )

  first_vocab = vocabularies[0]
  for vocab in vocabularies[1:]:
    if vocab != first_vocab:
      raise ValueError(
          '"inputs" and "targets" are not both present, and '
          'vocabularies are different.'
      )
  return (first_vocab, first_vocab)


def verify_matching_vocabs(cfg: DatasetConfig, model: Any):
  """Verify whether the task vocab matches the model vocab.

  The seqio Task and the Model both define their vocabularies
  separately, but these vocabularies must match or else the training/inference
  results will not be sensible. This functions validates that they do match,
  under the assumption that this is a standard Encoder-only, Decoder-only,
  or Encoder-decoder model.

  Args:
    cfg: The DatasetConfig of the training/inference task.
    model: A BaseTransformerModel model with input_vocabulary and
      output_vocabulary attributes.

  Raises:
    ValueError: If the task vocabulary does not match the model vocabulary.
  """
  ds_vocabs = get_vocabulary(cfg)
  if (
      ds_vocabs[0] != model.input_vocabulary
      or ds_vocabs[1] != model.output_vocabulary
  ):
    raise ValueError(
        'Model and Task vocabularies do not match:\n'
        f'  task={cfg.mixture_or_task_name}\n'
        f'  ds_vocabs=({ds_vocabs[0]}, {ds_vocabs[1]})\n'
        f'  model.input_vocabulary={model.input_vocabulary}\n'
        f'  model.output_vocabulary={model.output_vocabulary}\n'
    )




def get_dataset(
    cfg: DatasetConfig,
    shard_id: int,
    num_shards: int,
    feature_converter_cls: Callable[..., seqio.FeatureConverter],
    num_epochs: Optional[int] = None,
    continue_from_last_checkpoint: bool = False,
) -> tf.data.Dataset:
  """Returns a dataset from SeqIO based on a `DatasetConfig`."""
  if continue_from_last_checkpoint:
    raise ValueError(
        '`continue_from_last_checkpoint` must be set to False as this is not '
        'supported by this dataset fn.'
    )
  del continue_from_last_checkpoint

  if cfg.module:
    import_module(cfg.module)

  if cfg.batch_size % num_shards:
    raise ValueError(
        f'Batch size ({cfg.batch_size}) must be divisible by number of '
        f'shards ({num_shards}).'
    )

  seed = cfg.seed


  shard_info = seqio.ShardInfo(index=shard_id, num_shards=num_shards)

  if seed is None:
    # Use a shared timestamp across devices as the seed.
    seed = multihost_utils.broadcast_one_to_all(np.int32(time.time()))

  return get_dataset_inner(
      cfg, shard_info, feature_converter_cls, seed, num_epochs
  )


def get_dataset_inner(
    cfg: DatasetConfig,
    shard_info: seqio.ShardInfo,
    feature_converter_cls: Callable[..., seqio.FeatureConverter],
    seed: Optional[int] = None,
    num_epochs: Optional[int] = None,
):
  """Internal fn to load a dataset from SeqIO based on a `DatasetConfig`."""
  batch_size = cfg.batch_size // shard_info.num_shards
  if isinstance(cfg.mixture_or_task_name, seqio.DatasetProviderBase):
    mixture_or_task = cfg.mixture_or_task_name
  else:
    mixture_or_task = seqio.get_mixture_or_task(cfg.mixture_or_task_name)
  if seed is not None:
    if not str(jax.devices()[0]).startswith('MOCK_TPU'):
      multihost_assert_equal(
          np.array(seed),
          (
              f'`seed` is not same across hosts; {jax.process_index} has a seed'
              f' of {seed}'
          ),
      )
    logging.info(
        (
            "Initializing dataset for task '%s' with a replica batch size of %d"
            ' and a seed of %d'
        ),
        mixture_or_task.name,
        batch_size,
        seed,
    )

  in_memory_shuffle = cfg.shuffle
  return seqio.get_dataset(
      mixture_or_task_name=mixture_or_task,
      task_feature_lengths=cfg.task_feature_lengths,
      dataset_split=cfg.split,
      shuffle=in_memory_shuffle,
      num_epochs=num_epochs,
      feature_converter=feature_converter_cls(
          pack=cfg.pack, use_custom_packing_ops=cfg.use_custom_packing_ops
      ),
      shard_info=shard_info,
      use_cached=cfg.use_cached,
      seed=seed,
      trim_output_features=cfg.trim_output_features,
      batch_size=batch_size,
  )


class GetDatasetCallable(typing_extensions.Protocol):
  """Interface for a function returning a dataset (iterator)."""

  def __call__(
      self,
      cfg: DatasetConfig,
      shard_id: int,
      num_shards: int,
      feature_converter_cls: Callable[..., seqio.FeatureConverter],
      num_epochs: Optional[int] = None,
      continue_from_last_checkpoint: bool = True,
  ) -> Union[clu.data.dataset_iterator.DatasetIterator, tf.data.Dataset]:
    ...


class GetEvalDatasetCallable(typing_extensions.Protocol):
  """Interface for a function returning a dataset (iterator)."""

  def __call__(
      self,
      cfg: DatasetConfig,
      shard_id: int,
      num_shards: int,
      eval_steps: int,
      feature_converter_cls: Callable[..., seqio.FeatureConverter],
  ) -> Mapping[str, tf.data.Dataset]:
    ...


def get_training_eval_datasets(
    cfg: DatasetConfig,
    shard_id: int,
    num_shards: int,
    eval_steps: int,
    feature_converter_cls: Callable[..., seqio.FeatureConverter],
    deterministic: bool = False,
    model_dir: Optional[str] = None,
    start_step: int = 0,
) -> Mapping[str, tf.data.Dataset]:
  """Returns a mapping from eval task name to its dataset."""
  if isinstance(cfg.mixture_or_task_name, seqio.DatasetProviderBase):
    mixture_or_task = cfg.mixture_or_task_name
  else:
    mixture_or_task = seqio.get_mixture_or_task(cfg.mixture_or_task_name)
  datasets = {}
  get_dataset_fn = get_dataset
  if deterministic:
    assert model_dir is not None
    get_dataset_fn = functools.partial(
        get_deterministic_dataset, model_dir=model_dir, start_step=start_step
    )

  if cfg.batch_size % num_shards:
    raise ValueError(
        f'Batch size ({cfg.batch_size}) must be divisible by number of '
        f'shards ({num_shards}).'
    )

  def _repeat_shard_batch_take_cache(ds: tf.data.Dataset):
    # We shard and batch the full, repeated dataset to avoid issues with uneven
    # file shards.
    if not isinstance(ds, tf.data.Dataset):
      raise ValueError('Only tf.data.Dataset objects supported.')
    ds = (
        ds.unbatch()
        .repeat()
        .shard(num_shards, shard_id)
        .batch(cfg.batch_size // num_shards, drop_remainder=True)
        .take(eval_steps)
    )
    if cfg.use_memory_cache:
      return ds.cache()
    else:
      return ds

  for task in seqio.get_subtasks(mixture_or_task):
    if cfg.split not in task.splits:
      logging.info(
          "Task %s has no '%s' split; skipping training evaluation.",
          task.name,
          cfg.split,
      )
      continue
    logging.info('Loading task %s for training evaluation.', task.name)
    task_cfg = dataclasses.replace(
        cfg, mixture_or_task_name=task.name, batch_size=1
    )
    # We set `num_epochs` to be finite to avoid infinite loops on shards that
    # have input examples that are all filtered.
    datasets[task.name] = _repeat_shard_batch_take_cache(
        get_dataset_fn(
            task_cfg,
            shard_id=0,
            num_shards=1,
            feature_converter_cls=feature_converter_cls,
            num_epochs=eval_steps * cfg.batch_size,
            continue_from_last_checkpoint=False,
        )
    )

  if isinstance(mixture_or_task, seqio.Mixture):
    datasets[mixture_or_task.name] = _repeat_shard_batch_take_cache(
        get_dataset_fn(
            dataclasses.replace(cfg, batch_size=1),
            shard_id=0,
            num_shards=1,
            feature_converter_cls=feature_converter_cls,
            num_epochs=eval_steps * cfg.batch_size,
            continue_from_last_checkpoint=False,
        )
    )

  return datasets


def round_vocab_size_to_multiple(
    vocabulary: seqio.Vocabulary, divisor: int = 128
):
  """Round up vocabulary size for improved TPU performance."""
  size = vocabulary.vocab_size
  return size + -size % divisor


def flatten_dict_string_keys(x):
  """Flattens a nested dictionary to have string keys and '/' separators."""
  return traverse_util.flatten_dict(flax.core.unfreeze(x), sep='/')


class _RegexMap(collections.abc.Mapping):
  """Ordered mapping from regexes to values requiring a full match."""

  def __init__(self, kvs: Sequence[Tuple[str, Any]]):
    self._kvs = [(re.compile(k), v) for k, v in kvs]

  def __getitem__(self, key: str) -> Any:
    for pattern, v in self._kvs:
      if pattern.fullmatch(key):
        return v
    raise KeyError(f'No pattern matching key: {key}')

  def __len__(self) -> int:
    return len(self._kvs)

  def __iter__(self) -> Iterable[Tuple[re.Pattern, Any]]:
    return iter(self._kvs)


def override_params_axes_names(
    model_variables: flax_scope.FrozenVariableDict,
    params_axes_names_override: Sequence[Tuple[str, Tuple[str, ...]]] = (),
) -> flax_scope.FrozenVariableDict:
  """Applies parameter axis names overrides to axes variables.

  Args:
    model_variables: the original model variables containing the 'params_axes'
      collection.
    params_axes_names_override: a priority-ordered mapping from regex patterns
      (fully matching parameter names) to tuples containing string logical axis
      names to replace model-derived names.

  Returns:
    an updated set of model variables with the overrides applied to the
    'params_axes' collection.
  """
  params_axes_names_override_map = _RegexMap(params_axes_names_override)

  if 'params_axes' not in model_variables:
    raise ValueError(
        "Model variables do not contain a 'params_axes' collection to apply an "
        'override to.'
    )
  model_variables = model_variables.unfreeze()
  flat_params = traverse_util.flatten_dict(model_variables['params'])
  flat_params_axes = traverse_util.flatten_dict(model_variables['params_axes'])

  for key, param in flat_params.items():
    param_name = '/'.join(key)
    override = params_axes_names_override_map.get(param_name)
    if override is None:
      continue

    param_axes_key = key[:-1] + (f'{key[-1]}_axes',)

    curr_metadata = flat_params_axes.get(param_axes_key)

    if curr_metadata is None:
      logging.info('Adding axis names for %s: %s', param_name, override)
    else:
      assert isinstance(curr_metadata, flax_partitioning.AxisMetadata)
      logging.info(
          'Replacing axis names for %s (%s) with %s.',
          param_name,
          curr_metadata.names,
          override,
      )

    if param.ndim != len(override):
      raise ValueError(
          f'Provided axis name override for {param_name} does not match '
          f'param rank ({param.ndim}): {override}'
      )
    flat_params_axes[param_axes_key] = flax_partitioning.AxisMetadata(
        names=override
    )

  model_variables['params_axes'] = traverse_util.unflatten_dict(
      flat_params_axes
  )
  return flax.core.freeze(model_variables)


