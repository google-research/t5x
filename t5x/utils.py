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

"""General utility functions for t5x."""
import collections.abc
from concurrent.futures import thread
import dataclasses
import functools
import importlib
import os
import re
import time
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple, Type, Union
import warnings

from absl import logging
from flax import optim
from flax import traverse_util
import flax.core
from flax.core import scope as flax_scope
from flax.linen import partitioning as flax_partitioning

import jax
from jax import prng
import jax.numpy as jnp
import numpy as np
import seqio
from t5x import checkpoints
from t5x import multihost_utils
from t5x import partitioning
from t5x import state_utils
from t5x import train_state as train_state_lib
import tensorflow as tf
from tensorflow.io import gfile
import typing_extensions


Array = Union[np.ndarray, jnp.ndarray, jax.pxla.ShardedDeviceArray, tf.Tensor]
PyTreeDef = type(jax.tree_structure(None))
PartitionSpec = partitioning.PartitionSpec
DType = Union[np.dtype, type(jnp.bfloat16)]
Shape = Tuple[int, ...]

# TODO(adarob): Remove namespace mapping after client gin files are updated.
TensorBoardLogger = seqio.TensorBoardLogger

# -----------------------------------------------------------------------------
# Configurations
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class SaveCheckpointConfig:
  """Configuration for saving model checkpoints."""
  # The dtype to save ('float32' or 'bfloat16').
  dtype: str
  # Number of steps between writing checkpoints.
  period: Optional[int] = None
  # Number of most recent checkpoints to keep, or None to keep them all.
  keep: Optional[int] = None
  # Whether to save dataset checkpoints.
  save_dataset: bool = False
  # The checkpointer class to use.
  checkpointer_cls: Type[checkpoints.Checkpointer] = checkpoints.Checkpointer
  # Transformations to apply, in order, to the state before writing.
  state_transformation_fns: Sequence[checkpoints.SaveStateTransformationFn] = (
      dataclasses.field(default_factory=list))

  def __post_init__(self):
    if self.dtype not in ('float32', 'bfloat16'):
      raise ValueError(
          "`SaveCheckpointConfig.dtype` must be one of 'float32' or "
          f"'bfloat16'. Got {self.dtype}.")


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
  checkpointer_cls: Type[checkpoints.Checkpointer] = checkpoints.Checkpointer
  # Transformations to apply, in order, to the state after reading. These will
  # be applied after the `assignment_map` transformations.
  state_transformation_fns: Sequence[
      checkpoints.RestoreStateTransformationFn] = ()

  def __post_init__(self):
    if self.mode not in ('specific', 'latest', 'all'):
      raise ValueError(
          "`RestoreCheckpointConfig.mode` must be one of 'specific', 'latest', "
          f"or 'all'. Got {self.mode}.")
    if self.dtype not in (None, 'float32', 'bfloat16'):
      raise ValueError(
          "`RestoreCheckpointConfig.dtype` must be one of `None`, 'float32', "
          f"or 'bfloat16'. Got {self.dtype}.")
    if self.assignment_map is not None:
      # Turns `assignment_map` into a transformation function.
      assignment_map_fn = functools.partial(
          state_utils.apply_assignment_map, assignment_map=self.assignment_map)
      # Prepends the `assignment_map` transformation to the front of the list.
      self.state_transformation_fns = (assignment_map_fn,
                                       *self.state_transformation_fns)


@dataclasses.dataclass
class CheckpointConfig:
  """Configuration for checkpointing of model and dataset."""
  save: Optional[SaveCheckpointConfig] = None
  restore: Optional[RestoreCheckpointConfig] = None


@dataclasses.dataclass
class DatasetConfig:
  """Configuration for loading a dataset from a SeqIO Task or Mixture."""
  mixture_or_task_name: str
  task_feature_lengths: Mapping[str, int]
  split: str
  batch_size: int
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


#------------------------------------------------------------------------------
# Fast *nondeterministic* hardware RNG for faster Dropout
#------------------------------------------------------------------------------
def _hardware_uniform(
    rng_key: Array,
    shape: Shape,
    dtype: jnp.dtype = np.float32,
    minval: Array = np.float32(0),
    maxval: Array = np.float32(1)
) -> Array:
  """Random uniform method that uses non-deterministic accelerator hardware."""
  del rng_key  # non-deterministic prng.
  minval = jax.lax.convert_element_type(minval, dtype)
  maxval = jax.lax.convert_element_type(maxval, dtype)
  return jax.lax.rng_uniform(minval, maxval, shape)


# For dropout-only hardware rng.
def _hardware_bernoulli(
    rng_key: Array, p: np.ndarray = np.float32(0.5),
    shape: Shape = ()) -> Array:
  del rng_key  # non-deterministic prng.
  return jax.lax.rng_uniform(0.0, 1.0, shape) < p


def set_hardware_rng_ops():
  """Enable JAX Custom PRNG extension."""
  jax.config.update('jax_enable_custom_prng', True)
  # Use only fast TPU hardware PRNG with iterated-hash "split" substitute.
  # Expected to be deterministic for a fixed partitioning.
  # Monkey-patch JAX PRNGKey to use unsafe_rbg_prng_impl
  # TODO(levskaya): replace with jax global config option once we debug it.
  rbg_prng_key = functools.partial(prng.seed_with_impl,
                                   prng.unsafe_rbg_prng_impl)
  jax.random.PRNGKey = rbg_prng_key
  jax._src.random.PRNGKey = rbg_prng_key  # pylint: disable=protected-access


# -----------------------------------------------------------------------------
# Training utility functions.
# -----------------------------------------------------------------------------


def get_zeros_batch_like_spec(
    batch_spec: Mapping[str,
                        jax.ShapeDtypeStruct]) -> Mapping[str, jnp.ndarray]:
  return {k: jnp.zeros(t.shape, t.dtype) for k, t in batch_spec.items()}


def get_zeros_batch_like_dataset(dataset: tf.data.Dataset,
                                 batch_size=None) -> Mapping[str, jnp.ndarray]:
  reshape = lambda s: (batch_size,) + s[1:] if batch_size else tuple(s)
  batch_spec = {
      k: jax.ShapeDtypeStruct(reshape(t.shape), t.dtype.as_numpy_dtype)
      for k, t in dataset.element_spec.items()
  }
  return get_zeros_batch_like_spec(batch_spec)


class InitFnCallable(typing_extensions.Protocol):
  """A callable that initializes model variables."""

  def __call__(
      self, rng: Array, input_shapes: Mapping[str, Array],
      input_types: Optional[Mapping[str,
                                    DType]]) -> flax_scope.FrozenVariableDict:
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
    min_learning_rate: float = 1e-8) -> LearningRateCallable:
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
            step / warmup_steps, 1.0 + decay_factor * (warmup_steps - step))
      elif name == 'rsqrt_decay':
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'rsqrt_normalized_decay':
        ret *= jnp.sqrt(warmup_steps)
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'decay_every':
        ret *= (decay_factor**(step // steps_per_decay))
      elif name == 'cosine_decay':
        progress = jnp.maximum(0.0,
                               (step - warmup_steps) / float(steps_per_cycle))
        ret *= jnp.maximum(0.0,
                           0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))
      else:
        raise ValueError('Unknown factor %s.' % name)
    ret = jnp.maximum(ret, min_learning_rate)
    return jnp.asarray(ret, dtype=jnp.float32)

  return step_fn


# TODO(b/188897586): Add test coverage for the logic in this class.
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
  def __init__(self,
               optimizer_def: Optional[optim.OptimizerDef],
               init_fn: InitFnCallable,
               input_shapes: Mapping[str, Array],
               partitioner: partitioning.BasePartitioner,
               input_types: Optional[Mapping[str, DType]] = None):
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
          rng=rng, input_shapes=input_shapes, input_types=input_types)
      if optimizer_def:
        return train_state_lib.FlaxOptimTrainState.create(
            optimizer_def, initial_variables)
      return train_state_lib.InferenceState.create(initial_variables)

    self._partitioner = partitioner
    self.global_train_state_shape = jax.eval_shape(
        initialize_train_state, rng=jax.random.PRNGKey(0))
    self.train_state_axes = partitioner.get_mesh_axes(
        self.global_train_state_shape)
    self._initialize_train_state = initialize_train_state

    # Currently scanned layers require passing annotations through to the
    # point of the scan transformation to resolve an XLA SPMD issue.

    # init_fn is always(?) equal to model.get_initial_variables, fetch the model
    # instance from the bound method.
    model = init_fn.__self__  # pytype: disable=attribute-error
    if (hasattr(model, 'module') and hasattr(model.module, 'scan_layers') and
        model.module.scan_layers):
      if hasattr(model.module, 'spmd_annotations'):
        # update top-level module with spmd annotations.
        model.module = model.module.clone(
            parent=None, spmd_annotations=self.train_state_axes.params)

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
        out_axis_resources=self.train_state_axes)
    return p_initialize_train_state_fn(init_rng)

  def from_checkpoints(
      self,
      restore_cfgs: Sequence[RestoreCheckpointConfig],
      ds_iter: Optional[tf.data.Iterator] = None,
      init_rng: Optional[jnp.ndarray] = None,
  ) -> Iterable[train_state_lib.TrainState]:
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
    """

    def _restore_path(path, cfg):
      restore_checkpointer = cfg.checkpointer_cls(
          train_state=self.global_train_state_shape,
          partitioner=self._partitioner,
          checkpoints_dir='',  # unused for restore
          dataset_iterator=ds_iter if cfg.restore_dataset else None,
          restore_dtype=jnp.dtype(cfg.dtype) if cfg.dtype else None)

      from_tensorflow = gfile.exists(path + '.index')
      if from_tensorflow and cfg.state_transformation_fns:
        raise ValueError('Cannot initialize from a TensorFlow checkpoint using '
                         '`state_transformation_fns`.')
      if from_tensorflow:
        logging.info('Initializing parameters from TensorFlow checkpoint %s',
                     path)
        return restore_checkpointer.restore_from_tf_checkpoint(
            path, strict=cfg.strict)

      else:
        if cfg.fallback_to_scratch:
          if not cfg.state_transformation_fns:
            raise ValueError('`state_transformation_fns` must be provided with '
                             '`fallback_to_scratch`')
          if init_rng is None:
            raise ValueError('An `init_rng` must be provided with '
                             '`fallback_to_scratch`')
          fallback_state = self.from_scratch(init_rng).state_dict()
        else:
          fallback_state = None

        logging.info('Initializing parameters from specific T5X checkpoint %s',
                     path)
        return restore_checkpointer.restore(
            path=path,
            state_transformation_fns=cfg.state_transformation_fns,
            fallback_state=fallback_state)

    for restore_cfg in restore_cfgs:
      paths = ([restore_cfg.path]
               if isinstance(restore_cfg.path, str) else restore_cfg.path)
      if restore_cfg.mode == 'specific':
        logging.info('Restoring specific checkpoint(s): %s', paths)
        for path in paths:
          yield _restore_path(path, restore_cfg)
        return
      elif restore_cfg.mode in ('all', 'latest'):
        for ckpt_dir in paths:
          if not gfile.isdir(ckpt_dir):
            raise ValueError(
                'Checkpoint path(s) must be valid directories when using '
                "restore mode 'all' or 'latest'.")
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
            logging.info('No checkpoints found in specified directory: %s',
                         ckpt_dir)
            continue
          if restore_cfg.mode == 'latest':
            logging.info('Restoring latest T5X checkpoint.')
            ckpt_paths = ckpt_paths[-1:]
          logging.info('Restoring checkpoints for path(s): %s', ckpt_paths)
          for ckpt_path in ckpt_paths:
            yield _restore_path(ckpt_path, restore_cfg)
          return
      else:
        raise ValueError(
            f'Unsupported checkpoint restore mode: {restore_cfg.mode}')

  def from_checkpoint(
      self,
      ckpt_cfgs: Sequence[RestoreCheckpointConfig],
      *,
      ds_iter: Optional[tf.data.Iterator] = None,
      init_rng: Optional[jnp.ndarray] = None
  ) -> Optional[train_state_lib.TrainState]:
    """Restores (at most) 1 checkpoint using `from_checkpoints`, or dies."""
    train_states = list(
        self.from_checkpoints(ckpt_cfgs, ds_iter=ds_iter, init_rng=init_rng))
    if len(train_states) > 1:
      raise ValueError(
          f'Expected at most 1 checkpoint but got {len(train_states)} for '
          f'config(s): {ckpt_cfgs}')
    return (train_states[0]) if train_states else None

  def from_checkpoint_or_scratch(
      self,
      ckpt_cfgs: Sequence[RestoreCheckpointConfig],
      *,
      init_rng: Array,
      ds_iter: Optional[tf.data.Iterator] = None) -> train_state_lib.TrainState:
    """Initializes from checkpoint, if found, or from scratch."""
    return (self.from_checkpoint(ckpt_cfgs, ds_iter=ds_iter, init_rng=init_rng)
            or self.from_scratch(init_rng))


# -----------------------------------------------------------------------------
# Logging utility functions
# -----------------------------------------------------------------------------


def log_model_info(log_file: str, full_train_state: train_state_lib.TrainState,
                   partitioner: partitioning.BasePartitioner):
  """Log the variable shapes information and write to a file."""
  # Only write logs on host 0.
  if jax.process_index() != 0:
    return

  state_dict = full_train_state.state_dict()
  param_state_dict = state_dict['target']
  total_num_params = jax.tree_util.tree_reduce(
      np.add, jax.tree_map(np.size, param_state_dict))

  param_logical_axes = partitioner.get_logical_axes(
      full_train_state).state_dict()['target']

  param_mesh_axes = jax.tree_map(
      lambda x: tuple(x) if x is not None else None,
      partitioner.get_mesh_axes(full_train_state).state_dict()['target'])

  def _log_info_and_write_to_file(writer, format_str, *args):
    logging.info(format_str, *args)
    writer.write(format_str % args + '\n')

  with gfile.GFile(log_file, 'w') as writer:

    # Log params
    def _log_param(name: str, arr: np.ndarray,
                   logical_axes: Optional[partitioning.AxisNames],
                   mesh_axes: Optional[partitioning.PartitionSpec]):
      if logical_axes is None:
        shape_str = str(arr.shape)
      else:
        assert len(logical_axes) == len(arr.shape)
        shape_str = '({})'.format(', '.join(
            f'{name}={dimension}'
            for name, dimension in zip(logical_axes, arr.shape)))
      _log_info_and_write_to_file(
          writer, 'Variable %-80s size %-12s shape %-40s partition spec %s',
          name, arr.size, shape_str, mesh_axes)

    jax.tree_map(_log_param, state_utils.get_name_tree(param_state_dict),
                 param_state_dict, param_logical_axes, param_mesh_axes)

    _log_info_and_write_to_file(writer, 'Total number of parameters: %d',
                                total_num_params)

    # Add a blank line between params and states.
    _log_info_and_write_to_file(writer, '')

    # Log states
    def _log_state(name, arr):
      if arr is None:
        _log_info_and_write_to_file(writer, 'State    %-80s None', name)
      else:
        _log_info_and_write_to_file(writer,
                                    'State    %-80s size %-12s shape %s', name,
                                    arr.size, arr.shape)

    jax.tree_map(_log_state, state_utils.get_name_tree(state_dict['state']),
                 state_dict['state'])


# -----------------------------------------------------------------------------
# Utility functions for prediction and evaluation.
# -----------------------------------------------------------------------------


class InferStepCallable(typing_extensions.Protocol):

  def __call__(self, params: Mapping[str, Any],
               batch: Mapping[str, jnp.ndarray]) -> PyTreeDef:
    """Runs an inference step returning a prediction or score."""
    ...


def _deshard_and_remove_padding(all_inferences, all_indices):
  """Deshard and remove padded examples.

  Shape notation:

    Per replica set batch size: B
    Total number of batches: batch_count
    Number of replica sets: H

  Args:
    all_inferences: PyTree[B * batch_count, H, ...].
    all_indices: [B * batch_count, H].

  Returns:
    all_inferences in shape PyTree[total_examples, ...].
    all_indices in shape [total_exmamples].
  """
  # PyTree[batch_count * B, H, ...] -> PyTree[batch_count * B * H, ...]
  # batch_count * B * H is the total number of examples including padding
  # examples at the end if they exist.
  all_inferences = jax.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]),
                                all_inferences)

  # [batch_count * B, H] -> [batch_count * B * H]
  all_indices = all_indices.reshape(-1)

  # Remove padding.
  non_pad_idxs = np.where(all_indices >= 0)
  all_indices = all_indices[non_pad_idxs]
  all_inferences = jax.tree_map(lambda x: x[non_pad_idxs], all_inferences)
  return all_inferences, all_indices


def get_infer_fn(infer_step: InferStepCallable, batch_size: int,
                 train_state_axes: train_state_lib.TrainState,
                 partitioner: partitioning.BasePartitioner):
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
    batch_size: the global infer batch size.
    train_state_axes: Partitioning info for the train state object.
    partitioner: partitioner to use.

  Returns:
    predict_fn: a callable which takes in the enumerated infer dataset and an
      optimizer and runs the prediction.
  """
  infer_step = partitioner.partition(
      infer_step,
      in_axis_resources=(train_state_axes.params, PartitionSpec('data',)),
      out_axis_resources=PartitionSpec('data',))

  data_layout = partitioner.get_data_layout(batch_size)
  shard_id = data_layout.shard_id
  num_shards = data_layout.num_shards

  per_shard_batch_size = batch_size // num_shards

  def infer_fn(ds: tf.data.Dataset, train_state: train_state_lib.TrainState):
    ds_shapes = jax.tree_map(lambda x: jnp.array(x.shape), ds.element_spec)
    multihost_utils.assert_same(
        ds_shapes, 'Dataset element shapes do not agree across hosts. '
        'This could be an indication that the dataset is nondeterministic.')
    try:
      dataset_remainder = len(ds) % batch_size  # pytype:disable=wrong-arg-types
      logging.info('length of dataset = %s', len(ds))
    except TypeError as e:
      if str(e) == 'dataset length is unknown.':
        logging.warning(
            'The following error is likely due to the use of TensorFlow v1 in '
            'your dataset pipeline. Verify you are not importing from '
            '`tf.compat.v1` as part of your pipeline.')
      raise e

    if dataset_remainder:
      dataset_pad_amt = batch_size - dataset_remainder
      logging.info(
          'Padding infer dataset with %d examples for even per-replica shards.',
          dataset_pad_amt)
      # Pad with the first example using an index of -1 so seqio will ignore.
      pad_ds = ds.take(1).map(lambda i, x: (np.int64(-1), x)).repeat(
          dataset_pad_amt)
      ds = ds.concatenate(pad_ds)

    # Shard the infer dataset across replica sets.
    sharded_ds = ds.shard(num_shards, shard_id).batch(
        per_shard_batch_size, drop_remainder=True)
    multihost_utils.assert_same(
        jnp.array(len(sharded_ds)),
        'Dataset lengths do not agree across hosts.')

    logging.info(
        'The infer dataset is sharded into %d shards with per-shard '
        'batch size of %d', num_shards, per_shard_batch_size)

    # Run inference for each replica set.
    batched_results, all_indices = [], []
    for index, infer_batch in sharded_ds.as_numpy_iterator():
      # Run fast inference on batch.
      # [B, ...] -> [B, ...]
      batch_result = infer_step(train_state.params, infer_batch)
      logging.info('Inference of batch %s done.', index)
      # Issue asynchronous copy request which serves as prefetching to the host.
      # The result value is synchronized with host_allgather in the loop below.
      try:
        jax.tree_map(lambda x: x.copy_to_host_async(), batch_result)
      except AttributeError:
        # Similar to jax.device_get, we skip transfers for non DeviceArrays.
        pass
      batched_results.append(batch_result)
      all_indices.append(index)
    logging.info('Inference of all batches done.')
    all_inferences = []
    for batch_result in batched_results:
      # [B, ...] -> [H, B, ...]
      batch_result = multihost_utils.host_allgather(
          batch_result, num_shards, shard_id,
          data_layout.is_first_host_in_replica_set)
      all_inferences.append(batch_result)

    # List[H, B, ...] -> List[B, H, ...]
    all_inferences = jax.tree_map(lambda x: np.moveaxis(x, 0, 1),
                                  all_inferences)

    # List[B, H, ...] -> [B * batch_count, H, ...]
    all_inferences = jax.tree_multimap(lambda *args: np.concatenate(args),
                                       *all_inferences)
    # List[B] -> [B * batch_count]
    all_indices = np.concatenate(all_indices)
    # Collect all batches from across hosts.
    # [B * batch_count] -> [H, B * batch_count]
    all_indices = multihost_utils.host_allgather(
        all_indices, num_shards, shard_id,
        data_layout.is_first_host_in_replica_set)
    # [H, B * batch_count] -> [B * batch_count, H]
    all_indices = np.transpose(all_indices)
    all_inferences, all_indices = _deshard_and_remove_padding(
        all_inferences, all_indices)

    # Translate [B, ...] -> List[...] by flattening inferences making sure to
    # preserve structure of individual elements (inferences are not assumed to
    # be simple np.array). Finally, zip inferences with corresponding indices
    # and convert leaf np.arrays into lists.
    all_inferences, struct = jax.tree_flatten(all_inferences)
    all_inferences = map(
        functools.partial(jax.tree_unflatten, struct), zip(*all_inferences))
    indices_and_outputs = list(zip(all_indices, all_inferences))
    indices_and_outputs = jax.tree_map(lambda x: np.array(x).tolist(),
                                       indices_and_outputs)
    return indices_and_outputs

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
    if (str(e) ==
        'Attempted to add a new configurable after the config was locked.'):
      raise RuntimeError(
          'Your Task/Mixture module contains gin configurables that must be '
          'loaded before gin flag parsing. One fix is to add '
          f"'import {module}' in your gin file.")
    raise e


def get_vocabulary(
    cfg: DatasetConfig) -> Tuple[seqio.Vocabulary, seqio.Vocabulary]:
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
        'The use of `DatasetConfig.module` and `MIXTURE_OR_TASK_MODULE` is '
        'deprecated in favor of importing the module directly or via gin.',
        DeprecationWarning)
    import_module(cfg.module)

  provider = seqio.get_mixture_or_task(cfg.mixture_or_task_name)
  features = provider.output_features

  if 'inputs' in features and 'targets' in features:
    return (features['inputs'].vocabulary, features['targets'].vocabulary)

  # If a mix of PassThroughVocabularies and other Vocabularies are specified,
  # use the non-PassThroughVocabularies.
  # TODO(b/185912004): Remove this once a more general solution is implemented.
  vocabularies = list(
      f.vocabulary
      for f in features.values()
      if not isinstance(f.vocabulary, seqio.PassThroughVocabulary))

  # Otherwise, if all of the vocabs are PassThroughVocabularies, use those.
  if not vocabularies:
    vocabularies = list(f.vocabulary for f in features.values())

  # If there still aren't any vocabularies, raise an error.
  if not vocabularies:
    raise ValueError('"inputs" and "targets" are not both present, and '
                     'no vocabularies were set for any features.')

  first_vocab = vocabularies[0]
  for vocab in vocabularies[1:]:
    if vocab != first_vocab:
      raise ValueError('"inputs" and "targets" are not both present, and '
                       'vocabularies are different.')
  return (first_vocab, first_vocab)




def get_dataset(cfg: DatasetConfig,
                shard_id: int,
                num_shards: int,
                feature_converter_cls: Type[seqio.FeatureConverter],
                num_epochs: Optional[int] = None,
                continue_from_last_checkpoint: bool = False) -> tf.data.Dataset:
  """Returns a dataset from SeqIO based on a `DatasetConfig`."""
  if continue_from_last_checkpoint:
    raise ValueError(
        '`continue_from_last_checkpoint` must be set to False as this is not '
        'supported by this dataset fn.')
  del continue_from_last_checkpoint

  if cfg.module:
    import_module(cfg.module)

  if cfg.batch_size % num_shards:
    raise ValueError(
        f'Batch size ({cfg.batch_size}) must be divisible by number of '
        f'shards ({num_shards}).')


  shard_info = seqio.ShardInfo(index=shard_id, num_shards=num_shards)

  if cfg.seed is None:
    # Use a shared timestamp across devices as the seed.
    seed = multihost_utils.broadcast_one_to_all(np.int32(time.time()))
  else:
    seed = cfg.seed

  return get_dataset_inner(cfg, shard_info, feature_converter_cls, seed,
                           num_epochs)


def get_dataset_inner(cfg: DatasetConfig,
                      shard_info: seqio.ShardInfo,
                      feature_converter_cls: Type[seqio.FeatureConverter],
                      seed: Optional[int] = None,
                      num_epochs: Optional[int] = None):
  """Internal fn to load a dataset from SeqIO based on a `DatasetConfig`."""
  batch_size = cfg.batch_size // shard_info.num_shards
  if seed is not None:
    multihost_utils.assert_same(
        np.array(seed),
        f'`seed` is not same across hosts; {jax.process_index} has a seed of '
        f'{seed}')
    logging.info(
        "Initializing dataset for task '%s' with a replica batch size of %d and "
        'a seed of %d', cfg.mixture_or_task_name, batch_size, seed)

  ds = seqio.get_dataset(
      mixture_or_task_name=cfg.mixture_or_task_name,
      task_feature_lengths=cfg.task_feature_lengths,
      dataset_split=cfg.split,
      shuffle=cfg.shuffle,
      num_epochs=num_epochs,
      feature_converter=feature_converter_cls(
          pack=cfg.pack, use_custom_packing_ops=cfg.use_custom_packing_ops),  # pytype: disable=not-instantiable
      shard_info=shard_info,
      use_cached=cfg.use_cached,
      seed=seed)
  ds = ds.batch(batch_size, drop_remainder=True)
  return ds


class GetDatasetCallable(typing_extensions.Protocol):

  def __call__(self,
               cfg: DatasetConfig,
               shard_id: int,
               num_shards: int,
               feature_converter_cls: Callable[..., seqio.FeatureConverter],
               num_epochs: Optional[int] = None,
               continue_from_last_checkpoint: bool = True) -> tf.data.Dataset:
    ...


def get_training_eval_datasets(
    cfg: DatasetConfig,
    shard_id: int,
    num_shards: int,
    eval_steps: int,
    feature_converter_cls: Callable[..., seqio.FeatureConverter],
    get_dataset_fn: GetDatasetCallable = get_dataset,
) -> Mapping[str, tf.data.Dataset]:
  """Returns a mapping from eval task name to its dataset."""
  mixture_or_task = seqio.get_mixture_or_task(cfg.mixture_or_task_name)
  datasets = {}
  for task in seqio.get_subtasks(mixture_or_task):
    if cfg.split not in task.splits:
      logging.info("Task %s has no '%s' split; skipping training evaluation.",
                   task.name, cfg.split)
      continue
    logging.info('Loading task %s for training evaluation.', task.name)
    task_cfg = dataclasses.replace(cfg, mixture_or_task_name=task.name)
    # We set `num_epochs=eval_steps` to avoid infinite loops on shards that have
    # input examples but are filtered to be empty.
    datasets[task.name] = get_dataset_fn(
        task_cfg,
        shard_id,
        num_shards,
        feature_converter_cls,
        num_epochs=eval_steps,
        continue_from_last_checkpoint=False).repeat().take(eval_steps)

  if isinstance(mixture_or_task, seqio.Mixture):
    datasets[mixture_or_task.name] = get_dataset_fn(
        cfg,
        shard_id,
        num_shards,
        feature_converter_cls,
        num_epochs=eval_steps,
        continue_from_last_checkpoint=False).repeat().take(eval_steps)

  for task_name, ds in datasets.items():
    try:
      next(iter(ds))
    except StopIteration:
      raise ValueError(
          f"Shard {shard_id}/{num_shards} of task '{task_name}:{cfg.split}' "
          'is empty. Try resharding your dataset for more even splits or '
          'reducing data parallelism.')

  return datasets


def round_vocab_size_to_multiple(vocabulary: seqio.Vocabulary,
                                 divisor: int = 128):
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
    params_axes_names_override: Sequence[Tuple[str, Tuple[str, ...]]] = ()
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
        'override to.')
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
      logging.info('Replacing axis names for %s (%s) with %s.', param_name,
                   curr_metadata.names, override)

    if param.ndim != len(override):
      raise ValueError(
          f'Provided axis name override for {param_name} does not match '
          f'param rank ({param.ndim}): {override}')
    flat_params_axes[param_axes_key] = flax_partitioning.AxisMetadata(
        names=override)

  model_variables['params_axes'] = traverse_util.unflatten_dict(
      flat_params_axes)
  return flax.core.freeze(model_variables)


