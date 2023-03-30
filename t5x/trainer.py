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

"""Trainer and MetricsManager classes for use in train loop.

To create a custom trainer, subclass `BaseTrainer` and implement
`_partitioned_train_step` and `_partitioned_eval_step` methods,
possibly by re-using the utility functions provided in this module.
"""
import abc
import enum
import os
import threading
import time
from typing import Any, Dict, Iterator, Mapping, MutableMapping, Optional, Protocol, Sequence, TYPE_CHECKING, Tuple, Union

from absl import logging
import cached_property
from clu import asynclib
from clu import metric_writers
import clu.data
import clu.metrics
import clu.values
from flax.core import FrozenDict
import jax
from jax.experimental import multihost_utils
import jax.lax
import jax.numpy as jnp
import jax.random
import numpy as np
from t5x import metrics as metrics_lib
from t5x import models
from t5x import partitioning
from t5x import train_state as train_state_lib
from t5x import utils
import typing_extensions


Array = Union[np.ndarray, jnp.ndarray]
BatchSpec = Mapping[str, jax.ShapeDtypeStruct]
BatchType = Mapping[str, np.ndarray]
FlaxMutables = FrozenDict
Rng = jnp.ndarray
MetricMapType = MutableMapping[str, clu.metrics.Metric]
MetricMapSpec = Mapping[str, jax.ShapeDtypeStruct]
MetricValueMapType = Mapping[str, clu.values.Value]
ModelWeights = Any
MutableMetricMapType = Dict[str, clu.metrics.Metric]
PyTree = Any
PartitionSpec = partitioning.PartitionSpec

if TYPE_CHECKING:  # See b/163639353
  cached_property = property  # pylint: disable=invalid-name
else:
  cached_property = cached_property.cached_property


@jax.jit
def _merge_metrics(a, b):
  return jax.tree_util.tree_map(
      lambda a, b: a.merge(b), a, b, is_leaf=metrics_lib.is_metric_obj)


# Merges two metrics pytrees (mapping of metric_name (str) to clu.Metric object)
def merge_metrics(a, b):
  a, b = jax.tree_util.tree_map(utils.get_local_data, (a, b))
  return _merge_metrics(a, b)


class ArrayMapFuture(typing_extensions.Protocol):

  def result(self) -> Mapping[str, Array]:
    ...


class MetricValueMapFuture(typing_extensions.Protocol):

  def result(self) -> Mapping[str, clu.values.Value]:
    ...


class TimeFuture(typing_extensions.Protocol):

  def result(self) -> float:
    ...


class LearningRateCallable(typing_extensions.Protocol):

  def __call__(
      self,
      step: jnp.ndarray,
  ) -> jnp.ndarray:
    ...


class SummarizeMetricsCallable(typing_extensions.Protocol):
  """PyType template for a metrics summary function."""

  def __call__(self, metrics: MetricMapType, duration: float,
               num_steps: int) -> Mapping[str, jnp.ndarray]:
    """Summarizes metrics accumulated across multiple steps.

    Args:
      metrics: Metrics accumulated across multiple steps.
      duration: The duration of the run being summarized.
      num_steps: The number of steps the metrics are accumulated across.

    Returns:
      Summarized metrics.
    """
    ...


class PartitionedTrainCallable(typing_extensions.Protocol):
  """Protocol for a partitioned train step."""

  def __call__(
      self, train_state: train_state_lib.TrainState,
      batch: BatchType) -> Tuple[train_state_lib.TrainState, MetricMapType]:
    ...


class PartitionedEvalCallable(typing_extensions.Protocol):
  """Protocol for a partitioned eval step."""

  def __call__(self, train_state: train_state_lib.TrainState,
               batch: jnp.ndarray) -> MetricMapType:
    ...


class WeightMetricsComputer(object):
  """Decides which weight metrics to compute during training."""

  _WEIGHT_METRICS = [
      "weight_rms", "weight_gradient_rms", "weight_update_rms", "weight_max"
  ]

  @staticmethod
  def _make_rms_metrics(name, tree):
    """Calculates the root-mean-square metric for a pytree."""
    return {
        f"{name}/{k}": metrics_lib.AveragePerStep.from_model_output(
            jnp.sqrt(jnp.mean(jnp.square(v))))
        for k, v in utils.flatten_dict_string_keys(tree).items()
    }

  @staticmethod
  def _make_max_metrics(name, tree):
    """Calculates the L-inf norm for a pytree."""
    return {
        f"{name}/{k}":
        metrics_lib.AveragePerStep.from_model_output(jnp.max(jnp.abs(v)))
        for k, v in utils.flatten_dict_string_keys(tree).items()
    }

  def compute_metrics(
      self, gradients: ModelWeights,
      old_train_state: train_state_lib.TrainState,
      new_train_state: train_state_lib.TrainState) -> MutableMetricMapType:
    """Compute some metrics about weights after having updating these weights.

    Args:
      gradients: The gradients of all weights.
      old_train_state: The training state before applying the gradients.
      new_train_state: The training state after applying the gradients.

    Returns:
      A dictionary of Metrics, where the keys are either metric names, or of the
      form metric_name/parameter_name, depending on whether or not they are
      global to the model, or specific to each model parameter.
    """
    # TODO(reinerp): Extend weight stats logging with support for non-reduced
    # axes of tensors. For example, for stacked layers (QKV stacking or layer
    # stacking), we might not want to reduce over the stacking dimension, in
    # order to provide more localization in the logged stats.
    metrics = {}
    metrics.update(self._make_rms_metrics("weight_rms", new_train_state.params))
    metrics.update(self._make_rms_metrics("weight_gradient_rms", gradients))
    grad_norm = jnp.sqrt(
        jnp.sum(
            jnp.array([
                jnp.vdot(x, x) for x in jax.tree_util.tree_leaves(gradients)
            ])))
    metrics.update({
        "weight_gradient_norm":
            metrics_lib.AveragePerStep.from_model_output(grad_norm)
    })
    weight_update = jax.tree_util.tree_map(jnp.subtract, new_train_state.params,
                                           old_train_state.params)
    metrics.update(self._make_rms_metrics("weight_update_rms", weight_update))
    weight_update_by_weight = jax.tree_util.tree_map(jnp.divide, weight_update,
                                                     old_train_state.params)
    metrics.update(
        self._make_rms_metrics("weight_update_divided_by_weight_rms",
                               weight_update_by_weight))
    metrics.update(self._make_max_metrics("weight_max", new_train_state.params))

    return metrics


class _AsyncTimer(object):
  """A timer that computes computes durations between async jax operations.

  You should call close() to wait for threads started by this class to finish.
  """

  def __init__(self):
    # We use a thread pool with a single worker to ensure that calls to the
    # function are run in order (but in a background thread).
    self._pool = asynclib.Pool(thread_name_prefix="AsyncTimer", max_workers=1)
    self._start_future = None

  def close(self):
    self._pool.close()

  def __del__(self):
    self.close()

  def _get_completion_future(self, block_on: PyTree = ()) -> TimeFuture:
    """Returns Future containing time when `block_on` is ready."""

    def _get_completion_time():
      try:
        jax.block_until_ready(block_on)
      except RuntimeError as e:
        # If the buffer no longer exists, we assume it was completed.
        buffer_deleted_message = ("INVALID_ARGUMENT: BlockHostUntilReady() "
                                  "called on deleted or donated buffer")
        gda_buffer_deleted_message = ("INVALID_ARGUMENT: GetReadyFuture() "
                                      "called on deleted or donated buffer")
        if str(e) not in (buffer_deleted_message, gda_buffer_deleted_message):
          raise
      return time.time()

    return self._pool(_get_completion_time)()

  def start(self, block_on: PyTree = ()):
    """Starts timer after `block_on` is ready."""
    self._start_future = self._get_completion_future(block_on)

  def stop(self, block_on: PyTree = ()) -> TimeFuture:
    """Stops timer after `block_on` is ready, returning the duration."""
    if not self._start_future:
      raise ValueError("The timer hasn't been started.")

    start_future = self._start_future
    self._start_future = None
    stop_future = self._get_completion_future(block_on)
    return self._pool(lambda: stop_future.result() - start_future.result())()


class MetricsManager(object):
  """Manages a set of distributed metrics and their logging.

  Logging is disabled on all but host 0.

  Logs to:
    * TensorBoard
    * ABSL

  You should call close() to wait for threads started by this class to finish.
  """

  def __init__(
      self,
      name: str,
      summary_dir: Optional[str] = None,
  ):
    """MetricsManager constructor.

    Constructs an empty MetricWriter on all but host 0.

    Args:
      name: an identifier of the metrics to use when logging (e.g., 'train').
      summary_dir: the summary directory. If provided, TensorBoard summaries
        will be written to a `name` subdirectory.
    """
    self._name = name
    if jax.process_index() == 0:
      self._writer = metric_writers.create_default_writer(
          summary_dir,
          collection=name,
          asynchronous=True,
      )
    else:
      self._writer = metric_writers.MultiWriter([])
    self.summary_dir = os.path.join(summary_dir, name) if summary_dir else None
    self._writer_lock = threading.Lock()
    # We use a thread pool with a single worker to ensure that calls to the
    # function are run in order (but in a background thread).
    self._summary_pool = asynclib.Pool(
        thread_name_prefix="MetricsManager", max_workers=1)
    # Times the duration between steps.
    self._duration_timer = _AsyncTimer()

  def __del__(self):
    self.close()

  def close(self):
    try:
      self._summary_pool.close()
    finally:
      try:
        self._duration_timer.close()
      finally:
        if self._writer:
          self._writer.close()
          self._writer = None

  @property
  def summary_writer(self) -> metric_writers.MetricWriter:
    """Returns the MetricWriter used by this class."""
    # TODO(adarob): Make returned writer threadsafe.
    return self._writer

  def write_scalar(self, key: str, val: metric_writers.interface.Scalar,
                   step: int):
    """Writes scalar value to metric writers in a threadsafe manner."""
    step = int(utils.get_local_data(step))
    self.write_scalars(step, {key: val})

  def write_scalars(self, step: int,
                    scalars: Mapping[str, metric_writers.interface.Scalar]):
    """Writes scalar value to metric writers in a threadsafe manner."""
    step = utils.get_local_data(step)
    with self._writer_lock:
      self._writer.write_scalars(step, scalars)

  def start_duration_timer(self, block_on: PyTree = ()):
    """Starts the duration timer."""
    self._duration_timer.start(block_on=block_on)

  def write_metrics_summary(self, metrics: MetricMapType, step: int,
                            num_steps: int) -> MetricValueMapFuture:
    """Writes summary based on accumulated metrics in a background thread.

    Duration is automatically computed as the interval between completion of
    metrics fetching. This closely approximates the duration of `num_steps`,
    as the steps must be computes sequentually, and it is more accurate than
    computing the time since the call to the step function since its actual
    execution occurs asynchronously on the TPU/GPU device.

    Args:
      metrics: acculumated metric values.
      step: the current train step.
      num_steps: the number of steps the metrics are accumulated across.

    Returns:
      A mapping of name -> scalar value of the written summary. Only return the
        real scalar value on host 0. For other hosts, return None.
    """
    step = utils.get_local_data(step)

    # Must be called in the main thread to avoid race condition.
    duration_future = self._duration_timer.stop(block_on=metrics)

    def _summarize_and_write():
      # For thread safety we first copy the metrics to host.
      fetched_metrics = jax.tree_util.tree_map(jax.device_get, metrics)

      duration = duration_future.result()
      # We set the duration on time-related metrics.
      final_metrics = metrics_lib.set_time_metrics_duration(
          fetched_metrics, duration)
      # Set num_steps for Step metrics (AveragePerStep, StepsPerTime, ...)
      final_metrics = metrics_lib.set_step_metrics_num_steps(
          final_metrics, num_steps)

      # Ensure the metrics are not on device, which could lead to a deadlock.
      def _ensure_not_on_device(x):
        assert not isinstance(x, jax.Array)

      jax.tree_util.tree_map(_ensure_not_on_device, final_metrics)
      final_metrics = jax.tree_util.tree_map(utils.get_local_data,
                                             final_metrics)

      summary = {k: v.compute_value() for k, v in final_metrics.items()}
      with self._writer_lock:
        metric_writers.write_values(self._writer, int(step), summary)

      return summary

    return self._summary_pool(_summarize_and_write)()

  def flush(self):
    try:
      self._summary_pool.join()
    finally:
      self._writer.flush()


class PreemptionError(Exception):
  """Training has been interrupted and needs an emergency checkpoint."""


class BaseTrainer(abc.ABC):
  """Abstract base trainer class.

  Internally this uses MetricsManagers that start threads. You should
  use the trainer as a context manager, or call close() directly in
  order to wait for these threads to finish after training is done.
  """

  def __init__(self, model: models.BaseModel,
               train_state: train_state_lib.TrainState,
               partitioner: partitioning.BasePartitioner,
               eval_names: Sequence[str], summary_dir: Optional[str],
               train_state_axes: Any, rng: Rng):
    """Trainer constructor.

    Args:
      model: the instantiation of `BaseModel` to train.
      train_state: A train state with model parameters and optimizer state.
      partitioner: the partitioner to use.
      eval_names: names of evaluation datasets, which must match the keys of the
        mapping passed to `eval`.
      summary_dir: optional directory to write TensorBoard metrics to.
      train_state_axes: partitioning info for the train state to be used.
      rng: jax PRNGKey seed for random operations, to be combined with step
        number for a deterministic RNG.
    """
    self._model = model
    self._train_state_axes = train_state_axes
    self._base_rng = rng
    self._partitioner = partitioner
    self._compiled_train_step: Optional[PartitionedTrainCallable] = None
    self._compiled_eval_steps: MutableMapping[str, PartitionedEvalCallable] = {}
    self._compiled_eval_step_cache: MutableMapping[
        BatchSpec, PartitionedEvalCallable] = {}

    self._train_state_mutex = threading.RLock()
    self._train_state = train_state

    self.stop_training = False

    # Time since the trainer was made, this will record the "uptime" of the job.
    self._trainer_init_time = time.time()

    # The training metrics combine metrics added by the Model (e.g., loss and
    # accuracy) and Trainer (e.g., learning rate).
    self.train_metrics_manager = MetricsManager(
        "train", summary_dir=summary_dir)

    # The eval metrics only include metrics added by the Model.
    self.eval_metrics_managers = {  # pylint:disable=g-complex-comprehension
        n: MetricsManager(f"training_eval/{n[:113]}", summary_dir=summary_dir)
        for n in eval_names
    }

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()

  def close(self):
    """Stops all train metric managers threads."""
    self.train_metrics_manager.close()
    for mm in self.eval_metrics_managers.values():
      mm.close()

  def _get_step_rng(self, step: int) -> Rng:
    return jax.random.fold_in(self._base_rng, step)

  @property
  def train_state(self):
    with self._train_state_mutex:
      return self._train_state

  @train_state.setter
  def train_state(self, train_state: PyTree):
    with self._train_state_mutex:
      self._train_state = train_state

  def train(self,
            batch_iter: Union[Iterator[BatchType],
                              clu.data.dataset_iterator.DatasetIterator],
            num_steps: int,
            start_step: Optional[int] = None) -> ArrayMapFuture:
    """Runs the train loop for the given number of steps."""
    metrics = None
    # Use pre-compiled step, if available.
    train_step_fn = self._compiled_train_step or self._partitioned_train_step

    # We lock `train_state` access during the loop to avoid race conditions.
    with self._train_state_mutex:
      train_state = self.train_state
      # Compute step number on host to avoid communication overhead.
      start_step = int(
          start_step if start_step is not None else train_state.step)
      self.train_metrics_manager.start_duration_timer(block_on=train_state)
      for step_num in range(start_step, start_step + num_steps):
        logging.log_every_n_seconds(logging.INFO, "Training: step %d", 10,
                                    step_num)
        with jax.profiler.StepTraceAnnotation("train", step_num=step_num):
          batch = next(batch_iter)
          train_state, metrics_update = train_step_fn(train_state, batch)
          if metrics:
            metrics = merge_metrics(metrics, metrics_update)
          else:
            metrics = metrics_update

      self.train_state = train_state

    if metrics is not None:
      metrics["timing/uptime"] = clu.metrics.LastValue.from_model_output(
          jnp.asarray([time.time() - self._trainer_init_time]))

    return self.train_metrics_manager.write_metrics_summary(
        metrics, start_step + num_steps, num_steps)

  def compile_train(self, batch: BatchType) -> None:
    """Pre-compiles train step (if not yet compiled).

    Not required.

    If not called before `train`, compilation will occur automatically on the
    first step and JAX's "jit cache" will be used to avoid recompilation for
    future steps.

    Args:
      batch: A sample batch that may contain dummy values, but with correct
        shapes and dtypes.
    """
    tick = time.time()
    self._compiled_train_step = self._partitioner.compile(
        self._partitioned_train_step, self.train_state, batch)
    tock = time.time()
    self.train_metrics_manager.write_scalar("timing/compilation_seconds",  # pytype: disable=wrong-arg-types  # jax-ndarray
                                            tock - tick, self.train_state.step)

  def eval(
      self, batch_iters: Mapping[str,
                                 Iterator[BatchType]]) -> Mapping[str, Array]:
    """Runs evaluation loop over the iterator and writes summary."""
    eval_summaries = {}
    train_state = self.train_state
    for iter_name, batch_iter in batch_iters.items():
      logging.info("Evaluating: %s.", iter_name)
      metrics = None
      # Use a pre-compiled step function, if available.
      eval_step_fn = self._compiled_eval_steps.get(iter_name,
                                                   self._partitioned_eval_step)
      mm = self.eval_metrics_managers[iter_name]

      num_steps = 0
      mm.start_duration_timer(block_on=train_state)
      for batch in batch_iter:
        num_steps += 1
        utils.multihost_assert_equal(
            jnp.array(num_steps),
            "Eval step mismatch across hosts. Check for empty dataset shard.")
        if jax.process_count() > 1:
          batch = multihost_utils.host_local_array_to_global_array(
              batch, self._partitioner.mesh,
              self._partitioner.data_partition_spec)
        metrics_update = eval_step_fn(train_state, batch)
        if metrics:
          metrics = merge_metrics(metrics, metrics_update)
        else:
          metrics = metrics_update
      utils.multihost_assert_equal(
          jnp.array(-1),
          "Eval step mismatch across hosts. Check for empty dataset shard.")

      eval_summaries[iter_name] = mm.write_metrics_summary(  # pytype: disable=wrong-arg-types  # jax-ndarray
          metrics, train_state.step, num_steps)

    # TODO(adarob): Return futures.
    return {k: v.result() for k, v in eval_summaries.items()}

  def compile_eval(self, batches: Mapping[str, BatchType]) -> None:
    """Pre-compiles eval step (if not yet compiled).

    Not required.

    Pre-compiles the evaluation step for each evaluation dataset, reusing cached
    compilations where possible. In other words, if multiple evaluation datasets
    have equivalent shapes/dtypes for the batch and initial metrics,
    recompilation will be avoided.

    If not called before `eval`, compilation will occur automatically on the
    first step and JAX's "jit cache" will be used to avoid recompilation for
    future steps.

    Args:
      batches: a mapping from evaluation dataset name to a sample batch. The
        batch may contain dummy values, but the shapes and dtypes must be
        correct.
    """
    for eval_name, batch in batches.items():
      tick = time.time()
      cache_key: BatchSpec = FrozenDict(jax.eval_shape(lambda: batch))  # pylint:disable=cell-var-from-loop
      if cache_key not in self._compiled_eval_step_cache:
        if jax.process_count() > 1:
          batch = multihost_utils.host_local_array_to_global_array(
              batch, self._partitioner.mesh,
              self._partitioner.data_partition_spec)
        self._compiled_eval_step_cache[cache_key] = self._partitioner.compile(
            self._partitioned_eval_step, self.train_state, batch)
      self._compiled_eval_steps[eval_name] = self._compiled_eval_step_cache[
          cache_key]
      tock = time.time()
      self.eval_metrics_managers[eval_name].write_scalar(  # pytype: disable=wrong-arg-types  # jax-ndarray
          "timing/compilation_seconds", tock - tick, self.train_state.step)

  @property
  @abc.abstractmethod
  def _partitioned_train_step(self) -> PartitionedTrainCallable:
    """Partitioned train step."""
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def _partitioned_eval_step(self) -> PartitionedEvalCallable:
    """Partitioned eval step."""
    raise NotImplementedError


def accumulate_grads_microbatched(
    model: models.BaseModel,
    train_state: train_state_lib.TrainState,
    batch: BatchType,
    dropout_rng: Rng,
    num_microbatches: Optional[int],
    data_partition_spec: PartitionSpec = PartitionSpec("data"),
) -> Tuple[train_state_lib.TrainState, MutableMetricMapType,
           Optional[FlaxMutables]]:
  """Implements optional microbatched gradient accumulation.

  Args:
    model: the instantiation of `BaseModel` to train.
    train_state: A train state with model parameters and optimizer state.
    batch: input batch consisting of either - simply-padded batched features
      'encoder_input_tokens', 'decoder_input_tokens' 'decoder_target_tokens'
      'decoder_loss_weights'- packed, batched features with additional
      "(encoder|decoder)_segment_id", "(encoder|decoder)_position"
    dropout_rng: jax PRNGKey for dropout.
    num_microbatches: the number of microbatches to use, or None for direct
      training.
    data_partition_spec: the PartitionSpec to use for partitioning annotations
      on the batch.

  Returns:
   Accumulated gradients and incremental metrics.
  """
  batch_size = next(iter(batch.values())).shape[0]

  grad_fn = jax.value_and_grad(model.loss_fn, has_aux=True)

  # We assume that the model loss_fn supports flax mutables if and only if
  # the train state includes non-empty flax mutables.
  # Note: Default t5x models don't support flax_mutables. One needs to subclass
  # them and return flax_mutables from `get_initial_variables` and `loss_fn`.

  initial_flax_mutables = train_state.flax_mutables if train_state.flax_mutables else None

  if num_microbatches is None or num_microbatches <= 1:

    if initial_flax_mutables is None:
      (_, metrics), grad_accum = grad_fn(train_state.params, batch, dropout_rng)
      flax_mutables = None
    else:
      (_, (metrics,
           flax_mutables)), grad_accum = grad_fn(train_state.params, batch,
                                                 dropout_rng,
                                                 initial_flax_mutables)
  else:
    assert batch_size % num_microbatches == 0, (
        "Batch size isn't divided evenly by num_microbatches.")
    microbatch_size = batch_size // num_microbatches
    logging.info("using microbatches: %d microbatches, %d size",
                 num_microbatches, microbatch_size)

    def get_microbatch(batch: BatchType, idx: int) -> Mapping[str, jnp.ndarray]:
      """Fetch microbatch slice from possibly-packed input data."""
      offset = idx * microbatch_size
      length = microbatch_size
      starts = {k: [offset] + [0] * (b.ndim - 1) for k, b in batch.items()}
      limits = {k: [length] + list(b.shape[1:]) for k, b in batch.items()}
      return {
          k: jax.lax.dynamic_slice(b, starts[k], limits[k])
          for k, b in batch.items()
      }

    def metrics_and_grad(loop_cnt, dropout_rng, flax_mutables=None):
      dropout_rng, sub_dropout_rng = jax.random.split(dropout_rng)
      mbatch = get_microbatch(batch, loop_cnt)
      # We need to annotate the microbatch sharding as we would a batch.
      mbatch = jax.tree_util.tree_map(
          lambda x: partitioning.with_sharding_constraint(  # pylint: disable=g-long-lambda
              x, data_partition_spec),
          mbatch)
      if flax_mutables is None:
        (_, metrics), grad = grad_fn(train_state.params, mbatch,
                                     sub_dropout_rng)
      else:
        (_, (metrics, flax_mutables)), grad = grad_fn(train_state.params,
                                                      mbatch, sub_dropout_rng,
                                                      flax_mutables)
      return metrics, grad, flax_mutables

    def per_microbatch_train_step(
        loop_cnt: int, state: Tuple[jnp.ndarray, jnp.ndarray,
                                    Mapping[str, jnp.ndarray],
                                    Optional[FlaxMutables]]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Mapping[str, jnp.ndarray],
               Optional[FlaxMutables]]:
      (dropout_rng, grad_accum, prev_metrics, flax_mutables) = state
      metrics, grad, flax_mutables = metrics_and_grad(loop_cnt, dropout_rng,
                                                      flax_mutables)

      grad_accum = jax.tree_util.tree_map(jnp.add, grad_accum, grad)
      metrics = jax.lax.cond(loop_cnt == 0, lambda _: metrics,
                             lambda _: merge_metrics(prev_metrics, metrics),
                             None)
      return dropout_rng, grad_accum, metrics, flax_mutables

    # Initialize gradient accumulation loop state.
    accum_dtype = jnp.float32
    grad_accum_init = jax.tree_util.tree_map(
        lambda x: jnp.zeros(x.shape, accum_dtype), train_state.params)
    initial_metrics_shape, _, _ = jax.eval_shape(
        metrics_and_grad,
        loop_cnt=0,
        dropout_rng=dropout_rng,
        flax_mutables=initial_flax_mutables,
    )

    initial_metrics = {
        k: metrics_lib.shape_obj_to_defined_obj(v)
        for k, v in initial_metrics_shape.items()
    }
    loop_init = (dropout_rng, grad_accum_init, initial_metrics,
                 initial_flax_mutables)
    new_dropout_rng, grad_accum, metrics, flax_mutables = jax.lax.fori_loop(
        0, num_microbatches, per_microbatch_train_step, loop_init)

    del new_dropout_rng

  return grad_accum, metrics, flax_mutables


def apply_grads(
    train_state: train_state_lib.TrainState,
    grad_accum: ModelWeights,
    metrics: MutableMetricMapType,
    learning_rate: jnp.ndarray,
    weight_metrics_computer: Optional[WeightMetricsComputer],
    other_state_variables: Optional[Mapping[str, Any]] = None
) -> Tuple[train_state_lib.TrainState, MetricMapType]:
  """Applies gradients to the optimizer.

  Args:
    train_state: A train state that contains model and optimizer params.
    grad_accum: results of `accumulate_grads` call.
    metrics: incremental metrics from `accumulate_grads` call.
    learning_rate: the learning rate to use for this step.
    weight_metrics_computer: A WeightMetricsComputer instance, or None, to
      decide what metrics, if any, to log about weights and weight updates
      during training.
    other_state_variables: other variables to update the state with.

  Returns:
   The updated train state, metrics.
  """
  if other_state_variables is None:
    other_state_variables = {}
  # Update optimizer using accumulated gradient.
  new_train_state = train_state.apply_gradient(
      grad_accum, learning_rate=learning_rate, **other_state_variables)
  metrics["learning_rate"] = clu.metrics.Average.from_model_output(
      jnp.asarray([learning_rate]))
  metrics["learning_rate/current"] = clu.metrics.LastValue.from_model_output(
      jnp.asarray([learning_rate]))
  if weight_metrics_computer is not None:
    metrics.update(
        weight_metrics_computer.compute_metrics(grad_accum, train_state,
                                                new_train_state))
  return new_train_state, metrics


def eval_step(model: models.BaseModel, train_state: train_state_lib.TrainState,
              batch: jnp.ndarray) -> MetricMapType:
  """Default evaluation step."""
  _, metrics = model.eval_fn(train_state.params, batch)  # pytype: disable=wrong-arg-types  # jax-ndarray
  return metrics


def train_with_lr(
    train_state: train_state_lib.TrainState,
    batch: BatchType,
    learning_rate: jnp.ndarray,
    dropout_rng: Rng,
    model: models.BaseModel,
    num_microbatches: Optional[int],
    weight_metrics_computer: Optional[WeightMetricsComputer] = None,
    data_partition_spec: PartitionSpec = PartitionSpec("data")):
  """Main training function with LR schedule."""
  grad_accum, metrics, flax_mutables = (
      accumulate_grads_microbatched(model, train_state, batch, dropout_rng,
                                    num_microbatches, data_partition_spec))
  new_train_state, metrics = apply_grads(
      train_state,
      grad_accum,
      metrics,
      learning_rate,
      weight_metrics_computer,
      other_state_variables={"flax_mutables": flax_mutables}
      if flax_mutables else None)

  return new_train_state, metrics


class BaseTrainerConstructor(Protocol):
  """A function that returns a BaseTrainer."""

  def __call__(self, model: models.BaseModel,
               train_state: train_state_lib.TrainState,
               partitioner: partitioning.BasePartitioner,
               eval_names: Sequence[str], summary_dir: Optional[str],
               train_state_axes: Any, rng: Rng) -> BaseTrainer:
    ...


class Trainer(BaseTrainer):
  """Training loop with optional microbatches."""

  def __init__(self,
               model: models.BaseModel,
               train_state: train_state_lib.TrainState,
               partitioner: partitioning.BasePartitioner,
               eval_names: Sequence[str],
               summary_dir: Optional[str],
               train_state_axes: Any,
               rng: Rng,
               learning_rate_fn: LearningRateCallable,
               num_microbatches: Optional[int],
               weight_metrics_computer: Optional[WeightMetricsComputer] = None):
    """Trainer constructor.

    Args:
      model: the instantiation of `BaseModel` to train.
      train_state: a train state with parameters and optimizer state.
      partitioner: the partitioner to use.
      eval_names: names of evaluation datasets, which must match the keys of the
        mapping passed to `eval`.
      summary_dir: optional directory to write TensorBoard metrics to.
      train_state_axes: partitioning info for the optimizer to be used.
      rng: jax PRNGKey seed for random operations, to be combined with step
        number for a deterministic RNG.
      learning_rate_fn: returns the learning rate given the current step.
      num_microbatches: the number of microbatches to use, or None for direct
        training.
      weight_metrics_computer: A WeightMetricsComputer instance, or None, to
        decide what metrics, if any, to log about weights and weight updates
        during training.
    """
    self._learning_rate_fn = learning_rate_fn
    self._num_microbatches = num_microbatches
    self._weight_metrics_computer = weight_metrics_computer

    super().__init__(
        model=model,
        train_state=train_state,
        partitioner=partitioner,
        eval_names=eval_names,
        summary_dir=summary_dir,
        train_state_axes=train_state_axes,
        rng=rng)

  @cached_property
  def _partitioned_train_step(self) -> PartitionedTrainCallable:

    def train_step(train_state: train_state_lib.TrainState, batch: BatchType):
      return train_with_lr(
          train_state,
          batch,
          learning_rate=self._learning_rate_fn(train_state.step),
          dropout_rng=self._get_step_rng(train_state.step),  # pytype: disable=wrong-arg-types  # jax-ndarray
          model=self._model,
          num_microbatches=self._num_microbatches,
          weight_metrics_computer=self._weight_metrics_computer,
          data_partition_spec=self._partitioner.data_partition_spec)

    return self._partitioner.partition(
        train_step,
        in_axis_resources=(self._train_state_axes,
                           self._partitioner.data_partition_spec),
        out_axis_resources=(self._train_state_axes, None),
        donate_argnums=(0,))

  @cached_property
  def _partitioned_eval_step(self) -> PartitionedEvalCallable:
    return self._partitioner.partition(
        lambda *args, **kwargs: eval_step(self._model, *args, **kwargs),
        in_axis_resources=(self._train_state_axes,
                           self._partitioner.data_partition_spec),
        out_axis_resources=None)


def _warn_action_not_run(action, task, metric):
  logging.warning(
      "The action: %s that tracks metric: %s for task: %s is not run", action,
      metric, task)


# TODO(b/200701930): Support dynamic registration for enum.
@enum.unique
class ActionMode(enum.Enum):
  """Defines when to run a action.

  For example, TRAIN means to run an action after a TRAIN loop is done.
  """
  TRAIN = 1
  TRAIN_EVAL = 2
  INFER_EVAL = 3


class BaseAction(abc.ABC):
  """Base Action class for override. The action itself does nothing."""

  @abc.abstractmethod
  def run(self, train_state: train_state_lib.TrainState,
          metrics_by_task: Mapping[str, MetricValueMapType]) -> bool:
    """Runs an action for the given train_state and metrics.

    Args:
      train_state: The current train_state in the training loop.
      metrics_by_task: A map of metrics that is grouped by each task.

    Returns:
      A bool indicating whether training should be halted.
    """
    raise NotImplementedError("Action must define its run method.")


ActionMapType = Mapping[ActionMode, Sequence[BaseAction]]


class EarlyStoppingAction(BaseAction):
  """Terminates training when the specified metric is not improving.

  Checks whether the monitored metrics are decreasing after every `train` or
  `eval`, or `both`. If the loss is no longer decreasing for `patience` times,
  terminate the training process.
  """

  def __init__(self,
               metric: Tuple[str, str],
               mode: str,
               patience: int = 3,
               atol: float = 0.,
               rtol: float = 0.):
    """Constructs the EarlyStoppingAction.

    Args:
      metric: A metric to monitor when invoking the action. When the metric does
        not improve for a number of times (specified in patience), stop the
        training. The tuple takes 2 strings, whereas the first string defines
        the task to track, and the second defines the metric of the task to
        track. e.g.,: ('mt5_xnli_dev_test.all_langs', 'accuracy') would monitor
        the 'accuracy' for `mt5_xnli_dev_test.all_langs`.
      mode: One of `{"min", "max"}`. In `min` mode, training will stop when the
        quantity monitored has stopped decreasing; in `"max"` mode it will stop
        when the quantity monitored has stopped increasing;
      patience: The threshold of stopping criteria. Usually this is measured by
        number of steps.
      atol: Absolute tolerance in the monitored quantity to qualify as an
        improvement, i.e. a change of less than `atol`, will count as no
        improvement.
      rtol: Relative tolerance in the monitoried quantity to qualify as an
        improvement. This combined with `atol` defines whether a change is
        considered improvement. The total change is calculated as following:
        `delta = (atol + rtol * previous)` See `numpy.allclose` for detailed
        information.
    """
    self._task, self._metric = metric
    if mode not in ["min", "max"]:
      raise ValueError('mode must be in ["min", "max"]')
    self._mode = mode

    if atol < 0:
      raise ValueError("atol must be greater equal than 0")
    self._atol = atol

    if rtol < 0:
      raise ValueError("rtol must be greater equal than 0")
    self._rtol = rtol

    self._patience = patience
    self._metric_history = []

  def _compare_fn(self, current, previous):
    compare_fn = jnp.greater_equal if self._mode == "min" else jnp.less_equal
    delta = self._atol + self._rtol * abs(previous)
    if self._mode == "max":
      delta *= -1
    return compare_fn(current, previous - delta)

  def run(self, train_state: train_state_lib.TrainState,
          metrics_by_task: Mapping[str, MetricValueMapType]) -> bool:
    if self._task not in metrics_by_task.keys():
      logging.warning(
          "Monitoring task: %s does not exist in all task metrics. "
          "Available tasks are : %s", self._task, metrics_by_task.keys())
      _warn_action_not_run(type(self), self._task, self._metric)
      return False
    if self._metric not in metrics_by_task[self._task].keys():
      logging.warning("Metric : %s does not exist in metrics for task : %s",
                      self._metric, self._task)
      _warn_action_not_run(type(self), self._task, self._metric)
      return False

    m = metrics_by_task[self._task][self._metric]

    if isinstance(m, clu.values.Scalar):
      self._metric_history.append(m.value)

    # For metrics returned from action_mode=INFER_EVAL (i.e. seqio.Evaluator)
    elif isinstance(m, float):
      self._metric_history.append(m)
    else:
      logging.warning("Metric %s does not have Scalar type. Found %s.",
                      self._metric, type(m))
      _warn_action_not_run(type(self), self._task, self._metric)
      return False

    # Not enough history.
    if len(self._metric_history) < self._patience:
      return False

    if all(
        self._compare_fn(self._metric_history[i + 1], self._metric_history[i])
        for i in range(self._patience - 1)):
      logging.warning(
          "Requested `stop_training` in training loop (Details below).\n "
          "Metric: %s for Task: %s has not improved for %s iterations, detail "
          "history of the metric: %s", self._metric, self._task, self._patience,
          self._metric_history)
      return True
    # Remove extra histories that we don't need to keep.
    self._metric_history.pop(0)
    return False


class TerminateOnNanAction(BaseAction):
  """Terminates training when NaN loss is detected.

  Checks whether the loss metric for the given task is NaN or Inf and terminates
  training if so.
  """

  def __init__(self, task: str, metric: str = "loss"):
    """Constructs the TerminateOnNanAction.

    Args:
      task: Defines the task from which to track the given metric.
      metric: Defines a metric to track for NaN values (defaults to "loss").
    """
    self._task = task
    self._metric = metric

  def run(self, train_state: train_state_lib.TrainState,
          metrics_by_task: Mapping[str, MetricValueMapType]) -> bool:
    if self._task not in metrics_by_task.keys():
      logging.warning(
          "Monitoring task: %s does not exist in all task metrics. "
          "Available tasks are : %s", self._task, metrics_by_task.keys())
      _warn_action_not_run(type(self), self._task, self._metric)
      return False
    if self._metric not in metrics_by_task[self._task].keys():
      logging.warning("Metric : %s does not exist in metrics for task : %s",
                      self._metric, self._task)
      _warn_action_not_run(type(self), self._task, self._metric)
      return False

    metric = metrics_by_task[self._task][self._metric]

    if not isinstance(metric, clu.values.Scalar):
      logging.warning("Metric %s does not have Scalar type. Found %s.",
                      self._metric, type(metric))
      _warn_action_not_run(type(self), self._task, self._metric)
      return False

    value = metric.value
    if np.isnan(value) or np.isinf(value):
      logging.warning(
          "Requested `stop_training` in training loop (Details below).\n "
          "NaN encountered in metric for task : %s", self._task)
      return True

    return False
