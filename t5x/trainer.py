# Copyright 2021 The T5X Authors.
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
import concurrent.futures
import enum
import os
import time
from typing import Any, Dict, Iterator, Mapping, MutableMapping, Optional, Sequence, TYPE_CHECKING, Tuple, Union

from absl import logging
import cached_property
from clu import metric_writers
from clu.metrics import Metric
from flax.core import FrozenDict
import jax.lax
import jax.numpy as jnp
import jax.random
import numpy as np
from t5x import metrics as metrics_lib
from t5x import models
from t5x import multihost_utils
from t5x import partitioning
from t5x import train_state as train_state_lib
from t5x import utils
import typing_extensions


Array = Union[np.ndarray, jnp.ndarray]
BatchSpec = Mapping[str, jax.ShapeDtypeStruct]
BatchType = Mapping[str, np.ndarray]
Rng = jnp.ndarray
MetricMapType = MutableMapping[str, Metric]
MetricMapSpec = Mapping[str, jax.ShapeDtypeStruct]
ModelWeights = Any
MutableMetricMapType = Dict[str, Metric]
P = partitioning.PartitionSpec

if TYPE_CHECKING:  # See b/163639353
  cached_property = property  # pylint: disable=invalid-name
else:
  cached_property = cached_property.cached_property


class OptionalArrayMapFuture(typing_extensions.Protocol):

  def result(self) -> Optional[Mapping[str, Array]]:
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
      self, train_state: train_state_lib.TrainState, batch: BatchType,
      prev_metrics: MetricMapType
  ) -> Tuple[train_state_lib.TrainState, MetricMapType]:
    ...


class PartitionedEvalCallable(typing_extensions.Protocol):
  """Protocol for a partitioned eval step."""

  def __call__(
      self,
      train_state: train_state_lib.TrainState,
      batch: jnp.ndarray,
      prev_metrics: MetricMapType,
  ) -> MetricMapType:
    ...


class WeightMetricsComputer(object):
  """Decides which weight metrics to compute during training."""

  _WEIGHT_METRICS = [
      "weight_rms", "weight_gradient_rms", "weight_update_rms", "weight_max"
  ]

  def get_initial_metrics(
      self,
      initial_train_state: train_state_lib.TrainState) -> MutableMetricMapType:
    """Returns a set of zero-initialized metrics.

    Args:
      initial_train_state: A training state that determines the set of weights
        the model has, which in turn determines what weight statistics will be
        computed.

    Returns:
      A map of metrics, indexed by metric name.
    """

    initial_metrics = {
        "weight_gradient_norm": metrics_lib.Sum.from_model_output(0.)
    }
    targets = utils.flatten_dict_string_keys(
        initial_train_state.state_dict()["target"])
    for name in self._WEIGHT_METRICS:
      initial_metrics.update({
          f"{name}/{k}": metrics_lib.Sum.from_model_output(0.)
          for k in targets.keys()
      })
    return initial_metrics

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
    metrics.update(_make_rms_metrics("weight_rms", new_train_state.params))
    metrics.update(_make_rms_metrics("weight_gradient_rms", gradients))
    grad_norm = jnp.sqrt(
        jnp.sum(
            jnp.array([jnp.vdot(x, x) for x in jax.tree_leaves(gradients)])))
    metrics.update(
        {"weight_gradient_norm": metrics_lib.Sum.from_model_output(grad_norm)})
    metrics.update(
        _make_rms_metrics(
            "weight_update_rms",
            jax.tree_multimap(jnp.subtract, new_train_state.params,
                              old_train_state.params)))
    metrics.update(_make_max_metrics("weight_max", new_train_state.params))

    return metrics


class MetricsManager(object):
  """Manages a set of distributed metrics and their logging to Tensorboard.

  """

  def __init__(self, name: str, initial_accumulator: MetricMapType,
               summarize_fn: SummarizeMetricsCallable,
               summary_dir: Optional[str]):
    """MetricsManager constructor.

    Args:
      name: an identifier of the metrics to use when logging (e.g., 'train').
      initial_accumulator: a mapping from metric name to the initial values
        (clu.metric.Metric objects) for accumulation.
      summarize_fn: a callable to convert the mapping of accumulated metrics
        into a mapping of scalars to be logged.
      summary_dir: the summary directory. If provided, TensorBoard summaries
        will be written to a `name` subdirectory.
    """
    self._name = name
    self._initial_accumulator = initial_accumulator
    self._summarize_fn = summarize_fn
    self.summary_dir = os.path.join(summary_dir, name) if summary_dir else None
    writers = []

    self._summary_writer = None
    if self.summary_dir and jax.process_index() == 0:
      self._summary_writer = metric_writers.SummaryWriter(self.summary_dir)
    if self._summary_writer:
      writers.append(self._summary_writer)
    self._writer = metric_writers.MultiWriter(writers)

  @property
  def summary_writer(self) -> Optional[metric_writers.SummaryWriter]:
    """Returns the MultiWriter used by this class."""
    return self._summary_writer

  @property
  def initial_accumulator(self) -> MetricMapType:
    """Returns a metric map with initial values for accumulation."""
    return dict(self._initial_accumulator)

  def write_scalar(self, key: str, val, step: int):
    """Writes scalar value to TensorBoard, if host 0."""
    self._writer.write_scalars(step, {key: val})

  def write_metrics_summary(self, metrics: MetricMapType, step: int,
                            duration: float,
                            num_steps: int) -> Optional[Mapping[str, Array]]:
    """Writes summary based on accumulated metrics.

    Actual write only occurs on host 0.

    Args:
      metrics: acculumated metric values.
      step: the current train step.
      duration: the duration of the run being summarized.
      num_steps: the number of steps the metrics are accumulated across.

    Returns:
      A mapping of name -> scalar value of the written summary. Only return the
        real scalar value on host 0. For other hosts, return None.
    """
    if set(self.initial_accumulator) != set(metrics):
      raise ValueError(
          "Initial and accumulated metric names do not match: "
          f"{sorted(self.initial_accumulator)} vs {sorted(metrics)}")

    if jax.process_index() == 0:
      # Ensure that there are no TPU computations since this method may be
      # called from a separate thread.
      summary = self._summarize_fn(
          metrics=metrics, duration=duration, num_steps=num_steps)
      # TODO(b/203790423): Use CLU LoggingWriter.
      logging.info("%s metrics at step: %s, %s", self._name, step, summary)
      self._writer.write_scalars(step, summary)
      self._writer.flush()
      return summary
    return None


class PreemptionError(Exception):
  """Training has been interrupted and needs an emergency checkpoint."""


class BaseTrainer(abc.ABC):
  """Abstract base trainer class."""

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
    self._compiled_eval_step_cache: MutableMapping[Tuple[
        MetricMapSpec, BatchSpec], PartitionedEvalCallable] = {}

    self.train_state = train_state

    self.stop_training = False

    # The training metrics combine metrics added by the Model (e.g., loss and
    # accuracy) and Trainer (e.g., learning rate).
    # Pre-copy the initial accumulator to devices to reduce communication.
    on_device_initial_accumulator = self._partitioner.partition(
        lambda x: x, in_axis_resources=None, out_axis_resources=None)({
            **model.get_initial_metrics(),
            **self._get_initial_metrics()
        })
    self.train_metrics_manager = MetricsManager(
        "train",
        initial_accumulator=on_device_initial_accumulator,
        summarize_fn=lambda *args, **kwargs: {  # pylint:disable=g-long-lambda
            **model.summarize_metrics_fn(*args, **kwargs),
            **self._summarize_metrics_fn(*args, **kwargs)
        },
        summary_dir=summary_dir)

    # The eval metrics only include metrics added by the Model.
    self.eval_metrics_managers = {  # pylint:disable=g-complex-comprehension
        n: MetricsManager(
            f"training_eval/{n}",
            initial_accumulator=model.get_initial_metrics(),
            summarize_fn=model.summarize_metrics_fn,
            summary_dir=summary_dir) for n in eval_names
    }

    self._metrics_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1)

  def __del__(self):
    """Wait for metrics to be written before deletion."""
    self._metrics_executor.shutdown(wait=True)

  def _get_step_rng(self, step: int) -> Rng:
    return jax.random.fold_in(self._base_rng, step)

  def _device_copy_and_write_summary(self, summarize_fn, metrics, tick,
                                     num_steps, start_step):
    """Copy to device to avoid TPU computations in separate thread."""

    def _replicated_device_get(x):
      """Avoids unnecessary coms when getting replicated ShardedDeviceArray."""
      if isinstance(x, jax.pxla.ShardedDeviceArray):
        x = x.device_buffers[0]
      return x.copy()

    final_metrics = jax.tree_map(_replicated_device_get, metrics)
    # Take end time only after step computation is completed.
    tock = time.time()
    return summarize_fn(final_metrics, start_step + num_steps, tock - tick,
                        num_steps)

  def train(self,
            batch_iter: Iterator[BatchType],
            num_steps: int,
            start_step: Optional[int] = None) -> OptionalArrayMapFuture:
    """Runs the train loop for the given number of steps."""
    tick = time.time()
    metrics = self.train_metrics_manager.initial_accumulator
    # Compute step number on host to avoid communication overhead.
    start_step = int(
        start_step if start_step is not None else self.train_state.step)
    for step_num in range(start_step, start_step + num_steps):
      logging.log_every_n_seconds(logging.INFO, "Training: step %d", 10,
                                  step_num)
      # Use pre-compiled step, if available.
      train_step_fn = self._compiled_train_step or self._partitioned_train_step
      with jax.profiler.StepTraceAnnotation("train", step_num=step_num):
        batch = next(batch_iter)
        self.train_state, metrics = train_step_fn(self.train_state, batch,
                                                  metrics)

    return self._metrics_executor.submit(
        self._device_copy_and_write_summary,
        self.train_metrics_manager.write_metrics_summary, metrics, tick,
        num_steps, start_step)

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
        self._partitioned_train_step, self.train_state, batch,
        self.train_metrics_manager.initial_accumulator)
    tock = time.time()
    self.train_metrics_manager.write_scalar("timing/compilation_seconds",
                                            tock - tick, self.train_state.step)

  def eval(
      self, batch_iters: Mapping[str,
                                 Iterator[BatchType]]) -> Mapping[str, Array]:
    """Runs evaluation loop over the iterator and writes summary."""
    eval_summaries = {}
    for iter_name, batch_iter in batch_iters.items():
      logging.info("Evaluating: %s.", iter_name)
      tick = time.time()
      metrics = self.eval_metrics_managers[iter_name].initial_accumulator
      # Use a pre-compiled step function, if available.
      eval_step_fn = self._compiled_eval_steps.get(iter_name,
                                                   self._partitioned_eval_step)
      num_steps = 0
      for batch in batch_iter:
        num_steps += 1
        multihost_utils.assert_same(
            jnp.array(num_steps),
            "Eval step mismatch across hosts. Check for empty dataset shard.")
        metrics = eval_step_fn(self.train_state, batch, metrics)
      multihost_utils.assert_same(
          jnp.array(-1),
          "Eval step mismatch across hosts. Check for empty dataset shard.")

      # TODO(adarob): Write metrics in separate thread.
      eval_summary = self._device_copy_and_write_summary(
          self.eval_metrics_managers[iter_name].write_metrics_summary, metrics,
          tick, num_steps, self.train_state.step)

      if eval_summary is not None:
        eval_summaries[iter_name] = eval_summary
    return eval_summaries

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
      metrics = self.eval_metrics_managers[eval_name].initial_accumulator
      cache_key: Tuple[MetricMapSpec, BatchSpec] = (
          FrozenDict(jax.eval_shape(lambda: metrics)),  # pylint:disable=cell-var-from-loop
          FrozenDict(jax.eval_shape(lambda: batch)))  # pylint:disable=cell-var-from-loop
      if cache_key not in self._compiled_eval_step_cache:
        self._compiled_eval_step_cache[cache_key] = self._partitioner.compile(
            self._partitioned_eval_step, self.train_state, batch, metrics)
      self._compiled_eval_steps[eval_name] = self._compiled_eval_step_cache[
          cache_key]
      tock = time.time()
      self.eval_metrics_managers[eval_name].write_scalar(
          "timing/compilation_seconds", tock - tick, self.train_state.step)

  @property
  def _summarize_metrics_fn(self) -> SummarizeMetricsCallable:
    """Summary function for Trainer metrics (excludes model metrics)."""
    raise NotImplementedError

  @abc.abstractmethod
  def _get_initial_metrics(self) -> MutableMetricMapType:
    """Returns initial metrics map for Trainer (excludes model metrics)."""
    raise NotImplementedError

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
    model: models.BaseModel, train_state: train_state_lib.TrainState,
    batch: BatchType, dropout_rng: Rng, num_microbatches: Optional[int]
) -> Tuple[train_state_lib.TrainState, MutableMetricMapType]:
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

  Returns:
   Accumulated gradients and incremental metrics.
  """
  batch_size = next(iter(batch.values())).shape[0]

  grad_fn = jax.value_and_grad(model.loss_fn, has_aux=True)

  if num_microbatches is None or num_microbatches <= 1:
    (loss, (weight_sum, metrics)), grad_accum = grad_fn(train_state.params,
                                                        batch, dropout_rng)
    del loss, weight_sum
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

    def per_microbatch_train_step(
        loop_cnt: int, state: Tuple[jnp.ndarray, jnp.ndarray,
                                    Mapping[str, jnp.ndarray]]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Mapping[str, jnp.ndarray]]:
      (dropout_rng, grad_accum, prev_metrics) = state
      dropout_rng, sub_dropout_rng = jax.random.split(dropout_rng)
      mbatch = get_microbatch(batch, loop_cnt)
      # We need to annotate the microbatch sharding as we would a batch.
      mbatch = jax.tree_map(
          lambda x: partitioning.with_sharding_constraint(  # pylint: disable=g-long-lambda
              x, partitioning.PartitionSpec("data")),
          mbatch)

      (loss, (weight_sum, metrics)), grad = grad_fn(train_state.params, mbatch,
                                                    sub_dropout_rng)
      del loss, weight_sum
      grad_accum = jax.tree_multimap(jnp.add, grad_accum, grad)
      metrics = {k: v.merge(metrics[k]) for k, v in prev_metrics.items()}
      return dropout_rng, grad_accum, metrics

    # Initialize gradient accumulation loop state.
    accum_dtype = jnp.float32
    grad_accum_init = jax.tree_map(lambda x: jnp.zeros(x.shape, accum_dtype),
                                   train_state.params)
    loop_init = (dropout_rng, grad_accum_init, model.get_initial_metrics())
    # Run gradient accumulation loop.
    new_dropout_rng, grad_accum, metrics = jax.lax.fori_loop(
        0, num_microbatches, per_microbatch_train_step, loop_init)
    del new_dropout_rng

  return grad_accum, metrics


def apply_grads(
    train_state: train_state_lib.TrainState, grad_accum: ModelWeights,
    metrics: MutableMetricMapType, learning_rate: jnp.ndarray,
    weight_metrics_computer: Optional[WeightMetricsComputer]
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

  Returns:
   The updated train state, metrics.
  """
  # Update optimizer using accumulated gradient.
  new_train_state = train_state.apply_gradient(
      grad_accum, learning_rate=learning_rate)
  metrics["learning_rate"] = metrics_lib.Sum.from_model_output(learning_rate)
  if weight_metrics_computer is not None:
    metrics.update(
        weight_metrics_computer.compute_metrics(grad_accum, train_state,
                                                new_train_state))
  return new_train_state, metrics


def eval_step(model: models.BaseModel, train_state: train_state_lib.TrainState,
              batch: jnp.ndarray, prev_metrics: MetricMapType) -> MetricMapType:
  """Default evaluation step."""
  _, (_, metrics) = model.eval_fn(train_state.params, batch)
  metrics = {k: v.merge(metrics[k]) for k, v in prev_metrics.items()}
  return metrics


def _make_rms_metrics(name, tree):
  """Calculates the root-mean-square metric for a pytree."""
  return {
      f"{name}/{k}":
      metrics_lib.Sum.from_model_output(jnp.sqrt(jnp.mean(jnp.square(v))))
      for k, v in utils.flatten_dict_string_keys(tree).items()
  }


def _make_max_metrics(name, tree):
  """Calculates the L-inf norm for a pytree."""
  return {
      f"{name}/{k}": metrics_lib.Sum.from_model_output(jnp.max(jnp.abs(v)))
      for k, v in utils.flatten_dict_string_keys(tree).items()
  }


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

  @property
  def _summarize_metrics_fn(self) -> SummarizeMetricsCallable:

    def _summarize_trainer_metrics(metrics: MetricMapType, duration: float,
                                   num_steps: int) -> Mapping[str, jnp.ndarray]:
      """Produces summaries for metrics added by the trainer."""
      del duration
      summary_metrics = {"learning_rate": metrics["learning_rate"]}

      if self._weight_metrics_computer is not None:
        trainer_metric_names = set(
            self._weight_metrics_computer.get_initial_metrics(self.train_state))
        summary_metrics.update(
            {k: v for k, v in metrics.items() if k in trainer_metric_names})

      for k, v in summary_metrics.items():
        # All of the Sum metrics should be divided by num_steps, since they've
        # been accumulated that many times.
        if isinstance(v, metrics_lib.Sum):
          summary = v.compute() / num_steps
        else:
          summary = v.compute()
        summary_metrics[k] = summary

      return summary_metrics

    return _summarize_trainer_metrics

  def _get_initial_metrics(self) -> MutableMetricMapType:
    initial_metrics = {"learning_rate": metrics_lib.Sum.from_model_output(0.)}
    if self._weight_metrics_computer is not None:
      initial_metrics.update(
          self._weight_metrics_computer.get_initial_metrics(self.train_state))
    return initial_metrics

  @cached_property
  def _partitioned_train_step(self) -> PartitionedTrainCallable:

    def train_with_lr(train_state: train_state_lib.TrainState, batch: BatchType,
                      prev_metrics: MetricMapType):

      learning_rate = self._learning_rate_fn(train_state.step)
      dropout_rng = self._get_step_rng(train_state.step)
      grad_accum, metrics = (
          accumulate_grads_microbatched(self._model, train_state, batch,
                                        dropout_rng, self._num_microbatches))
      new_train_state, metrics = apply_grads(train_state, grad_accum, metrics,
                                             learning_rate,
                                             self._weight_metrics_computer)
      metrics = {k: v.merge(metrics[k]) for k, v in prev_metrics.items()}
      return new_train_state, metrics

    return self._partitioner.partition(
        train_with_lr,
        in_axis_resources=(self._train_state_axes, P("data",), None),
        out_axis_resources=(self._train_state_axes, None),
        donate_argnums=(0,))

  @cached_property
  def _partitioned_eval_step(self) -> PartitionedEvalCallable:
    return self._partitioner.partition(
        lambda *args, **kwargs: eval_step(self._model, *args, **kwargs),
        in_axis_resources=(self._train_state_axes, P("data",), None),
        out_axis_resources=None)


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
          metrics_by_task: Mapping[str, Array]) -> bool:
    """Runs an action for the given train_state and metrics.

    Args:
      train_state: The current train_state in the training loop.
      metrics_by_task: A map of metrics that is grouped by each task.

    Returns:
      A bool indicating whether training should be halted.
    """
    raise NotImplementedError("Action must define its run method.")


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
        track.
        e.g.,: ('mt5_xnli_dev_test.all_langs', 'accuracy') would monitor the
          'accuracy' for `mt5_xnli_dev_test.all_langs`.
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
        considered improvement.
        The total change is calculated as following: `delta = (atol + rtol *
          previous)` See `numpy.allclose` for detailed information.
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
          metrics_by_task: Mapping[str, MetricMapType]) -> bool:
    if self._task not in metrics_by_task.keys():
      logging.warning(
          "Monitoring task: %s does not exist in all task metrics. "
          "Available tasks are : %s", self._task, metrics_by_task.keys())
      logging.warning(
          "The action that tracks metric : %s for task : %s is not run",
          self._metric, self._task)
      return False
    else:
      self._metric_history.append(metrics_by_task[self._task][self._metric])

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
          metrics_by_task: Mapping[str, MetricMapType]) -> bool:
    if self._task not in metrics_by_task.keys():
      logging.warning(
          "Monitoring task: %s does not exist in all task metrics. "
          "Available tasks are : %s", self._task, metrics_by_task.keys())
      logging.warning("TerminateOnNanAction for task : %s is not run.",
                      self._task)
      return False
    if self._metric not in metrics_by_task[self._task].keys():
      logging.warning("Metric : %s does not exist in metrics for task : %s",
                      self._metric, self._task)
      logging.warning("TerminateOnNanAction for task : %s is not run.",
                      self._task)
      return False

    metric = metrics_by_task[self._task][self._metric]
    if np.isnan(metric) or np.isinf(metric):
      logging.warning(
          "Requested `stop_training` in training loop (Details below).\n "
          "NaN encountered in metric for task : %s", self._task)
      return True

    return False
