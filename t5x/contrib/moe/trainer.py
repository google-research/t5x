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

"""Trainer with Mixture of Experts support."""

from typing import Any, Callable, Optional, Sequence, TYPE_CHECKING

import cached_property
from t5x import models
from t5x import train_state as train_state_lib
from t5x import trainer
from t5x.contrib.moe import partitioning
from t5x.contrib.moe import training_utils

BatchType = trainer.BatchType
LearningRateCallable = trainer.LearningRateCallable
MetricMapType = trainer.MetricMapType
PartitionSpec = partitioning.PartitionSpec
PartitionedTrainCallable = trainer.PartitionedTrainCallable
Rng = trainer.Rng

if TYPE_CHECKING:  # See b/163639353
  cached_property = property  # pylint: disable=invalid-name
else:
  cached_property = cached_property.cached_property


class MoeTrainer(trainer.Trainer):
  """T5X trainer with overrides for Mixture of Experts support."""

  def __init__(
      self,
      model: models.BaseModel,
      train_state: train_state_lib.TrainState,
      partitioner: partitioning.MoePjitPartitioner,
      eval_names: Sequence[str],
      summary_dir: Optional[str],
      train_state_axes: Any,
      rng: Rng,
      learning_rate_fn: LearningRateCallable,
      num_microbatches: Optional[int],
      num_experts: int,
      sharded_match_fn: Optional[Callable[
          [str], bool]] = training_utils.match_fn(r'.*expert.*'),
      weight_metrics_computer: Optional[trainer.WeightMetricsComputer] = None):
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
      num_experts: Global number of experts. Used to scale sharded parameter
        gradients.
      sharded_match_fn: Filter function for distinguishing sharded (MoE)
        parameters from replicated parameters. Used to identify the sharded
        parameter gradients that need to be rescaled under pjit training.
      weight_metrics_computer: A WeightMetricsComputer instance, or None, to
        decide what metrics, if any, to log about weights and weight updates
        during training.
    """
    super().__init__(
        model=model,
        train_state=train_state,
        partitioner=partitioner,
        eval_names=eval_names,
        summary_dir=summary_dir,
        train_state_axes=train_state_axes,
        rng=rng,
        learning_rate_fn=learning_rate_fn,
        num_microbatches=num_microbatches,
        weight_metrics_computer=weight_metrics_computer)

    self._num_experts = num_experts
    self._sharded_match_fn = sharded_match_fn
    self.data_partition_spec = partitioner.data_partition_spec

  @cached_property
  def _partitioned_train_step(self) -> PartitionedTrainCallable:
    """Same as a regular T5X train step, but scales expert parameter gradients.

    We must scale expert parameter gradients by the number of experts to account
    for pjit's implicit averaging over partitioned parameter gradients.

    Returns:
      Partitioned train step function.
    """

    def train_with_lr(train_state: train_state_lib.TrainState,
                      batch: BatchType):
      grad_accum, metrics, flax_mutables = (
          trainer.accumulate_grads_microbatched(
              self._model,
              train_state,
              batch,
              self._get_step_rng(train_state.step),
              self._num_microbatches,
              data_partition_spec=self.data_partition_spec))

      # Only difference between this train step and regular T5X train step:
      scaled_grads = training_utils.scale_sharded_grads(
          grad_accum, self._sharded_match_fn, scale_factor=self._num_experts)

      new_train_state, metrics = trainer.apply_grads(
          train_state,
          scaled_grads,
          metrics,
          self._learning_rate_fn(train_state.step),
          self._weight_metrics_computer,
          other_state_variables={'flax_mutables': flax_mutables}
          if flax_mutables else None)
      return new_train_state, metrics

    return self._partitioner.partition(
        train_with_lr,
        in_axis_resources=(self._train_state_axes, self.data_partition_spec),
        out_axis_resources=(self._train_state_axes, None),
        donate_argnums=(0,))
