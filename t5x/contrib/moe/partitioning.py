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

"""Pjit partitioner with Mixture of Experts overrides."""

from typing import Callable, Optional, Union

from flax import core as flax_core
from t5x import partitioning
from t5x import train_state as train_state_lib

from t5x.contrib.moe import training_utils

FlaxOptimTrainState = train_state_lib.FlaxOptimTrainState
InferenceState = train_state_lib.InferenceState
HardwareMesh = partitioning.HardwareMesh
LogicalAxisRules = partitioning.LogicalAxisRules
PartitionSpec = partitioning.PartitionSpec
TrainState = train_state_lib.TrainState


class MoePjitPartitioner(partitioning.PjitPartitioner):
  """Pjit partitioner with overrides for Mixture of Experts support."""

  def __init__(self,
               num_partitions: Optional[int] = None,
               model_parallel_submesh: Optional[HardwareMesh] = None,
               params_on_devices: bool = True,
               logical_axis_rules: Optional[LogicalAxisRules] = None,
               state_filter_fn: Optional[Callable[[str], bool]] = None):
    """Configures the partitioner.

    Args:
      num_partitions: Specifies the size of the model parallel submesh to be
        automatically selected for the current topology. See
        `model_parallel_submesh` for details on how this submesh is used.
        Mutually exclusive with `model_parallel_submesh`.
      model_parallel_submesh: 4-tuple that specifies the `(x, y, z, c)` submesh
        model-parallel device tile -- an axis of accelerator parallelism
        orthogonal to data parallelism. See t5x/partitioning.py for details.
        This argument is mutually exclusive with `num_partitions`.
      params_on_devices: Whether to keep the params on devices. If False, params
        stay in the host memory.
      logical_axis_rules: A priority-ordered sequence of KV tuples that maps
        logical axis names to either `None` (not sharded), 'model' (to shard
        across the model-parallel submesh), or 'data' (to shard across the
        data-parallel submesh).
      state_filter_fn: Function to identify which optimizer state axis rules
        should be overridden to be sharded along the 'expert' axis. If None
        (default), Adafactor expert sharding overrides are used.
    """
    super().__init__(
        num_partitions=num_partitions,
        model_parallel_submesh=model_parallel_submesh,
        params_on_devices=params_on_devices,
        logical_axis_rules=logical_axis_rules)
    self._state_filter_fn = state_filter_fn

  def get_logical_axes(
      self, train_state: Union[FlaxOptimTrainState, InferenceState]
  ) -> Union[FlaxOptimTrainState, InferenceState]:
    """Returns a copy of TrainState with Optional[AxisNames] as leaves."""
    logical_axes = train_state.as_logical_axes()

    if isinstance(logical_axes, InferenceState):
      # InferenceState does not contain any optimizer state, so we skip all
      # expert partitioning overrides.
      return logical_axes

    if self._state_filter_fn:
      state_filter_fn = self._state_filter_fn
    else:
      # Use default T5X Adafactor expert sharding overrides for factored
      # kernels.
      #
      # The kernel terms  (`m` and `v`) that are not captured below are trivial
      # placeholders and should not be sharded. The only nontrivial bias term
      # (`v`) automatically inherits the correct expert partition specs through
      # the T5X Adafactor factor rules (see derive_factor_rules():
      # https://github.com/google-research/t5x/blob/main/t5x/adafactor.py#L591).
      #
      # TODO(jamesleethorp): Revisit once other T5X optimizers are available.
      state_filter_fn = training_utils.match_fn(r'.*expert.*/kernel/v_.*')

    # Prepend 'expert' axis to MoE state terms (identified by state_filter_fn)
    # so they are sharded along the 'expert' axis.
    prepend_expert = lambda x: PartitionSpec(  # pylint: disable=g-long-lambda
        'expert',) + x if x else PartitionSpec('expert',)
    optimizer_axes = logical_axes._optimizer  # pylint: disable=protected-access
    state_dict = flax_core.unfreeze(optimizer_axes.state_dict())
    state_dict['state']['param_states'] = training_utils.tree_map_with_names(
        prepend_expert, state_dict['state']['param_states'], state_filter_fn)

    return train_state.restore_state(state_dict)


def standard_logical_axis_rules() -> partitioning.LogicalAxisRules:
  """Returns partitioning rules for Mixture of Experts models."""
  return (
      ('expert', 'data'),  # Partition experts along the data axis
      ('expert_mlp', 'model'),  # Expert MLPs partitioned along model axis
      ('expert_group', None),  # Replicated axis for all-to-all constraints
      ('unmodeled', None),  # Replicated weights
  )
