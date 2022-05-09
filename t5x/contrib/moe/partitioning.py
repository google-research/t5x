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

from typing import Any, Callable, Optional, Sequence, Union

from absl import logging
from flax import core as flax_core
import jax
import numpy as np
from t5x import partitioning as t5x_partitioning
from t5x import train_state as train_state_lib

from t5x.contrib.moe import training_utils

DataLayout = t5x_partitioning.DataLayout
FlaxOptimTrainState = train_state_lib.FlaxOptimTrainState
HardwareMesh = t5x_partitioning.HardwareMesh
InferenceState = train_state_lib.InferenceState
LogicalAxisRules = t5x_partitioning.LogicalAxisRules
PartitionSpec = t5x_partitioning.PartitionSpec
Pytree = Any
TrainState = train_state_lib.TrainState


class MoePjitPartitioner(t5x_partitioning.PjitPartitioner):
  """Pjit partitioner with overrides for Mixture of Experts support.

  This MoE partitioner has two overrides relative to the default partitioner:
  (1) It prepends an 'expert' axis to all MoE optimizer state terms, so that
      they are sharded along the 'expert' axis; see get_logical_axes().
  (2) In cases where model parallelism is used and the number of experts is less
      than the number of devices, we treat the 'model' axis as a secondary data
      axis. This allows us to decouple expert parallelism ('data' mesh axis)
      from data parallelism ('data' and 'model' axes).
  """

  def __init__(self,
               num_experts: int,
               num_partitions: Optional[int] = None,
               model_parallel_submesh: Optional[HardwareMesh] = None,
               params_on_devices: bool = True,
               logical_axis_rules: Optional[LogicalAxisRules] = None,
               state_filter_fn: Optional[Callable[[str], bool]] = None):
    """Configures the partitioner.

    Args:
      num_experts: Total number of experts across all devices.
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
    # If True, treat 'model' axis as secondary data axis.
    self.two_data_axes = _override_model_axis(num_experts, num_partitions,
                                              model_parallel_submesh)
    if self.two_data_axes:
      # Override num_partitions to repurpose the 'model' axis as a secondary
      # data axis, along which only the batch is sharded. Experts will be
      # replicated along this secondary data axis.
      num_partitions = jax.device_count() // num_experts

      # Override user specified model parallel submesh. Rely on T5X partitioning
      # to determine new submesh from updated `num_partitions`.
      logging.info(
          'Overriding user specified `model_parallel_submesh`=%s to support '
          'expert parallelism for updated `num_partitions`=%d',
          model_parallel_submesh, num_partitions)
      model_parallel_submesh = None

    super().__init__(
        num_partitions=num_partitions,
        model_parallel_submesh=model_parallel_submesh,
        params_on_devices=params_on_devices,
        logical_axis_rules=logical_axis_rules)

    self._state_filter_fn = state_filter_fn

  def get_data_layout(self,
                      batch_size: Optional[int] = None,
                      host_index: Optional[int] = None) -> DataLayout:
    """Returns filled `DataLayout` based on the partitioned model layout.

    Overrides default data layout in case were both mesh axes ('data' and
    'model') are treated as data axes.

    Args:
      batch_size: If set, indicates the requested batch size. If not set, the
        batch size is inferred from the layout.
      host_index: Indicates the host index to use for the calculations, if not
        set - use JAX-provided one. Should be in [0, num_hosts) interval and the
        order should match the order of corresponding CPU devices in
        `jax.devices()`.

    Returns:
      Filled `DataLayout` structure.
    """
    if self.two_data_axes:
      if host_index is not None:
        raise NotImplementedError('Explicit host_index is not yet implemented.')
      mesh_size = self._local_chunker.global_mesh.shape[
          'data'] * self._local_chunker.global_mesh.shape['model']
      batch_size = batch_size or mesh_size
      if batch_size % mesh_size:
        raise ValueError(
            f'Batch size ({batch_size}) must be divisible by corresponding '
            f'mesh size ({mesh_size}).')
      num_shards = self._local_chunker.num_chunks['data']
      if batch_size % num_shards:
        raise ValueError(
            f'Batch size ({batch_size}) must be divisible by number of '
            f'replicas ({num_shards}).')
      replica_id = self._local_chunker.get_local_chunk_info(
          (batch_size,), ('data', 'model')).replica_id
      return DataLayout(
          batch_size=batch_size,
          shard_id=self._local_chunker.chunk_ids['data'],
          num_shards=num_shards,
          is_first_host_in_replica_set=(replica_id == 0))
    else:
      return super().get_data_layout(batch_size, host_index)

  def get_logical_axes(
      self, train_state: Union[FlaxOptimTrainState, InferenceState]
  ) -> Union[FlaxOptimTrainState, InferenceState]:
    """Returns a copy of TrainState with Optional[AxisNames] as leaves.

    Overrides the default logical axes by prepending the 'expert' axis to any
    MoE optimizer state terms (identified by self._state_filter_fn) so they are
    correctly sharded along the 'expert' axis.

    Args:
      train_state: Object holding all relevant training of inference state.

    Returns:
      State object matching structure of input train_state but with axis names
      as leaves.
    """
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
      # the T5X Adafactor factor rules (see derive_logical_axes()):
      # https://github.com/google-research/t5x/blob/main/t5x/adafactor.py#L591).
      #
      # TODO(jamesleethorp): Revisit once other T5X optimizers are available.
      state_filter_fn = training_utils.match_fn(r'.*expert.*/kernel/v_.*')

    prepend_expert = lambda x: PartitionSpec(  # pylint: disable=g-long-lambda
        'expert',) + x if x else PartitionSpec('expert',)
    optimizer_axes = logical_axes._optimizer  # pylint: disable=protected-access
    state_dict = flax_core.unfreeze(optimizer_axes.state_dict())
    state_dict['state']['param_states'] = training_utils.tree_map_with_names(
        prepend_expert, state_dict['state']['param_states'], state_filter_fn)

    return train_state.restore_state(state_dict)

  def partition(
      self,
      fn: Callable,  # pylint: disable=g-bare-generic
      in_axis_resources: Pytree,
      out_axis_resources: Pytree,
      static_argnums: Union[int, Sequence[int]] = (),
      donate_argnums: Union[int, Sequence[int]] = ()
  ) -> t5x_partitioning.PjittedFnWithContext:
    """Partitions the computation using pjit.

    Overrides the default pjit partitioning in cases where expert and data axes
    are decoupled -- wherein we treat the 'model' axis as a secondary data axis.

    Args:
      fn: Function to partition.
      in_axis_resources: Pytree of structure matching that of arguments to `fn`,
        with all actual arguments replaced by resource assignment
        specifications.
      out_axis_resources: Like `in_axis_resources`, but specifies resource
        assignment for function outputs.
      static_argnums: Specifies which positional arguments to treat as static
        (compile-time constant) in the partitioned function.
      donate_argnums: Specifies which argument buffers are "donated" to the
        computation.

    Returns:
      A partitioned version of the input function.
    """
    if self.two_data_axes:
      # Both axes are used for data parallelism in this case, so we override the
      # partition specs.
      in_axis_resources = _override_partition_specs(in_axis_resources)
      out_axis_resources = _override_partition_specs(out_axis_resources)

    pjitted = t5x_partitioning.pjit(
        fn,
        in_axis_resources=in_axis_resources,
        out_axis_resources=out_axis_resources,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
        backend=self._backend)

    return t5x_partitioning.PjittedFnWithContext(pjitted, self.mesh,
                                                 self._logical_axis_rules)


def standard_logical_axis_rules(
    num_experts: int,
    num_partitions: Optional[int] = None,
    model_parallel_submesh: Optional[HardwareMesh] = None,
    activation_partitioning_dims: int = 1,
    parameter_partitioning_dims: int = 1,
    additional_rules: Optional[LogicalAxisRules] = None):
  """Returns partitioning rules for MoE models.

  The partitioning rules vary based on whether the expert and data axes need to
  be decoupled; see also MoePjitPartitioner for details of when expert and data
  axes need to be decouple.

  Args:
    num_experts: Total number of experts across all devices.
    num_partitions: Size of the model parallel submesh. Model parallelism is
      only used if num_model_partitions > 1. Ignored if model_parallel_submesh
      is specified.
    model_parallel_submesh: 4-tuple that specifies the `(x, y, z, c)` submesh
      model-parallel device tile -- an axis of accelerator parallelism
      orthogonal to data parallelism. Model parallelism is only used if
      np.prod(model_parallel_submesh) > 1. Mutually exclusive with
      `num_partitions`.
    activation_partitioning_dims: Enables 2-D activation sharding when set to 2.
    parameter_partitioning_dims: Enables 2-D parameter sharding when set to 2.
    additional_rules: Additional rules (a sequence of tuples) that will be
      appended to the standard rules.

  Returns:
    Sequence of logical axis rules.
  """

  default_rules = t5x_partitioning.standard_logical_axis_rules(
      activation_partitioning_dims, parameter_partitioning_dims)
  moe_rules = [
      ('expert', 'data'),  # Shard experts along the data axis
      ('expert_mlp', 'model'),  # Expert MLPs partitioned along model axis
      ('expert_group', None),  # Replicated axis for all-to-all constraints
      ('expert_replicas', None),  # Experts replicated along this axis
      ('unmodeled', None),  # Replicated weights
  ]
  standard_rules = list(default_rules) + moe_rules
  if additional_rules:
    standard_rules.extend(additional_rules)

  if _override_model_axis(num_experts, num_partitions, model_parallel_submesh):
    overridden_rules = []
    for logical_axis, mesh_axis in standard_rules:
      if logical_axis == 'batch':
        # Because we now treat the 'model' axis as a second data axis, we want
        # to shard batches across both axes.
        overridden_mesh_axis = ('data', 'model')
      elif logical_axis == 'expert_replicas':
        # "model" axis is repurposed as a second data axis, along which experts
        # are replicated.
        overridden_mesh_axis = 'model'
      elif mesh_axis == 'model':
        # Any weights ordinarily partitioned along the model axis, should be
        # explicitly replicated.
        overridden_mesh_axis = None
      else:
        overridden_mesh_axis = mesh_axis
      overridden_rules.append((logical_axis, overridden_mesh_axis))

    return overridden_rules

  else:
    return standard_rules


def data_partition_spec(two_data_axes: bool) -> PartitionSpec:
  """Returns data partitioning spec.

  Args:
    two_data_axes: If True, use 'model' axis as secondary data axis. Otherwise,
      only use 'data' axis for data sharding.

  Returns:
    Mesh dependent partition spec.
  """
  if two_data_axes:
    # Use 'model' axis as secondary data axis. Shard batches across both axes.
    return PartitionSpec(('data', 'model'),)
  else:
    return PartitionSpec('data',)


def _override_model_axis(
    num_experts: int, num_partitions: Optional[int],
    model_parallel_submesh: Optional[HardwareMesh]) -> bool:
  """Returns true iff there is no model parallelism & num experts < num devices.

  Args:
    num_experts: Total number of experts across all devices.
    num_partitions: Size of the model parallel submesh. Model parallelism is
      only used if num_model_partitions > 1. Mutually exclusive with
      `model_parallel_submesh`.
    model_parallel_submesh: 4-tuple that specifies the `(x, y, z, c)` submesh
      model-parallel device tile -- an axis of accelerator parallelism
      orthogonal to data parallelism. Model parallelism is only used if
      np.prod(model_parallel_submesh) > 1. Mutually exclusive with
      `num_partitions`.

  Returns:
    True if there is no model parallelism & num experts < num devices; False
    otherwise.
  """
  if (num_partitions is None) == (model_parallel_submesh is None):
    raise ValueError(
        'One, and only one, of {num_partitions, model_parallel_submesh} must '
        'be specified. Received: %s and %s' %
        (num_partitions, model_parallel_submesh))

  if num_experts == 0 or jax.device_count() <= num_experts:
    # No expert replication required. No need to override model mesh axis.
    return False

  return ((num_partitions is not None and num_partitions <= 1) or
          (model_parallel_submesh is not None and
           np.prod(model_parallel_submesh) <= 1))


def _override_partition_specs(resources: Pytree):
  """Override axis resources for two data axes setup.

  In the two data axes setup, we treat the 'model' axis as a secondary data
  axis. To this end, we override any hardcoded, raw partition specs:
  - PartitionSpec('data',) -> PartitionSpec(('data', 'model'),)
  - PartitionSpec('model',) -> None
  There is no need to override any params or optimizer state as these will
  inherit the correct specs from the logical axis rules; see
  standard_logical_axis_rules().

  Args:
    resources: Axis resource assignment specifications.

  Returns:
    Axis resources with partition specs overridden to use 'model' as secondary
    'data' axis.
  """

  def _maybe_overridde_spec(axis_resource: Pytree):
    """Overrides "data" and "model" partition specs; leaves others unchanged."""
    if axis_resource == PartitionSpec('data',):
      # Shard all batches across both axes.
      return PartitionSpec(('data', 'model'),)
    elif axis_resource == PartitionSpec('model',):
      # No model parallelism.
      return None
    else:
      return axis_resource

  if resources is None:
    return resources
  elif not isinstance(resources, Sequence):
    return _maybe_overridde_spec(resources)
  else:
    overridden_resources = []
    for resource in resources:
      overridden_resources.append(_maybe_overridde_spec(resource))
  return tuple(overridden_resources)
