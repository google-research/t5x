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

from typing import Any, Callable, Optional, Sequence, Tuple, Union

from absl import logging
import cached_property
from flax import core as flax_core
import jax
from jax.experimental.maps import Mesh
import numpy as np
from t5x import adafactor
from t5x import optimizers
from t5x import partitioning as base_partitioning
from t5x import train_state as train_state_lib
from t5x.contrib.moe import training_utils

DataLayout = base_partitioning.DataLayout
FlaxOptimTrainState = train_state_lib.FlaxOptimTrainState
HardwareMesh = base_partitioning.HardwareMesh
InferenceState = train_state_lib.InferenceState
JaxDevice = jax.lib.xla_client.Device
LogicalAxisRules = base_partitioning.LogicalAxisRules
PartitionSpec = base_partitioning.PartitionSpec
Pytree = Any
TrainState = train_state_lib.TrainState


def get_cpu_mesh() -> Mesh:
  """Trivial MoE mesh for CPU Testing."""
  base_cpu_mesh = base_partitioning.get_cpu_mesh()
  # Add extra dimension for new 'expert' axis.
  devices = np.expand_dims(base_cpu_mesh.devices, axis=-1)
  return Mesh(devices, ['data', 'expert', 'model'])


def get_gpu_mesh() -> Mesh:
  """Simple MoE mesh for GPUs."""
  base_gpu_mesh = base_partitioning.get_gpu_mesh(jax.local_device_count())
  # Move devices from the 'model' to the 'expert' axis.
  devices = np.expand_dims(base_gpu_mesh.devices, axis=-1)
  return Mesh(devices, ['data', 'expert', 'model'])


def default_moe_mesh(num_experts: int,
                     num_partitions: Optional[int] = None,
                     model_parallel_submesh: Optional[HardwareMesh] = None,
                     backend: Optional[str] = None) -> Mesh:
  """Construct default xmap/pjit mesh for MoE.

  Unlike the vanilla T5X mesh, this mesh has three resource axes:
  - 'expert': a 1D submesh with length that divides into `num_experts`,
  - 'model': specified by the provided `model_parallel_submesh` shape, and
  - 'data', which covers the rest of the mesh.

  Relative to the vanilla T5X mesh, the `expert` axis is constructed by
  factoring along the 'data' axis length.

  Args:
    num_experts: Total number of experts across all devices.
    num_partitions: Specifies the size of the model parallel submesh to be
      automatically selected for the current topology. See
      `model_parallel_submesh` for details on how this submesh is used. Mutually
      exclusive with `model_parallel_submesh`.
    model_parallel_submesh: 4-tuple that specifies the `(x, y, z, c)` submesh
      model-parallel device tile. See also t5x/partitioning.py for details. This
      argument is mutually exclusive with `num_partitions`.
    backend: Fetch devices from the pinned backend, if specified. This is useful
      for explicitly specifying the devices other than relying on
      jax_platform_name.

  Returns:
    xmap/pjit 3D Mesh with 'data', 'expert' and 'model' mesh axes.
  """
  # Base mesh has shape ('data', 'model').
  logging.info('For MoE, first construct vanilla T5X (data, model) mesh.')
  base_default_mesh = base_partitioning.default_mesh(num_partitions,
                                                     model_parallel_submesh,
                                                     backend)
  data_axis_size, model_axis_size = base_default_mesh.devices.shape

  # Factor out the largest divisor of 'data' axis satisfying <= `num_experts`.
  expert_axis_size = num_experts
  while data_axis_size % expert_axis_size != 0:
    expert_axis_size -= 1

  # Reshape mesh to ('data', 'expert', 'model').
  devices = base_default_mesh.devices.reshape(-1, expert_axis_size,
                                              model_axis_size)
  global_mesh = Mesh(devices, ['data', 'expert', 'model'])
  logging.info('Overridden MoE global_mesh axes_names: %s',
               global_mesh.axis_names)
  logging.info('Overridden MoE global_mesh devices: %s', global_mesh.devices)
  return global_mesh


class MoePjitPartitioner(base_partitioning.PjitPartitioner):
  """Pjit partitioner with overrides for Mixture of Experts support.

  This MoE partitioner overrides the default partitioner to use the MoE friendly
  ('data', 'expert', 'model') mesh. MoE params and state are partitioned along
  the 'expert' axis. Data is partitioned along both of the 'data' AND 'expert'
  axes.

  Additionally, when training with T5X's Adafactor optimizer, it handles an edge
  case where the MoE optimizer state terms do NOT automatically inherit the
  'expert' axis annotation from the model params; see get_logical_axes().
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
        across the model-parallel submesh), 'data' (to shard across the
        data-parallel submesh), or 'expert' (for expert parallelism).
      state_filter_fn: Function to identify which optimizer state axis rules
        should be overridden to be sharded along the 'expert' axis. If None
        (default), Adafactor expert sharding overrides are used.
    """
    if logical_axis_rules is None:
      logical_axis_rules = standard_logical_axis_rules()

    super().__init__(
        num_partitions=num_partitions,
        model_parallel_submesh=model_parallel_submesh,
        params_on_devices=params_on_devices,
        logical_axis_rules=logical_axis_rules)

    self._num_experts = num_experts
    self._state_filter_fn = state_filter_fn

  @property
  def data_partition_spec(self) -> PartitionSpec:
    """Returns MoE data partitioning spec.

    Data is sharded across the 'expert' and 'data' axes.

    Returns:
      Mesh dependent partition spec.
    """
    return PartitionSpec(('expert', 'data'),)

  @cached_property.cached_property
  def mesh(self) -> Mesh:
    """Overrides default T5X mesh with ('data', 'expert', 'model') mesh."""
    return default_moe_mesh(self._num_experts, self._num_partitions,
                            self._model_parallel_submesh, self._backend)

  def get_data_layout(self,
                      batch_size: Optional[int] = None,
                      host_index: Optional[int] = None) -> DataLayout:
    """Returns filled `DataLayout` based on the partitioned model layout.

    Overrides default data layout for MoE, where we treat 'data' and 'expert'
    axes as "data" axes.

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
    if host_index is not None:
      raise NotImplementedError('Explicit host_index is not yet implemented.')

    num_data_partitions = self._local_chunker.global_mesh.shape['data']
    num_expert_partitions = self._local_chunker.global_mesh.shape['expert']

    data_mesh_size = num_data_partitions * num_expert_partitions
    batch_size = batch_size or data_mesh_size
    if batch_size % data_mesh_size:
      raise ValueError(
          f'Batch size ({batch_size}) must be divisible by entire data mesh '
          f'size ({data_mesh_size}). Note that for MoE, the data mesh spans '
          'both the "expert" and "data" virtual mesh axes.')

    num_shards = self._local_chunker.num_chunks[
        'data'] * self._local_chunker.num_chunks['expert']
    if batch_size % num_shards:
      raise ValueError(
          f'Batch size ({batch_size}) must be divisible by total number of '
          f'shards ({num_shards}) across "data" and "expert" mesh axes.')

    # Partition the batch over both of the 'expert' and 'data' axes.
    global_array_shape = (num_expert_partitions,
                          batch_size // num_expert_partitions)
    replica_id = self._local_chunker.get_local_chunk_info(
        global_array_shape, ('expert', 'data')).replica_id

    return DataLayout(
        batch_size=batch_size,
        shard_id=(self._local_chunker.chunk_ids['data'] +
                  self._local_chunker.chunk_ids['expert'] *
                  self._local_chunker.num_chunks['data']),
        num_shards=num_shards,
        is_first_host_in_replica_set=(replica_id == 0))

  def get_logical_axes(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self, train_state: Union[FlaxOptimTrainState, InferenceState]
  ) -> Union[FlaxOptimTrainState, InferenceState]:
    """Returns a copy of TrainState with Optional[AxisNames] as leaves.

    Overrides the default logical axes by prepending the 'expert' axis to any
    MoE optimizer state terms (identified by self._state_filter_fn); this is
    useful for T5X's Adafactor optimizer, which does not propagate param
    annotations to the optimizer state when the optimizers is factored.

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
    else:
      train_state: FlaxOptimTrainState

    state_filter_fn = (
        self._state_filter_fn or _infer_state_filter_fn(train_state))
    if state_filter_fn is None:
      # No state updates required.
      return logical_axes

    prepend_expert = lambda x: PartitionSpec(  # pylint: disable=g-long-lambda
        *('expert',) + x) if x else PartitionSpec('expert',)
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
  ) -> base_partitioning.PjittedFnWithContext:
    """Partitions the computation using pjit.

    Overrides the default pjit partitioning to ensure that data is sharded along
    both of the 'data' and 'expert' axes.

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
    # Override the partition specs to use 'data' AND 'expert' axes for data
    # parallelism.
    in_axis_resources = override_partition_specs(in_axis_resources)
    out_axis_resources = override_partition_specs(out_axis_resources)

    pjitted = base_partitioning.pjit(
        fn,
        in_axis_resources=in_axis_resources,
        out_axis_resources=out_axis_resources,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
        backend=self._backend)

    return base_partitioning.PjittedFnWithContext(pjitted, self.mesh,
                                                  self._logical_axis_rules)


def standard_logical_axis_rules(
    activation_partitioning_dims: int = 1,
    parameter_partitioning_dims: int = 1,
    additional_rules: Optional[LogicalAxisRules] = None):
  """Returns partitioning rules for MoE models.

  MoE params and state are partitioned along the 'expert' axis. Data is
  partitioned along both of the 'data' AND 'expert' axes.

  The partitioning rules vary based on whether the expert and data axes need to
  be decoupled; see also MoePjitPartitioner for details of when expert and data
  axes need to be decouple.

  Buyer beware: 2D parameter sharding (`parameter_partitioning_dims=2`) is
  technically supported but untested.

  Args:
    activation_partitioning_dims: Enables 2-D activation sharding when set to 2.
    parameter_partitioning_dims: Enables 2-D parameter sharding when set to 2.
    additional_rules: Additional rules (a sequence of tuples) that will be
      appended to the standard rules.

  Returns:
    Sequence of logical axis rules.
  """
  _ = base_partitioning.global_mesh_defined()

  if parameter_partitioning_dims == 2:
    raise logging.warning(
        '2D parameter sharding (`parameter_partitioning_dims=2`) is supported '
        'but untested for MoE.')

  default_rules = base_partitioning.standard_logical_axis_rules(
      activation_partitioning_dims, parameter_partitioning_dims)
  moe_rules = [
      ('expert', 'expert'),  # Shard experts along the expert axis
      ('expert_mlp', 'model'),  # Expert MLPs partitioned along model axis
      ('expert_replicas', 'data'),  # Experts replicated along "pure" data axis
      ('unmodeled', None),  # Replicated weights
  ]
  standard_rules = list(default_rules) + moe_rules
  if additional_rules:
    standard_rules.extend(additional_rules)

  overridden_rules = []
  for logical_axis, mesh_axis in standard_rules:
    if logical_axis == 'batch':
      # Data is sharded across both 'data' and 'expert axes.
      overridden_mesh_axis = ('expert', 'data')
    else:
      overridden_mesh_axis = mesh_axis
    overridden_rules.append((logical_axis, overridden_mesh_axis))

  return overridden_rules


def compute_num_model_partitions(
    num_model_partitions: Optional[int],
    model_parallel_submesh: Optional[HardwareMesh]) -> int:
  """Returns number of model partitions.

  Args:
    num_model_partitions: Specifies the size of the model parallel submesh.
    model_parallel_submesh: 4-tuple that specifies the `(x, y, z, c)` submesh
      model-parallel device tile

  Returns:
    Size of model parallel submesh.

  Raises:
    ValueError if neither num_model_partitions nor model_parallel_submesh are
    specified, or if they are inconsistent.
  """
  if num_model_partitions is None and model_parallel_submesh is None:
    raise ValueError('At least one of num_model_partitions and '
                     'model_parallel_submesh must be specified.')

  if num_model_partitions is not None:
    if (model_parallel_submesh is not None and
        num_model_partitions != np.prod(model_parallel_submesh)):
      raise ValueError(
          'num_model_partitions and model_parallel_submesh are inconsistent. '
          'Received: %s and %s' %
          (num_model_partitions, model_parallel_submesh))
    return num_model_partitions
  else:
    return np.prod(model_parallel_submesh)


def override_partition_specs(resources: Pytree):
  """Override raw axis resources so data is sharded over 'data' & 'expert' axes.

  Here, we only override any raw partition specs that are hardcoded in T5X
  libraries:
  PartitionSpec('data',) -> PartitionSpec(('expert', 'data'),)

  NOTE: We do not (and there is no need) to override any params or optimizer
  state (which appear as large Pytrees) as these will inherit the correct specs
  from the logical axis rules; see also standard_logical_axis_rules().

  Args:
    resources: Axis resource assignment specifications.

  Returns:
    Axis resources with partition specs overridden to use 'model' as secondary
    'data' axis.
  """

  def _maybe_override_spec(axis_resource: Pytree):
    """Overrides raw "data" partition specs; leaves others unchanged."""
    if axis_resource == PartitionSpec('data',):
      # Shard all data across 'data' and 'expert' axes.
      return PartitionSpec(('expert', 'data'),)
    else:
      return axis_resource

  if isinstance(resources, PartitionSpec):
    return _maybe_override_spec(resources)
  elif isinstance(resources, Sequence):
    overridden_resources = []
    for resource in resources:
      overridden_resources.append(_maybe_override_spec(resource))
    return tuple(overridden_resources)
  else:
    return resources


def _infer_state_filter_fn(
    train_state: FlaxOptimTrainState) -> Optional[Callable[[str], bool]]:
  """Infers relevant regex matching sharded expert model state for optimizer.

  The model state generally inherits the correct partitioning specs from the
  model parameters. In such cases, no state_filter_fn is required. However,
  T5X's custom Adafactor optimizer, when factored, requires overrides to the
  `v_col` and `v_row` kernel terms; see
  https://github.com/google-research/t5x/blob/main/t5x/adafactor.py#L591. For
  those cases, we use the state_filter_fn to identify the factored kernel terms
  that need to be partitioned along the expert axis.

  Args:
    train_state: Object holding optimizer and optimizer state (parameters).

  Returns:
    Function to identify which model state is sharded along 'expert' axis.

  Raises:
    ValueError if optimizer (on train state) is not a recognized optimizer type.
  """
  optimizer = train_state._optimizer  # pylint: disable=protected-access
  optimizer_def = optimizer.optimizer_def

  if isinstance(optimizer_def, optimizers.OptaxWrapper):
    # T5X wrapped optax optimizers inherit the correct specs, so no state
    # updates will be required.
    return None

  if not isinstance(optimizer_def, adafactor.Adafactor):
    raise ValueError('Unrecognized optimizer type. Expecting '
                     'optimizers.OptaxWrapper or adafactor.Adafactor. '
                     f'Received: {optimizer_def}')

  if optimizer_def.hyper_params.factored:
    # Factored kernel terms (`v_col` and `v_row`) need to be identified for
    # expert sharding.
    return training_utils.match_fn(r'.*expert.*/kernel/v_.*')
  else:
    # Non-factored kernel terms (`v`) inherit the correct specs, so no state
    # updates will be required.
    return None


