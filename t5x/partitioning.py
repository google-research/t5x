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

# pytype: skip-file
# Lint as: python3
"""Utilities for partitioning."""

import abc
import collections
import dataclasses
import re
from typing import Any, Callable, Optional, Sequence, TYPE_CHECKING, Tuple, Union

from absl import logging
import cached_property
from flax.linen import partitioning as flax_partitioning
from flax.traverse_util import flatten_dict
from flax.traverse_util import unflatten_dict
import jax
from jax import numpy as jnp
from jax import random
from jax.experimental.maps import Mesh
from jax.experimental.maps import mesh
from jax.experimental.pjit import pjit as jax_pjit
from jax.experimental.pjit import with_sharding_constraint as jax_pjit_wsc
from jax.interpreters.sharded_jit import PartitionSpec
import numpy as np
from t5x import train_state as train_state_lib

JaxDevice = jax.lib.xla_client.Device
TpuMesh = Tuple[int, int, int, int]  # (x, y, z, num_cores).
OtherMesh = Tuple[int, int]
HardwareMesh = Union[TpuMesh, OtherMesh]
PyTreeDef = type(jax.tree_structure(None))
TrainState = train_state_lib.TrainState
LogicalAxisRules = Sequence[Tuple[str, Optional[str]]]

if TYPE_CHECKING:  # See b/163639353
  cached_property = property  # pylint: disable=invalid-name
else:
  cached_property = cached_property.cached_property


class AxisNames(tuple):
  """Tuple of strings specifying name for each axis.

  We create a separate class for this so JAX's pytree utilities can distinguish
  it from a tuple that should be treated as a pytree, instead treating it as a
  leaf.
  """

  def __new__(cls, *names):
    return tuple.__new__(AxisNames, names)

  def __repr__(self):
    return 'AxisNames%s' % tuple.__repr__(self)


# pjit wrappers for cpu fallback.
# -----------------------------------------------------------------------------
# TODO(levskaya): upstream this fallback behavior to jax pjit.
def pjit(
    fun: Callable,  # pylint: disable=g-bare-generic
    in_axis_resources,
    out_axis_resources,
    static_argnums: Union[int, Sequence[int]] = (),
    donate_argnums: Union[int, Sequence[int]] = ()):
  """Wrapper for pjit that calls normal jit on cpu."""
  if jax.devices()[0].platform == 'cpu':
    return jax.jit(
        fun, static_argnums=static_argnums, donate_argnums=donate_argnums)
  else:
    return jax_pjit(
        fun,
        in_axis_resources,
        out_axis_resources,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums)


def with_sharding_constraint(x, axis_resources):
  """Wrapper for pjit with_sharding_constraint, no-op on cpu or outside pjit."""
  if jax.devices()[0].platform == 'cpu' or not global_mesh_defined():
    return x
  else:
    return jax_pjit_wsc(x, axis_resources)




# pjit Mesh creation functions.
# -----------------------------------------------------------------------------
def bounds_from_last_device(
    last_device: jax.lib.xla_client.Device) -> HardwareMesh:
  """Get the bound from the given last device."""
  # Must be passed the device at the highest-coordinate corner of the
  # relevant mesh, which is a requirement we know is satisfied by the last
  # device in jax.devices().
  if hasattr(last_device, 'coords'):
    x, y, z = last_device.coords
    return x + 1, y + 1, z + 1, last_device.core_on_chip + 1
  else:
    # On non-TPU platforms, the "mesh" is hosts x devices per host in order
    # to take advantage of faster within-host interconnect.
    return jax.host_count(), jax.local_device_count()


def get_coords(device: jax.lib.xla_client.Device) -> HardwareMesh:
  """Returns the coordinates of the given device."""
  if hasattr(device, 'coords'):
    return (*device.coords, device.core_on_chip)
  return (device.process_index, device.id % jax.local_device_count())


def global_mesh_defined():
  """Checks if global xmap/pjit mesh resource environment is defined."""
  maps_env = jax.experimental.maps.thread_resources.env
  return maps_env.physical_mesh.devices.shape != ()  # pylint: disable=g-explicit-bool-comparison


def get_mesh(model_parallel_submesh: HardwareMesh,
             input_devices: Sequence[JaxDevice] = (),
             input_local_devices: Sequence[JaxDevice] = (),
             tile_by_host_if_needed: bool = True) -> Mesh:
  """Construct an xmap/pjit Mesh for the given model-parallel submesh.

  The resulting mesh has two resource axes: 'model', with the provided submesh
  shape, and 'data', which covers the rest of the mesh.

  Args:
    model_parallel_submesh: a HardwareMesh spec, namely (x,y,z,core) on TPU for
      a single model-parallel replica's "tile" in the physical device mesh. The
      first three elements (`x`, `y`, and `z`) should be factors of the pod
      slice; e.g., if you are using df_4x8, then `x` should be a factor of 4
      (one of 1, 2, 4), `y` should be a factor of 8 (one of 1, 2, 4, 8), and `z`
      must be 1, because TPU v3 slices are only 2D. `z` can be >1 for
      TPU v4 (and maybe later TPUs) that allow 3D slices. `core` is the
      number of cores to use from each TPU node. As communication is usually
      fastest inside the same node, if you need a tile of more than 1 core, then
      you should first increase `core`: e.g., for TPU v3, (1,1,1,2) is
        better than (2,1,1,1). To pick a good spec, try a few possible values
        until you get high TPU utilization.
    input_devices: the devices to use, will use jax.devices() if this is not
      set.
    input_local_devices: the local devices to use, will use jax.local_devices()
      if this is not set.
    tile_by_host_if_needed: JAX currently requires that the parts of any sharded
      array that are located on one host's local devices form a single
      contiguous slice. A best effort will be made to achieve this without
      "tiling" the device assignment over hosts (which can reduce XLA collective
      performance). If this flag is True, then the device assignment will be
      tiled over hosts if necessary to satisfy this constraint and create a
      buildable mesh; if false, mesh construction will fail instead.

  Returns:
    A xmap / pjit Mesh containing the virtual device mesh with data, model axes.
  """
  input_devices = input_devices or jax.devices()
  input_local_devices = input_local_devices or jax.local_devices(0)
  last_device = input_devices[-1]
  global_hardware_mesh = bounds_from_last_device(last_device)
  mesh_ndim = len(global_hardware_mesh)
  local_hardware_mesh = bounds_from_last_device(input_local_devices[-1])
  mesh_err = (
      f'each dimension of the model parallel submesh {model_parallel_submesh} '
      'must be a factor of the corresponding dimension of the global device '
      f'mesh {global_hardware_mesh}')
  assert not any(
      g % m
      for g, m in zip(global_hardware_mesh, model_parallel_submesh)), f"{mesh_err} {global_hardware_mesh} {model_parallel_submesh}"
  assert not any(
      g % l for g, l in zip(global_hardware_mesh, local_hardware_mesh))
  devices = np.empty(global_hardware_mesh, dtype=np.object)
  for device in input_devices:
    device_coords = get_coords(device)
    devices[device_coords] = device
  tile_by_host = tile_by_host_if_needed
  if len(global_hardware_mesh) == 4:
    # enable contiguous local chunks without host tiling by making Z major
    gx, gy, gz, gc = global_hardware_mesh
    mx, my, mz, mc = model_parallel_submesh
    if (mx == gx > 1 and my == mz == 1) or (mx == 1 and my == gy > 1 and
                                            mz == gz > 1):
      logging.info('ensuring YZ plane has a Z-major device order')
      # YZ should be ZY
      assert mc == gc, (mc, gc)
      global_hardware_mesh = gx, gz, gy, gc
      model_parallel_submesh = mx, mz, my, mc
      devices = devices.swapaxes(1, 2)
      tile_by_host = False
    if (my == gy > 1 and mx == mz == 1) or (my == 1 and mx == gx > 1 and
                                            mz == gz > 1):
      logging.info('ensuring XZ plane has a Z-major device order')
      # XZ should be ZX
      assert mc == gc, (mc, gc)
      global_hardware_mesh = gz, gy, gx, gc
      model_parallel_submesh = mz, my, mx, mc
      devices = devices.swapaxes(0, 2)
      tile_by_host = False
  if tile_by_host:
    logging.warning(
        'Tiling device assignment mesh by hosts, which may lead to '
        'reduced XLA collective performance. To avoid this, modify '
        'the model parallel submesh or run with more tasks per host.')
    tile_err = (
        'to tile the mesh by hosts, each dimension of the model parallel '
        'submesh must be either a factor or a multiple of the corresponding '
        'dimension of the per-host submesh')

    def dh_dd_mh_md(g: int, m: int, l: int) -> Tuple[int]:
      """Split a global mesh dimension into four tiling components.

      Args:
        g: global mesh bounds dimension size
        m: model-parallel submesh bounds dimension size
        l: local submesh bounds dimension size

      Returns:
        The resulting tuple divides the dimension into the hosts component of
        the data-parallel submesh, the devices component of the data-parallel
        submesh, the hosts component of the model-parallel submesh, and the
        devices component of the model-parallel submesh.
      """
      d = g // m
      if m >= l:
        assert not m % l, tile_err
        return (d, 1, m // l, l)
      else:
        assert not l % m, tile_err
        return (d // (l // m), l // m, 1, m)

    # e.g. [(x_data_hosts, x_data_devs, x_model_hosts, x_model_devs), ...]
    dh_dd_mh_md_tups = map(dh_dd_mh_md, global_hardware_mesh,
                           model_parallel_submesh, local_hardware_mesh)
    # reshape to e.g. (x_dh, x_dd, x_mh, x_md, y_dh, ...)
    devices = devices.reshape(*(s for t in dh_dd_mh_md_tups for s in t))  # pylint: disable=g-complex-comprehension
    # TODO(jekbradbury): reorder local subgroups for ring locality
    # Transpose to [data_host], [data_device], [model_host], [model_device]
    # block ordering e.g. (x_dh, y_dh, ..., x_dd, y_dd, ...)
    devices = devices.transpose(*(4 * i for i in range(mesh_ndim)),
                                *(4 * i + 1 for i in range(mesh_ndim)),
                                *(4 * i + 2 for i in range(mesh_ndim)),
                                *(4 * i + 3 for i in range(mesh_ndim)))
  else:
    # e.g. [(x_data, x_model), (y_data, y_model), ...]
    model_data_tups = [
        (g // m, m)
        for g, m in zip(global_hardware_mesh, model_parallel_submesh)
    ]
    # reshape to e.g. (x_data, x_model, y_data, y_model...)
    devices = devices.reshape(*(s for t in model_data_tups for s in t))  # pylint: disable=g-complex-comprehension
    # TODO(jekbradbury): reorder small subgroups for ring locality
    # transpose to e.g. (x_data, y_data, ..., x_model, ...)
    devices = devices.transpose(*(2 * i for i in range(mesh_ndim)),
                                *(2 * i + 1 for i in range(mesh_ndim)))
  # reshape to (data, model)
  devices = devices.reshape(-1, np.prod(model_parallel_submesh))
  global_mesh = Mesh(devices, ['data', 'model'])
  logging.info('global_mesh axes_names: %s', global_mesh.axis_names)
  logging.info('global_mesh devices: %s', global_mesh.devices)
  return global_mesh


def get_cpu_mesh() -> Mesh:
  """Trivial mesh for CPU Testing."""
  devices = np.empty((jax.host_count(), jax.local_device_count()),
                     dtype=np.object)
  for device in jax.devices():
    devices[device.process_index, device.id % jax.local_device_count()] = device
  return Mesh(devices, ['data', 'model'])


def get_gpu_mesh() -> Mesh:
  """Simple mesh for GPUs."""
  devices = np.empty((jax.host_count(), jax.local_device_count()),
                     dtype=np.object)
  for device in jax.devices():
    devices[device.process_index, device.id % jax.local_device_count()] = device
  return Mesh(devices, ['data', 'model'])


def default_mesh(num_partitions: int,
                 model_parallel_submesh: Optional[HardwareMesh] = None) -> Mesh:
  """Attempt to return a default mesh for simple cases.

  Args:
    num_partitions: number of partitions to use, will be ignored if
      model_parallel_submesh is provided.
    model_parallel_submesh: 4-tuple that specifies the x,y,z,c submesh to use as
      the model-parallel device tile.

  Returns:
    xmap/pjit 2D Mesh with 'data', 'model' mesh axes.
  """
  last_device = jax.devices()[-1]
  platform = last_device.platform
  device_kind = last_device.device_kind
  bounds = bounds_from_last_device(last_device)

  if model_parallel_submesh:
    return get_mesh(model_parallel_submesh)

  if platform == 'cpu':
    return get_cpu_mesh()
  elif platform == 'gpu':
    return get_gpu_mesh()

  mps = None
  if device_kind in ('TPU v2', 'TPU v3'):
    if num_partitions == 1:
      mps = (1, 1, 1, 1)
    elif num_partitions == 2:
      mps = (1, 1, 1, 2)
    elif num_partitions == 4:
      mps = (2, 1, 1, 2)
    elif num_partitions == 8:
      mps = (2, 2, 1, 2)
    elif num_partitions == 16:
      mps = (4, 2, 1, 2)
  # assume the use of megacore on TPU v4
  elif device_kind == 'TPU v4' and bounds[3] == 1:
    if num_partitions == 1:
      mps = (1, 1, 1, 1)
    elif num_partitions == 2:
      mps = (2, 1, 1, 1)
    elif num_partitions == 4:
      if bounds[0] >= 4:
        mps = (4, 1, 1, 1)
      else:
        mps = (2, 2, 1, 1)
    elif num_partitions == 8:
      if bounds[2] >= 8:
        mps = (1, 1, 8, 1)
      else:
        mps = (4, 2, 1, 1)
    elif num_partitions == 16:
      if bounds[2] >= 16:
        mps = (1, 1, 16, 1)
      elif bounds[0] >= 8:
        mps = (8, 2, 1, 1)
      else:
        mps = (4, 4, 1, 1)

  if mps is None:
    raise ValueError('No default mesh for this configuration: specify '
                     'config.model_parallel_submesh explicitly.')
  return get_mesh(mps)


# Data chunking helper.
# -----------------------------------------------------------------------------
@dataclasses.dataclass
class LocalChunkInfo:
  # The logical slice of an array located on this host's local devices.
  slice: Tuple[slice, ...]
  # A unique index for this host/local chunk among chunks with the same slice.
  replica_id: int


class LocalChunker:
  """Utility class to aid chunking of sharded arrays in multihost settings."""

  def __init__(self, global_mesh: Mesh):
    self.global_mesh = global_mesh
    local_mesh = global_mesh.local_mesh
    first_local_device = local_mesh.devices.reshape(-1)[0]
    host_location = collections.OrderedDict(
        zip(
            global_mesh.shape.keys(),
            list(zip(*np.nonzero(
                global_mesh.devices == first_local_device)))[0]))
    self.num_chunks = collections.OrderedDict()
    self.chunk_ids = collections.OrderedDict()
    self.mesh_axes = list(global_mesh.shape.keys())
    for mesh_axis in self.mesh_axes:
      num_devices_per_chunk = local_mesh.shape[mesh_axis]
      self.num_chunks[mesh_axis] = (
          global_mesh.shape[mesh_axis] // num_devices_per_chunk)
      self.chunk_ids[mesh_axis] = (
          host_location[mesh_axis] // num_devices_per_chunk)

  def get_local_chunk_info(
      self, global_shape: Tuple[int, ...],
      mesh_axes: Sequence[Optional[str]]) -> LocalChunkInfo:
    """Get the local chunk info for a given array shape and sharded axes.

    Args:
      global_shape: the global, unsharded shape of the array to chunk.
      mesh_axes: a sequence of names (or None) of equal rank to `global_shape`
        that specifies which mesh dimensions the array is sharded along.

    Returns:
      LocalChunkInfo containing the logical slices of the array found on this
      host's local devices, as well as the replica index for this chunk among
      chunks with the same slice.  The latter is used to determine which
      host should write this chunk during checkpointing and in calls to
      host_allgather to ensure that only one copy of a particular result is
      gathered.
    """
    local_slice = [slice(None) for dim in global_shape]
    sharded_mesh_axes = set()
    for i, (mesh_axis, size) in enumerate(zip(mesh_axes, global_shape)):
      if not mesh_axis:
        continue
      sharded_mesh_axes.add(mesh_axis)
      if not isinstance(mesh_axis, str):
        raise NotImplementedError('TODO(jekbradbury)')
      chunk_id = self.chunk_ids[mesh_axis]
      chunk_size = size // self.num_chunks[mesh_axis]
      local_slice[i] = slice(chunk_id * chunk_size, (chunk_id + 1) * chunk_size)

    replicated_mesh_axes = [
        mesh_axis for mesh_axis in self.mesh_axes
        if mesh_axis not in sharded_mesh_axes
    ]
    replica_id = 0
    for mesh_axis in replicated_mesh_axes:
      chunk_id = self.chunk_ids[mesh_axis]
      replica_id = replica_id * self.num_chunks[mesh_axis] + chunk_id

    return LocalChunkInfo(tuple(local_slice), replica_id)


# Model parallel sharding specifications.
# -----------------------------------------------------------------------------
def _insert(tpl, idx, x):
  tmp = list(tpl)
  tmp.insert(idx, x)
  return tuple(tmp)


def standard_logical_axis_rules() -> LogicalAxisRules:
  """Default sharding rules for P5X model in terms of logical axes names."""
  return (
      ('batch', 'data'),
      ('vocab', 'model'),
      ('embed', None),
      ('mlp', 'model'),
      ('heads', 'model'),
      ('kv', None),
      ('joined_kv', 'model'),  # joined heads+kv dim in 2D attn param layouts
      ('relpos_buckets', None),
      ('length', None),
      ('layers', None),
      ('stack', None),
  )


# NB: This needs to be top-level for the jax compilation cache.
def _id_fn(x, ix):
  """Identity function for copying parameters to the devices, sharded."""
  # A pure identity such as `lambda x, *: x` can get optimized away, so we
  # include a random.split as a cheap function that cannot be optimized away.
  return x, random.split(jnp.array([ix, ix], dtype=jnp.uint32))


@dataclasses.dataclass
class DataLayout:
  """Represents data layout for the partitioned model."""
  batch_size: int
  shard_id: int
  num_shards: int
  is_first_host_in_replica_set: bool


PartitionedCallable = Callable[..., Any]
CompiledPartitionedCallable = Callable[..., Any]


class BasePartitioner(metaclass=abc.ABCMeta):
  """Interface for partitioning computations."""

  def __init__(self,
               num_partitions: Optional[int],
               model_parallel_submesh: Optional[HardwareMesh] = None,
               params_on_devices: bool = True):
    """Configures the partitioner.

    Args:
      num_partitions: the number of partitions to use. Ignored if
        `model_parallel_submesh` is provided.
      model_parallel_submesh: 4-tuple that specifies the x,y,z,c submesh to use
        as the model-parallel device tile. This submesh is used for the larger
        of the two parameter dimensions, and, if 2-D activation sharding is
        enabled, for the model dimension of activations. The rest of the mesh is
        used for data parallelism and, if 2-D parameter sharding is enabled, the
        other parameter dimension.
      params_on_devices: whether to keep the params on devices, if False -
        params stay in the host memory. Note that some partitioners might ignore
        this setting, for example if they don't support storing all params on
        device memory.
    """
    self._num_partitions = num_partitions
    self._model_parallel_submesh = model_parallel_submesh
    self._params_on_devices = params_on_devices

  def get_data_layout(self,
                      batch_size: Optional[int] = None,
                      host_index: Optional[int] = None) -> DataLayout:
    """Returns filled `DataLayout` based on the partitioned model layout.

    Args:
      batch_size: if set, indicates the requested batch size. The exception will
        be raised if this batch size is not compatible with the layout. If not
        set, the batch size is inferred from the layout.
      host_index: indicates the host index to use for the calculations, if not
        set - use JAX-provided one. Should be in [0, num_hosts) interval and the
        order should match the order of corresponding CPU devices in
        `jax.devices()`.

    Returns:
      Filled `DataLayout` structure.
    """
    if host_index is not None:
      raise NotImplementedError('Explicit host_index is not yet implemented.')
    mesh_size = self._local_chunker.global_mesh.shape['data']
    batch_size = batch_size or mesh_size
    if batch_size % mesh_size:
      raise ValueError(
          f'Batch size ({batch_size}) must be divisible by corresponding mesh '
          f'size ({mesh_size}).')
    num_shards = self._local_chunker.num_chunks['data']
    if batch_size % num_shards:
      raise ValueError(
          f'Batch size ({batch_size}) must be divisible by number of '
          f'replicas ({num_shards}).')
    replica_id = self._local_chunker.get_local_chunk_info((batch_size,),
                                                          ['data']).replica_id
    return DataLayout(
        batch_size=batch_size,
        shard_id=self._local_chunker.chunk_ids['data'],
        num_shards=num_shards,
        is_first_host_in_replica_set=(replica_id == 0))

  def get_local_chunk_info(
      self, global_shape: Tuple[int, ...],
      mesh_axes: Sequence[Optional[str]]) -> LocalChunkInfo:
    """Returns the local chunk info for a given array shape and sharded axes."""
    return self._local_chunker.get_local_chunk_info(global_shape, mesh_axes)

  @property
  def params_on_devices(self):
    return self._params_on_devices

  def move_params_to_devices(self, train_state: TrainState,
                             train_state_axes: TrainState) -> TrainState:
    """Moves the optimizer parameters to devices."""
    p_id_fn = self.partition(
        _id_fn,
        in_axis_resources=(train_state_axes, None),
        out_axis_resources=(train_state_axes, None),
        donate_argnums=(0,))
    train_state, _ = p_id_fn(train_state, jnp.ones((), dtype=jnp.uint32))
    return train_state

  @property
  @abc.abstractmethod
  def _local_chunker(self):
    """Returns the chunker that matches the parameters of this partitioner."""
    raise NotImplementedError

  def get_logical_axes(self, train_state: TrainState) -> TrainState:
    """Returns a copy of TrainState with Optional[AxisNames] as leaves."""
    # By default, return None for the logical axes.
    return train_state.restore_state(
        jax.tree_map(lambda x: None, train_state.state_dict()))

  def get_mesh_axes(self, train_state: TrainState) -> TrainState:
    """Returns a copy of TrainState with Optional[PartitionSpecs] as leaves."""
    raise NotImplementedError

  @abc.abstractmethod
  def partition(
      self,
      fn: Callable,  # pylint: disable=g-bare-generic
      in_axis_resources,
      out_axis_resources,
      static_argnums: Union[int, Sequence[int]] = (),
      donate_argnums: Union[int, Sequence[int]] = ()
  ) -> PartitionedCallable:
    """Partitions the computation using partitioner-specific implementation."""
    raise NotImplementedError

  @abc.abstractmethod
  def compile(self, partitioned_fn: PartitionedCallable,
              *args) -> CompiledPartitionedCallable:
    """Compiles and returns the partitioned function, or the original.

    Args:
      partitioned_fn: The partitioned function.
      *args: Sample arguments to the partitioned function matching the input
        shapes that will be passed to the compiled function.

    Returns:
      The compiled function, or the original if this partitioner does not
      support compilation.
    """
    raise NotImplementedError


class PjittedFnWithContext(PartitionedCallable):
  """Wraps pjitted function to apply the appropriate contexts."""

  def __init__(self,
               pjitted_fn,
               partition_mesh: Mesh,
               logical_axis_rules: flax_partitioning.LogicalRules = ()):
    self._pjitted_fn = pjitted_fn
    self._mesh = partition_mesh
    self._logical_axis_rules = logical_axis_rules

  def __call__(self, *args):
    with mesh(self._mesh.devices,
              self._mesh.axis_names), flax_partitioning.axis_rules(
                  self._logical_axis_rules):
      return self._pjitted_fn(*args)

  def lower(self, *args):
    with mesh(self._mesh.devices,
              self._mesh.axis_names), flax_partitioning.axis_rules(
                  self._logical_axis_rules):
      return self._pjitted_fn.lower(*args)


class BasePjitPartitioner(BasePartitioner):
  """Partitioner that uses T5X version of jax.pjit."""

  @cached_property
  def _local_chunker(self) -> LocalChunker:
    return LocalChunker(self._mesh)

  @cached_property
  def _mesh(self) -> Mesh:
    return default_mesh(self._num_partitions, self._model_parallel_submesh)

  def partition(
      self,
      fn: Callable,  # pylint: disable=g-bare-generic
      in_axis_resources,
      out_axis_resources,
      static_argnums: Union[int, Sequence[int]] = (),
      donate_argnums: Union[int, Sequence[int]] = ()
  ) -> PjittedFnWithContext:
    pjitted = pjit(
        fn,
        in_axis_resources=in_axis_resources,
        out_axis_resources=out_axis_resources,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums)

    return PjittedFnWithContext(pjitted, self._mesh)

  def compile(self, partitioned_fn: PjittedFnWithContext,
              *args) -> CompiledPartitionedCallable:
    return partitioned_fn.lower(*args).compile()


class ModelBasedPjitPartitioner(BasePjitPartitioner):
  """Partitioner that uses P5X version of jax.pjit and model annotations."""

  def __init__(self,
               num_partitions: Optional[int],
               model_parallel_submesh: Optional[HardwareMesh] = None,
               params_on_devices: bool = True,
               logical_axis_rules: Optional[LogicalAxisRules] = None):
    super().__init__(
        num_partitions=num_partitions,
        model_parallel_submesh=model_parallel_submesh,
        params_on_devices=params_on_devices)
    if logical_axis_rules is None:
      logical_axis_rules = standard_logical_axis_rules()
    self._logical_axis_rules = logical_axis_rules

  def partition(
      self,
      fn: Callable,  # pylint: disable=g-bare-generic
      in_axis_resources,
      out_axis_resources,
      static_argnums: Union[int, Sequence[int]] = (),
      donate_argnums: Union[int, Sequence[int]] = ()
  ) -> PjittedFnWithContext:
    pjitted = pjit(
        fn,
        in_axis_resources=in_axis_resources,
        out_axis_resources=out_axis_resources,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums)

    return PjittedFnWithContext(pjitted, self._mesh, self._logical_axis_rules)

  def get_logical_axes(self, train_state: TrainState) -> TrainState:
    """Returns a copy of TrainState with Optional[AxisNames] as leaves."""
    if 'params_axes' not in train_state.other_variables:
      raise ValueError('Missing model params_axes collection.')

    params_axes = flax_partitioning.get_axis_names(
        train_state.other_variables['params_axes'])

    optimizer_axes = train_state._optimizer.optimizer_def.derive_logical_axes(  # pylint: disable=protected-access
        train_state._optimizer, params_axes)  # pylint: disable=protected-access

    return train_state.restore_state(optimizer_axes.state_dict())

  def get_mesh_axes(self, train_state: TrainState) -> TrainState:
    """Returns a copy of TrainState with Optional[PartitionSpecs] as leaves."""
    logical_axes = self.get_logical_axes(train_state)
    with flax_partitioning.axis_rules(self._logical_axis_rules):
      mesh_axes_dict = jax.tree_map(flax_partitioning.logical_to_mesh_axes,
                                    logical_axes.state_dict())

    return logical_axes.restore_state(mesh_axes_dict)


