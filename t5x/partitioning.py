# Copyright 2024 The T5X Authors.
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

"""Utilities for partitioning."""

import abc
import collections
import dataclasses
import functools
import typing
from typing import Any, Callable, Optional, Sequence, Set, Tuple, Union

from absl import logging
import cached_property
from flax import traverse_util
from flax.linen import partitioning as flax_partitioning
import jax
from jax import numpy as jnp
from jax import random
from jax.experimental import multihost_utils
from jax.experimental.mesh_utils import create_hybrid_device_mesh
from jax.experimental.pjit import pjit
from jax.interpreters import pxla
from jax.sharding import Mesh
from jax.sharding import PartitionSpec
import numpy as np
from t5x import train_state as train_state_lib
from t5x.te_helper import TransformerEngineHelper

JaxDevice = jax.Device
TpuMesh = Tuple[int, int, int, int]  # (x, y, z, num_cores).
OtherMesh = Tuple[int, int]
HardwareMesh = Union[TpuMesh, OtherMesh]
TrainState = train_state_lib.TrainState
LogicalAxisRules = Sequence[Tuple[str, Optional[str]]]

if typing.TYPE_CHECKING:  # See b/163639353
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


def with_sharding_constraint(x, axis_resources):
  """Wrapper for lax.with_sharding_constraint, no-op on cpu or outside pjit."""
  if jax.devices()[0].platform == 'cpu' or not global_mesh_defined():
    return x
  else:
    return jax.lax.with_sharding_constraint(x, axis_resources)


# pjit Mesh creation functions.
# -----------------------------------------------------------------------------
def bounds_from_last_device(last_device: jax.Device) -> HardwareMesh:
  """Get the bound from the given last device."""
  # Must be passed the device at the highest-coordinate corner of the
  # relevant mesh, which is a requirement we know is satisfied by the last
  # device in jax.devices().
  if hasattr(last_device, 'coords') and len(last_device.coords) == 3:
    x, y, z = last_device.coords
    return x + 1, y + 1, z + 1, last_device.core_on_chip + 1
  else:
    # On non-TPU platforms, the "mesh" is hosts x devices per host in order
    # to take advantage of faster within-host interconnect.
    return jax.process_count(), jax.local_device_count()


def get_coords(device: jax.Device) -> HardwareMesh:
  """Returns the coordinates of the given device."""
  if hasattr(device, 'coords'):
    return (*device.coords, device.core_on_chip)
  return (device.process_index, device.id % jax.local_device_count())


def global_mesh_defined():
  """Checks if global xmap/pjit mesh resource environment is defined."""
  maps_env = pxla.thread_resources.env
  return maps_env.physical_mesh.devices.shape != ()  # pylint: disable=g-explicit-bool-comparison


def get_mesh(model_parallel_submesh: HardwareMesh,
             input_devices: Sequence[JaxDevice] = (),
             input_local_devices: Sequence[JaxDevice] = (),
             tile_by_host_if_needed: bool = True,
             backend: Optional[str] = None) -> Mesh:
  """Construct an xmap/pjit Mesh for the given model-parallel submesh.

  The resulting mesh has two resource axes: 'model', with the provided submesh
  shape, and 'data', which covers the rest of the mesh.

  Args:
    model_parallel_submesh: a HardwareMesh spec, namely (x,y,z,core) on TPU for
      a single model-parallel replica's "tile" in the physical device mesh. The
      first three elements (`x`, `y`, and `z`) should be factors of the pod
      slice; e.g., if you are using df_4x8, then `x` should be a factor of 4
      (one of 1, 2, 4), `y` should be a factor of 8 (one of 1, 2, 4, 8), and `z`
      must be 1, because TPU v3 slices are only 2D. `z` can be >1 for TPU v4
      (and maybe later TPUs) that allow 3D slices. `core` is the number of cores
      to use from each TPU node. As communication is usually fastest inside the
      same node, if you need a tile of more than 1 core, then
      you should first increase `core`: e.g., for TPU v3, (1,1,1,2) is better
        than (2,1,1,1). To pick a good spec, try a few possible values until you
        get high TPU utilization.
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
    backend: get devices from the pinned backend, if specified. This is
      useful for explicitly specifying the devices other than relying on
      jax_platform_name.

  Returns:
    A xmap / pjit Mesh containing the virtual device mesh with data, model axes.
  """
  input_devices = input_devices or jax.devices(backend)
  input_local_devices = input_local_devices or jax.local_devices(0, backend)
  # Sort input_devices based on coords, as backends might not return devices
  # in order.
  last_device = sorted(input_devices, key=get_coords)[-1]
  last_input_local_devices = sorted(input_local_devices, key=get_coords)[-1]
  logging.info('last device coords : %r\nlast local device coords: %r',
               get_coords(last_device), get_coords(last_input_local_devices))
  global_hardware_mesh = bounds_from_last_device(last_device)
  mesh_ndim = len(global_hardware_mesh)
  local_hardware_mesh = bounds_from_last_device(last_input_local_devices)
  mesh_err = (
      f'each dimension of the model parallel submesh {model_parallel_submesh} '
      'must be a factor of the corresponding dimension of the global device '
      f'mesh {global_hardware_mesh}')
  assert not any(
      g % m
      for g, m in zip(global_hardware_mesh, model_parallel_submesh)), mesh_err
  assert not any(
      g % l for g, l in zip(global_hardware_mesh, local_hardware_mesh))
  devices = np.empty(global_hardware_mesh, dtype=object)
  for device in input_devices:
    device_coords = get_coords(device)
    devices[device_coords] = device
  tile_by_host = tile_by_host_if_needed
  if len(global_hardware_mesh) == 4:
    # enable contiguous local chunks without host tiling by making Z major
    global_hardware_mesh = typing.cast(Tuple[int, int, int, int],
                                       global_hardware_mesh)
    model_parallel_submesh = typing.cast(Tuple[int, int, int, int],
                                         model_parallel_submesh)
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

    def dh_dd_mh_md(g: int, m: int, l: int) -> Tuple[int, int, int, int]:
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
  logging.info('global_mesh axis_names: %s', global_mesh.axis_names)
  logging.info('global_mesh devices: %s', global_mesh.devices)
  logging.info('global_mesh devices shape: %s', global_mesh.devices.shape)
  return global_mesh


def get_cpu_mesh() -> Mesh:
  """Trivial mesh for CPU Testing."""
  devices = np.empty(
      (jax.process_count(), jax.local_device_count()), dtype=object
  )
  for device in jax.devices():
    devices[device.process_index, device.id % jax.local_device_count()] = device
  return Mesh(devices, ['data', 'model'])


def get_gpu_mesh(num_partitions: int) -> Mesh:
  """Mesh for GPUs that preferentially places 'model' on NVLink."""
  nvlink_size = jax.local_device_count()
  dcn_size = jax.process_count()
  nvlink_mp = min(num_partitions, nvlink_size)
  nvlink_dp, extra1 = divmod(nvlink_size, nvlink_mp)
  dcn_mp, extra2 = divmod(num_partitions, nvlink_mp)
  assert not (extra1 or extra2), ('number of partitions on GPU must be a factor'
                                  ' or multiple of the number of local devices')
  dcn_dp = dcn_size // dcn_mp

  devices = create_hybrid_device_mesh(
      mesh_shape=[nvlink_dp, nvlink_mp],
      dcn_mesh_shape=[dcn_dp, dcn_mp],
      process_is_granule=True)

  global_mesh = Mesh(devices, ['data', 'model'])
  logging.info('global_mesh axis_names: %s', global_mesh.axis_names)
  logging.info('global_mesh devices: %s', global_mesh.devices)
  return global_mesh


def default_mesh(
    num_partitions: int,
    model_parallel_submesh: Optional[HardwareMesh] = None,
    backend: Optional[str] = None,
    ici_mesh_shape: Optional[HardwareMesh] = None,
    dcn_mesh_shape: Optional[HardwareMesh] = None,
) -> Mesh:
  """Attempt to return a default mesh for simple cases.

  Args:
    num_partitions: number of partitions to use, will be ignored if
      model_parallel_submesh is provided.
    model_parallel_submesh: 4-tuple that specifies the x,y,z,c submesh to use as
      the model-parallel device tile.
    backend: get devices from the pinned backend, if specified. This is useful
      for explicitly specifying the devices other than relying on
      jax_platform_name.
    ici_mesh_shape: Shape of the logical mesh used for SPMD parallelism in each
      slice. The meaning of each mesh axis is defined by mesh_axis_names, so
      these two params must be the same length. If dcn_mesh_shape is present,
      the overall mesh is the product of ici_mesh_shape and dcn_mesh_shape. For
      example, an ici_mesh_shape of [2, 3, 4] with mesh_axis_names ['replica',
      'data', 'model'] indicates 2-way replica parallelism, 3-way data
      parallelism, and 4-way model parallelism over 24 devices. None, the
      default, is equivalent to a sequence of ones and means that the model is
      placed on a single device.
    dcn_mesh_shape: Shape of the logical mesh used for SPMD parallelism over
      multiple slices. The overall mesh is the product of ici_mesh_shape and
      dcn_mesh_shape, and the meaning of each mesh axis is defined by
      mesh_axis_names, so these three params must be the same length.

  Returns:
    xmap/pjit 2D Mesh with 'data', 'model' mesh axes if single-slice, otherwise
    3D Mesh with 'replica', 'data', and 'model' mesh axes.
  """
  devices = jax.devices(backend)
  last_device = devices[-1]
  platform = last_device.platform
  device_kind = last_device.device_kind
  bounds = bounds_from_last_device(last_device)

  if ici_mesh_shape is not None and dcn_mesh_shape is not None:
    device_mesh = create_hybrid_device_mesh(
        ici_mesh_shape,
        dcn_mesh_shape,
        devices=devices,
    )
    multi_slice_global_mesh = Mesh(device_mesh, ['replica', 'data', 'model'])
    logging.info(
        'multi_slice_global_mesh axis_names: %s',
        multi_slice_global_mesh.axis_names,
    )
    logging.info(
        'multi_slice_global_mesh devices: %s', multi_slice_global_mesh.devices
    )
    logging.info(
        'multi_slice_global_mesh devices shape: %s',
        multi_slice_global_mesh.devices.shape,
    )
    return multi_slice_global_mesh

  if model_parallel_submesh:
    return get_mesh(model_parallel_submesh, backend=backend)

  if platform == 'cpu':
    return get_cpu_mesh()
  elif platform == 'gpu':
    return get_gpu_mesh(num_partitions)

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
  elif (device_kind == 'TPU v4' or
        device_kind == 'TPU v4 lite') and bounds[3] == 1:
    if num_partitions == 1:
      mps = (1, 1, 1, 1)
    elif num_partitions == 2:
      mps = (1, 2, 1, 1)
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
      elif bounds[0] >= 4:
        mps = (4, 4, 1, 1)
      else:
        mps = (2, 2, 4, 1)

  if mps is None:
    raise ValueError(
        'No default mesh for this configuration: specify '
        'config.model_parallel_submesh explicitly. \n'
        f'Platform: {platform}\n'
        f'Device kind: {device_kind}\n'
        f'Num partitions: {num_partitions}\n'
        f'Bounds: {bounds}'
    )
  return get_mesh(mps, backend=backend)


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
      chunks with the same slice. The latter is used to determine which
      host should write this chunk during checkpointing.
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

    replica_id = self.get_replica_id(sharded_mesh_axes)

    return LocalChunkInfo(tuple(local_slice), replica_id)

  def get_shard_id(self, sharded_mesh_axes: str | Set[Optional[str]]) -> int:
    """Given mesh axes used for sharding, computes current host's shard id.

    To give an example, let's say there are two axes globally: replica, data,
    and model, the mesh axes for sharding is ('replica', 'data'), which means we
    are going to partition an array along 'replica' and 'data' axes.
    The shard_id is to show the index of the current local host along the
    sharding axes (in this example, it's 'replica' and 'data' axes).

    More concretely, let's say we have 4 local hosts, and we use 'replica' and
    'data' axes for data parallel (2 hosts along the replica axis, and 2 host
    along the data axis). The host located in ('replica': 0, 'data': 0), we
    should assign data shard-0 to it. For host ('replica': 0, 'data': 1), we
    assign shard-1. For host ('replica': 1, 'data': 0), we assign shard-2.
    For host ('replica': 1, 'data': 1), we assign shard-3.

    Note: the host location along 'replica' and 'data' axes, e.g.,
    ('replica': 0, 'data': 0) is named chunk_id and stored in
    self._local_chunker.chunk_ids[axis].

    Args:
      sharded_mesh_axes: the mesh axes for sharding.

    Returns:
      the index of the current local host along the sharding axes.
    """
    if isinstance(sharded_mesh_axes, str):
      sharded_mesh_axes = (sharded_mesh_axes,)

    shard_id = 0
    for mesh_axis in sharded_mesh_axes:
      chunk_id = self.chunk_ids[mesh_axis]
      shard_id = shard_id * self.num_chunks[mesh_axis] + chunk_id

    return shard_id

  def get_replica_id(self, sharded_mesh_axes: str | Set[Optional[str]]) -> int:
    """Given mesh axes used for sharding, computes current host's replica id.

    To give an example, let's say there are two axes globally: data, and model,
    the mesh axes for sharding is ('data', ), which means we are going to
    partition an array along 'data' axis and replicate it along 'model' axis.
    The replica_id is to show the index of the current local host along the
    'model' axis.

    Args:
      sharded_mesh_axes: the mesh axes for sharding.

    Returns:
      the index of the current local host along the non-sharding axes (i.e.,
      replicating axes).
    """
    if isinstance(sharded_mesh_axes, str):
      sharded_mesh_axes = (sharded_mesh_axes,)

    replicated_mesh_axes = [
        mesh_axis for mesh_axis in self.mesh_axes
        if mesh_axis not in sharded_mesh_axes
    ]
    replica_id = 0
    for mesh_axis in replicated_mesh_axes:
      chunk_id = self.chunk_ids[mesh_axis]
      replica_id = replica_id * self.num_chunks[mesh_axis] + chunk_id

    return replica_id


def standard_logical_axis_rules(
    activation_partitioning_dims: int = 1,
    parameter_partitioning_dims: int = 1,
    additional_rules: Optional[LogicalAxisRules] = None) -> LogicalAxisRules:
  """Default sharding rules for T5X model in terms of logical axis names.

  Args:
    activation_partitioning_dims: enables 2-D activation sharding when set to 2.
    parameter_partitioning_dims: enables 2-D parameter sharding when set to 2.
    additional_rules: additional rules (a sequence of tuples) that will be
      appended to the standard rules.

  Returns:
    Sequence of logical axis rules
  """
  logging.info(
      '`activation_partitioning_dims` = %d, `parameter_partitioning_dims` = %d',
      activation_partitioning_dims, parameter_partitioning_dims)

  if activation_partitioning_dims == 1 and parameter_partitioning_dims == 1:
    rules = [
        ('batch', 'data'),
        ('vocab', 'model'),
        ('embed', None),
        ('mlp', 'model'),
        ('heads', 'model'),
        ('kv', None),
        ('joined_kv', 'model'),  # joined heads+kv dim in 2D attn param layouts
    ]
  elif activation_partitioning_dims == 2 and parameter_partitioning_dims == 1:
    rules = [
        ('batch', 'data'),
        ('vocab', 'model'),
        ('mlp', 'model'),
        ('heads', 'model'),
        ('kv', None),
        ('joined_kv', 'model'),
        ('embed', 'model'),
    ]
  elif activation_partitioning_dims == 1 and parameter_partitioning_dims == 2:
    rules = [
        ('batch', 'data'),
        ('vocab', 'model'),
        ('mlp', 'model'),
        ('heads', 'model'),
        ('kv', None),
        ('joined_kv', 'model'),
        ('embed', 'data'),
    ]
  elif activation_partitioning_dims == 2 and parameter_partitioning_dims == 2:
    rules = [
        ('batch', 'data'),
        ('vocab', 'model'),
        ('mlp', 'model'),
        ('heads', 'model'),
        ('kv', None),
        ('joined_kv', 'model'),
        ('embed', 'model'),
        ('embed', 'data'),
    ]
  else:
    raise ValueError(
        f'`activation_partitioning_dims` = {activation_partitioning_dims} '
        f'`parameter_partitioning_dims` = {parameter_partitioning_dims} '
        'is not supported.')

  # Add the common rules for the replicated logical axes names.
  replicated_rules = [
      ('relpos_buckets', None),
      ('abspos_buckets', None),
      ('length', None),
      ('layers', None),
      ('stack', None),
      ('mlp_activations', None),
  ]
  rules.extend(replicated_rules)

  if additional_rules:
    rules.extend(additional_rules)

  rules = TransformerEngineHelper.extend_logical_axis_rules(rules)

  return rules


# NB: This needs to be top-level for the jax compilation cache.
def _id_fn(x, ix):
  """Identity function for copying parameters to the devices, sharded."""
  # A pure identity such as `lambda x, *: x` can get optimized away, so we
  # include a random.split as a cheap function that cannot be optimized away.
  y = random.split(random.PRNGKey(jnp.array(ix, dtype=jnp.uint32)))
  return x, y


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
  """Interface for partitioning computations across hardware devices."""

  def __init__(
      self,
      num_partitions: Optional[int] = None,
      model_parallel_submesh: Optional[HardwareMesh] = None,
      params_on_devices: bool = True,
      backend: Optional[str] = None,
      ici_mesh_shape: Optional[HardwareMesh] = None,
      dcn_mesh_shape: Optional[HardwareMesh] = None,
  ):
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
      backend: get devices from the pinned backend, if specified. This is useful
        for explicitly specifying the devices other than relying on
        jax_platform_name.
      ici_mesh_shape: Shape of the logical mesh used for SPMD parallelism in
        each slice. The meaning of each mesh axis is defined by mesh_axis_names,
        so these two params must be the same length. If dcn_mesh_shape is
        present, the overall mesh is the product of ici_mesh_shape and
        dcn_mesh_shape. For example, an ici_mesh_shape of [2, 3, 4] with
        mesh_axis_names ['replica', 'data', 'mdl'] indicates 2-way replica
        parallelism, 3-way data parallelism, and 4-way model parallelism over 24
        devices. None, the default, is equivalent to a sequence of ones and
        means that the model is placed on a single device.
      dcn_mesh_shape: Shape of the logical mesh used for SPMD parallelism over
        multiple slices. The overall mesh is the product of ici_mesh_shape and
        dcn_mesh_shape, and the meaning of each mesh axis is defined by
        mesh_axis_names, so these three params must be the same length.
    """

    if not num_partitions and not model_parallel_submesh:
      raise ValueError('At least one of `num_partitions` or '
                       '`model_parallel_submesh` must be set.')

    if model_parallel_submesh is not None and len(model_parallel_submesh) != 4:
      logging.error(
          (
              '`model_parallel_submesh` must be either None or a 4-tuple. Got'
              ' `model_parallel_submesh`=%r. A ValueError will be raised'
              ' beginning March 1, 2022.'
          ),
          model_parallel_submesh,
      )

    if bool(num_partitions) and bool(model_parallel_submesh):
      logging.error(
          'At most one of `num_partitions` or `model_parallel_submesh` can be '
          'set. Got `num_partitions=%r` and `model_parallel_submesh`=%r. A '
          'ValueError will be raised beginning March 21, 2022.',
          num_partitions,
          model_parallel_submesh,
      )

    self._num_partitions = num_partitions
    self._model_parallel_submesh = model_parallel_submesh
    self._params_on_devices = params_on_devices
    if ici_mesh_shape is None or dcn_mesh_shape is None:
      self._data_axis = 'data'
    else:
      self._data_axis = ('replica', 'data')
    self._backend = backend
    self._ici_mesh_shape = ici_mesh_shape
    self._dcn_mesh_shape = dcn_mesh_shape

  @property
  def mesh(self) -> Mesh:
    raise NotImplementedError

  @property
  def data_partition_spec(self) -> PartitionSpec:
    return PartitionSpec(self._data_axis)

  @property
  def data_mesh_size(self) -> int:
    """Data mesh size.

    Data mesh size is defined as the number of global devices involved to
    carry out data parallel. Let's say we have a global mesh: ('replica': 2,
    'data': 4, 'model': 2), and axes 'replica' and 'data' are responsible for
    the data parallel, that means we have 2*4 = 8 devices involved - i.e., data
    mesh size is 8.

    Returns:
      the id of the shard for the axes being replicated among the devices used
      to shard the sharded_mesh_axes.
    """
    data_submesh_sizes = (
        [self.mesh.shape[self._data_axis]]
        if isinstance(self._data_axis, str)
        else [self.mesh.shape[axis] for axis in self._data_axis]
    )
    data_mesh_size = functools.reduce(lambda x, y: x * y, data_submesh_sizes)
    return data_mesh_size

  @property
  def data_shards(self) -> int:
    """Number of data shards.

    Let's say we are dealing with 2 slices of df4x2 TPUs. In data pipeline
    we need prepare / send one data shard to each local host. This means, we
    need 4 shards since we have 4 local hosts. How to infer the number of hosts
    from mesh information? In this case, we have a global mesh: ('replica': 2,
    'data': 8, 'model': 2). Each local host (i.e., df2x2) has this local mesh:
    ('replica': 1, 'data': 4, 'model': 2). By dividing global mesh with local
    mesh, we can get the count of hosts.

    Returns:
      Number of data shards. Each shard will be sent to one local host.
    """
    data_chunks = (
        [self._local_chunker.num_chunks[self._data_axis]]
        if isinstance(self._data_axis, str)
        else [self._local_chunker.num_chunks[axis] for axis in self._data_axis]
    )
    data_shards = functools.reduce(lambda x, y: x * y, data_chunks)
    return data_shards

  @property
  def data_shard_id(self) -> int:
    """Data shard id for the current host.

    Returns:
      Index of data shard that will be sent to the current local host.
    """
    return self._local_chunker.get_shard_id(self._data_axis)

  def get_data_layout(
      self, batch_size: Optional[int] = None, host_index: Optional[int] = None
  ) -> DataLayout:
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
    if self._data_axis is None:
      return DataLayout(
          batch_size=batch_size,
          shard_id=0,
          num_shards=1,
          is_first_host_in_replica_set=(jax.process_index() == 0))

    batch_size = batch_size or self.data_mesh_size
    if batch_size % self.data_mesh_size:
      raise ValueError(
          f'Batch size ({batch_size}) must be divisible by corresponding '
          f'data mesh size ({self.data_mesh_size}).'
      )

    if batch_size % self.data_shards:
      raise ValueError(
          f'Batch size ({batch_size}) must be divisible by number of '
          f'data shards ({self.data_shards}).'
      )
    replica_id = self._local_chunker.get_replica_id(self._data_axis)
    return DataLayout(
        batch_size=int(batch_size),
        shard_id=int(self.data_shard_id),
        num_shards=int(self.data_shards),
        is_first_host_in_replica_set=(replica_id == 0),
    )

  def get_local_chunk_info(
      self, global_shape: Tuple[int, ...],
      mesh_axes: Sequence[Optional[str]]) -> LocalChunkInfo:
    """Returns the local chunk info for a given array shape and sharded axes."""
    return self._local_chunker.get_local_chunk_info(global_shape, mesh_axes)

  @property
  def params_on_devices(self):
    return self._params_on_devices

  @params_on_devices.setter
  def params_on_devices(self, value):
    self._params_on_devices = value

  def move_params_to_devices(self, train_state: TrainState,
                             train_state_axes: TrainState) -> TrainState:
    """Moves the optimizer parameters to devices."""
    p_id_fn = self.partition(
        _id_fn,
        in_axis_resources=(train_state_axes, None),
        out_axis_resources=(train_state_axes, None),
        donate_argnums=(0,))
    if jax.process_count() > 1:
      train_state = host_local_array_to_global_array(
          train_state, self.mesh, train_state_axes
      )
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
        jax.tree.map(lambda x: None, train_state.state_dict())
    )

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
    """Partitions the computation using partitioner-specific implementation.

    Args:
      fn: the function to partition.
      in_axis_resources: Pytree of structure matching that of arguments to `fn`,
        with all actual arguments replaced by resource assignment
        specifications. It is also valid to specify a pytree prefix (e.g. one
        value in place of a whole subtree), in which case the leaves get
        broadcast to all values in that subtree.
        The valid resource assignment specifications are:
          `None`: in which case the value will be replicated on all devices
          `PartitionSpec`: a tuple of length at most equal to the rank of the
            partitioned value. Each element can be a `None`, a mesh axis or a
            tuple of mesh axes, and specifies the set of resources assigned to
            partition the value's dimension matching its position in the spec.
      out_axis_resources: Like `in_axis_resources`, but specifies resource
        assignment for function outputs.
      static_argnums: an optional int or collection of ints that specify which
        positional arguments to treat as static (compile-time constant) in the
        partitioned function.
      donate_argnums: an optional int or collection of ints that specify which
        argument buffers are "donated" to the computation. It is safe to donate
        argument buffers if you no longer need them once the computation has
        finished.

    Returns:
      A partitioned version of the input function.
    """
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

  def __call__(self, *args, **kwargs):
    with Mesh(self._mesh.devices,
              self._mesh.axis_names), flax_partitioning.axis_rules(
                  self._logical_axis_rules):
      return self._pjitted_fn(*args, **kwargs)

  def lower(self, *args, **kwargs):
    with Mesh(self._mesh.devices,
              self._mesh.axis_names), flax_partitioning.axis_rules(
                  self._logical_axis_rules):
      return self._pjitted_fn.lower(*args, **kwargs)

  def lower_and_compile(self, *args, **kwargs):
    with Mesh(self._mesh.devices,
              self._mesh.axis_names), flax_partitioning.axis_rules(
                  self._logical_axis_rules):
      return self._pjitted_fn.lower(*args, **kwargs).compile()



class BasePjitPartitioner(BasePartitioner):
  """Partitioner that uses T5X version of jax.pjit."""

  @cached_property
  def _local_chunker(self) -> LocalChunker:
    return LocalChunker(self.mesh)

  @cached_property
  def mesh(self) -> Mesh:
    return default_mesh(
        self._num_partitions,
        self._model_parallel_submesh,
        self._backend,
        self._ici_mesh_shape,
        self._dcn_mesh_shape,
    )

  def partition(
      self,
      fn: Callable,  # pylint: disable=g-bare-generic
      in_axis_resources,
      out_axis_resources,
      static_argnums: Union[int, Sequence[int]] = (),
      donate_argnums: Union[int, Sequence[int]] = (),
  ) -> PjittedFnWithContext:
    pjitted = pjit(
        fn,
        in_shardings=in_axis_resources,
        out_shardings=out_axis_resources,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
    )

    return PjittedFnWithContext(pjitted, self.mesh)

  def compile(self, partitioned_fn: PjittedFnWithContext,
              *args) -> CompiledPartitionedCallable:
    return partitioned_fn.lower_and_compile(*args)


class PjitPartitioner(BasePjitPartitioner):
  """Partitioner that uses named axes and jax.pjit."""

  def __init__(
      self,
      num_partitions: Optional[int] = None,
      model_parallel_submesh: Optional[HardwareMesh] = None,
      params_on_devices: bool = True,
      backend: Optional[str] = None,
      ici_mesh_shape: Optional[HardwareMesh] = None,
      dcn_mesh_shape: Optional[HardwareMesh] = None,
      logical_axis_rules: Optional[LogicalAxisRules] = None,
  ):
    """PjitPartitioner constructor.

    See https://github.com/google-research/text-to-text-transfer-transformer/blob/main/README.mdx/usage/partitioning for details.

    Args:
      num_partitions: an integer that specifies the size of the model parallel
        submesh to be automatically selected for the current topology. See
        `model_parallel_submesh` for details on how this submesh is used.
        Mutually exclusive with `model_parallel_submesh`.
      model_parallel_submesh: is a 4-tuple that specifies the `(x, y, z, c)`
        submesh model-parallel device tile, an axis of accelerator parallelism
        orthogonal to data parallelism. Array axes in a model's parameters or
        activations can be sharded over this submesh using axis rules (see
        `logical_axis_rules`) that map them to 'model'. The effective number of
        model sub-partitions is equal to `np.prod(model_parallel_submesh)` and
        must evenly divide the total number of devices (i.e.,
        `jax.device_count() % np.prod(model_parallel_submesh) == 0`). The rest
        of the TPU mesh is the data parallel submesh, providing
        `jax.device_count() // np.prod(model_parallel_submesh)` partitions. It
        is used for data (batch) parallelism and to shard other array axes that
        are mapped to 'data'. This argument is mutually exclusive with
        `num_partitions`.
      params_on_devices: whether to keep the params on devices, if False -
        params stay in the host memory. Note that some partitioners might ignore
        this setting, for example if they don't support storing all params on
        device memory.
      backend: get devices from the pinned backend, if specified. This is useful
        for explicitly specifying the devices other than relying on
        jax_platform_name.
      ici_mesh_shape: Shape of the logical mesh used for SPMD parallelism in
        each slice. The meaning of each mesh axis is defined by mesh_axis_names,
        so these two params must be the same length. If dcn_mesh_shape is
        present, the overall mesh is the product of ici_mesh_shape and
        dcn_mesh_shape. For example, an ici_mesh_shape of [2, 3, 4] with
        mesh_axis_names ['replica', 'data', 'model'] indicates 2-way replica
        parallelism, 3-way data parallelism, and 4-way model parallelism over 24
        devices. None, the default, is equivalent to a sequence of ones and
        means that the model is placed on a single device.
      dcn_mesh_shape: Shape of the logical mesh used for SPMD parallelism over
        multiple slices. The overall mesh is the product of ici_mesh_shape and
        dcn_mesh_shape, and the meaning of each mesh axis is defined by
        mesh_axis_names, so these three params must be the same length.
      logical_axis_rules: a priority-ordered sequence of KV tuples that maps
        logical axis names to either `None` (not sharded), 'model' (to shard
        across the model-parallel submesh), or 'data' (to shard across the
        data-parallel submesh).
    """
    super().__init__(
        num_partitions=num_partitions,
        model_parallel_submesh=model_parallel_submesh,
        params_on_devices=params_on_devices,
        backend=backend,
        ici_mesh_shape=ici_mesh_shape,
        dcn_mesh_shape=dcn_mesh_shape,
    )
    if logical_axis_rules is None:
      logical_axis_rules = standard_logical_axis_rules()
    if ici_mesh_shape is not None and dcn_mesh_shape is not None:
      # Split batch over new replica axis.
      logical_axis_rules = (
          (k, ('replica', 'data') if k == 'batch' else v)
          for k, v in logical_axis_rules
      )
    self._logical_axis_rules = tuple(logical_axis_rules)
    (self._data_axis,) = flax_partitioning.logical_to_mesh_axes(
        ['batch'], self._logical_axis_rules
    )

  def partition(
      self,
      fn: Callable,  # pylint: disable=g-bare-generic
      in_axis_resources,
      out_axis_resources,
      static_argnums: Union[int, Sequence[int]] = (),
      donate_argnums: Union[int, Sequence[int]] = ()
  ) -> PjittedFnWithContext:
    """Partitions the function using jax.pjit."""
    pjitted = pjit(
        fn,
        in_shardings=in_axis_resources,
        out_shardings=out_axis_resources,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
    )

    return PjittedFnWithContext(pjitted, self.mesh, self._logical_axis_rules)

  @property
  def logical_axis_rules(self):
    """Returns the logical axis rules."""
    return self._logical_axis_rules

  def get_logical_axes(self, train_state: TrainState) -> TrainState:
    """Returns a copy of TrainState with Optional[AxisNames] as leaves."""
    return train_state.as_logical_axes()

  def get_mesh_axes(self, train_state: TrainState) -> TrainState:
    """Returns a copy of TrainState with Optional[PartitionSpecs] as leaves."""
    logical_axes = self.get_logical_axes(train_state)

    def _logical_to_mesh_axes(param_name, logical_axes):
      if logical_axes is None:
        return None
      elif logical_axes is traverse_util.empty_node:
        return traverse_util.empty_node
      try:
        return flax_partitioning.logical_to_mesh_axes(logical_axes,
                                                      self._logical_axis_rules)
      except ValueError as e:
        raise ValueError(f'Failed to map logical axes for {param_name}') from e

    flat_logical_axes = traverse_util.flatten_dict(
        logical_axes.state_dict(), keep_empty_nodes=True, sep='/')
    flat_mesh_axes = {
        k: _logical_to_mesh_axes(k, v) for k, v in flat_logical_axes.items()
    }

    return logical_axes.restore_state(
        traverse_util.unflatten_dict(flat_mesh_axes, sep='/'))


# arr_tree is a PyTree of jax.Array or np.ndarray and
# pspecs is PyTree[jax.sharding.PartitionSpec]
def host_local_array_to_global_array(arr_tree, mesh: jax.sharding.Mesh, pspecs):
  pspecs = jax.tree.map(
      lambda x: PartitionSpec() if x is None else x,
      pspecs,
      is_leaf=lambda x: x is None,
  )
  return multihost_utils.host_local_array_to_global_array(
      arr_tree, mesh, pspecs
  )
