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

"""Mixture-of-Experts checkpoint overrides."""

import os
from typing import Any, Optional, Union

import clu.data
import jax
import jax.config
from jax.experimental.array_serialization import serialization as array_serialization
from jax.experimental.pjit import pjit
import jax.numpy as jnp
import numpy as np
from t5x import checkpoint_importer
from t5x import checkpoints
from t5x import partitioning
from t5x import train_state as train_state_lib
import tensorflow as tf
import tensorstore as ts

LazyAwaitableArray = checkpoint_importer.LazyAwaitableArray
_ParameterInfo = checkpoints._ParameterInfo  # pylint: disable=protected-access
PartitionSpec = partitioning.PartitionSpec


class UpcycleCheckpointer(checkpoints.Checkpointer):
  """Modified Checkpointer for sparse upcycling (dense-to-sparse) runs.

  This subclass calls modified _read_ts, namely _read_upcycle_ts, which
  broadcasts the checkpoint's dense MLP weights to the model's sparse, expert
  weights. This enables sparsifying dense checkpoints. See also _read_upcycle_ts
  for more details.
  """

  def __init__(
      self,
      train_state: train_state_lib.TrainState,
      partitioner: partitioning.BasePartitioner,
      checkpoints_dir: str,
      num_experts: int,
      dataset_iterator: Optional[
          Union[tf.data.Iterator, clu.data.dataset_iterator.DatasetIterator]
      ] = None,
      *,
      keep: Optional[int] = None,
      save_dtype: jnp.dtype = np.float32,
      restore_dtype: Optional[jnp.dtype] = None,
      keep_dataset_checkpoints: Optional[int] = None,
  ):
    """Checkpointer constructor.

    Args:
      train_state: A train state to be used to determine the structure of the
        parameter tree, and the *full* (non-partitioned) parameter shapes and
        dtypes. Saved and restored train states must match this structure.
      partitioner: The partitioner to use for determining the local chunks
        mapping or to perform params partitioning on restore.
      checkpoints_dir: a path to a directory to save checkpoints in and restore
        them from.
      num_experts: Global number of experts.
      dataset_iterator: An optional iterator to save/restore.
      keep: An optional maximum number of checkpoints to keep. If more than this
        number of checkpoints exist after a save, the oldest ones will be
        automatically deleted to save space.
      save_dtype: Dtype to cast targets to before saving.
      restore_dtype: Optional dtype to cast targets to after restoring. If None,
        no parameter casting is performed.
      keep_dataset_checkpoints: An optional maximum number of data iterators to
        keep. If more than this number of data iterators exist after a save, the
        oldest ones will be automatically deleted to save space.
    """
    super().__init__(
        train_state=train_state,
        partitioner=partitioner,
        checkpoints_dir=checkpoints_dir,
        dataset_iterator=dataset_iterator,
        keep=keep,
        save_dtype=save_dtype,
        restore_dtype=restore_dtype,
        keep_dataset_checkpoints=keep_dataset_checkpoints,
    )

    self._num_experts = num_experts

  def _create_lazy_awaitable_array(
      self,
      param_info: _ParameterInfo,
      maybe_ts_spec: Any,
      ckpt_path: str,
      restore_dtype: Optional[jnp.dtype],
  ) -> LazyAwaitableArray:
    """Creates LazyArray from tensorstore and optionally broadcasts it.

    Does not materialize the array immediately.

    The only difference of this method from that of the parent class is that
    this one calls _read_upcycle_ts instead of _read_ts, which also performs
    broadcasting the MoE weights and optimizer states for sparsely upcycled
    models.

    Args:
      param_info: Information about how to read the parameter, host based sliced
        reads and the like.
      maybe_ts_spec: The tensorstore spec to read the parameter or some other
        object. If this is an array then we will do a host based sliced read on
        it (provided the param_info says to). Anything else we just return.
      ckpt_path: A base location to use when resolving the relative paths in the
        tensorstore spec.
      restore_dtype: Type to restore as. None indicates that no cast is
        requested.

    Returns:
      LazyArray object. If it is an MLP parameter kernel that needs to be
      "sparsified", then the MLP parameter kernel is broadcast to all experts.
    """
    mesh = self._partitioner.mesh
    axes = param_info.axes

    async def get_fn():
      nonlocal mesh
      nonlocal axes
      arr = await _read_upcycle_ts(
          param_info,
          maybe_ts_spec,
          ckpt_path,
          self._num_experts,
          restore_dtype=restore_dtype,
          mesh=mesh,
          axes=axes,
      )

      is_sharded_jax_array = (
          isinstance(arr, jax.Array) and not arr.is_fully_addressable
      )
      if (
          isinstance(arr, (np.ndarray, jnp.ndarray))
          and not is_sharded_jax_array
      ):
        if axes is None:
          axes = PartitionSpec(
              None,
          )
        if restore_dtype is not None:
          arr = arr.astype(restore_dtype)
        arr = jax.make_array_from_callback(
            arr.shape,
            jax.sharding.NamedSharding(mesh, axes),
            lambda idx: arr[idx],
        )
      return arr

    return LazyAwaitableArray.from_tensor_store_spec_or_array(
        maybe_ts_spec, get_fn, dtype=restore_dtype
    )


async def _read_upcycle_ts(
    param_info: _ParameterInfo,
    maybe_ts_spec: Any,
    ckpt_path: str,
    num_experts: int,
    restore_dtype: Optional[jnp.dtype] = None,
    mesh: Optional[jax.sharding.Mesh] = None,
    axes: Optional[jax.sharding.PartitionSpec] = None,
):
  """Reads array from tensorstore and handles broadcasting of expert weights.

  If both `mesh` and `axes` are provided, the method will attempt to restore the
  array as a GlobalDeviceArray.

  This method is adapted from _read_ts() in t5x/checkpoints.py. This variant
  broadcasts dense MLP weights from the checkpoint to the sparse, expert weights
  of the model.

  Args:
    param_info: Information about how to read the parameter, host based sliced
      reads and the like.
    maybe_ts_spec: The tensorstore spec to read the parameter or some other
      object. If this is an array then we will do a host based sliced read on it
      (provided the param_info says to). Anything else we just return.
    ckpt_path: A base location to use when resolving the relative paths in the
      tensorstore spec.
    num_experts: Global number of experts.
    restore_dtype: type to restore as. None indicates that no cast is requested.
    mesh: Mesh object for GDA restoration.
    axes: MeshAxes object for GDA restoration.

  Returns:
    The array. Depending on the value `maybe_ts_spec` it might be read from
    tensorstore, or it might be returned as is. Depending on the values in
    param_info (specifically the `local_chunk_info`) it might be the full value
    or a specific slice. If it is an expert parameter, then it is broadcast to
    all experts.
  """
  if param_info:
    param_name = param_info.name
    m_or_v = param_name.endswith('/m') or param_name.endswith('/v')
    is_expert_param = 'expert/' in param_name

  # If saved as a numpy array, but a partitioned read is requested, return a
  # slice of the array for that host. Otherwise, return the whole thing.
  if isinstance(maybe_ts_spec, np.ndarray) and param_info:
    if mesh is not None and axes is not None:
      # Using GDA, return global array without selecting local chunk.
      return maybe_ts_spec
    elif param_info.local_chunk_info:
      return maybe_ts_spec[param_info.local_chunk_info.slice]
    else:
      return maybe_ts_spec
  # If we have anything else that isn't a tensorstore spec just return it.
  elif not isinstance(maybe_ts_spec, ts.Spec):
    return maybe_ts_spec

  tmp_ts_spec_dict = maybe_ts_spec.to_json()
  # Remove non-required params so that we can open Tensorstore
  # that was created with a different set of params.
  del tmp_ts_spec_dict['metadata']['chunks']
  del tmp_ts_spec_dict['metadata']['compressor']

  # Convert the relative path in the spec to a path based on the checkpoint
  # location. Path and gcs bucket (if applicable) information is updated
  # in-place.
  checkpoints._update_ts_path_from_relative_to_absolute(  # pylint:disable=protected-access
      os.path.dirname(ckpt_path), tmp_ts_spec_dict
  )

  if param_info.shape is not None:
    ts_spec_arr_shape = tuple(tmp_ts_spec_dict['metadata']['shape'])
    # Check that the shapes of the array on disk match the expected shape based
    # on the optimizer that is being restored.
    if (not m_or_v) and is_expert_param:
      shapes_match = ts_spec_arr_shape == param_info.shape[1:]
    else:
      shapes_match = ts_spec_arr_shape == param_info.shape
    if not shapes_match:
      raise ValueError(
          f'Shape of `{param_info.name}` in checkpoint '
          f'{ts_spec_arr_shape} does not match expected '
          f'{param_info.shape}.'
      )

  if (
      'dtype' in tmp_ts_spec_dict and tmp_ts_spec_dict['dtype'] == 'uint16'
  ) or (
      'dtype' in tmp_ts_spec_dict['metadata']
      and tmp_ts_spec_dict['metadata']['dtype'] == '<u2'
  ):
    error_message = (
        'Found unsupported uint16 type in Tensorstore spec: '
        f'{tmp_ts_spec_dict}. Please update saved types to bfloat16.'
    )
    raise ValueError(error_message)

  if restore_dtype is not None:
    tmp_ts_spec_dict = {
        'base': tmp_ts_spec_dict,
        'driver': 'cast',
        'dtype': jnp.dtype(restore_dtype).name,
    }

  if mesh is None or axes is None:
    # Read the array.
    t = await ts.open(tmp_ts_spec_dict, open=True)
    if param_info.local_chunk_info is not None:
      # Just read the subsection we care about.
      t = t[param_info.local_chunk_info.slice]
    arr = await t.read()
  else:
    # If provided, read as GDA.
    if (not m_or_v) and is_expert_param:
      # The checkpoint kernels do not have an expert axis, so we override the
      # specs for "expected" expert parameters from ('expert', ...) to (...).
      checkpoint_axes = jax.sharding.PartitionSpec(*axes[1:])
    else:
      checkpoint_axes = axes

    arr = await array_serialization.async_deserialize(
        jax.sharding.NamedSharding(mesh, checkpoint_axes), tmp_ts_spec_dict
    )

  if (not m_or_v) and is_expert_param:
    if mesh is not None and axes is not None:

      def upcycle(arr):
        """Reads slice from numpy array and broadcasts to experts.

        Args:
          arr: Checkpoint array to be sparsely upcycled.

        Returns:
          Array broadcast to number of expert multiples.
        """
        sl = param_info.local_chunk_info.slice
        # Since sl is the slice generated for the new model, we need to ignore
        # the first (expert) dimension to deal with the checkpoint states.
        arr_slice = arr[sl[1:]]
        return np.repeat(arr_slice[None], num_experts, axis=0)

      with mesh:
        upcycled_axes = axes
        arr = pjit(
            upcycle,
            in_shardings=checkpoint_axes,
            out_shardings=upcycled_axes,
        )(arr)

  return arr
