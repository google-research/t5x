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

"""Utilities for synchronizing and communication across multiple hosts."""
import functools
from typing import Optional
import zlib

import jax
import numpy as np

PyTreeDef = type(jax.tree_structure(None))


# NB: This needs to be top-level for the jax compilation cache.
@functools.partial(jax.pmap, axis_name='hosts')
def _host_allgather_psum(x: PyTreeDef) -> PyTreeDef:
  """Host psum for host_allgather."""
  return jax.lax.psum(x, 'hosts')


def broadcast_one_to_all(in_tree: PyTreeDef,
                         is_source: Optional[bool] = None) -> PyTreeDef:
  """Broadcast data from a source host (host 0 by default) to all other hosts.

  Args:
    in_tree: pytree of arrays - each array *must* have the same shape across the
      hosts.
    is_source: optional bool denoting whether the caller is the source. Only
      'source host' will contribute the data for the broadcast. If None, then
      host 0 is used.

  Returns:
    A pytree matching in_tree where the leaves now all contain the data from the
    first host.
  """
  if is_source is None:
    is_source = jax.process_index() == 0

  def pre_pmap(x):
    if is_source:
      return np.concatenate([
          x[None, ...],
          np.repeat([np.zeros_like(x)],
                    jax.local_device_count() - 1, 0)
      ])
    else:
      return np.repeat([np.zeros_like(x)], jax.local_device_count(), 0)

  def post_pmap(x):
    return jax.device_get(x)[0]

  in_tree = jax.tree_map(pre_pmap, in_tree)
  in_tree = jax.device_get(_host_allgather_psum(in_tree))
  return jax.tree_map(post_pmap, in_tree)


def sync_devices(name: str):
  """Creates a barrier across all hosts/devices."""
  h = np.int32(zlib.crc32(name.encode()))
  assert_same(h, f"sync_devices name mismatch ('{name}')")


def host_allgather(in_tree: PyTreeDef, num_replica_sets: int,
                   replica_set_id: int,
                   is_first_host_in_replica_set: bool) -> PyTreeDef:
  """Gather data from across hosts/replica sets.

  Args:
    in_tree: pytree of arrays - each array _must_ have the same shape across the
      hosts.
    num_replica_sets: int denoting the number of replica sets (least common
      multiples of hosts and replicas) in the computation.
    replica_set_id: int denoting which replica set the current host belongs to.
    is_first_host_in_replica_set: bool denoting whether the current host is the
      first one in its replica set. Only that first host will contribute the
      data for the all-gather from its replica set.

  Returns:
    A pytree matching in_tree where each leaf array has a new leading
    dimension of size num_replica_sets, carrying the data copied from all hosts.
  """
  num_local_devices = jax.local_device_count()

  # We collect data per-host by creating two new axes: a pmap outer axis, and
  # an inner 'host' axis. The latter is filled based on process_index, and the
  # outer only has this single nonzero entry. Thus after a psum, we collect the
  # first member of the outer axis and have a new 'host' dimension such that
  # the returned leaves contain the data gathered from other hosts.
  def pre_pmap(x):
    y = np.zeros((num_local_devices, num_replica_sets, *x.shape), dtype=x.dtype)
    if is_first_host_in_replica_set:
      y[0, replica_set_id] = x
    return y

  def post_pmap(x):
    return jax.device_get(x)[0]

  return jax.tree_map(post_pmap,
                      _host_allgather_psum(jax.tree_map(pre_pmap, in_tree)))


def assert_same(in_tree: PyTreeDef, fail_message: str = ''):
  """Verifies that all the hosts have the same tree of values`."""
  expected = broadcast_one_to_all(in_tree)
  if not jax.tree_util.tree_all(
      jax.tree_map(lambda *x: np.all(np.equal(*x)), in_tree, expected)):
    raise AssertionError(
        f'{fail_message} Expected: {expected}; got: {in_tree}.')
