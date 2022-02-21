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

"""Utilities for processing optimizer states."""

import re
from typing import Any, Mapping, Optional, Sequence, Tuple

from absl import logging
from flax import traverse_util


def tensorstore_leaf(_, value):
  """Detect if the node is a serialized tensorstore spec.

  Args:
    _: The unused name of the current item.
    value: The value of the possible leaf.

  Returns:
    True if the value represents a tensorstore spec, False otherwise.
  """
  # It is a tensorstore leaf if it at least has `driver`, `kvstore` and
  # `metadata` in its keys, sometime they have additional ones like `dtype` or
  # `transform`.
  return set(value.keys()) >= {"driver", "kvstore", "metadata"}


def flatten_state_dict(state_dict, keep_empty_nodes: bool = False):
  """Flatten a dictionary until an array or tensorstore is reached.

  Args:
    state_dict: Optimizer state as nested dictionary.
    keep_empty_nodes: Whether to keep empty node, for example, empty param
      states from simple optimizers or non-touched parameter states in a
      multioptimizer.

  Returns:
    Flattened dictionary, though keeping tensor store state unflattened.
  """
  return traverse_util.flatten_dict(
      state_dict,
      is_leaf=tensorstore_leaf,
      keep_empty_nodes=keep_empty_nodes,
      sep="/")


def get_name_tree(state_dict, keep_empty_nodes: bool = False):
  """Returns new state_dict with leaves as joined path keys separated by "/"."""
  return traverse_util.unflatten_dict({
      k: "/".join(k) for k in traverse_util.flatten_dict(
          state_dict, keep_empty_nodes=keep_empty_nodes)
  })


def intersect_state(
    state_dict: Mapping[str, Any],
    intersect_state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
  """Drops non-matching entries from `state_dict`.

  Args:
    state_dict: nested dict of optimizer state
    intersect_state_dict: nested dict of entries to keep

  Returns:
    nested dict like `state_dict` but with extra keys removed
  """
  state_dict_flat = flatten_state_dict(state_dict)
  intersect_state_dict_flat = flatten_state_dict(intersect_state_dict)

  for k in list(state_dict_flat):
    if k not in intersect_state_dict_flat:
      state_dict_flat.pop(k)
      logging.warning("Ignoring param=%s from checkpoint", k)

  state_dict = traverse_util.unflatten_dict(state_dict_flat, sep="/")

  return state_dict


def merge_state(state_dict: Mapping[str, Any],
                from_scratch_state: Mapping[str, Any]) -> Mapping[str, Any]:
  """Inserts new entries into `state_dict`.

  Args:
    state_dict: nested dict of optimizer state
    from_scratch_state: nested dict of entries to insert

  Returns:
    a nested dict like `state_dict` but with extra entries from
      `from_scratch_state` inserted
  """
  state_dict_flat = flatten_state_dict(state_dict)
  from_scratch_state_flat = flatten_state_dict(from_scratch_state)

  for k in from_scratch_state_flat:
    if k not in state_dict_flat:
      logging.warning("Initializing param=%s from scratch", k)
      state_dict_flat[k] = from_scratch_state_flat[k]

  state_dict = traverse_util.unflatten_dict(state_dict_flat, sep="/")

  return state_dict


def apply_assignment_map(ckpt_optimizer_state,
                         optimizer_state,
                         assignment_map: Sequence[Tuple[str, Optional[str]]],
                         require_all_rules_match: bool = True,
                         *,
                         is_resuming: bool = False):
  """Applies an assignment map to a checkpoint optimizer state.

  In contrast to previous implementations, this has a switch whether to require
  that all rules match, and has somewhat-custom-but-sensible replacement rules:

   1. old keys that are matched are removed.
   2. old keys that don't match are retained.
   3. if two new keys map to the same old key, they both get assigned to its
      value.
   4. if a new key isn't mapped but is in the checkpoint, it is copied over.
   5. new keys with None-valued replacement patterns are removed.

  Args:
    ckpt_optimizer_state: Optimizer state in the checkpoint (usually, previous
      model).
    optimizer_state: optimizer state in the current model.
    assignment_map: List of tuples (matcher, replacement) where matcher is a
      regex, and replacement is a string replacement (possibly with
      regex-compatible group match codes) or None if the matching state should
      be dropped.
    require_all_rules_match: Whether to require that all rules match.
    is_resuming: Whether we are resuming a training run (True) or initializing a
      new one (False).

  Returns:
    New, remapped optimizer state.
  """
  if is_resuming:
    # Do not apply the transformation when resuming after a temporary stop.
    # This ensures that the transformation will only happen once.
    return ckpt_optimizer_state

  flat_ckpt = flatten_state_dict(ckpt_optimizer_state)
  unmapped_old_keys = flat_ckpt.copy()
  result = {}
  explicitly_skipped_keys = set()
  flat_opt = flatten_state_dict(optimizer_state)

  used_patterns = set()
  for k in flat_opt:
    for pattern, repl in assignment_map:
      p_match = re.fullmatch(pattern, k)
      if p_match:
        # Skip initialization if the replacement pattern for this key is None.
        if repl is None:
          explicitly_skipped_keys.add(k)
          used_patterns.add(pattern)
          logging.info(
              "Skipping optimizer param=%s, which had a None "
              "replacement using pattern=%s in the assignment map.", k, pattern)
          break

        old_k = p_match.expand(repl)
        used_patterns.add(pattern)

        # Remove the old key, but read the value from the original dict since
        # it's OK if it was referenced twice.
        unmapped_old_keys.pop(old_k, None)
        try:
          result[k] = flat_ckpt[old_k]
          logging.info(
              "Assigning checkpoint param=%s to optimizer param=%s "
              "using pattern=%s", old_k, k, pattern)
        except KeyError:
          raise ValueError(
              f"Parameter '{old_k}' does not exist in restore checkpoint. "
              f"Must be one of: {sorted(flat_ckpt.keys())}")
        break

  # Now re-add the unmapped keys. This is a 2-step process so that the `pop()`
  # call above doesn't mis-fire if the assignment map "rotates" a chain of keys.
  for key, v in unmapped_old_keys.items():
    if key not in explicitly_skipped_keys:
      result[key] = v

  # If any new keys weren't mapped, but are in the old checkpoint, copy those.
  for key in set(flat_opt) - set(result):
    if key in explicitly_skipped_keys:
      pass
    elif key in flat_ckpt:
      result[key] = flat_ckpt[key]
    else:
      logging.warning(
          "Skipping key=%s which did not match assignment map or checkpoint.",
          key)

  if require_all_rules_match and len(assignment_map) != len(used_patterns):
    unused_patterns = set(p for p, _ in assignment_map) - used_patterns
    unused_patterns_str = ", ".join(f"'{p}'" for p in unused_patterns)
    raise ValueError("Unused patterns in `assignment_map`: {" +
                     unused_patterns_str + "}")

  return traverse_util.unflatten_dict(result, sep="/")
