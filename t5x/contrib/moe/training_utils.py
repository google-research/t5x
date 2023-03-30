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

"""Extensions to Jax/Flax core functions for Mixture of Experts training.

"""

import dataclasses
import re
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import flax
import jax
import numpy as np
from t5x import train_state

# Type Stubs
ParamTree = Any
Gradients = Union[flax.core.FrozenDict, train_state.TrainState]


def match_fn(prefix: Optional[str]) -> Callable[[str], bool]:
  """Creates a function returning true iff a string matches the prefix.

  Args:
    prefix: Regex prefix to match. If none, then return match function will not
      match any strings.

  Returns:
    Prefix match function.
  """
  if not prefix:
    return lambda name: False
  params_regex = re.compile(f'^{prefix}')
  return lambda name: params_regex.match(name) is not None


def scale_sharded_grads(grads: Gradients,
                        sharded_match_fn: Optional[Callable[[str], bool]],
                        scale_factor: float) -> Gradients:
  """Scales sharded grads, identified by sharded_match_fn, by scale_factor.

  Args:
    grads: Parameter gradients.
    sharded_match_fn: Filter function for distinguishing sharded parameters from
      replicated parameters.
    scale_factor: Amount by which to scale sharded parameter gradients.

  Returns:
    Gradients matching input, expect with sharded parameter gradients rescaled.
  """
  if sharded_match_fn:
    names_and_grads, tree_def = _tree_flatten_with_names(grads)
    scaled_grads = [
        grad * scale_factor if sharded_match_fn(name) else grad
        for name, grad in names_and_grads
    ]
    return tree_def.unflatten(scaled_grads)
  else:
    return grads


def tree_map_with_names(f, param_tree, match_name_fn=lambda name: True):
  """Like jax.tree_map but with a filter on the leaf path name.

  Args:
    f: The function to be applied to each parameter in `param_tree`.
    param_tree: The tree of parameters `f` should be applied to.
    match_name_fn: This function is called with each tree leave's path name,
      which has a path-like format ('a/b/c'), and decides whether `f` should be
      applied to that leaf or the leaf should be kept as-is.

  Returns:
    A tree identical in structure to `param_tree` but with the leaves the
    result of calling `f` on them in the cases where `match_name_fn` returns
    True for that leaf's path name.
  """
  names_and_vals, tree_def = _tree_flatten_with_names(param_tree)
  vals = [f(v) if match_name_fn(name) else v for name, v in names_and_vals]
  return tree_def.unflatten(vals)


def _tree_flatten_with_names(
    tree: ParamTree,
) -> Tuple[Sequence[Tuple[str, Any]], jax.tree_util.PyTreeDef]:
  """Like jax.tree_util.tree_flatten but also fetches leaf names.

  Specialized to parameter trees of the form {'key0': {'subkey0': Any}, ...}.

  Args:
    tree: Tree of parameters to flatten.

  Returns:
    - A list of leaf name and value pairs: [(name, value), ...].
    - A tree definition object representing the structure of the flattened tree.
  """
  # PyTrees don't treat None values as leaves, so we explicitly declare them as
  # such.
  vals, tree_def = jax.tree_util.tree_flatten(tree, is_leaf=lambda x: x is None)

  # 'Fake' token tree that is use to track jax internal tree traversal and
  # adjust our custom tree traversal to be compatible with it.
  tokens = range(len(vals))
  token_tree = tree_def.unflatten(tokens)
  val_names, perm = zip(*_traverse_with_names(token_tree))
  inv_perm = np.argsort(perm)

  # Custom traversal should visit the same number of leaves.
  if len(val_names) != len(vals):
    raise ValueError(f'Pytree traversal detected {len(val_names)} names, '
                     f'but {len(vals)} leafs.\nTreeDef is:\n{tree_def}')

  return [(val_names[i], v) for i, v in zip(inv_perm, vals)], tree_def


def _traverse_with_names(
    param_tree: ParamTree) -> Iterable[Tuple[str, ParamTree]]:
  """Traverses nested dicts/dataclasses and emits (leaf_name, leaf_val)."""
  if dataclasses.is_dataclass(param_tree):
    param_tree = flax.serialization.to_state_dict(param_tree)
  if isinstance(param_tree, (dict, flax.core.FrozenDict)):
    keys = sorted(param_tree.keys())
    for key in keys:
      for path, v in _traverse_with_names(param_tree[key]):
        yield (key + '/' + path).rstrip('/'), v
  else:
    yield '', param_tree
