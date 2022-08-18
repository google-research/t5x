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

"""T5X Optimizer Support.

Tools for wrapping Optax optimizers and handling SPMD annotations for use with
pjit.

Additional support for the legacy Adafactor implementation.
"""

import functools
from typing import Any, Optional, Union, Sequence, Tuple, Mapping

import flax

# just used for transitional type definitions

from flax import serialization
from flax import struct
from flax import traverse_util
from flax.core import frozen_dict
from flax.serialization import from_state_dict
from flax.serialization import to_state_dict
import jax
import jax.numpy as jnp
import optax

freeze = flax.core.frozen_dict.freeze
unfreeze = flax.core.frozen_dict.unfreeze

Dtype = Any


@struct.dataclass
class OptimizerState:
  step: jnp.ndarray
  param_states: Any


class OptimizerDef:
  """Base class for an optimizer definition."""

  def __init__(self, hyper_params):
    self.hyper_params = hyper_params

  def apply_gradient(self, hyper_params, params, state, grads):
    """Applies a gradient for a set of parameters."""
    raise NotImplementedError()

  def init_state(self, params):
    raise NotImplementedError()

  def update_hyper_params(self, **hyper_param_overrides):
    """Updates the hyper parameters with a set of overrides.

    Args:
      **hyper_param_overrides: the hyper parameters updates will override the
        defaults specified in the `OptimizerDef`. Pass `hyper_params=...` to
        replace all hyper parameters.

    Returns:
      The new hyper parameters.
    """
    hp = hyper_param_overrides.pop('hyper_params', self.hyper_params)
    if hyper_param_overrides:
      hp = hp.replace(**hyper_param_overrides)
    return hp

  def create(self, target):
    """Creates a new optimizer for the given target.

    Args:
      target: the object to be optimized. This is typically a variable dict
        returned by `flax.linen.Module.init()`, but it can also be a container
        of variables dicts, e.g. `(v1, v2)` and  `('var1': v1, 'var2': v2)` are
          valid inputs as well.

    Returns:
      An instance of `Optimizer`.
    """
    opt_def = self
    state = opt_def.init_state(target)
    return Optimizer(opt_def, state, target)

  def state_dict(self, target, state):
    return to_state_dict({
        'target': to_state_dict(target),
        'state': to_state_dict(state)
    })

  def restore_state(self, opt_target, opt_state, state_dict):
    """Restore the optimizer target and state from the state dict.

    Args:
      opt_target: the optimizer target.
      opt_state: the optimizer state.
      state_dict: the state dict containing the desired new state of the
        optimizer.

    Returns:
      a tuple of the optimizer target and state with the restored values from
      the state dict.
    """

    opt_target = from_state_dict(opt_target, state_dict['target'])
    opt_state = from_state_dict(opt_state, state_dict['state'])
    return opt_target, opt_state


class Optimizer(struct.PyTreeNode):
  """Legacy flax optimizer class.

  Optimizer carries the target and optimizer state. The optimizer is updated
  using the method apply_gradient.

  Attributes:
    optimizer_def: The optimizer definition.
    state: The initial state of the optimizer.
    target: The target to optimizer.
  """

  optimizer_def: OptimizerDef = struct.field(pytree_node=False)
  state: Any = struct.field(pytree_node=True)
  target: Any = struct.field(pytree_node=True)

  def apply_gradient(self, grads, **hyper_param_overrides):
    """Applies a pytree of gradients to the target.

    Args:
      grads: A pytree of gradients.
      **hyper_param_overrides: the hyper parameters passed to apply_gradient
        will override the defaults specified in the `OptimizerDef`. Pass
        `hyper_params=...` to replace all hyper parameters.

    Returns:
      A new optimizer with the updated target and state.
    """
    hyper_params = self.optimizer_def.update_hyper_params(
        **hyper_param_overrides)
    new_target, new_state = self.optimizer_def.apply_gradient(
        hyper_params, self.target, self.state, grads)
    return self.replace(target=new_target, state=new_state)

  def state_dict(self):
    return self.optimizer_def.state_dict(self.target, self.state)

  def restore_state(self, state):
    target, state = self.optimizer_def.restore_state(self.target, self.state,
                                                     state)
    return self.replace(target=target, state=state)


# Transitional Type Definitions

OptimizerType = Optimizer
OptimizerStateType = Union[OptimizerState, Mapping[str, Any]]
OptimizerDefType = OptimizerDef


# Optax Elementwise Wrapper


class OptaxStatePartitionRules:
  """Collection of rules to partition optax states.

  These rules work for optimizers whose states are simply replications of
  params, e.g., Adam. Optimizers that aim to save memory by factoring states,
  e.g., Adafactor, SM3, are not supported currently.
  """

  # Rules mapping a particular optax state to a callable returning the state
  # with arrays replaced by t5x PartitionSpec or None.
  #
  # NOTE(levskaya): This is not an entirely exhaustive list, add to this list
  # to support additional optimizers / transformations.
  #
  # pylint: disable=g-long-lambda

  _RULES = {

      # Leaf Optax States:
      optax.AddNoiseState:
          lambda state, params_axes: optax.AddNoiseState(
              count=None, rng_key=None),
      optax.DifferentiallyPrivateAggregateState:
          lambda state, params_axes: optax.DifferentiallyPrivateAggregateState(
              rng_key=None),
      optax.EmaState:
          lambda state, params_axes: optax.EmaState(
              count=None, ema=params_axes),
      optax.EmptyState:
          lambda state, params_axes: optax.EmptyState(),
      optax.TraceState:
          lambda state, params_axes: optax.TraceState(trace=params_axes),
      optax.ScaleByAdamState:
          lambda state, params_axes: optax.ScaleByAdamState(
              count=None, mu=params_axes, nu=params_axes),
      optax.ScaleByBeliefState:
          lambda state, params_axes: optax.ScaleByBeliefState(
              count=None, mu=params_axes, nu=params_axes),
      optax.ScaleByRssState:
          lambda state, params_axes: optax.ScaleByRssState(
              sum_of_squares=params_axes),
      optax.ScaleByRmsState:
          lambda state, params_axes: optax.ScaleByRmsState(nu=params_axes),
      optax.ScaleByRStdDevState:
          lambda state, params_axes: optax.ScaleByRStdDevState(
              mu=params_axes, nu=params_axes),
      optax.ScaleBySM3State:
          lambda state, params_axes: optax.ScaleBySM3State(
              mu=params_axes, nu=params_axes),
      optax.ScaleByTrustRatioState:
          lambda state, params_axes: optax.ScaleByTrustRatioState(),
      optax.ScaleByScheduleState:
          lambda state, params_axes: optax.ScaleByScheduleState(count=None),
      optax.ScaleByFromageState:
          lambda state, params_axes: optax.ScaleByFromageState(count=None),
      optax.ZeroNansState:
          lambda state, params_axes: optax.ZeroNansState(found_nan=None),
      # FactoredState

      # Recursive, Combinator Optax States:

      # MaskedState
      optax.MaskedState:
          lambda state, params_axes: optax.MaskedState(
              inner_state=OptaxStatePartitionRules.derive_optax_logical_axes(
                  state.inner_state, params_axes)),
      optax.InjectHyperparamsState:
          lambda state, params_axes: optax.InjectHyperparamsState(
              count=None,
              hyperparams=jax.tree_map(lambda x: None, state.hyperparams),
              inner_state=OptaxStatePartitionRules.derive_optax_logical_axes(
                  state.inner_state, params_axes)),
      optax.MultiStepsState:
          lambda state, params_axes: optax.MultiStepsState(
              mini_step=None,
              gradient_step=None,
              inner_opt_state=OptaxStatePartitionRules.
              derive_optax_logical_axes(  # pylint: disable=line-too-long
                  state.inner_opt_state, params_axes),
              acc_grads=params_axes),
      optax.ApplyIfFiniteState:
          lambda state, params_axes: optax.ApplyIfFiniteState(
              notfinite_count=None,
              last_finite=None,
              total_notfinite=None,
              inner_state=OptaxStatePartitionRules.derive_optax_logical_axes(
                  state.inner_state, params_axes)),
      optax.MaybeUpdateState:
          lambda state, params_axes: optax.MaybeUpdateState(
              inner_state=OptaxStatePartitionRules.derive_optax_logical_axes(
                  state.inner_state, params_axes),
              step=None),
      optax.MultiTransformState:
          lambda state, params_axes: optax.MultiTransformState(
              inner_states=OptaxStatePartitionRules.derive_optax_logical_axes(
                  state.inner_states, params_axes)),
      # LookaheadState
      # SplitRealAndImaginaryState
  }
  # pylint: enable=g-long-lambda

  @classmethod
  def _is_optax_state(cls, x):
    """Returns true if an object is an optax state.

    Note that in optax states are simply derived from NamedTuple, so we have to
    do some hacky name matching.

    Args:
      x: object.

    Returns:
      True if x is an optax state.
    """
    # A solution from stack overflow. Note that isinstance(x, NamedTuple) would
    # not work.
    is_named_tuple = (
        isinstance(x, tuple) and hasattr(x, '_asdict') and
        hasattr(x, '_fields'))
    result = is_named_tuple and type(x).__name__.endswith('State')
    return result

  @classmethod
  def derive_optax_logical_axes(cls, optax_state, params_axes):
    """Derived logical axes for optax state."""
    # Flatten the optax state but do not go into the registered states.
    flattened_state, tree_def = jax.tree_flatten(
        optax_state, is_leaf=cls._is_optax_state)

    def derive_fn(x):
      if type(x) not in cls._RULES:
        if cls._is_optax_state(x):
          raise ValueError(
              f'Encountered unregistered optax state type {type(x).__name__}')
        return None
      return cls._RULES[type(x)](x, params_axes)

    flattened_axes = [derive_fn(x) for x in flattened_state]
    derived_axes = jax.tree_unflatten(tree_def, flattened_axes)
    return derived_axes


@struct.dataclass
class _OptaxWrapperHyperParams:
  """Dummy hyper params struct, not used."""
  # Required by t5x trainer. Unused as learning rate scheduling is done using
  # optax.Schedule.
  learning_rate: Optional[float] = None


class OptaxWrapper(OptimizerDef):
  """Wrapper to make optax optimizer compatible with T5X."""

  def __init__(self, optax_optimizer: optax.GradientTransformation):
    """Initializer.

    Args:
      optax_optimizer: An optax optimizer.
    """
    self.optax_optimizer = optax_optimizer
    super().__init__(hyper_params=_OptaxWrapperHyperParams())

  def init_state(self, params):
    """Create initial state based on the params to optimize.

    Args:
      params: PyTree of parameters to optimize.

    Returns:
      Initial optimizer state.
    """
    state = OptimizerState(
        step=0, param_states=self.optax_optimizer.init(params))
    return state

  def apply_gradient(self, hyper_params, params, state, grads):
    """Applies gradient.

    Args:
      hyper_params: Unused hyper parameters.
      params: PyTree of the parameters.
      state: A named tuple containing the state of the optimizer.
      grads: PyTree of the gradients for the parameters.

    Returns:
      A tuple containing the new parameters and the new optimizer state.
    """
    del hyper_params

    updates, new_optax_state = self.optax_optimizer.update(
        grads, state.param_states, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, OptimizerState(
        step=state.step + 1, param_states=new_optax_state)

  def derive_logical_axes(self, optimizer, param_logical_axes):
    """Derives optimizer state logical axes from params logical axes.

    Args:
      optimizer: `optimizers.Optimizer` instance.
      param_logical_axes: A PyTree where each leaf is a t5x PartitionSpec.

    Returns:
      An `optimizers.Optimizer` instance, with all the leafs replaced by t5x
      PartitionSpec or None (no partition).
    """
    optimizer_logical_axes = jax.tree_map(lambda x: None,
                                          optimizer.state_dict())
    optimizer_logical_axes['target'] = param_logical_axes

    optax_state_axes = OptaxStatePartitionRules.derive_optax_logical_axes(
        optimizer.state.param_states, param_logical_axes)

    optimizer_logical_axes['state']['param_states'] = (
        serialization.to_state_dict(optax_state_axes))

    return optimizer.restore_state(frozen_dict.unfreeze(optimizer_logical_axes))

  def state_dict(self, target, state):
    """Override state dict function.

    We need to override this function because many optax transformations use
    `optax.EmptyState`, which produces empty dict in the state dict. This causes
    the T5 training loop to fail in multiple places. As a remedy, we will
    filter out the generated state dict so that there are no empty dict in the
    output.

    The restore_state function is also overridden to reconstruct those empty
    dict.

    Args:
      target: Pytree of target variables.
      state: Pytree of optimizer state.

    Returns:
      A nested state.
    """
    state_dict = to_state_dict(state)

    # This step removes any empty dict (recursively) in the state dict.
    state_dict = traverse_util.unflatten_dict(
        traverse_util.flatten_dict(state_dict, sep='/'), sep='/')

    return to_state_dict({
        'target': to_state_dict(target),
        'state': state_dict,
    })

  def restore_state(self, opt_target, opt_state, state_dict):
    """Override to restore empty dicts corresponding to `optax.EmptyState`.

    Args:
      opt_target: the optimizer target.
      opt_state: the optimizer state.
      state_dict: the state dict containing the desired new state of the
        optimizer.

    Returns:
      a tuple of the optimizer target and state with the restored values from
      the state dict.
    """
    opt_target = from_state_dict(opt_target, state_dict['target'])

    # Get all the possible keys in the reference optimizer state.
    flat_ref_opt_state_dict = traverse_util.flatten_dict(
        to_state_dict(opt_state), keep_empty_nodes=True, sep='/')

    flat_src_opt_state_dict = dict(
        traverse_util.flatten_dict(state_dict['state'], sep='/'))
    # Adding the empty paths back to flat_src_opt_state_dict.
    for k, v in flat_ref_opt_state_dict.items():
      if k in flat_src_opt_state_dict:
        continue
      # The key is not in the input state dict, presumably because it
      # corresponds to an empty dict.
      if v != traverse_util.empty_node:
        raise ValueError(
            f'Failed to restore optimizer state, path {k} is not present '
            'in the input optimizer state dict.')
      flat_src_opt_state_dict[k] = v

    # Restore state from the enhanced state dict.
    opt_state = from_state_dict(
        opt_state,
        traverse_util.unflatten_dict(flat_src_opt_state_dict, sep='/'))
    return opt_target, opt_state


# Optax wrapper and elementary wrapped optax optimizers.


def wrap_optax_optimizer(optax_optimizer):
  """Converts optax optimizer constructor to a wrapped T5X-compatible optimizer.

  Args:
    optax_optimizer: an optax optimizer creation function that returns an optax
      GradientTransformation.

  Returns:
    A function that takes the same arguments as the original optax creation
    function but instead returns a wrapped OptimizerDef-compatible interface for
    using the optimizer with T5X.
  """

  @functools.wraps(optax_optimizer)
  def wrapped_optimizer(*args, **kwargs) -> OptimizerDef:
    return OptaxWrapper(optax_optimizer(*args, **kwargs))

  return wrapped_optimizer


def chain(
    transformations: Sequence[optax.GradientTransformation]
) -> optax.GradientTransformation:
  return optax.chain(*transformations)


chain = wrap_optax_optimizer(chain)
adabelief = wrap_optax_optimizer(optax.adabelief)
adagrad = wrap_optax_optimizer(optax.adagrad)
adam = wrap_optax_optimizer(optax.adam)
adamw = wrap_optax_optimizer(optax.adamw)
fromage = wrap_optax_optimizer(optax.fromage)
lars = wrap_optax_optimizer(optax.lars)
lamb = wrap_optax_optimizer(optax.lamb)
noisy_sgd = wrap_optax_optimizer(optax.noisy_sgd)
radam = wrap_optax_optimizer(optax.radam)
rmsprop = wrap_optax_optimizer(optax.rmsprop)
sgd = wrap_optax_optimizer(optax.sgd)
yogi = wrap_optax_optimizer(optax.yogi)
dpsgd = wrap_optax_optimizer(optax.dpsgd)

# Excluded optimizers:
# TODO(levskaya): add shampoo, sm3
# We use our own generalized adafactor implementations.
# adafactor = wrap_optax_optimizer(optax.adafactor)
# We may use a more complete quantized implementation of SM3
# sm3 = wrap_optax_optimizer(optax.sm3)

# Inlined Legacy Generalized Multioptimizer


class _Marker:
  """Used to mark unoptimized leaves."""

  def __init__(self):
    self._indices = []


def _tree_of_paths(tree):
  """Converts a (frozen) nested dictionary into a (frozen) dict of paths."""
  is_frozen = isinstance(tree, flax.core.frozen_dict.FrozenDict)
  flat_tree = traverse_util.flatten_dict(unfreeze(tree))
  path_tree = traverse_util.unflatten_dict(
      {k: '/'.join(k) for k in flat_tree.keys()})
  if is_frozen:
    path_tree = freeze(path_tree)
  return path_tree


def _subtree_from_traversal(traversal, tree):
  """Creates a (frozen) tree subset given a traversal."""
  is_frozen = isinstance(tree, flax.core.frozen_dict.FrozenDict)
  flat_tree = {}
  for path, leaf in zip(
      traversal.iterate(_tree_of_paths(tree)), traversal.iterate(tree)):
    flat_tree[path] = leaf
  new_tree = traverse_util.unflatten_dict(
      {tuple(k.split('/')): v for k, v in flat_tree.items()})
  if is_frozen:
    new_tree = freeze(new_tree)
  return new_tree


def _update_subtree_of_traversal(traversal, tree, update):
  """Updates a (frozen) tree's subset given a traversal and update subtree."""
  is_frozen = isinstance(tree, flax.core.frozen_dict.FrozenDict)
  flat_tree = traverse_util.flatten_dict(unfreeze(tree))
  flat_tree = {'/'.join(k): v for k, v in flat_tree.items()}
  for path, leaf in zip(
      traversal.iterate(_tree_of_paths(update)), traversal.iterate(update)):
    flat_tree[path] = leaf
  nested_d = traverse_util.unflatten_dict(
      {tuple(k.split('/')): v for k, v in flat_tree.items()})
  if is_frozen:
    nested_d = freeze(nested_d)
  return nested_d


class MultiOptimizer(OptimizerDef):
  """Generalized Multioptimizer.

  NB: Although this is provided for legacy support, it is still quite general
  and should work fine with wrapped optax optimizers.  But do note that the more
  canonical way of mixing multiple optimizers inside optax uses optax.masked or
  optax.multi_transform instead.

  A MultiOptimizer is subclass of :class:`OptimizerDef` and useful for applying
  separate optimizer algorithms to various subsets of the model parameters.

  The example below creates two optimizers using
  :class:`flax.traverse_util.ModelParamTraversal`:
  one to optimize ``kernel`` parameters and to optimize ``bias`` parameters.
  Note each optimizer is created with a different learning rate::

    kernels = traverse_util.ModelParamTraversal(
        lambda path, _: 'kernel' in path)
    biases = traverse_util.ModelParamTraversal(lambda path, _: 'bias' in path)
    kernel_opt = optimizers.adam(learning_rate=0.01)
    bias_opt = optimizers.adam(learning_rate=0.1)
    opt_def = MultiOptimizer((kernels, kernel_opt), (biases, bias_opt))
    optimizer = opt_def.create(model)

  In order to train only a subset of the parameters, you can simply use a single
  :class:`flax.traverse_util.ModelParamTraversal` instance.

  If you want to update the learning rates of both optimizers online with
  different learning rate schedules, you should update the learning rates when
  applying the gradient. In the following example, the second optimizer is not
  doing any optimization during the first 1000 steps::

    hparams = optimizer.optimizer_def.hyper_params
    new_optimizer = optimizer.apply_gradient(
        grads,
        hyper_params=[
          hparams[0].replace(learning_rate=0.2),
          hparams[1].replace(learning_rate=jnp.where(step < 1000, 0., lr)),
        ])
  """

  def __init__(
      self, traversals_and_optimizers: Sequence[Tuple[traverse_util.Traversal,
                                                      OptimizerDef]]):
    """Create a new MultiOptimizer.

    See docstring of :class:`MultiOptimizer` for more details.

    Args:
      traversals_and_optimizers: pairs of flax.traverse_util.Traversal and
        `optimizers.OptimizerDef` instances.
    """
    traversals, sub_optimizers = zip(*traversals_and_optimizers)
    hyper_params = [opt.hyper_params for opt in sub_optimizers]
    super().__init__(hyper_params)
    self.traversals = traversals
    self.sub_optimizers = sub_optimizers

  def init_state(self, params):
    param_states = jax.tree_map(lambda x: _Marker(), params)
    overlap = False
    for idx, traversal in enumerate(self.traversals):
      for match in traversal.iterate(param_states):
        match._indices.append(idx)  # pylint: disable=protected-access
        overlap |= len(match._indices) > 1  # pylint: disable=protected-access
    if overlap:
      raise ValueError(
          'Multiple optimizers match the same leaves : ' +
          str(jax.tree_map(lambda match: match._indices, param_states)))  # pylint: disable=protected-access

    param_states = jax.tree_map(lambda x: _Marker(), params)
    for focus, opt_def in zip(self.traversals, self.sub_optimizers):
      ps = _subtree_from_traversal(focus, params)
      ss = opt_def.init_state(ps)
      param_states = _update_subtree_of_traversal(focus, param_states,
                                                  ss.param_states)
    # Update state to None when param is not optimized by any sub optimizer.
    param_states = jax.tree_map(
        lambda x: (None if isinstance(x, _Marker) else x), param_states)
    return OptimizerState(jnp.asarray(0, dtype=jnp.int32), param_states)

  def apply_gradient(self, hyper_params, params, state, grads):
    new_params = params
    it = zip(self.traversals, self.sub_optimizers, hyper_params)
    new_param_states = jax.tree_map(lambda x: _Marker(), params)
    for focus, opt_def, hp in it:
      ps = _subtree_from_traversal(focus, params)
      gs = _subtree_from_traversal(focus, grads)
      ss = _subtree_from_traversal(focus, state.param_states)
      prev_ss = OptimizerState(state.step, ss)
      new_ps, new_ss = opt_def.apply_gradient(hp, ps, prev_ss, gs)
      new_params = _update_subtree_of_traversal(focus, new_params, new_ps)
      new_param_states = _update_subtree_of_traversal(focus, new_param_states,
                                                      new_ss.param_states)
    # Update state to None when param is not optimized by any sub optimizer.
    new_param_states = jax.tree_map(
        lambda x: (None if isinstance(x, _Marker) else x), new_param_states)
    return new_params, OptimizerState(state.step + 1, new_param_states)

  def update_hyper_params(self, **hyper_param_overrides):
    """Updates the hyper parameters with a set of overrides.

    This method is called from :meth:`Optimizer.apply_gradient` to create the
    hyper parameters for a specific optimization step.
    MultiOptimizer will apply the overrides for each sub optimizer.

    Args:
      **hyper_param_overrides: the hyper parameters updates will override the
        defaults specified in the `OptimizerDef`. Pass `hyper_params=...` to
        replace all hyper parameters.

    Returns:
      The new hyper parameters.
    """
    hps = hyper_param_overrides.pop('hyper_params', self.hyper_params)
    if hyper_param_overrides:
      hps = [hp.replace(**hyper_param_overrides) for hp in hps]
    return hps

  def set_param_axes(self, param_logical_axes):
    """Derives factorization rules from model parameter logical axes."""
    for focus, opt_def in zip(self.traversals, self.sub_optimizers):
      pla_subtree = _subtree_from_traversal(focus, param_logical_axes)
      if hasattr(opt_def, 'set_param_axes'):
        opt_def.set_param_axes(pla_subtree)

  def derive_logical_axes(self, optimizer, param_logical_axes):
    """Derives optimizer logical partitioning from model logical partitions."""
    param_states = jax.tree_map(lambda x: _Marker(),
                                optimizer.state.param_states)
    for focus, opt_def in zip(self.traversals, self.sub_optimizers):
      if hasattr(opt_def, 'derive_logical_axes'):
        ps = _subtree_from_traversal(focus, param_logical_axes)
        ss = _subtree_from_traversal(focus, optimizer.state.param_states)
        new_opt = opt_def.derive_logical_axes(
            Optimizer(opt_def, OptimizerState(None, ss), ps), ps)
        param_states = _update_subtree_of_traversal(focus, param_states,
                                                    new_opt.state.param_states)
    # Update axes to None when param is not optimized by any sub optimizer.
    param_states = jax.tree_map(
        lambda x: (None if isinstance(x, _Marker) else x), param_states)
    return Optimizer(optimizer.optimizer_def,
                     OptimizerState(None, param_states), param_logical_axes)

  # TODO(levskaya): add traversal handling for state_dict / restore_state
  # this is required to make this work w. optax optimizers...
