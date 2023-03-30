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

"""Adafactor Optimizer.

Specialized Adafactor implementation for T5X with:
 - custom factorization specification rules.
 - support for stacked parameters from scanned layers and parameter fusions.

Why do we need custom factorization?  In the Adafactor paper, scalar, vector and
matrix parameters are considered. This is sufficiently general because higher
dimensional parameters can be reshaped. In practice, there are situations where
higher dimensional parameters are desirable.  For example, consider the
multi-headed attention. It has projection kernels.  This is naturally
represented as 3-dimensional array [d_model, num_head, head_dim]. Keeping the
3-dimensional structure can be beneficial for performance optimization, e.g., by
giving compilers additional degree of freedom to do layout optimization.

The default heuristic behavior for the second-moment estimator can lead to an
unexpected result because it assumes that the parameters are matrices (vectors
and scalars are not factored). The dimensions are sorted and the smaller
dimension is assigned to the row dim and the larger dim to the col dim (unless
the two largest dims have an equal size and then the original ordering of the
dimensions is used). Then `v_row` (i.e., the optimizer state for the row) is
obtained by removing the col dim. In other words, `rank(v_row) = rank(v) - 1`.
If the parameter is higher dimensional, v_row and v_col are higher dimensional.
Therefore, the outer product of v_row and v_col do not necessarily corresponds
to the row rank approximation that minimizes the generalized Kullback-Leibler
divergence (the original Adafactor formulation).

This Adafactor implementation generalized the default behavior such that we
obtain the correct second moment estimator even for higher dimensional
parameters.

"""
import enum
import re
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
from flax import struct
from flax.core import freeze
from flax.core import FrozenDict
from flax.core import unfreeze
from flax.serialization import from_state_dict
from flax.serialization import to_state_dict
from flax.traverse_util import flatten_dict
from flax.traverse_util import unflatten_dict
import jax
import jax.numpy as jnp
import numpy as np
from t5x import utils
from t5x.optimizers import OptimizerDef
from t5x.optimizers import OptimizerState

Dtype = Any


class FactorDim(enum.Enum):
  # Don't factorize this dimension.
  NONE = None
  # A batch-like dimension that we should not average over.
  BATCH = 1
  ROW = 2
  COLUMN = 3


# Sentinel value signifying the legacy heuristic factorization rule.
class HeuristicRule(enum.Enum):
  token = 1


HEURISTIC_RULE = HeuristicRule.token
FactorRule = Union[HeuristicRule, Tuple[FactorDim]]


def _restore(target, flat):
  state_dict = unflatten_dict({tuple(k.split('/')): v for k, v in flat.items()})
  if isinstance(target, FrozenDict):
    return freeze(state_dict)
  else:
    return state_dict


def _insert(tpl, idx, x):
  tmp = list(tpl)
  tmp.insert(idx, x)
  return tuple(tmp)


def standard_logical_factor_rules():
  return freeze({
      'vocab': FactorDim.COLUMN,
      'embed': FactorDim.ROW,
      'mlp': FactorDim.COLUMN,
      'heads': FactorDim.COLUMN,
      'kv': FactorDim.COLUMN,
      'joined_kv': FactorDim.COLUMN,
      'relpos_buckets': FactorDim.NONE,
      'layers': FactorDim.BATCH,  # used in scanned layers
      'stack': FactorDim.BATCH,  # used in stacked params
      # 'batch', 'length' should not occur in parameters
      'q_wi_fused': FactorDim.COLUMN,
      'o_wo_fused': FactorDim.COLUMN,
      'multiquery_heads': FactorDim.COLUMN,
      'kv_fused': FactorDim.COLUMN,
      'layer_norm_scale': FactorDim.NONE,
      'mlp_activations': FactorDim.COLUMN,
  })


def factor_name_to_factordim(name):
  if not isinstance(name, str):
    return name
  name = name.lower()
  return {
      'row': FactorDim.ROW,
      'col': FactorDim.COLUMN,
      'column': FactorDim.COLUMN,
      'batch': FactorDim.BATCH,
      'none': FactorDim.NONE,
      'unfactorized': FactorDim.NONE
  }[name]


class HParamMap:
  """Maps parameter path names to hparams.

  Names of parameters nested in a PyTree (e.g., an Optimizer) are formed by
  joining the names along the path to the parameter leaf with '/'.
  """

  def __init__(self, rules):
    self._rules = [(re.compile(r), p) for r, p in rules]

  def __getitem__(self, key: str) -> Any:
    for r, p in self._rules:
      if r.search(key):
        return p
    raise KeyError(f'No factor rule found for parameter: {key}')

  def __call__(self, params):
    """Returns a copy of the params with mapped hparams in leaves."""
    flat_state_dict = flatten_dict(to_state_dict(params))
    flat_rules_dict = {k: self['/'.join(k)] for k in flat_state_dict.keys()}
    return from_state_dict(params, unflatten_dict(flat_rules_dict))


@struct.dataclass
class _AdafactorHyperParams:
  """Hparams for Adafactor optimizer."""
  learning_rate: Optional[float]
  factored: bool
  multiply_by_parameter_scale: Union[bool, HParamMap]
  beta1: Optional[float]
  decay_rate: float
  step_offset: int
  clipping_threshold: Optional[float]
  weight_decay_rate: Optional[float]
  min_dim_size_to_factor: int
  epsilon1: float
  epsilon2: float
  factor_map: Optional[HParamMap] = None
  logical_factor_rules: Any = None
  weight_decay_rate_lr_exponent: Optional[float] = None
  global_norm_clip_threshold: Optional[float] = None
  max_parameter_scale: Optional[float] = None
  skip_nan_updates: Optional[bool] = False


@struct.dataclass
class _AdafactorParamState:
  v_row: np.ndarray  # used in normal factored version
  v_col: np.ndarray
  v: np.ndarray  # only used without factoring
  m: np.ndarray  # only used with momentum


class Adafactor(OptimizerDef):
  """Adafactor optimizer.

  Adafactor is described in https://arxiv.org/abs/1804.04235.
  """

  def __init__(self,
               learning_rate: Optional[float] = None,
               factored: bool = True,
               multiply_by_parameter_scale: Union[bool, HParamMap] = True,
               beta1: Optional[float] = None,
               decay_rate: float = 0.8,
               step_offset: int = 0,
               clipping_threshold: Optional[float] = 1.0,
               weight_decay_rate: Optional[float] = None,
               min_dim_size_to_factor: int = 128,
               epsilon1: float = 1e-30,
               epsilon2: float = 1e-3,
               dtype_momentum: Dtype = jnp.float32,
               factor_map: Optional[HParamMap] = None,
               logical_factor_rules: Optional[Mapping[str, FactorDim]] = None,
               weight_decay_rate_lr_exponent: Optional[float] = None,
               global_norm_clip_threshold: Optional[float] = None,
               max_parameter_scale: Optional[float] = None,
               skip_nan_updates: Optional[bool] = False):
    """Constructor for the Adafactor optimizer.


    Args:
      learning_rate: float: learning rate.  NB: the natural scale for adafactor
        LR is markedly different from Adam, one doesn't use the 1/sqrt(hidden)
        correction for this optimizer with attention-based models.
      factored: boolean: whether to use factored second-moment estimator for 2d
        variables.
      multiply_by_parameter_scale: boolean: if True, then scale provided
        learning_rate by parameter norm. if False, provided learning_rate is
        absolute step size.
      beta1: an optional float value between 0 and 1, enables momentum and uses
        extra memory if non-None! None by default.
      decay_rate: float: controls second-moment exponential decay schedule.
      step_offset: for finetuning, one may optionally set this to the starting
        step-number of the finetuning phase to reset the second moment
        accumulators after pretraining. Does not affect the momentum even if it
        was used during pretraining.
      clipping_threshold: an optional float >= 1, if None no update clipping.
      weight_decay_rate: optional rate at which to decay weights.
      min_dim_size_to_factor: only factor accumulator if two array dimensions
        are at least this size.
      epsilon1: Regularization constant for squared gradient.
      epsilon2: Regularization constant for parameter scale.
      dtype_momentum: dtype of momentum buffers.
      factor_map: hparam-map from key path to manual factorization rules.
      logical_factor_rules: factorization rules provided as a set of mappings
        from logical axis name to ROW, COLUMN, BATCH, or NONE.  Supersedes
        factor_map if `set_param_axes` is called.
      weight_decay_rate_lr_exponent: If present, weight decay rate is computed
        as (learning_rate ** weight_decay_rate_lr_exponent).  If
        weight_decay_rate is also present, then multiply by it.
      global_norm_clip_threshold: If set, will clip gradients by global norm
        before Adafactor stats are applied.
      max_parameter_scale: If set, clips the parameter scale to a maximum value,
        which helps prevent parameters from growing without bound.
      skip_nan_updates: If set, any parameter that would have been updated by a
        NaN value after a applying gradients will be kept with the earlier value
        it had.
    """
    if not factored and factor_map is not None:
      raise ValueError('Adafactor factored is False but factorization rules '
                       'have been provided.')
    if not isinstance(multiply_by_parameter_scale, (bool, HParamMap)):
      raise TypeError(
          '`multiply_by_parameter_scale` must be either bool or `HParamMap` '
          f'type. Got {type(multiply_by_parameter_scale)}')

    if not isinstance(factor_map, (type(None), HParamMap)):
      raise TypeError(
          '`factor_map` must be either None or `HParamMap` type. Got '
          f'{type(factor_map)}')

    hyper_params = _AdafactorHyperParams(
        learning_rate, factored, multiply_by_parameter_scale, beta1, decay_rate,
        step_offset, clipping_threshold, weight_decay_rate,
        min_dim_size_to_factor, epsilon1, epsilon2, factor_map,
        logical_factor_rules, weight_decay_rate_lr_exponent,
        global_norm_clip_threshold, max_parameter_scale, skip_nan_updates)
    self.dtype_momentum = jax.dtypes.canonicalize_dtype(dtype_momentum)
    super().__init__(hyper_params)

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, Adafactor):
      return False
    return (
        self.hyper_params == other.hyper_params
        and self.dtype_momentum == other.dtype_momentum
    )

  def __hash__(self) -> int:
    return id(self)

  @staticmethod
  def _decay_rate_pow(i: int, exponent: float = 0.8) -> float:
    """Default Adafactor second-moment decay schedule."""
    t = jnp.array(i, jnp.float32) + 1.0
    return 1.0 - t**(-exponent)

  @staticmethod
  def _parse_rule(
      rule: Optional[FactorRule],
      shape: Sequence[int],
      path: str,
      fallback_to_heuristics=True
  ) -> Tuple[Tuple[int, ...], Optional[Union[HeuristicRule, Tuple[Tuple[
      int, ...], Tuple[int, ...]]]]]:
    """Parses specification and return factored dims and dims for averaging.

    Adafactor needs to know the two largest dimensions to factorize along.
    Traditionally it used a heuristic, but we want finer control over these
    factorization dimensions.  Additionally, there are situations where
    parameters are batched together for e.g. scanned layers and QKV fusion,
    and we want to ensure that the scale updates and clipping thresholds are
    calculated _within_ each array and not across the entire batched array.

    Args:
      rule: the rule is either None (default to heuristic behavior) or a tuple
        of the same rank as the `param` array containing a FactorDim.ROW or
        FactorDim.COLUMN to mark dimensions to factorize in two row and column
        sets, and optionally dimensions marked FactorDim.BATCH to denote batched
        dimensions that should not be averaged over. e.g. (BATCH, ROW, COLUMN,
        COLUMN)
      shape: shape of the variable
      path: '/' joined parameter path.
      fallback_to_heuristics: whether to fallback to heuristic factorization
        rule. For most cases this should be set to `True`.

    Returns:
      tuple of: tuple of dimensions to average over, 2-tuple of dimensions to
      factorize over.
    """
    param_ndim = len(shape)

    if rule is None:
      # No factorization.
      return tuple(np.arange(param_ndim)), None

    if rule is HEURISTIC_RULE:
      if param_ndim > 2:
        raise ValueError(
            f'A parameter with rank strictly higher than 2 must have an '
            f'explicit factorization rule: {path}, {shape}')
      # Even if no explicit rule is provided for the param, we still want to
      # average over all the dimensions for computing the RMS scale.
      return tuple(np.arange(param_ndim)), HEURISTIC_RULE

    if len(rule) != param_ndim:
      raise ValueError(f'Factorization rule {rule} has incorrect rank '
                       f'for param of rank {param_ndim}: {path}, {shape}')

    row_dims = tuple(idx for idx, d in enumerate(rule) if d == FactorDim.ROW)
    col_dims = tuple(idx for idx, d in enumerate(rule) if d == FactorDim.COLUMN)
    batched_dims = tuple(
        idx for idx, d in enumerate(rule) if d == FactorDim.BATCH)
    averaging_dims = tuple(np.delete(np.arange(param_ndim), batched_dims))
    factor_dims = (row_dims, col_dims)
    if factor_dims == ((), ()):
      factor_dims = None

    if fallback_to_heuristics and param_ndim <= 2 and not batched_dims:
      logging.warning(
          'Since rank of parameter %s %d is less than or equal to 2, the '
          'factorization method falls back to heuristics and the provided '
          'factor rule %s is ignored.', path, param_ndim, rule)
      return tuple(np.arange(param_ndim)), HEURISTIC_RULE

    return averaging_dims, factor_dims

  def _factored_dims(
      self, shape: Sequence[int]) -> Optional[Tuple[Tuple[int], Tuple[int]]]:
    """Whether to use a factored second moment estimator.

    If there are not two dimensions of size >= min_dim_size_to_factor, then we
    do not factor. If we do factor the accumulator, then this function returns a
    tuple of the two largest axes to reduce over.

    Args:
      shape: a Shape

    Returns:
      None or a tuple of ints
    """
    if not self.hyper_params.factored or len(shape) < 2:
      return None
    sorted_dims = np.argsort(shape)
    if shape[sorted_dims[-2]] < self.hyper_params.min_dim_size_to_factor:
      return None
    return (int(sorted_dims[-2]),), (int(sorted_dims[-1]),)

  def init_param_state(self, param, path):
    shape = param.shape
    state = {k: jnp.zeros((1,)) for k in ['v_row', 'v_col', 'v', 'm']}
    if self.hyper_params.factored:
      factor_rule = (
          self.hyper_params.factor_map[path]
          if self.hyper_params.factor_map else HEURISTIC_RULE)
    else:
      factor_rule = None
    _, factored_dims = self._parse_rule(factor_rule, param.shape, path)
    if factored_dims is HEURISTIC_RULE:
      factored_dims = self._factored_dims(shape)
    if factored_dims is not None:
      d1, d0 = factored_dims
      vr_shape = np.delete(shape, d0)
      vc_shape = np.delete(shape, d1)
      state['v_row'] = jnp.zeros(vr_shape, dtype=jnp.float32)
      state['v_col'] = jnp.zeros(vc_shape, dtype=jnp.float32)
    else:
      state['v'] = jnp.zeros(param.shape, dtype=jnp.float32)
    if self.hyper_params.beta1 is not None:
      state['m'] = jnp.zeros(param.shape, dtype=self.dtype_momentum)
    return _AdafactorParamState(**state)

  def init_state(self, params):
    params_flat = utils.flatten_dict_string_keys(params)
    param_states_flat = [
        self.init_param_state(param, path)
        for path, param in params_flat.items()
    ]
    param_states_flat = {
        k: v for k, v in zip(params_flat.keys(), param_states_flat)
    }
    param_states = _restore(params, param_states_flat)
    state = OptimizerState(jnp.asarray(0, dtype=jnp.int32), param_states)
    return state

  def apply_param_gradient(self, step, hyper_params, param, state, grad, path):
    assert hyper_params.learning_rate is not None, 'no learning rate provided.'
    learning_rate = hyper_params.learning_rate
    beta1 = hyper_params.beta1
    decay_rate = hyper_params.decay_rate
    step_offset = hyper_params.step_offset
    multiply_by_parameter_scale = hyper_params.multiply_by_parameter_scale
    max_parameter_scale = hyper_params.max_parameter_scale
    clipping_threshold = hyper_params.clipping_threshold
    weight_decay_rate = hyper_params.weight_decay_rate
    epsilon1 = hyper_params.epsilon1
    epsilon2 = hyper_params.epsilon2
    if hyper_params.weight_decay_rate_lr_exponent:
      weight_decay_rate = (
          (weight_decay_rate or 1.0) *
          learning_rate**hyper_params.weight_decay_rate_lr_exponent)

    if self.hyper_params.factored:
      factor_rule = (
          self.hyper_params.factor_map[path]
          if self.hyper_params.factor_map else HEURISTIC_RULE)
    else:
      factor_rule = None
    averaging_dims, factored_dims = self._parse_rule(factor_rule, param.shape,
                                                     path)

    grad = grad.astype(jnp.float32)

    updates = {k: jnp.zeros((1,)) for k in ['v_row', 'v_col', 'v', 'm']}
    decay_rate = self._decay_rate_pow(step - step_offset, exponent=decay_rate)
    update_scale = learning_rate

    if isinstance(multiply_by_parameter_scale, HParamMap):
      multiply_by_parameter_scale = multiply_by_parameter_scale[path]
    if multiply_by_parameter_scale:
      param_scale = jnp.sqrt(
          jnp.mean(param * param, axis=averaging_dims, keepdims=True))
      # Clip param_scale to a minimum value of epsilon2.
      param_scale = jnp.maximum(param_scale, epsilon2)
      # Clip param_scale to a maximum value, if specified.
      if max_parameter_scale is not None:
        param_scale = jnp.minimum(param_scale, max_parameter_scale)
      update_scale *= param_scale
    mixing_rate = 1.0 - decay_rate

    grad_sqr = grad * grad + epsilon1
    if factored_dims is HEURISTIC_RULE:
      factored_dims = self._factored_dims(param.shape)
    if factored_dims is not None:
      d1, d0 = factored_dims
      new_v_row = (
          decay_rate * state.v_row + mixing_rate * jnp.mean(grad_sqr, axis=d0))
      new_v_col = (
          decay_rate * state.v_col + mixing_rate * jnp.mean(grad_sqr, axis=d1))
      updates['v_row'] = new_v_row
      updates['v_col'] = new_v_col
      reduced_d1 = tuple(d - len([e for e in d0 if e < d]) for d in d1)

      row_col_mean = jnp.mean(new_v_row, axis=reduced_d1, keepdims=True)
      row_factor = (new_v_row / row_col_mean)**-0.5
      col_factor = (new_v_col)**-0.5
      y = (
          grad * jnp.expand_dims(row_factor, axis=d0) *
          jnp.expand_dims(col_factor, axis=d1))
    else:
      new_v = decay_rate * state.v + mixing_rate * grad_sqr
      updates['v'] = new_v
      y = grad * (new_v)**-0.5

    if clipping_threshold is not None:
      clipping_denom = (
          jnp.maximum(
              1.0,
              jnp.sqrt(jnp.mean(y * y, axis=averaging_dims, keepdims=True)) /
              clipping_threshold))
      y /= clipping_denom

    subtrahend = update_scale * y
    if beta1 is not None:
      new_m = beta1 * state.m + (1.0 - beta1) * subtrahend
      subtrahend = new_m
      updates['m'] = new_m.astype(self.dtype_momentum)

    if weight_decay_rate is not None:
      new_param = (1.0 - weight_decay_rate) * param - subtrahend
    else:
      new_param = param - subtrahend

    if hyper_params.skip_nan_updates:
      updates['v_row'] = jnp.where(
          jnp.isnan(updates['v_row']), state.v_row, updates['v_row'])
      updates['v_col'] = jnp.where(
          jnp.isnan(updates['v_col']), state.v_col, updates['v_col'])
      updates['v'] = jnp.where(jnp.isnan(updates['v']), state.v, updates['v'])
      updates['m'] = jnp.where(jnp.isnan(updates['m']), state.m, updates['m'])
      new_param = jnp.where(jnp.isnan(new_param), param, new_param)
    new_state = _AdafactorParamState(**updates)

    return new_param.astype(param.dtype), new_state

  def apply_gradient(self, hyper_params, params, state, grads):
    """Applies a gradient for a set of parameters.

    Args:
      hyper_params: a named tuple of hyper parameters.
      params: the parameters that should be updated.
      state: a named tuple containing the state of the optimizer
      grads: the gradient tensors for the parameters.

    Returns:
      A tuple containing the new parameters and the new optimizer state.
    """
    step = state.step
    # We assume that params, param_states, and grads are all dict-like here.
    params_flat_dict = utils.flatten_dict_string_keys(params)
    params_paths = params_flat_dict.keys()
    params_flat = params_flat_dict.values()
    # extra paranoia to guarantee identical value ordering
    states_flat = utils.flatten_dict_string_keys(state.param_states)
    states_flat = [states_flat[k] for k in params_paths]
    grads_flat = utils.flatten_dict_string_keys(grads)
    grads_flat = [grads_flat[k] for k in params_paths]

    if hyper_params.global_norm_clip_threshold:
      # Paper: http://proceedings.mlr.press/v28/pascanu13.pdf
      # TF: https://www.tensorflow.org/api_docs/python/tf/clip_by_global_norm
      squared_l2_norms = [jnp.sum(jnp.square(g)) for g in grads_flat]
      global_norm = jnp.sqrt(jnp.sum(jnp.array(squared_l2_norms)))
      scale = hyper_params.global_norm_clip_threshold * jnp.minimum(
          1.0 / hyper_params.global_norm_clip_threshold, 1.0 / global_norm)
      grads_flat = [g * scale for g in grads_flat]

    out = [
        self.apply_param_gradient(step, hyper_params, param, state, grad, path)
        for param, state, grad, path in zip(params_flat, states_flat,
                                            grads_flat, params_paths)
    ]

    new_params_flat, new_states_flat = list(zip(*out)) if out else ((), ())
    new_params_flat = {k: v for k, v in zip(params_paths, new_params_flat)}
    new_states_flat = {k: v for k, v in zip(params_paths, new_states_flat)}
    new_params = _restore(params, new_params_flat)
    new_param_states = _restore(params, new_states_flat)
    new_state = OptimizerState(step + 1, new_param_states)

    return new_params, new_state

  def set_param_axes(self, param_logical_axes):
    """Sets Adafactor factorization map from logical axis names tree."""
    logical_factor_rules = self.hyper_params.logical_factor_rules
    if logical_factor_rules is None:
      return

    # pylint:disable=invalid-name
    NONE = FactorDim.NONE
    COLUMN = FactorDim.COLUMN
    ROW = FactorDim.ROW

    # pylint:enable=invalid-name

    def apply_rules(axes):
      # Partially factorized params are marked as unfactorized, preserving
      # only BATCH axis annotations. We also check for incompletely factorized
      # params that have ROW, COLUMN but also accidental NONE dimensions and
      # raise an error in that case.
      axis_rules = tuple(logical_factor_rules[x] for x in axes)
      axis_rules = tuple(factor_name_to_factordim(x) for x in axis_rules)
      if ROW in axis_rules and COLUMN in axis_rules and NONE in axis_rules:
        raise ValueError(f'Incomplete adafactor spec {axis_rules} for {axes}!')
      if ROW not in axis_rules or COLUMN not in axis_rules:
        axis_rules = tuple(
            NONE if x in (ROW, COLUMN) else x for x in axis_rules)
      return axis_rules

    factor_map = jax.tree_util.tree_map(apply_rules, param_logical_axes)
    factor_map = utils.flatten_dict_string_keys(factor_map)

    self.hyper_params = self.hyper_params.replace(factor_map=factor_map)

  def derive_logical_axes(self, optimizer_state, param_logical_axes):
    """Derives optimizer logical partitioning from model logical partitions."""
    optimizer_logical_axes = jax.tree_util.tree_map(
        lambda x: None, optimizer_state.state_dict())
    optimizer_logical_axes['target'] = param_logical_axes

    def factor_rule(logical_axes, adafactor_leaf):
      return dict(
          v_row=None,
          v_col=None,
          v=logical_axes if adafactor_leaf['v'].shape != (1,) else None,
          m=logical_axes if self.hyper_params.beta1 else None)

    optimizer_logical_axes['state']['param_states'] = jax.tree_util.tree_map(
        factor_rule, unfreeze(param_logical_axes),
        optimizer_state.state_dict()['state']['param_states'])

    return optimizer_state.restore_state(unfreeze(optimizer_logical_axes))
