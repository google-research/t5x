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

"""Train state for passing around objects during training."""

from typing import Any, Mapping, MutableMapping, Optional, Tuple

from flax import traverse_util
import flax.core
from flax.core import scope as flax_scope
from flax.linen import partitioning as flax_partitioning
import flax.serialization
import flax.struct
import jax.numpy as jnp
from t5x import optimizers

import typing_extensions

EMPTY_DICT = flax.core.freeze({})
FrozenDict = flax_scope.FrozenDict
FrozenVariableDict = flax_scope.FrozenVariableDict
MutableVariableDict = flax_scope.MutableVariableDict
VariableDict = flax_scope.VariableDict


@typing_extensions.runtime_checkable
class TrainState(typing_extensions.Protocol):
  """TrainState interface."""

  @property
  def step(self) -> jnp.ndarray:
    """The current training step as an integer scalar."""
    ...

  @property
  def params(self) -> FrozenVariableDict:
    """The parameters of the model as a PyTree matching the Flax module."""
    ...

  @property
  def param_states(self) -> FrozenVariableDict:
    """The optimizer states of the parameters as a PyTree."""
    ...

  @property
  def flax_mutables(self) -> FrozenVariableDict:
    """Flax mutable collection."""
    ...

  def state_dict(self) -> MutableVariableDict:
    """Returns a mutable representation of the state for checkpointing."""
    ...

  def restore_state(self, state_dict: Mapping[str, Any]) -> 'TrainState':
    """Restores the object state from a state dict."""
    ...

  def replace_params(self, params: VariableDict) -> 'TrainState':
    ...

  def replace_flax_mutables(self, flax_mutables: FrozenDict) -> 'TrainState':
    ...

  def replace_step(self, step: jnp.ndarray) -> 'TrainState':
    ...

  def apply_gradient(self,
                     grads,
                     learning_rate,
                     flax_mutables=EMPTY_DICT) -> 'TrainState':
    """Applies gradient, increments step, and returns an updated TrainState."""
    ...

  def as_logical_axes(self) -> 'TrainState':
    """Replaces `param` and `param-states` with their logical axis names."""
    ...


def _validate_params_axes(params_axes, params):
  axis_names = flax_partitioning.get_axis_names(params_axes)
  missing_params_axes = (
      set(traverse_util.flatten_dict(params, sep='/')) -
      set(traverse_util.flatten_dict(axis_names, sep='/')))
  if missing_params_axes:
    raise ValueError(
        f'Missing axis names for parameters: {missing_params_axes}')


def _split_variables_and_axes(
    variables_and_axes: FrozenVariableDict
) -> Tuple[FrozenVariableDict, FrozenVariableDict]:
  """Splits `variables_and_axes` into two separate dicts with the same keys."""
  # For each `key`, `key_axes` (if any) are its axes in `variables_and_axes`.
  variables = {}
  axes = {}
  for k, v in variables_and_axes.items():
    if k.endswith('_axes'):
      axes[k[:-5]] = v  # k without "_axes".
      _validate_params_axes(v, variables_and_axes[k[:-5]])  # k without "_axes".
    else:
      variables[k] = v
  return flax.core.freeze(variables), flax.core.freeze(axes)


class FlaxOptimTrainState(flax.struct.PyTreeNode):
  """Simple train state for holding parameters, step, optimizer state."""
  _optimizer: optimizers.OptimizerType
  # Contains axis metadata (e.g., names) matching parameter tree.
  params_axes: Optional[FrozenVariableDict] = None
  # Flax mutable fields.
  flax_mutables: FrozenDict = EMPTY_DICT
  # Contains axis metadata (e.g., names) matching flax_mutables tree.
  flax_mutables_axes: Optional[FrozenVariableDict] = None

  @classmethod
  def create(
      cls,
      optimizer_def: optimizers.OptimizerDefType,
      model_variables: FrozenVariableDict,
  ) -> 'FlaxOptimTrainState':
    other_variables, params = flax.core.pop(model_variables, 'params')
    if 'params_axes' in other_variables:
      other_variables, params_axes = flax.core.pop(
          other_variables, 'params_axes'
      )
      _validate_params_axes(params_axes, params)
    else:
      params_axes = None

    # Split other_variables into mutables and their corresponding axes.
    flax_mutables, flax_mutables_axes = _split_variables_and_axes(
        other_variables
    )

    # If the optimizer supports `set_param_axes`, then assume that the model
    # code is emitting these axes and use it.
    if hasattr(optimizer_def, 'set_param_axes'):
      if params_axes is None:
        raise ValueError(
            'The optimizer supports params_axes for model-based '
            'partitioning, but the model is not emitting them.'
        )
      # `get_axis_names` removes "_axes" suffix in the leaf name and replaces
      # `AxisMetadata` with `PartitionSpec`.
      axis_names = flax_partitioning.get_axis_names(params_axes)
      optimizer_def.set_param_axes(axis_names)

    optimizer = optimizer_def.create(params)
    flax_mutables_axes = flax_mutables_axes or None
    return FlaxOptimTrainState(
        optimizer,
        params_axes=params_axes,
        flax_mutables=flax_mutables,
        flax_mutables_axes=flax_mutables_axes)

  @property
  def step(self) -> jnp.ndarray:
    return self._optimizer.state.step

  @property
  def params(self) -> FrozenVariableDict:
    return self._optimizer.target

  @property
  def param_states(self) -> FrozenVariableDict:
    return self._optimizer.state.param_states

  def state_dict(self) -> MutableVariableDict:
    state_dict = self._optimizer.state_dict()
    if self.flax_mutables:
      state_dict['flax_mutables'] = flax.core.unfreeze(self.flax_mutables)
    return state_dict

  def apply_gradient(self,
                     grads,
                     learning_rate,
                     flax_mutables=EMPTY_DICT) -> 'FlaxOptimTrainState':
    new_optimizer = self._optimizer.apply_gradient(
        grads, learning_rate=learning_rate)
    return self.replace(_optimizer=new_optimizer, flax_mutables=flax_mutables)

  def replace_params(self, params: VariableDict) -> 'FlaxOptimTrainState':
    return self.replace(_optimizer=self._optimizer.replace(target=params))

  def replace_flax_mutables(self,
                            flax_mutables: FrozenDict) -> 'FlaxOptimTrainState':
    return self.replace(flax_mutables=flax_mutables)

  def replace_step(self, step: jnp.ndarray) -> 'FlaxOptimTrainState':
    state_dict = self.state_dict()
    state_dict['state']['step'] = step
    return self.restore_state(state_dict)

  def restore_state(self, state_dict: VariableDict) -> 'FlaxOptimTrainState':
    new_optimizer = self._optimizer.restore_state(state_dict)
    return self.replace(
        _optimizer=new_optimizer,
        flax_mutables=flax.core.freeze(state_dict['flax_mutables'])
        if 'flax_mutables' in state_dict else EMPTY_DICT)

  def as_logical_axes(self) -> 'FlaxOptimTrainState':
    if not hasattr(self._optimizer.optimizer_def, 'derive_logical_axes'):
      raise ValueError(
          f"Optimizer '{self._optimizer.optimizer_def.__class__.__name__}' "
          'requires a `derive_logical_axes` method to be used with named axis '
          'partitioning.')
    flax_mutables_axes = self.flax_mutables_axes or EMPTY_DICT
    return FlaxOptimTrainState(
        _optimizer=self._optimizer.optimizer_def.derive_logical_axes(
            self._optimizer,
            flax_partitioning.get_axis_names(self.params_axes)),
        flax_mutables=flax_partitioning.get_axis_names(flax_mutables_axes))


class InferenceState(flax.struct.PyTreeNode):
  """State compatible with FlaxOptimTrainState without optimizer state."""

  step: jnp.ndarray
  params: flax_scope.FrozenVariableDict
  params_axes: Optional[flax_scope.FrozenVariableDict] = None
  flax_mutables: flax_scope.FrozenDict = EMPTY_DICT
  flax_mutables_axes: Optional[flax_scope.FrozenVariableDict] = None

  @classmethod
  def create(cls, model_variables: FrozenVariableDict) -> 'InferenceState':
    other_variables, params = flax.core.pop(model_variables, 'params')
    if 'params_axes' in other_variables:
      other_variables, params_axes = flax.core.pop(
          other_variables, 'params_axes'
      )
      _validate_params_axes(params_axes, params)
    else:
      params_axes = None

    # Split other_variables into mutables and their corresponding axes.
    flax_mutables, flax_mutables_axes = _split_variables_and_axes(
        other_variables
    )
    flax_mutables_axes = flax_mutables_axes or None
    return InferenceState(
        step=jnp.array(0),
        params=params,
        params_axes=params_axes,
        flax_mutables=flax_mutables,
        flax_mutables_axes=flax_mutables_axes,
    )

  @property
  def param_states(self) -> FrozenVariableDict:
    """The optimizer states of the parameters as a PyTree."""
    raise NotImplementedError('InferenceState has no optimizer states.')

  def apply_gradient(self, *args, **kwargs) -> 'InferenceState':
    raise NotImplementedError(
        'InferenceState does not support `apply_gradient`.')

  def state_dict(self) -> MutableMapping[str, Any]:
    state_dict = {
        'target': flax.core.unfreeze(self.params),
        'state': {
            'step': self.step
        }
    }
    if self.flax_mutables:
      state_dict['flax_mutables'] = flax.core.unfreeze(self.flax_mutables)
    return state_dict

  def replace_step(self, step: jnp.ndarray) -> 'InferenceState':
    return self.replace(step=step)

  def replace_params(self, params: FrozenVariableDict) -> 'InferenceState':
    return self.replace(params=params)

  def replace_flax_mutables(self,
                            flax_mutables: FrozenDict) -> 'InferenceState':
    return self.replace(flax_mutables=flax_mutables)

  def restore_state(self, state_dict: Mapping[str, Any]) -> 'InferenceState':
    return self.replace(
        params=flax.core.freeze(state_dict['target']),
        step=state_dict['state']['step'],
        flax_mutables=flax.core.freeze(state_dict['flax_mutables'])
        if 'flax_mutables' in state_dict else EMPTY_DICT)

  def as_logical_axes(self) -> 'InferenceState':
    # Set step to None so that when the logical axes are processed by the
    # flax.partitioning.logical_to_mesh_axes function, it will be skipped
    # because jax.tree_map will short circut and never call the function on the
    # step.
    flax_mutables_axes = self.flax_mutables_axes or EMPTY_DICT
    return InferenceState(
        step=None,
        params=flax_partitioning.get_axis_names(self.params_axes),
        flax_mutables=flax_partitioning.get_axis_names(flax_mutables_axes))
