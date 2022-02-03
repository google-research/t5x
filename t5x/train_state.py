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

"""Train state for passing around objects during training."""

from typing import Any, Mapping, MutableMapping, Optional

from flax import optim
import flax.core
from flax.core import scope as flax_scope
import flax.serialization
import flax.struct
import jax.numpy as jnp

import typing_extensions

EMPTY_DICT = flax.core.freeze({})


class TrainState(typing_extensions.Protocol):
  """TrainState interface."""

  @property
  def step(self) -> int:
    ...

  @property
  def params(self) -> flax_scope.FrozenVariableDict:
    ...

  @property
  def param_states(self) -> flax_scope.FrozenVariableDict:
    ...

  @property
  def axes_variables(self) -> Optional[flax_scope.FrozenVariableDict]:
    ...

  @property
  def flax_mutables(self) -> Optional[flax_scope.FrozenVariableDict]:
    ...

  def state_dict(self) -> Mapping[str, Any]:
    ...

  def apply_gradient(self,
                     grads,
                     learning_rate,
                     flax_mutables=EMPTY_DICT) -> 'TrainState':
    ...

  def restore_state(self, state_dict: Mapping[str, Any]) -> 'TrainState':
    ...

  def update_step(self, step: int) -> 'TrainState':
    ...


class InferenceTrainState(flax.struct.PyTreeNode):
  """Simple train state for holding parameters, step, optimizer state."""

  step: jnp.ndarray
  params: flax_scope.FrozenVariableDict
  axes_variables: Optional[flax_scope.FrozenVariableDict] = None
  # Flax mutable fields.
  flax_mutables: flax_scope.FrozenDict = EMPTY_DICT

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

  def restore_state(self, state_dict: Mapping[str,
                                              Any]) -> 'InferenceTrainState':
    return self.replace(
        params=flax.core.freeze(state_dict['target']),
        step=state_dict['state']['step'],
        flax_mutables=flax.core.freeze(state_dict['flax_mutables'])
        if 'flax_mutables' in state_dict else EMPTY_DICT)


class FlaxOptimTrainState(flax.struct.PyTreeNode):
  """Simple train state for holding parameters, step, optimizer state."""
  _optimizer: optim.Optimizer
  axes_variables: Optional[flax_scope.FrozenVariableDict] = None
  # Flax mutable fields.
  flax_mutables: flax_scope.FrozenDict = EMPTY_DICT

  @property
  def step(self) -> jnp.ndarray:
    return self._optimizer.state.step

  @property
  def params(self) -> flax_scope.FrozenVariableDict:
    return self._optimizer.target

  @property
  def param_states(self) -> flax_scope.FrozenVariableDict:
    return self._optimizer.state.param_states

  @property
  def optimizer_name(self):
    """Returns the name of the used optimizer."""
    return self._optimizer.optimizer_def.__class__.__name__

  def state_dict(self) -> MutableMapping[str, Any]:
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

  def restore_state(self, state_dict: MutableMapping[str, Any]) -> 'TrainState':
    new_optimizer = self._optimizer.restore_state(state_dict)
    return self.replace(
        _optimizer=new_optimizer,
        flax_mutables=flax.core.freeze(state_dict['flax_mutables'])
        if 'flax_mutables' in state_dict else EMPTY_DICT)

  def update_step(self, step: int) -> 'FlaxOptimTrainState':
    return self.replace(
        _optimizer=self._optimizer.replace(
            state=self._optimizer.state.replace(step=step)),
        flax_mutables=self.flax_mutables)

  @classmethod
  def from_flax_optimizer(
      cls,
      optimizer: optim.Optimizer,
      axes_variables: Optional[flax_scope.FrozenVariableDict] = None,
      flax_mutables: Optional[flax_scope.FrozenDict] = EMPTY_DICT
  ) -> 'FlaxOptimTrainState':
    return cls(
        _optimizer=optimizer,
        axes_variables=axes_variables,
        flax_mutables=flax_mutables)
