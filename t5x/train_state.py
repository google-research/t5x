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

from typing import Any, Dict, Optional

from absl import logging
from flax import core as flax_core
from flax import optim
from flax import struct
from flax.core import scope as flax_scope
import jax
import jax.numpy as jnp

PyTreeDef = type(jax.tree_structure(None))
EMPTY_DICT = flax_core.freeze({})


class TrainState(struct.PyTreeNode):
  """Simple train state for holding parameters, step, optimizer state.

  This copy will soon become an interface rather than an implementation.
  """
  _optimizer: optim.Optimizer
  # Variables related with axes specification.
  axes_variables: Optional[flax_scope.FrozenVariableDict] = None
  # Flax mutable fields.
  flax_mutables: Optional[flax_scope.FrozenDict] = EMPTY_DICT

  @property
  def step(self) -> jnp.ndarray:
    return self._optimizer.state.step

  @property
  def params(self) -> PyTreeDef:
    return self._optimizer.target

  @property
  def param_states(self) -> PyTreeDef:
    return self._optimizer.state.param_states

  @property
  def optimizer_name(self):
    """Returns the name of the used optimizer."""
    return self._optimizer.optimizer_def.__class__.__name__

  def state_dict(self) -> Dict[str, Any]:
    state_dict = self._optimizer.state_dict()
    if self.flax_mutables:
      state_dict['flax_mutables'] = flax_core.unfreeze(self.flax_mutables)
    return state_dict

  def apply_gradient(self,
                     grads,
                     learning_rate,
                     flax_mutables=EMPTY_DICT) -> 'TrainState':
    new_optimizer = self._optimizer.apply_gradient(
        grads, learning_rate=learning_rate)
    return self.replace(_optimizer=new_optimizer, flax_mutables=flax_mutables)

  def restore_state(self, state_dict: Dict[str, Any]) -> 'TrainState':
    new_optimizer = self._optimizer.restore_state(state_dict)
    return self.replace(
        _optimizer=new_optimizer,
        flax_mutables=flax_core.freeze(state_dict['flax_mutables'])
        if 'flax_mutables' in state_dict else EMPTY_DICT)

  def update_step(self, step: int) -> 'TrainState':
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
  ) -> 'TrainState':
    logging.error(
        '`from_flax_optimizer` is deprecated and will be removed shortly. '
        'Please instantiate `FlaxOptimTrainState` directly instead.')
    return cls(
        _optimizer=optimizer,
        axes_variables=axes_variables,
        flax_mutables=flax_mutables)


class FlaxOptimTrainState(TrainState):
  """Train state flax.optim.Optimizer-based optimization."""

  @classmethod
  def from_flax_optimizer(
      cls,
      optimizer: optim.Optimizer,
      axes_variables: Optional[flax_scope.FrozenVariableDict] = None,
      flax_mutables: Optional[flax_scope.FrozenDict] = EMPTY_DICT
  ) -> 'FlaxOptimTrainState':
    raise NotImplementedError('Initialize `FlaxOptimTrainState` directly.')
