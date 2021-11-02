# Copyright 2021 The T5X Authors.
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

from flax import optim
from flax import struct
from flax.core import scope as flax_scope
import jax
import jax.numpy as jnp

PyTreeDef = type(jax.tree_structure(None))


class TrainState(struct.PyTreeNode):
  """Simple train state for holding parameters, step, optimizer state."""
  _optimizer: optim.Optimizer
  # Non-parameter variables.
  other_variables: Optional[flax_scope.FrozenVariableDict] = None
  stop_training: bool = struct.field(default=False, pytree_node=False)

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
    return self._optimizer.state_dict()

  def apply_gradient(self, grads, learning_rate) -> 'TrainState':
    new_optimizer = self._optimizer.apply_gradient(
        grads, learning_rate=learning_rate)
    return self.replace(_optimizer=new_optimizer)

  def restore_state(self, state_dict: Dict[str, Any]) -> 'TrainState':
    new_optimizer = self._optimizer.restore_state(state_dict)
    return self.replace(_optimizer=new_optimizer, stop_training=False)

  def update_step(self, step: int) -> 'TrainState':
    return self.replace(
        _optimizer=self._optimizer.replace(
            state=self._optimizer.state.replace(step=step)))

  @classmethod
  def from_flax_optimizer(
      cls,
      optimizer: optim.Optimizer,
      other_variables: Optional[flax_scope.FrozenVariableDict] = None
  ) -> 'TrainState':
    return cls(_optimizer=optimizer, other_variables=other_variables)
