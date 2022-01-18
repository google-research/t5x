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

"""Tests for train_state."""

from absl.testing import absltest
from flax import core as flax_core
from flax import linen as nn
from flax import optim
import jax
import numpy as np
from t5x import train_state as train_state_lib


class TrainStateTest(absltest.TestCase):

  def test_create_train_state(self):
    """Tests creating a train state."""
    model = nn.Dense(10)
    inputs = np.ones([2, 3], dtype=np.float32)
    params = model.init(jax.random.PRNGKey(0), inputs)['params']
    optimizer_def = optim.GradientDescent(learning_rate=0.1)
    optimizer = optimizer_def.create(params)
    flax_mutables = flax_core.freeze({'flax_mutable1': np.ones(10)})
    state = train_state_lib.TrainState.from_flax_optimizer(
        optimizer, flax_mutables=flax_mutables)
    self.assertEqual(state.step, 0)
    self.assertIsInstance(state._optimizer, optim.Optimizer)
    self.assertEqual(state.state_dict()['flax_mutables'],
                     flax_core.unfreeze(flax_mutables))
    jax.tree_multimap(np.testing.assert_array_equal, params, state.params)


if __name__ == '__main__':
  absltest.main()
