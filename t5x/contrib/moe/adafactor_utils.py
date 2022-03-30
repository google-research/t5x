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

"""Adafactor logical rules for Mixture of Experts models."""

from flax import core as flax_core
from t5x import adafactor

FactorDim = adafactor.FactorDim
FrozenDict = flax_core.FrozenDict


def logical_factor_rules() -> FrozenDict:
  """Logical factor rules for Mixture of Experts."""
  rules = flax_core.unfreeze(adafactor.standard_logical_factor_rules())
  rules.update({
      'expert': FactorDim.BATCH,
      'expert_mlp': FactorDim.COLUMN,
      'unmodeled': FactorDim.NONE
  })
  return flax_core.freeze(rules)
