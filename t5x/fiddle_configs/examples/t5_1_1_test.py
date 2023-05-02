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

"""Tests for t5_1_1 examples."""

from absl.testing import absltest

import fiddle as fdl
from t5x import config_utils
from t5x.fiddle_configs.examples import t5_1_1


def _prepare_config(config: fdl.Buildable) -> fdl.Buildable:
  config = config_utils.prepare_to_summarize(config)
  # Avoid executing config during fdl.Build
  return fdl.cast(fdl.Partial, config)


class T511Test(absltest.TestCase):

  def test_partial_build_small_wmt_finetune(self):
    config = t5_1_1.small_wmt_finetune()
    config = _prepare_config(config)
    fdl.build(config)

  def test_partial_build_small_wmt_eval(self):
    config = t5_1_1.small_wmt_eval()
    config = _prepare_config(config)
    fdl.build(config)


if __name__ == '__main__':
  absltest.main()
