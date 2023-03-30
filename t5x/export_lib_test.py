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

"""Tests for variable shape input in export_lib."""

from absl.testing import absltest
from absl.testing import parameterized
from t5x import export_lib
import tensorflow as tf  # type: ignore


class ExportLibTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.shape_config = export_lib.VariableInputShapeConfig(
        allow_variable_sequence_length=True, allowed_sequence_lengths=(10, 15)
    )

  @parameterized.parameters(
      (0, 10), (4, 10), (10, 10), (11, 15), (15, 15), (16, 15)
  )
  def test_variable_shape_config_rounding(self, input_length, expected_out):
    self.assertEqual(
        export_lib._round_up_sequence_length(
            self.shape_config, tf.constant(input_length)
        ),
        tf.constant(expected_out),
    )


if __name__ == "__main__":
  absltest.main()
