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

"""Tests for disable_gc_during_import."""

# pylint: disable=g-import-not-at-top,unused-import

import builtins
import gc
import importlib
import os
import sys
from absl.testing import absltest
from absl.testing import parameterized
from t5x import disable_gc_during_import

_ORIGINAL_BUILTIN_IMPORT_FN = builtins.__import__


def assert_gc_disabled_during_import():
  # Side effect of importing module is asserting gc is disabled.
  if sys.modules.get("t5x.assert_gc_disabled_during_import_test_util"):
    sys.modules.pop("t5x.assert_gc_disabled_during_import_test_util", None)

  import t5x.assert_gc_disabled_during_import_test_util


class DisableGcDuringImportTest(parameterized.TestCase):

  def setUp(self):
    super(DisableGcDuringImportTest, self).setUp()
    builtins.__import__ = _ORIGINAL_BUILTIN_IMPORT_FN
    os.environ["EXPERIMENTAL_DISABLE_GC_DURING_IMPORT"] = "true"

  def tearDown(self):
    super(DisableGcDuringImportTest, self).tearDown()
    builtins.__import__ = _ORIGINAL_BUILTIN_IMPORT_FN
    os.environ.pop("EXPERIMENTAL_DISABLE_GC_DURING_IMPORT")

  def test_gc_enabled_after_one_import_import_builtin(self):
    disable_gc_during_import.try_disable_gc_during_import()

    self.assertTrue(gc.isenabled())
    # Some arbitrary import; not particularly important.
    import enum

    assert_gc_disabled_during_import()

    self.assertTrue(gc.isenabled())

  def test_gc_enabled_after_two_imports_import_builtin(self):
    disable_gc_during_import.try_disable_gc_during_import()
    # from t5x import disable_gc_during_import

    self.assertTrue(gc.isenabled())
    # Some arbitrary imports; not particularly important which ones.
    import contextlib
    import enum

    assert_gc_disabled_during_import()

    self.assertTrue(gc.isenabled())

  def test_test_utils_appropriately_detect_when_gc_enabled(self):
    with self.assertRaisesRegex(ValueError, "Expected gc to be disabled"):
      assert_gc_disabled_during_import()


if __name__ == "__main__":
  absltest.main()
