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

"""Disables gc during each top-level import.

Only takes effect when environment variable
EXPERIMENTAL_DISABLE_GC_DURING_IMPORT
is true.

Some libraries like SeqIO have lots of side-effects during import time.
In some cases, disabling garbage collection for each top-level import can save
minutes of startup time.

This should be _relatively_ safe, because we don't expect that it's often that
1. There's sufficient memory pressure during an import to cause an OOM, and
2. That memory pressure would have been sufficiently alleviated by garbage
   collection.
"""

import builtins
import contextlib
import gc
import os


@contextlib.contextmanager
def disabled_gc():
  """When used as context manager, prevents garbage collection in scope."""
  if not gc.isenabled():
    # GC is already disabled; don't make any changes.
    yield
    return

  gc.disable()
  try:
    yield
  finally:
    # We know that the original state was enabled because
    # we didn't return above.
    gc.enable()


_original_importlib_import = builtins.__import__


def gc_disabled_import(*args, **kwargs):
  with disabled_gc():
    return _original_importlib_import(*args, **kwargs)


def try_disable_gc_during_import():
  if os.environ.get('EXPERIMENTAL_DISABLE_GC_DURING_IMPORT'):
    builtins.__import__ = gc_disabled_import
