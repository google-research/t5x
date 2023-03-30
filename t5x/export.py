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

r"""Exports a T5X model.


"""
import os
from typing import Sequence
from absl import logging

# Set Linen to add profiling information when constructing Modules.
# Must be set before flax imports.
# pylint:disable=g-import-not-at-top
os.environ.setdefault('FLAX_PROFILE', 'true')

import jax
from t5x import export_lib

if __name__ == '__main__':
  # pylint:disable=g-import-not-at-top
  from absl import app
  from absl import flags
  import gin
  from t5x import gin_utils
  # pylint:enable=g-import-not-at-top

  FLAGS = flags.FLAGS

  jax.config.parse_flags_with_absl()


  flags.DEFINE_multi_string(
      'gin_file',
      default=None,
      help='Path to gin configuration file. Multiple paths may be passed and '
      'will be imported in the given order, with later configurations  '
      'overriding earlier ones.')

  flags.DEFINE_multi_string(
      'gin_bindings',
      default=[],
      help='Individual gin bindings. Also used to integrate gin and XManager.')

  flags.DEFINE_list(
      'gin_search_paths',
      default=['t5x/configs'],
      help='Comma-separated list of gin config path prefixes to be prepended '
      'to suffixes given via `--gin_file`. If a file appears in. Only the '
      'first prefix that produces a valid path for each suffix will be '
      'used.')

  def main(argv: Sequence[str]):
    """Wrapper for g3pdb post mortems."""
    _main(argv)

  def _main(argv: Sequence[str]):
    """True main function."""
    if len(argv) > 1:
      raise app.UsageError('Too many command-line arguments.')

    save_with_gin = gin.configurable(export_lib.save)

    gin_utils.parse_gin_flags(FLAGS.gin_search_paths, FLAGS.gin_file,
                              FLAGS.gin_bindings)
    logging.info('Creating inference function...')
    save_with_gin()

  gin_utils.run(main)
