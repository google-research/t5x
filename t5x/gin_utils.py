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

"""Utilities for using gin configurations with T5X binaries."""
import os
from typing import Optional, Sequence, Union

from absl import app
from absl import logging
from clu import metric_writers
import gin
import jax
import tensorflow as tf



@gin.configurable
def get_gin_config_str(show_provenance: bool = False) -> str:
  """Utility function retrieving configuration in string form.

  This is only necessary to to provide a configurable name to toggle the
  show_provenance parameter.

  Args:
    show_provenance: Flag indicating whether to show (where possible) the
      provenance of configuration settings.

  Returns:
    Current gin configuration as string.
  """
  # The following ensures that existing configs will not fail on old gin version
  # that do not have the show_provenance parameter yet and makes this feature
  # opt-in.
  if show_provenance:
    return gin.config_str(show_provenance=True)
  else:
    return gin.config_str()


def parse_gin_flags(gin_search_paths: Sequence[str],
                    gin_files: Sequence[str],
                    gin_bindings: Sequence[str],
                    skip_unknown: Union[bool, Sequence[str]] = False,
                    finalize_config: bool = True):
  """Parses provided gin files override params.

  Args:
    gin_search_paths: paths that will be searched for gin files.
    gin_files: paths to gin config files to be parsed. Files will be parsed in
      order with conflicting settings being overriden by later files. Paths may
      be relative to paths in `gin_search_paths`.
    gin_bindings: individual gin bindings to be applied after the gin files are
      parsed. Will be applied in order with conflicting settings being overriden
      by later ones.
    skip_unknown: whether to ignore unknown bindings or raise an error (default
      behavior). Alternatively, a list of configurable names to skip if unknown.
    finalize_config: whether to finalize the config so that it cannot be
      modified (default behavior).
  """
  # We import t5.data here since it includes gin configurable functions commonly
  # used by task modules.
  # TODO(adarob): Strip gin from t5.data and remove this import.
  # Register .gin file search paths with gin
  for gin_file_path in gin_search_paths:
    gin.add_config_file_search_path(gin_file_path)


  # Parse config files and bindings passed via flag.
  gin.parse_config_files_and_bindings(
      gin_files,
      gin_bindings,
      skip_unknown=skip_unknown,
      finalize_config=finalize_config)
  logging.info('Gin Configuration:')
  for line in get_gin_config_str().splitlines():
    logging.info('%s', line)


def rewrite_gin_args(args: Sequence[str]) -> Sequence[str]:
  """Rewrite `--gin.NAME=VALUE` flags to `--gin_bindings=NAME=VALUE`."""

  def _rewrite_gin_arg(arg):
    if not arg.startswith('--gin.'):
      return arg
    if '=' not in arg:
      raise ValueError(
          "Gin bindings must be of the form '--gin.<param>=<value>', got: " +
          arg)
    # Strip '--gin.'
    arg = arg[6:]
    name, value = arg.split('=', maxsplit=1)
    r_arg = f'--gin_bindings={name} = {value}'
    logging.info('Rewritten gin arg: %s', r_arg)
    return r_arg

  return [_rewrite_gin_arg(arg) for arg in args]


@gin.register
def summarize_gin_config(model_dir: str,
                         summary_writer: Optional[metric_writers.MetricWriter],
                         step: int):
  """Writes gin config to the model dir and TensorBoard summary."""
  if jax.process_index() == 0:
    config_str = get_gin_config_str()
    tf.io.gfile.makedirs(model_dir)
    # Write the config as JSON.
    with tf.io.gfile.GFile(os.path.join(model_dir, 'config.gin'), 'w') as f:
      f.write(config_str)
    # Include a raw dump of the json as a text summary.
    if summary_writer is not None:
      summary_writer.write_texts(step, {'config': gin.markdown(config_str)})
      summary_writer.flush()


def run(main):
  """Wrapper for app.run that rewrites gin args before parsing."""
  app.run(
      main,
      flags_parser=lambda a: app.parse_flags_with_usage(rewrite_gin_args(a)))  # pytype: disable=wrong-arg-types


# ====================== Configurable Utility Functions ======================


@gin.configurable
def sum_fn(var1=gin.REQUIRED, var2=gin.REQUIRED):
  """sum function to use inside gin files."""
  return var1 + var2


@gin.configurable
def bool_fn(var1=gin.REQUIRED):
  """bool function to use inside gin files."""
  return bool(var1)


@gin.configurable
def string_split_fn(text=gin.REQUIRED,
                    separator=gin.REQUIRED,
                    maxsplit=-1,
                    index=None):
  """String split function to use inside gin files."""
  values = text.split(separator, maxsplit)
  if index is None:
    return values
  else:
    return values[index]
