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

"""Utilities for configuring T5X binaries."""
import copy
import inspect
from typing import Callable, Optional, TypeVar

from absl import app
from absl import flags
from absl import logging
from clu import metric_writers
from etils import epath
import fiddle as fdl
from fiddle import absl_flags as fdl_flags
from fiddle import selectors
from fiddle.experimental import serialization
import jax
from t5x import gin_utils


FLAGS = flags.FLAGS


def using_fdl():
  """Returns true if any Fiddle configuration flags are set."""
  return (
      FLAGS.fdl_config is not None
      or FLAGS.fdl_config_file is not None
      or FLAGS.fdl_help
  )


T = TypeVar('T')


def prepare_to_summarize(config: fdl.Buildable[T]) -> fdl.Buildable[T]:
  """Update `config` so its `summarize_fiddle_config` calls can access it."""
  new_config = copy.deepcopy(config)
  # Here because get_current_fiddle_config is a lambda that returns config,
  # and not config itself, this does not cause recursive trouble for fiddle.
  selectors.select(new_config, direct_summarize_fiddle_config).set(
      get_current_fiddle_config=lambda: config
  )
  return new_config


def sanitize_summary_getter(config: fdl.Buildable[T]) -> fdl.Buildable[T]:
  """Update `config` to remove `get_current_fiddle_config` calls."""
  new_config = copy.deepcopy(config)
  # Here because get_current_fiddle_config is a lambda that returns config,
  # and not config itself, this does not cause recursive trouble for fiddle.
  selectors.select(new_config, direct_summarize_fiddle_config).set(
      get_current_fiddle_config=None
  )
  return new_config


def direct_summarize_fiddle_config(
    model_dir: str,
    summary_writer: Optional[metric_writers.MetricWriter],
    step: int,
    get_current_fiddle_config: Optional[Callable[[], fdl.Buildable]] = None,
):
  """Writes fiddle config to the model dir and TensorBoard summary.

  When passing this function to your fiddle config, don't pass the private
  function; instead pass the `fdl.Partial` version of `summarize_fiddle_config`.

  Args:
    model_dir: Model directory to write to.
    summary_writer: MetricWriter, if any.
    step: Current step.
    get_current_fiddle_config: This will be filled in by
      `t5x.config_utils.prepare_to_summarize()`.
  """
  if jax.process_index() != 0:
    return
  if not get_current_fiddle_config:
    raise ValueError(
        'get_current_fiddle_config() not provided.  Please pass your fiddle '
        'config through t5x.config_utils.prepare_to_summarize prior to '
        'building it.'
    )
  config = get_current_fiddle_config()
  config = sanitize_summary_getter(config)
  config_str = str(config)

  model_dir_path = epath.Path(model_dir)
  model_dir_path.mkdir(parents=True, exist_ok=True)

  # Write the config.
  (model_dir_path / 'fiddle_config.txt').write_text(config_str)

  # Try to serialize to json as well
  try:
    config_json = serialization.dump_json(config)
    (model_dir_path / 'fiddle_config.json').write_text(config_json)
  except serialization.UnserializableValueError as e:
    logging.warning(
        'Unable to JSON Serialize fiddle config, skipping. Error: %s', e
    )

  if summary_writer is not None:
    summary_writer.write_texts(step, {'fiddle_config': config_str})
    summary_writer.flush()


# Pass this when configuring the argument `summarize_config_fn`.
summarize_fiddle_config = fdl.Partial(direct_summarize_fiddle_config)


def config_with_fiddle(
    function: Callable[..., T],
) -> fdl.Buildable[Callable[..., T]]:
  """Configure and build a T5X launcher from Fiddle command line flags.

  The output config, when called via `fdl.build()`, will execute `function`.

  Args:
    function: A function that launches a T5X job, e.g., `train`, `eval`, ...

  Returns:
    The buildable of the function or object, depending on whether
      `--fdl_config_file` or `--fdl_config` was passed.

  Raises:
    AssertionError: If `not using_fdl()`.
    ValueError: If both fiddle and gin arguments were passed on the command
      line.
    ValueError: If both `--fdl_config_file` and `--fdl_config` were passed.
    ValueError: If the object built via `--fdl_config` does not build as a
      call to `function`.
  """
  assert using_fdl(), 'No fiddle command line flags found'
  if (FLAGS.fdl_config_file or FLAGS.fdl_config) and (
      FLAGS.gin_file or FLAGS.gin_bindings
  ):
    raise ValueError(
        'Must pass exactly one of `--fdl_config_file`, `--fdl_config`, or '
        '`--gin_file` / `--gin_bindings`. Got: '
        f'--fdl_config_file={FLAGS.fdl_config_file} '
        f'--fdl_config={FLAGS.fdl_config} '
        f'--gin_file={FLAGS.gin_file}.'
        f'--gin_bindings={FLAGS.gin_bindings}.'
    )
  if FLAGS.fdl_config_file and FLAGS.fdl_config:
    raise ValueError(
        'Must pass exactly one of `--fdl_config_file` or `--fdl_config`.  Got: '
        f'--fdl_config_file={FLAGS.fdl_config_file} '
        f'--fdl_config={FLAGS.fdl_config}.'
    )

  if FLAGS.fdl_config_file:
    # Fill in the launcher function args using a fiddle config json.
    config = fdl_flags.create_buildable_from_flags(function)
  elif FLAGS.fdl_config:
    # Build a launcher object using a fiddle config module+function.
    config = fdl_flags.create_buildable_from_flags(function, allow_imports=True)
  else:
    raise AssertionError('Should not get to this point.')

  # If this is a fdl.Config<function, ...> we want to convert it to
  # fdl.Partial<function, ...> so that function() does not execute when
  # fdl.build(config) is called.
  config = fdl.cast(fdl.Partial, config)

  config_module = inspect.getmodule(config.__fn_or_cls__)
  function_module = inspect.getmodule(function)

  # Best effort to ensure that config and function match, even if the json
  # defines a different alias to the same module, like __main__ ~= t5x.train.
  if (config.__fn_or_cls__.__qualname__ != function.__qualname__) or (
      inspect.getsource(config_module) != inspect.getsource(function_module)
  ):

    def module_and_name(fn: Callable[..., T]) -> str:
      return '.'.join((fn.__module__, fn.__qualname__))

    raise ValueError(
        'Expected fiddle flags to configure function '
        f'{module_and_name(function)} but it configured '
        f'{module_and_name(config.__fn_or_cls__)}.\n\nConfig: {config}'
    )

  # Ensure that summarize_fiddle_config calls will work.
  config = prepare_to_summarize(config)

  return config


def run(main):
  """Wrapper for app.run that rewrites jax, gin, and fiddle flags."""

  def flags_parser(args):
    args = gin_utils.rewrite_gin_args(args)
    return fdl_flags.flags_parser(args)

  jax.config.parse_flags_with_absl()
  if using_fdl():
    app.run(main, flags_parser=flags_parser)
  else:
    gin_utils.run(main)
