# Copyright 2024 The T5X Authors.
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

"""Utilities for discovering checkpoints on disk.

This library contains only the relatively small/simple functions needed to
identify checkpoint directories. They can be used on a controller-type job that
doesn't need the ability to actually read the checkpoints and can thus
significantly reduce its binary size by not linking all the Jax libraries.
"""

import os
import re
from typing import Optional, Sequence, Tuple

from etils import epath
import gin
from tensorflow.io import gfile


_TRAIN_DS_PREFIX = 'train_ds'


@gin.configurable
def get_checkpoint_prefix(prefix='checkpoint'):
  return prefix


def all_steps(checkpoints_dir: str) -> Sequence[int]:
  """Returns list of available step numbers in ascending order."""
  glob_pattern = os.path.join(checkpoints_dir, 'checkpoint_*')
  checkpoint_paths = gfile.glob(glob_pattern)
  re_pattern = re.compile(r'.*/checkpoint_(\d+)$')
  matches = [re_pattern.match(ckpt) for ckpt in checkpoint_paths]
  return sorted(int(match.group(1)) for match in matches if match)


def all_dataset_checkpoint_steps(checkpoints_dir: str) -> Sequence[int]:
  """Returns available dataset checkpoint step numbers in ascending order."""
  glob_pattern = os.path.join(
      checkpoints_dir, 'checkpoint_*', f'{_TRAIN_DS_PREFIX}-*'
  )
  train_ds_paths = gfile.glob(glob_pattern)
  re_pattern = re.compile(r'.*/checkpoint_(\d+)/.*$')
  matches = [re_pattern.match(path) for path in train_ds_paths]
  return sorted(set(int(match.group(1)) for match in matches if match))


def latest_step(checkpoints_dir: str) -> Optional[int]:
  """Returns latest step number or None if no checkpoints exist."""
  steps = all_steps(checkpoints_dir)
  if not steps:
    return None
  return steps[-1]


def get_checkpoint_dir(
    checkpoints_dir: epath.PathLike,
    step: int,
    step_format_fixed_length: Optional[int] = None,
) -> epath.PathLike:
  """Returns path to a checkpoint dir given a parent directory and step."""
  step_str = (
      f'{step:0{step_format_fixed_length}d}'
      if step_format_fixed_length is not None
      else str(step)
  )
  return os.path.join(checkpoints_dir, f'{get_checkpoint_prefix()}_{step_str}')


def get_step_from_checkpoint_dir(checkpoints_dir: str) -> Tuple[str, int]:
  """Returns a step number and the parent directory."""
  if checkpoints_dir.endswith('/'):
    checkpoints_dir = checkpoints_dir[:-1]
  parent, checkpoint = os.path.split(checkpoints_dir)
  if get_checkpoint_prefix() not in checkpoint:
    raise ValueError('Found improperly formatted checkpoint directory.')
  return parent, int(checkpoint.replace(f'{get_checkpoint_prefix()}_', ''))
