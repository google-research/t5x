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

"""Checkpoint helper functions for managing checkpoints.

Supports marking checkpoints as pinned to exclude them from the checkpointer
removal process.
"""

import enum
import os
from typing import Any, BinaryIO, Optional

from absl import logging
from etils import epath
import msgpack
from tensorflow.io import gfile

# PINNED file in the checkpoint directory indicates that the checkpoint should
# not be removed during the automatic pruning of old checkpoints.
_PINNED_CHECKPOINT_FILENAME = 'PINNED'

PyTree = Any


def pinned_checkpoint_filepath(ckpt_dir: str) -> str:
  """Full path of the pinned checkpoint file."""
  return os.path.join(ckpt_dir, _PINNED_CHECKPOINT_FILENAME)


def is_pinned_checkpoint(ckpt_dir: str) -> bool:
  """Returns whether the checkpoint is pinned, and should NOT be removed."""
  pinned_ckpt_file = pinned_checkpoint_filepath(ckpt_dir)
  if gfile.exists(pinned_ckpt_file):
    return True
  return False


def pin_checkpoint(ckpt_dir: str, txt: str = '1') -> None:
  """Pin a checkpoint so it does not get deleted by the normal pruning process.

  Creates a PINNED file in the checkpoint directory to indicate the checkpoint
  should be excluded from the deletion of old checkpoints.

  Args:
    ckpt_dir: The checkpoint step dir that is to be always kept.
    txt: Text to be written into the checkpoints ALWAYS_KEEP me file.
  """
  pinned_ckpt_file = pinned_checkpoint_filepath(ckpt_dir)
  with gfile.GFile(pinned_ckpt_file, 'w') as f:
    logging.debug('Write %s file : %s.', pinned_ckpt_file, txt)
    f.write(txt)


def unpin_checkpoint(ckpt_dir: str) -> None:
  """Removes the pinned status of the checkpoint so it is open for deletion."""
  if not is_pinned_checkpoint(ckpt_dir):
    logging.debug('%s is not PINNED. Nothing to do here.', ckpt_dir)
    return
  try:
    pinned_ckpt_file = pinned_checkpoint_filepath(ckpt_dir)
    logging.debug('Remove %s file.', pinned_ckpt_file)
    gfile.rmtree(pinned_ckpt_file)
  except IOError:
    logging.exception('Failed to unpin %s', ckpt_dir)


def remove_checkpoint_dir(ckpt_dir: str) -> None:
  """Removes the checkpoint dir if it is not pinned."""
  if not is_pinned_checkpoint(ckpt_dir):
    logging.info('Deleting checkpoint: %s', ckpt_dir)
    gfile.rmtree(ckpt_dir)
  else:
    logging.info('Keeping pinned checkpoint: %s', ckpt_dir)


def remove_dataset_checkpoint(ckpt_dir: str, train_ds_prefix: str) -> None:
  """Removes dataset checkpoints if the checkpoint is not pinned."""
  if not is_pinned_checkpoint(ckpt_dir):
    train_ds_pattern = os.path.join(ckpt_dir, train_ds_prefix + '*')
    logging.info('Deleting dataset checkpoint: %s', train_ds_pattern)
    for file in gfile.glob(train_ds_pattern):
      gfile.remove(file)
  else:
    logging.info('Keeping pinned checkpoint: %s', ckpt_dir)


def _read_msgpack_keys(file_like: BinaryIO) -> PyTree:
  """Returns a tree containing all keys but no values from a msgpack file."""
  unpacker = msgpack.Unpacker(file_like)
  num_keys = unpacker.read_map_header()
  ret = {}

  # Contains references to the parent tree for each key to visit in the
  # msgpack file traversal.
  visit_stack = [ret for _ in range(num_keys)]
  while visit_stack:
    parent_dict = visit_stack.pop()
    key = unpacker.unpack()
    if isinstance(key, bytes):
      key = str(unpacker.unpack(), 'utf-8')

    # Check if the value object is map.
    try:
      n = unpacker.read_map_header()
      ref = parent_dict[key] = {}
      visit_stack.extend(ref for _ in range(n))
    except msgpack.UnpackValueError:
      # Not a map so skip unpacking the value object and record the current key.
      unpacker.skip()
      parent_dict[key] = None

  return ret


def _contains_ts_spec(tree: PyTree) -> bool:
  """Returns whether the a Pytree contains a serialized ts.Spec object."""
  to_visit = [tree]
  while to_visit:
    cur = to_visit.pop()
    if cur.keys() >= {'driver', 'kvstore', 'metadata'}:
      return True
    to_visit.extend(v for v in cur.values() if isinstance(v, dict))
  return False


# Constant copied from orbax/checkpoint/pytree_checkpoint_handler.py
_METADATA_FILE = '_METADATA'


def _contains_orbax_metadata(ckpt_path: str) -> bool:
  metadata = os.path.join(os.path.dirname(ckpt_path), _METADATA_FILE)
  return gfile.exists(metadata)


class CheckpointTypes(enum.Enum):
  ORBAX = 'ORBAX'
  T5X = 'T5X'
  T5X_TF = 'T5X_TF'


def _warn_if_unexpected_type(
    checkpoint_path, checkpoint_type, expected, extra_warn_log
):
  """Warns the user if unexpected type found."""
  if expected is None or checkpoint_type == expected:
    return

  logging.warning(
      'Expected the checkpoint at %s to be %s format, but'
      ' the actual detected format was %s.',
      checkpoint_path,
      expected,
      checkpoint_type,
  )
  logging.warning(extra_warn_log)


def detect_checkpoint_type(
    checkpoint_path: epath.PathLike, expected: Optional[CheckpointTypes] = None
) -> CheckpointTypes:
  """Returns the checkpoint type by reading the `.checkpoint` metadata file.

  Args:
    checkpoint_path: The path of the `.checkpoint` file.
    expected: The expected checkpoint type. If the checkpoint type is not as
      expected, this function will log a warning but will not raise an error.

  Returns:
    The checkpoint type.
  """
  if _contains_orbax_metadata(checkpoint_path):
    checkpoint_type = CheckpointTypes.ORBAX
    _warn_if_unexpected_type(
        checkpoint_path,
        checkpoint_type,
        expected,
        f'Found `{_METADATA_FILE}` in the checkpoint directory, which only '
        'appears in Orbax checkpoints',
    )
    return checkpoint_type

  with gfile.GFile(checkpoint_path, 'rb') as fp:
    raw_contents = fp.read(21)
    if raw_contents == b'model_checkpoint_path':
      checkpoint_type = CheckpointTypes.T5X_TF
      _warn_if_unexpected_type(
          checkpoint_path,
          checkpoint_type,
          expected,
          'The checkpoint file was not a msgpack, and had the string '
          '"model_checkpoint_path", so it was assumed to be in the T5X '
          'TensorFlow format.',
      )
      return checkpoint_type

    # Assume that if the msgpack file has exactly 'version' and 'optimizer' as
    # keys, it is a T5X checkpoint. Checkpoints that were created a long time
    # ago may not contain these keys, so there is a backup ts.Spec check
    # as well.
    fp.seek(0)
    key_tree = _read_msgpack_keys(fp)
    if set(key_tree.keys()) == {'version', 'optimizer'}:
      checkpoint_type = CheckpointTypes.T5X
      _warn_if_unexpected_type(
          checkpoint_path,
          checkpoint_type,
          expected,
          'Top-level keys in the msgpack file were "version" and "optimizer", '
          'thus the checkpoint was assumed to be in the T5X format.',
      )
      return checkpoint_type
    elif _contains_ts_spec(key_tree):
      # If the checkpoint contains a ts.Spec, it could either be a T5X
      # checkpoint or an early version Flax checkpoint. The latter is
      # essentially deprecated but should also be handled by the T5X
      # Checkpointer, so we return T5X here for simplicity.
      checkpoint_type = CheckpointTypes.T5X
      _warn_if_unexpected_type(
          checkpoint_path,
          checkpoint_type,
          expected,
          'Found ts.Spec in the checkpoint msgpack file, thus the checkpoint'
          ' was assumed to be in the T5X format.',
      )
      return checkpoint_type
    else:
      checkpoint_type = CheckpointTypes.ORBAX
      _warn_if_unexpected_type(
          checkpoint_path,
          checkpoint_type,
          expected,
          'Did not detect ts.Spec nor the {"version", "optimizer"} keys in the'
          'checkpoint msgpack file, so the checkpoint was assumed to be '
          'written with Orbax.',
      )
      return checkpoint_type
