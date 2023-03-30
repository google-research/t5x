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

"""Tests for t5x.checkpoint_utils."""

import os
import traceback

from absl.testing import absltest
from t5x import checkpoint_utils
from tensorflow.io import gfile

TESTDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata")


class CheckpointsUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.checkpoints_dir = self.create_tempdir()
    self.ckpt_dir_path = self.checkpoints_dir.full_path
    self.pinned_ckpt_file = os.path.join(self.ckpt_dir_path, "PINNED")
    self.checkpoints_dir.create_file("checkpoint")
    # Create a `train_ds` file representing the dataset checkpoint.
    train_ds_basename = "train_ds-00000-of-00001"
    self.train_ds_file = os.path.join(self.ckpt_dir_path, train_ds_basename)
    self.checkpoints_dir.create_file(train_ds_basename)

  def test_always_keep_checkpoint_file(self):
    self.assertEqual(
        "/path/to/ckpt/dir/PINNED",
        checkpoint_utils.pinned_checkpoint_filepath("/path/to/ckpt/dir"))

  def test_is_pinned_checkpoint_false_by_default(self):
    # Ensure regular checkpoint without PINNED file.
    self.assertFalse(gfile.exists(os.path.join(self.ckpt_dir_path, "PINNED")))

    # Validate checkpoints are not pinned by default.
    self.assertFalse(checkpoint_utils.is_pinned_checkpoint(self.ckpt_dir_path))

  def test_is_pinned_checkpoint(self):
    # Ensure the checkpoint directory as pinned.
    pinned_ckpt_testdata = os.path.join(TESTDATA, "pinned_ckpt_dir")
    pinned_file = os.path.join(pinned_ckpt_testdata, "PINNED")
    self.assertTrue(gfile.exists(pinned_file))

    # Test and validate.
    self.assertTrue(checkpoint_utils.is_pinned_checkpoint(pinned_ckpt_testdata))

  def test_is_pinned_missing_ckpt(self):
    self.assertFalse(
        checkpoint_utils.is_pinned_checkpoint(
            os.path.join(self.ckpt_dir_path, "ckpt_does_not_exist")))

  def test_pin_checkpoint(self):
    # Ensure directory isn't already pinned.
    self.assertFalse(gfile.exists(self.pinned_ckpt_file))

    # Test.
    checkpoint_utils.pin_checkpoint(self.ckpt_dir_path)

    # Validate.
    self.assertTrue(gfile.exists(self.pinned_ckpt_file))
    with open(self.pinned_ckpt_file) as f:
      self.assertEqual("1", f.read())

  def test_pin_checkpoint_txt(self):
    checkpoint_utils.pin_checkpoint(self.ckpt_dir_path, "TEXT_IN_PINNED")
    self.assertTrue(os.path.exists(os.path.join(self.ckpt_dir_path, "PINNED")))
    with open(self.pinned_ckpt_file) as f:
      self.assertEqual("TEXT_IN_PINNED", f.read())

  def test_unpin_checkpoint(self):
    # Mark the checkpoint directory as pinned.
    self.checkpoints_dir.create_file("PINNED")
    self.assertTrue(checkpoint_utils.is_pinned_checkpoint(self.ckpt_dir_path))

    # Test.
    checkpoint_utils.unpin_checkpoint(self.ckpt_dir_path)

    # Validate the "PINNED" checkpoint file got removed.
    self.assertFalse(gfile.exists(os.path.join(self.ckpt_dir_path, "PINNED")))

  def test_unpin_checkpoint_does_not_exist(self):
    missing_ckpt_path = os.path.join(self.ckpt_dir_path, "ckpt_does_not_exist")
    self.assertFalse(gfile.exists(missing_ckpt_path))

    # Test. Assert does not raise error.
    try:
      checkpoint_utils.unpin_checkpoint(missing_ckpt_path)
    except IOError:
      # TODO(b/172262005): Remove traceback.format_exc() from the error message.
      self.fail("Unpin checkpoint failed with: %s" % traceback.format_exc())

  def test_remove_checkpoint_dir(self):
    # Ensure the checkpoint directory is setup.
    assert gfile.exists(self.ckpt_dir_path)

    # Test.
    checkpoint_utils.remove_checkpoint_dir(self.ckpt_dir_path)

    # Validate the checkpoint directory got removed.
    self.assertFalse(gfile.exists(self.ckpt_dir_path))

  def test_remove_checkpoint_dir_pinned(self):
    # Mark the checkpoint directory as pinned so it does not get removed.
    self.checkpoints_dir.create_file("PINNED")

    # Test.
    checkpoint_utils.remove_checkpoint_dir(self.ckpt_dir_path)

    # Validate the checkpoint directory still exists.
    self.assertTrue(gfile.exists(self.ckpt_dir_path))

  def test_remove_dataset_checkpoint(self):
    # Ensure the checkpoint directory is setup.
    assert gfile.exists(self.ckpt_dir_path)

    # Test.
    checkpoint_utils.remove_dataset_checkpoint(self.ckpt_dir_path, "train_ds")

    # Validate the checkpoint directory got removed.
    self.assertFalse(gfile.exists(self.train_ds_file))
    self.assertTrue(gfile.exists(self.ckpt_dir_path))

  def test_remove_dataset_checkpoint_pinned(self):
    # Mark the checkpoint directory as pinned so it does not get removed.
    self.checkpoints_dir.create_file("PINNED")

    # Test.
    checkpoint_utils.remove_dataset_checkpoint(self.ckpt_dir_path, "train_ds")

    # Validate the checkpoint directory still exists.
    self.assertTrue(gfile.exists(self.train_ds_file))
    self.assertTrue(gfile.exists(self.ckpt_dir_path))

if __name__ == "__main__":
  absltest.main()
