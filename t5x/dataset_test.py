# Copyright 2022 The T5X Authors.
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

"""Tests for dataset."""

from absl.testing import absltest
import jax
import seqio
from seqio import test_utils as seqio_test_utils
from t5x import dataset
import tensorflow as tf


# Replace this with seqio.PassThroughFeatureConverter when it is available.
class TestFeatureConverter(seqio.FeatureConverter):

  def __init__(self, **unused_kwargs):  # pylint: disable=super-init-not-called
    pass

  def __call__(self, ds, task_feature_lengths):
    del task_feature_lengths
    return ds

  def _convert_features(self, ds, task_feature_lengths):
    pass

  def get_model_feature_lengths(self, task_feature_lengths):
    pass


class DatasetTest(seqio_test_utils.FakeTaskTest):

  def test_seqio_dataset_manager(self):

    def get_hardcoded_data(
        split,
        shuffle_files=False,
        seed=None,
        shard_info=None,
        ndfeatures=False,
    ):
      del split, shuffle_files, seed, shard_info, ndfeatures
      output = [
          {
              "prefix": "this",
              "suffix": "hello",
          },
          {
              "prefix": "that",
              "suffix": "world",
          },
      ]
      return tf.data.Dataset.from_generator(
          lambda: output,
          output_types={
              "prefix": tf.string,
              "suffix": tf.string
          },
          output_shapes={
              "prefix": [],
              "suffix": []
          },
      )

    function_source = seqio.FunctionDataSource(
        dataset_fn=get_hardcoded_data,
        splits=["train", "validation"],
    )
    self.add_task(
        "t5x_task",
        source=function_source,
    )
    manager = dataset.SeqIoDatasetManager(
        task_feature_lengths={
            "inputs": 2,
            "targets": 2,
        },
        split="train",
        shuffle=False,
        data_layout=dataset.DataLayout(batch_size=2, shard_id=0, num_shards=1),
        seed=1,
        feature_converter_cls=TestFeatureConverter,
        num_epochs=1,
    )
    actual_tensors = list(manager.get_iterator("t5x_task"))
    # Numpy arrays are harder to assert.
    actual = jax.tree_map(lambda a: a.numpy().tolist(), actual_tensors)

    self.assertLen(actual, 1)
    # Tokens in `inputs` and `targets` are too specific to seqio testutils for
    # assertion.
    self.assertContainsSubset(
        {
            # Seqio testutils's default preprocessor adds "complete:" to prefix.
            "inputs_pretokenized": [b"complete: this", b"complete: that"],
            "targets_pretokenized": [b"hello", b"world"],
        },
        actual[0],
    )


if __name__ == "__main__":
  absltest.main()
