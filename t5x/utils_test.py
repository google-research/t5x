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

"""Tests for t5x.utils."""

import dataclasses
import os
import re

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized

import flax.core
from flax.linen import partitioning as flax_partitioning
import jax
import numpy as np

import seqio
from t5x import partitioning
from t5x import test_utils
from t5x import utils
import tensorflow as tf

mock = absltest.mock
Evaluator = seqio.Evaluator
PartitionSpec = partitioning.PartitionSpec
AxisMetadata = flax_partitioning.AxisMetadata

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()

FLAGS = flags.FLAGS


def get_mock_train_state(params, param_states=None, step=0):
  """Returns a mock TrainState."""
  step = np.array(step) if step is not None else None
  state = mock.Mock(param_states=param_states, step=step)
  state_dict = dict(
      target=params, state=dict(param_states=param_states, step=step))
  return mock.Mock(
      params=params,
      param_states=param_states,
      step=step,
      state_dict=lambda: state_dict,
      optimizer=mock.Mock(
          target=params, state=state, state_dict=lambda: state_dict),
  )


class UtilsTest(parameterized.TestCase):

  def round_vocab_size_to_multiple(self):
    self.assertEqual(utils.round_vocab_size_to_multiple(1), 128)
    self.assertEqual(utils.round_vocab_size_to_multiple(128), 128)
    self.assertEqual(utils.round_vocab_size_to_multiple(129), 256)
    self.assertEqual(utils.round_vocab_size_to_multiple(129), 256)
    self.assertEqual(
        utils.round_vocab_size_to_multiple(25600, divisor=384), 256128)

  def test_get_zeros_batch_like_spec(self):
    test_utils.assert_same(
        utils.get_zeros_batch_like_spec({
            "i": jax.ShapeDtypeStruct((2, 5), dtype=np.int32),
            "j": jax.ShapeDtypeStruct((1,), dtype=np.float32),
        }), {
            "i": np.zeros((2, 5), dtype=np.int32),
            "j": np.zeros((1,), dtype=np.float32)
        })

  def test_get_zeros_batch_like_dataset(self):
    ds = tf.data.Dataset.from_tensors({
        "i": np.arange(10, dtype=np.int32).reshape((2, 5)),
        "j": np.ones((1,), dtype=np.float32)
    })

    test_utils.assert_same(
        utils.get_zeros_batch_like_dataset(ds), {
            "i": np.zeros((2, 5), dtype=np.int32),
            "j": np.zeros((1,), dtype=np.float32)
        })

    test_utils.assert_same(
        utils.get_zeros_batch_like_dataset(ds, batch_size=4), {
            "i": np.zeros((4, 5), dtype=np.int32),
            "j": np.zeros((4,), dtype=np.float32)
        })

  def test_log_model_info(self):
    log_file = self.create_tempfile()

    mock_train_state = get_mock_train_state(
        params={
            "a": {
                "aa": jax.ShapeDtypeStruct(shape=(2, 3), dtype=np.int32)
            },
            "c": jax.ShapeDtypeStruct(shape=(7, 8), dtype=np.int32)
        },
        param_states={
            "a": {
                "aa": {
                    "v_row": jax.ShapeDtypeStruct(shape=(2,), dtype=np.int32),
                    "v_col": jax.ShapeDtypeStruct(shape=(3,), dtype=np.int32)
                }
            },
            "c": {
                "v_row": jax.ShapeDtypeStruct(shape=(2, 4), dtype=np.int32),
                "v_col": None
            }
        })

    mock_logical_axes = get_mock_train_state(
        params={
            "a": {
                "aa": partitioning.AxisNames("a1", None)
            },
            "c": partitioning.AxisNames(None, "a1")
        },
        param_states={
            "a": {
                "aa": {
                    "v_row": partitioning.AxisNames(None,),
                    "v_col": partitioning.AxisNames(None,)
                }
            },
            "c": {
                "v_row": partitioning.AxisNames("a1",),
                "v_col": partitioning.AxisNames("a2",)
            }
        },
        step=None)

    mock_mesh_axes = get_mock_train_state(
        params={
            "a": {
                "aa": PartitionSpec("b1", None)
            },
            "c": PartitionSpec(None, "b1")
        },
        param_states={
            "a": {
                "aa": {
                    "v_row": partitioning.AxisNames(None,),
                    "v_col": partitioning.AxisNames(None,)
                }
            },
            "c": {
                "v_row": partitioning.AxisNames("b1",),
                "v_col": partitioning.AxisNames("b2",)
            }
        },
        step=None)

    partitioner = mock.Mock(
        get_logical_axes=lambda _: mock_logical_axes,
        get_mesh_axes=lambda _: mock_mesh_axes)

    utils.log_model_info(log_file.full_path, mock_train_state, partitioner)

    self.assertEqual(
        re.sub(r"\s+", " ", log_file.read_text()),
        "Variable a/aa size 6 shape (a1=2, None=3) partition spec ('b1', None) "
        "Variable c size 56 shape (None=7, a1=8) partition spec (None, 'b1') "
        "Total number of parameters: 62 "
        "Variable param_states/a/aa/v_col size 3 shape (None=3) partition spec (None,) "
        "Variable param_states/a/aa/v_row size 2 shape (None=2) partition spec (None,) "
        "Variable param_states/c/v_col None "
        "Variable param_states/c/v_row size 8 shape (2, 4) partition spec ('b1',) "
        "Variable step size 1 shape () partition spec None ")


  @mock.patch.object(utils, "get_dataset")
  def test_get_training_eval_datasets(self, mock_get_dataset):
    # Register a mock SeqIO mixture.
    task1 = mock.create_autospec(seqio.Task, instance=True)
    task1.name = "mock_task1"
    task1.splits = set(["train", "test"])
    task2 = mock.create_autospec(seqio.Task, instance=True)
    task2.name = "mock_task2"
    task2.splits = set(["train", "test"])
    seqio.TaskRegistry.add_provider("mock_task1", task1)
    seqio.TaskRegistry.add_provider("mock_task2", task2)
    mixture = seqio.Mixture(
        "mock_mix", ["mock_task1", "mock_task2"], default_rate=1.0)
    seqio.MixtureRegistry.add_provider("mock_mix", mixture)

    mock_get_dataset.return_value = tf.data.Dataset.from_tensor_slices(
        range(100))

    # Verify calls to utils.get_dataset
    cfg = utils.DatasetConfig(
        mixture_or_task_name="mock_mix",
        task_feature_lengths={},
        split="test",
        batch_size=2,
        shuffle=False,
        seed=23)

    _ = utils.get_training_eval_datasets(
        cfg,
        shard_id=0,
        num_shards=1,
        eval_steps=2,
        feature_converter_cls=seqio.FeatureConverter,
        get_dataset_fn=utils.get_dataset)

    expected_calls = [
        mock.call(
            dataclasses.replace(cfg, mixture_or_task_name="mock_task1"),
            0,
            1,
            seqio.FeatureConverter,
            continue_from_last_checkpoint=False,
            num_epochs=2),
        mock.call(
            dataclasses.replace(cfg, mixture_or_task_name="mock_task2"),
            0,
            1,
            seqio.FeatureConverter,
            continue_from_last_checkpoint=False,
            num_epochs=2),
        mock.call(
            dataclasses.replace(cfg, mixture_or_task_name="mock_mix"),
            0,
            1,
            seqio.FeatureConverter,
            continue_from_last_checkpoint=False,
            num_epochs=2)
    ]
    mock_get_dataset.assert_has_calls(expected_calls)

  def test_override_params_axes_names(self):
    model_variables = flax.core.freeze({
        "params": {
            "logits_dense": np.zeros((2, 4)),
            "mlp": {
                "wo": {
                    "kernel": np.zeros((4, 6)),
                    "bias": np.zeros(6),
                }
            }
        },
        "params_axes": {
            "logits_dense_axes": AxisMetadata(names=("vocab", "embed")),
            "mlp": {
                "wo": {
                    "kernel_axes": AxisMetadata(names=("embed", "mlp"))
                }
            }
        }
    })

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Model variables do not contain a 'params_axes' collection to apply an "
        "override to."):
      utils.override_params_axes_names({"params": model_variables["params"]},
                                       [("mlp/wo/kernel", ("embed",))])

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Provided axis name override for mlp/wo/kernel does not match param "
        "rank (2): ('embed',)"):
      utils.override_params_axes_names(model_variables,
                                       [("mlp/wo/kernel", ("embed",))])

    overridden_variables = utils.override_params_axes_names(
        model_variables,
        [
            ("wo/kernel", ("batch",)),  # unused since not a full match
            (".*/wo/kernel", ("batch", "embed")),  # this one is used
            ("mlp/wo/kernel", ("embed",)),  # unused since already matched
            ("mlp/wo/bias", ("embed",)),  # used
        ])

    jax.tree_multimap(
        np.testing.assert_equal, overridden_variables,
        flax.core.freeze({
            "params": {
                "logits_dense": np.zeros((2, 4)),
                "mlp": {
                    "wo": {
                        "kernel": np.zeros((4, 6)),
                        "bias": np.zeros(6),
                    }
                }
            },
            "params_axes": {
                "logits_dense_axes": AxisMetadata(names=("vocab", "embed")),
                "mlp": {
                    "wo": {
                        "kernel_axes": AxisMetadata(names=("batch", "embed")),
                        "bias_axes": AxisMetadata(names=("embed",)),
                    }
                }
            }
        }))


if __name__ == "__main__":
  absltest.main()
