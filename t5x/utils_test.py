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
from typing import Optional

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import flax.core
from flax.linen import partitioning as flax_partitioning
import jax
import numpy as np
import seqio
from t5x import checkpoints
from t5x import partitioning
from t5x import test_utils
from t5x import train_state as train_state_lib
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

  @parameterized.named_parameters(
      dict(testcase_name="write_to_file", write_to_log_file=True),
      dict(testcase_name="do_not_write_to_file", write_to_log_file=False),
  )
  def test_log_model_info(self, write_to_log_file):
    log_file = self.create_tempfile() if write_to_log_file else None

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

    with self.assertLogs(level="INFO") as logs:
      utils.log_model_info(log_file and log_file.full_path, mock_train_state,
                           partitioner)

    relevant_logs = [
        re.sub(r"\s+", " ", output)
        for record, output in zip(logs.records, logs.output)
        if "t5x/utils.py" in record.pathname
    ]
    self.assertLen(relevant_logs, 9)
    self.assertIn(
        "Variable a/aa size 6 shape (a1=2, None=3) partition spec ('b1', None)",
        relevant_logs[0])
    self.assertIn(
        "Variable c size 56 shape (None=7, a1=8) partition spec (None, 'b1')",
        relevant_logs[1])

    if write_to_log_file:
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
  def test_get_training_eval_datasets_task(self, mock_get_dataset):
    task = mock.create_autospec(seqio.Task, instance=True)
    task.name = "mock_task"
    task.splits = set(["train", "test"])
    seqio.TaskRegistry.add_provider("mock_task", task)

    mock_get_dataset.return_value = tf.data.Dataset.range(10).batch(1)
    mock_fc_cls = mock.Mock()

    cfg = utils.DatasetConfig(
        mixture_or_task_name="mock_task",
        task_feature_lengths={},
        split="test",
        batch_size=4,
        shuffle=False,
        seed=None)

    # Single shard.
    ds = utils.get_training_eval_datasets(
        cfg,
        shard_id=0,
        num_shards=1,
        eval_steps=3,
        feature_converter_cls=mock_fc_cls)

    mock_get_dataset.assert_called_once_with(
        dataclasses.replace(cfg, batch_size=1),
        shard_id=0,
        num_shards=1,
        feature_converter_cls=mock_fc_cls,
        num_epochs=12,
        continue_from_last_checkpoint=False)

    self.assertSameElements(ds.keys(), ["mock_task"])
    jax.tree_map(np.testing.assert_equal, list(ds["mock_task"]), [
        np.array([0, 1, 2, 3]),
        np.array([4, 5, 6, 7]),
        np.array([8, 9, 0, 1]),
    ])

    # 2 shards, shard 0
    mock_get_dataset.reset_mock()
    ds = utils.get_training_eval_datasets(
        cfg,
        shard_id=0,
        num_shards=2,
        eval_steps=3,
        feature_converter_cls=mock_fc_cls)

    # Call the underlying function loading all shards since the fn shards at the
    # example level.
    mock_get_dataset.assert_called_once_with(
        dataclasses.replace(cfg, batch_size=1),
        shard_id=0,
        num_shards=1,
        feature_converter_cls=mock_fc_cls,
        num_epochs=12,
        continue_from_last_checkpoint=False)

    self.assertSameElements(ds.keys(), ["mock_task"])
    jax.tree_map(np.testing.assert_equal, list(ds["mock_task"]), [
        np.array([0, 2]),
        np.array([4, 6]),
        np.array([8, 0]),
    ])

    # 2 shards, shard 1
    mock_get_dataset.reset_mock()
    ds = utils.get_training_eval_datasets(
        cfg,
        shard_id=1,
        num_shards=2,
        eval_steps=3,
        feature_converter_cls=mock_fc_cls)

    # Call the underlying function loading all shards since the fn shards at the
    # example level.
    mock_get_dataset.assert_called_once_with(
        dataclasses.replace(cfg, batch_size=1),
        shard_id=0,
        num_shards=1,
        feature_converter_cls=mock_fc_cls,
        num_epochs=12,
        continue_from_last_checkpoint=False)

    self.assertSameElements(ds.keys(), ["mock_task"])
    jax.tree_map(np.testing.assert_equal, list(ds["mock_task"]), [
        np.array([1, 3]),
        np.array([5, 7]),
        np.array([9, 1]),
    ])

    # 3 shards
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Batch size (4) must be divisible by number of shards (3)."):
      _ = utils.get_training_eval_datasets(
          cfg,
          shard_id=0,
          num_shards=3,
          eval_steps=3,
          feature_converter_cls=mock_fc_cls)

  @mock.patch.object(utils, "get_dataset")
  def test_get_training_eval_datasets_mixture(self, mock_get_dataset):
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

    mock_get_dataset.return_value = tf.data.Dataset.range(10).batch(1)

    # Verify calls to utils.get_dataset
    cfg = utils.DatasetConfig(
        mixture_or_task_name="mock_mix",
        task_feature_lengths={},
        split="test",
        batch_size=4,
        shuffle=False,
        seed=23)

    res = utils.get_training_eval_datasets(
        cfg,
        shard_id=0,
        num_shards=2,
        eval_steps=3,
        feature_converter_cls=seqio.FeatureConverter)

    expected_calls = [
        mock.call(
            dataclasses.replace(
                cfg, mixture_or_task_name="mock_task1", batch_size=1),
            shard_id=0,
            num_shards=1,
            feature_converter_cls=seqio.FeatureConverter,
            continue_from_last_checkpoint=False,
            num_epochs=12),
        mock.call(
            dataclasses.replace(
                cfg, mixture_or_task_name="mock_task2", batch_size=1),
            shard_id=0,
            num_shards=1,
            feature_converter_cls=seqio.FeatureConverter,
            continue_from_last_checkpoint=False,
            num_epochs=12),
        mock.call(
            dataclasses.replace(
                cfg, mixture_or_task_name="mock_mix", batch_size=1),
            shard_id=0,
            num_shards=1,
            feature_converter_cls=seqio.FeatureConverter,
            continue_from_last_checkpoint=False,
            num_epochs=12)
    ]
    mock_get_dataset.assert_has_calls(expected_calls)

    self.assertSameElements(res.keys(),
                            ["mock_task1", "mock_task2", "mock_mix"])
    for ds in res.values():
      jax.tree_map(np.testing.assert_equal, list(ds), [
          np.array([0, 2]),
          np.array([4, 6]),
          np.array([8, 0]),
      ])

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

    jax.tree_map(
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


@dataclasses.dataclass
class MockTrainState:
  path: Optional[str] = None
  from_scratch: Optional[bool] = None


class MockCheckpointer(checkpoints.Checkpointer):

  def __init__(self, *args, **kwargs):
    pass

  # restore should return TrainState, but we force it to return Mock with path
  # for simplicity.
  def restore(self, path, *args, **kwargs):
    return MockTrainState(path=path, from_scratch=False)


class TrainStateInitializerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    def _partition(train_state, in_axis_resources, out_axis_resources):
      del train_state, in_axis_resources, out_axis_resources
      partitioned_fn = lambda _: MockTrainState(from_scratch=True)
      return partitioned_fn

    partitioner = mock.Mock(get_mesh_axes=lambda _: None, partition=_partition)
    mock_inference_state_create = self.enter_context(
        mock.patch.object(train_state_lib.InferenceState, "create"))
    mock_inference_state_create.return_value = None

    shapes = {
        "ones": (1, 1),
        "twos": (2, 2),
        "threes": (3, 3),
    }
    types = {
        "ones": int,
        "twos": float,
        "threes": int,
    }

    def _init_fn(rng, input_shapes, input_types):
      del rng
      return {
          "ones":
              np.ones(input_shapes["ones"], dtype=input_types["ones"]),
          "twos":
              np.ones(input_shapes["twos"], dtype=input_types["twos"]) * 2,
          "threes":
              np.ones(input_shapes["threes"], dtype=input_types["threes"]) * 3
      }

    init_fn = mock.Mock()
    init_fn.__call__ = _init_fn
    init_fn.__self__ = None

    self.train_state_init = utils.TrainStateInitializer(None, init_fn, shapes,
                                                        partitioner, types)

    self.ckptdir = self.create_tempdir(name="primary_checkpoints")
    steps = (2, 3)
    self.paths = []
    for s in steps:
      step_dir = self.ckptdir.mkdir(f"checkpoint_{s}")
      step_dir.create_file("checkpoint")
      self.paths += [step_dir.full_path]

  def test_from_checkpoints_specific(self):
    # multiple paths
    ckpt_cfg = utils.RestoreCheckpointConfig(
        path=self.paths, mode="specific", checkpointer_cls=MockCheckpointer)
    restored = self.train_state_init.from_checkpoints([ckpt_cfg])
    self.assertSequenceEqual(self.paths, [state.path for state in restored])
    with self.assertRaisesRegex(ValueError, r"^Expected at most 1 checkpoint"):
      self.train_state_init.from_checkpoint([ckpt_cfg])

  def test_from_checkpoints_latest(self):
    # only restore single latest
    ckpt_cfg = utils.RestoreCheckpointConfig(
        path=self.ckptdir.full_path,
        mode="latest",
        checkpointer_cls=MockCheckpointer)
    restored = list(self.train_state_init.from_checkpoints([ckpt_cfg]))
    assert len(restored) == 1
    self.assertEqual(self.paths[-1], restored[0].path)
    restored = self.train_state_init.from_checkpoint([ckpt_cfg])
    self.assertEqual(self.paths[-1], restored.path)

  def test_from_checkpoints_multiple_configs(self):
    # uses first checkpoint with files present.
    ckpt_cfg = utils.RestoreCheckpointConfig(
        path=self.ckptdir.full_path,
        mode="latest",
        checkpointer_cls=MockCheckpointer)
    secondary_ckptdir = self.create_tempdir(name="secondary_checkpoints")
    for s in (4, 5):
      step_dir = secondary_ckptdir.mkdir(f"checkpoint_{s}")
      step_dir.create_file("checkpoint")
    secondary_ckpt_cfg = utils.RestoreCheckpointConfig(
        path=secondary_ckptdir.full_path,
        mode="latest",
        checkpointer_cls=MockCheckpointer)
    restored = self.train_state_init.from_checkpoint(
        [ckpt_cfg, secondary_ckpt_cfg])
    self.assertEqual(self.paths[-1], restored.path)

  def test_from_checkpoints_multiple_configs_one_empty(self):
    # skips empty_checkpoints directory with no checkpoints present.
    ckpt_cfg = utils.RestoreCheckpointConfig(
        path=self.ckptdir.full_path,
        mode="latest",
        checkpointer_cls=MockCheckpointer)
    empty_ckptdir = self.create_tempdir(name="empty_checkpoints")
    empty_ckpt_cfg = utils.RestoreCheckpointConfig(
        path=empty_ckptdir.full_path,
        mode="latest",
        checkpointer_cls=MockCheckpointer)
    restored = self.train_state_init.from_checkpoint([empty_ckpt_cfg, ckpt_cfg])
    self.assertEqual(self.paths[-1], restored.path)

  def test_from_scratch(self):
    self.assertTrue(
        self.train_state_init.from_scratch(jax.random.PRNGKey(13)).from_scratch)

  def test_from_checkpoint_or_scratch(self):
    ckpt_cfg = utils.RestoreCheckpointConfig(
        path=self.ckptdir.full_path,
        mode="latest",
        checkpointer_cls=MockCheckpointer)
    empty_ckptdir = self.create_tempdir(name="empty_checkpoints")
    empty_ckpt_cfg = utils.RestoreCheckpointConfig(
        path=empty_ckptdir.full_path,
        mode="latest",
        checkpointer_cls=MockCheckpointer)

    init_rng = jax.random.PRNGKey(13)

    # ckpt_cfg has checkpoints, restore from there
    restored = self.train_state_init.from_checkpoint_or_scratch(
        [empty_ckpt_cfg, ckpt_cfg], init_rng=init_rng)
    self.assertEqual(self.paths[-1], restored.path)
    self.assertFalse(restored.from_scratch)

    # no checkpoints available, init from scratch
    initialized = self.train_state_init.from_checkpoint_or_scratch(
        [empty_ckpt_cfg], init_rng=init_rng)
    self.assertTrue(initialized.from_scratch)


if __name__ == "__main__":
  absltest.main()
