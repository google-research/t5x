# Copyright 2021 The T5X Authors.
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

"""Tests for t5x.train."""
import functools
import glob
import os
import tempfile

from absl import flags
from absl.testing import absltest
import gin
import jax
import seqio
import t5.data.tasks  # pylint:disable=unused-import
from t5x import gin_utils
from t5x import partitioning
from t5x import test_utils
from t5x import train as train_lib
from t5x import trainer
from t5x import utils
import tensorflow as tf

# Work-around for GIN `runs` configs expecting to find
# the `train` at the root scope of the `__main__`.
train = train_lib.train
FLAGS = flags.FLAGS

flags.DEFINE_list(
    "gin_search_paths",
    default=[""],
    help="Comma-separated list of gin config path prefixes.")
flags.DEFINE_list(
    "gin_test_configs",
    default=[],
    help=("Comma-separated list of gin test configs to use. Each config "
          "results in a separate test case run."))


def flag_configs():
  return [(os.path.basename(cfg), cfg) for cfg in FLAGS.gin_test_configs]


EXPECTED_T5_METRICS = {
    "train": {
        "accuracy": (0.0, 0.0),
        "loss": (490.0, 1e2),
        "loss_per_nonpadding_target_token": (10.2, 1e-1),
        "loss_per_all_target_tokens": (1.91, 1e-2),
        "z_loss": (0.0, 0.0),
        "cross_ent_loss": (490.0, 1e2),
        "cross_ent_loss_per_all_target_tokens": (1.91, 1e-2),
        "z_loss_per_all_target_tokens": (0.0, 0.0),
        "learning_rate": (0.03, 1e-2)
    },
    "training_eval/wmt19_ende_v003": {
        "accuracy": (0.0, 0.0),
        "loss": (627.0, 1),
        "loss_per_nonpadding_target_token": (9.8, 1e-1),
        "loss_per_all_target_tokens": (2.45, 1e-2),
        "z_loss": (0.0, 0.0),
        "cross_ent_loss": (627.0, 1),
        "cross_ent_loss_per_all_target_tokens": (2.45, 1e-2),
        "z_loss_per_all_target_tokens": (0.0, 0.0)
    }
}

class TrainTest(absltest.TestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    for p in FLAGS.gin_search_paths + train_lib._DEFAULT_GIN_SEARCH_PATHS:
      gin.add_config_file_search_path(p)

  @absltest.mock.patch.object(
      seqio.Task,
      "get_dataset",
      side_effect=test_utils.get_fake_tokenized_dataset)
  def _run_test(self, cfg_path, expected_metrics, unused_fn=None):
    workdir = tempfile.mkdtemp()
    gin.clear_config(clear_constants=True)
    if cfg_path:
      configured_train = gin.configurable(train)
      gin.parse_config_files_and_bindings([cfg_path], "")
    else:
      train_dataset_cfg = utils.DatasetConfig(
          module="t5.data.mixtures",
          mixture_or_task_name="wmt19_ende_v003",
          task_feature_lengths={
              "inputs": 32,
              "targets": 32
          },
          split="train",
          batch_size=8,
          shuffle=False,
          pack=False,
          use_cached=False,
          seed=0)
      eval_dataset_cfg = utils.DatasetConfig(
          module="t5.data.mixtures",
          mixture_or_task_name="wmt19_ende_v003",
          task_feature_lengths={
              "inputs": 32,
              "targets": 32
          },
          split="validation",
          batch_size=8,
          shuffle=False,
          pack=False,
          use_cached=False,
          seed=0)
      ckpt_cfg = utils.CheckpointConfig(
          save=utils.SaveCheckpointConfig(dtype="float32", period=4))
      partitioner = partitioning.ModelBasedPjitPartitioner(num_partitions=2)
      trainer_cls = functools.partial(
          trainer.Trainer,
          learning_rate_fn=utils.create_learning_rate_scheduler(
              factors="constant * rsqrt_decay",
              base_learning_rate=1.0,
              warmup_steps=1000),
          num_microbatches=None)
      configured_train = functools.partial(
          train,
          model=test_utils.get_t5_test_model(),
          train_dataset_cfg=train_dataset_cfg,
          train_eval_dataset_cfg=eval_dataset_cfg,
          infer_eval_dataset_cfg=None,
          checkpoint_cfg=ckpt_cfg,
          partitioner=partitioner,
          trainer_cls=trainer_cls,
          total_steps=3,
          eval_steps=2,
          eval_period=1000,
          random_seed=0,
          summarize_config_fn=gin_utils.summarize_gin_config)

    step, train_state = configured_train(model_dir=workdir)
    content = os.listdir(workdir)
    self.assertEqual(step, 3)
    self.assertIn("checkpoint_3", content)

    for step_type in expected_metrics:
      for e in tf.compat.v1.train.summary_iterator(
          glob.glob(os.path.join(workdir, step_type, "*"))[0]):
        if e.step != 3:
          continue
        for v in e.summary.value:
          if v.tag in expected_metrics[step_type]:
            actual_val = tf.make_ndarray(v.tensor)
            exp_val, exp_delta = expected_metrics[step_type][v.tag]
            tf.compat.v1.logging.info("%s %s %f", step_type, v.tag, actual_val)
            self.assertAlmostEqual(actual_val, exp_val, delta=exp_delta)

    # This should load last checkpoint and do no training steps, returning
    # the optimizer loaded from checkpoint.
    new_step, new_train_state = configured_train(model_dir=workdir)
    self.assertEqual(new_step, step)
    test_utils.assert_same(train_state.state_dict(),
                           new_train_state.state_dict())

  def test_train_t5_gin(self):
    if flag_configs():
      return
    self._run_test("t5x/examples/t5/t5_1_1/examples/test_train_t5_tiny.gin",
                   EXPECTED_T5_METRICS)

  def test_train_t5_nogin(self):
    if flag_configs():
      return
    self._run_test(None, EXPECTED_T5_METRICS)

  def test_train_flag_configs(self):
    for (name, cfg_path) in flag_configs():
      with self.subTest(name=name):
        self._run_test(cfg_path, EXPECTED_T5_METRICS)


if __name__ == "__main__":
  jax.config.parse_flags_with_absl()
  absltest.main()
