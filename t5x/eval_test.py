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

"""Tests for t5x.evaluate."""
import functools
import glob
import json
import os
import tempfile

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import gin
import jax
import seqio
import t5.data
import t5.data.tasks  # pylint:disable=unused-import
from t5x import eval as eval_lib
from t5x import model_inference as minfer
from t5x import partitioning
from t5x import test_utils
from t5x import utils


# Work-around for GIN `runs` configs expecting to find
# the `evaluate` at the root scope of the `__main__`.
evaluate = eval_lib.evaluate
FLAGS = flags.FLAGS
mock = absltest.mock

flags.DEFINE_list(
    'gin_search_paths',
    default=[''],
    help='Comma-separated list of gin config path prefixes.')
flags.DEFINE_list(
    'gin_test_configs',
    default=None,
    help=('List of gin test configs to use. Each config results '
          'in a separate test case run.'))


def extend_tests_list(*args):
  jax.config.parse_flags_with_absl()
  if FLAGS.gin_test_configs:
    return tuple(
        list(args) + [(os.path.basename(p), p) for p in FLAGS.gin_test_configs])
  else:
    return tuple(args)


def get_fake_tokenized_dataset_no_pretokenized(*_, split='validation', **__):
  return test_utils.get_fake_tokenized_dataset(split=split).map(
      lambda x: {k: v for k, v in x.items() if not k.endswith('_pretokenized')})


class EvalTest(parameterized.TestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    for p in FLAGS.gin_search_paths + eval_lib._DEFAULT_GIN_SEARCH_PATHS:
      gin.add_config_file_search_path(p)

  @parameterized.named_parameters(*extend_tests_list(
      ('t5_tiny', 'test/eval_t5_tiny.gin',
       test_utils.get_fake_tokenized_dataset),
      ('t5_tiny_no_gin', None, test_utils.get_fake_tokenized_dataset),
      ('no_pretokenized_features', 'test/eval_t5_tiny.gin',
       get_fake_tokenized_dataset_no_pretokenized),
      ))
  @mock.patch(
      't5x.utils.get_vocabulary',
      side_effect=lambda x: (t5.data.get_default_vocabulary(),) * 2)
  @mock.patch.object(seqio.Task, 'get_dataset')
  def test_t5x_local_evaluate(self, cfg_path, get_dataset_fn,
                              get_dataset_mock, unused_get_vocabulary_mock_fn):
    get_dataset_mock.side_effect = get_dataset_fn
    workdir = tempfile.mkdtemp()
    ckpt_path = 't5x/google/testdata/test_t5_tiny.checkpoint_0'
    gin.clear_config(clear_constants=True)
    if cfg_path:
      configured_evaluate = gin.configurable(evaluate)
      gin.parse_config_files_and_bindings([cfg_path], [
          f"utils.RestoreCheckpointConfig.path = '{ckpt_path}'",
          "utils.DatasetConfig.split = 'validation'"
      ])
    else:
      dataset_cfg = utils.DatasetConfig(
          module='t5.data.mixtures',
          mixture_or_task_name='wmt19_ende_v003',
          task_feature_lengths={
              'inputs': 512,
              'targets': 512
          },
          split='validation',
          batch_size=8,
          shuffle=False,
          seed=0)
      ckpt_cfg = utils.RestoreCheckpointConfig(path=ckpt_path, mode='specific')
      partitioner = partitioning.PjitPartitioner(num_partitions=1)
      configured_evaluate = functools.partial(
          evaluate,
          model=test_utils.get_tiny_t5_model(),
          dataset_cfg=dataset_cfg,
          restore_checkpoint_cfg=ckpt_cfg,
          partitioner=partitioner)
    configured_evaluate(output_dir=workdir)

    if cfg_path:
      # Load predictions.
      prediction_file_paths = glob.glob(
          os.path.join(workdir, 'inference_eval', '*[0-9].jsonl'))
      with open(prediction_file_paths[0], 'r') as reader:
        outputs = [json.loads(line) for line in reader.readlines()]

      self.assertLen(outputs, 8)
      for output in outputs:
        self.assertIsInstance(output['prediction'], str)



if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
