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

"""Fiddle versions of examples in t5x/examples/t5_1_1/examples/*.gin."""
import fiddle as fdl

import seqio

# Load task and mixture registrations.
# pylint: disable=unused-import
from t5.data import mixtures
from t5.data import tasks
# pylint: disable=unused-import

from t5x import config_utils
from t5x import eval as t5x_eval
from t5x import partitioning
from t5x import utils

from t5x.fiddle_configs.configs import finetune
from t5x.fiddle_configs.configs import pretrain
from t5x.fiddle_configs.models import t5_1_1


def small_wmt_finetune() -> fdl.Buildable:
  config = t5_1_1.small_config(dropout_rate=0.0)
  model = t5_1_1.model(
      config=config,
      loss_normalizing_factor=233472,
  )
  return finetune.train(
      mixture_or_task_name='wmt_t2t_ende_v003',
      model_dir=None,  # To be set via --fdl_set="model_dir=..."
      model=model,
      task_feature_lengths={'inputs': 256, 'targets': 256},
      # 1000000 pre-trained steps + 20000 fine-tuning steps.
      train_steps=1_020_000,
      initial_checkpoint_path=(
          'gs://t5-data/pretrained_models/t5x/'
          't5_1_1_small/checkpoint_1000000'
      ),
      use_cached_tasks=False,
  )


def small_wmt_eval() -> fdl.Buildable:
  config = t5_1_1.small_config(dropout_rate=0.0)
  model = t5_1_1.model(
      config=config,
  )
  return fdl.Config(
      t5x_eval.evaluate,
      model=model,
      partitioner=fdl.Config(
          partitioning.PjitPartitioner,
          num_partitions=1,
      ),
      dataset_cfg=fdl.Config(
          utils.DatasetConfig,
          mixture_or_task_name='wmt_t2t_ende_v003',
          task_feature_lengths=None,  # Auto-computes the max lengths.
          split='test',
          batch_size=32,
          shuffle=False,
          seed=42,
      ),
      inference_evaluator_cls=fdl.Partial(
          seqio.Evaluator,
          logger_cls=[
              fdl.Partial(seqio.PyLoggingLogger),
              fdl.Partial(seqio.TensorBoardLogger),
              fdl.Partial(seqio.JSONLogger),
          ],
          num_examples=None,  # Use all examples in the dataset.
          use_memory_cache=True,
      ),
      summarize_config_fn=config_utils.summarize_fiddle_config,
      restore_checkpoint_cfg=fdl.Config(
          utils.RestoreCheckpointConfig,
          path=None,  # Set via --fdl_set="restore_checkpoint_cfg.path=..."
          mode='specific',
      ),
      output_dir=None,  # Set via --fdl_set="output_dir=..."
  )


def small_c4_pretrain() -> fdl.Buildable:
  config = t5_1_1.small_config(dropout_rate=0.0)
  model = t5_1_1.model(
      config=config,
  )
  return pretrain.train(
      mixture_or_task_name='c4_v220_span_corruption',
      model_dir=None,  # To be set via --fdl_set="model_dir=..."
      model=model,
      task_feature_lengths={'inputs': 512, 'targets': 114},
      train_steps=10000,
      batch_size=256,
  )
