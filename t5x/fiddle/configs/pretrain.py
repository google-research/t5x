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

"""Fiddle-config helpers equivalent to t5x/configs/runs/pretrain.gin."""
from collections.abc import Mapping
import copy
from typing import Optional, Union

import fiddle as fdl
import seqio
from t5x import config_utils
from t5x import models
from t5x import partitioning
from t5x import train as t5x_train  # import __main__ as train_script
from t5x import trainer
from t5x import utils


# Defaults, commonly overridden
USE_CACHED_TASKS = True
BATCH_SIZE = 128

# Defaults, sometimes overridden
EVAL_STEPS = 20
EVAL_PERIOD = 1000

# HW RNG is faster than SW, but has limited determinism.
# Most notably it is not deterministic across different
# submeshes.
USE_HARDWARE_RNG = False
# None always uses faster, hardware RNG
RANDOM_SEED = None

MixtureOrTask = Union[str, seqio.Task, seqio.Mixture]


def train(
    model: fdl.Buildable[models.BaseTransformerModel],
    model_dir: Optional[str],
    train_steps: int,
    mixture_or_task_name: MixtureOrTask,
    task_feature_lengths: Mapping[str, int],
    eval_steps: int = EVAL_STEPS,
    eval_period: int = EVAL_PERIOD,
    random_seed: Optional[int] = RANDOM_SEED,
    mixture_or_task_module: Optional[str] = None,
    use_hardware_rng: bool = USE_HARDWARE_RNG,
    batch_size: int = BATCH_SIZE,
    use_cached_tasks: bool = USE_CACHED_TASKS,
) -> fdl.Buildable:
  """Generate a configuration for running T5X `train()` launcher."""
  return fdl.Config(
      t5x_train.train,
      model=model,
      model_dir=model_dir,
      train_dataset_cfg=train_dataset_config(
          mixture_or_task_name=mixture_or_task_name,
          task_feature_lengths=copy.copy(task_feature_lengths),
          batch_size=batch_size,
          use_cached_tasks=use_cached_tasks,
          mixture_or_task_module=mixture_or_task_module,
      ),
      train_eval_dataset_cfg=train_eval_dataset_config(
          mixture_or_task_name=mixture_or_task_name,
          task_feature_lengths=copy.copy(task_feature_lengths),
          batch_size=batch_size,
          use_cached_tasks=use_cached_tasks,
          mixture_or_task_module=mixture_or_task_module,
      ),
      infer_eval_dataset_cfg=None,
      checkpoint_cfg=checkpoint_config(),
      partitioner=fdl.Config(
          partitioning.PjitPartitioner,
          num_partitions=1,
          model_parallel_submesh=None,
          logical_axis_rules=fdl.Config(
              partitioning.standard_logical_axis_rules
          ),
      ),
      trainer_cls=fdl.Partial(
          trainer.Trainer,
          num_microbatches=None,
          learning_rate_fn=fdl.ArgFactory(
              utils.create_learning_rate_scheduler,
              factors='constant * rsqrt_decay',
              base_learning_rate=1.0,
              # 10k to keep consistent with T5/MTF defaults.
              warmup_steps=10000,
          ),
      ),
      total_steps=train_steps,
      eval_steps=eval_steps,
      eval_period=eval_period,
      random_seed=random_seed,
      use_hardware_rng=use_hardware_rng,
      summarize_config_fn=config_utils.summarize_fiddle_config,
  )


def checkpoint_config() -> fdl.Buildable[utils.CheckpointConfig]:
  return fdl.Config(
      utils.CheckpointConfig,
      restore=fdl.Config(utils.RestoreCheckpointConfig, path=[]),
      save=fdl.Config(
          utils.SaveCheckpointConfig,
          period=1000,
          dtype='float32',
          keep=None,  # keep all checkpoints,
          save_dataset=False,  # don't checkpoint dataset state
      ),
  )


def train_dataset_config(
    mixture_or_task_name: MixtureOrTask,
    task_feature_lengths: Mapping[str, int],
    batch_size: int,
    use_cached_tasks: bool,
    mixture_or_task_module: Optional[str],
) -> fdl.Buildable[utils.DatasetConfig]:
  return fdl.Config(
      utils.DatasetConfig,
      mixture_or_task_name=mixture_or_task_name,
      task_feature_lengths=task_feature_lengths,
      split='train',
      batch_size=batch_size,
      shuffle=True,
      seed=None,  # use a new seed each run/restart
      use_cached=use_cached_tasks,
      pack=True,
      module=mixture_or_task_module,
  )


def train_eval_dataset_config(
    mixture_or_task_name: MixtureOrTask,
    task_feature_lengths: Mapping[str, int],
    batch_size: int,
    use_cached_tasks: bool,
    mixture_or_task_module: Optional[str],
) -> fdl.Buildable[utils.DatasetConfig]:
  return fdl.Config(
      utils.DatasetConfig,
      mixture_or_task_name=mixture_or_task_name,
      task_feature_lengths=task_feature_lengths,
      split='validation',
      batch_size=batch_size,
      shuffle=False,
      seed=42,
      use_cached=use_cached_tasks,
      pack=True,
      module=mixture_or_task_module,
  )
