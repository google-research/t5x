from __gin__ import dynamic_registration

import __main__ as eval_script
from t5.data import mixtures
from t5x import partitioning
from t5x import utils

include "t5x/examples/t5/t5_1_1/base.gin"  # defines %MODEL.

CHECKPOINT_PATH = %gin.REQUIRED  # passed via commandline
EVAL_OUTPUT_DIR = %gin.REQUIRED  # passed via commandline

DROPOUT_RATE = 0.0  # unused boilerplate
MIXTURE_OR_TASK_NAME = "wmt_t2t_ende_v003"

eval_script.evaluate:
  model = %MODEL  # imported from separate gin file
  dataset_cfg = @utils.DatasetConfig()
  restore_checkpoint_cfg = @utils.RestoreCheckpointConfig()
  output_dir = %EVAL_OUTPUT_DIR

utils.DatasetConfig:
  mixture_or_task_name = %MIXTURE_OR_TASK_NAME
  task_feature_lengths = None  # Auto-computes the max feature lengths.
  split = 'test'
  batch_size = 32
  shuffle = False
  seed = 42

partitioning.PjitPartitioner.num_partitions = 2

utils.RestoreCheckpointConfig:
  path = %CHECKPOINT_PATH
  mode = 'specific'
