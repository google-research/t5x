#! /bin/bash
# Assumes you are using a SLURM cluster. Edit flags under --multiprocess_gpu below to suit your setup

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

set -x

TFDS_DATA_DIR="/t5x_home/datasets/"
T5X_DIR=${PWD}

# Arguments
FT_TASK=$1       # currently supported: mnli2, squad1
T5_SIZE=$2       # Model size (small, base, large, xl, xxl)
PREC="$3"        # Precision (float32, float16, bfloat16)
NUM_GPUS=$4      # Number of GPUs (1, 2, 4, 8)
BSIZE_PER_GPU=$5 # Batch size per GPU 
LOG_DIR=$6       # Output log directory
MODEL_DIR_LOCAL=${7:-"model_dir"}
MODEL_DIR=${PWD}/${MODEL_DIR_LOCAL}
NUM_MICROBATCHES=${8:-0}

# If true, this will duplicate the last checkpoint in MODEL_DIR and add a date/time string. It will finetune on this directory. Useful if running many experiments on the same pretrained checkpoint.
MAKE_FT_DIR=${9:-false} # 'true' or 'false'. 

export XLA_FLAGS="--xla_gpu_simplify_all_fp_conversions --xla_gpu_all_reduce_combine_threshold_bytes=136314880 ${XLA_FLAGS}"

case $MAKE_FT_DIR in
  true)
    NEW_DIR=ft_${MODEL_DIR_LOCAL}_$(date +%F_%H-%M-%S)
    OLD_DIR=${MODEL_DIR}
    MODEL_DIR=${PWD}/${NEW_DIR}
    mkdir ${MODEL_DIR}
    cp -r ${OLD_DIR}/checkpoint_* ${MODEL_DIR}
    ;;
  false)
    ;;
  *)
    echo "Warning, MAKE_NEW_DIR set to neither true nor false. Not making new directory for finetuning"
    ;;
esac
    

echo $MODEL_DIR

case $FT_TASK in
  mnli2)
    ;;
  squad1)
    ;;
  *)
    echo $FT_TASK may not be supported. Try mnli2 or squad1
    ;;
esac

# Global batch size
BSIZE=$(( NUM_GPUS * BSIZE_PER_GPU * SLURM_JOB_NUM_NODES ))

python3 -u ${T5X_DIR}/t5x/train.py \
  --gin_file="t5x/contrib/gpu/t5/t5_1_1/examples/${T5_SIZE}_${FT_TASK}_finetune_adam.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.network.T5Config.dtype=\"${PREC}\" \
  --tfds_data_dir=${TFDS_DATA_DIR} \
  --gin.train/utils.DatasetConfig.batch_size=${BSIZE} \
  --gin.trainer.Trainer.num_microbatches=${NUM_MICROBATCHES} \
  --gin.train_eval/utils.DatasetConfig.batch_size=${BSIZE} \
  --gin.infer_eval/utils.DatasetConfig.batch_size=${BSIZE} \
  --multiprocess_gpu \
  --coordinator_address=${SLURM_LAUNCH_NODE_IPADDR}:12345 \
  --process_count=${SLURM_NTASKS} \
  --process_index=${SLURM_PROCID}
