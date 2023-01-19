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

# uncomment the next line to enable benchmarking. This activates the STEP_CT setting below and will delete the MODEL_DIR on every run
#BENCHMARK_MODE=True
STEP_CT=500 # only used if BENCHMARK_MODE is set
STAT_PERIOD=100 #only used if BENCHMARK_MODE is set

# Arguments
T5_SIZE=$1       # Model size (small, base, large, xl, xxl)
PREC="$2"        # Precision (float32, float16, bfloat16)
GPUS_PER_NODE=$3      # Number of GPUs (1, 2, 4, 8)
BSIZE_PER_GPU=$4 # Batch size per GPU (varies with model size)
MODEL_DIR_LOCAL=${5:-"model_dir_${OPTIMIZER}"}
MODEL_DIR=${PWD}/${MODEL_DIR_LOCAL}
NUM_MICROBATCHES=${6:-0}
MP=${7:-1} 

echo Model Parallel partitions: ${MP}

# Setting XLA flags
export XLA_FLAGS="--xla_gpu_simplify_all_fp_conversions --xla_gpu_all_reduce_combine_threshold_bytes=136314880 ${XLA_FLAGS}"

echo $MODEL_DIR

# Global batch size
GLOBAL_BATCH_SIZE=$(( GPUS_PER_NODE * BSIZE_PER_GPU * SLURM_JOB_NUM_NODES / MP))

case $BENCHMARK_MODE in
  True)
    rm -rf "${MODEL_DIR}/*"
    echo BENCHMARKING
    python3 -u ${T5X_DIR}/t5x/train.py \
      --gin_file="t5x/contrib/gpu/t5/t5_1_1/examples/${T5_SIZE}_pile_pretrain.gin" \
      --gin.MODEL_DIR=\"${MODEL_DIR}\" \
      --gin.TRAIN_STEPS=${STEP_CT}\
      --gin.network.T5Config.dtype=\"${PREC}\" \
      --gin.train.stats_period=${STAT_PERIOD} \
      --tfds_data_dir=${TFDS_DATA_DIR} \
      --gin.trainer.Trainer.num_microbatches=${NUM_MICROBATCHES} \
      --gin.train/utils.DatasetConfig.batch_size=${GLOBAL_BATCH_SIZE} \
      --gin.train_eval/utils.DatasetConfig.batch_size=${GLOBAL_BATCH_SIZE} \
      --gin.infer_eval/utils.DatasetConfig.batch_size=${GLOBAL_BATCH_SIZE} \
      --gin.partitioning.PjitPartitioner.num_partitions=${MP} \
      --multiprocess_gpu \
      --coordinator_address=${SLURM_LAUNCH_NODE_IPADDR}:12345 \
      --process_count=${SLURM_NTASKS} \
      --process_index=${SLURM_PROCID}
    ;;

  *)
    echo TRAINING
    python3 -u ${T5X_DIR}/t5x/train.py \
    --gin_file="t5x/contrib/gpu/t5/t5_1_1/examples/${T5_SIZE}_pile_pretrain.gin" \
    --gin.MODEL_DIR=\"${MODEL_DIR}\" \
    --gin.network.T5Config.dtype=\"${PREC}\" \
    --tfds_data_dir=${TFDS_DATA_DIR} \
    --gin.trainer.Trainer.num_microbatches=${NUM_MICROBATCHES} \
    --gin.train/utils.DatasetConfig.batch_size=${GLOBAL_BATCH_SIZE} \
    --gin.train_eval/utils.DatasetConfig.batch_size=${GLOBAL_BATCH_SIZE} \
    --gin.infer_eval/utils.DatasetConfig.batch_size=${GLOBAL_BATCH_SIZE} \
    --gin.partitioning.PjitPartitioner.num_partitions=${MP} \
    --multiprocess_gpu \
    --coordinator_address=${SLURM_LAUNCH_NODE_IPADDR}:12345 \
    --process_count=${SLURM_NTASKS} \
    --process_index=${SLURM_PROCID}
  ;;

esac
set +x
