#! /bin/bash

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

# Setting XLA flags
export XLA_FLAGS="--xla_gpu_simplify_all_fp_conversions --xla_gpu_all_reduce_combine_threshold_bytes=136314880 ${XLA_FLAGS}"

TFDS_DATA_DIR="/t5x_home/datasets/"
T5X_DIR=${PWD}

# Arguments
FT_TASK=$1       # currently supported: mnli2, squad1
T5_SIZE=$2       # Model size (small, base, large)
PREC="$3"        # Precision (float32, float16, bfloat16)
NUM_GPUS=$4      # Number of GPUs (1, 2, 4, 8)
BSIZE_PER_GPU=$5 # Batch size per GPU (varies with model size)
LOG_DIR=$6       # Output log directory
MODEL_DIR_LOCAL=${7:-"model_dir"}
MODEL_DIR=${PWD}/${MODEL_DIR_LOCAL}
NUM_MICROBATCHES=${8:-0}
# If true, this will duplicate the last checkpoint in MODEL_DIR and add a date/time string. It will finetune on this directory. Useful if running many experiments on the same pretrained checkpoint.
# Otherwise, be wary of overriding your pretrained checkpoint with the finetuned one.
MAKE_FT_DIR=${9:-false} # 'true' or 'false'. 
ENABLE_FP8=${10:-1}
[[ $ENABLE_FP8 -eq 1 ]] && PREC='bfloat16' # Required for t5x te fp8 to work
TRANSPOSE_BS=${11:-1}
FUSE_QKV=${12:-1}
PACK=${13:-0}

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
    echo $FT_TASK may not be supported. Try mnli2, or squad1
    ;;
esac

echo "Please make sure ${NUM_GPUS} is the number of visible CUDA devices you have"

# Global batch size
BSIZE=$(( NUM_GPUS * BSIZE_PER_GPU  ))

python3 -u ${T5X_DIR}/t5x/train.py \
  --gin_file="t5x/contrib/gpu/t5/t5_1_1/examples/${T5_SIZE}_${FT_TASK}_finetune_adam.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.network.T5Config.dtype=\"${PREC}\" \
  --tfds_data_dir=${TFDS_DATA_DIR} \
  --gin.train/utils.DatasetConfig.batch_size=${BSIZE} \
  --gin.trainer.Trainer.num_microbatches=${NUM_MICROBATCHES} \
  --gin.train_eval/utils.DatasetConfig.batch_size=${BSIZE} \
  --gin.infer_eval/utils.DatasetConfig.batch_size=${BSIZE} \
  --gin.train/utils.DatasetConfig.pack=${PACK} \
  --gin.train_eval/utils.DatasetConfig.pack=${PACK} \
  --gin.train.te_config_cls=@te_helper.TransformerEngineConfig \
  --gin.te_helper.TransformerEngineConfig.enable_fp8=${ENABLE_FP8} \
  --gin.te_helper.TransformerEngineConfig.fp8_format=\"hybrid\" \
  --gin.network.T5Config.transpose_batch_sequence=${TRANSPOSE_BS} \
  --gin.network.T5Config.fuse_qkv_params=${FUSE_QKV} \
  2>&1 | tee \
  ${LOG_DIR}/ft_${T5_SIZE}_gpu_${NUM_GPUS}_${PREC}_gbs_${BSIZE}_fp8_${ENABLE_FP8}_fuseqkv_${FUSE_QKV}_transbs_${TRANSPOSE_BS}.log
