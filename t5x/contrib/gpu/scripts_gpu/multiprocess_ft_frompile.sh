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

export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_all_reduce_combine_threshold_bytes=8589934592 ${XLA_FLAGS}"

#! Change these values !#
FT_TASK=${FT_TASK:=mnli2}       # currently supported: mnli2, squad1
PREC=${PREC:="bfloat16"}
T5_SIZE=${T5_SIZE:="large"}
BSIZE_PER_GPU=${BSIZE_PER_GPU:=32}
ENC_SL=${ENC_SL:=512}
DEC_SL=${DEC_SL:=128}
NUM_MICROBATCHES=${NUM_MICROBATCHES:=1}
ENABLE_FP8=${ENABLE_FP8:=1}
[[ $ENABLE_FP8 -eq 1 ]] && PREC='bfloat16' # Required for t5x te fp8 to work
TP_SIZE=${TP_SIZE:=1}
TRANSPOSE_BS=${TRANSPOSE_BS:=1}
FUSE_QKV=${FUSE_QKV:=1}
PACK=${PACK:=0}
GPUS_PER_NODE=${GPUS_PER_NODE:=8}
CHECKPOINT_DISABLE=${CHECKPOINT_DISABLE:=0}

MODEL_DIR=${MODEL_DIR:=model_dir}
MODEL_DIR=${T5X_WORKSPACE_DIR}/${MODEL_DIR}

# If true, this will duplicate the last checkpoint in MODEL_DIR and add a date/time string. It will finetune on this directory. Useful if running many experiments on the same pretrained checkpoint.
MAKE_FT_DIR=${9:-false} # 'true' or 'false'. 

: ${WITH_NSYS:=0}

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

case $UNINSTALL_TE in
  0)
    ;;
  *)
    pip uninstall -y transformer_engine
    ;;
esac

# Global batch size
BSIZE=$(( GPUS_PER_NODE * BSIZE_PER_GPU * SLURM_JOB_NUM_NODES / TP_SIZE))
export GPU_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE - 1)) )
export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Set nsys command if WITH_NSYS=1
test "${WITH_NSYS}" == 1 && export NSYS_CMD="nsys profile -t nvtx,cuda,cublas -o ${CONFIG} -f true -c cudaProfilerApi"

# Set multiprocesses command if WITH_MP=1
test "${WITH_MP}" == 1 && export MP_ARGS="--multiprocess_gpu --coordinator_address=${SLURM_LAUNCH_NODE_IPADDR}:12345 --process_count=${SLURM_NTASKS} --process_index=${SLURM_PROCID}"

# Disable checkpointing if desired
test "${CHECKPOINT_DISABLE}" == 1 && export CHECK_DISABLE_ARGS="--gin.utils.CheckpointConfig.save=None"

$NSYS_CMD \
python3 -u ${T5X_DIR}/t5x/train.py \
  --gin_file="t5x/contrib/gpu/t5/t5_1_1/examples/${T5_SIZE}_${FT_TASK}_finetune_adam.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.network.T5Config.dtype=\"${PREC}\" \
  --gin.TASK_FEATURE_LENGTHS="{\"inputs\": ${ENC_SL}, \"targets\": ${DEC_SL}}" \
  --gin.partitioning.PjitPartitioner.num_partitions=${TP_SIZE} \
  --gin.trainer.Trainer.num_microbatches=${NUM_MICROBATCHES} \
  --gin.train/utils.DatasetConfig.batch_size=${BSIZE} \
  --tfds_data_dir=${TFDS_DATA_DIR} \
  --gin.train.stats_period=1000 \
  --gin.train.eval_period=1000 \
  --gin.train.gc_period=2000 \
  --gin.train.te_config_cls=@te_helper.TransformerEngineConfig \
  --gin.te_helper.TransformerEngineConfig.enabled=${ENABLE_FP8} \
  --gin.te_helper.TransformerEngineConfig.fp8_format=\"hybrid\" \
  --gin.network.T5Config.transpose_batch_sequence=${TRANSPOSE_BS} \
  --gin.network.T5Config.fuse_qkv_params=${FUSE_QKV} \
  --gin.train_eval/utils.DatasetConfig.batch_size=${BSIZE} \
  --gin.infer_eval/utils.DatasetConfig.batch_size=${BSIZE} \
  --gin.train/utils.DatasetConfig.pack=${PACK} \
  --gin.train_eval/utils.DatasetConfig.pack=${PACK} \
  ${CHECK_DISABLE_ARGS} \
  ${MP_ARGS}
set +x
