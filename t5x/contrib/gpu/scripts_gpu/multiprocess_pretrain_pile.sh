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

set -eoux pipefail

# uncomment the next line to enable benchmarking. This activates the STEP_CT setting below and will delete the MODEL_DIR on every run
#BENCHMARK_MODE=True
STAT_PERIOD=100 #only used if BENCHMARK_MODE is set

export BASE_XLA_FLAGS="${BASE_XLA_FLAGS:---xla_gpu_enable_triton_gemm=false --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_all_reduce_combine_threshold_bytes=8589934592}"
export XLA_FLAGS="${BASE_XLA_FLAGS} ${XLA_FLAGS:-}"

#! Change these values !#
TRAIN_STEPS=${TRAIN_STEPS:=1000000}
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
CHECKPOINT_DISABLE=${CHECKPOINT_DISABLE:=0}
ADDITIONAL_CLI_ARGS=${ADDITIONAL_CLI_ARGS:-}

MODEL_DIR_LOCAL=${MODEL_DIR:=model_dir}
MODEL_DIR=${T5X_WORKSPACE_DIR}/${MODEL_DIR_LOCAL}

WITH_MP=${WITH_MP:-1}
# If using slurm:
#   - NUM_PROCESSES, GPUS_PER_NODE are inferred from slurm output env variables
# If calling this outside of slurm:
#   - Set NUM_PROCESSES, GPUS_PER_NODE
#   - Set ADDITIONAL_CLI_ARGS="--coordinator_address=<ip-add-with-port> --process_count=${NUM_PROCESSES} --process_index=<proc-id>"
NUM_PROCESSES=${NUM_PROCESSES:-${SLURM_STEP_NUM_TASKS}}
if [[ -z "$SLURM_STEP_TASKS_PER_NODE" ]]; then
  GPUS_PER_NODE=${GPUS_PER_NODE:-8}
else
  # SLURM_STEP_TASKS_PER_NODE can look like "2(x2)" or "2(x2),1" for homogeneous and heterogeneous setups respectively. This script will only support homogeneous requests
  if [[ "$SLURM_STEP_TASKS_PER_NODE" == *,* ]]; then
      echo "SLURM_STEP_TASKS_PER_NODE=$SLURM_STEP_TASKS_PER_NODE but this script does not support heterogeneous GPU allocations"
      exit 1
  fi
  # This egrep infers the number of homoegeneous tasks per node:
  # If SLURM_STEP_TASKS_PER_NODE=8(x2) then GPUS_PER_NODE=8
  GPUS_PER_NODE=$(echo "$SLURM_STEP_TASKS_PER_NODE" | egrep -o '^[^\(]+')
fi

# nsys setting
WITH_NSYS=${WITH_NSYS:-0}

# Global batch size
BSIZE=$(( BSIZE_PER_GPU * NUM_PROCESSES / TP_SIZE))
DP_SIZE=$((NUM_PROCESSES / TP_SIZE))

GPU_DEVICES=$(seq -s ',' 0 $((GPUS_PER_NODE - 1)))
export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

if [[ "${WITH_NSYS}" -eq 1 ]]; then
  CONFIG=PILE_MODEL-${T5_SIZE}_GBS-${BSIZE}_DP-${DP_SIZE}_TP-${TP_SIZE}_FP8-${ENABLE_FP8}_FUSEQKV-${FUSE_QKV}_TRANSBS-${TRANSPOSE_BS}_ID-${HOSTID}
  NSYS_CMD="nsys profile -t nvtx,cuda,cublas -o ${CONFIG} -f true -c cudaProfilerApi"
fi

if [[ "${WITH_MP}" -eq 1 ]]; then
    ADDITIONAL_CLI_ARGS+=" --multiprocess_gpu"
fi

if [[ "${CHECKPOINT_DISABLE}" -eq 1 ]]; then
    ADDITIONAL_CLI_ARGS+=" --gin.utils.CheckpointConfig.save=None"
fi

echo "MODEL DIR: ${MODEL_DIR}"

case ${BENCHMARK_MODE:-} in
  True)
    rm -rf "${MODEL_DIR}/*"
    ADDITIONAL_CLI_ARGS+=" --gin.train.stats_period=${STAT_PERIOD}"
    echo BENCHMARKING
  ;;
  
  *)
    echo TRAINING
  ;;
 esac

${NSYS_CMD:-} \
python3 -u -m t5x.train \
  --gin_file="${T5X_DIR}/t5x/contrib/gpu/t5/t5_1_1/examples/${T5_SIZE}_pile_pretrain.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.network.T5Config.dtype=\"${PREC}\" \
  --gin.TASK_FEATURE_LENGTHS="{\"inputs\": ${ENC_SL}, \"targets\": ${DEC_SL}}" \
  --gin.partitioning.PjitPartitioner.num_partitions=${TP_SIZE} \
  --gin.trainer.Trainer.num_microbatches=${NUM_MICROBATCHES} \
  --gin.BATCH_SIZE=${BSIZE} \
  --gin.train/utils.DatasetConfig.batch_size=${BSIZE} \
  --gin.train/utils.DatasetConfig.pack=${PACK} \
  --gin.train_eval/utils.DatasetConfig.pack=${PACK} \
  --tfds_data_dir=${TFDS_DATA_DIR} \
  --gin.train.stats_period=1000 \
  --gin.train.eval_period=1000 \
  --gin.train.gc_period=${TRAIN_STEPS} \
  --gin.train.te_config_cls=@te_helper.TransformerEngineConfig \
  --gin.te_helper.TransformerEngineConfig.enable_fp8=${ENABLE_FP8} \
  --gin.te_helper.TransformerEngineConfig.fp8_format=\"hybrid\" \
  --gin.network.T5Config.transpose_batch_sequence=${TRANSPOSE_BS} \
  --gin.network.T5Config.fuse_qkv_params=${FUSE_QKV} \
  --gin.TRAIN_STEPS=${TRAIN_STEPS} \
  ${ADDITIONAL_CLI_ARGS}

echo Finished
