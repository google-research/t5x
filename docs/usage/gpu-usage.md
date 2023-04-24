# GPU Scripts and Usage

# Warning!
An updated version of T5x with optimized GPU performance (18-80% perf gains!) and new features, including FP8 with [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) and H100 support can be found here: [NVIDIA Rosetta](https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/t5x). 
-----
**NVIDIA no longer recommends using this repository and won't be updating it further.**
-----

The [t5x/contrib/gpu/scripts_gpu](../../t5x/contrib/gpu/scripts_gpu) directory contains scripts optimized for GPU usage and includes FP8 support via [Transformer Engine](https://github.com/NVIDIA/TransformerEngine).

Install with `pip install -r pile_requirements.txt` to get all pile dependencies.

## Building the container
We provide a fully built and ready-to-use container here: [ghcr.io/nvidia/t5x:te-fp8-reference](ghcr.io/nvidia/t5x:te-fp8-reference)  
If you'd like you build your own,
The Dockerfile in `t5x/contrib/gpu` will build a container with all gpu/pile dependencies. It can be built with `t5x/contrib/gpu/docker/build.sh <name>` 

## Running interactively
Note: this should only be done with singlenode jobs and/or for downloading the pile. Use `t5x/contrib/gpu/docker/interactive_pull_and_launch.sh`. This takes arguments for the URL to pull a container from and the location of the dataset directory to mount. For example:

`t5x/contrib/gpu/docker/interactive_pull_and_launch.sh [URL] /my/dataset/dir`

## Downloading The Pile
We use The Pile for our pretraining experiments. If you would like to as well, run `download_the_pile.py` to download it. The download is approximately 1TB. It will download to the directory set in the environment variable: `TFDS_DATA_DIR`. After that, set the `TFDS_DATA_DIR` to the same directory in your scripts to use.

## Single Node runs
Pretraining and Finetuning can be done with `singlenode_*.sh`. These will build a T5X model with the Adam optimizer and relevant parameters. These will allow multi-gpu on one host.

## Multi Node runs
For a SLURM+pyxis cluster, `example*.sub` files provide example slurm submit files (edit with your details), which call `multiprocess*.sh` to execute training. You can add a binding script in the `.sub` file for your cluster, or remove it entirely (dropping some throughput)

## Convergence and performance
For our Pile convergence runs, we used a Global batch size of 2304 for XXL and 2016-2048 for all other models, where GBS is defined as #GPUs * BS/GPU / Tensor Parallel(TP). Below are example (tested) hardware topologies on NVIDIA DGX A100 (8x A100-SXM4-80G) and H100-SXM-80G nodes.

| size                                    | GPU              | Precision | #GPUs |  TP   | BS / GPU | Sequences/Sec | Seq/Sec/GPU | Est. Walltime | GPU-days | MNLI 2.0 - matched | SQuAD v1.1 (EM/F1) | Convergence Log                                                                              | Config | 
| ----                                    | ------------     | --------- | ----- | ----- | -------- | ------------- | ----------- | ------------- | -------- |------------------ | ------------------ | ---------------                                                                              | ----   |
| [T5-v1.1-small](../t5/t5_1_1/small.gin) | A100 80G SXM     | bf16      | 8     | 1     | 256      | ~5712         | 714         | 4.2 days      | 33       | 83.06%             | 78.33 / 86.63      | [log](https://tensorboard.dev/experiment/lWnHal7PRnOLeZuewyWVxQ/#scalars&_smoothingWeight=0) | [pile](../t5/t5_1_1/examples/small_pile_pretrain.gin)
| [T5-v1.1-large](../t5/t5_1_1/large.gin) | A100 80G SXM     | bf16      | 64    | 1     | 32       | ~4853         | 75.8        | 4.8 days      | 309     | 90.50%             | 87.31 / 94.04      | [log](https://tensorboard.dev/experiment/aOxJBIvTQBeTJ8XGXxaL6Q/#scalars&_smoothingWeight=0) |[pile](../t5/t5_1_1/examples/large_pile_pretrain.gin)
| [T5-v1.1-xl](../t5/t5_1_1/xl.gin)       | A100 80G SXM     | bf16      | 144   | 1     | 8        | ~3021         | 21.0        | 7.9 days      | 1,133   | N/A(perf test)     | N/A (perf test)  |                |[pile](../t5/t5_1_1/examples/xl_pile_pretrain.gin)
| [T5-v1.1-xl](../t5/t5_1_1/xl.gin)       | A100 80G SXM     | bf16      | 256   | 1     | 8        | ~4322         | 16.9        | 5.5 days      | 1,408   | 91.15%             | 89.36 / 95.29      | [log](https://tensorboard.dev/experiment/vuRoEYgkRgWiEtbvgxlOqw/#scalars&_smoothingWeight=0) |[pile](../t5/t5_1_1/examples/xl_pile_pretrain.gin)
| [T5-v1.1-xxl](../t5/t5_1_1/xxl.gin)     | A100 80G SXM     | bf16      | 512   | 8     | 36       | ~1887         | 3.69        | 12.6 days     | 6,431  |N/A(partial run)   | N/A(partial run)   |                  |[pile](../t5/t5_1_1/examples/xxl_pile_pretrain.gin)
| [T5-v1.1-large](../t5/t5_1_1/large.gin) | **H100 80G SXM** | TE-fp8    | 64    | 1     | 32       | ~10156        | **158.7**   | **2.3 days**  | **147** | 89.1%              | 86.36 / 93.5       |                 |[pile](../t5/t5_1_1/examples/large_pile_pretrain.gin)
| [T5-v1.1-xl](../t5/t5_1_1/xl.gin)       | **H100 80G SXM** | TE-fp8    | 144   | 1     | 14       | ~7257         | **50.4**    | **3.3 days**  | **475** | N/A (perf test)    | N/A (perf test)    |                 |[pile](../t5/t5_1_1/examples/xl_pile_pretrain.gin)
| [T5-v1.1-xl](../t5/t5_1_1/xl.gin)       | **H100 80G SXM** | TE-fp8    | 256   | 1     | 8        | ~9688         | **37.8**    | **2.4 days**  | **614** | N/A (perf test)    | N/A (perf test)    |                 |[pile](../t5/t5_1_1/examples/xl_pile_pretrain.gin)

Note: Convergence (as shown in log) was not necessarily done with the hardware topology listed, but the listed topology is tested. Estimated Walltime is calculated assuming full throughput (seq/sec) continuously. In practice, there are compilation overheads at the beginning of each run/restart(in cluster settings) + checkpointing overheads (if any).

Other hyperparameters are specified in the associated pile `gin` files in the `contrib/gpu/t5/t5_1_1/examples` directory.

## Pretraining run commands

### Multinode
Arguments are set by environment variable as such:

`PREC={PRECISION} T5_SIZE={SIZE} BSIZE_PER_GPU={BSIZE} ..... sbatch -N {NODE_CT} t5x/contrib/gpu/t5/scripts_gpu/example_slurm_pretrain_pile.sub {GPUS_PER_NODE}`

All parameters can be found in the relevant script.

### Example Pretraining Commands
Assumes 8GPU 80GB A100/H100 Nodes. `ENABLE_FP8` uses transformer engine (included in container) and requires H100

* Note: To use, FP8 set `ENABLE_FP8` to `1`. This will automatically set `PREC` to `bfloat16` as is required by internals for `FP8` usage.
#### [T5-v1.1-small](../t5/t5_1_1/small.gin) (60M):
```sh
PREC=bfloat16 T5_SIZE=small BSIZE_PER_GPU=256 TRAIN_STEPS=1000000 NUM_MICROBATCHES=1 ENABLE_FP8=1 TP_SIZE=1 \
sbatch -N1 t5x/contrib/gpu/t5/scripts_gpu/example_slurm_pretrain_pile.sub
```

#### [T5-v1.1-large](../t5/t5_1_1/large.gin) (770M):
```sh
PREC=bfloat16 T5_SIZE=large BSIZE_PER_GPU=32 TRAIN_STEPS=1000000 NUM_MICROBATCHES=1 ENABLE_FP8=1 TP_SIZE=1 \
sbatch -N8 t5x/contrib/gpu/t5/scripts_gpu/example_slurm_pretrain_pile.sub
```

#### [T5-v1.1-xl](../t5/t5_1_1/xl.gin) (3B):
```sh
PREC=bfloat16 T5_SIZE=large BSIZE_PER_GPU=8 TRAIN_STEPS=1000000 NUM_MICROBATCHES=1 ENABLE_FP8=1 TP_SIZE=1 \
sbatch -N 32 t5x/contrib/gpu/t5/scripts_gpu/example_slurm_pretrain_pile.sub
```

### Example Finetuning Commands
Finetuning commands simply change the script and have an additional `{FT_TASK}` as the first argument (along with relevant hyperparameter changes). Your `MODEL_DIR` should contain the pretrained checkpoint to restore from.

#### MNLI v2:
```sh
FT_TASK=mnli2 PREC=bfloat16 T5_SIZE={SIZE} BSIZE_PER_GPU={BSIZE} NUM_MICROBATCHES=1 ENABLE_FP8=1 TP_SIZE=1 \
sbatch -N{NODE_CT} t5x/contrib/gpu/t5/scripts_gpu/example_slurm_ft_frompile.sub
```

#### SQuAD v1.1:
```sh
FT_TASK=squad1 PREC=bfloat16 T5_SIZE={SIZE} BSIZE_PER_GPU={BSIZE} NUM_MICROBATCHES=1 ENABLE_FP8=1 TP_SIZE=1 \
sbatch -N{NODE_CT} t5x/contrib/gpu/t5/scripts_gpu/example_slurm_ft_frompile.sub

```

## Performance Settings:
There are 3 major performance settings: `ENABLE_FP8`, `FUSE_QKV` and `TRANSPOSE_BS` (all of which are controllable via env var in the commands above).
We recommend always enabling `TRANSPOSE_BS` (default), but only using `FUSE_QKV` when using `ENABLE_FP8` for optimal performance.

On all finetuning runs, we use a Global Batch Size of 256 with bfloat16 precision + FP8.

WARNING: Finetuning is configured by default to save every checkpoint and delete none (to avoid accidentally deleting your pretrained checkpoint). Watch your disk space! This behavior can be changed in `t5x/configs/runs/finetune_{TASK}.gin`, however this puts the pretrained checkpoint at risk unless backed up.

### Singlenode (single process)
small:

```sh
t5x/contrib/gpu/scripts_gpu/singlenode_pretrain_pile.sh \
  small \
  bfloat16 \
  8 \
  256 \
  {LOGDIR - create before running} \
  {MODEL_DIR} \
  {GRADIENT_ACCUMULATION (1 by default)} \
  {ENABLE_FP8 (1 by default)} \
  {TRANSPOSE_BS (1 by default)} \
  {FUSE_QKV (1 by default)} \
  {PACK (0 by default)}
```

WARNING: Finetuning is configured by default to save every checkpoint and delete none (to avoid accidentally deleting your pretrained checkpoint). Watch your disk space! This behavior can be changed in `t5x/configs/runs/finetune_{TASK}.gin`, however this puts the pretrained checkpoint at risk unless backed up.

Finetuning:
MNLI v2:
```sh
t5x/contrib/gpu/scripts_gpu/singlenode_ft_frompile.sh \
  mnli2 \
  small \
  bfloat16 \
  8 \
  256 \
  {LOGDIR - create before running} \
  {MODEL_DIR(to restore pretrained checkpoint from)} \
  {GRADIENT_ACCUMULATION (1 by default)} \
  {MAKE_FT_DIR (false by default)}
  {ENABLE_FP8 (1 by default)} \
  {TRANSPOSE_BS (1 by default)} \
  {FUSE_QKV (1 by default)} \
  {PACK (0 by default)}
```

WARNING: Finetuning is configured by default to save every checkpoint and delete none (to avoid accidentally deleting your pretrained checkpoint). Watch your disk space! This behavior can be changed in `t5x/configs/runs/finetune_{TASK}.gin`, however this puts the pretrained checkpoint at risk unless backed up.
# Changelog
- Added Transformer Engine + FP8 support
- Added the Transposed Batch-Sequence GPU optimization
- A100 Perf gains! (BF16)
  - 80% speedup - T5-small
  - 23% speedup - T5-large
  - 18% speedup - T5-xl
  - 40% speedup - T5-xxl 
- H100 FP8 support, with gains over A100
  - 2.08x faster - T5-large (FP8)
  - 2.24x faster - T5-xl (FP8)
