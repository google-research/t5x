# GPU Scripts

# Warning!
An updated version of T5x with optimized GPU performance (18-80% perf gains!) and new features, including FP8 with [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) and H100 support can be found here: [NVIDIA Rosetta](https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/t5x). 
-----
**NVIDIA no longer recommends using this repository and won't be updating it further.**
-----

The [t5x/contrib/gpu](../../t5x/contrib/gpu) directory contains scripts optimized for GPU usage.

Install with `pip install -r pile_requirements.txt` to get all pile dependencies.

## Building the container
The Dockerfile in `t5x/contrib/gpu` given will build a container with all gpu/pile dependencies. It can be built with `t5x/contrib/gpu/docker/build.sh <name>` 

## Running interactively
Note: this should only be done with singlenode jobs and/or for downloading the pile. Use `t5x/contrib/gpu/docker/interactive_pull_and_launch.sh`. This takes arguments for the URL to pull a container from and the location of the dataset directory to mount. For example:

`t5x/contrib/gpu/docker/interactive_pull_and_launch.sh [URL] /my/dataset/dir`

## Downloading The Pile
Run `download_the_pile.py` to download the pile. It will download to the directory set in the environment variable: `TFDS_DATA_DIR`. After that, set the `TFDS_DATA_DIR` to the same directory in your scripts to use.

## Single Node runs
Pretraining and Finetuning can be done with `singlenode_*.sh`. These will build a T5X model with the Adam optimizer and relevant parameters. These will allow multi-gpu on one host.

## Multi Node runs
For a SLURM+pyxis cluster, `example*.sub` files provide example slurm submit files (edit with your details), which call `multiprocess*.sh` to execute training. You can add a binding script in the `.sub` file for your cluster, or remove it entirely (dropping some throughput)

## Convergence
For our Pile convergence runs, we used a Global batch size of 2304 for XXL and 2048 for all other models, where GBS is defined as #GPUs * BS/GPU / Tensor Parallel(TP). Below are example (tested) hardware topologies on NVIDIA DGX A100 (8x A100 80G) nodes.

| size | #GPUs |  TP   | BS / GPU | Sequences/Sec | Estimated Walltime | MNLI 2.0 - matched | SQuAD v1.1 (EM/F1) | Convergence Log | 
| ---- | ----- | ----- | -------- | ------------- | ------------------ | ------------------ | ------------------ | --------------- |
| small| 8     | 1     | 256      | ~3168         | 7.48 days          | 83.06%             | 78.33 / 86.63      | [log](https://tensorboard.dev/experiment/lWnHal7PRnOLeZuewyWVxQ/#scalars&_smoothingWeight=0) |
| large| 64    | 1     | 32       | ~3886         | 6.10 days          | 90.50%             | 87.31 / 94.04      | [log](https://tensorboard.dev/experiment/aOxJBIvTQBeTJ8XGXxaL6Q/#scalars&_smoothingWeight=0) |
| xl   | 256   | 1     | 8        | ~3652         | 6.49 days          | 91.15%             | 89.36 / 95.29      | [log](https://tensorboard.dev/experiment/vuRoEYgkRgWiEtbvgxlOqw/#scalars&_smoothingWeight=0) |
| xxl  | 512   | 8     | 36       | ~1346         | 19.81 days         | N/A(partial run)   | N/A(partial run)   | N/A(partial run)|

Note: Convergence (as shown in log) was not necessarily done with the hardware topology listed, but the listed topology is tested. Estimated Walltime is calculated assuming full throughput (seq/sec) continuously. In practice, there are compilation overheads at the beginning of each run/restart(in cluster settings) + checkpointing overheads (if any).

(More perf improvements coming soon!)

Other hyperparameters are specified in the associated pile `gin` files in the `contrib/gpu/t5/t5_1_1/examples` directory.

## Pretraining run commands

### Singlenode
small:

`t5x/contrib/gpu/t5/scripts_gpu/singlenode_pretrain_pile.sh small bfloat16 8 256 {LOGDIR - create before running} {MODEL_DIR} {GRADIENT_ACCUMULATION (1 by default)}`

Finetuning:
MNLI v2:
`t5x/contrib/gpu/t5/scripts_gpu/singlenode_ft_frompile.sh mnli2 small bfloat16 8 256 {LOGDIR - create before running} {MODEL_DIR(to restore pretrained checkpoint from)} {GRADIENT_ACCUMULATION}`


### Multinode
Arguments are as such:

`sbatch -N {NODE_CT} t5x/contrib/gpu/t5/scripts_gpu/example_slurm_pretrain_pile.sub {MODEL_SIZE} {MODEL_PREC} {GPU/NODE} {BS/GPU} {MODEL_DIR} {GRADIENT_ACCUMULATION} {TENSOR_PARALLEL}`

small:

`sbatch -N 1 t5x/contrib/gpu/t5/scripts_gpu/example_slurm_pretrain_pile.sub small bfloat16 8 256 {MODEL_DIR} 1 1`

large:

`sbatch -N 8 t5x/contrib/gpu/t5/scripts_gpu/example_slurm_pretrain_pile.sub large bfloat16 8 32 {MODEL_DIR} 1 1`

xl:

`sbatch -N 32 t5x/contrib/gpu/t5/scripts_gpu/example_slurm_pretrain_pile.sub xl bfloat16 8 8 {MODEL_DIR} 1 1`

Finetuning commands simply change the script and have an additional `{FT_TASK}` as the first argument (along with relevant hyperparameter changes). Your `MODEL_DIR` should contain the pretrained checkpoint to restore from. 

MNLI v2:

`sbatch -N {NODE_CT} t5x/contrib/gpu/t5/scripts_gpu/example_slurm_ft_frompile.sub mnli2 {MODEL_SIZE} {MODEL_PREC} {GPU/NODE} {BS/GPU} {MODEL_DIR} {GRADIENT_ACCUMULATION} {TENSOR_PARALLEL}`

SQuAD v1.1

`sbatch -N {NODE_CT} t5x/contrib/gpu/t5/scripts_gpu/example_slurm_ft_frompile.sub squad1 {MODEL_SIZE} {MODEL_PREC} {GPU/NODE} {BS/GPU} {MODEL_DIR} {GRADIENT_ACCUMULATION} {TENSOR_PARALLEL}`

On all finetuning runs, we use a Global Batch Size of 128 with bfloat16 precision.

WARNING: Finetuning is configured by default to save every checkpoint and delete none (to avoid accidentally deleting your pretrained checkpoint). Watch your disk space! This behavior can be changed in `t5x/configs/runs/finetune_{TASK}.gin`, however this puts the pretrained checkpoint at risk unless backed up.