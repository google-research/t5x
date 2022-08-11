# Auxiliary Job


## Introduction

This page outlines the steps needed to use the auxiliary job capabilities
available in T5X.

## Overview

There are a variety of situations in which running a single job is insufficient
or suboptimal. For example, consider the following scenarios:

+   You want to keep track of evaluation (`infer_eval` or `train_eval`) metrics
    per checkpoint, but evaluation takes a very long time due to having a large
    eval dataset, slow decoding, or multiple tasks to evaluate.

+   You want to finetune every checkpoint on a downstream task as you train.

+   You have customized evaluation code that you want to run on every checkpoint
    as you train, but that does not naturally fit within a `seqio.Evaluator`
    framework.

In cases like these, users can make use of the auxiliary job functionality. At a
high-level, the auxiliary job will launch a new job every time a new checkpoint
is saved. This new job can either re-use the `train.py` binary (e.g. for
continuous finetuning) or a different one. For example, this allows users to
perform continuous evaluation (using `eval.py`) without slowing down the
training job. We will provide detailed examples showing how to use the auxiliary
job for these use-cases.

When this new job is launched, the controller will replace four gin macros:
`MODEL_DIR`, `MIXTURE_OR_TASK_NAME`,`INITIAL_CHECKPOINT_PATH`, `TRAIN_STEPS`.
The second of these is set by the user-controlled flag (more on this below), and
the third one is equal to the last checkpoint seen. Aside from this, users are
free to modify the configuration as needed. Beyond gin macros, the auxiliary job
can also have different resource requirements, priority, and even cell placement
from the train job.

## Example 1: Separate evaluation job.

### Step 1: Choose a model architecture.

Similar to pretraining, we will need some gin configuration. For this example,
we will use the T5-1.1-Base model.

### Step 2: Choose a SeqIO Task/Mixture for training and evaluation.

In this example, we will use the classic task of English-French translation from
WMT14, which is conveniently available as a SeqIO task in the tasks file from
the T5 tasks under the name `'wmt_enfr14_v003'`.

### Step 3: Write a Gin config.

Unlike pretraining or finetuning, we will need two gin files for this setup: one
for the training job, and one for the auxiliary job. The train gin file will
have the same requirements as the gin file for pretraining or finetuning. The
auxiliary job gin file can leverage these gin files or be its own independent
gin file, depending on the user’s choice. For this example, we will make a new
gin which is mostly a wrapper around `pretrain.gin` with some additional
hardcoded features. We will use this gin file for the train job and `eval.gin`
for the auxiliary job.

### Step 4: Launch your experiment.

Our sample script will be quite similar to the one used in pretraining and
finetuning, but with a few additional flags which we describe below.

+   `auxiliary_job_mixtures`: This is a comma-separated list of mixtures. A
    separate auxiliary job will be run for each mixture and will replace the gin
    macro `MIXTURE_OR_TASK_NAME`. Note that you need this flag even if you are
    using a custom binary, which does not need a mixture since otherwise no
    auxiliary job will run.

+   `auxiliary_job_gin_file`: This is identical to `gin_file`, except it is used
    for the auxiliary job instead of the train job.

+   `replace_gin_file`: If True, this auxiliary launcher will not use any of the
    gin files from train job. This is necessary when using a binary different
    from `train.py`, since the top-level functions will not match.

+   `auxiliary_job_cell`: The cell in which to run your job. Note that this can
    be different from the training cell.

+   `auxiliary_job_platform`: The platform to use for the auxiliary. Note that
    this can be different from the one use for the train job, allowing users to
    use smaller configurations for evaluation than needed for training.

+   `auxiliary_job_build_target`: The binary to use for auxiliary job.

+   `final_auxiliary_job_steps`: This flag controls how many additional steps to
    take when using the auxiliary job for finetuning. Setting to 0 enables
    continuous evaluation.

We provide the sample script below.

```sh
declare -a ARGS=(
--cell=iz
--platform=jd=2x2
--final_auxiliary_job_steps=0
--replace_gin_file=True
--auxiliary_job_mixtures=wmt14_enfr_v003
--auxiliary_job_gin_file=t5x/examples/t5/t5_1_1/examples/base_wmt14enfr_eval.gin
--auxiliary_job_cell=iz
--auxiliary_job_platform=jd=2x2
--auxiliary_job_build_target_path=//t5x:eval
--gin_file=t5x/examples/t5/t5_1_1/examples/base_wmt14enfr_train.gin
)

gxm t5x/google/xm_launch.py "${ARGS[@]}"
```

## Example 2: Continuous finetuning job.

In this example, we will be pretraining a model on a span corruption task on the
C4 dataset, and finetuning it on the WMT'14 English-French translation task. As
before, we will launch a new auxiliary job once every checkpoint is saved.
However, instead of using the `eval.py` binary, we will use the `train.py`
binary.

### Step 1: Choose a model architecture.

We will use the T5-1.1-Base model as in the previous example.

### Step 2: Choose a SeqIO Task/Mixture for training and evaluation.

For pretraining, we re-use the span coprruption task `c4_v220_span_corruption`
available in the T5 mixtures `tasks.py` file.

### Step 3: Write a Gin config.

As before, we need our gin files to contain all the desired macros in them. We
thus create two new gin files: `base_c4_pretrain.gin` for the train job and
`base_wmtenfr14_finetune.gin` for the auxiliary job.

### Step 4: Launch your experiment.

Our script is quite similar to the first example, with the same flags as before
but with the appropiate changes. The main distinction is that we must change the
flag `final_auxiliary_job_steps` to be non-zero to start finetuning. We will
settle for a modest 200 steps for the sake of demonstration (and evaluate every
100 steps), but users should use larger steps in realistic scenarios. We also
use `train.py` binary instead of `eval.py`.

We provide the sample script below.

```sh
declare -a ARGS=(
--cell=iz
--platform=jd=2x2
--final_auxiliary_job_steps=200
--replace_gin_file=True
--auxiliary_job_mixtures=wmt14_enfr_v003
--auxiliary_job_gin_file=t5x/examples/t5/t5_1_1/examples/base_wmt14enfr_finetune.gin
--auxiliary_job_cell=iz
--auxiliary_job_platform=jd=2x2
--auxiliary_job_build_target_path=//t5x:train
--gin_file=t5x/examples/t5/t5_1_1/examples/base_c4_pretrain.gin
)

gxm t5x/google/xm_launch.py "${ARGS[@]}"
```

## Common Gotchas.

We outline a few common error patterns that we have encountered.

+   **Not passing a value for the `auxiliary_mixtures` flag.** Even if you have
    the desired task in your gin file, or you use a differently named macro, you
    should still pass a value for this flag, since launch script will launch a
    new job per value of this flag.

+   **Not setting `replace_gin_file=True` when using a different binary from
    train.py.** This will usually yield an error that there is no `train`
    function.

+   **No metrics being logged.** It can be tempting to use gin files usually
    used for evaluation. However, one must ensure that the corresponding SeqIO
    evaluators still log to the tensorboard, otherwise you won’t see the
    metrics.

+   **Slow `train_eval`.** While the approach outlined above separates out the
    infer_eval job, it may be that even train_eval is too slow. In these
    situations, we suggest adding the metrics from train_eval into the
    `metrics_fn` argument of the SeqIO task and have them be computed in the
    auxiliary job as well.

+   **Using `CHECKPOINT_PATH` rather `INITIAL_CHECKPOINT_PATH`.** For legacy
    reasons, the auxiliary job uses the macro `INITIAL_CHECKPOINT_PATH` rather
    than `CHECKPOINT_PATH` as found in `eval.gin`. Make sure to use the latter
    macro building your gin scripts.

+   **Gin macros being ignored when passed through the format
    `gin.{MACRO}={VAL}`.** In the current setup, you must include all gin macros
    in the gin script. Attempting to pass them as additional flags will usually
    not work.

+   **Not setting `final_auxiliary_job_steps=0` when performing continuous
    evaluation.** The current parameter controller uses this as a check. When
    this is true, it will replace the `EVAL_OUTPUT_DIR` folder with the current
    `MODEL_DIR`, so that the evaluation metrics are saved in the right place and
    the metrics are showed correctly on the tensorboard.
