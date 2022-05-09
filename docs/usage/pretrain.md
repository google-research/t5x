# Pretraining a model


## Introduction

This page outlines the steps to pretrain a model with T5X on common tasks
defined with [SeqIO](https://github.com/google/seqio/blob/main/README.md).

## Overview

Pretraining a model with T5X consists of the following steps:

1.  Choose the model architecture.
2.  Choose the SeqIO Task/Mixture to for training.
3.  Write a Gin file that configures the model, SeqIO Task/Mixture and other
    details of your pretraining run.
4.  Launch your experiment locally or on XManager.
5.  Monitor your experiment.

These steps are explained in detail in the following sections. An example run
that trains a T5 1.1 Small checkpoint from scratch on the C4 dataset using the
span corruption pretraining objective is also showcased.

## Step 1: Choose a model architecture

To train a model, you need a Gin config file that defines the model params. For
your convenience, Gin configs for common models have been made available for use
in T5X. Following is a list of these models and their Gin locations.

Model                                 | Gin File Location
------------------------------------- | -----------------
T5 Small                              | [t5_1_0/small.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/t5_1_0/small.gin)
T5 Base                               | [t5_1_0/base.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/t5_1_0/base.gin)
T5 Large                              | [t5_1_0/large.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/t5_1_0/large.gin)
T5 3B                                 | [t5_1_0/3B.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/t5_1_0/3B.gin)
T5 11B                                | [t5_1_0/11B.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/t5_1_0/11B.gin)
T5 1.1 Small                          | [t5_1_1/small.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/t5_1_1/small.gin)
T5 1.1 Base                           | [t5_1_1/base.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/t5_1_1/base.gin)
T5 1.1 Large                          | [t5_1_1/large.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/t5_1_1/large.gin)
T5 1.1 XL                             | [t5_1_1/xl.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/t5_1_1/xl.gin)
T5 1.1 XXL                            | [t5_1_1/xxl.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/t5_1_1/xxl.gin)
MT5 Small                             | [mt5/small.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/mt5/small.gin)
MT5 Base                              | [mt5/base.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/mt5/base.gin)
MT5 Large                             | [mt5/large.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/mt5/large.gin)
MT5 XL                                | [mt5/xl.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/mt5/xl.gin)
MT5 XXL                               | [mt5/xxl.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/mt5/xxl.gin)

For the example run, you will use the T5 1.1 Small model. The Gin file for this
model is located at
[`/t5x/examples/t5/t5_1_1/1_1_small.gin`](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/t5_1_1/small.gin).

## Step 2: Choose a SeqIO Task/Mixture

A SeqIO Task encapsulates the data source, the preprocessing logic to be
performed on the data before querying the model, the postprocessing logic to be
performed on model outputs, and the metrics to be computed given the
postprocessed outputs and targets. A SeqIO Mixture denotes a collection of Tasks
and enables pretraining a model on multiple Tasks simultaneously.

Many common datasets and benchmarks, e.g. [GLUE](https://gluebenchmark.com/),
[SuperGLUE](https://super.gluebenchmark.com/),
[WMT](https://www.tensorflow.org/datasets/catalog/wmt_t2t_translate),
[SQUAD](https://rajpurkar.github.io/SQuAD-explorer/),
[CNN/Daily Mail](https://github.com/abisee/cnn-dailymail), etc. have been
implemented as SeqIO Tasks/Mixtures and can be used directly. These
Tasks/Mixtures are defined in
[`third_party/py/t5/data/tasks.py`](https://github.com/google-research/text-to-text-transfer-transformer/tree/main/t5/data/tasks.py)
and
[`third_party/py/t5/data/mixtures.py`](https://github.com/google-research/text-to-text-transfer-transformer/tree/main/t5/data/mixtures.py).

For the example run, you will train the model on
[`c4_v220_span_corruption`](https://github.com/google-research/text-to-text-transfer-transformer/tree/main/t5/data/tasks.py?l=42&rcl=370153959)
Task that implements the span corruption pretraining objective using the C4
dataset. This is the final pretraining Task used in the
[T5 paper](https://arxiv.org/pdf/1910.10683.pdf%C3%82%C2%A0).

TIP: Want to use a custom Task or Mixture? See section below called "Adding
SeqIO Task/Mixture modules and Gin files"

## Step 3: Write a Gin Config

After choosing the model architecture and SeqIO Task/Mixture for your run, the
next step is to configure your run using Gin. If you're not familiar with Gin,
reading the [T5X Gin Primer](gin.md) is recommended.

T5X provides a Gin file that configures the T5X trainer for pretraining (located
at
[`runs/pretrain.gin`](https://github.com/google-research/t5x/tree/main/t5x/configs/runs/pretrain.gin)),
and expects a few params from you. These params can be specified in a separate
Gin file, or via commandline flags. Following are the required params:

+   `TRAIN_STEPS`: Number of training steps. For the example run, set this to
    `100_000`.
+   `MIXTURE_OR_TASK_NAME`: This is the SeqIO Task or Mixture name to run (from
    Step 2). For the example run, set this to `'c4_v220_span_corruption'`.
+   `TASK_FEATURE_LENGTHS`: This is a dict mapping feature key to maximum int
    length for that feature. After preprocessing, features are truncated to the
    provided value. For the example run, set this to `{"inputs": 512, "targets":
    114}`, following the original T5 pretraining setup.
+   `MODEL_DIR`: A path to write pretrained checkpoints to. When launching using
    XManager, this path is automatically set and can be accessed from the
    XManager Artifacts page. When running locally using Blaze, you can
    explicitly pass a directory using a flag. Launch commands are provided in
    the next step.

In addition to the above params, you will need to import
[`pretrain.gin`](https://github.com/google-research/t5x/tree/main/t5x/configs/runs/pretrain.gin)
and the Gin file for the pretrained model, which for the example run is
[`t5_1_1/small.gin`](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/t5_1_1/small.gin).

```gin
include 't5x/configs/runs/pretrain.gin'
include 't5x/examples/t5/t5_1_1/small.gin'
```

Note that the `include` statements can use relative paths in this example for
which You will pass an appropriate `gin_search_paths` flag to locate these files
when launching your run. However, we recommend that you use absolute paths
because it can be more difficult to locate the gin files speicified via relative
paths without inspecting the launch command.

You will also need to import the Python module(s) that register SeqIO Tasks and
Mixtures used in your run. For the example run, we add `import t5.data.mixtures`
since it is where 'glue_v002_proportional' is registered. Note that this module
must also be included as a dependency in the T5X trainer
[binary](https://github.com/google-research/t5x/tree/main/t5x/BUILD;l=74;rcl=398627055). Most
common Task/Mixture modules, such as this one, are already included. If your
module is not included, see the [Advanced Topics section](#custom-t5x-binaries)
at the end of this tutorial for instructions to add it.

Finally, your Gin file should look like this:

```gin
include 't5x/examples/t5/t5_1_1/small.gin'
include 't5x/configs/runs/pretrain.gin'

# Register necessary SeqIO Tasks/Mixtures.
import t5.data.mixtures

MIXTURE_OR_TASK_NAME = "c4_v220_span_corruption"
TASK_FEATURE_LENGTHS = {"inputs": 512, "targets": 114}
TRAIN_STEPS = 10000
DROPOUT_RATE = 0.0
BATCH_SIZE = 256
```

See
[`t5x/examples/t5/t5_1_1/examples/small_c4_pretrain.gin`](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/t5_1_1/examples/small_c4_pretrain.gin)
for this example.


## Step 4: Launch your experiment

To launch your experiment locally (for debugging only; larger checkpoints may
cause issues), run the following on commandline:

```sh
MODEL_DIR="/tmp/pretrain-model/"
python -m t5x.train \
  --gin_file=t5x/examples/t5/t5_1_1/c4_pretrain_small.gin \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --alsologtostderr
```

Note that multiple comma-separated paths can be passed to the `gin_search_paths`
flag, and these paths should contain all Gin files used or included in your
experiment.


## Next Steps

Now that you have successfully pretrained a model, here are some topics you
might want to explore next:

+   [Fine-tuning a model.](finetune)
+   [Evaluating a fine-tuned model.](eval)
+   [Running inference on a fine-tuned model.](infer)

We also touch upon a few advanced topics related to pretraining below that might
be useful, especially when customizing your pretraining job.

## Advanced Topics

### `train`, `train_eval` {#train-eval .no-toc}

A
[`DatasetConfig`](https://github.com/google-research/t5x/tree/main/t5x/utils.py?l=113&rcl=375475889)
object is used to configure loading SeqIO Tasks/Mixtures for training and eval.
If you take a closer look at
[`runs/pretrain.gin`](https://github.com/google-research/t5x/tree/main/t5x/configs/runs/pretrain.gin),
you will see that there are two `DatasetConfig` objects defined and passed to
the train function: `train_dataset_cfg` and `train_eval_dataset_cfg`. Here's a
brief description of these configs:

+   `train`: This configures the Task/Mixture that the model will be pretrained
    on.
+   `train_eval`: This configures the Task/Mixture that is used to compute
    training metrics on the eval split, e.g. perplexity. These metrics are
    defined in the
    [`Model`](https://github.com/google-research/t5x/tree/main/t5x/models.py;l=257-266;rcl=394045248)
    class and the eval fn is located
    [here](https://github.com/google-research/t5x/tree/main/t5x/trainer.py?l=212&rcl=371778063).

### Deterministic training {.no-toc}

A training run may consist of various randomized operations, e.g. dataset
shuffling, dropout, etc. However, it is often useful to have deterministic
training, meaning that the random operations are reproducible and robust to
preemption/restarts. To make your pretraining deterministic, in addition to
the params configured in `pretrain.gin`, you need to add the following configs:

+   sets the dataset seed to a fixed value: `train/utils.DatasetConfig.seed =
    42`.
+   sets the dropout seed to a fixed value: `train_script.train.random_seed =
    42`.
+   enables dataset checkpointing: `utils.SaveCheckpointConfig.save_dataset =
    True`. This means that the dataset iterator is checkpointed periodically
    during training, and in case of preemptions, training resumes from the
    latest dataset checkpoint to ensure deterministic behavior. The
    checkpointing frequency is set using `utils.SaveCheckpointConfig.period`
    (`1000` by default), meaning that the dataset is checkpointed after
    processing `1000` batches (batches, not examples; batch size can be
    overridden using `train/DatasetConfig.batch_size` and is set to `128` by
    default).


### Defining a custom SeqIO Task/Mixture to pretrain on {.no-toc}

Refer to [SeqIO documentation](https://github.com/google/seqio/blob/main/README.md).
