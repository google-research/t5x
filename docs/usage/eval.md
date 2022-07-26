# Evaluating a Model


## Introduction

This page outlines the steps to evaluate a model with T5X on downstream tasks
defined with [SeqIO](https://github.com/google/seqio/blob/main/README.md).

Refer to this tutorial when you have an existing model that you want to
evaluate. If you would like to fine-tune your model before evaluation, please
refer to the [fine-tuning](finetune) tutorial. You can run evals as part of your
fine-tuning run as well.

## Overview

Evaluating a model with T5X consists of the following steps:

1.  Choose the model to evaluate.
1.  Choose the SeqIO Task/Mixture to evaluate the model on.
1.  Write a Gin file that configures the model, SeqIO Task/Mixture and other
    details of your eval run.
1.  Launch your experiment locally or on XManager.
1.  Monitor your experiment and parse metrics.

These steps are explained in detail in the following sections. An example run
that evaluates a fine-tuned T5-1.1-Small checkpoint on the
[(Open Domain) Natural Questions benchmark](https://ai.google.com/research/NaturalQuestions/)
is also showcased.

## Step 1: Choose a model

To evaluate a model, you need a Gin config file that defines the model params,
and the model checkpoint to load from. For this example, a T5-1.1-Small model
fine-tuned on the
[`natural_questions_open_test`](https://github.com/google-research/google-research/tree/master/t5_closed_book_qa/t5_cbqa/tasks.py?l=141&rcl=370261021)
SeqIO Task will be used:

+   Model checkpoint -
    [`cbqa/small_ssm_nq/model.ckpt-1110000`](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/cbqa/small_ssm_nq/)
+   Model Gin file -
    [`t5x/configs/models/t5_1_1_small.gin`](https://github.com/google-research/t5x/tree/main/t5x/google/examples/flaxformer_t5/configs/models/t5_1_1_small.gin).

If you would like to fine-tune your model before evaluation, please follow the
[fine-tuning](finetune) tutorial, and continue to Step 2.

## Step 2: Choose a SeqIO Task/Mixture

A SeqIO Task encapsulates the data source, the preprocessing logic to be
performed on the data before querying the model, the postprocessing logic to be
performed on model outputs, and the metrics to be computed given the
postprocessed outputs and targets. A SeqIO Mixture denotes a collection of Tasks
and enables fine-tuning a model on multiple Tasks simultaneously.

Many common datasets and benchmarks, e.g. [GLUE](https://gluebenchmark.com/),
[SuperGLUE](https://super.gluebenchmark.com/),
[WMT](https://www.tensorflow.org/datasets/catalog/wmt_t2t_translate),
[SQUAD](https://rajpurkar.github.io/SQuAD-explorer/),
[CNN/Daily Mail](https://github.com/abisee/cnn-dailymail), etc. have been
implemented as SeqIO Tasks/Mixtures and can be used directly. These
Tasks/Mixtures are defined in
[`t5/data/tasks.py`](https://github.com/google-research/text-to-text-transfer-transformer/tree/main/t5/data/tasks.py) and
[`t5/data/mixtures.py`](https://github.com/google-research/text-to-text-transfer-transformer/tree/main/t5/data/mixtures.py).

For the example run, you will evaluate the model on the Natural Questions
benchmark, which has been implemented as the `natural_questions_open` Task in
[`/third_party/google_research/google_research/t5_closed_book_qa/t5_cbqa/tasks.py`](https://github.com/google-research/google-research/tree/master/t5_closed_book_qa/t5_cbqa/tasks.py?l=98&rcl=370261021).
Here's an example of a single row of preprocessed data from this Task:

```python
{
    'inputs_pretokenized': 'nq question: what was the main motive of salt march',
    'inputs': [3, 29, 1824, 822, 10, 125, 47,  8, 711, 10280, 13, 3136, 10556, 1]
    'targets_pretokenized': 'challenge to British authority',
    'targets': [1921, 12, 2390, 5015, 1],
    'answers': ['challenge to British authority']
}
```

## Step 3: Write a Gin Config

After choosing the model and SeqIO Task/Mixture for your run, the next step is
to configure your run using Gin. If you're not familiar with Gin, reading the
[T5X Gin Primer](gin.md) is recommended.

T5X provides a Gin file that configures the T5X eval job (located at
[`t5x/configs/runs/eval.gin`](https://github.com/google-research/t5x/tree/main/t5x/configs/runs/eval.gin)),
and expects a few params from you. These params can be specified in a separate
Gin file, or via commandline flags. Following are the required params:

+   `CHECKPOINT_PATH`: This is the path to the model checkpoint (from Step 1).
    For the example run, set this to
    `'gs://t5-data/pretrained_models/cbqa/small_ssm_nq/model.ckpt-1110000'`.
+   `MIXTURE_OR_TASK_NAME`: This is the SeqIO Task or Mixture name to run eval
    on (from Step 2). For the example run, set this to
    `'natural_questions_open'`.
+   `EVAL_OUTPUT_DIR`: A path to write eval outputs to. When launching using
    XManager, this path is automatically set and can be accessed from the
    XManager Artifacts page. When running locally using Blaze, you can
    explicitly pass a directory using a flag. Launch commands are provided in
    the next step.

In addition to the above params, you will need to import
[`eval.gin`](https://github.com/google-research/t5x/tree/main/t5x/configs/runs/eval.gin) and the
Gin file for the model, which for the example run is
[`t5_1_1_small.gin`](https://github.com/google-research/t5x/tree/main/t5x/google/examples/flaxformer_t5/configs/models/t5_1_1_small.gin).

```gin
include 'runs/eval.gin'
include 'models/t5_small.gin'
```

Note that the `include` statements use relative paths in this example. You will
pass an appropriate `gin_search_paths` flag to locate these files when launching
your run. Absolute paths to Gin files can also be used, e.g.

```gin
include 't5x/configs/runs/eval.gin'
include 't5x/google/examples/flaxformer_t5/configs/models/t5_1_1_small.gin'
```

You will also need to import the Python module(s) that register SeqIO Tasks and
Mixtures used in your run. For the example run, we add `import
google_research.t5_closed_book_qa.t5_cbqa.tasks`
since it is where 'glue_v002_proportional' is registered.

If you choose a module that is not included as a dependency in the T5X trainer
[binary](https://github.com/google-research/t5x/tree/main/t5x/BUILD;l=76;rcl=398627055), or if you
have defined your gin config file in a location other than the
[T5X config directory](https://github.com/google-research/t5x/tree/main/t5x/configs/), you will
need to follow the instructions in the
[Advanced Topics section](#custom-t5x-binaries) to link in the custom gin file
and/or task definition.

Note that for most common Task/Mixtures, such as the `glue_v002_proportional`
used in this tutorial, the necessary modules are already included. It is also
possible to skip writing a Gin file and instead pass the params as flags when
launching the eval job (see instructions in Step 4).

Finally, your Gin file should look like this:

```gin
include 't5x/configs/runs/eval.gin'
include 't5x/google/examples/flaxformer_t5/configs/models/t5_1_1_small.gin'

# Register necessary SeqIO Tasks/Mixtures.
import google_research.t5_closed_book_qa.t5_cbqa.tasks

CHECKPOINT_PATH = 'gs://t5-data/pretrained_models/cbqa/small_ssm_nq/model.ckpt-1110000'
MIXTURE_OR_TASK_NAME = 'natural_questions_open'
```

See
[`t5_1_1_small_cbqa_natural_questions.gin`](https://github.com/google-research/t5x/tree/main/t5x/google/examples/flaxformer_t5/configs/examples/eval/t5_1_1_small_cbqa_natural_questions.gin)
for this example.

In this example, we run the evaluation on one checkpoint. It is common to
evaluate with multiple checkpoints. We provide an easy way to do so *without*
having to recompile the model graph for each checkpoints. This is simply done by
adding `utils.RestoreCheckpointConfig.mode = "all"` to a gin file. Our
`t5x/configs/runs/eval.gin` uses "specific" mode.

## Step 4: Launch your experiment

To launch your experiment locally (for debugging only; larger checkpoints may
cause issues), run the following on commandline:

```sh
EVAL_OUTPUT_DIR="/tmp/model-eval/"
python -m t5x.eval \
  --gin_file=t5x/google/examples/flaxformer_t5/configs/examples/eval/t5_1_1_small_cbqa_natural_questions.gin \
  --gin.EVAL_OUTPUT_DIR=\"${EVAL_OUTPUT_DIR}\" \
  --alsologtostderr
```

Note that relative paths can be used to locate the gin files. For that, multiple
comma-separated paths can be passed to the `gin_search_paths` flag, and these
paths should contain all Gin files used or included in your experiment.


You can have a look inside
[`eval.gin`](https://github.com/google-research/t5x/tree/main/t5x/configs/runs/eval.gin) to see
other useful parameters that it is possible to pass in, including dataset split,
batch size, and random seed.

## Step 5: Monitor your experiment and parse metrics


After evaluation has completed, you can parse metrics into CSV format using the
following script:

```sh
EVAL_OUTPUT_DIR= # from Step 4 if running locally, from XManager Artifacts otherwise
VAL_DIR="$EVAL_OUTPUT_DIR/inference_eval"
python -m t5.scripts.parse_tb \
  --summary_dir="$VAL_DIR" \
  --seqio_summaries \
  --out_file="$VAL_DIR/results.csv" \
  --alsologtostderr
```

## Next Steps

Now that you have successfully evaluated a model on the Natural Questions
benchmark, here are some topics you might want to explore next:

+   [Running inference on a model.](infer)
+   [Fine-tuning a model.](finetune)
+   [Training a model from scratch.](pretrain)

We also touch upon a few advanced topics related to evaluations below that might
be useful, especially when customizing your eval job.

## Advanced Topics


### Defining a custom SeqIO Task/Mixture to evaluate on {.no-toc}

Refer to [SeqIO documentation](https://github.com/google/seqio/blob/main/README.md).

### Defining a custom metric to evaluate

The best way to define a custom metric is to define a new SeqIO Task/Mixture
that contains this custom metric. Please refer to the SeqIO Documentation on
[custom metrics](https://github.com/google/seqio/blob/main/README.md#metrics).
