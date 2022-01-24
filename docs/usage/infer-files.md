# Running inference on a Model


## Introduction

This page outlines the steps to run inference a model with T5X on files
containing
[TensorFlow Examples](https://www.tensorflow.org/api_docs/python/tf/train/Example).

## Overview

Running inference on a model with T5X using TF Example files consists of the
following steps:

1.  Choose the model to run inference on.
1.  Choose the TF Example files to run inference on.
1.  Write a Gin file that configures the model, file source and other details of
    your inference run.
1.  Launch your experiment locally or on XManager.
1.  Monitor your experiment and access predictions.

These steps are explained in detail in the following sections. An example run
that runs inference on a fine-tuned T5-1.1-Small checkpoint on `tfrecord` files
containing the
[(Open Domain) Natural Questions benchmark](https://ai.google.com/research/NaturalQuestions/)
is also showcased.

## Step 1: Choose a model

To run inference on a model, you need a Gin config file that defines the model
params, and the model checkpoint to load from. For this example, a T5-1.1-Small
model fine-tuned on the
[`natural_questions_open_test`](https://github.com/google-research/google-research/tree/master/t5_closed_book_qa/t5_cbqa/tasks.py?l=141&rcl=370261021)
SeqIO Task will be used:

+   Model checkpoint -
    [`cbqa/small_ssm_nq/model.ckpt-1110000`](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/cbqa/small_ssm_nq/)
+   Model Gin file -
    [`models/t5_1_1_small.gin`](https://github.com/google-research/t5x/tree/main/t5x/google/examples/flaxformer_t5/configs/models/t5_1_1_small.gin).

If you would like to fine-tune your model before inference, please follow the
[fine-tuning](finetune) tutorial, and continue to Step 2.

## Step 2: Choose a TF Example file source

T5X supports running inference on `tfrecord`, `recordio` and `sstable` files
containing TF Examples. For the example run, you will run inference on
`tfrecord` files containing the `'natural_questions_open'` dataset located here:
`/path/to/tfds/data/dir/natural_questions_open/1.0.0/natural_questions_open-validation.tfrecord*`.
Here's an example of a single row of data from this file (you can explore this
file further using [GQUI](http://shortn/_oNuDhg7jwN)):

```json
{ # (tensorflow.Example) size=101B
  features: { # (tensorflow.Features) size=99B
    feature: { # (tensorflow.Features.FeatureEntry) size=27B
      key: "answer" # size=6
      value: { # (tensorflow.Feature) size=17B
        bytes_list: { # (tensorflow.BytesList) size=15B
          value: [ "Jason Flemyng" ] # size=13
        } # features.feature[0].value.bytes_list
      } # features.feature[0].value
    } # features.feature[0]
    feature: { # (tensorflow.Features.FeatureEntry) size=68B
      key: "question" # size=8
      value: { # (tensorflow.Feature) size=56B
        bytes_list: { # (tensorflow.BytesList) size=54B
          value: [ "who played hyde in league of extraordinary gentlemen" ] # size=52
        } # features.feature[1].value.bytes_list
      } # features.feature[1].value
    } # features.feature[1]
  } # features
}
```

## Step 3: Write a Gin Config

After choosing the model and file source for your run, the next step is to
configure your run using Gin. If you're not familiar with Gin, reading the
[T5X Gin Primer](gin.md) is recommended. T5X provides a Gin file that configures
the T5X inference job (located at
[`t5x/configs/runs/infer_from_tfexample_file.gin`](https://github.com/google-research/t5x/tree/main/t5x/configs/runs/infer_from_tfexample_file.gin))
to run inference on TF Example files, and expects a few params from you. These
params can be specified in a separate Gin file, or via commandline flags.
Following are the required params:

+   `CHECKPOINT_PATH`: This is the path to the model checkpoint (from Step 1).
    For the example run, set this to
    `'gs://t5-data/pretrained_models/cbqa/small_ssm_nq/model.ckpt-1110000'`.
+   `TF_EXAMPLE_FILE_PATHS`: This is a list of paths or glob patterns to read TF
    Examples from. For the example run, set this to
    `['/path/to/tfds/data/dir/natural_questions_open/1.0.0/natural_questions_open-validation.tfrecord*']`.
+   `TF_EXAMPLE_FILE_TYPE`: This is the TF Example file format. Currently
    supported file formats are `tfrecord`, `recordio` and `sstable`. For the
    example run, set this to `'tfrecord'`.
+   `FEATURE_LENGTHS`: This is a dict mapping feature key to maximum int length
    for that feature. the TF Example features are truncated to the provided
    value. For the example run, set this to `{'inputs': 38, 'targets': 18}`,
    which is the maximum token length for the test set.
+   `INFER_OUTPUT_DIR`: A path to write inference outputs to. When launching
    using XManager, this path is automatically set and can be accessed from the
    XManager Artifacts page. When running locally using Blaze, you can
    explicitly pass a directory using a flag. Launch commands are provided in
    the next step.

In addition to the above params, you may also need to override the
`create_task_from_tfexample_file.inputs_key` param based on the data format (it
is set to `'inputs'` by default. For the example run, the `'question'` key
contains the input (see Step 2), so add the following to your Gin config:

```gin
create_task_from_tfexample_file.inputs_key = 'question'
```

Additionally, you will need to import the
[`infer_from_tfexample_file.gin`](https://github.com/google-research/t5x/tree/main/t5x/configs/runs/infer_from_tfexample_file.gin)
and the Gin file for the model, which for the example run is
[`t5_1_1_small.gin`](https://github.com/google-research/t5x/tree/main/t5x/google/examples/flaxformer_t5/configs/models/t5_1_1_small.gin).

```gin
include 'runs/infer_from_tfexample_file.gin'
include 'models/t5_1_1_small.gin'
```

Note that the `include` statements use relative paths in this example. You will
pass an appropriate `gin_search_paths` flag to locate these files when launching
your run. Absolute paths to Gin files can also be used, e.g.

```gin
include 't5x/configs/runs/infer_from_tfexample_file.gin'
include 't5x/google/examples/flaxformer_t5/configs/models/t5_1_1_small.gin'
```

Finally, your Gin file should look like this:

```gin
include 'runs/infer_from_tfexample_file.gin'
include 'models/t5_1_1_small.gin'

CHECKPOINT_PATH = 'gs://t5-data/pretrained_models/cbqa/small_ssm_nq/model.ckpt-1110000'
TF_EXAMPLE_FILE_PATHS = ['/path/to/tfds/data/dir/natural_questions_open/1.0.0/natural_questions_open-validation.tfrecord*']
TF_EXAMPLE_FILE_TYPE = 'tfrecord'
FEATURE_LENGTHS = {'inputs': 38, 'targets': 18}
create_task_from_tfexample_file.inputs_key = 'question'
```

See
[`t5x/configs/examples/inference/t5_1_1_small_cbqa_natural_questions_tfexample.gin`](https://github.com/google-research/t5x/tree/main/t5x/google/examples/flaxformer_t5/configs/examples/inference/t5_1_1_small_cbqa_natural_questions_tfexample.gin)
for this example. Make sure that your Gin file is linked as a data dependency to
the T5X inference
[binary](https://github.com/google-research/t5x/tree/main/t5x/BUILD;l=74;rcl=398627055). If your
Gin file is not included, see the
[Advanced Topics section](#custom-t5x-binaries) at the end of this tutorial for
instructions to add it, or skip writing a Gin file and pass the above params as
flags when launching the inference job (see instructions in Step 4).

## Step 4: Launch your experiment

To launch your experiment locally (for debugging only; larger checkpoints may
cause issues), run the following on commandline:

```sh
INFER_OUTPUT_DIR="/tmp/model-infer/"
python -m t5x.infer \
  --gin_file=t5x/google/examples/flaxformer_t5/configs/examples/inference/t5_1_1_small_cbqa_natural_questions_tfexample.gin \
  --gin.INFER_OUTPUT_DIR=\"${INFER_OUTPUT_DIR}\" \
  --alsologtostderr
```

Note that multiple comma-separated paths can be passed to the `gin_search_paths`
flag, and these paths should contain all Gin files used or included in your
experiment.


After inference has completed, you can view predictions in the `jsonl` files in
the output dir. JSON data is written in chunks and combined at the end of the
inference run. Refer to [Sharding](#sharding) and
[Checkpointing](#checkpointing) sections for more details.

## Next Steps

Now that you have successfully run inference on a model, here are some topics
you might want to explore next:

+   [Fine-tuning a model.](finetune)
+   [Evaluating a model.](eval)
+   [Training a model from scratch.](pretrain)

We also touch upon a few advanced topics related to inference below that might
be useful, especially when customizing your inference job.

## Advanced Topics

### Dataset Sharding {#sharding .no-toc}

You can run inference in parallel across multiple TPU slices by setting the
`num_shards` flag when running using XManager. When `num_shards > 1`, the
dataset is interleaved among the shards and the predictions are combined in the
end; hence the order of examples in the data source and the predictions in the
output json files will not match (order is guaranteed to match for `num_shards =
1` or the number of input file shards).

### Dataset Checkpointing {#checkpointing .no-toc}

You can control dataset checkpointing frequency by overriding the
`infer.checkpoint_period` in
[runs/infer.gin](https://github.com/google-research/t5x/tree/main/t5x/configs/runs/infer.gin),
which is set to `100` by default. This means that the dataset is checkpointed
after running inferences on `checkpoint_period` batches (batches, not examples;
you can control batch size by overriding `utils.DatasetConfig.batch_size` in
[runs/infer.gin](https://github.com/google-research/t5x/tree/main/t5x/configs/runs/infer.gin), it
is set to `32` by default).


### Defining a custom SeqIO Task/Mixture to run inference on {.no-toc}

Refer to [SeqIO documentation](https://github.com/google/seqio/blob/main/README.md).
