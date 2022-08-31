# Gin Primer


[Gin](https://github.com/google/gin-config/blob/main/README.md) is a lightweight configuration framework for Python,
based on dependency injection. While T5X does not employ gin in its core
libraries, it is used to configure runs of the `train`, `eval`, and `infer`
scripts. This usage is a bit different (and more limited) than how gin is
typically applied, so this primer should be useful even for those who may be
familiar with gin from other libaries (e.g., T5 or Mesh TensorFlow).

Nevertheless, you may still find it helpful to refer to the
[gin documentation](https://github.com/google/gin-config/blob/main/README.md) for more background.

[TOC]

## Gin in T5X Scripts

Rather than plumbing run arguments and hyperparameters through via limited set
of command-line flags or a flat configuration schema, T5X's gin integration
allows you to parameterize the top-level run functions (`train`, `evaluate`, and
`infer`) as well as any object or function that is passed to them. This enables
a vast amount of flexibility over your runs without needing to modify any code
within the core T5X library.

For example, you can implement a Python class in your own codebase (e.g., a
custom model or trainer) and use gin to pass an instance of it to the T5X XM
launcher without having to fork any code. Previously you needed to implement
every experimental idea in the core library (no matter how widely used it would
be) and add a ConfigDict flag to enable/disable it, resulting in significant
code debt over time.

On the other hand, gin can sometimes be too powerful, allowing users the ability
to bind arguments throughout a codebase, which makes it difficult or impossible
to update "private" internal interfaces. However, by limiting configurability to
a single top-level function and its arguments we can better control the
configurable surface to public interfaces and user-owned code, and also avoid
unintended side effects.

### An Example

Let's look at the `evaluate` call signature from
[eval.py](https://github.com/google-research/t5x/tree/main/t5x/eval.py) as an example:

```py
def evaluate(*,
             model: models.BaseModel,
             dataset_cfg: utils.DatasetConfig,
             restore_checkpoint_cfg: utils.RestoreCheckpointConfig,
             partitioner: partitioning.BasePartitioner,
             output_dir: str):
  """Evaluation function.

  Args:
    model: The model object to use for inference.
    dataset_cfg: Specification for the dataset to infer based on.
    restore_checkpoint_cfg: Specification for the model parameter checkpoint to
      load.
    partitioner: The partitioner for the model parameters and
      data across devices.
    output_dir: Path to directory to write temporary files and final results.
  """
  ...
```

In the binary, the user-provided gin configuration file will be parsed. It
specifies which values should be bound to the `evaluate` argument, after which
we can directly call the fully-bound function without any arguments. Basically,
we are creating a custom closure of `evaluate` (a la `functools.partial`) but
specifying the arguments via gin instead of Python.

Furthermore, this ability to bind custom arguments is recursive. Not only can we
bind the arguments of `evaluate`, but we can also bind the constructor and
method arguments of the instance of `models.BaseModel` that we pass to
`evaluate`.

Let's now look at an example of a gin configuration for parameterizing
`evaluate`, specifically evaluating a
[T5 model fine-tuned for closed book question answering](http://goo.gle/t5-cbqa)
on [Natural Questions Open](https://ai.google.com/research/NaturalQuestions):

```py
from __gin__ import dynamic_registration

import __main__ as eval_script
from t5x import models
from t5x import partitioning
from t5x import utils

MODEL = %gin.REQUIRED

eval_script.evaluate:
  model = %MODEL
  output_dir = '/tmp/t5x_eval'
  dataset_cfg = @utils.DatasetConfig()
  partitioner = @partitioning.PjitPartitioner()
  restore_checkpoint_cfg = @utils.RestoreCheckpointConfig()

# Load model with overrides.
include 'models/t5_large.gin'
models.EncoderDecoderModel.predict_batch_with_aux.num_decodes = 1

utils.DatasetConfig:
  mixture_or_task_name = 'natural_questions_open'
  split = 'test'
  task_feature_lengths = None
  batch_size = 32
  shuffle = False
  seed = 0
  use_cached = False
  pack = False
  use_custom_packing_ops = False
  module = 'google_research.t5_closed_book_qa.t5_cbqa.tasks'

partitioning.PjitPartitioner:
  num_partitions = 1

utils.RestoreCheckpointConfig:
  mode = 'specific'
  path = 'gs://t5-data/pretrained_models/cbqa/large_ssm_nqo'
  assignment_map = None
  strict = True
  dtype = None
```

Let's go through this block-by-block.

```py
from __gin__ import dynamic_registration
```

The first line imports a new gin feature (see cl/372624800 for more details) to
allow us to register functions and objects for configuration from within the gin
file itself without having to modify or decorate functions from the imported
packages.

```py
import __main__ as eval_script
from t5x import models
from t5x import utils
```

The second block imports the modules containing the components we plan to
configure in this file and is required for dynamic registration. Note that only
those functions and objects that we specify below will actually be configured,
not everything in the module. Also, as is the case in Python, the binary module
is referred as `__main__`, although we rename it to `eval_script` for clarity in
the rest of the config.

```py
MODEL = %gin.REQUIRED
```

The third block creates a
[gin macro](https://github.com/google/gin-config/tree/master/docs/index.md#gin-macros)
(essentially a lazy reference) and for now sets it to refer to the special macro
`gin.REQUIRED`, which will cause a failure during parsing of the configuration
if not updated via a later assignment in the config file or command-line flags
(see [below](#command-line-usage)).

```py
eval_script.evaluate:
  model = %MODEL
  output_dir = '/tmp/t5x_eval'
  dataset_cfg = @utils.DatasetConfig()
  partitioner = @partitioning.PjitPartitioner()
  restore_checkpoint_cfg = @utils.RestoreCheckpointConfig()
```

The fourth block specifies the binding for the `evaluate` function. For `model`,
we pass the value of the `MODEL` macro (to be defined later). For `output_dir`
we pass a string path. For `dataset_cfg`, `restore_checkpoint_cfg`, and
`partitioner`, we pass instantiations of `DatasetConfig`,
`RestoreCheckpointConfig`, and `PjitPartitioner`, which are defined in
[utils.py](https://github.com/google-research/t5x/tree/main/t5x/utils.py) and
[partitioning.py](https://github.com/google-research/t5x/tree/main/t5x/partitioning.py)
respectively. The '@' prefix tells gin that the following is a configured
function or class, and the '()' suffix signifies that it should be called (in
the cases of class, this means calling the constructor). If we wanted to pass in
the closure (or a partially bound) function instead of its return value, we
would leave off the parentheses.

The remainder of the file deals with defining the `MODEL` macro and fully
binding these constructors.

```py
# Load model with overrides.
include 't5x/examples/t5/t5_1_1/large.gin'
models.EncoderDecoderModel.predict_batch_with_aux.num_decodes = 1
```

Although we could define `MODEL = model.EncoderDecoderModel()` here, we prefer
to create a separate gin file that defines it. This makes it easier to reuse
parts of the common configurations. All of the bindings in the newly included
file are read and override any conflicting ones defined so far in this file.
It's equivalent to copy and pasting the contents of the included file at this
location in the config. If you want to see how the model itself is instantiated,
you can refer to
[t5_1_1/large.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/t5_1_1/large.gin)
(which simply overrides a few values from
[t5_1_1/base.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/t5_1_1/base.gin)).

The final line of this block shows an example of how you can modify the default
arguments of the `EncoderDecoderModel` instance referenced by `%MODEL`, in this
case changing the default beam size it will use during prediction. Notice that
since we are only binding one argument here, we choose to write it on a single
line instead of using the block binding syntax used elsewhere in the file.

```py
utils.DatasetConfig:
  mixture_or_task_name = 'natural_questions_open'
  split = 'test'
  task_feature_lengths = None
  batch_size = 32
  shuffle = False
  seed = 0
  use_cached = False
  pack = False
  use_custom_packing_ops = False
  module = 'google_research.t5_closed_book_qa.t5_cbqa.tasks'

partitioning.PjitPartitioner:
  num_partitions = 1

utils.RestoreCheckpointConfig:
  mode = 'specific'
  path = 'gs://t5-data/pretrained_models/cbqa/large_ssm_nqo'
  assignment_map = None
  strict = True
  dtype = None
```

The last 3 blocks are fairly straightforward. They are effectively setting the
attributes of these dataclasses by binding values to their constructors that
will be used when they are instantiated and passed to `evaluate`, as specified
in the fourth block.

### Scoping

The above example lacks one key component of gin:
[scopes](https://github.com/google/gin-config/blob/main/README.md#4-configuring-the-same-function-in-different-ways-scopes).

What happens if you need to use a class or function multiple times but with
different bound values?

A clear example of this is in the top-level `train` function (in
[train.py](https://github.com/google-research/t5x/tree/main/t5x/train.py)). The call signature
includes 3 different instances of `utils.DatasetConfig`: one for the train
dataset, one for the "train-eval" dataset (used for evaluation with teacher
forcing), and one for the "infer-eval" dataset (used for evaluation with
inference/decoding).

The solution is to prefix each instance with a unique identifier both when
specifying where it is to be passed to `train` and when binding its arguments.
For example, the gin file might look like the following (skipping the irrelevant
bits):

```py
...

train_script.train:
  train_dataset_cfg = @train/utils.DatasetConfig()
  train_eval_dataset_cfg = @train_eval/utils.DatasetConfig()
  infer_eval_dataset_cfg = @infer_eval/utils.DatasetConfig()
  ...

train/utils.DatasetConfig:
  mixture_or_task_name = 'train_mixture'
  split = 'train'
  ...

train_eval/utils.DatasetConfig:
  mixture_or_task_name = 'eval_mixture'
  split = 'validation'
  ...

infer_eval/utils.DatasetConfig:
  mixture_or_task_name = 'eval_mixture'
  split = 'test'
  ...
```

We have therefore configured 3 different scoped-versions of
`utils.DatasetConfig` producing 3 separate instances that are passed to `train`.

Note that these three scopes will all inherit from the base scope, so if you
want to set a shared binding, you may directly configure `utils.DatasetConfig`
without a scope prefix.

## Command-Line Usage

So now that you have a gin config, how do you pass it to the script? There are
two ways: gin files and override flags.

1.  **Gin Files** You have already seen an example of a gin file above. You can
    specify the gin file(s) to use in your script via the `--gin_file` flag. If
    you want to load multiple gin files, you can set the flag multiple times and
    the files will be loaded in order, with the second potentially overriding
    the first when there are conflicts. It is possible to supply a
    comma-separate list of search prefixes via `--gin_search_paths` and then
    only specify the relative path to the `--gin_file` flags. However, we
    strongly recommend against using `--gin_search_paths`. Using absolute paths
    via the `--gin_file` flags will reduce sources of ambiguity and improve
    the consistency of your scripts.

1.  **Override Flags** Gin flags allow for more fine-grained overrides of any
    configurable aspect of your run. These flags follow the single-line binding
    format from the above example with the addition of a `--gin.` prefix. For
    example, if you want to override the dataset shuffling, you can set
    `--gin.utils.DatasetConfig.shuffle=False`. In the train setting where there
    are multiple datasets, you must supply the appropriate scope, e.g.,
    `--gin.train/utils.DatasetConfig.shuffle=False`. These bindings are
    processed in order *after* the gin files are loaded, and therefore overwrite
    any previously assigned value in the gin files.

**Note:** when supplying a string, dict, list, or tuple value via a flag, you
must put it in quotes. In the case of strings, it requires escaped quotes
(`\"<string>\"`). For example: `--gin.utils.DatasetConfig.split=\"validation\"`,
`--gin.utils.DatasetConfig.task_feature_lengths="{'inputs': 512, 'targets':
84}"`, and `--gin.dense.MlpBlock.activations="('dense', 'gelu')"`

### An Example

An example where you may need multiple files is with the `train` script.

You can first specify which model you want to train by supplying a gin file
containing its definition, for example:
[t5_1_1/small.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/t5_1_1/small.gin).

You may then specify a run config that supplies some of the common defaults. For
example, if you are doing pretraining you can use
[runs/pretrain.gin](https://github.com/google-research/t5x/tree/main/t5x/configs/runs/pretrain.gin),
and if you are doing finetuning, you can use
[runs/finetune.gin](https://github.com/google-research/t5x/tree/main/t5x/configs/runs/finetune.gin).

We can apply these two files with the following command:

```sh
python -m t5x.train \
  --gin_file=t5x/examples/t5/t5_1_1/small.gin \
  --gin_file=t5x/configs/runs/finetune.gin \
  --logtostderr
```

However, running this command will give you an error like the following:

```sh
ValueError: MODEL_DIR/macro.value set to `%gin.REQUIRED` but not subsequently overridden.
```

This is because the config still includes some `gin.REQUIRED` macros that you'll
need to override with the details of your run. At the top of
[runs/finetune.gin](https://github.com/google-research/t5x/tree/main/t5x/configs/runs/finetune.gin)
you'll see the list of required overrides, which we will populate for finetuning
on WMT in the updated launch command here:

```sh
python -m t5x.train \
  --gin_file=t5x/examples/t5/t5_1_1/small.gin \
  --gin_file=t5x/configs/runs/finetune.gin \
  --gin.MIXTURE_OR_TASK_NAME=\"wmt_t2t_ende_v003\" \
  --gin.MIXTURE_OR_TASK_MODULE=\"t5.data.mixtures\" \
  --gin.TASK_FEATURE_LENGTHS="{'inputs': 256, 'targets': 256}" \
  --gin.TRAIN_STEPS=1_020_000 \
  --gin.MODEL_DIR=\"/tmp/t5_1_1_base_finetune_gin\" \
  --gin.INITIAL_CHECKPOINT_PATH=\"gs://t5-data/pretrained_models/t5x/t5_1_1_small/checkpoint_1000000\" \
  --logtostderr
```

Note you may still override any registered bindings. For example, to disable
inference evaluation you may add `--gin.train.infer_eval_dataset_cfg=None`.

### A File-only Example

At the beginning of the primer, we saw a fully-specified run config. We can do
something similar with the previous example to create a self-contained run
configuration.
[t5_1_1/examples/base_wmt_finetune.gin](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/t5_1_1/examples/small_wmt_finetune.gin)
is just such an example that allows you to exactly duplicate the previous launch
command simply by calling:

```sh
python -m t5x.train \
  --gin_file=t5x/examples/t5/t5_1_1/examples/small_wmt_finetune.gin \
  --logtostderr
```

## Logging

After your gin files and flag overrides are parsed, the complete configuration
will be logged to INFO, written to `config.gin` in the output directory, and
added to a TensorBoard summary.

It is highly recommended that you review this generated config to ensure that
your overrides are working as expected.
