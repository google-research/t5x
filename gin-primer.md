# Gin Primer


[Gin](https://github.com/google/gin-config) is a lightweight configuration
framework for Python, based on dependency injection. While T5X does not employ
gin in its core libraries, it is used to configure runs of the `train`, `eval`,
and `infer` scripts. This usage is a bit different (and more limited) than how
gin is typically applied, so this primer should be useful even for those who may
be familiar with gin from other libaries (e.g., T5 or Mesh TensorFlow).

Nevertheless, you may still find it helpful to refer to the
[gin documentation](https://github.com/google/gin-config) for more background.


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
a single top-level function and its arguments, we can better control the
configurable surface to public interfaces and user-owned code and also avoid
unintended side effects.

### An Example


Let's look at the `evaluate` call signature from `t5x/eval.py` as an example:

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

In the eval.py script, the user-provided gin configuration file will be parsed.
It specifies which values should be bound to the `evaluate` argument, after
which we can directly call the fully-bound function without any arguments.
Basically, we are creating a custom closure of `evaluate` (a la
`functools.partial`) but specifying the arguments via gin instead of Python.

Furthermore, this ability to bind custom arguments is recursive. Not only can we
bind the arguments of `evaluate`, but we can also bind the constructor and
method arguments of the instance of `models.BaseModel` that we pass to
`evaluate`.

Let's now look at an example of a gin configuration for parameterizing
`evaluate`, specifically using the same the gin file
[base_wmt_eval.gin](t5x/examples/t5/t5_1_1/examples/base_wmt_eval.gin) used in the English
to German translation task in the [README](README.md).

```py
from __gin__ import dynamic_registration

import __main__ as eval_script
from t5.data import mixtures
from t5x import partitioning
from t5x import utils

include "t5x/examples/t5/t5_1_1/base.gin"  # defines %MODEL.

CHECKPOINT_PATH = %gin.REQUIRED  # passed via commandline
EVAL_OUTPUT_DIR = %gin.REQUIRED  # passed via commandline
DROPOUT_RATE = 0.0  # unused boilerplate
MIXTURE_OR_TASK_NAME = "wmt_t2t_ende_v003"

eval_script.evaluate:
  model = %MODEL  # imported from separate gin file
  dataset_cfg = @utils.DatasetConfig()
  partitioner = @partitioning.ModelBasedPjitPartitioner()
  restore_checkpoint_cfg = @utils.RestoreCheckpointConfig()
  output_dir = %EVAL_OUTPUT_DIR

utils.DatasetConfig:
  mixture_or_task_name = %MIXTURE_OR_TASK_NAME
  task_feature_lengths = None  # Auto-computes the max feature lengths.
  split = 'test'
  batch_size = 32
  shuffle = False
  seed = 42

partitioning.ModelBasedPjitPartitioner.num_partitions = 2

utils.RestoreCheckpointConfig:
  path = %CHECKPOINT_PATH
  mode = 'specific'
```


Let's go through this block-by-block.

```py
from __gin__ import dynamic_registration
```

The first line imports a new gin feature to allow us to register functions and
objects for configuration from within the gin file itself without having to
modify or decorate functions from the imported packages.

```py
import __main__ as eval_script
from t5.data import mixtures
from t5x import partitioning
from t5x import utils
```

The second block imports the modules containing the components we plan to
configure in this file and is required for dynamic registration. Note that only
those functions and objects that we specify below will actually be configured,
not everything in the module. Also, as is the case in Python, the run script
(`t5x/eval.py` in this case) module is referred as `__main__`, although we
rename it to `eval_script` for clarity in the rest of the config.


```py
include "t5x/examples/t5/t5_1_1/base.gin"  # defines %MODEL
```

The third block loads in a separate gin file that defines the `%MODEL` macro.
Although we could define `MODEL = model.EncoderDecoderModel()` here, we prefer
to create a separate gin file that defines it. This makes it easier to reuse
parts of the common configurations. All of the bindings in the newly included
file are read and override any conflicting ones defined so far in this file.
It's equivalent to copy and pasting the contents of the included file at this
location in the config. If you want to see how the model itself is instantiated,
you can refer to [base.gin](t5x/examples/t5/t5_1_1/base.gin).


```py
CHECKPOINT_PATH = %gin.REQUIRED  # passed via commandline
EVAL_OUTPUT_DIR = %gin.REQUIRED  # passed via commandline
DROPOUT_RATE = 0.0  # unused boilerplate
MIXTURE_OR_TASK_NAME = "wmt_t2t_ende_v003"
```

The fourth block creates a [gin
macro](https://github.com/google/gin-config/blob/master/docs/index.md#gin-macros)
(essentially a lazy reference) and for now sets it to refer to the special macro
`gin.REQUIRED`. This will cause a failure during parsing of the configuration
if not updated via a later assignment in the config file or command-line flags
(see [below](#command-line-usage)). In the [README](README.md), we passed the
assignments via command-line flags.

```py
eval_script.evaluate:
  model = %MODEL  # imported from separate gin file
  dataset_cfg = @utils.DatasetConfig()
  partitioner = @partitioning.ModelBasedPjitPartitioner()
  restore_checkpoint_cfg = @utils.RestoreCheckpointConfig()
  output_dir = %EVAL_OUTPUT_DIR
```

The fifth block specifies the binding for the `evaluate` function. For `model`,
we pass the value of the `MODEL` macro defined in the third block. For
`output_dir` we use the `%EVAL_OUTPUT_DIR` to be passed via command-line flag.
For `dataset_cfg`, `restore_checkpoint_cfg`, and `partitioner`, we pass
instantiations of `DatasetConfig`, `RestoreCheckpointConfig`, and
`ModelBasedPjitPartitioner`, which are defined in `t5x/utils.py` and
`t5x/partitioning.py`, respectively. The '@' prefix tells gin that the following
is a configured function or class, and the '()' suffix signifies that it should
be called (in the cases of class, this means calling the constructor). If we
wanted to pass in the closure (or a partially bound) function instead of its
return value, we would leave off the parentheses.


```py
utils.DatasetConfig:
  mixture_or_task_name = %MIXTURE_OR_TASK_NAME
  task_feature_lengths = None  # Auto-computes the max feature lengths.
  split = 'test'
  batch_size = 32
  shuffle = False
  seed = 42

partitioning.ModelBasedPjitPartitioner.num_partitions = 2

utils.RestoreCheckpointConfig:
  path = %CHECKPOINT_PATH
  mode = 'specific'
```

The last 3 blocks are fairly straightforward. They are effectively setting the
attributes of these dataclasses by binding values to their constructors that
will be used when they are instantiated and passed to `evaluate`, as specified
in the fifth block. Note that there are two styles of binding values of the
constructor. Since we are only binding one argument in
`partitioning.ModelBasedPjitPartitioner.num_partitions = 2`, we choose to write
it on a single line instead of using the block binding syntax used elsewhere in
the file.


### Scoping

The above example lacks one key component of gin:
[scopes](https://github.com/google/gin-config/blob/master/docs/index.md#scoping).

What happens if you need to use a class or function multiple times but with
different bound values?

A clear example of this is in the top-level `train` function in `train.py`. The
call signature includes 3 different instances of `utils.DatasetConfig`: one for
the train dataset, one for the "train-eval" dataset (used for evaluation with
teacher forcing), and one for the "infer-eval" dataset (used for evaluation with
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
    the first when there are conflicts. Rather than passing the full path for
    each file, you may also supply a comma-separate list of search prefixes via
    `--gin_search_paths` and then only specify the relative to the `--gin_file`
    flags. If multiple search paths are provided, the first to yield a valid
    path for each gin file will be used.

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
must put it in quotes. In the case of strings, it requires "triple quotes"
(`"'<string>'"`). For example: `--gin.utils.DatasetConfig.split="'validation'"`,
`--gin.utils.DatasetConfig.task_feature_lengths="{'inputs': 512, 'targets':
84}"`, and `--gin.network.T%Config.mlp_activations="('dense', 'gelu')"`.

### An Example

An example where you may need multiple files is with the `train` script.

You can first specify which model you want to train by supplying a gin file
containing its definition, for example:
[t5x/examples/t5/t5_1_1/base.gin](t5x/examples/t5/t5_1_1/base.gin).

You may then specify a run config that supplies some of the common defaults. For
example, if you are doing pretraining you can use
[t5x/configs/runs/pretrain.gin](t5x/configs/runs/pretrain.gin), and if you are
doing finetuning, you can use
[t5x/configs/runs/finetune.gin](t5x/configs/runs/finetune.gin).

We can apply these two files with the following command:

```sh
T5X_DIR="..."  # directory where the t5x is cloned.
TFDS_DATA_DIR="..."

python3 ${T5X_DIR}/t5x/train.py \
  --tfds_data_dir=${TFDS_DATA_DIR} \
  --gin_file="t5x/examples/t5/t5_1_1/base.gin" \
  --gin_file="t5x/configs/runs/finetune.gin"
```

However, running this command will give you an error like the following:

```sh
ValueError: MODEL_DIR/macro.value set to `%gin.REQUIRED` but not subsequently overridden.
```

This is because the config still includes some `gin.REQUIRED` macros that you'll
need to override with the details of your run. At the top of
`t5x/configs/runs/finetune.gin` you'll see the list of required overrides, which
we will populate for finetuning on WMT in the updated launch command here:

```sh
T5X_DIR="..."  # directory where the t5x is cloned.
TFDS_DATA_DIR="..."
MODEL_DIR="..."

python3 ${T5X_DIR}/t5x/train.py \
  --tfds_data_dir=${TFDS_DATA_DIR} \
  --gin_file="t5x/examples/t5/t5_1_1/base.gin" \
  --gin_file="t5x/configs/runs/finetune.gin" \
  --gin.MIXTURE_OR_TASK_NAME="'wmt_t2t_ende_v003'" \
  --gin.MIXTURE_OR_TASK_MODULE="'t5.data.mixtures'" \
  --gin.TASK_FEATURE_LENGTHS="{'inputs': 256, 'targets': 256}" \
  --gin.TRAIN_STEPS=1_100_000 \
  --gin.MODEL_DIR="'${MODEL_DIR}'" \
  --gin.USE_CACHED_TASKS=False \
  --gin.INITIAL_CHECKPOINT_PATH="'gs://t5-data/pretrained_models/t5.1.1.base/model.ckpt-1000000'"
```

Note you may still override any registered bindings. For example, to disable
inference evaluation you may add `--gin.train.infer_eval_dataset_cfg=None`.

### A File-only approach

At the beginning of the primer, we saw a fully-specified run config. We can do
something similar with the previous example to create a self-contained run
configuration. We create one gin file with the configurations specified and
other gin files included. In the launch command, we only need to include this
"top level" gin file. The examples in the README use this approach.


## Logging

After your gin files and flag overrides are parsed, the complete configuration
will be logged to INFO, written to `config.gin` in the output directory, and
added to a TensorBoard summary.

It is highly recommended that you review this generated config to ensure that
your overrides are working as expected, especially if the same configurable is
set in multiple gin files.
