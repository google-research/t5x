# T5X

*Go to [T5X ReadTheDocs Documentation Page](https://t5x.readthedocs.io/).*

T5X is a modular, composable, research-friendly framework for high-performance,
configurable, self-service training, evaluation, and inference of sequence
models (starting with language) at many scales.

It is essentially a new and improved implementation of the
[T5 codebase](https://github.com/google-research/text-to-text-transfer-transformer)
(based on [Mesh TensorFlow](https://github.com/tensorflow/mesh)) in [JAX](https://github.com/google/jax) and [Flax](https://github.com/google/flax). To learn
more, see the [T5X Paper](https://arxiv.org/abs/2203.17189).

Below is a quick start guide for training models with TPUs on Google Cloud. For
additional tutorials and background, see the [complete documentation](docs/index.md).

## Quickstart (Recommended)

T5X can be run with [XManager](https://github.com/deepmind/xmanager) on
[Vertex AI](https://cloud.google.com/vertex-ai). Vertex AI is a platform for
training that creates TPU instances and runs code on the TPUs. Vertex AI will
also shut down the TPUs when the jobs terminate. This is signifcantly easier
than managing GCE VMs and TPU VM instances.

1. Follow the pre-requisites and directions to install [XManager](https://github.com/deepmind/xmanager).

2. Request TPU quota as required. GCP projects come with 8 cores by default,
which is enough to run one training experiment on a single TPU host. If you want
to run multi-host training or run multiple trials in parallel, you will need
more quota. Navigate to [Quotas](https://console.cloud.google.com/quotas).

  The quota you want is:

  * Service: `Vertex AI API`
  * Dimensions (location): `us-central1`
  * If you want to run single-host experiments:
    * `Custom model training TPU V2 cores per region`
    * `Custom model training TPU V3 cores per region`
  * If you want to run multi-host experiments:
    * `Custom model training TPU V2 pod cores per region`
    * `Custom model training TPU V3 pod cores per region`

  TIP: You won't be able to run single-host experiments with multi-host quota.
  (i.e. you can't run `tpu_v2=8` using `TPU V2 pod`)


3. Launch the xmanager script located at `t5x/scripts/xm_launch.py`.

As a running example, we use the WMT14 En-De translation which is described in
more detail in the Examples section below.

```sh
export GOOGLE_CLOUD_BUCKET_NAME=...
export TFDS_DATA_DIR=gs://$GOOGLE_CLOUD_BUCKET_NAME/t5x/data
export MODEL_DIR=gs://$GOOGLE_CLOUD_BUCKET_NAME/t5x/$(date +%Y%m%d)

# Pre-download dataset in multi-host experiments.
tfds build wmt_t2t_translate --data_dir=$TFDS_DATA_DIR

git clone https://github.com/google-research/t5x
cd ./t5x/

python3 ./t5x/scripts/xm_launch.py \
  --gin_file=t5x/examples/t5/t5_1_1/examples/base_wmt_from_scratch.gin \
  --model_dir=$MODEL_DIR \
  --tfds_data_dir=$TFDS_DATA_DIR
```

Check `gs://$GOOGLE_CLOUD_BUCKET_NAME/t5x/` for the output artifacts, which can
be read by TensorBoard.

## GPU Usage
UPDATE!: Nvidia has released an updated version of this repository with H100 FP8 support and broad GPU performance improvements here: [NVIDIA Rosetta](https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/t5x)

T5X can be run easily on GPUs either in single-node configurations or multi-node configurations with a SLURM+pyxis cluster. Further instructions at [Rosetta T5X README](https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/t5x/README.md). The `t5x/contrib/gpu/scripts_gpu` folder contains example scripts for pretraining T5X on [The Pile](https://pile.eleuther.ai/) and for finetuning on SQuAD and MNLI. These scripts and associated `gin` configurations also contain additional GPU optimizations for better throughput.

We now have support for:
- [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) FP8
- Improved performance on H100/A100 GPUs

## Installation

Note that all the commands in this document should be run in the commandline of
the TPU VM instance unless otherwise stated.

1.  Follow the
    [instructions](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm#install_the_google_cloud_sdk)
    to set up a Google Cloud Platform (GCP) account and enable the Cloud TPU
    API.

    **Note:** T5X also works with GPU, please follow instructions in [t5x/contrib/gpu/scripts_gpu](https://github.com/google-research/t5x/blob/main/t5x/contrib/gpu/scripts_gpu/README.md) if you'd like to use GPU version.

2.  Create a
    [Cloud TPU VM instance](https://cloud.google.com/blog/products/compute/introducing-cloud-tpu-vms)
    following
    [this instruction](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm#create-vm).
    We recommend that you develop your workflow in a single v3-8 TPU (i.e.,
    `--accelerator-type=v3-8`) and scale up to pod slices once the pipeline is
    ready. In this README, we focus on using a single v3-8 TPU. See
    [here](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm) to
    learn more about TPU architectures.

3.  With Cloud TPU VMs, you ssh directly into the host machine of the TPU VM.
    You can install packages, run your code run, etc. in the host machine. Once
    the TPU instance is created, ssh into it with

    ```sh
    gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE}
    ```

    where `TPU_NAME` and `ZONE` are the name and the zone used in step 2.

4.  Install T5X and the dependencies.

    ```sh
    git clone --branch=main https://github.com/google-research/t5x
    cd t5x

    python3 -m pip install -e '.[tpu]' -f \
      https://storage.googleapis.com/jax-releases/libtpu_releases.html

    ```


5.  Create Google Cloud Storage (GCS) bucket to store the dataset and model
    checkpoints. To create a GCS bucket, see these
    [instructions](https://cloud.google.com/storage/docs/creating-buckets).

6.  (optional) If you prefer working with Jupyter/Colab style environment
    you can setup a custom Colab runtime by following steps from
    [t5x/notebooks](https://github.com/google-research/t5x/blob/main/t5x/notebooks/README.md).

## Example: English to German translation

As a running example, we use the WMT14 En-De translation. The raw dataset is
available in TensorFlow Datasets as
["wmt_t2t_translate"](https://www.tensorflow.org/datasets/catalog/wmt_t2t_translate).

T5 casts the translation task such as the following

```py
{'en': 'That is good.', 'de': 'Das ist gut.'}
```

to the form called "text-to-text":

```py
{'inputs': 'translate English to German: That is good.', 'targets': 'Das ist gut.'}
```

This formulation allows many different classes of language tasks to be expressed
in a uniform manner and a single encoder-decoder architecture can handle them
without any task-specific parameters. For more detail, refer to the [T5 paper
(Raffel et al. 2019)][t5_paper].

For a scalable data pipeline and an evaluation framework, we use
[`SeqIO`](https://github.com/google/seqio), which was factored out of the [T5
library][t5_github]. A `seqio.Task` packages together the raw dataset, vocabulary,
preprocessing such as tokenization and evaluation metrics such as
[BLEU](https://aclanthology.org/P02-1040.pdf) and provides a
[`tf.data`](https://www.tensorflow.org/guide/data) instance.

[The T5 library][t5_github] provides a number of `seqio.Task`s that were used in the
[T5 paper][t5_paper]. In this example, we use [wmt_t2t_ende_v003](https://github.com/google-research/text-to-text-transfer-transformer/blob/d81c0bab2a41b4d5dfbe4971de32f7d67df65f31/t5/data/tasks.py#L212).

Before training or fine-tuning you need to download ["wmt_t2t_translate"]
(https://www.tensorflow.org/datasets/catalog/wmt_t2t_translate) dataset first.

```sh
# Data dir to save the processed dataset in "gs://data_dir" format.
TFDS_DATA_DIR="..."

# Make sure that dataset package is up-to-date.
python3 -m pip install --upgrade tfds-nightly

# Pre-download dataset.
tfds build wmt_t2t_translate ${TFDS_DATA_DIR}
```

### Training

To run a training job, we use the `t5x/train.py` script.

```sh
# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
MODEL_DIR="..."
T5X_DIR="..."  # directory where the T5X repo is cloned.
TFDS_DATA_DIR="..."

python3 ${T5X_DIR}/t5x/train.py \
  --gin_file="t5x/examples/t5/t5_1_1/examples/base_wmt_from_scratch.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --tfds_data_dir=${TFDS_DATA_DIR}
```

The configuration for this training run is defined in the Gin file
[base_wmt_from_scratch.gin](t5x/examples/t5/t5_1_1/examples/base_wmt_from_scratch.gin).
[Gin-config](https://github.com/google/gin-config) is a library to handle
configurations based on dependency injection. Among many benefits, Gin allows
users to pass custom components such as a custom model to the T5X library
without having to modify the core library. The [custom
components](#custom-components) section shows how this is done.

While the core library is independent of Gin, it is central to the examples we
provide. Therefore, we provide a short [introduction][gin-primer] to Gin in the
context of T5X.  All the configurations are written to a file "config.gin" in
`MODEL_DIR`. This makes debugging as well as reproducing the experiment much
easier.

In addition to the `config.json`, `model-info.txt` file summarizes the model
parameters (shape, names of the axes, partitioning info) as well as the
optimizer states.



#### TensorBoard

To monitor the training in [TensorBoard](https://www.tensorflow.org/tensorboard), it is much easier (due to
authentification issues) to launch the TensorBoard on your own machine and _not_ in
the TPU VM. So in the commandline where you ssh'ed into the TPU VM, launch the
TensorBoard with the `logdir` pointing to the `MODEL_DIR`.

```sh
# NB: run this on your machine not TPU VM!
MODEL_DIR="..."  # Copy from the TPU VM.
tensorboard --logdir=${MODEL_DIR}
```

Or you can launch the TensorBoard inside a Colab. In a Colab cell, run

```python
from google.colab import auth
auth.authenticate_user()
```

to authorize the Colab to access the GCS bucket and launch the TensorBoard.

```python
%load_ext tensorboard
model_dir = "..."  # Copy from the TPU VM.
%tensorboard --logdir=model_dir
```


### Fine-tuning

We can leverage the benefits of self-supervised pre-training by initializing
from one of our pre-trained models. Here we use the T5.1.1 Base checkpoint.

```sh
# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
MODEL_DIR="..."

# Data dir to save the processed dataset in "gs://data_dir" format.
TFDS_DATA_DIR="..."
T5X_DIR="..."  # directory where the T5X repo is cloned.

python3 ${T5X_DIR}/t5x/train.py \
  --gin_file="t5x/examples/t5/t5_1_1/examples/base_wmt_finetune.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --tfds_data_dir=${TFDS_DATA_DIR}
```

**Note:** when supplying a string, dict, list, tuple value, or a bash variable
via a flag, you must put it in quotes. In the case of strings, it requires
escaped quotes (`\"<string>\"`). For example:
`--gin.utils.DatasetConfig.split=\"validation\"` or
`--gin.MODEL_DIR=\"${MODEL_DIR}\"`.

Gin makes it easy to change a number of configurations. For example, you can
change the `partitioning.PjitPartitioner.num_partitions` (overriding
the value in
[base_wmt_from_scratch.gin](t5x/examples/t5/t5_1_1/examples/base_wmt_from_scratch.gin))
to chanage the parallelism strategy and pass it as a commandline arg.

```sh
--gin.partitioning.PjitPartitioner.num_partitions=8
```

### Evaluation

To run the offline (i.e. without training) evaluation, you can use `t5x/eval.py`
script.

```sh
EVAL_OUTPUT_DIR="..."  # directory to write eval output
T5X_DIR="..."  # directory where the t5x is cloned, e.g., ${HOME}"/t5x".
TFDS_DATA_DIR="..."
CHECKPOINT_PATH="..."

python3 ${T5X_DIR}/t5x/eval.py \
  --gin_file="t5x/examples/t5/t5_1_1/examples/base_wmt_eval.gin" \
  --gin.CHECKPOINT_PATH=\"${CHECKPOINT_PATH}\" \
  --gin.EVAL_OUTPUT_DIR=\"${EVAL_OUTPUT_DIR}\" \
  --tfds_data_dir=${TFDS_DATA_DIR}
```


### Inference

To run inference, you can use `t5x/infer.py` script. Here we use the same
`seqio.Task`, but for inference we do not use the targets features other than
logging them alongside the prediction in a JSON file.

```sh
INFER_OUTPUT_DIR="..."  # directory to write infer output
T5X_DIR="..."  # directory where the t5x is cloned, e.g., ${HOME}"/t5x".
TFDS_DATA_DIR="..."
CHECKPOINT_PATH="..."

python3 ${T5X_DIR}/t5x/infer.py \
  --gin_file="t5x/examples/t5/t5_1_1/examples/base_wmt_infer.gin" \
  --gin.CHECKPOINT_PATH=\"${CHECKPOINT_PATH}\" \
  --gin.INFER_OUTPUT_DIR=\"${INFER_OUTPUT_DIR}\" \
  --tfds_data_dir=${TFDS_DATA_DIR}
```

### Exporting as TensorFlow Saved Model

Pretrained model can be exported as TensorFlow Saved Model, and deployed
to Vertex AI Prediction service using [Optimized TensorFlow Runtime]
(https://cloud.google.com/vertex-ai/docs/predictions/optimized-tensorflow-runtime).
Please note that exported model won't work on OSS based
[TensorFlow Model Server](https://github.com/tensorflow/serving).

```sh
T5X_DIR="..."  # directory where the t5x is cloned, e.g., ${HOME}"/t5x".
CHECKPOINT_PATH="..."

BATCH_SIZE=None
BEAM_SIZE=1

# Use 'bfloat16' if you plan to run exported model on NVIDIA A100 or newer GPUs,
# for other GPUs use 'float32'.
ACTIVATION_DTYPE=bfloat16

# Version numbers must be numeric. We generate one based on datetime.
VERSION=$(date +%Y%m%d%H%M%S)

NAME=t5x_base_${ACTIVATION_DTYPE}  # Model name.

# Path to export model to. Note that export script is going to add _cpu suffix
# after model name.
OUTPUT=${CHECKPOINT_PATH}/saved_model.${NAME}/${VERSION}

declare -a ARGS=(
--gin_file=t5x/examples/t5/t5_1_1/base.gin
--gin_file=t5x/t5x/configs/runs/export.gin
--gin.TASK_FEATURE_LENGTHS="{'inputs': 256, 'targets': 256}"
--gin.CHECKPOINT_PATH=\"${CHECKPOINT_PATH}\"
--gin.MODEL_NAME=\"/ml/${USER}/t5x_base\"
--gin.MODEL_OUTPUT_DIR=\"${OUTPUT}\"
--gin.BEAM_SIZE=${BEAM_SIZE}
--gin.BATCH_SIZE=${BATCH_SIZE}
--gin.export_lib.save.partitioner=None
--gin.export_lib.save.warmup_examples="['hello world']"
--gin.export_lib.ExportableModule.use_batch_function=False
--gin.export_lib.ExportableModule.use_gpu=False
--gin.export_lib.ExportableModule.jit_compile=False
--gin.ACTIVATION_DTYPE=\"${ACTIVATION_DTYPE}\"
--gin.network.T5Config.dtype=\"${ACTIVATION_DTYPE}\"
--gin.utils.RestoreCheckpointConfig.dtype=\"${ACTIVATION_DTYPE}\"
--gin.DROPOUT_RATE=0.0
)

(python3 ${T5X_DIR}/t5x/export.py "${ARGS[@]}")
```

For detailed arguments definition refer to [export.gin]
(t5x/configs/runs/export.gin).

You can run XL and smaller models on NVIDIA A100 40GB, and XXL models on
NVIDIA A100 80GB.

## Custom components

[The translation example](#example-english-to-german-translation) uses the
encoder-decoder model that T5X provides as well as the dataset from the T5
library. This section shows how you can use your own dataset and a model and
pass via Gin.

### Example: custom dataset in a user directory

For this example, we have the following directory structure with
`${HOME}/dir1/user_dir` representing a user directory with custom components.

```
${HOME}
└── dir1
    └── user_dir
        ├── t5_1_1_base_de_en.gin
        └── tasks.py
```

As an example, let's define a new dataset. Here we use the same Translation
dataset but we define the translation task in the opposite direction, i.e.,
German to English intead of English to German. We define this task in `tasks.py`

```py
# ${HOME}/dir1/user_dir/tasks.py

import functools
import seqio
import tensorflow_datasets as tfds
from t5.evaluation import metrics
from t5.data import preprocessors

vocabulary = seqio.SentencePieceVocabulary(
    'gs://t5-data/vocabs/cc_all.32000/sentencepiece.model', extra_ids=100)
output_features = {
    'inputs': seqio.Feature(vocabulary=vocabulary),
    'targets': seqio.Feature(vocabulary=vocabulary)
}

seqio.TaskRegistry.add(
    'wmt_t2t_de_en_v003',
    source=seqio.TfdsDataSource(tfds_name='wmt_t2t_translate/de-en:1.0.0'),
    preprocessors=[
        functools.partial(
            preprocessors.translate,
            source_language='de', target_language='en'),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[metrics.bleu],
    output_features=output_features)
```

In the Gin file, most of the settings are equivalent to those used in the
[En->De example](#example-english-to-german-translation). So we include the Gin
file from that example. To use "wmt_t2t_de_en_v003" task we just defined, we
need to import the task module "tasks.py". Note that we use a relative path
defined with respect to the user directory. This will be specified as a
flag.

```py
# ${HOME}/dir1/user_dir/t5_1_1_base_de_en.gin
from __gin__ import dynamic_registration
import tasks  # This imports the task defined in dir1/user_dir/tasks.py.

include "t5x-tmp/t5x/examples/t5/t5_1_1/examples/base_wmt_from_scratch.gin"
MIXTURE_OR_TASK_NAME = "wmt_t2t_de_en_v003"
```

Finally, we launch training passing the user directory as a flag
`gin_search_paths` such that the Gin file and python modules can be specified
with relative paths.

```sh
PROJECT_DIR=${HOME}"/dir1/user_dir"
T5X_DIR="..."  # directory where the t5x is cloned.
TFDS_DATA_DIR="..."
MODEL_DIR="..."
export PYTHONPATH=${PROJECT_DIR}

python3 ${T5X_DIR}/t5x/train.py \
  --gin_search_paths=${PROJECT_DIR} \
  --gin_file="t5_1_1_base_de_en.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --tfds_data_dir=${TFDS_DATA_DIR}
```

## Checkpoints

### Native Checkpoints

We have released the checkpoints of many of the original T5 models and their
variants a native T5X format for maximal efficiency.
See the [complete list](https://github.com/google-research/t5x/blob/main/docs/models.md) including the
matching Gin configuration files.

These are converted from the public [Mesh TensorFlow
checkpoints](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md#t511)
.


### Compatibility with the Mesh TensorFlow checkpoints
The Mesh TensorFlow checkpoints trained using the [T5 library][t5_github] can be
directly loaded into T5X. For example, we can rerun the fine-tuning example
initializing from the MTF checkpoint by changing the `INIT_CHECKPOINT` Gin
macro.

```sh
# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
MODEL_DIR="..."

# Data dir to save the processed dataset in "gs://data_dir" format.
TFDS_DATA_DIR="..."
T5X_DIR="..."  # directory where the T5X repo is cloned.

python3 ${T5X_DIR}/t5x/train.py \
  --gin_file="t5x/examples/t5/t5_1_1/examples/base_wmt19_ende_train.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.MIXTURE_OR_TASK_NAME=\"wmt_t2t_ende_v003\" \
  --gin.INIT_CHECKPOINT=\"gs://t5-data/pretrained_models/t5.1.1.base/model.ckpt-1000000\" \
  --tfds_data_dir=${TFDS_DATA_DIR}
```

Note that restoring directly from the Mesh TensorFlow checkpoints can be
inefficient if heavy model parallelism is used for large models. This is
because each host loads the entire copy of the model first and then keep only
the relevant slices dictated by the model parallelism specification. If you have
Mesh TensorFlow checkpoints that you run often, we recommend converting the
checkpoints to T5X native format using the
[convert_tf_checkpoint script](t5x/scripts/convert_tf_checkpoint.py).


## Citing T5X
Please use the following bibtex entry to cite T5X.

```
@article{roberts2022t5x,
  url = {https://arxiv.org/abs/2203.17189},
  author = {Roberts, Adam and Chung, Hyung Won and Levskaya, Anselm and Mishra, Gaurav and Bradbury, James and Andor, Daniel and Narang, Sharan and Lester, Brian and Gaffney, Colin and Mohiuddin, Afroz and Hawthorne, Curtis and Lewkowycz, Aitor and Salcianu, Alex and van Zee, Marc and Austin, Jacob and Goodman, Sebastian and Soares, Livio Baldini and Hu, Haitang and Tsvyashchenko, Sasha and Chowdhery, Aakanksha and Bastings, Jasmijn and Bulian, Jannis and Garcia, Xavier and Ni, Jianmo and Chen, Andrew and Kenealy, Kathleen and Clark, Jonathan H. and Lee, Stephan and Garrette, Dan and Lee-Thorp, James and Raffel, Colin and Shazeer, Noam and Ritter, Marvin and Bosma, Maarten and Passos, Alexandre and Maitin-Shepard, Jeremy and Fiedel, Noah and Omernick, Mark and Saeta, Brennan and Sepassi, Ryan and Spiridonov, Alexander and Newlan, Joshua and Gesmundo, Andrea},
  title = {Scaling Up Models and Data with $\texttt{t5x}$ and $\texttt{seqio}$},
  journal={arXiv preprint arXiv:2203.17189},
  year = {2022},
}
```


## Note
This is not an officially supported Google product

[t5_paper]: https://arxiv.org/abs/1910.10683
[t5_github]: https://github.com/google-research/text-to-text-transfer-transformer
[gin-primer]: docs/usage/gin.md
