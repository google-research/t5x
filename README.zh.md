# T5X

[English](README.md) | [简体中文](README.zh.md)

*前往 [T5X ReadTheDocs 文档主页](https://t5x.readthedocs.io/)。*

T5X 是一个模块化、可组合、对研究友好的框架，用于在多尺度上对序列模型（从语言模型起步）进行高性能、可配置且自服务的训练、评估与推理。

它本质上是基于 [JAX](https://github.com/google/jax) 和 [Flax](https://github.com/google/flax) 对原 [T5 代码库](https://github.com/google-research/text-to-text-transfer-transformer)（原基于 [Mesh TensorFlow](https://github.com/tensorflow/mesh)）的全新改进实现。欲了解更多信息，请参阅 [T5X 论文](https://arxiv.org/abs/2203.17189)。

以下是在 Google Cloud 上使用 TPU 训练模型的快速入门指南。有关其他教程与背景知识，请参阅[完整文档](docs/index.md)。

## 快速开始（推荐）

T5X 可以在 [Vertex AI](https://cloud.google.com/vertex-ai) 上配合 [XManager](https://github.com/deepmind/xmanager) 运行。Vertex AI 是一个训练平台，可创建 TPU 实例并在 TPU 上运行代码。当作业终止时，Vertex AI 还会自动关停 TPU。这比管理 GCE 虚拟机和 TPU VM 实例要简单得多。

1. 遵循前置条件与指引安装 [XManager](https://github.com/deepmind/xmanager)。

2. 根据需要申请 TPU 配额（Quota）。GCP 项目默认配备 8 个核心，足以在单个 TPU 主机上运行一次训练实验。如果您希望运行多主机训练或并行运行多个试验，则需要更多配额。请前往[配额页面](https://console.cloud.google.com/quotas)。

  所需配额项为：

  * 服务（Service）：`Vertex AI API`
  * 维度（位置，Dimensions）：`us-central1`
  * 如需运行单主机实验：
    * `每个区域自定义模型定型 TPU V2 核心`（Custom model training TPU V2 cores per region）
    * `每个区域自定义模型定型 TPU V3 核心`（Custom model training TPU V3 cores per region）
  * 如需运行多主机实验：
    * `每个区域自定义模型定型 TPU V2 Pod 核心`（Custom model training TPU V2 pod cores per region）
    * `每个区域自定义模型定型 TPU V3 Pod 核心`（Custom model training TPU V3 pod cores per region）

  提示（TIP）：无法使用多主机配额来运行单主机实验（例如，不能使用 `TPU V2 pod` 配额来运行 `tpu_v2=8`）。


3. 启动位于 `t5x/scripts/xm_launch.py` 的 xmanager 脚本。

作为贯穿本文的示例，我们使用 WMT14 英德（En-De）翻译任务，下文的示例章节将对其进行更详尽的说明。

```sh
export GOOGLE_CLOUD_BUCKET_NAME=...
export TFDS_DATA_DIR=gs://$GOOGLE_CLOUD_BUCKET_NAME/t5x/data
export MODEL_DIR=gs://$GOOGLE_CLOUD_BUCKET_NAME/t5x/$(date +%Y%m%d)

# 在多主机实验中预先下载数据集。
tfds build wmt_t2t_translate --data_dir=$TFDS_DATA_DIR

git clone https://github.com/google-research/t5x
cd ./t5x/

python3 ./t5x/scripts/xm_launch.py \
  --gin_file=t5x/examples/t5/t5_1_1/examples/base_wmt_from_scratch.gin \
  --model_dir=$MODEL_DIR \
  --tfds_data_dir=$TFDS_DATA_DIR
```

检查 `gs://$GOOGLE_CLOUD_BUCKET_NAME/t5x/` 中的输出工件，可通过 TensorBoard 进行读取。

## GPU 使用
注意：NVIDIA 已发布了本仓库的更新版本，支持 H100 FP8 并带来了广泛的 GPU 性能提升。详情及使用说明请访问 [NVIDIA Rosetta](https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/t5x) 仓库。

T5X 可以轻松在 GPU 上运行，既支持单节点配置，也支持结合 SLURM+pyxis 集群的多节点配置。更多说明请参见 [t5x/contrib/gpu](https://github.com/google-research/t5x/blob/main/t5x/contrib/gpu/README.md)。`t5x/contrib/gpu/scripts_gpu` 目录包含在 [The Pile](https://pile.eleuther.ai/) 上预训练 T5X 以及在 SQuAD 和 MNLI 上微调的示例脚本。这些脚本及相关的 `gin` 配置还包含额外的 GPU 吞吐量优化。更多示例和指引可在 NVIDIA 维护的 [NVIDIA Rosetta](https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/t5x) 仓库中找到，该仓库支持 H100 FP8 并具有全面的 GPU 性能改进。


## 安装指南

请注意，除非另有说明，本文档中的所有命令均应在 TPU VM 实例的命令行中运行。

1.  按照[说明](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm#install_the_google_cloud_sdk)设置 Google Cloud Platform (GCP) 账号并启用 Cloud TPU API。

    **注意：** T5X 同样支持 GPU，如需使用 GPU 版本，请参考 [t5x/contrib/gpu](https://github.com/google-research/t5x/blob/main/t5x/contrib/gpu/README.md) 中的说明。

2.  按照[此指南](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm#create-vm)创建 [Cloud TPU VM 实例](https://cloud.google.com/blog/products/compute/introducing-cloud-tpu-vms)。建议先在单个 v3-8 TPU（即 `--accelerator-type=v3-8`）上开发工作流，待流水线准备就绪后再扩展到 Pod 切片。在本 README 中，我们重点介绍使用单台 v3-8 TPU。参阅[此处](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm)以了解有关 TPU 架构的更多信息。

3.  对于 Cloud TPU VM，可以直接通过 SSH 连接到 TPU VM 的宿主机中。在宿主机内可以安装软件包、运行代码等。创建 TPU 实例后，通过以下命令 SSH 登录：

    ```sh
    gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE}
    ```

    其中 `TPU_NAME` 和 `ZONE` 为步骤 2 中使用的名称和区域。

4.  安装 T5X 及其依赖项。

    ```sh
    git clone --branch=main https://github.com/google-research/t5x
    cd t5x

    python3 -m pip install -e '.[tpu]' -f \
      https://storage.googleapis.com/jax-releases/libtpu_releases.html

    ```


5.  创建 Google Cloud Storage (GCS) 存储分区以存放数据集和模型检查点。创建 GCS 存储分区请参阅[这些说明](https://cloud.google.com/storage/docs/creating-buckets)。

6.  （可选）如果您更偏好 Jupyter/Colab 风格的环境，可以按照 [t5x/notebooks](https://github.com/google-research/t5x/blob/main/t5x/notebooks/README.md) 中的步骤设置自定义 Colab 运行时。

## 示例：英语到德语翻译

作为贯穿本文的示例，我们使用 WMT14 英德翻译任务。原始数据集可在 TensorFlow Datasets 中以 ["wmt_t2t_translate"](https://www.tensorflow.org/datasets/catalog/wmt_t2t_translate) 获取。

T5 将翻译任务（如下所示）：

```py
{'en': 'That is good.', 'de': 'Das ist gut.'}
```

转换为称为“文本到文本（text-to-text）”的形式：

```py
{'inputs': 'translate English to German: That is good.', 'targets': 'Das ist gut.'}
```

这种表述方式使得许多不同类别的语言任务能够以统一的方式表达，单个编码器-解码器架构无需任何任务专属参数即可处理它们。更多详情请参阅 [T5 论文 (Raffel et al. 2019)][t5_paper]。

为了实现可扩展的数据流水线和评估框架，我们使用从 [T5 库][t5_github] 中解耦拆分出的 [`SeqIO`](https://github.com/google/seqio)。一个 `seqio.Task` 将原始数据集、词表、预处理（如分词/tokenization）以及评估指标（如 [BLEU](https://aclanthology.org/P02-1040.pdf)）打包在一起，并提供一个 [`tf.data`](https://www.tensorflow.org/guide/data) 实例。

[T5 库][t5_github] 提供了许多在 [T5 论文][t5_paper] 中使用的 `seqio.Task`。在本示例中，我们使用 [wmt_t2t_ende_v003](https://github.com/google-research/text-to-text-transfer-transformer/blob/d81c0bab2a41b4d5dfbe4971de32f7d67df65f31/t5/data/tasks.py#L212)。

在训练或微调之前，需要先下载 ["wmt_t2t_translate"](https://www.tensorflow.org/datasets/catalog/wmt_t2t_translate) 数据集。

```sh
# 用于保存处理后数据集的数据目录，格式为 "gs://data_dir"。
TFDS_DATA_DIR="..."

# 确保数据集软件包为最新版本。
python3 -m pip install --upgrade tfds-nightly

# 预先下载数据集。
tfds build wmt_t2t_translate ${TFDS_DATA_DIR}
```

### 训练

运行训练作业使用 `t5x/train.py` 脚本：

```sh
# 模型目录，用于保存日志、检查点等，格式为 "gs://model_dir"。
MODEL_DIR="..."
T5X_DIR="..."  # 克隆 T5X 仓库所在的目录。
TFDS_DATA_DIR="..."

python3 ${T5X_DIR}/t5x/train.py \
  --gin_file="t5x/examples/t5/t5_1_1/examples/base_wmt_from_scratch.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --tfds_data_dir=${TFDS_DATA_DIR}
```

该训练运行的配置定义在 Gin 文件 [base_wmt_from_scratch.gin](t5x/examples/t5/t5_1_1/examples/base_wmt_from_scratch.gin) 中。[Gin-config](https://github.com/google/gin-config) 是一个基于依赖注入处理配置的库。在众多优势中，Gin 允许用户将自定义组件（如自定义模型）传递给 T5X 库，而无需修改核心库代码。[自定义组件](#自定义组件)章节展示了具体操作方式。

虽然核心库独立于 Gin，但 Gin 在我们提供的示例中起到了核心作用。因此，我们在 T5X 的背景下提供了简短的 [Gin 入门介绍][gin-primer]。所有配置都会写入 `MODEL_DIR` 下的 "config.gin" 文件中，这使得调试和复现实验变得极为简便。

除了 `config.json`，`model-info.txt` 文件还概述了模型参数（形状、轴名称、分区信息）以及优化器状态。



#### TensorBoard

为了在 [TensorBoard](https://www.tensorflow.org/tensorboard) 中监控训练，由于身份验证原因，在您自己的机器上（而*不是*在 TPU VM 内）启动 TensorBoard 会容易得多。因此，在您 SSH 进入 TPU VM 的命令行中，将 `logdir` 指向 `MODEL_DIR` 启动 TensorBoard：

```sh
# 注意：在您自己的本地机器上运行，不要在 TPU VM 上运行！
MODEL_DIR="..."  # 从 TPU VM 中复制。
tensorboard --logdir=${MODEL_DIR}
```

或者您也可以在 Colab 内部启动 TensorBoard。在 Colab 单元格中运行：

```python
from google.colab import auth
auth.authenticate_user()
```

以授权 Colab 访问 GCS 存储分区并启动 TensorBoard：

```python
%load_ext tensorboard
model_dir = "..."  # 从 TPU VM 中复制。
%tensorboard --logdir=model_dir
```


### 微调

我们可以通过从预训练模型中初始化来利用自监督预训练的优势。此处我们使用 T5.1.1 Base 检查点：

```sh
# 模型目录，用于保存日志、检查点等，格式为 "gs://model_dir"。
MODEL_DIR="..."

# 用于保存处理后数据集的数据目录，格式为 "gs://data_dir"。
TFDS_DATA_DIR="..."
T5X_DIR="..."  # 克隆 T5X 仓库所在的目录。

python3 ${T5X_DIR}/t5x/train.py \
  --gin_file="t5x/examples/t5/t5_1_1/examples/base_wmt_finetune.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --tfds_data_dir=${TFDS_DATA_DIR}
```

**注意：** 通过命令行参数传入字符串、字典、列表、元组值或 bash 变量时，必须用引号引起来。如果是字符串，则需要转义引号（`\"<string>\"`）。例如：
`--gin.utils.DatasetConfig.split=\"validation\"` 或
`--gin.MODEL_DIR=\"${MODEL_DIR}\"`。

Gin 可以非常方便地更改多项配置。例如，您可以修改 `partitioning.PjitPartitioner.num_partitions`（覆盖 [base_wmt_from_scratch.gin](t5x/examples/t5/t5_1_1/examples/base_wmt_from_scratch.gin) 中的值）以调整并行策略，并作为命令行参数传入：

```sh
--gin.partitioning.PjitPartitioner.num_partitions=8
```

### 评估

要运行离线评估（即不进行训练），可以使用 `t5x/eval.py` 脚本：

```sh
EVAL_OUTPUT_DIR="..."  # 写入评估输出的目录
T5X_DIR="..."  # 克隆 t5x 所在的目录，例如 ${HOME}"/t5x"。
TFDS_DATA_DIR="..."
CHECKPOINT_PATH="..."

python3 ${T5X_DIR}/t5x/eval.py \
  --gin_file="t5x/examples/t5/t5_1_1/examples/base_wmt_eval.gin" \
  --gin.CHECKPOINT_PATH=\"${CHECKPOINT_PATH}\" \
  --gin.EVAL_OUTPUT_DIR=\"${EVAL_OUTPUT_DIR}\" \
  --tfds_data_dir=${TFDS_DATA_DIR}
```


### 推理

要运行推理，可以使用 `t5x/infer.py` 脚本。此处我们使用相同的 `seqio.Task`，但对于推理，除了将目标特征与预测结果一同记录在 JSON 文件中之外，我们不使用它们：

```sh
INFER_OUTPUT_DIR="..."  # 写入推理输出的目录
T5X_DIR="..."  # 克隆 t5x 所在的目录，例如 ${HOME}"/t5x"。
TFDS_DATA_DIR="..."
CHECKPOINT_PATH="..."

python3 ${T5X_DIR}/t5x/infer.py \
  --gin_file="t5x/examples/t5/t5_1_1/examples/base_wmt_infer.gin" \
  --gin.CHECKPOINT_PATH=\"${CHECKPOINT_PATH}\" \
  --gin.INFER_OUTPUT_DIR=\"${INFER_OUTPUT_DIR}\" \
  --tfds_data_dir=${TFDS_DATA_DIR}
```

### 导出为 TensorFlow SavedModel

预训练模型可以导出为 TensorFlow SavedModel，并使用 [Optimized TensorFlow Runtime](https://cloud.google.com/vertex-ai/docs/predictions/optimized-tensorflow-runtime) 部署到 Vertex AI Prediction 服务。请注意，导出的模型无法在基于开源的 [TensorFlow Model Server](https://github.com/tensorflow/serving) 上运行。

```sh
T5X_DIR="..."  # 克隆 t5x 所在的目录，例如 ${HOME}"/t5x"。
CHECKPOINT_PATH="..."

BATCH_SIZE=None
BEAM_SIZE=1

# 如果计划在 NVIDIA A100 或更新的 GPU 上运行导出模型，请使用 'bfloat16'；
# 对于其他 GPU，请使用 'float32'。
ACTIVATION_DTYPE=bfloat16

# 版本号必须为数字。我们基于当前日期时间生成版本号。
VERSION=$(date +%Y%m%d%H%M%S)

NAME=t5x_base_${ACTIVATION_DTYPE}  # 模型名称。

# 模型导出路径。注意导出脚本会在模型名称后添加 _cpu 后缀。
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

详细参数定义请参阅 [export.gin](t5x/configs/runs/export.gin)。

您可以在 NVIDIA A100 40GB 上运行 XL 及更小的模型，在 NVIDIA A100 80GB 上运行 XXL 模型。

## 自定义组件

[翻译示例](#示例英语到德语翻译)使用了 T5X 提供的编码器-解码器模型以及来自 T5 库的数据集。本节展示如何使用您自己的数据集和模型并通过 Gin 进行传递。

### 示例：用户目录中的自定义数据集

在此示例中，目录结构如下，`${HOME}/dir1/user_dir` 代表包含自定义组件的用户目录：

```
${HOME}
└── dir1
    └── user_dir
        ├── t5_1_1_base_de_en.gin
        └── tasks.py
```

作为示例，我们定义一个新数据集。这里使用相同的翻译数据集，但将翻译任务的方向反转，即德译英而不是英译德。我们在 `tasks.py` 中定义此任务：

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

在 Gin 文件中，大部分设置与[英德翻译示例](#示例英语到德语翻译)中使用的设置相同，因此我们引入了该示例的 Gin 文件。为了使用刚才定义的 "wmt_t2t_de_en_v003" 任务，我们需要导入任务模块 "tasks.py"。请注意，我们使用的是相对于用户目录定义的相对路径，这将通过标志（flag）进行指定：

```py
# ${HOME}/dir1/user_dir/t5_1_1_base_de_en.gin
from __gin__ import dynamic_registration
import tasks  # 导入 dir1/user_dir/tasks.py 中定义的任务。

include "t5x-tmp/t5x/examples/t5/t5_1_1/examples/base_wmt_from_scratch.gin"
MIXTURE_OR_TASK_NAME = "wmt_t2t_de_en_v003"
```

最后，我们启动训练，将用户目录作为 `gin_search_paths` 参数传入，以便 Gin 文件和 Python 模块可以通过相对路径指定：

```sh
PROJECT_DIR=${HOME}"/dir1/user_dir"
T5X_DIR="..."  # 克隆 t5x 所在的目录。
TFDS_DATA_DIR="..."
MODEL_DIR="..."
export PYTHONPATH=${PROJECT_DIR}

python3 ${T5X_DIR}/t5x/train.py \
  --gin_search_paths=${PROJECT_DIR} \
  --gin_file="t5_1_1_base_de_en.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --tfds_data_dir=${TFDS_DATA_DIR}
```

## 检查点

### 原生检查点

我们已经以原生 T5X 格式发布了许多原始 T5 模型及其变体的检查点，以实现最高效率。
请参阅[完整列表](https://github.com/google-research/t5x/blob/main/docs/models.md)，包括匹配的 Gin 配置文件。

这些检查点是从公开的 [Mesh TensorFlow 检查点](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md#t511)转换而来的。


### 与 Mesh TensorFlow 检查点的兼容性
使用 [T5 库][t5_github] 训练的 Mesh TensorFlow 检查点可以直接加载到 T5X 中。例如，我们可以通过更改 `INIT_CHECKPOINT` Gin 宏从 MTF 检查点初始化重新运行微调示例：

```sh
# 模型目录，用于保存日志、检查点等，格式为 "gs://model_dir"。
MODEL_DIR="..."

# 用于保存处理后数据集的数据目录，格式为 "gs://data_dir"。
TFDS_DATA_DIR="..."
T5X_DIR="..."  # 克隆 T5X 仓库所在的目录。

python3 ${T5X_DIR}/t5x/train.py \
  --gin_file="t5x/examples/t5/t5_1_1/examples/base_wmt19_ende_train.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.MIXTURE_OR_TASK_NAME=\"wmt_t2t_ende_v003\" \
  --gin.INIT_CHECKPOINT=\"gs://t5-data/pretrained_models/t5.1.1.base/model.ckpt-1000000\" \
  --tfds_data_dir=${TFDS_DATA_DIR}
```

请注意，对于大型模型，如果使用较重的模型并行，直接从 Mesh TensorFlow 检查点恢复可能会效率低下。这是因为每个主机首先加载整个模型副本，然后仅保留模型并行规范所指定的相应切片。如果您有经常运行的 Mesh TensorFlow 检查点，我们建议使用 [convert_tf_checkpoint 脚本](t5x/scripts/convert_tf_checkpoint.py)将检查点转换为 T5X 原生格式。


## 引用 T5X
引用 T5X 请使用以下 BibTeX 条目：

```
@article{roberts2022t5x,
  url = {https://arxiv.org/abs/2203.17189},
  author = {Roberts, Adam and Chung, Hyung Won and Levskaya, Anselm and Mishra, Gaurav and Bradbury, James and Andor, Daniel and Narang, Sharan and Lester, Brian and Gaffney, Colin and Mohiuddin, Afroz and Hawthorne, Curtis and Lewkowycz, Aitor and Salcianu, Alex and van Zee, Marc and Austin, Jacob and Goodman, Sebastian and Soares, Livio Baldini and Hu, Haitang and Tsvyashchenko, Sasha and Chowdhery, Aakanksha and Bastings, Jasmijn and Bulian, Jannis and Garcia, Xavier and Ni, Jianmo and Chen, Andrew and Kenealy, Kathleen and Clark, Jonathan H. and Lee, Stephan and Garrette, Dan and Lee-Thorp, James and Raffel, Colin and Shazeer, Noam and Ritter, Marvin and Bosma, Maarten and Passos, Alexandre and Maitin-Shepard, Jeremy and Fiedel, Noah and Omernick, Mark and Saeta, Brennan and Sepassi, Ryan and Spiridonov, Alexander and Newlan, Joshua and Gesmundo, Andrea},
  title = {Scaling Up Models and Data with $\texttt{t5x}$ and $\texttt{seqio}$},
  journal={arXiv preprint arXiv:2203.17189},
  year = {2022},
}
```


## 说明
这并非官方支持的 Google 产品。

[t5_paper]: https://arxiv.org/abs/1910.10683
[t5_github]: https://github.com/google-research/text-to-text-transfer-transformer
[gin-primer]: docs/usage/gin.md

---

> **免责声明**：本文档由社区志愿者进行本地化翻译，旨在促进跨语言的技术交流与学习。如翻译内容与[英文原版](README.md)存在出入，请以英文原版为准。
