# T5X


Note: T5X is community-supported since ~2023. For critical use cases, consider
using libraries like TuneLab (go/tunelab) and Gemax Prod (go/gemax-prod). See
https://github.com/google-research/text-to-text-transfer-transformer/blob/main/README.mdx-to-gemax-prod for useful tips on transitioning.

## Overview

T5X is a modular, composable, research-friendly framework for high-performance,
configurable, self-service training, evaluation, and inference of sequence
models (starting with language) at many scales.

It is essentially a new and improved implementation of the
[T5 codebase](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/README.md) (based on Mesh TensorFlow) in JAX and Flax. To learn
more, see the [T5X Paper](https://arxiv.org/abs/2203.17189).

## Getting Started

Here are some quick tutorials to help you get started with common use-cases on
T5X:

#### [Introductory Colabs](tutorials.md)

If you are new to T5X, we recommend starting with our introductory Colab series,
which introduces core concepts of both T5X and SeqIO. More colabs will be added
to this series regularly!

#### [Fine-tuning a model](usage/finetune.md)

This tutorial outlines the steps to fine-tune an existing pre-trained model with
T5X on common downstream Tasks/Mixtures available on SeqIO. This is one of the
simplest and most common use cases of T5X. If you're new to T5X, this tutorial
is the recommended starting point.

#### [Running evaluation on a model](usage/eval.md)

This tutorial outlines the steps to evaluate a model with T5X on downstream
Tasks/Mixtures defined in SeqIO.

#### [Running inference on a model](usage/infer.md)

This tutorial outlines the steps to run inference on a model with T5X.

#### [Training a model from scratch](usage/pretrain.md)

This tutorial outlines the steps to pretrain a model with T5X on Tasks/Mixtures
defined in SeqIO.

#### [Gin Primer](usage/gin.md)

This tutorial provides a quick introduction to Gin, a lightweight configuration
framework for Python that is used to configure training, eval and inference jobs
on T5X.

#### [Partitioning Primer](usage/partitioning.md)

This tutorial provides background on what model and data partitioning are and
how it can be configured in T5X.

#### [Metrics Overview](usage/metrics.md)

This tutorial provides an overview of how metrics can be used and customized to
evaluate T5X models.

