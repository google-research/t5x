# Mixture of Experts


This repo contains overrides and configs for training sparse Mixture of Experts
(MoE) models with T5X. The existing setups and examples all use [Flaxformer](https://github.com/google/flaxformer).

## Training standard MoE architectures

If you are looking train a T5X variant of a popular Mesh Tensorflow MoE model 
(e.g. [Switch Transformer](https://arxiv.org/abs/2101.03961) or [Sparsely-Gated Mixture-of-Experts](https://arxiv.org/abs/1701.06538)) or adapt existing
MoE models, then the easiest way to get started is to plug one of the
[(Flaxformer) model gin configs](https://github.com/google/flaxformer/tree/main/flaxformer/t5x/configs/moe/models)
into the [T5X Quickstart guide](https://github.com/google-research/t5x). To customize the default MoE models, you can override aspects of the underlying [(Flaxformer) architecture gin config](https://github.com/google/flaxformer/blob/main/flaxformer/t5x/configs/moe/architectures/moe.gin).

## Using MoE in your existing model

Alternatively, if you already have your own existing T5X/Flaxformer model
architecture and wish to add MoE layers, you can directly use the
[Flaxformer MoeLayer](https://github.com/google/flaxformer/blob/b725bd2a51d70e866d819c92de166fbf24425e6a/flaxformer/architectures/moe/moe_layers.py#L67).
Currently, the MoeLayer is constrained to use
[Flaxformer MlpBlock(s)](https://github.com/google/flaxformer/blob/b725bd2a51d70e866d819c92de166fbf24425e6a/flaxformer/components/dense.py#L185)
as experts. As a point of reference: MoeLayer(s) are integrated with the Flaxformer T5
architecture through the
[SparseEncoder](https://github.com/google/flaxformer/blob/b725bd2a51d70e866d819c92de166fbf24425e6a/flaxformer/architectures/moe/moe_architecture.py#L36)
and
[SparseDecoder](https://github.com/google/flaxformer/blob/b725bd2a51d70e866d819c92de166fbf24425e6a/flaxformer/architectures/moe/moe_architecture.py#L162).
These classes allow us to interleave sparse MoE and dense MLP blocks through the
`sparse_layout` attribute.

## Expert routing mechanisms

A number of routing mechanisms are supported:

*   Switch routing (or top-1 "tokens choose" routing) based on the
    [Switch Transformer](https://arxiv.org/abs/2101.03961)
*   General Top-k "tokens choose" routing of the form used in
    [Sparsely-Gated Mixture-of-Experts](https://arxiv.org/abs/1701.06538),
    [Vision MoE](https://arxiv.org/abs/2106.05974),
    [Designing Effective Sparse Expert Models](https://arxiv.org/abs/2202.08906)
    and many other MoE works
*   "Experts choose" routing introduced in
    [Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368)

See the
[Flaxformer router codebase](https://github.com/google/flaxformer/blob/main/flaxformer/architectures/moe/routing.py) for details.

