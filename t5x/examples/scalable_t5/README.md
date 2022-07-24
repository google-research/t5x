# Scalable T5

This directory is very similar to the vanilla T5X "T5" example, but demonstrates
a host of techniques needed to scale model training to giant models run on
large TPU or GPU cluster environments using XLA's SPMD capabilities.  See the
notes for the main "t5" example for general details on setup and execution.

## Intermediate variable annotations

In larger models, with multi-axis model parallelism, it is typically necessary
to provide additional constraint annotations beyond those for the input and
output parameters for a function.  We do this using a special version of the
`pjit` annotation function `with_sharding_constraint` that uses _logical_ axis
names instead of raw mesh axes.  This allows us to avoid tightly coupling a
specific partitioning plan to the model code itself.  Instead, we merely need
to annotate the axis names used in the model in a coherent scheme, and later
map these logical axes to the physical mesh axes using a small set of rules.
Example usage can be seen in `network.py`.

## Scan over layers

One challenge with giant models is the increasing amount of compilation time
required to handle extremely large layer stacks in XLA.  At the size of a full
TPU pod this compile time cost can become quite extreme.  To remedy this,
instead of handing the compiler a huge stack of unrolled layers, we can use
native XLA control flow constructs to simplify the computational graph given
from JAX.  For giant models this can drop the compile time from hour(s) to
minutes, and even at base-scale can be roughly 5x faster.

In this case, we want to use the [XLA While Op](xla-while) via JAX's
[scan](jax-scan) control flow construct to express the idea that we're looping
over identically-defined layers when using a deep transformer network.  We do
this via a custom Flax version of scan called `scan_with_axes` that also handles
the parameter logical axis name metadata needed for partitioning.

## Rematerialization / Checkpointing

"Rematerialization" or "checkpointing" is a technique for trading off compute
time for lower peak memory utilization when performing reverse-mode automatic
differentiation.  JAX offers several different default rematerialization
"policies" that dictate which kinds of intermediate values are preserved from
the forward-pass to the backwards-pass calculation, and which are discarded to
be recomputed anew in the backwards-pass.


[xla-while]: https://www.tensorflow.org/xla/operation_semantics#while
[jax-scan]: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html
