# Data, Model, and Activation Partitioning


## What is partitioning?

Partitioning (sometimes referred to sharding) is a core feature of T5X, provided
primarily via the
[`jax.pjit`](https://github.com/google/jax/tree/main/jax/experimental/pjit.py) backend.
With partitioning, you can shard data, model parameters, and intermediate
activations across the hardware mesh to reduce per-device memory requirements
and increase parallelism.

This allows you to take advantage of additional TPU (or GPU) devices to train or
infer from models too big to fit in the memory of a single device, use larger
batch sizes, and/or train faster.

## Partitioning with named axes

With T5X, the recommended way of configuring partitioning is to use the
`PjitPartitioner`, which relies on the use of logical axis name
annotations (for parameters and activations).

*Logical axis names* are a user-configured shorthand for grouping
axes/dimensions of various arrays (parameters or activations) in a model
implementation. For example, one might refer to the axes of the inputs to a
model as `('batch', 'length', 'vocab')`. If the parameters of the embedding
matrix are labelled `('vocab', 'embed')` then the activations following
embedding should be named `('batch', 'length', 'embed')`. A list of canonical
logical axis names can be found [below](#canonical-logical-axis-names).

For `jax.pjit` to know how to partition these arrays across the hardware, the
logical axis names must be mapped to physical axes of the accelerator (TPU/GPU)
mesh. In T5X, the two primary hardware axes are named `'data'` and `'model'`,
referring to the default mappings for data- and model-parallelism, respectively.
Names notwithstanding, you are free to map model parameters/activations across
the "data" axis. In fact, this is what is done in "2D" parameter/activation
sharding. See [`t5x.partitioning.standard_logical_axis_rules`][standard-rules]
and the [example mappings below](#example-configurations) to see how this works
in practice.

The following subsections explain how to configure partitioning in T5X using
named axes.

### Configuring `PjitPartitioner`

[`PjitPartitioner`][pjit-partitioner] has three primary
constructor arguments, typically set via gin: `num_partitions`,
`model_parallel_submesh`, and `logical_axis_rules`.

`num_partitions` and `model_parallel_submesh` provide two mutually-exclusive
methods of specifying the submesh of devices to use for model (parameter and
activation) partitioning.

`model_parallel_submesh (Tuple[int, int, int, int])` is a 4-tuple that specifies
the `(x, y, z, c)` submesh model-parallel device tile, an axis of accelerator
parallelism orthogonal to data parallelism. Array axes in a model's parameters
or activations can be sharded over this submesh using axis rules (see
`logical_axis_rules`) that map them to `'model'`. The effective number of model
sub-partitions is equal to `np.prod(model_parallel_submesh)` and must evenly
divide the total number of devices (i.e., `jax.device_count() %
np.prod(model_parallel_submesh) == 0`). The rest of the TPU mesh is the data
parallel submesh, providing `jax.device_count() //
np.prod(model_parallel_submesh)` partitions. It is used for data (batch)
parallelism and to shard other array axes that are mapped to `'data'`.

Alternatively, `num_partitions (int)` accepts an integer that specifies the size
of the model parallel submesh to be automatically selected for the current
topology.

The third key argument is `logical_axis_rules (Sequence[Tuple[str,
Optional[str]]])`. This argument accepts an priority-ordered sequence of KV
tuples that maps logical axis names to hardware resources, using `'model'` and
`'data'`as a convention for the two primary hardware axes. Therefore, each
logical axis can be mapped to one of:

*   `None` to disable sharding, and thus be fully-replicated,
*   `'model'` to shard across the model-parallel submesh, or
*   `'data'` to shard across the data-parallel submesh.

Note that the same key can be mapped to multiple values. For each array,
mappings are applied in priority order. If a hardware resource has already been
assigned in to a different axis and multiple keys exist, a latter mapping may be
used.

For example, consider the following set of logical axis rules:

```py
[
  ('head', 'model'),
  ('embed', 'model'),
  ('embed', 'data'),
  ('vocab', 'model'),
]
```

For an array with logical axes `('embed', 'head')`, `'head'` will first be
mapped to `'model'`, since it comes first in the priority list. Next, `'embed'`
will be mapped to `'data'`, since `'model'` has already been used. However, an
array with logical axes `('vocab', 'embed')` will receive the mapping `(None,
'model')` since `'embed'` has a higher priority than `'vocab'`.

T5X provides the default
[`t5x.partitioning.standard_logical_axis_rules`][standard-rules] function to
generate canonical logical axis rule sets depending on how many mesh dimensions
you wish to shard activations and parameters (defaults to 1), with the
assumption that you are using
[canonical logical axis names](#canonical-logical-axis-names).

### Configuring logical axis annotations within the model

In order for your model to be partitionable by the `PjitPartitioner`,
you must apply logical axis name annotations to your model parameters and
activations.

These annotations can be provided through the utilities in
[`flax.linen.partitioning`](https://github.com/google/flax/tree/main/flax/linen/partitioning.py).

Instead of calling `self.param` to create parameters within your model
implementation, you should use the `flax.linen.partitioning.param_with_axes()`
api from Flax to communicate axis names for each parameter.

```py
from flax.linen import partitioning

scale = partitioning.param_with_axes(
    'scale', self.scale_init, (features,), jnp.float32, axes=('embed',))
```

Similarly, instead of calling `jax.pjit.with_sharding_constraint` or
`t5x.partitioning.with_sharding_constraint` to specify hardware axes for
activation partitioning, you should use
`flax.linen.partitioning.with_sharding_constraint` providing logical axis names.

```py
from flax.linen import partitioning

...
output = jnp.dot(x, embedding)
output = with_sharding_constraint(output, ('batch', 'length', 'embed'))
return output
```

See the [Minimal](https://github.com/google-research/t5x/tree/main/t5x/examples/) and
[Flaxformer](https://github.com/google/flaxformer/tree/main/flaxformer/architectures/t5/)
T5 implementations for examples of how these annotations are applied in
practice.

### Overriding Axis Names from External Codebase

You may wish to incorporate Flax modules from an external codebase into your
model implementation that uses `self.param` instead of
`flax.linen.partitioning.param_with_axes`, or that may use axis names that are
incompatible with your codebase.

To deal with this situation, we provide the `utils.override_params_axes_names`
helper function. This helper can be called at the end of
`Model.get_initial_variables` to apply a priority-ordered mapping from regex
patterns (fully matching parameter names) to tuples containing string logical
axis names to replace model-derived names.

For example, the following configuration provides logical axis names for an
external module called 'external_mlp' used in every layer of model's encoder,
without modifying any other modules:

```py
class MyCustomEncoderDecoderModel(models.EncoderDecoderModel):

  def get_initial_variables():
    self,
    rng: jnp.ndarray,
    input_shapes: Mapping[str, Array],
    input_types: Optional[Mapping[str, jnp.dtype]] = None
  ) -> flax_scope.FrozenVariableDict:
    initial_variables = super().get_initial_variables(
        rng=rng, input_shapes=input_shapes, input_types=input_types)
    return utils.override_params_axes_names(
        initial_variables,
        params_axes_names_override=[
            ('encoder/layer_\\d/external_mlp/kernel':, ('embed', 'mlp')),
            ('encoder/layer_\\d/external_mlp/bias':, ('mlp',)),
        ])
```

Note: It is not possible to add or modify activation partitioning in an external
module.

### Canonical Logical Axis Names

We recommend you use logical axis names from the following list for
compatibility with
[`t5x.partitioning.standard_logical_axis_rules`][standard-rules]. If you wish to
use a non-canonical axis name, you will need to pass a custom set of axis rules
to the `PjitPartitioner`.

*   `"embed"`: This is the common "activation_dim" in the network, first emitted
    by the embeding layer.
*   `"heads"`: Number of heads for attention / relative position biases.
*   `"kv"`: For query/key/value hidden dimensions of each head.
*   `"joined_kv"`: For "heads * kv" fused dimension of attention matrices, when
    the kernel is reshaped such that "heads" and "kv" are packed in the same
    dimension.
*   `"mlp"`: Intermediate dimension of the feed-forward layer.
*   `"vocab"`: For embeddings, the input/output vocabulary size.
*   `"mlp_activations"`: For fused MLP matrices that have a dimension for the
    activation function index.
*   `"stack"`: For KV and QKV fused attention implementations, the manual
    parameter-fusion stacked dimension.
*   `"abspos_buckets"` / `"relpos_buckets"`: The dimension for positional bias
    buckets.

## Example configurations

You can override the default 1D sharding configuration (e.g., in gin) by
modifying the arguments to
[`t5x.partitioning.standard_logical_axis_rules`][standard-rules]. For example,
for full (parameter and activation) 2D partitioning, you can set:

```py
from t5x import partitioning

train_script.train:
  partitioner = @partitioning.PjitPartitioner()

partitioning.PjitPartitioner:
  num_partitions = 1
  logical_axis_rules= @partitioning.standard_logical_axis_rules()

partitioning.standard_logical_axis_rules:
  activation_partitioning_dims = 2
  parameter_partitioning_dims = 2
```

Alternatively, you can set the rules manually, experimenting with some of the
following options:

*   Only data parallelism:

```py
partitioning.PjitPartitioner.logical_axis_rules = [
    ('batch', 'data'),
    ('vocab', None),
    ('embed', None),
    ('mlp', None),
    ('heads', None),
    ('kv', None),
    ('joined_kv', None),
    ('relpos_buckets', None),
    ('abspos_buckets', None),
    ('length', None),
    ('layers', None),
    ('stack', None),
    ('mlp_activations', None),
]
```

*   Data parallel with parameter gather, aka
    [ZeRO-3](https://arxiv.org/abs/1910.02054), aka "2D parameter partitioning
    with trivial MP submesh":

```py
partitioning.PjitPartitioner.logical_axis_rules = [
    ('batch', 'data'),
    # all weight matrices have this axis; activations already shard it along 'data'
    ('embed', 'data'),
    ('vocab', None),
    ('mlp', None),
    ('heads', None),
    ('kv', None),
    ('joined_kv', None),
    ('relpos_buckets', None),
    ('abspos_buckets', None),
    ('length', None),
    ('layers', None),
    ('stack', None),
    ('mlp_activations', None),
]
```

*   Data and model parallel with replicated activations, aka
    [Megatron](https://arxiv.org/abs/1909.08053), aka "1D parameter
    partitioning":

```py
partitioning.PjitPartitioner.logical_axis_rules = [
    ('batch', 'data'),
    ('mlp', 'model'),
    ('heads', 'model'),
    ('vocab', 'model'),
    ('embed', None),
    ('kv', None),
    ('joined_kv', None),
    ('relpos_buckets', None),
    ('abspos_buckets', None),
    ('length', None),
    ('layers', None),
    ('stack', None),
    ('mlp_activations', None),
]
```

*   Data and model parallel with sharded activations, perhaps the same as
    [Optimus](https://arxiv.org/abs/2104.05343), aka "1D parameter partitioning
    with 2D activation partitioning":

```py
partitioning.PjitPartitioner.logical_axis_rules = [
    ('batch', 'data'),
    ('mlp', 'model'),
    ('heads', 'model'),
    ('vocab', 'model'),
    # shard remaining activations; weight matrices already have axes mapped to 'model'
    ('embed', 'model'),
    ('kv', None),
    ('joined_kv', None),
    ('relpos_buckets', None),
    ('abspos_buckets', None),
    ('length', None),
    ('layers', None),
    ('stack', None),
    ('mlp_activations', None),
]
```

*   Full 2D sharding, aka [GShard](https://arxiv.org/abs/2105.04663); aka "2D
    parameter + activation partitioning":

```py
partitioning.PjitPartitioner.logical_axis_rules = [
    ('batch', 'data'),
    ('mlp', 'model'),
    ('heads', 'model'),
    ('vocab', 'model'),
    # shard both activations and weight matrices on the remaining available axis
    ('embed', 'model'),
    ('embed', 'data'),
    ('kv', None),
    ('joined_kv', None),
    ('relpos_buckets', None),
    ('abspos_buckets', None),
    ('length', None),
    ('layers', None),
    ('stack', None),
    ('mlp_activations', None),
]
```

<!---TODO(b/214235006): Use symbol reference instead of line number+rcl.-->

[standard-rules]: https://github.com/google-research/t5x/tree/main/t5x/partitioning.py?l=438&rcl=421294093
[pjit-partitioner]: https://github.com/google-research/t5x/tree/main/t5x/partitioning.py?l=674&rcl=421291812
