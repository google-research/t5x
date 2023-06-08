# Data, Model, and Activation Partitioning


TL;DR: The recommended way of specifying partitions in T5X.

**Partitioning** is the process dividing and replicating machine learning *model
parameters*, *activations*, and *data* across the accelerator devices (TPU/GPU)
in order to:

*   Train and infer from models too large to fit in the memory of a single
    device
*   Use extremely large batch sizes
*   Train faster

## How to Partition

Partitioning in T5X is configured in two steps:

1.  Specify logical axes names for parameter and activation array dimensions
2.  Map the logical names to the physical axes of the accelerator mesh

Let's take a closer look at each of these steps.

**Note:** In T5X, partitioning is primarily provided through the
[jax.pjit][pjit] backend via `PjitPartitioner` using the Gin configuration
framework.

### Specify logical axis names

**Logical axis names** are a user-configured shorthand for grouping *axes* (aka
*dimensions*) of either parameter or activation arrays in a model
implementation.

For example, you could refer to the axes of the inputs to a model as `('batch',
'length', 'vocab')`. If the parameters of the embedding matrix are labelled
`('vocab', 'embed')` then the activations following embedding should be named
`('batch', 'length', 'embed')`.

**Description**      | **Logical Axis Names**
-------------------- | ------------------------------
Inputs to model      | `('batch', 'length', 'vocab')`
Embedding parameters | `('vocab', 'embed')`
Activations          | `('batch', 'length', 'embed')`

**How to configure logical axis names**

Logical axis annotations can be provided through the utilities in
[`flax.linen.partitioning`][lan].

Instead of calling `self.param` to create parameters within your model
implementation, use the `flax.linen.partitioning.param_with_axes` API to
communicate axis names for each parameter.

```py
from flax.linen import partitioning

scale = partitioning.param_with_axes(
    'scale', self.scale_init, (features,), jnp.float32, axes=('embed',))
```

For an example in context, see [`layers.py`][param_with_axes].

Tip: We recommend you use the *canonical* logical axis names listed
[below](#canonical-logical-axis-names).

To specify the logical axes for *activation partitioning*, provide the logical
axes names to `flax.linen.partitioning.with_sharding_constraint` (instead of
using `jax.pjit.with_sharding_constraint` or
`t5x.partitioning.with_sharding_constraint`).

```py
from flax.linen import partitioning

...
output = jnp.dot(x, embedding)
output = with_sharding_constraint(output, ('batch', 'length', 'embed'))
return output
```

### Map logical names to device

For `jax.pjit` to know how to partition these arrays across the hardware, the
logical axis names must be mapped to the physical axes of the accelerator mesh.

**Note:** A *mesh* is an n-dimensional array of TPU (or GPU) processors,
connected by a network. The TPUv3 processor is limited to 2D meshes. The TPUv4
processor can handle 3D meshes.

In T5X, the two primary *hardware* axes are named `'data'` and `'model'`,
referring to the default mappings for data- and model-parallelism.

> **Note:** You are actually free to map model parameters or activations across
> the `'data'` axis. In fact, this is what is done in 2D parameter/activation
> sharding. To see how this works in practice, see:
>
> *   [The example mappings](#example-configurations) below
> *   [`t5x.partitioning.standard_logical_axis_rules`][standard-rules]
>     implementation


#### Configuring `PjitPartitioner`

`PjitPartitioner` has three primary constructor arguments:

*   `model_parallel_submesh`
*   `num_partitions`
*   `logical_axis_rules`

The `model_parallel_submesh` and `num_partitions` arguments provide two
mutually-exclusive methods of specifying the submesh of devices to use for model
partitioning. As a rule of thumb:

*   Use `model_parallel_submesh` when you want to specify how the logical names
    are mapped to the device
*   Use`num_partitions` for an automatic mapping

**Using `model_parallel_submesh`**

The `PjitPartitioner` constructor argument that provides the most control is:

```
model_parallel_submesh(Tuple[int, int, int, int])
```

It is a 4-tuple that specifies the `(x, y, z, c)` *model-parallel* submesh–an
axis of accelerator parallelism orthogonal to data parallelism. Axes in a
model's parameter or activation arrays can be sharded over this submesh using
axis rules that map them to `'model'`.

**Note:** The effective number of model subpartitions is equal to
`np.prod(model_parallel_submesh)` and must evenly divide the total number of
devices. Specifically: \
`jax.device_count() % np.prod(model_parallel_submesh) == 0`.

The rest of the TPU mesh is the *data parallel* submesh, providing
`jax.device_count() // np.prod(model_parallel_submesh)` partitions. It is used
for data (aka *batch*) parallelism and to shard other array axes that are mapped
to `'data'`.

**Using `num_partitions`**

Alternatively,

```
num_partitions(int)
```

accepts an integer that specifies the size of the model parallel submesh to be
*automatically* selected for the current topology.

**Using `logical_axis_rules`**

The third key argument is

```
logical_axis_rules(Sequence[Tuple[str, Optional[str]]])
```

This argument accepts a priority-ordered sequence of key-value (KV) tuples.
These tuples map the logical axis names to hardware resources, using `'model'`
and `'data'` as the two primary hardware axes. Specifically, each logical axis
can be mapped to one of:

*   `None` to disable sharding, and thus be fully-replicated
*   `'model'` to shard across the model-parallel submesh
*   `'data'` to shard across the data-parallel submesh

The same key can be mapped to multiple values. For each array, mappings are
applied in priority order. If a hardware resource has already been assigned in
to a different axis and multiple keys exist, a latter mapping may be used.

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

T5X provides the `t5x.partitioning.standard_logical_axis_rules()` function to
generate canonical logical axis rule sets depending on how many mesh dimensions
you wish to shard. This assumes that you are using
[canonical logical axis names](#canonical-logical-axis-names).

For details, see
[`t5x.partitioning.standard_logical_axis_rules()`][standard-rules].

## Other Stuff

### Overriding axis names from an external codebase

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
external module called 'external_mlp' used in every layer of the model's
encoder, without modifying any other modules:

```py
class MyCustomEncoderDecoderModel(models.EncoderDecoderModel):

  def get_initial_variables(
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

**Note:** It is not possible to add or modify activation partitioning in an
external module.

### Canonical logical axis names

Use the following logical axis names to be compatible with
[`t5x.partitioning.standard_logical_axis_rules`][standard-rules]:

| Logical Axis Name    | Description                                          |
| -------------------- | ---------------------------------------------------- |
| `"embed"`            | The common "activation_dim" in the network, first    |
:                      : emitted by the embedding layer.                      :
| `"heads"`            | Number of heads for attention/relative position      |
:                      : biases.                                              :
| `"kv"`               | For query/key/value hidden dimensions of each head.  |
| `"joined_kv"`        | For "heads * kv" fused dimension of attention        |
:                      : matrices, when the kernel is reshaped such that      :
:                      : "heads" and "kv" are packed in the same dimension.   :
| `"mlp"`              | Intermediate dimension of the feed-forward layer.    |
| `"vocab"`            | For embeddings, the input/output vocabulary size.    |
| `"mlp_activations"`  | For fused MLP matrices that have a dimension for the |
:                      : activation function index.                           :
| `"stack"`            | For KV and QKV fused attention implementations, the  |
:                      : manual parameter-fusion stacked dimension.           :
| `"abspos_buckets"` / | The dimension for positional bias buckets.           |
: `"relpos_buckets"`   :                                                      :

If you wish to use a non-canonical axis name, you will need to pass a custom set
of axis rules to the `PjitPartitioner`.

--------------------------------------------------------------------------------

## Example configurations

### Automatic - Full 2D partitioning

You can override the default 1D sharding configuration by modifying the
arguments to [`t5x.partitioning.standard_logical_axis_rules`][standard-rules].
For example, for full parameter and activation 2D partitioning you can set:

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

### Manual configurations

Alternatively, you can manually set the rules, experimenting with some of the
following options:

*   [Data-only parallelism](#data-only-parallelism)
*   [Data parallel with parameter gather](#data-parallel-with-parameter-gather)
*   [Data and model parallel with replicated activations](#data-and-model-parallel-with-replicated-activations)
*   [Data and model parallel with sharded activations](#data-and-model-parallel-with-sharded-activations)
*   [Full 2D sharding](#full-2d-sharding)

#### Data-only parallelism

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

#### Data parallel with parameter gather

An example of 2D parameter partitioning with trival MP submesh, such as
[ZeRO-3][ZeRO-3].

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

#### Data and model parallel with replicated activations

An example of 1D parameter partitioning, such as [Megatron][megatron].

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

#### Data and model parallel with sharded activations

An example of 1D parameter partitioning with 2D activation partitioning, such as
[Optimus][optimus].

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

#### Full 2D sharding

An example of 2D parameter and activation partitioning, such as
[GShard][gshard].

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

## Recommended reading

[Basic model and data partitioning for inference in P5X](https://docs.google.com/document/d/1bU8IuufbgkY0Wg8okyrEPnu3S5rfYGqVPMbUmeorFVo/edit)
by brandonthorpe@, luyaoxu@

<!-- Reference links -->

<!---TODO(b/214235006): Use symbol reference instead of line number+rcl.-->

[ZeRO-3]: https://arxiv.org/abs/1910.02054
[gshard]: https://arxiv.org/abs/2105.04663
[flaxformer]: https://github.com/google/flaxformer/tree/main/flaxformer/architectures/t5/
[lan]: https://github.com/google/flax/tree/main/flax/linen/partitioning.py
[megatron]: https://arxiv.org/abs/1909.08053
[minimal]: https://github.com/google-research/t5x/blob/main/t5x/examples/
[optimus]: https://arxiv.org/abs/2104.05343
[param_with_axes]: https://github.com/google-research/t5x/blob/main/t5x/examples/t5/layers.py;rcl=427300354;l=462
[pjit]: https://github.com/google/jax/tree/main/jax/experimental/pjit.py
[standard-rules]: https://github.com/google-research/t5x/blob/main/t5x/partitioning.py?l=438&rcl=421294093
