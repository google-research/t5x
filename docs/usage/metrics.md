# Metrics Overview


## Introduction

T5X provides a flexible and customizable library for managing metrics. Metrics
in T5X rely on [CLU](https://github.com/google/CommonLoopUtils/blob/main/README.md), which broadly provides utilities for writing
training loops but specifically provides metric libraries that are extended by
T5X.


NOTE: This document currently only applies to train and 'train_eval' metrics,
not to 'infer_eval' metrics, which are implemented using SeqIO. We plan to unify
these three in the future.

## Metrics and Writers

CLU provides `Metric` and `MetricWriter` classes. Full details are provided in
[go/clu-metrics](https://github.com/google/CommonLoopUtils/blob/main/README.md-metrics), but a simplified summary will suffice for our
purposes.

[`clu.metrics.Metric`](https://github.com/google/CommonLoopUtils/tree/main/clu/metrics.py?q=symbol:%5CbMetric%5Cb)
provides an abstract interface for metrics. The interface can be simply
represented by the following:

```py
class Metric:

  @classmethod
  def from_model_output(cls, *args, **kwargs) -> Metric:
    # creates a Metric from model output (i.e. loss arrays).
    pass

  def merge(self, other) -> Metric:
    # combines a Metric from the current step with that of a previous step.
    pass

  def compute(self) -> Union[float, np.ndarray]:
    # computes the writable value of a metric (as a float, array, etc.)
    pass

  def compute_value(self) -> clu.values.Value:
    # computes metric as a writable type (Scalar, Image, Histogram, etc.)
    # defaults to Scalar
    return clu.values.Scalar(self.compute())
```

`Metric`s can then be extended into concrete representations, such as Sum:

```py
@flax.struct.dataclass
class Sum(Metric):

  total: float

  @classmethod
  def from_model_output(cls, values: np.ndarray) -> Metric:
    return cls(total=np.sum(values))

  def merge(self, other) -> Metric:
    return type(self)(total = self.total + other.total)

  def compute(self) -> Union[float, array]:
    return self.total

  # If Metric is non-Scalar, return a different Value type as needed.
  def compute_value() -> clu.values.Value:
    return clu.values.Scalar(self.compute())
```

We will elaborate in more detail [below](#a-metric-example) on how Metrics are
practically used in T5X.

In addition to CLU provided metrics like Average and Accuracy, T5X provides a
few specialized metrics, like TimeRate and AveragePerStep. A full list of CLU
metrics is provided at
[clu/metrics.py](https://github.com/google/CommonLoopUtils/tree/main/clu/metrics.py) while T5X metrics
are listed in [t5x/metrics.py](https://github.com/google-research/t5x/tree/main/t5x/metrics.py). We
will elaborate on specialized metrics like TimeRate and AveragePerStep
[below](#special-t5x-metrics).

Given a constructed `Metric` object, we can use a `MetricWriter` to write it in
a readable form to some destination.

`MetricWriter` again has a simple interface, represented by the following

```py
class MetricWriter:

  def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
    pass

  def write_images(self, step: int, images: Mapping[str, Array]):
    pass

  ...
```

A `MetricWriter` implements a specific write method for each type, including
scalars, images, audios, texts, histograms, and hyperparameters.

CLU provides a convenience method for easily writing metric values of diverse
types,
[`clu.metric_writers.write_values`](https://github.com/google/CommonLoopUtils/tree/main/clu/metric_writers/utils.py?q=symbol:%5Cbwrite_values%5Cb).

```
def write_values(writer: MetricWriter, step: int,
                 metrics: Mapping[str, Union[values.Value, values.ArrayType,
                                             values.ScalarType]]):
```

Given a mapping of string to
[`clu.values.Value`](https://github.com/google/CommonLoopUtils/tree/main/clu/values.py?q=symbol:%5CbValue%5Cb),
the method automatically calls the writer's appropriate write method. Such a
mapping can be easily obtained by calling `metric.compute_value()`.

`MetricWriter` is subclassed by several specific writers, which enable writing
to the console, TF summary files, XManager, and others. See
[source](https://github.com/google/CommonLoopUtils/tree/main/clu/metric_writers/) for full details. By
default, the T5X
[`MetricsManager`](https://github.com/google-research/t5x/tree/main/t5x/trainer.py?q=symbol:%5CbMetricsManager%5Cb)
logs metrics to
[TensorBoard](https://github.com/google/CommonLoopUtils/tree/main/clu/metric_writers/summary_writer.py),
[XManager](https://github.com/google/CommonLoopUtils/tree/main/clu/metric_writers/google/xm_measurement_writer.py),
and
[INFO logs](https://github.com/google/CommonLoopUtils/tree/main/clu/metric_writers/logging_writer.py).
In the future, the set of writers used will be made more easily customizable.

## Usage in T5X

In a T5X Model, we have a `loss_fn` that returns a dictionary of metrics,
mapping string name to `Metric` objects. In the simplest case, this may involve
creating a dictionary such as the following:

```py
metrics = {
  'nonpadding_tokens_fraction': Average(mask.sum(), count=mask.size()),
  'accuracy': Accuracy.from_model_output(
      logits=logits, labels=targets.astype(jnp.int32), mask=mask)
}
```

`Metric` objects can either be intialized directly or by using
`from_model_output`.

The metrics created on one particular training step (one call of the loss
function) are accumulated over subsequent steps (using the `merge` method).

NOTE: Unlike in previous versions of T5X, "initial metrics" should not be
defined, since the first metrics returned from `loss_fn` are treated as the
initial metrics for later accumulation.

Finally, in order to summarize the metrics into writable forms, we can simply
use the following:

```py
summary = {k: m.compute() for k, m in metrics.items()}
```

Typically, the above call will not be necessary, since the T5X `BaseModel`
already includes it automatically.

### A Metric Example

Let's imagine that we want to create a metric that tracks the loss per number of
tokens. One (bad) way of doing this would be the following:

```py {.bad}
# create metrics and return from loss_fn
metrics = {
  'loss': Sum(total=jnp.sum(loss))
  'num_tokens': Sum(total=num_tokens)
}

# run for many steps, metrics get merged and accumulated

# summarize metrics
summary = {
  'loss_per_all_target_tokens':
      metrics['loss'].compute() / metrics['num_tokens'].compute()
}
```

If this looks familiar, then you may be used to the old way of handling metrics
in T5X. This is obviously less than ideal, since we track two "metrics" that
we're not interested in, which we use to compute the actual one metric we want.

A better way of implementing this could be more like this:

```py
# create metrics and return from loss_fn
metrics = {
  'loss_per_all_target_tokens': Average(total=jnp.sum(loss), count=num_tokens)
}

# run for many steps, metrics get merged and accumulated

# summarize metrics
summary = {k: m.compute() for k, m in metrics.items()}
```

There are a few advantages of this change. First, we don't need to implement any
new logic in the summarization step - we can simply reuse the generic logic.
Second, our metric, `loss_per_all_target_tokens`, is explicitly created as an
`Average`, and is tracked throughout training, with no extraneous intermediate
metrics.

NOTE: If you see live code like the former example, it is part of our ongoing
migration towards new-style metrics in T5X. Please help us clean it up!

### Special T5X Metrics

A few metrics are somewhat more complicated to use, largely due to limitations
of the T5X training library. Metrics can be found at
[`t5x/metrics.py`](https://github.com/google-research/t5x/tree/main/t5x/metrics.py).

#### `AveragePerStep`

When dealing with per-step metrics, use `AveragePerStep`. This could correspond
to metrics such as loss per step. It cannot be implemented simply using a
standard `Average` metric because the loss function, where the metric is
initially computed, may be run multiple times if we have multiple microbatches.
If we have two microbatches, this results in the metric being initialized twice
per step. Thus, we defer setting number of steps at creation time and set it
before the metrics are summarized.

For example, we need to initialize `z_loss` and `steps_per_second` as follows:

```py
'z_loss': AveragePerStep.from_model_output(z_loss)
```

Then, before summarization
[`set_step_metrics_num_steps(metrics, num_steps)`](https://github.com/google-research/t5x/tree/main/t5x/metrics.py;l=222)
is called automatically to set the number of steps for relevant metrics.

#### `TimeRate`

Another special metric is `TimeRate`, which is used to measure metrics over a
period of time. Our complication here is that the start time of the metric
cannot be set when the metric is created, since creation happens inside a JAX
compiled function. Instead, we must set the duration on the host.

For example, we can initialize a `seqs_per_second` metric as follows:

```py
'timing/seqs_per_second': TimeRate(numerator=num_examples)
```

Before summarization,
[`set_time_rate_metrics_duration(metrics, duration)`](https://github.com/google-research/t5x/tree/main/t5x/metrics.py;l=209)
is called automatically called to set the duration of time-related metrics.

#### `StepsPerTime`

This metric represents the sythesis of the above two, which can represent a
metric such as `steps_per_second`.

```py
'timing/steps_per_second': StepsPerTime()
```

NOTE: Unless you are also overriding
[Trainer](https://github.com/google-research/t5x/tree/main/t5x/trainer.py;l=314), you likely only
need to worry about initializing metrics correctly, and not about making later
adjustments for duration and number of microbatches.
