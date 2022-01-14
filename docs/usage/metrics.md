# Metrics Overview


TIP: See [below](#migration-instructions) for concrete instructions on how to
align your metrics with the redesigned T5X metric library.

## Introduction

T5X provides a flexible and customizable library for managing metrics. Metrics
in T5X rely on [CLU](go/clu), which broadly provides utilities for writing
training loops but specifically provides metric libraries that are extended by
T5X.


NOTE: This document currently only applies to train and 'train_eval' metrics,
not to 'infer_eval' metrics, which are implemented using SeqIO. We plan to unify
these three in the future.

## Metrics and Writers

CLU provides `Metric` and `MetricWriter` classes. Full details are provided in
[go/clu-metrics](go/clu-metrics), but a simplified summary will suffice for our
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
```

TODO(cpgaffney): Update depending on resolution of CLU generic write interface
discussion.

We will elaborate in more detail [below](#a-metric-example) on how Metrics are
practically used in T5X.

In addition to CLU provided metrics like Average and Accuracy, T5X provides a
few specialized metrics, like TimeRate and MicrobatchAdjusted. A full list of
CLU metrics is provided at
[clu/metrics.py](https://github.com/google/CommonLoopUtils/tree/main/clu/metrics.py) while T5X metrics
are listed in [t5x/metrics.py](https://github.com/google-research/t5x/tree/main/t5x/metrics.py). We
will elaborate on specialized metrics like TimeRate and MicrobatchAdjusted
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
defined, since the metrics returned from the first training step are simply used
as the initial metrics. Specifically, the `get_initial_metrics` functions
provided by BaseTrainer and BaseModel will be deprecated in the future.

Finally, in order to summarize the metrics into writable forms, we can simply
use the following:

```py
summary = {k: m.compute() for k, m in metrics.items()}
```

Typically, the above call will not be necessary, since the T5X `BaseModel`
already includes it automatically. Any model inheriting from this class can rely
on its implementation of `summarize_metrics_fn`.

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
of the T5X training library.

#### MicrobatchAdjusted

One such metric is `MicrobatchAdjusted`, which should be used to wrap another
metric that involves any "per-step" or "step-per" computation, such as loss per
step, or steps per second. This is due to the fact that while the loss function,
where the metric is initially computed, runs once per training step, it may be
run multiple times if we have multiple microbatches. If we have two
microbatches, this results in the metric being initialzed twice per step. We
need to add an adjustment for such metrics.

For example, we need to initialize `z_loss` as follows:

```py
'z_loss': MicrobatchAdjusted(
  metric=clu_metrics.Average(total=z_loss, count=1), per_step=True)
```

Then, later in the training loop,
[`set_microbatch_adjusted_metrics_microbatches(metrics, num_microbatches)`](https://github.com/google-research/t5x/tree/main/t5x/metrics.py;l=222)
is called automatically to ensure that per-step metrics are properly adjusted.

#### TimeRate

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

NOTE: Unless you are also overriding
[Trainer](https://github.com/google-research/t5x/tree/main/t5x/trainer.py;l=314), you likely only
need to worry about initializing metrics correctly, and not about making later
adjustments for duration and number of microbatches.

## Migration Instructions

If your T5X models override functions such as `get_initial_metrics` or
`summarize_metrics_fn`, please align with the current version of the T5X metrics
library by using the following instructions.

1.  Remove `get_initial_metrics` from models inheriting from T5X
    [`BaseModel`](https://github.com/google-research/t5x/tree/main/t5x/models.py?q=symbol:%5CbBaseModel%5Cb).

2.  Ensure all metrics dicts are mappings of string name to `clu.metrics.Metric`
    object, rather than mappings of string name to scalar values. See
    [above](#metrics-and-writers) for more details.

3.  Wherever metrics are computed (typically in a model's `loss_fn`), use a
    `Metric` subclass that fully describes the behavior of the metric. If the
    metric is an average, use `clu.metrics.Average`. If it is an accuracy
    metric, consider using `clu.metrics.Accuracy`. At any point, you must be
    able to call `metric.compute()` and receive the correct value of the metric.
    See [examples](#a-metric-example) for more details.

4.  Remove `summarize_metrics_fn` from models once the previous step is
    complete, as summarization now happens automatically.

