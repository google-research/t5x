# Decoding


This page outlines the decoding functions that T5X provides out-of-the-box and
how custom decoding functions can be used for a Transformer model, i.e., an
instance of
[`BaseTransformerModel`](https://github.com/google-research/t5x/tree/main/t5x/models.py?q=symbol:%5CbBaseTransformerModel%5Cb).
Here we refer to decoding as a process of generating a sequence of items from a
fixed alphabet (e.g., generating token ids from the vocabulary).

There are two major ways to configure the decoding routine. The first method is
to define a decode function that follows the `DecodeFnCallable` signature. This
is more restrictive as it enforces the call signature but users don't need to
modify the model code.

The second method is to subclass a model class and override
`predict_batch_with_aux` method. While this provides more flexibility, it
requires rewriting the method.

## Option 1: defining a decoding function

If a desired decoding process can follow `DecodeFnCallable`, it can be
registered as a private attribute of a
[`BaseTransformerModel`](https://github.com/google-research/t5x/tree/main/t5x/models.py?q=symbol:%5CbBaseTransformerModel%5Cb)
by passing it as a `decode_fn` argument to its constructor.

### Decoding function call signature

`DecodeFnCallable` has the following call signature


It takes in `inputs`, which is an int32 array with a shape `[batch_size,
max_decode_len]`. This is an input tokens to the decoder. For the standard
encoder-decoder models like T5, this is initialized as zeros with a desired
decoding length. The decoding function will populate the array with the sampled
token ids and return.

For a decoder-only architectures such as a Prefix Language Model, `inputs` can
be a concatenated sequence of "inputs" and "targets" tokens ids.

`tokens_to_logits` is a callable that takes in a batch of token ids and the
current autoregressive cache, performs the forward pass and returns the
resulting logits resulting and an updated cache. Note that for incremental
decoding, this function operates with a single token, i.e., the length dimension
is assumed to be 1.

`DecodeFnCallable` is designed to be as general as possible. This results in
some of the arguments being somewhat generic for a specialized decoding
algorithm. For example, `num_decodes` refers to the number of decoded samples to
be returned. In the case of beam search, `num_decodes` corresponds to what is
commonly known as `beam_size`, with returned sequences sorted by the beam
scores. For temperature sampling, we perform `num_decodes` *independent*
sampling procedures with different random seeds and sort them by the log
probability of the generated sequences.

For custom decoding functions, there might be additional arguments. To support
these, we provide `**kwargs`.

Another usage of `**kwargs` is calling `decoding_fn` multiple times without
recompiling the model. This pattern is used in
[Prediction Service](https://github.com/google-research/t5x/tree/main/t5x/google/prediction_service/README.md).
For a compiled model, different values of `alpha` can be passed e.g.,
`decoder_params = {"alpha": 0.7}` where `decoder_params` is the argument to
`predict_batch_with_aux`. It is unpacked and passed to `beam_search` function.
Note that the Prediction Service uses
[`predict_batch_with_aux`](https://github.com/google-research/t5x/tree/main/t5x/models.py?q=func:%5Cbpredict_batch_with_aux%5Cb),
which is one of the two public methods. This method is useful if auxiliary
outputs (e.g., scores of the predictions) are to be returned. The other method
is
[`predict_batch`](https://github.com/google-research/t5x/tree/main/t5x/models.py?q=func:%5Cbpredict_batch%5Cb),
which simply returns the predictions.

### Beam search

The following lines can be added to a gin file in order to use
[beam search](https://github.com/google-research/t5x/tree/main/t5x/decoding.py;l=881;rcl=446762159)
as a decoding function for an encoder-decoder model.

```gin
models.EncoderDecoderModel.predict_batch_with_aux.num_decodes = 4
models.EncoderDecoderModel.decode_fn = @decoding.beam_search
decode.beam_search.alpha = 0.6
```

Note that we skip the gin boilerplate code such as gin dynamic registration.
Please refer to [T5X Gin Primer](gin.md) for more details.

The beam search behavior is controlled by the arguments passed to `beam_search`.
We provide details for a few of them below.

#### `num_decodes`

If `num_decodes` are configured with `gin.register`, it is overridden by the
value explicitly passed by the caller e.g.,
`models.EncoderDecoderModel.predict_batch_with_aux`. This is because the
information about `num_decodes` is needed to prepare the encoder inputs and
outputs expanded by `num_decodes` times in the batch dimension.

We recommend that `num_decodes` be specified *only* in
`models.EncoderDecoderModel.predict_batch_with_aux`.

#### `alpha`

This is the brevity penalty introduced in
[Wu et al. 2016](https://arxiv.org/abs/1609.08144) to penalize short sequences.

#### `max_decode_len`

For evaluation, we typically don't want to truncate the examples by a specified
sequence length. Therefore, we dynamically obtain the length information from
the batch of examples. The default behavior of `seqio.Evaluator` is to use the
maximum length of a task but, this can be overridden.

Since the length information is provided dynamically, we don't set
`max_decode_len` in gin. Instead we pass the relevant `inputs` array to
`beam_search` whose length is the dynamically determined maximum length.

If `max_decode_len` is explicitly specified via gin, this will override the
implictly determined length information unless it is passed by
`predict_batch_with_aux`.

### Temperature sampling

[Temperature sampling](https://github.com/google-research/t5x/tree/main/t5x/decoding.py;l=37;rcl=446762159)
can be used for multiple decoding strategies. The following lines configures
temperature sampling as a `decode_fn`.

```gin
models.EncoderDecoderModel.predict_batch_with_aux.num_decodes = 1
models.EncoderDecoderModel.decode_fn = @decoding.temperature_sample
decoding.temperature_sample:
  temperature = 0.5
  topk = 20
```

Similar specification can be used for other model types by replacing
`models.EncoderDecoderModel` with the relevant model class, e.g.
`models.PrefixLanguageModel`.

The sampling behavior is controlled by the arguments passed to
`temperature_sample`. We provide details for a few of them below.

#### `temperature`

A probabilistic model outputs a probability distribution over a pre-defined
alphabet. For example, a language model outputs *logits*, which are unnormalized
probability values for each item in the vocabulary. We use a language model as a
running example. A sampling process involves *sampling* from the predicted
distribution one item at a time conditioned on the previously generated items
until a given number of items are generated or a sentinel token that represents
the end of sequence is generated.

Temperature modifies the unnormalized probability distribution at each step. For
each item $$i$$ in the vocabulary, its probability predicted by the model is
given by

$$p_i \propto \exp\left(\frac{x_i}{T} \right)$$

where $$T$$ is the temperature and $$x_i$$ is the logits value corresponding to
item $$i$$. As $$T \to 0$$, the distribution puts all probability mass to the
item with the highest probability. In other words, the sampling process becomes
a greedy search.

In the other extreme, as $$T \to \infty$$, the predicted distribution becomes
uniform.

#### `topk`

By specifying strictly positive integer value for `topk`, the sampling process
in each step is limited to the `k` items with highest probabilities. `topk` also
uses `temperature` to modify the logits corresponding to the top `k` items.

#### `topp`

By specifying non-zero positive float value for `topp`, the sampling process is
limited to a subset of the vocabulary $$V^{(p)} \subset V$$, which is defined by
the smallest set such that

$$\sum_{i \in V^{(p)}} p_i \ge p$$

where $$p_i$$ is the conditional dsitribution at each time step for item $$i$$.
This is called "Nucleus sampling", which was introduced by
[Holtzman et al. ICLR 2020](https://openreview.net/forum?id=rygGQyrFvH).
Currently, temperature is not used if `topp` is used.

Note that only one of `topk` or `topp` can be used.

## Option2: subclassing a model class

If `DecodeFnCallable` is not flexible enough for your custom decoding function,
you can subclass the model class and override `predict_batch_with_aux` method.
While the model class can be any instance of
[`BaseTransformerModel`](https://github.com/google-research/t5x/tree/main/t5x/models.py?q=symbol:%5CbBaseTransformerModel%5Cb),
we recommend that you subclass the existing models such as
[`EncoderDecoderModel`](https://github.com/google-research/t5x/tree/main/t5x/models.py?q=symbol:%5CbEncoderDecoderModel%5Cb)
and only override `predict_batch_with_aux` method.

`predict_batch_with_aux` method also has a required call signature, but it is
significantly more flexible. It should return a tuple of predicted sequence
array and auxiliary outputs such as score.
