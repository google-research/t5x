# T5X with Transformer Engine Summary #

**Highlight:**
1. Add `TransformerEngineHelper` to allow users to switch with or without Transformer Engine.
2. Add the feature of transposing batch_size and sequence to accelerate performance.
2. Hide FP8 metadata in `flax_mutable`. The flax_mutable is a variable collection that originally is declared by T5X.

## The *.gin files ##
They are configurations to set up T5X. The major change is to replace the AdaFactor optimizer with AdamW because of performance concerns. In old XLA, using AdaFactor will generate a lot of D2D copies and slow down the performance. Although the issue was resolved, we used AdamW to verify convergence and performance tests for now.

## network.py ##
1. The `TransformerEngineHelper` is a singleton to manage ON/OFF Transformer Engine, to hide the if-else statement inside. The pseudo code is like:
    ```python
    class TransformerEngineHelper:
        @staticmethod:
        def foo(x):
            if _IS_TRANSFORMER_ENGINE_INSTALLED and use_te:
                y = TransformerEngine.foo(x)
            else:
                y = T5X.foo(x)
            return y
    ```
2. The input tensor is BATCH_SEQ_HIDDEN format (i.e., batch_size, sequence, ...) by default. If `cfg.transpose_batch_sequence` is True, transpose input tensor to SEQ_BATCH_HIDDEN format because using SEQ_BATCH_HIDDEN is faster for now. It might not be necessary after integrating cuDNN MHA. And according to `output_format` to decide whether to transpose output tensor or not. It is for easy debugging.
3. The reason to rename the mask from `encoder_mask`/`decoder_mask` to `attention_mask` is to align the kwargs of TransformerLayer between T5X and Transformer Engine. The original T5X TransformerLayer has a different parameter list than the Transformer Engine. It blocks us from making a functor to switch two of them. The pseudo code is like:
    ```python
    if use_te:
        TransformerLayer = te.TransformerLayer
    else:
        TransformerLayer = t5x.TransformerLayer

    y = TransformerLayer(x, attention_mask=mask)
    ```
4. The `TransformerEngineHelper.get_attn_mask(*_mask)` is used to convert the T5X mask to the format required by Transformer Engine. In T5X, `1` means keep and `0` means drop, but in Transformer Engine, the meaning is reversed.

## utils.py ##
1. The `jax.eval_shape` has to be wrapped by `TransformerEngineHelper.eval_shape_guard()` because the `ShardingResource` must be set first. Otherwise, xmap cannot infer the shape of each layer of the model, and an exception will be thrown.
2. The `flax_mutables` is a variable collection that contains FP8 metadata and sharding information (e.g., named logical axis). It is required by FP8 training and tensor parallelism.

## trainer.py ##
1. At the code: `grad_fn = jax.value_and_grad(model.loss_fn, argnums=(0, 3), ...)`, the number `0` refers to 1st argument of loss_fn, and the number `3` refers to 4th argument of loss_fn. The 1st argument is input tensor. The 4th argument is the `flax_mutables` which contains FP8 metadata. In order to get the updated FP8 metadata after 1 training step, we need to ask JAX to differentiate `flax_mutables`. Note that, in fact, FP8 metadata is NOT calculated by differentiation. The FP8 metadata is maintained by the Transformer Engine. It is a trick to get the updated FP8 metadata because we didn't find other interfaces or approaches to get it.
2. At the code:
    ```diff
    - initial_flax_mutables = train_state.flax_mutables if train_state.flax_mutables else None
    + initial_flax_mutables = train_state.flax_mutables if train_state.flax_mutables else {}
    ```
    The `None` should be a T5X bug. It will trigger exceptions if `flax_mutables` needs to be filled into JAX routines. Although T5X declares the `flax_mutables`, it actually doesn't use it. Thus, T5X developers weren't aware of this issue.
3. The `grad_accum` becomes a list of variable collection because two variables are differentiated. The 1st is model parameters. The 2nd is FP8 metadata.
4. At the code:
    ```python
    grad_accum = (grad_accum[0],
        TransformerEngineHelper.update_fp8_metas(
        grad_accum[1], flax_mutables, train_state.step))
    ```
    It is a workaround due to the T5X (or JAX) bug. We don't know the root-cause yet and don't have time to investigate it. The bug is that T5X always misses 1 time of accumulating gradients. For example, if the accumulation step is 10, T5X should run micro-batch 10 times and accumulate the gradient of each micro-batch but it only accumulates gradient 9 times. If the accumulation step is 1, T5X doesn't update the gradient. Thus, the workaround is to accumulate the gradient 1 time manually.

## train_state.py ##
1. Add `flax_mutables_axes`, so xmap can know how to do the sharding for FP8 metadata.

## train.py ##
1. Import `TransformerEngineHelper` and initialize it.

## te_helper.py ##
1. A new file contains the `TransformerEngineHelper` implementation. Note that it uses Transformer Engine internal API - `FP8Helper.initialize` and `FP8Helper.finalize`. It is a trade off between the number of lines of code changes and the recommended way for enabling FP8 training. The recommended approach is:
    ```python
    with te.fp8_autocast(fp8_format, ...):
        model = Net()
        variable_collection = model.init(rng, inputs)
        state = TrainState.create(apply_fn=model.apply, ...)
        train_epoch(state, dataset)
    ```
    It is equal to:
    ```python
    FP8Helper.initialize(fp8_format, ...) # allocate FP8 metadata and setup
    model = Net()
    variable_collection = model.init(rng, inputs)
    state = TrainState.create(apply_fn=model.apply, ...)
    train_epoch(state, dataset)
    FP8Helper.finalize() # release FP8 metadata
    ```

## partitioning.py ##
1. Append the sharding rules needed by Transformer Engine after T5X's rues

## models.py ##
1. Add `eval_fn` because a new argument - `flax_mutable` is needed.
2. Add `predict_batch` because a new argument - `flax_mutable` is needed.
3. At the code:
    ```python
    module.apply(
        {'params': params, **flax_mutable},
        ...
    )
    ```
    The module.apply only accepts 1 variable collection, so model parameters and FP8 metadata need to be merged before filled into apply.
4. The `cache_offset` indicates which dimension is batch_size, for beam-search. Thus, it must be changed if `cfg.transpose_batch_sequence` is True.

## run_t5x_*.sh ##
1. They are shell scripts for convenience in running experiments.

