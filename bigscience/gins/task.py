import functools

import seqio
from t5.data import preprocessors, get_default_vocabulary
from t5.data.preprocessors import select_random_chunk, reduce_concat_tokens, split_tokens

from t5x.partitioning import LogicalAxisRules

# --- Seqio ---
seqio.add_global_cache_dirs(['gs://bigscience-t5x/seqio_cached_tasks'])

TaskRegistry = seqio.TaskRegistry

def full_lm(dataset, sequence_length, output_features):
    """Full language modeling objective"""
    ds = dataset
    ds = select_random_chunk(ds, output_features=output_features, feature_key='targets', max_length=65536)
    ds = seqio.preprocessors.append_eos(ds, output_features)
    ds = reduce_concat_tokens(ds, feature_key='targets', batch_size=128)
    ds = split_tokens(ds, max_tokens_per_segment=sequence_length['targets'])
    # ds = trim_and_pad_dataset(ds, sequence_length) # I feel this should be interesting, we should use `split_tokens_to_targets_length`
    return ds

TaskRegistry.add(
    "c4_v220_full_lm",
    source=seqio.TfdsDataSource(tfds_name="c4/en:2.2.0"),
    preprocessors=[
        functools.partial(
            preprocessors.rekey, key_map={
                "inputs": None,
                "targets": "text"
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        full_lm,
    ],
    output_features={
        "targets": seqio.Feature(
            vocabulary=get_default_vocabulary(), add_eos=True)
    },
    metric_fns=[])

# --- Improve sharding ---

def fully_sharded_logical_axis_rules() -> LogicalAxisRules:
    """Fully sharded rules for P5X model in terms of logical axes names."""
    return (
      ('batch', 'data'),
      ('vocab', 'model'),
      ('mlp', 'model'),
      ('heads', 'model'),
      ('joined_kv', 'model'),
      ('kv', None),
      ('embed', 'model'),
      ('embed', 'data'),
      ('relpos_buckets', None),
      ('length', None),
      ('layers', None),
      ('stack', None),
    )
