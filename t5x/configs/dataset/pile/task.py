import functools
import seqio
from t5.data import preprocessors

from t5x.configs.dataset.pile.utils import PileDatasetFnCallable

vocabulary = seqio.SentencePieceVocabulary(
    'gs://t5-data/vocabs/cc_all.32000/sentencepiece.model', extra_ids=100)
output_features = {
    'inputs': seqio.Feature(vocabulary=vocabulary),
    'targets': seqio.Feature(vocabulary=vocabulary)
}

DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=vocabulary, add_eos=True,
        required=False),
    "targets": seqio.Feature(
        vocabulary=vocabulary, add_eos=True)
}

seqio.TaskRegistry.add(
    'pile_t2t_span_corruption',
    source=seqio.FunctionDataSource(dataset_fn=PileDatasetFnCallable(), splits=["train", "val"]),
    preprocessors=[
        functools.partial(
            preprocessors.rekey, key_map={
                "inputs": None,
                "targets": "text"
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(required=True),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[]
)

# Prefix language modeling pretraining task used in Raffel et al., 2019.
seqio.TaskRegistry.add(
    "pile_t2t_prefix_lm",
    source=seqio.FunctionDataSource(dataset_fn=PileDatasetFnCallable(), splits=["train", "val"]),
    preprocessors=[
        functools.partial(
            preprocessors.rekey, key_map={
                "inputs": None,
                "targets": "text"
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(required=True),
        preprocessors.prefix_lm,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[]
)