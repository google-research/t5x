# To enable importing version.py directly, we add its path to sys.path.
import seqio

from t5x import adafactor
from t5x import decoding
from t5x import models
from t5x.examples.decoder_only import network

decoder_only_model = models.DecoderOnlyModel(
    network.DecoderWrapper(
        network.TransformerConfig(
            vocab_size = 32128,  # vocab size rounded to a multiple of 128 for TPU efficiency
            dtype = 'bfloat16',
            emb_dim = 768,
            num_heads = 12,
            num_layers = 12,
            head_dim = 64,
            mlp_dim = 2048,
            mlp_activations = ('gelu', 'linear'),
            dropout_rate = 0.1,
            logits_via_embedding = True
        )
    ),
    seqio.SentencePieceVocabulary(
        "gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model"
    ),
    adafactor.Adafactor(),
    decode_fn=decoding.temperature_sample,
)

import ipdb
ipdb.set_trace()