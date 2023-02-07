"""
From: https://github.com/EleutherAI/the-pile
the_pile dataset
"""

import tensorflow_datasets as tfds
import tensorflow as tf
import io
import zstandard
import jsonlines
import os
import time
from itertools import chain
"""
Tips for Colab - Change _PILE_SPLITS below to increments of 8 to allow downloading and storing in GCS
After every 8 parts, tfds will flush the tempfiles from local and it will be cached on GCS, allowing reuse
preventing th need to redownload again. Example below

_download: Skipping download of http://eaidata.bmk.sh/data/pile/train/26.jsonl.zst: File cached in gs://your_bucket/datasets/cached/downloads/eaidata.bmk.sh_pile_train_26.jsonlCue2aNl9cxodxAvl9vIacuexGWYSoJAt4Rpcy19pqds.zst
_download: Skipping download of http://eaidata.bmk.sh/data/pile/train/27.jsonl.zst: File cached in gs://your_bucket/datasets/cached/downloads/eaidata.bmk.sh_pile_train_27.jsonlt8W_PLYeC4bZeaNMqMhe0-lhS3ijPL7RjvILWsMZlhQ.zst
_download: Downloading http://eaidata.bmk.sh/data/pile/train/28.jsonl.zst into gs://your_bucket/datasets/cached/downloads/eaidata.bmk.sh_pile_train_28.jsonl7Fj9nvI6std-e0H2ScxDKMpTWEC8iJMI8OT2vxLw2A4.zst.tmp.576c9ac11d30419b8ea8f30a5157ee53...
_download: Downloading http://eaidata.bmk.sh/data/pile/train/29.jsonl.zst into gs://your_bucket/datasets/cached/downloads/eaidata.bmk.sh_pile_train_29.jsonl1syFpl-ESnwk__9_6Xrj_OO5mRxpmaxQG7bZ_5d2sZc.zst.tmp.2f7f6afb86d74e988dcdb71d59b0d3f2...


Use tfds.disable_progress_bar() to prevent javascript issues
This uses pysimdjson for faster parsing of json. The entire dataset should be completed in around 12 hours on Colab.

"""

_USAGE_EXAMPLE = """
This can be run in a script or in a notebook.

_GCS_BUCKET = 'gs://your_gcs_bucket/path'

import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/path/to/adc.json' # if building to store in GCS
os.environ['TFDS_DATA_DIR'] = _GCS_BUCKET

import tensorflow_datasets as tfds
from the_pile import tfds_pile
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '<|padding|>'})

def simple_tokenization(item):
    return tokenizer.encode(item['text'], return_tensors='tf')

tfds.disable_progress_bar() # optional - will help with colab since tqdm breaks often

ds = tfds.load(name="ThePile", try_gcs=True)

# Have not tested below
ds.map(simple_tokenization, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# or 
ds.map(lambda item: simple_tokenization(item), num_parallel_calls=tf.data.experimental.AUTOTUNE)

"""

try:
  import simdjson as json
except ImportError:
  print('Installing simdjson library')
  os.system('pip install -q pysimdjson')
  import simdjson as json
  parser = json.Parser()

parser = json.Parser()
_DESCRIPTION = """
The Pile is a large, diverse, open source language modelling data set 
that consists of many smaller datasets combined together. 
The objective is to obtain text from as many modalities as possible to 
ensure that models trained using The Pile will have much broader generalization abilities.
We are currently developing Version 1, with an ultimate goal of 1 TiB of English text. 
After the completion of Version 1, our next goal is a fully-multilingual, 10TiB text dataset.
"""

_CITATION = """
"""
_DATASET_MODES = ["lm"]

_PILE_URL = 'https://the-eye.eu/public/AI/pile/train/{}.jsonl.zst'
_PILE_SPLITS = 30

_URLS = {
    'the_pile': {
        'train': [
            _PILE_URL.format(str(i).zfill(2)) for i in range(_PILE_SPLITS)
        ],
        'test': 'https://the-eye.eu/public/AI/pile/test.jsonl.zst',
        'validation': 'https://the-eye.eu/public/AI/pile/val.jsonl.zst',
    }
}

_VERSION = tfds.core.Version('1.0.0')
_RELEASE_NOTES = {
    '1.0.0': 'Initial release.',
}

_NAME = 'the_pile'
_FILE_FORMAT = 'jsonlines'


def json_parser(x):
  global parser
  try:
    line = parser.parse(x).as_dict()
    return line
  except ValueError:
    return x


class PileReader:

  def __init__(self, filenames, para_joiner='\n\n'):
    if not isinstance(filenames, list):
      filenames = [filenames]
    self.filenames = filenames
    self.para_joiner = para_joiner

  def _read_fn(self, filename):
    print(filename)
    with tf.io.gfile.GFile(filename, 'rb+') as f:
      cctx = zstandard.ZstdDecompressor()
      reader_stream = io.BufferedReader(cctx.stream_reader(f))
      reader = jsonlines.Reader(reader_stream, loads=json_parser)
      print('reader made')
      for item in reader:
        result = dict()
        if isinstance(item, str):
          result['text'] = item
        else:
          text = item['text']
          if isinstance(text, list):
            text = self.para_joiner.join(text)
          result['text'] = text

        yield result

  def __iter__(self):
    print(self.filenames)
    #for item in chain.from_iterable([self._read_fn(filename) for filename in self.filenames]):
    #    return item
    #for filename in self.filenames:
    #    return self._read_fn(filename)
    return chain.from_iterable(
        [self._read_fn(filename) for filename in self.filenames])


class ThePileConfig(tfds.core.BuilderConfig):

  def __init__(self, *, mode=None, **kwargs):
    super(ThePileConfig, self).__init__(name=mode,
                                        description="The Pile dataset",
                                        **kwargs)


class ThePile(tfds.core.GeneratorBasedBuilder):
  BUILDER_CONFIGS = [
      ThePileConfig(version=_VERSION, mode=mode) for mode in _DATASET_MODES
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({'text': tfds.features.Text()}),
        supervised_keys=("text", "text"),
        homepage='https://github.com/EleutherAI/The-Pile',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    dl_manager.verify_ssl = False
    dl_paths = dl_manager.download(_URLS['the_pile'])
    print(dl_paths)
    return {
            'train': self._generate_examples(dl_paths['train']),
            'validation': self._generate_examples(dl_paths['validation']),
            'test': self._generate_examples(dl_paths['test']),
    }

  def _generate_examples(self, paths):
    pipeline = PileReader(paths)
    #print('pipeline', pipeline)
    for x, result in enumerate(pipeline):
      if result:
        idx = f'{x}_the_pile'
        yield idx, {'text': result['text']}
