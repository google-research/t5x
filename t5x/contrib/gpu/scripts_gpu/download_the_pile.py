import t5x.contrib.gpu.scripts_gpu.seqio_tasks
import tensorflow_datasets as tfds

# This will download 'ThePile' to TFDS_DATA_DIR (environment variable).
ds = tfds.load('ThePile')
