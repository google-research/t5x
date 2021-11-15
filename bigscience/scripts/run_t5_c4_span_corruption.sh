# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
MODEL_DIR="gs://bigscience/t5x/t5_c4_span_corruption"

# Data dir to save the processed dataset in "gs://data_dir" format.
T5X_DIR="/home/thomas/code/t5x"  # directory where the T5X repo is cloned.

python3 ${T5X_DIR}/t5x/train.py \
  --gin_file="bigscience/gins/t5_c4_span_corruption.gin" \
  --gin.DROPOUT_RATE=0.1 \
  --gin.MODEL_DIR="'${MODEL_DIR}'"
