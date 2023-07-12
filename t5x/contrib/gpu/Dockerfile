ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:22.08-tf2-py3
FROM ${FROM_IMAGE_NAME}

# Install the latest jax
RUN pip install jax[cuda]==0.4.1 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# setup directory paths for T5x
ENV TFDS_DATA_DIR=/t5x_home/datasets/
ENV T5X_DIR=/t5x_home/
ENV T5X_WORKSPACE_DIR=/t5x_home/workspace
ENV PYTHONPATH=/t5x_home/
WORKDIR /t5x_home

# install the requirements for T5x
COPY . .
RUN pip install -e '.[gpu]'
