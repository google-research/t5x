ARG FROM_IMAGE_NAME=ghcr.io/nvidia/jax-toolbox-internal:5061977725-te
FROM ${FROM_IMAGE_NAME}

# setup directory paths for T5x
ENV TFDS_DATA_DIR=/t5x_home/datasets/
ENV T5X_DIR=/t5x_home/
ENV T5X_WORKSPACE_DIR=/t5x_home/workspace
ENV PYTHONPATH=/t5x_home/

WORKDIR /t5x_home

# install the requirements for T5x
COPY . .
RUN pip install -e '.[gpu]'
