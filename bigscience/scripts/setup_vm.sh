sudo pip uninstall jax jaxlib -y
pip3 install -U pip
pip3 install jax jaxlib
gsutil cp gs://cloud-tpu-tpuvm-v4-artifacts/wheels/libtpu/latest/libtpu_tpuv4-0.1.dev* .
pip3 install libtpu_tpuv4-0.1.dev*

mkdir -p ~/code
cd ~/code

git clone https://github.com/bigscience-workshop/t5x.git
pushd t5x
git checkout thomas/add_train_script_span_corruption
pip install -e .
popd

# TODO: figure if this is actually important
sudo rm /usr/local/lib/python3.8/dist-packages/tensorflow/core/kernels/libtfkernel_sobol_op.so

