# Prepare Colab Runtime

Currently the [default public Colab](https://colab.research.google.com/) cannot be easily used to run T5X models. Here we provide an alternative, i.e., creating a custom jupyter kernel/runtime via Google Cloud TPU VM. One can then use the `Connect to a local runtime` option run the notebooks in this folder.

## Create TPU VM
You should follow T5X's main README.md [installation guide](https://github.com/google-research/t5x#installation) to setup a GCP account.

Then create a TPU VM via the command below (make sure to change `TPUVMNAME` and `TPUVMZONE` accordingly)

```
export TPUVMNAME=xxxx;
export TPUVMZONE=xxxxxxx;
export TPUTYPE=v3-8;
export APIVERSION=v2-alpha

gcloud alpha compute tpus tpu-vm create ${TPUVMNAME} --zone=${TPUVMZONE} --accelerator-type=${TPUTYPE} --version=${APIVERSION}
```

## ssh to TPU VM
You need to set proper firewall rules to be able to ssh into the VM.

```
gcloud compute firewall-rules create default-allow-ssh --allow tcp:22
```

ssh into the VM with port forwarding (`8888` is often used for ipython notebook kernel)

```
gcloud compute tpus tpu-vm ssh ${TPUVMNAME} --zone=${TPUVMZONE} -- -L 8888:localhost:8888
```

## Prepare python env

Create a python environment via

```
sudo apt update
sudo apt install -y python3.9 python3.9-venv
python3.9 -m venv t5_venv
```

Then install T5X with its dependencies.

```
source t5_venv/bin/activate
python3 -m pip install -U pip setuptools wheel ipython
pip install flax
git clone --branch=main https://github.com/google-research/t5x
cd t5x
python3 -m pip install -e '.[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
cd -
```

After this, we can test if we can accessed TPU successfully by (should print out a list of TPU devices)

```
python3 -c "import jax; print(jax.local_devices())"
```

At last, we prepare necessary packages to allow the jupyter kernel can be access by our colab notebooks.

```
pip install notebook
pip install --upgrade jupyter_http_over_ws>=0.0.7
jupyter serverextension enable --py jupyter_http_over_ws
```

## Launch runtime

Use the command below to launch the prepared runtime.

```
jupyter notebook   --NotebookApp.allow_origin='https://colab.research.google.com'   --port=8888   --NotebookApp.port_retries=0
```

from the log of the above command, you can see an http link starting with `http://localhost:8888/?token`s. Copy and paste it into the `Connect to a local runtime` option and now you should be able to run T5X colab notebooks.
