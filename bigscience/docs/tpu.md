# Getting started with TPUv4s

This is intended to complete the Alpha User Guide. If you don't have access to it, ask on Slack :). 

1. **Get added to the `bigscience` GCP project**. You can ask Colin Raffel to add you on Slack, just give him the e-mail you need linked. 
2. **Install the `gcloud` CLI: https://cloud.google.com/sdk/docs/install**. Most TPUv4 related features are not yet available in the GCP Console, so you will need to go through the command line. 
3. **Set your default project to bigscience:** `gcloud config set project bigscience`.
4. **Enable the TPU tools in the CLI:**
```
gcloud services enable tpu.googleapis.com
gcloud alpha compute tpus tpu-vm service-identity create --zone=us-central2-b
```
5. **Create a subnetwork for your TPUs**. /!\ This step is not mentionned in the Alpha User Guide, but is necessary.
```
SUBNET=yourname-tpusubnet
gcloud compute networks subnets create ${SUBNET} --network=default --range=10.12.0.0/20 --region=us-central2 --enable-private-ip-google-access
```
6. **Spin-up your TPUs**. For now, we will focus on a single `v4-8`, things are more complicated when using an actual pod. We have quota for up to a `v4-1024`.
```
TPU_NAME=yourname-tpu
ZONE=us-central2-b
ACCELERATOR_TYPE=v4-8
RUNTIME_VERSION=v2-alpha-tpuv4
gcloud alpha compute tpus tpu-vm create ${TPU_NAME} --zone ${ZONE} --accelerator-type ${ACCELERATOR_TYPE} --version ${RUNTIME_VERSION} --subnetwork ${SUBNET}
```

7. **Connect to your TPU: `gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE}`**. It might prompt you to create an SSH key for `gcloud` specifically, just follow the process.

8. **Setup JAX on your TPU:**
```
sudo pip uninstall jax jaxlib -y
pip3 install -U pip
pip3 install jax jaxlib
gsutil cp gs://cloud-tpu-tpuvm-v4-artifacts/wheels/libtpu/latest/libtpu_tpuv4-0.1.dev* .
pip3 install libtpu_tpuv4-0.1.dev*
```

9. **Test JAX on your TPU.** This should return `8`, but will hang if you are using a pod (you have to run it on each VM then ;)).
```
python3
import jax
jax.device_count()
```

10. **Kill your TPU:** `gcloud alpha compute tpus tpu-vm delete ${TPU_NAME} --zone ${ZONE}`