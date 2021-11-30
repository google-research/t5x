# Set your default project to bigscience
gcloud config set project bigscience

# Enable the TPU tools in the CLI
gcloud services enable tpu.googleapis.com
gcloud alpha compute tpus tpu-vm service-identity create --zone=us-central2-b

# Spin-up your TPUs
NAME=$1
if [[ $NAME = "" ]]
then
  echo "Please feed in the name of the machine you're going to use"
  exit
fi

echo "Using $NAME as name"
SUBNET=julien-tpusubnet # to be updated to a single subnet at some point
TPU_NAME=$NAME-tpu
ZONE=us-central2-b
#ACCELERATOR_TYPE=v4-8
#RUNTIME_VERSION=v2-alpha-tpuv4
ACCELERATOR_TYPE=v4-128
RUNTIME_VERSION=v2-alpha-tpuv4-pod
gcloud alpha compute tpus tpu-vm create ${TPU_NAME} --zone ${ZONE} --accelerator-type ${ACCELERATOR_TYPE} --version ${RUNTIME_VERSION} --subnetwork ${SUBNET}

# Launch your TPU
# sh bigscience/scripts/start_tpu_instance enc_dec_c4_span_corruption

# Connect to your TPU
# gcloud alpha compute tpus tpu-vm ssh enc_dec_c4_span_corruption-tpu --zone us-central2-b
