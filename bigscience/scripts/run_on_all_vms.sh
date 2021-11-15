# When working with pods, one has to send command to all tpus workers
NAME=$1
COMMAND=$2
TPU_NAME=$NAME-tpu
ZONE=us-central2-b

echo $COMMAND

# TODO: wrap this in tmux in order for command not to be killed upon lost of ssh connection.
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} --worker=all --command="$COMMAND"