# When working with pods, one has to send command to all tpus workers
NAME=$1
COMMAND=$2
TPU_NAME=$NAME-tpu
ZONE=us-central2-b

echo $COMMAND

# TODO: wrap this in tmux in order for command not to be killed upon lost of ssh connection.
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} --worker=all --command="$COMMAND" -- -t

# Example to run t5_c4_span_corruption
#  - run setup vms: sh bigscience/scripts/run_on_all_vms.sh enc_dec_c4_span_corruption "$(cat bigscience/scripts/setup_vm.sh)"
#  - run t5_c4_span_corruption: sh bigscience/scripts/run_on_all_vms.sh enc_dec_c4_span_corruption "cd code/t5x; git pull; sh bigscience/scripts/launch_command_in_tmux.sh \"sh bigscience/scripts/pretrain.sh enc_dec_c4_span_corruption\""
#  - kill zombie process: sh bigscience/scripts/run_on_all_vms.sh enc_dec_c4_span_corruption "killall -u thomas"