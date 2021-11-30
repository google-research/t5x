COMMAND=$1

if tmux new -s pretrain "${COMMAND}" | grep -x "duplicate session: pretrain"; then
  echo "Already running pretrain tmux session, please manually check what is running before killing it.
  To review: please connect to a worker, and run \`tmux a -t pretrain\`.
  To kill all sessions: \`run_on_all_vms.sh \${NAME} \"tmux kill-ses -t pretrain\"\`
  "
  exit
fi