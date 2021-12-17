pushd ~/code/t5x

ORIGINAL_EXPERIMENT_NAME=$1

if [[ $ORIGINAL_EXPERIMENT_NAME == *t0_adapt* ]]
then
  echo "I don't know how much T0 adaptation one has to do, so I don't know the correct checkpoint"
  exit 1
else
  CHECKPOINT_STEP=32768
fi

PYTHONPATH=$(pwd)/bigscience/gins python3 $(pwd)/t5x/eval_harness.py \
   --gin_file_="bigscience/gins/c_dec_xxl.gin" \
   --gin_file_="bigscience/gins/eval_harness.gin" \
   --gin.INFER_OUTPUT_DIR="'.'"  \
   --gin.DROPOUT_RATE=0.0 \
   --gin.CHECKPOINT_PATH="'gs://bigscience-t5x/arch_objective_exps_v2/$ORIGINAL_EXPERIMENT_NAME/checkpoint_$CHECKPOINT_STEP'" \
   --results_path /home/thomas/arch_objective_exps_v2/"$ORIGINAL_EXPERIMENT_NAME".json \
   --tasks=arc_challenge,arc_easy,boolq,copa,headqa,hellaswag,lambada,logiqa,mathqa,mc_taco,mrpc,multirc,openbookqa,piqa,prost,pubmedqa,qnli,qqp,race,rte,sciq,sst,triviaqa,webqs,wic,winogrande,wnli,wsc
