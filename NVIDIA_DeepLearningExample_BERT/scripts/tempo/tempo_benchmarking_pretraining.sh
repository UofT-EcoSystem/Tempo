#!/bin/bash

## Default arguments
train_batch_size=${1:-8192}
learning_rate=${2:-"6e-3"}
precision=${3:-"fp16"}
num_gpus=${4:-$(nvidia-smi -L | wc -l)}
warmup_proportion=${5:-"0.2843"}
train_steps=${6:-7038}
save_checkpoint_steps=${7:-10000}
resume_training=${8:-"false"}
create_logfile=${9:-"true"}
accumulate_gradients=${10:-"true"}
gradient_accumulation_steps=${11:-128}
seed=${12:-12439}
job_name=${13:-"bert_lamb_pretraining"}
allreduce_post_accumulation=${14:-"true"}
allreduce_post_accumulation_fp16=${15:-"true"}
train_batch_size_phase2=${16:-4096}
learning_rate_phase2=${17:-"4e-3"}
warmup_proportion_phase2=${18:-"0.128"}
train_steps_phase2=${19:-1563}
gradient_accumulation_steps_phase2=${20:-512}
#change this for other datasets
DATASET=pretrain/phase1/unbinned/parquet
DATA_DIR_PHASE1=${21:-$BERT_PREP_WORKING_DIR/${DATASET}/}
#change this for other datasets
DATASET2=pretrain/phase2/bin_size_64/parquet
DATA_DIR_PHASE2=${22:-$BERT_PREP_WORKING_DIR/${DATASET2}/}
CODEDIR=${23:-"/workspace/bert"}
init_checkpoint=${24:-"None"}
VOCAB_FILE=vocab/vocab
RESULTS_DIR=$CODEDIR/results
CHECKPOINTS_DIR=$RESULTS_DIR/checkpoints
wikipedia_source=${25:-$BERT_PREP_WORKING_DIR/wikipedia/source/}
num_dask_workers=${26:-$(nproc)}
num_shards_per_worker=${27:-128}
num_workers=${28:-4}
num_nodes=1
sample_ratio=${29:-0.9}
phase2_bin_size=${30:-64}
masking=${31:-static}
BERT_CONFIG=${32:-bert_configs/large.json}

# gradient accumulation and allreduce
# allreduce_post_accumulation      - If set to true, performs allreduce only after the defined number of gradient accumulation steps.
# allreduce_post_accumulation_fp16 - If set to true, performs allreduce after gradient accumulation steps in FP16.
precision='fp32'
accumulate_gradients="false"
allreduce_post_accumulation="false"
allreduce_post_accumulation_fp16="false"

# phase1
micro_train_batch_size=32
gradient_accumulation_steps=1
train_batch_size=`expr $micro_train_batch_size \* $gradient_accumulation_steps`
train_steps=-1

# phase2
micro_train_batch_size_phase2=8
gradient_accumulation_steps_phase2=1
train_batch_size_phase2=`expr $micro_train_batch_size_phase2 \* $gradient_accumulation_steps_phase2`
train_steps_phase2=500

bash scripts/tempo/tempo_run_pretraining.sh $train_batch_size $learning_rate $precision $num_gpus $warmup_proportion $train_steps $save_checkpoint_steps $resume_training $create_logfile $accumulate_gradients $gradient_accumulation_steps $seed $job_name $allreduce_post_accumulation $allreduce_post_accumulation_fp16 $train_batch_size_phase2 $learning_rate_phase2 $warmup_proportion_phase2 $train_steps_phase2 $gradient_accumulation_steps_phase2 $DATA_DIR_PHASE1 $DATA_DIR_PHASE2 $CODEDIR $init_checkpoint $wikipedia_source $num_dask_workers $num_shards_per_worker $num_workers $sample_ratio $phase2_bin_size $masking $BERT_CONFIG