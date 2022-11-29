#!/usr/bin/env bash

## Default arguments
init_checkpoint=${1:-"/workspace/bert/checkpoints/bert_large_pretrained_amp.pt"}
epochs=${2:-"2.0"}
batch_size=${3:-"4"}
learning_rate=${4:-"3e-5"}
warmup_proportion=${5:-"0.1"}
precision=${6:-"fp16"}
num_gpu=${7:-"4"}
seed=${8:-"1"}
squad_dir=${9:-"$BERT_PREP_WORKING_DIR/download/squad/v1.1"}
vocab_file=${10:-"$BERT_PREP_WORKING_DIR/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt"}
OUT_DIR=${11:-"/workspace/bert/results/SQuAD"}
mode=${12:-"train"}
CONFIG_FILE=${13:-"/workspace/bert/bert_configs/large.json"}
max_steps=${14:-"-1"}


# Experiments
# Original + FP16
batch_size=8
precision="fp16"
max_steps=200
OUT_DIR="/workspace/bert/results/SQuAD_BSZ-${batch_size}_PRE-${precision}_STEPS-${max_steps}"
./scripts/tempo/unpatch_tempo.sh
bash scripts/tempo/tempo_run_squad.sh $init_checkpoint $epochs $batch_size $learning_rate $warmup_proportion $precision $num_gpu $seed $squad_dir $vocab_file $OUT_DIR $mode $CONFIG_FILE $max_steps
pkill -9 python

# Original + FP32
batch_size=4
precision="fp32"
max_steps=200
OUT_DIR="/workspace/bert/results/SQuAD_BSZ-${batch_size}_PRE-${precision}_STEPS-${max_steps}"
./scripts/tempo/unpatch_tempo.sh
bash scripts/tempo/tempo_run_squad.sh $init_checkpoint $epochs $batch_size $learning_rate $warmup_proportion $precision $num_gpu $seed $squad_dir $vocab_file $OUT_DIR $mode $CONFIG_FILE $max_steps
pkill -9 python

# Tempo + FP16
batch_size=11
precision="fp16"
max_steps=200
OUT_DIR="/workspace/bert/results/SQuAD_BSZ-${batch_size}_PRE-${precision}_STEPS-${max_steps}_Tempo"
./scripts/tempo/patch_tempo.sh
bash scripts/tempo/tempo_run_squad.sh $init_checkpoint $epochs $batch_size $learning_rate $warmup_proportion $precision $num_gpu $seed $squad_dir $vocab_file $OUT_DIR $mode $CONFIG_FILE $max_steps
pkill -9 python

# Tempo + FP32
batch_size=6
precision="fp32"
max_steps=200
OUT_DIR="/workspace/bert/results/SQuAD_BSZ-${batch_size}_PRE-${precision}_STEPS-${max_steps}_Tempo"
./scripts/tempo/patch_tempo.sh
bash scripts/tempo/tempo_run_squad.sh $init_checkpoint $epochs $batch_size $learning_rate $warmup_proportion $precision $num_gpu $seed $squad_dir $vocab_file $OUT_DIR $mode $CONFIG_FILE $max_steps
pkill -9 python