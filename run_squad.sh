#!/bin/bash

source ~/.bashrc
conda activate scene

NUM_PERT=1
NUM_PERM=1
NUM_RETV=1
WEIGHT_PERT=1.0
WEIGHT_PERM=1.0
WEIGHT_RETV=1.0
WEIGHT_DECAY=0.01
EPOCHS=3
DATSET=squad
MODEL_NAME=csarron/roberta-base-squad-v1


accelerate launch train.py \
  --model_name_or_path ${MODEL_NAME} \
  --per_device_train_batch_size 32 \
  --num_train_epochs ${EPOCHS} \
  --learning_rate 2e-5 \
  --custom_warmup_steps 0 \
  --weight_decay 0.01 \
  --dataset_name ${DATSET} \
  --pad_to_max_length \
  --max_seq_length 384 \
  --doc_stride 128 \
  --version_2_with_negative \
  --num_perturbation_examples_per_batch ${NUM_PERT} \
  --num_permutation_examples_per_batch ${NUM_PERM} \
  --num_retrieval ${NUM_RETV} \
  --weight_perturb ${WEIGHT_PERT} --weight_permute ${WEIGHT_PERM} --weight_retrieval ${WEIGHT_RETV} \
  --remove_no_answer \
  --use_paraphrase_detector \
  --output_dir ./checkpoints/${DATSET}/roberta_pert_${NUM_PERT}w${WEIGHT_PERT}_perm_${NUM_PERM}w${WEIGHT_PERM}_retr_${NUM_RETV}w${WEIGHT_RETV}_epochs_${EPOCHS}
