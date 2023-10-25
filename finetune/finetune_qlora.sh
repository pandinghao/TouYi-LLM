#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

GPUS_PER_NODE=2
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6002

MODEL="Qwen_model/Qwen/Qwen-7B" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="data/processed/stage1_train.json"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
torchrun $DISTRIBUTED_ARGS finetune.py \
    --output_dir output_qwen \
    --model_name_or_path Qwen_model/Qwen/Qwen-7B \
    --data_path $DATA \
    --num_train_epochs 10 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --gradient_accumulation_steps 2 \
    --logging_dir ./tb_logs \
    --logging_steps 10 \
    --eval_data_path data/processed/stage1_eval.json \
    --evaluation_strategy steps \
    --eval_steps 550 \
    --learning_rate 2e-4 \
    --model_max_length 1024 \
    --use_lora \
    --q_lora \
    --save_steps 1100 \
    --save_total_limit 20 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.1 \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --gradient_checkpointing \
    --disable_tqdm False \
    --optim paged_adamw_32bit \
    --seed 42 \
    --fp16 True \
    --report_to tensorboard \
    --dataloader_num_workers 0 \
    --save_strategy steps \
    --weight_decay 0.05 \
    --max_grad_norm 0.3 \
    --remove_unused_columns false