#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

GPUS_PER_NODE=2
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6002
#nohup sh finetune/finetune_qlora_from8800.sh > logs/train_stage1_1026_startfrom8800.log 2>&1 &
MODEL="/root/autodl-tmp/Qwen_model/Qwen/Qwen-7B" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="/root/autodl-tmp/data/processed/stage_2/stage2_train.json"
#DATA="/root/autodl-tmp/data/processed/stage_2/cmeie_dev.json"
NER_EVAL_DATA="/root/autodl-tmp/data/processed/stage_2/cmeee_dev.json"
RE_EVAL_DATA="/root/autodl-tmp/data/processed/stage_2/cmeie_dev.json"
BIO_DIA_EVAL_DATA="/root/autodl-tmp/data/processed/stage_2/cmedqa_dev.json"
TOUYI_DIA_EVAL_DATA="/root/autodl-tmp/data/processed/stage_2/touyi_dev.json"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
torchrun $DISTRIBUTED_ARGS finetune.py \
    --output_dir /root/autodl-tmp/output_qwen_stage2 \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --num_train_epochs 5 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_dir ./tb_logs \
    --logging_steps 10 \
    --eval_data_name NER RE BIODIA \
    --eval_data_path $NER_EVAL_DATA $RE_EVAL_DATA $BIO_DIA_EVAL_DATA \
    --evaluation_strategy steps \
    --eval_steps 1154 \
    --learning_rate 2e-4 \
    --model_max_length 1024 \
    --use_lora \
    --q_lora \
    --save_steps 1154 \
    --save_total_limit 50 \
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
    --remove_unused_columns false \
    --predict_with_generate \
    --max_generated_tokens 500 \
    --generation_top_p 0.9 \
    --generation_temperature 0.2 \
    --resume_from_checkpoint /root/autodl-tmp/output_qwen_stage2/checkpoint-5770 
